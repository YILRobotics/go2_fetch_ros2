#include "fetch_low_level/motor_crc.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fetch_interfaces/msg/control_state.hpp>
#include <fetch_interfaces/msg/control_timing.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace fetch_low_level {
namespace {
constexpr size_t kJoints = 12;
constexpr size_t kObservationSize = 49;

// Bit positions in LowState::wireless_remote. These match Unitree's KeyMap.
constexpr size_t kStartButton = 2;
constexpr size_t kSelectButton = 3;
constexpr size_t kAButton = 8;
using Clock = std::chrono::steady_clock;
double ms(Clock::duration d)
{
  return std::chrono::duration<double, std::milli>(d).count();
}
bool finite3(const std::array<float, 3> &v)
{
  return std::all_of(v.begin(), v.end(), [](float x){
        return std::isfinite(x);
      });
}
struct TrtTiming
{
  float host_input{};
  float h2d{};
  float enqueue{};
  float execute{};
  float d2h{};
  float sync_wait{};
};

class TrtLogger final : public nvinfer1::ILogger
{
  void log(Severity s, const char *m) noexcept override
  {
    if (s <= Severity::kWARNING)
      fprintf(stderr, "TensorRT: %s\n", m);
  }
};

// Owns one static-shape TensorRT execution context and all CUDA buffers.
// infer() performs no allocation; host buffers are pinned for asynchronous copies.
class TensorRtPolicy
{
 public:
  explicit TensorRtPolicy(const std::string &path)
  {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
      throw std::runtime_error("Cannot open engine: " + path);
    std::vector<char> bytes(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(bytes.data(), bytes.size());
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_)
      throw std::runtime_error("Cannot deserialize engine: " + path);
    context_.reset(engine_->createExecutionContext());
    if (!context_)
      throw std::runtime_error("Cannot create TensorRT context");
    if (engine_->getNbIOTensors() != 2)
      throw std::runtime_error("Low-level engine must have one input and one output");
    for (int i = 0; i < 2; ++i)
    {
      const char *n = engine_->getIOTensorName(i);
      auto mode = engine_->getTensorIOMode(n);
      if (mode == nvinfer1::TensorIOMode::kINPUT)
        input_name_ = n;
      else output_name_ = n;
    }
    auto in = engine_->getTensorShape(input_name_);
    auto out = engine_->getTensorShape(output_name_);
    auto volume = [](nvinfer1::Dims d){
          size_t v = 1;
          for (int i = 0; i < d.nbDims; ++i)
          {
            if (d.d[i] < 1)
              throw std::runtime_error("Dynamic TensorRT shapes unsupported");
            v *= d.d[i];
          }
          return v;
        };
    if (volume(in) != kObservationSize || volume(out) != kJoints)
      throw std::runtime_error("Engine shape does not match 49 inputs/12 outputs");
    check(cudaMallocHost(reinterpret_cast<void **>(&host_in_),
        kObservationSize * sizeof(float)), "cudaMallocHost input");
    check(cudaMallocHost(reinterpret_cast<void **>(&host_out_),
        kJoints * sizeof(float)), "cudaMallocHost output");
    check(cudaMalloc(&device_in_, kObservationSize * sizeof(float)), "cudaMalloc input");
    check(cudaMalloc(&device_out_, kJoints * sizeof(float)), "cudaMalloc output");
    check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    check(cudaEventCreate(&event_start_), "cudaEventCreate start");
    check(cudaEventCreate(&event_h2d_), "cudaEventCreate h2d");
    check(cudaEventCreate(&event_execute_), "cudaEventCreate execute");
    check(cudaEventCreate(&event_d2h_), "cudaEventCreate d2h");
    if (!context_->setTensorAddress(input_name_,
      device_in_) || !context_->setTensorAddress(output_name_, device_out_))
      throw std::runtime_error("TensorRT binding failed");
  }
  ~TensorRtPolicy()
  {
    if (event_start_)
      cudaEventDestroy(event_start_);
    if (event_h2d_)
      cudaEventDestroy(event_h2d_);
    if (event_execute_)
      cudaEventDestroy(event_execute_);
    if (event_d2h_)
      cudaEventDestroy(event_d2h_);
    if (stream_)
      cudaStreamDestroy(stream_);
    if (device_in_)
      cudaFree(device_in_);
    if (device_out_)
      cudaFree(device_out_);
    if (host_in_)
      cudaFreeHost(host_in_);
    if (host_out_)
      cudaFreeHost(host_out_);
  }
  TrtTiming infer(
      const std::array<float, kObservationSize> &in,
      std::array<float, kJoints> &out)
  {
    TrtTiming t;
    auto a = Clock::now();
    std::copy(in.begin(), in.end(), host_in_);
    auto b = Clock::now();
    check(cudaEventRecord(event_start_, stream_), "event start");
    check(cudaMemcpyAsync(device_in_, host_in_, sizeof(in), cudaMemcpyHostToDevice, stream_),
        "H2D");
    check(cudaEventRecord(event_h2d_, stream_), "event h2d");
    auto c = Clock::now();
    if (!context_->enqueueV3(stream_))
      throw std::runtime_error("TensorRT enqueue failed");
    auto d = Clock::now();
    check(cudaEventRecord(event_execute_, stream_), "event execute");
    check(cudaMemcpyAsync(host_out_, device_out_, sizeof(out), cudaMemcpyDeviceToHost, stream_),
        "D2H");
    check(cudaEventRecord(event_d2h_, stream_), "event d2h");
    auto e = Clock::now();
    check(cudaStreamSynchronize(stream_), "sync");
    auto f = Clock::now();
    std::copy(host_out_, host_out_ + kJoints, out.begin());
    t.host_input = ms(b - a);
    t.enqueue = ms(d - c);
    t.sync_wait = ms(f - e);
    cudaEventElapsedTime(&t.h2d, event_start_, event_h2d_);
    cudaEventElapsedTime(&t.execute, event_h2d_, event_execute_);
    cudaEventElapsedTime(&t.d2h, event_execute_, event_d2h_);
    return t;
  }
 private:
  static void check(cudaError_t e, const char *w)
  {
    if (e != cudaSuccess)
      throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
  }
  TrtLogger logger_;
  // TensorRT 10 interfaces use normal virtual destructors; destroy() was removed.
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  const char *input_name_{}, *output_name_{};
  float *host_in_{}, *host_out_{};
  void *device_in_{}, *device_out_{};
  cudaStream_t stream_{};
  cudaEvent_t event_start_{}, event_h2d_{}, event_execute_{}, event_d2h_{};
};
}

class LowLevelPolicyNode final : public rclcpp::Node
{
 public:
  LowLevelPolicyNode() : Node("low_level_policy_node")
  {
    declare_parameters();
    load_parameters();
    init_command();
    parameter_callback_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &p){
        return update_parameters(p);
      });
    if (control_mode_ != "hierarchical_lowcmd")
    {
      RCLCPP_INFO(get_logger(),
          "Low-level motor control disabled for control_mode=%s",
          control_mode_.c_str());
      return;
    }
    active_ = true;

    // Motor command/state topics use normal ROS messages. Control state and
    // command enable use transient-local QoS so restarted peers see the latest gate.
    lowcmd_pub_ = create_publisher<unitree_go::msg::LowCmd>(lowcmd_topic_, 10);
    relay_pub_ = create_publisher<unitree_go::msg::LowState>(inekf_topic_, 10);
    state_pub_ = create_publisher<fetch_interfaces::msg::ControlState>("/go2_fetch/control_state",
      rclcpp::QoS(1).reliable().transient_local());
    timing_pub_ = create_publisher<fetch_interfaces::msg::ControlTiming>(
      "/go2_fetch/control_timing",
      10);
    lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(lowstate_topic_,
      rclcpp::SensorDataQoS(),
      [this](unitree_go::msg::LowState::SharedPtr m){
        {std::lock_guard<std::mutex> l(state_mutex_);
         lowstate_ = *m;
         have_state_ = m->tick != 0;
        } keys_.store(uint16_t(m->wireless_remote[2]) | (uint16_t(m->wireless_remote[3]) << 8U));
        relay_pub_->publish(*m);
      });
    cmd_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>("/go2_fetch/high_level_cmd",
      10,
      [this](geometry_msgs::msg::TwistStamped::SharedPtr m){
        std::array<float, 3> v{float(m->twist.linear.x), float(m->twist.linear.y),
                               float(m->twist.angular.z)};
        std::lock_guard<std::mutex> l(command_mutex_);
        if (finite3(v))
        {
          command_ = v;
          command_received_ = Clock::now();
          have_command_ = true;
        }
      });
    enabled_sub_ = create_subscription<std_msgs::msg::Bool>("/go2_fetch/high_level_cmd_enabled",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](std_msgs::msg::Bool::SharedPtr m){
        command_enabled_.store(m->data);
      });

    // TensorRT owns a dedicated worker; the control thread owns all /lowcmd writes.
    policy_ = std::make_unique<TensorRtPolicy>(engine_path_);
    // Warm the engine before the control thread starts so CUDA initialization
    // cannot consume the first control-cycle deadline.
    {std::array<float, kObservationSize> warmup{};
     std::array<float, kJoints> output{};
     for (int i = 0; i < 50; ++i)
       (void)policy_->infer(warmup, output);
    }
    inference_thread_ = std::thread([this] {
        inference_loop();
      });
    control_thread_ = std::thread([this] {
        control_loop();
      });
    publish_state("waiting for lowstate");
  }
  ~LowLevelPolicyNode() override
  {
    if (!active_)
      return;
    stop_.store(true);
    inference_cv_.notify_all();
    if (control_thread_.joinable())
      control_thread_.join();
    publish_safe_hold();
    // A CUDA call cannot be cancelled safely. Give a healthy worker time to
    // finish; a truly wedged worker requires process restart after safe hold.
    for (int i = 0; i < 100 && inference_busy_.load(); ++i)
      std::this_thread::sleep_for(1ms);
    if (inference_busy_.load())
    {
      RCLCPP_ERROR(get_logger(), "TensorRT worker did not stop; exiting process after safe hold");
      std::_Exit(2);
    }
    if (inference_thread_.joinable())
      inference_thread_.join();
  }
 private:
  void declare_parameters()
  {
    declare_parameter("control_mode", "hierarchical_lowcmd");
    declare_parameter("send_commands", false);
    declare_parameter("low_level_policy_path", "");
    declare_parameter("lowcmd_topic", "/lowcmd");
    declare_parameter("lowstate_topic", "/lowstate");
    declare_parameter("inekf_lowstate_topic", "/inekf_lowstate");
    declare_parameter("control_rate_hz", 50.0);
    declare_parameter("command_timeout_s", 0.25);
    declare_parameter("expected_inference_ms", 20.0);
    declare_parameter("inference_timeout_factor", 2.0);
    declare_parameter("max_consecutive_deadline_misses", 3);
    declare_parameter("action_scale", 0.25);
    declare_parameter("ang_vel_scale", 0.2);
    declare_parameter("dof_pos_scale", 1.0);
    declare_parameter("dof_vel_scale", 0.05);
    declare_parameter("foot_force_offset", std::vector<double>{4, 1, 5, 5});
    declare_parameter("foot_force_scale", 75.0);
    declare_parameter("foot_force_clip_max", 150.0);
    declare_parameter("leg_joint2motor_idx", std::vector<int64_t>{3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11,
                                                                  8});
    declare_parameter("default_angles", std::vector<double>{.1, -.1, .1, -.1, .8, .8, 1, 1, -1.5,
                                                            -1.5, -1.5, -1.5});
    declare_parameter("kps", std::vector<double>(12, 25));
    declare_parameter("kds", std::vector<double>(12,
      .5));
    declare_parameter("torque_limits", std::vector<double>{22, 22, 20, 20, 22, 22, 20, 22, 22, 22,
                                                           20, 22});
    declare_parameter("max_cmd", std::vector<double>{.4, .4, .5});
    declare_parameter("safe_hold_kp", 5.0);
    declare_parameter("safe_hold_kd", 1.0);
  }
  template<class T, size_t N> void array_param(const char *n, std::array<T, N> &out)
  {
    auto v = get_parameter(n).as_double_array();
    if (v.size() != N)
      throw std::runtime_error(std::string(n) + " must have " + std::to_string(N) + " values");
    for (size_t i = 0; i < N; ++i)
      out[i] = T(v[i]);
  }
  // Runtime-safe values are copied into the cached control configuration under
  // one lock. Engine paths, topics, and loop rate remain restart-only.
  rcl_interfaces::msg::SetParametersResult update_parameters(
      const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    std::lock_guard<std::mutex> lock(config_mutex_);
    try
    {
      for (const auto &p : parameters)
      {
           const auto &n = p.get_name();
           if (n == "command_timeout_s")
             timeout_s_ = p.as_double();
           else if (n == "expected_inference_ms")
             inference_expected_ms_ = p.as_double();
           else if (n == "inference_timeout_factor")
             inference_factor_ = p.as_double();
           else if (n == "max_consecutive_deadline_misses")
             max_misses_ = p.as_int();
           else if (n == "action_scale")
             action_scale_ = p.as_double();
           else if (n == "ang_vel_scale")
             ang_scale_ = p.as_double();
           else if (n == "dof_pos_scale")
             pos_scale_ = p.as_double();
           else if (n == "dof_vel_scale")
             vel_scale_ = p.as_double();
           else if (n == "foot_force_scale")
             force_scale_ = p.as_double();
           else if (n == "foot_force_clip_max")
             force_clip_ = p.as_double();
           else if (n == "safe_hold_kp")
             safe_kp_ = p.as_double();
           else if (n == "safe_hold_kd")
             safe_kd_ = p.as_double();
           else if (n == "default_angles" || n == "kps" || n == "kds" || n == "torque_limits" ||
               n == "max_cmd" || n == "foot_force_offset")
           {
             auto v = p.as_double_array();
             size_t expected = n == "max_cmd"?3:n == "foot_force_offset"?4:12;
             if (v.size() != expected)
               throw std::runtime_error(n + " has wrong length");
             if (n == "default_angles")
               for (size_t i = 0; i < 12; ++i)
                 defaults_[i] = v[i];
             else if (n == "kps")
               for (size_t i = 0; i < 12; ++i)
                 kps_[i] = v[i];
             else if (n == "kds")
               for (size_t i = 0; i < 12; ++i)
                 kds_[i] = v[i];
             else if (n == "torque_limits")
               for (size_t i = 0; i < 12; ++i)
                 limits_[i] = v[i];
             else if (n == "max_cmd")
               for (size_t i = 0; i < 3; ++i)
                 max_cmd_[i] = v[i];
             else for (size_t i = 0; i < 4; ++i)
                 force_offset_[i] = v[i];
           }
           else
           {
             result.reason = n + " is restart-only";
             return result;
           }
      }
      if (timeout_s_ < 0 || inference_expected_ms_ <= 0 || inference_factor_ <= 0 ||
          force_scale_ <= 0 || force_clip_ <= 0 || max_misses_ < 1)
      {
        throw std::runtime_error("updated values must be positive");
      }
      inference_timeout_ms_ = inference_expected_ms_ * inference_factor_;
      result.successful = true;
      return result;
    }
    catch (const std::exception &error)
    {
      result.reason = error.what();
      return result;
    }
  }
  void load_parameters()
  {
    control_mode_ = get_parameter("control_mode").as_string();
    commands_enabled_ = get_parameter("send_commands").as_bool();
    engine_path_ = get_parameter("low_level_policy_path").as_string();
    if (control_mode_ == "hierarchical_lowcmd" && engine_path_.empty())
      throw std::runtime_error("low_level_policy_path is required");
    lowcmd_topic_ = get_parameter("lowcmd_topic").as_string();
    lowstate_topic_ = get_parameter("lowstate_topic").as_string();
    inekf_topic_ = get_parameter("inekf_lowstate_topic").as_string();
    rate_ = get_parameter("control_rate_hz").as_double();
    timeout_s_ = get_parameter("command_timeout_s").as_double();
    inference_expected_ms_ = get_parameter("expected_inference_ms").as_double();
    inference_factor_ = get_parameter("inference_timeout_factor").as_double();
    inference_timeout_ms_ = inference_expected_ms_ * inference_factor_;
    max_misses_ = get_parameter("max_consecutive_deadline_misses").as_int();
    action_scale_ = get_parameter("action_scale").as_double();
    ang_scale_ = get_parameter("ang_vel_scale").as_double();
    pos_scale_ = get_parameter("dof_pos_scale").as_double();
    vel_scale_ = get_parameter("dof_vel_scale").as_double();
    force_scale_ = get_parameter("foot_force_scale").as_double();
    force_clip_ = get_parameter("foot_force_clip_max").as_double();
    safe_kp_ = get_parameter("safe_hold_kp").as_double();
    safe_kd_ = get_parameter("safe_hold_kd").as_double();
    array_param("default_angles", defaults_);
    array_param("kps", kps_);
    array_param("kds", kds_);
    array_param("torque_limits", limits_);
    array_param("max_cmd", max_cmd_);
    array_param("foot_force_offset", force_offset_);
    auto idx = get_parameter("leg_joint2motor_idx").as_integer_array();
    if (idx.size() != 12)
      throw std::runtime_error("leg_joint2motor_idx must have 12 values");
    for (size_t i = 0; i < 12; ++i)
      joint_map_[i] = size_t(idx[i]);
    if (rate_ <= 0 || force_scale_ <= 0 || inference_timeout_ms_ <= 0)
      throw std::runtime_error("invalid positive control parameter");
  }
  void init_command()
  {
    cmd_.head = {0xFE, 0xEF};
    cmd_.level_flag = 0xFF;
    for (auto &m:cmd_.motor_cmd)
    {
      m.mode = 0x0A;
      m.q = 2.146e9f;
      m.dq = 16000;
      m.kp = m.kd = m.tau = 0;
    }
  }
  bool pressed(size_t bit)const
  {
    return (keys_.load() & (1U << bit)) != 0;
  }
  void publish_state(const std::string &detail)
  {
    fetch_interfaces::msg::ControlState m;
    m.stamp = now();
    m.state = state_.load();
    m.detail = detail;
    state_pub_->publish(m);
  }
  void transition(uint8_t s, const char *d)
  {
    state_.store(s);
    phase_start_ = Clock::now();
    publish_state(d);
    RCLCPP_INFO(get_logger(), "Control state: %s", d);
  }
  // TensorRT runs on a separate worker so a stuck CUDA call cannot prevent the
  // control thread from switching to FAULT and publishing safe-hold commands.
  void inference_loop()
  {
    while (!stop_.load())
    {
      std::unique_lock<std::mutex> lock(inference_mutex_);
      inference_cv_.wait(lock, [this] {
          return stop_.load() || inference_pending_;
        });
      if (stop_.load())
        return;
      auto in = inference_input_;
      inference_pending_ = false;
      inference_busy_.store(true);
      lock.unlock();
      try
      {
        std::array<float, kJoints> out{};
        auto timing = policy_->infer(in, out);
        lock.lock();
        inference_output_ = out;
        inference_timing_ = timing;
        inference_ready_ = true;
        inference_error_.clear();
        lock.unlock();
      }
      catch (const std::exception &error)
      {
        lock.lock();
        inference_error_ = error.what();
        inference_ready_ = true;
        lock.unlock();
      }
      inference_busy_.store(false);
      inference_done_cv_.notify_one();
    }
  }
  bool infer(
      const std::array<float, kObservationSize> &obs,
      std::array<float, kJoints> &out,
      TrtTiming &timing)
  {
    std::unique_lock<std::mutex> lock(inference_mutex_);
    if (inference_busy_.load())
      return false;
    inference_input_ = obs;
    inference_ready_ = false;
    inference_pending_ = true;
    inference_cv_.notify_one();
    if (!inference_done_cv_.wait_for(lock,
      std::chrono::duration<double, std::milli>(inference_timeout_ms_), [this] {
        return inference_ready_;
      }))
      return false;
    if (!inference_error_.empty())
      throw std::runtime_error(inference_error_);
    out = inference_output_;
    timing = inference_timing_;
    return true;
  }
  // Absolute steady-clock deadlines avoid the drift produced by repeated
  // sleep_for() calls. Repeated overruns are treated as a control fault.
  void control_loop()
  {
    const auto period =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / rate_));
    auto deadline = Clock::now();
    while (!stop_.load() && rclcpp::ok())
    {
      deadline += period;
      const auto start = Clock::now();
      unitree_go::msg::LowState s;
      bool have;
      {
       std::lock_guard<std::mutex> lock(state_mutex_);
        s = lowstate_;
        have = have_state_;
      }
      if (!have)
      {
        std::this_thread::sleep_until(deadline);
        continue;
      }
      try
      {
        step(s, start, deadline);
      }
      catch (const std::exception &error)
      {
        fault(error.what());
      }
      auto nowt = Clock::now();
      if (nowt > deadline)
      {
        if (++misses_ >= max_misses_)
          fault("repeated 50 Hz deadline misses");
        deadline = nowt;
      }
      else
      {
        misses_ = 0;
        std::this_thread::sleep_until(deadline);
      }
    }
  }
  // State transitions are intentionally local to this node. High-level command
  // generation cannot bypass START/A or interfere with pose interpolation.
  void step(
      const unitree_go::msg::LowState &s,
      Clock::time_point start,
      Clock::time_point deadline)
  {
    std::lock_guard<std::mutex> config_lock(config_mutex_);
    const auto state = state_.load();
    if (state == fetch_interfaces::msg::ControlState::ZERO_TORQUE)
    {
      zero_torque();
      if (pressed(kStartButton))
        transition(fetch_interfaces::msg::ControlState::MOVE_TO_DEFAULT, "MOVE_TO_DEFAULT");
      return;
    }
    if (state == fetch_interfaces::msg::ControlState::MOVE_TO_DEFAULT)
    {
      move_default(s);
      return;
    }
    if (state == fetch_interfaces::msg::ControlState::WAIT_FOR_A)
    {
      hold_defaults(60, 5);
      if (pressed(kAButton))
        transition(fetch_interfaces::msg::ControlState::RUNNING, "RUNNING");
      publish();
      return;
    }
    if (state == fetch_interfaces::msg::ControlState::FAULT)
    {
      publish_safe_hold(s);
      return;
    }
    if (pressed(kSelectButton))
    {
      ground_start_ = s;
      transition_ground_ = true;
      transition(fetch_interfaces::msg::ControlState::MOVE_TO_DEFAULT, "MOVE_TO_GROUND");
      return;
    }
    run_policy(s, start, deadline);
  }
  // MOVE_TO_DEFAULT is also reused for the SELECT-to-ground interpolation.
  // transition_ground_ distinguishes the two trajectories.
  void move_default(const unitree_go::msg::LowState &s)
  {
    if (transition_ground_)
    {
      double p = std::min(1.0,
        std::chrono::duration<double>(Clock::now() - phase_start_).count() / 0.6);
      static const std::array<double, 12> lie{0, 1.36, -2.65, 0, 1.36, -2.65, -.2, 1.36, -2.65, .2,
                                              1.36, -2.65};
      for (size_t i = 0; i < 12; ++i)
        set_motor(i, (1 - p) * ground_start_.motor_state[i].q + p * lie[i], 60, 5);
      publish();
      if (p >= 1)
      {
        transition_ground_ = false;
        transition(fetch_interfaces::msg::ControlState::ZERO_TORQUE, "ZERO_TORQUE");
      }
      return;
    }
    if (!startup_captured_)
    {
      startup_ = s;
      startup_captured_ = true;
    }
    double t = std::chrono::duration<double>(Clock::now() - phase_start_).count();
    static const std::array<double, 12> crouch{0, 1.36, -2.65, 0, 1.36, -2.65, -.2, 1.36, -2.65, .2,
                                               1.36, -2.65};
    for (size_t i = 0; i < 12; ++i)
    {
      const auto mapping = std::find(joint_map_.begin(), joint_map_.end(), i);
      const size_t policy_index = std::distance(joint_map_.begin(), mapping);
      const double default_position = defaults_[policy_index];

      double target_position = default_position;
      if (t < 1.0)
      {
        // Phase 1: measured pose -> crouched pose.
        target_position = (1.0 - t) * startup_.motor_state[i].q + t * crouch[i];
      }
      else if (t < 2.0)
      {
        // Phase 2: crouched pose -> policy default pose.
        const double blend = t - 1.0;
        target_position = (1.0 - blend) * crouch[i] + blend * default_position;
      }
      // Phase 3 holds the default pose until the four-second sequence ends.
      set_motor(i, target_position, 60, 5);
    }
    publish();
    if (t >= 4)
    {
      startup_captured_ = false;
      transition(fetch_interfaces::msg::ControlState::WAIT_FOR_A, "WAIT_FOR_A");
    }
  }
  void run_policy(const unitree_go::msg::LowState &s,
      Clock::time_point start,
      Clock::time_point deadline)
  {
    auto t0 = Clock::now();
    std::array<float, 3> command{};
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      const double command_age =
          std::chrono::duration<double>(Clock::now() - command_received_).count();
      if (command_enabled_.load() && have_command_ && command_age <= timeout_s_)
      {
        command = command_;
      }
      // Otherwise command remains zero. The locomotion policy keeps balancing.
    }
    for (size_t i = 0; i < 3; ++i)
    {
      command[i] = std::clamp(command[i], -float(max_cmd_[i]), float(max_cmd_[i]));
    }

    // Observation layout must exactly match training:
    // angular velocity, projected gravity, velocity command, joint position
    // error, joint velocity, reordered foot forces, previous action.
    std::array<float, kObservationSize> obs{};
    size_t n = 0;
    for (float v:s.imu_state.gyroscope)
      obs[n++] = v * ang_scale_;
    auto &q = s.imu_state.quaternion;
    obs[n++] = 2 * (-q[3] * q[1] + q[0] * q[2]);
    obs[n++] = -2 * (q[3] * q[2] + q[0] * q[1]);
    obs[n++] = 1 - 2 * (q[0] * q[0] + q[3] * q[3]);
    for (float v:command)
      obs[n++] = v;
    for (size_t i = 0; i < 12; ++i)
      obs[n++] = (s.motor_state[joint_map_[i]].q - defaults_[i]) * pos_scale_;
    for (size_t i = 0; i < 12; ++i)
      obs[n++] = s.motor_state[joint_map_[i]].dq * vel_scale_;
    const size_t reorder[4]{1, 0, 3, 2};
    for (size_t i:reorder)
      obs[n++] =
          std::clamp(double(s.foot_force[i]) - force_offset_[i], 0.0, force_clip_) / force_scale_;
    for (float v:last_action_)
      obs[n++] = v;
    auto obs_done = Clock::now();
    std::array<float, 12> action{};
    TrtTiming trt{};
    auto inf_start = Clock::now();
    if (!infer(obs, action, trt))
    {
      fault("TensorRT inference watchdog timeout");
      return;
    }
    auto inf_done = Clock::now();
    last_action_ = action;
    for (size_t i = 0; i < 12; ++i)
      set_motor(joint_map_[i], action[i] * action_scale_ + defaults_[i], kps_[i], kds_[i]);
    auto motor_done = Clock::now();
    auto torque_start = Clock::now();
    apply_limits(s);
    auto torque_done = Clock::now();
    set_crc(cmd_);
    auto crc_done = Clock::now();
    lowcmd_pub_->publish(cmd_);
    auto pub_done = Clock::now();
    fetch_interfaces::msg::ControlTiming tm;
    tm.stamp = now();
    tm.step = ++step_;
    tm.input_ms = ms(t0 - start);
    tm.observation_ms = ms(obs_done - t0);
    tm.inference_ms = ms(inf_done - inf_start);
    tm.trt_host_input_ms = trt.host_input;
    tm.trt_h2d_ms = trt.h2d;
    tm.trt_enqueue_ms = trt.enqueue;
    tm.trt_execute_ms = trt.execute;
    tm.trt_d2h_ms = trt.d2h;
    tm.trt_sync_wait_ms = trt.sync_wait;
    tm.motor_build_ms = ms(motor_done - inf_done);
    tm.torque_limit_ms = ms(torque_done - torque_start);
    tm.crc_ms = ms(crc_done - torque_done);
    tm.publish_ms = ms(pub_done - crc_done);
    tm.total_ms = ms(pub_done - start);
    tm.deadline_lateness_ms = std::max(0.0, ms(pub_done - deadline));
    tm.consecutive_deadline_misses = misses_;
    timing_pub_->publish(tm);
  }
  void set_motor(size_t i, double q, double kp, double kd)
  {
    auto &m = cmd_.motor_cmd[i];
    m.q = q;
    m.dq = 0;
    m.kp = kp;
    m.kd = kd;
    m.tau = 0;
  }
  void hold_defaults(double kp, double kd)
  {
    for (size_t i = 0; i < 12; ++i)
      set_motor(joint_map_[i], defaults_[i], kp, kd);
  }
  void zero_torque()
  {
    for (size_t i = 0; i < 12; ++i)
      set_motor(i, 0, 0, 0);
    publish();
  }
  void apply_limits(const unitree_go::msg::LowState &s)
  {
    // Clamp estimated PD + feed-forward torque by adjusting the position target.
    // This preserves the configured gains and zero feed-forward torque.
    for (size_t p = 0; p < 12; ++p)
    {
      size_t i = joint_map_[p];
      auto &m = cmd_.motor_cmd[i];
      double tau = m.kp * (m.q - s.motor_state[i].q) + m.kd * (m.dq - s.motor_state[i].dq) + m.tau;
      double c = std::clamp(tau, -limits_[p], limits_[p]);
      if (std::abs(tau - c) > 1e-6 && std::abs(m.kp) > 1e-6)
        m.q = s.motor_state[i].q + (c - m.kd * (m.dq - s.motor_state[i].dq) - m.tau) / m.kp;
    }
  }
  void publish()
  {
    if (!commands_enabled_)
      return;
    set_crc(cmd_);
    lowcmd_pub_->publish(cmd_);
  }
  void publish_safe_hold()
  {
    unitree_go::msg::LowState s;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      s = lowstate_;
    }
    publish_safe_hold(s);
  }
  void publish_safe_hold(const unitree_go::msg::LowState &s)
  {
    for (size_t i = 0; i < 12; ++i)
      set_motor(i, s.motor_state[i].q, safe_kp_, safe_kd_);
    publish();
  }
  void fault(const std::string &why)
  {
    if (state_.exchange(fetch_interfaces::msg::ControlState::FAULT) !=
        fetch_interfaces::msg::ControlState::FAULT)
    {
      RCLCPP_ERROR(get_logger(), "Control fault: %s", why.c_str());
      publish_state(why);
    }
  }

  // Cached configuration. The control loop never calls get_parameter().
  std::string control_mode_, engine_path_, lowcmd_topic_, lowstate_topic_, inekf_topic_;
  bool active_{false}, commands_enabled_{false};
  double rate_{50}, timeout_s_{.25}, inference_expected_ms_{20}, inference_factor_{2},
  inference_timeout_ms_{40}, action_scale_{.25}, ang_scale_{.2}, pos_scale_{1}, vel_scale_{.05},
  force_scale_{75}, force_clip_{150}, safe_kp_{5}, safe_kd_{1};
  int64_t max_misses_{3};
  std::array<size_t, 12>joint_map_{};
  std::array<double, 12>defaults_{}, kps_{}, kds_{}, limits_{};
  std::array<double, 3>max_cmd_{};
  std::array<double, 4>force_offset_{};
  // Latest robot state and reusable command message.
  unitree_go::msg::LowCmd cmd_;
  unitree_go::msg::LowState lowstate_, startup_, ground_start_;
  std::mutex state_mutex_, command_mutex_, inference_mutex_, config_mutex_;
  bool have_state_{false}, startup_captured_{false}, transition_ground_{false},
  have_command_{false};
  std::atomic<uint16_t> keys_{0};
  std::array<float, 3>command_{};
  Clock::time_point command_received_{}, phase_start_{Clock::now()};
  // Cross-thread state shared by ROS callbacks, control, and inference workers.
  std::atomic<uint8_t>state_{fetch_interfaces::msg::ControlState::ZERO_TORQUE};
  std::atomic<bool>command_enabled_{false}, stop_{false}, inference_busy_{false};
  std::array<float, kObservationSize>inference_input_{};
  std::array<float, 12>inference_output_{}, last_action_{};
  TrtTiming inference_timing_{};
  bool inference_pending_{false}, inference_ready_{false};
  std::string inference_error_;
  std::condition_variable inference_cv_, inference_done_cv_;
  std::thread inference_thread_, control_thread_;
  uint64_t step_{0};
  uint32_t misses_{0};
  std::unique_ptr<TensorRtPolicy>policy_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Publisher<unitree_go::msg::LowState>::SharedPtr relay_pub_;
  rclcpp::Publisher<fetch_interfaces::msg::ControlState>::SharedPtr state_pub_;
  rclcpp::Publisher<fetch_interfaces::msg::ControlTiming>::SharedPtr timing_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enabled_sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
};
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<fetch_low_level::LowLevelPolicyNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception &error)
  {
    fprintf(stderr, "low_level_policy_node fatal: %s\n", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
