#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fetch_interfaces/msg/control_state.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <unitree_api/msg/request.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std::chrono_literals;

namespace fetch_policy {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr size_t kJoints = 12;
constexpr size_t kObservationSize = 52;
constexpr size_t kActionSize = 3;
constexpr int64_t kSportStopMove = 1003;
constexpr int64_t kSportMove = 1008;
constexpr double kPi = 3.14159265358979323846;

double steady_seconds()
{
  return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
}

double milliseconds(SteadyClock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

float clipped(float value, float low = -100.0F, float high = 100.0F)
{
  return std::clamp(value, low, high);
}

double normalize_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

std::filesystem::path expand_user_path(const std::string & value)
{
  if (value == "~" || value.rfind("~/", 0) == 0) {
    if (const char * home = std::getenv("HOME")) {
      return std::filesystem::path(home) / (value.size() > 2 ? value.substr(2) : "");
    }
  }
  return std::filesystem::path(value);
}

struct TrtTiming
{
  float host_input{};
  float h2d{};
  float enqueue{};
  float execute{};
  float d2h{};
  float sync_wait{};
  float total{};
};

class TrtLogger final : public nvinfer1::ILogger
{
  void log(Severity severity, const char * message) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "TensorRT: %s\n", message);
    }
  }
};

class TensorRtPolicy
{
public:
  explicit TensorRtPolicy(const std::string & path)
  {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
      throw std::runtime_error("Cannot open TensorRT engine: " + path);
    }
    const auto end = stream.tellg();
    if (end <= 0) {
      throw std::runtime_error("TensorRT engine is empty: " + path);
    }
    std::vector<char> bytes(static_cast<size_t>(end));
    stream.seekg(0);
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("Cannot create TensorRT runtime");
    }
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) {
      throw std::runtime_error("Cannot deserialize TensorRT engine: " + path);
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("Cannot create TensorRT execution context");
    }

    for (int32_t index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char * name = engine_->getIOTensorName(index);
      const auto mode = engine_->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        if (input_name_ != nullptr) {
          throw std::runtime_error("High-level engine must have exactly one input");
        }
        input_name_ = name;
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        if (output_name_ != nullptr) {
          throw std::runtime_error("High-level engine must have exactly one output");
        }
        output_name_ = name;
      }
    }
    if (input_name_ == nullptr || output_name_ == nullptr) {
      throw std::runtime_error("High-level engine must have one input and one output");
    }
    if (engine_->getTensorDataType(input_name_) != nvinfer1::DataType::kFLOAT ||
      engine_->getTensorDataType(output_name_) != nvinfer1::DataType::kFLOAT)
    {
      throw std::runtime_error("High-level engine input and output must be float32");
    }
    if (volume(engine_->getTensorShape(input_name_)) != kObservationSize ||
      volume(engine_->getTensorShape(output_name_)) != kActionSize)
    {
      throw std::runtime_error("High-level engine shape does not match 52 inputs/3 outputs");
    }

    check(cudaMallocHost(reinterpret_cast<void **>(&host_input_), sizeof(float) * kObservationSize),
      "cudaMallocHost input");
    check(cudaMallocHost(reinterpret_cast<void **>(&host_output_), sizeof(float) * kActionSize),
      "cudaMallocHost output");
    check(cudaMalloc(&device_input_, sizeof(float) * kObservationSize), "cudaMalloc input");
    check(cudaMalloc(&device_output_, sizeof(float) * kActionSize), "cudaMalloc output");
    check(cudaStreamCreate(&cuda_stream_), "cudaStreamCreate");
    check(cudaEventCreate(&event_start_), "cudaEventCreate start");
    check(cudaEventCreate(&event_h2d_), "cudaEventCreate h2d");
    check(cudaEventCreate(&event_execute_), "cudaEventCreate execute");
    check(cudaEventCreate(&event_d2h_), "cudaEventCreate d2h");
    if (!context_->setTensorAddress(input_name_, device_input_) ||
      !context_->setTensorAddress(output_name_, device_output_))
    {
      throw std::runtime_error("Failed to bind TensorRT policy buffers");
    }

    std::array<float, kObservationSize> warmup_input{};
    std::array<float, kActionSize> warmup_output{};
    for (int index = 0; index < 50; ++index) {
      (void)infer(warmup_input, warmup_output);
    }
  }

  ~TensorRtPolicy()
  {
    if (event_start_) {cudaEventDestroy(event_start_);}
    if (event_h2d_) {cudaEventDestroy(event_h2d_);}
    if (event_execute_) {cudaEventDestroy(event_execute_);}
    if (event_d2h_) {cudaEventDestroy(event_d2h_);}
    if (cuda_stream_) {cudaStreamDestroy(cuda_stream_);}
    if (device_input_) {cudaFree(device_input_);}
    if (device_output_) {cudaFree(device_output_);}
    if (host_input_) {cudaFreeHost(host_input_);}
    if (host_output_) {cudaFreeHost(host_output_);}
  }

  TrtTiming infer(
    const std::array<float, kObservationSize> & input,
    std::array<float, kActionSize> & output)
  {
    const auto call_start = SteadyClock::now();
    std::copy(input.begin(), input.end(), host_input_);
    const auto host_done = SteadyClock::now();
    check(cudaEventRecord(event_start_, cuda_stream_), "record start");
    check(cudaMemcpyAsync(
        device_input_, host_input_, sizeof(input), cudaMemcpyHostToDevice, cuda_stream_), "H2D");
    check(cudaEventRecord(event_h2d_, cuda_stream_), "record h2d");
    const auto enqueue_start = SteadyClock::now();
    if (!context_->enqueueV3(cuda_stream_)) {
      throw std::runtime_error("TensorRT policy inference failed");
    }
    const auto enqueue_done = SteadyClock::now();
    check(cudaEventRecord(event_execute_, cuda_stream_), "record execute");
    check(cudaMemcpyAsync(
        host_output_, device_output_, sizeof(output), cudaMemcpyDeviceToHost, cuda_stream_), "D2H");
    check(cudaEventRecord(event_d2h_, cuda_stream_), "record d2h");
    const auto queued_done = SteadyClock::now();
    check(cudaStreamSynchronize(cuda_stream_), "stream synchronize");
    const auto sync_done = SteadyClock::now();
    std::copy(host_output_, host_output_ + kActionSize, output.begin());

    TrtTiming timing;
    timing.host_input = static_cast<float>(milliseconds(host_done - call_start));
    timing.enqueue = static_cast<float>(milliseconds(enqueue_done - enqueue_start));
    timing.sync_wait = static_cast<float>(milliseconds(sync_done - queued_done));
    timing.total = static_cast<float>(milliseconds(sync_done - call_start));
    check(cudaEventElapsedTime(&timing.h2d, event_start_, event_h2d_), "timing h2d");
    check(cudaEventElapsedTime(&timing.execute, event_h2d_, event_execute_), "timing execute");
    check(cudaEventElapsedTime(&timing.d2h, event_execute_, event_d2h_), "timing d2h");
    return timing;
  }

private:
  static size_t volume(nvinfer1::Dims dimensions)
  {
    size_t result = 1;
    for (int32_t index = 0; index < dimensions.nbDims; ++index) {
      if (dimensions.d[index] < 1) {
        throw std::runtime_error("Dynamic TensorRT policy shapes are not supported");
      }
      result *= static_cast<size_t>(dimensions.d[index]);
    }
    return result;
  }

  static void check(cudaError_t error, const char * operation)
  {
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
  }

  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  const char * input_name_{};
  const char * output_name_{};
  float * host_input_{};
  float * host_output_{};
  void * device_input_{};
  void * device_output_{};
  cudaStream_t cuda_stream_{};
  cudaEvent_t event_start_{};
  cudaEvent_t event_h2d_{};
  cudaEvent_t event_execute_{};
  cudaEvent_t event_d2h_{};
};

struct RemoteController
{
  uint16_t keys{};
  float lx{};
  float ly{};
  float rx{};
  float ry{};

  template<typename Container>
  void set(const Container & data)
  {
    if (data.size() < 24) {
      keys = 0;
      lx = ly = rx = ry = 0.0F;
      return;
    }
    keys = static_cast<uint16_t>(data[2]) |
      (static_cast<uint16_t>(data[3]) << 8U);
    std::memcpy(&lx, data.data() + 4, sizeof(float));
    std::memcpy(&rx, data.data() + 8, sizeof(float));
    std::memcpy(&ry, data.data() + 12, sizeof(float));
    std::memcpy(&ly, data.data() + 20, sizeof(float));
  }

  bool pressed(size_t index) const
  {
    return (keys & (1U << index)) != 0U;
  }
};

struct TimedCubeState
{
  double stamp{};
  std::array<float, 2> position{};
  std::array<float, 2> velocity{};
};

struct RobotSnapshot
{
  unitree_go::msg::LowState low_state;
  RemoteController remote;
  std::array<float, 2> robot_position{};
  std::array<float, 2> robot_velocity_world{};
  std::array<float, 4> base_velocity{};
  std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};
  double yaw{};
  double last_robot_odom{-std::numeric_limits<double>::infinity()};
  bool have_low_state{};
};

}  // namespace

class HighLevelPolicyNode final : public rclcpp::Node
{
public:
  HighLevelPolicyNode()
  : Node("high_level_policy_node")
  {
    declare_parameters();
    load_parameters();
    policy_ = std::make_unique<TensorRtPolicy>(engine_path_);
    RCLCPP_INFO(get_logger(), "TensorRT warm-up finished: %s", engine_path_.c_str());

    initialize_visualization();
    if (!fake_observations_mode_) {
      initialize_ros_io();
    } else {
      RCLCPP_WARN(get_logger(),
        "FAKE OBSERVATION MODE ENABLED: robot subscriptions are not initialized and commands are disabled.");
    }
    if (!commands_enabled_) {
      RCLCPP_WARN(get_logger(), "Robot command output is disabled.");
    }
    if (fake_cube_observation_mode_ && !fake_observations_mode_) {
      RCLCPP_WARN(get_logger(),
        "Fake cube observation enabled: robot observations remain real, but cube state comes from parameters.");
    }
    if (get_parameter("plot_on_exit").as_bool()) {
      RCLCPP_WARN(get_logger(),
        "plot_on_exit is retained for compatibility but plotting is not supported by the C++ node.");
    }

    parameter_callback_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        return on_parameters(parameters);
      });

    if (get_parameter("start_policy_on_startup").as_bool()) {
      worker_ = std::thread([this]() {supervisor_loop();});
    }
  }

  ~HighLevelPolicyNode() override
  {
    stop_.store(true);
    low_state_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (uses_sport_mode()) {
      publish_sport_stop();
    }
  }

private:
  void declare_parameters()
  {
    declare_parameter("start_policy_on_startup", true);
    declare_parameter("control_mode", "hierarchical_lowcmd");
    declare_parameter("send_commands", false);
    declare_parameter("use_high_level_policy", true);
    declare_parameter("high_level_toggle_button", "X");
    declare_parameter("goal_set_button", "Y");
    declare_parameter("cube_recovery_toggle_button", "B");
    declare_parameter("joystick_command_scale", std::vector<double>{1.0, 1.0, 1.0});
    declare_parameter("goal_xy", std::vector<double>{0.0, 0.0});
    declare_parameter("goal_radius", 0.2);
    declare_parameter("cube_goal_stop_radius", 0.3);
    declare_parameter("cube_goal_hold_s", 0.6);
    declare_parameter("robot_goal_clear_radius", 0.35);
    declare_parameter("cube_state_timeout_s", 0.5);
    declare_parameter("cube_target_age_s", 0.065);
    declare_parameter("cube_stale_stop_ramp_s", 1.0);
    declare_parameter("cube_recovery_angular_cmd", 0.5);
    declare_parameter("cube_recovery_front_angle_deg", 20.0);
    declare_parameter("cube_recovery_max_rotation_deg", 360.0);
    declare_parameter("fake_observations_mode", true);
    declare_parameter("fake_observation_seed", int64_t{0});
    declare_parameter("fake_observation_min", -1.0);
    declare_parameter("fake_observation_max", 1.0);
    declare_parameter("fake_log_every_n_steps", int64_t{100});
    declare_parameter("fake_cube_observation_mode", false);
    declare_parameter("fake_cube_position_xy", std::vector<double>{0.8, 0.0});
    declare_parameter("fake_cube_velocity_xy", std::vector<double>{0.0, 0.0});
    declare_parameter("fake_cube_publish_period_s", 0.05);
    declare_parameter("kalman_odom_topic", "/go2_odometry/filtered");
    declare_parameter("cube_state_topic", "/go2_fetch/cube_state");
    declare_parameter("lowstate_topic", "/lowstate");
    declare_parameter("sport_request_topic", "/api/sport/request");
    declare_parameter("policy_world_frame", "odom");
    declare_parameter("lf_foot_frame", "FL_foot");
    declare_parameter("lf_foot_tf_timeout_s", 0.02);
    declare_parameter("cube_state_tf_timeout_s", 0.05);
    declare_parameter("robot_twist_in_body_frame", true);
    declare_parameter("cube_marker_topic", "/go2_fetch/cube_marker");
    declare_parameter("cube_dimensions", std::vector<double>{0.16, 0.16, 0.16});
    declare_parameter("goal_marker_topic", "/go2_fetch/goal_marker");
    declare_parameter("goal_marker_publish_period_s", 0.2);
    declare_parameter("command_velocity_marker_topic", "/go2_fetch/command_velocity_marker");
    declare_parameter("current_velocity_marker_topic", "/go2_fetch/current_velocity_marker");
    declare_parameter("command_velocity_marker_frame", "base");
    declare_parameter("command_velocity_marker_z_offset", 0.25);
    declare_parameter("command_velocity_marker_scale", 0.25);
    declare_parameter("velocity_marker_rate_hz", 15.0);
    declare_parameter("high_level_policy_path",
      "logs/rsl_rl/unitree_go2_pushcube_4l/2026-05-15_02-52-05_cam_6/exported/policy.engine");
    declare_parameter("high_level_rate_hz", 15.384615);
    declare_parameter("high_level_num_obs", int64_t{52});
    declare_parameter("sport_move_publish_rate_hz", 15.0);
    declare_parameter("sport_stop_on_disable", true);
    declare_parameter("sport_command_log_every_n_steps", int64_t{50});
    declare_parameter("sport_command_scale", std::vector<double>{-1.0, 1.0, 1.0});
    declare_parameter("num_actions", int64_t{12});
    declare_parameter("max_cmd", std::vector<double>{0.6, 0.4, 0.8});
    declare_parameter("leg_joint2motor_idx",
      std::vector<int64_t>{3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8});
    declare_parameter("default_angles",
      std::vector<double>{0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5});
    declare_parameter("foot_force_offset", std::vector<double>{4.0, 0.0, 5.0, 5.0});
    declare_parameter("foot_force_clip_max", 150.0);
    declare_parameter("foot_force_scale", 100.0);
    declare_parameter("high_level_command_log_period_s", 0.0);
    declare_parameter("high_level_timing_warn_threshold_ms", 65.0);
    declare_parameter("plot_on_exit", false);
    declare_parameter("analysis_pdf_path", "analyse_robot.png");
    declare_parameter("observation_csv_path", "");
  }

  template<size_t N>
  std::array<float, N> double_array_parameter(const char * name) const
  {
    const auto values = get_parameter(name).as_double_array();
    if (values.size() != N) {
      throw std::runtime_error(std::string(name) + " must contain " + std::to_string(N) + " values");
    }
    std::array<float, N> result{};
    std::transform(values.begin(), values.end(), result.begin(),
      [](double value) {return static_cast<float>(value);});
    return result;
  }

  void load_parameters()
  {
    control_mode_ = get_parameter("control_mode").as_string();
    if (control_mode_ != "hierarchical_lowcmd" && control_mode_ != "unitree_sport_high_level") {
      throw std::runtime_error(
              "control_mode must be hierarchical_lowcmd or unitree_sport_high_level");
    }
    fake_observations_mode_ = get_parameter("fake_observations_mode").as_bool();
    fake_cube_observation_mode_.store(get_parameter("fake_cube_observation_mode").as_bool());
    commands_enabled_ = get_parameter("send_commands").as_bool() && !fake_observations_mode_;
    const auto engine_path = expand_user_path(get_parameter("high_level_policy_path").as_string());
    engine_path_ = engine_path.string();
    if (engine_path.extension() != ".engine" || !std::filesystem::is_regular_file(engine_path))
    {
      throw std::runtime_error("high_level_policy_path must point to an existing .engine file: " + engine_path_);
    }
    if (get_parameter("high_level_num_obs").as_int() != static_cast<int64_t>(kObservationSize)) {
      throw std::runtime_error("high_level_num_obs must be 52");
    }
    if (get_parameter("num_actions").as_int() != static_cast<int64_t>(kJoints)) {
      throw std::runtime_error("num_actions must be 12");
    }
    max_command_ = double_array_parameter<3>("max_cmd");
    default_angles_ = double_array_parameter<12>("default_angles");
    foot_force_offset_ = double_array_parameter<4>("foot_force_offset");
    goal_ = double_array_parameter<2>("goal_xy");
    goal_radius_ = static_cast<float>(get_parameter("goal_radius").as_double());
    const auto map = get_parameter("leg_joint2motor_idx").as_integer_array();
    if (map.size() != kJoints) {
      throw std::runtime_error("leg_joint2motor_idx must contain 12 values");
    }
    for (size_t index = 0; index < kJoints; ++index) {
      if (map[index] < 0 || map[index] >= 20) {
        throw std::runtime_error("leg_joint2motor_idx contains an invalid motor index");
      }
      joint_map_[index] = static_cast<size_t>(map[index]);
    }
    fake_rng_.seed(static_cast<uint64_t>(get_parameter("fake_observation_seed").as_int()));
  }

  bool uses_sport_mode() const
  {
    return control_mode_ == "unitree_sport_high_level";
  }

  void initialize_visualization()
  {
    fake_cube_state_publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("cube_state_topic").as_string(), 10);
    cube_marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      get_parameter("cube_marker_topic").as_string(), 10);
    goal_marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      get_parameter("goal_marker_topic").as_string(), 10);
    command_marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      get_parameter("command_velocity_marker_topic").as_string(), 10);
    current_marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      get_parameter("current_velocity_marker_topic").as_string(), 10);

    const auto fake_period = std::chrono::duration<double>(
      std::max(get_parameter("fake_cube_publish_period_s").as_double(), 1.0e-3));
    fake_cube_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(fake_period),
      [this]() {publish_fake_cube_state();});
    const auto goal_period = std::chrono::duration<double>(
      std::max(get_parameter("goal_marker_publish_period_s").as_double(), 1.0e-3));
    goal_marker_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(goal_period),
      [this]() {publish_goal_marker();});
  }

  void initialize_ros_io()
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);

    odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("kalman_odom_topic").as_string(), 10,
      [this](nav_msgs::msg::Odometry::SharedPtr message) {odometry_callback(*message);});
    cube_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("cube_state_topic").as_string(), 10,
      [this](nav_msgs::msg::Odometry::SharedPtr message) {cube_callback(*message);});
    low_state_subscription_ = create_subscription<unitree_go::msg::LowState>(
      get_parameter("lowstate_topic").as_string(), 10,
      [this](unitree_go::msg::LowState::SharedPtr message) {
        {
          std::lock_guard<std::mutex> lock(data_mutex_);
          robot_.low_state = *message;
          robot_.remote.set(message->wireless_remote);
          robot_.have_low_state = message->tick != 0;
        }
        low_state_cv_.notify_one();
      });

    high_level_command_publisher_ = create_publisher<geometry_msgs::msg::TwistStamped>(
      "/go2_fetch/high_level_cmd", 10);
    const auto state_qos = rclcpp::QoS(1).reliable().transient_local();
    command_enabled_publisher_ = create_publisher<std_msgs::msg::Bool>(
      "/go2_fetch/high_level_cmd_enabled", state_qos);
    control_state_subscription_ = create_subscription<fetch_interfaces::msg::ControlState>(
      "/go2_fetch/control_state", state_qos,
      [this](fetch_interfaces::msg::ControlState::SharedPtr message) {
        control_state_.store(message->state);
      });
    if (uses_sport_mode()) {
      sport_publisher_ = create_publisher<unitree_api::msg::Request>(
        get_parameter("sport_request_topic").as_string(), 10);
    }
  }

  rcl_interfaces::msg::SetParametersResult on_parameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    std::lock_guard<std::recursive_mutex> lock(supervisor_mutex_);
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    try {
      for (const auto & parameter : parameters) {
        if (parameter.get_name() == "fake_cube_observation_mode") {
          if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
            throw std::runtime_error("fake_cube_observation_mode must be a Boolean");
          }
          fake_cube_observation_mode_.store(parameter.as_bool());
          if (parameter.as_bool()) {
            publish_fake_cube_state();
          }
          RCLCPP_INFO(get_logger(), "Cube observation source changed to: %s",
            parameter.as_bool() ? "fake cube parameters" : "cube_state_topic");
        } else if (parameter.get_name() == "use_high_level_policy") {
          if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
            throw std::runtime_error("use_high_level_policy must be a Boolean");
          }
          next_high_level_time_ = -std::numeric_limits<double>::infinity();
          if (!parameter.as_bool()) {
            set_policy_enabled(false);
            RCLCPP_INFO(get_logger(), "High-level policy disabled by parameter");
          } else {
            RCLCPP_INFO(get_logger(),
              "High-level policy allowed by parameter; press the remote toggle button to enable it");
          }
        }
      }
    } catch (const std::exception & error) {
      result.successful = false;
      result.reason = error.what();
    }
    return result;
  }

  void odometry_callback(const nav_msgs::msg::Odometry & message)
  {
    const auto & orientation = message.pose.pose.orientation;
    const double yaw = std::atan2(
      2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
      1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z));
    std::array<float, 2> velocity{
      static_cast<float>(message.twist.twist.linear.x),
      static_cast<float>(message.twist.twist.linear.y)};
    if (get_parameter("robot_twist_in_body_frame").as_bool()) {
      const auto x = velocity[0];
      const auto y = velocity[1];
      velocity[0] = static_cast<float>(std::cos(yaw) * x - std::sin(yaw) * y);
      velocity[1] = static_cast<float>(std::sin(yaw) * x + std::cos(yaw) * y);
    }
    std::lock_guard<std::mutex> lock(data_mutex_);
    robot_.robot_position = {
      static_cast<float>(message.pose.pose.position.x),
      static_cast<float>(message.pose.pose.position.y)};
    robot_.robot_velocity_world = velocity;
    robot_.base_velocity = {
      static_cast<float>(message.twist.twist.linear.x),
      static_cast<float>(message.twist.twist.linear.y),
      static_cast<float>(message.twist.twist.linear.z),
      static_cast<float>(message.twist.twist.angular.z)};
    robot_.quaternion = {
      static_cast<float>(orientation.w), static_cast<float>(orientation.x),
      static_cast<float>(orientation.y), static_cast<float>(orientation.z)};
    robot_.yaw = yaw;
    robot_.last_robot_odom = steady_seconds();
  }

  void cube_callback(const nav_msgs::msg::Odometry & message)
  {
    if (fake_cube_observation_mode_.load()) {
      return;
    }
    std::array<float, 2> position{
      static_cast<float>(message.pose.pose.position.x),
      static_cast<float>(message.pose.pose.position.y)};
    std::array<float, 2> velocity{
      static_cast<float>(message.twist.twist.linear.x),
      static_cast<float>(message.twist.twist.linear.y)};
    const auto target_frame = get_parameter("policy_world_frame").as_string();
    const auto source_frame = message.header.frame_id;
    if (!source_frame.empty() && source_frame != target_frame) {
      try {
        const auto transform = tf_buffer_->lookupTransform(
          target_frame, source_frame, rclcpp::Time(message.header.stamp));
          // rclcpp::Duration::from_seconds(get_parameter("cube_state_tf_timeout_s").as_double()));
        const auto & rotation = transform.transform.rotation;
        const double yaw = std::atan2(
          2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
          1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z));
        const auto px = position[0];
        const auto py = position[1];
        const auto vx = velocity[0];
        const auto vy = velocity[1];
        position = {
          static_cast<float>(std::cos(yaw) * px - std::sin(yaw) * py +
          transform.transform.translation.x),
          static_cast<float>(std::sin(yaw) * px + std::cos(yaw) * py +
          transform.transform.translation.y)};
        velocity = {
          static_cast<float>(std::cos(yaw) * vx - std::sin(yaw) * vy),
          static_cast<float>(std::sin(yaw) * vx + std::cos(yaw) * vy)};
        cube_tf_warning_.store(false);
      } catch (const std::exception & error) {
        if (!cube_tf_warning_.exchange(true)) {
          RCLCPP_WARN(get_logger(), "Cube state TF %s <- %s unavailable; discarding state: %s",
            target_frame.c_str(), source_frame.c_str(), error.what());
        }
        return;
      }
    } else {
      cube_tf_warning_.store(false);
    }

    double stamp = rclcpp::Time(message.header.stamp).seconds();
    if (stamp <= 0.0) {
      stamp = now().seconds();
    }
    std::lock_guard<std::mutex> lock(data_mutex_);
    cube_position_ = position;
    cube_velocity_ = velocity;
    if (cube_history_.empty() || stamp >= cube_history_.back().stamp) {
      cube_history_.push_back({stamp, position, velocity});
      if (cube_history_.size() > 256) {
        cube_history_.pop_front();
      }
    }
    last_cube_state_time_ = steady_seconds();
    cube_stale_logged_.store(false);
  }

  void apply_fake_cube_observation()
  {
    const auto position = double_array_parameter<2>("fake_cube_position_xy");
    const auto velocity = double_array_parameter<2>("fake_cube_velocity_xy");
    std::lock_guard<std::mutex> lock(data_mutex_);
    cube_position_ = position;
    cube_velocity_ = velocity;
    last_cube_state_time_ = steady_seconds();
  }

  void publish_fake_cube_state()
  {
    if (!fake_cube_observation_mode_.load()) {
      return;
    }
    apply_fake_cube_observation();
    std::array<float, 2> position;
    std::array<float, 2> velocity;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      position = cube_position_;
      velocity = cube_velocity_;
    }
    nav_msgs::msg::Odometry message;
    message.header.stamp = now();
    message.header.frame_id = get_parameter("policy_world_frame").as_string();
    message.child_frame_id = "cube";
    message.pose.pose.position.x = position[0];
    message.pose.pose.position.y = position[1];
    message.pose.covariance[0] = 0.01;
    message.pose.covariance[7] = 0.01;
    message.twist.twist.linear.x = velocity[0];
    message.twist.twist.linear.y = velocity[1];
    message.twist.covariance[0] = 0.04;
    message.twist.covariance[7] = 0.04;
    fake_cube_state_publisher_->publish(message);
    publish_cube_marker(message.header.stamp, message.header.frame_id, position);
  }

  void publish_cube_marker(
    const builtin_interfaces::msg::Time & stamp, const std::string & frame,
    const std::array<float, 2> & position)
  {
    const auto dimensions = double_array_parameter<3>("cube_dimensions");
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame;
    marker.ns = "cube_state";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = position[0];
    marker.pose.position.y = position[1];
    marker.pose.position.z = dimensions[2] / 2.0F;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = dimensions[0];
    marker.scale.y = dimensions[1];
    marker.scale.z = dimensions[2];
    marker.color.r = 1.0F;
    marker.color.g = 1.0F;
    marker.color.a = 0.6F;
    cube_marker_publisher_->publish(marker);
  }

  void publish_goal_marker()
  {
    std::lock_guard<std::recursive_mutex> lock(supervisor_mutex_);
    if (!goal_reached_) {
      const auto robot = snapshot_robot();
      if (cube_state_current()) {
        goal_reached_ = goal_condition(robot).should_stop;
      } else {
        cube_goal_enter_time_ = -std::numeric_limits<double>::infinity();
      }
    }
    const auto goal = goal_;
    const float radius = std::abs(static_cast<float>(
      get_parameter("cube_goal_stop_radius").as_double()));
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = get_parameter("policy_world_frame").as_string();
    marker.ns = "goal_region";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = goal[0];
    marker.pose.position.y = goal[1];
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 2.0F * radius;
    marker.scale.y = 2.0F * radius;
    marker.scale.z = 0.01;
    marker.color.r = goal_reached_ ? 0.2F : 1.0F;
    marker.color.g = goal_reached_ ? 1.0F : 0.0F;
    marker.color.b = 0.2F;
    marker.color.a = 0.7F;
    goal_marker_publisher_->publish(marker);
  }

  void supervisor_loop()
  {
    try {
      if (fake_observations_mode_) {
        fake_observation_loop();
      } else {
        wait_for_low_state();
        if (!stop_.load()) {
          control_loop();
        }
      }
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "High-level supervisor stopped: %s", error.what());
    }

    {
      std::lock_guard<std::recursive_mutex> lock(supervisor_mutex_);
      if (!fake_observations_mode_ && control_state_.load() ==
        fetch_interfaces::msg::ControlState::RUNNING)
      {
        command_ = {};
        publish_high_level_command(false);
      }
      if (uses_sport_mode()) {
        publish_sport_stop();
      }
    }
    if (!stop_.load() && rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  void wait_for_low_state()
  {
    std::unique_lock<std::mutex> lock(data_mutex_);
    low_state_cv_.wait(lock, [this]() {return stop_.load() || robot_.have_low_state;});
    if (!stop_.load()) {
      RCLCPP_INFO(get_logger(), "Connected to robot");
    }
  }

  void control_loop()
  {
    const double rate = uses_sport_mode() ?
      get_parameter("sport_move_publish_rate_hz").as_double() :
      get_parameter("high_level_rate_hz").as_double();
    if (rate <= 0.0) {
      throw std::runtime_error("policy loop rate must be greater than zero");
    }
    const auto period = std::chrono::duration_cast<SteadyClock::duration>(
      std::chrono::duration<double>(1.0 / rate));
    auto deadline = SteadyClock::now();
    while (!stop_.load() && rclcpp::ok()) {
      deadline += period;
      const auto start = SteadyClock::now();
      if (uses_sport_mode()) {
        sport_step();
        if (sport_exit_requested_) {
          break;
        }
      } else if (control_state_.load() == fetch_interfaces::msg::ControlState::RUNNING) {
        hierarchical_step();
      }
      const auto finished = SteadyClock::now();
      if (finished > deadline) {
        RCLCPP_WARN(get_logger(), "High-level deadline missed by %.2f ms",
          milliseconds(finished - deadline));
        deadline = finished;
      } else {
        std::this_thread::sleep_until(deadline);
      }
      warn_if_slow(milliseconds(finished - start));
    }
  }

  void fake_observation_loop()
  {
    const double rate = get_parameter("high_level_rate_hz").as_double();
    if (rate <= 0.0) {
      throw std::runtime_error("high_level_rate_hz must be greater than zero");
    }
    auto minimum = get_parameter("fake_observation_min").as_double();
    auto maximum = get_parameter("fake_observation_max").as_double();
    if (minimum > maximum) {
      std::swap(minimum, maximum);
    }
    std::uniform_real_distribution<float> distribution(
      static_cast<float>(minimum), static_cast<float>(maximum));
    const auto period = std::chrono::duration_cast<SteadyClock::duration>(
      std::chrono::duration<double>(1.0 / rate));
    auto deadline = SteadyClock::now();
    while (!stop_.load() && rclcpp::ok()) {
      deadline += period;
      {
        std::lock_guard<std::recursive_mutex> lock(supervisor_mutex_);
        ++counter_;
        for (auto & value : observation_) {
          value = distribution(fake_rng_);
        }
        if (policy_enabled_) {
          last_trt_timing_ = policy_->infer(observation_, action_);
        } else {
          for (auto & value : action_) {
            value = distribution(fake_rng_);
          }
        }
        clamp_action();
        publish_velocity_markers_if_due({});
        const auto log_every = std::max<int64_t>(
          1, get_parameter("fake_log_every_n_steps").as_int());
        if (counter_ % static_cast<uint64_t>(log_every) == 0U) {
          RCLCPP_INFO(get_logger(),
            "Fake high-level policy step %lu: high_obs=52 high_action=[%.3f %.3f %.3f] commands_sent=false",
            counter_, action_[0], action_[1], action_[2]);
        }
      }
      std::this_thread::sleep_until(deadline);
    }
  }

  RobotSnapshot snapshot_robot() const
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return robot_;
  }

  void hierarchical_step()
  {
    std::lock_guard<std::recursive_mutex> lock(supervisor_mutex_);
    ++counter_;
    const auto robot = snapshot_robot();
    update_remote_controls(robot);
    const auto inputs = policy_inputs(robot);
    bool command_enabled = false;
    if (policy_enabled_) {
      if (!cube_state_fresh()) {
        stop_for_stale_cube();
      } else if (!stop_if_goal_reached(robot)) {
        cube_tracking_lost_ = false;
        update_policy_command(robot, inputs);
        command_enabled = true;
      }
    }
    update_recovery_toggle(robot);
    if (cube_recovery_active_) {
      update_recovery_command(robot);
      command_enabled = cube_recovery_active_;
    } else if (!policy_enabled_) {
      const double ramp = std::max(get_parameter("cube_stale_stop_ramp_s").as_double(), 0.0);
      const double elapsed = steady_seconds() - stale_ramp_start_time_;
      if (std::isfinite(stale_ramp_start_time_) && ramp > 0.0 && elapsed < ramp) {
        const float remaining = static_cast<float>(1.0 - elapsed / ramp);
        for (size_t index = 0; index < kActionSize; ++index) {
          command_[index] = stale_ramp_start_command_[index] * remaining;
        }
      } else {
        stale_ramp_start_time_ = -std::numeric_limits<double>::infinity();
        command_ = joystick_command(robot.remote);
      }
      command_enabled = true;
    }
    publish_high_level_command(command_enabled);
    publish_velocity_markers_if_due(robot);
  }

  void sport_step()
  {
    std::lock_guard<std::recursive_mutex> lock(supervisor_mutex_);
    ++counter_;
    const auto robot = snapshot_robot();
    update_sport_select(robot.remote);
    update_remote_controls(robot);
    const auto inputs = policy_inputs(robot);
    if (policy_enabled_) {
      if (!cube_state_fresh()) {
        stop_for_stale_cube();
      } else if (!stop_if_goal_reached(robot)) {
        cube_tracking_lost_ = false;
        update_policy_command(robot, inputs);
        publish_sport_move();
      }
    }
    update_recovery_toggle(robot);
    if (cube_recovery_active_) {
      update_recovery_command(robot);
      if (cube_recovery_active_) {
        publish_sport_move();
      } else {
        publish_sport_stop();
      }
    } else if (!policy_enabled_) {
      command_ = {};
      publish_sport_stop();
    }
    publish_velocity_markers_if_due(robot);
  }

  struct PolicyInputs
  {
    std::array<float, 3> angular_velocity{};
    std::array<float, 3> gravity{};
    std::array<float, kJoints> joint_position{};
    std::array<float, kJoints> joint_velocity{};
  };

  PolicyInputs policy_inputs(const RobotSnapshot & robot) const
  {
    PolicyInputs inputs;
    for (size_t index = 0; index < 3; ++index) {
      inputs.angular_velocity[index] = robot.low_state.imu_state.gyroscope[index];
    }
    const auto & q = robot.low_state.imu_state.quaternion;
    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];
    inputs.gravity = {
      2.0F * (-qz * qx + qw * qy),
      -2.0F * (qz * qy + qw * qx),
      -1.0F + 2.0F * (qx * qx + qy * qy)};
    for (size_t index = 0; index < kJoints; ++index) {
      const auto motor_index = joint_map_[index];
      inputs.joint_position[index] =
        robot.low_state.motor_state[motor_index].q - default_angles_[index];
      inputs.joint_velocity[index] = robot.low_state.motor_state[motor_index].dq;
    }
    return inputs;
  }

  void update_policy_command(const RobotSnapshot & robot, const PolicyInputs & inputs)
  {
    const double current = steady_seconds();
    const double period = 1.0 / std::max(get_parameter("high_level_rate_hz").as_double(), 1.0e-6);
    if (current >= next_high_level_time_) {
      observation_ = build_observation(robot, inputs);
      last_trt_timing_ = policy_->infer(observation_, action_);
      if (!std::isfinite(next_high_level_time_)) {
        next_high_level_time_ = current + period;
      } else {
        while (next_high_level_time_ <= current) {
          next_high_level_time_ += period;
        }
      }
    }
    clamp_action();
    previous_command_ = command_;
    log_command(current, robot);
  }

  void clamp_action()
  {
    for (size_t index = 0; index < kActionSize; ++index) {
      command_[index] = std::clamp(action_[index], -max_command_[index], max_command_[index]);
    }
  }

  std::array<float, kObservationSize> build_observation(
    const RobotSnapshot & robot, const PolicyInputs & inputs)
  {
    std::array<float, 2> cube_position;
    std::array<float, 2> cube_velocity;
    if (fake_cube_observation_mode_.load()) {
      apply_fake_cube_observation();
      std::lock_guard<std::mutex> lock(data_mutex_);
      cube_position = cube_position_;
      cube_velocity = cube_velocity_;
    } else {
      const double target = now().seconds() - get_parameter("cube_target_age_s").as_double();
      std::lock_guard<std::mutex> lock(data_mutex_);
      const auto selected = std::find_if(cube_history_.rbegin(), cube_history_.rend(),
        [target](const TimedCubeState & state) {return state.stamp <= target;});
      if (selected == cube_history_.rend()) {
        throw std::runtime_error("No timestamped cube state is available");
      }
      cube_position = selected->position;
      cube_velocity = selected->velocity;
    }
    const auto foot_position = lookup_left_front_foot();
    const auto foot_force = corrected_foot_force(robot.low_state);
    const auto rotation = world_to_base_rotation(robot.quaternion);
    const auto rotate = [&rotation](const std::array<float, 2> & vector) {
        return std::array<float, 2>{
          static_cast<float>(rotation[0] * vector[0] + rotation[1] * vector[1]),
          static_cast<float>(rotation[2] * vector[0] + rotation[3] * vector[1])};
      };
    const std::array<float, 2> cube_to_goal_world{
      goal_[0] - cube_position[0], goal_[1] - cube_position[1]};
    const std::array<float, 2> foot_to_cube_world{
      cube_position[0] - foot_position[0], cube_position[1] - foot_position[1]};
    const auto cube_velocity_base = rotate(cube_velocity);
    const auto cube_to_goal_base = rotate(cube_to_goal_world);
    const auto foot_to_cube_base = rotate(foot_to_cube_world);

    std::array<float, kObservationSize> observation{};
    size_t offset = 0;
    const auto append = [&observation, &offset](float value) {
        observation.at(offset++) = value;
      };
    for (float value : inputs.angular_velocity) {append(clipped(value) * 0.2F);}
    for (float value : inputs.gravity) {append(clipped(value));}
    for (float value : inputs.joint_position) {append(clipped(value));}
    for (float value : inputs.joint_velocity) {append(clipped(value) * 0.05F);}
    for (float value : previous_command_) {append(clipped(value));}
    append(clipped(robot.robot_position[0] - goal_[0]) * 0.5F);
    append(clipped(robot.robot_position[1] - goal_[1]) * 0.5F);
    for (float value : robot.robot_velocity_world) {append(clipped(value));}
    append(clipped(cube_position[0] - goal_[0]) * 0.5F);
    append(clipped(cube_position[1] - goal_[1]) * 0.5F);
    for (float value : cube_velocity_base) {append(clipped(value));}
    append(0.0F);
    append(0.0F);
    append(clipped(goal_radius_));
    for (float value : cube_to_goal_base) {append(clipped(value) * 0.5F);}
    for (float value : foot_to_cube_base) {append(clipped(value));}
    const float force_scale = static_cast<float>(get_parameter("foot_force_scale").as_double());
    if (force_scale <= 0.0F) {
      throw std::runtime_error("foot_force_scale must be greater than zero");
    }
    for (float value : foot_force) {append(clipped(value, 0.0F, 150.0F) / force_scale);}
    if (offset != kObservationSize ||
      !std::all_of(observation.begin(), observation.end(), [](float value) {return std::isfinite(value);}))
    {
      throw std::runtime_error("PushCube observation is invalid");
    }
    append_observation_csv(observation);
    return observation;
  }

  std::array<double, 4> world_to_base_rotation(const std::array<float, 4> & quaternion) const
  {
    double qw = quaternion[0];
    double qx = quaternion[1];
    double qy = quaternion[2];
    double qz = quaternion[3];
    const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (!std::isfinite(norm) || norm < 1.0e-8) {
      throw std::runtime_error("robot quaternion is invalid");
    }
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;
    return {
      1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qz * qw),
      2.0 * (qx * qy - qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz)};
  }

  std::array<float, 2> lookup_left_front_foot() const
  {
    try {
      const auto transform = tf_buffer_->lookupTransform(
        get_parameter("policy_world_frame").as_string(),
        get_parameter("lf_foot_frame").as_string(), tf2::TimePointZero);
        // tf2::durationFromSec(get_parameter("lf_foot_tf_timeout_s").as_double()));
      return {static_cast<float>(transform.transform.translation.x),
        static_cast<float>(transform.transform.translation.y)};
    } catch (const std::exception & error) {
      throw std::runtime_error(std::string("Required LF foot TF is unavailable: ") + error.what());
    }
  }

  std::array<float, 4> corrected_foot_force(const unitree_go::msg::LowState & state) const
  {
    const float maximum = static_cast<float>(get_parameter("foot_force_clip_max").as_double());
    if (maximum <= 0.0F) {
      throw std::runtime_error("foot_force_clip_max must be greater than zero");
    }
    const std::array<float, 4> corrected{
      state.foot_force[1] - foot_force_offset_[1],
      state.foot_force[0] - foot_force_offset_[0],
      state.foot_force[3] - foot_force_offset_[3],
      state.foot_force[2] - foot_force_offset_[2]};
    std::array<float, 4> result{};
    std::transform(corrected.begin(), corrected.end(), result.begin(),
      [maximum](float value) {return std::clamp(value, 0.0F, maximum);});
    return result;
  }

  void append_observation_csv(const std::array<float, kObservationSize> & observation)
  {
    const auto path_value = get_parameter("observation_csv_path").as_string();
    if (path_value.empty()) {
      return;
    }
    const auto path = expand_user_path(path_value);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    const bool header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream output(path, std::ios::app);
    if (!output) {
      throw std::runtime_error("Cannot append observation CSV: " + path.string());
    }
    if (header) {
      output << "ros_time_s";
      for (size_t index = 0; index < kObservationSize; ++index) {
        output << ",obs_" << index;
      }
      output << '\n';
    }
    output << std::setprecision(10) << now().seconds();
    for (float value : observation) {
      output << ',' << value;
    }
    output << '\n';
  }

  bool cube_state_fresh()
  {
    if (fake_observations_mode_ || fake_cube_observation_mode_.load()) {
      return true;
    }
    const double timeout = get_parameter("cube_state_timeout_s").as_double();
    if (timeout <= 0.0) {
      return true;
    }
    double last;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      last = last_cube_state_time_;
    }
    const double age = steady_seconds() - last;
    if (age <= timeout) {
      return true;
    }
    if (!cube_stale_logged_.exchange(true)) {
      RCLCPP_WARN(get_logger(),
        "No fresh cube state; disabling high-level commands until tracking resumes (age=%.3fs timeout=%.3fs)",
        age, timeout);
    }
    return false;
  }

  bool cube_state_current() const
  {
    if (fake_observations_mode_ || fake_cube_observation_mode_.load()) {
      return true;
    }
    const double timeout = get_parameter("cube_state_timeout_s").as_double();
    std::lock_guard<std::mutex> lock(data_mutex_);
    return timeout <= 0.0 || steady_seconds() - last_cube_state_time_ <= timeout;
  }

  void stop_for_stale_cube()
  {
    cube_goal_enter_time_ = -std::numeric_limits<double>::infinity();
    cube_tracking_lost_ = true;
    const auto ramp_start = command_;
    set_policy_enabled(false);
    const double ramp = std::max(get_parameter("cube_stale_stop_ramp_s").as_double(), 0.0);
    if (!uses_sport_mode() && ramp > 0.0) {
      stale_ramp_start_command_ = ramp_start;
      stale_ramp_start_time_ = steady_seconds();
      command_ = ramp_start;
    }
    publish_command_marker();
    if (uses_sport_mode()) {
      publish_sport_stop();
    }
  }

  struct GoalCondition
  {
    bool should_stop{};
    double cube_distance{};
    double robot_distance{};
  };

  GoalCondition goal_condition(const RobotSnapshot & robot)
  {
    std::array<float, 2> cube;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      cube = cube_position_;
    }
    const double cube_distance = std::hypot(cube[0] - goal_[0], cube[1] - goal_[1]);
    const double robot_distance = std::hypot(
      robot.robot_position[0] - goal_[0], robot.robot_position[1] - goal_[1]);
    const double cube_radius = std::abs(get_parameter("cube_goal_stop_radius").as_double());
    if (cube_distance <= cube_radius) {
      if (!std::isfinite(cube_goal_enter_time_)) {
        cube_goal_enter_time_ = steady_seconds();
      }
    } else {
      cube_goal_enter_time_ = -std::numeric_limits<double>::infinity();
    }
    const bool held = cube_distance <= cube_radius &&
      steady_seconds() - cube_goal_enter_time_ >=
      std::max(get_parameter("cube_goal_hold_s").as_double(), 0.0);
    const bool should_stop = held && robot_distance >
      std::abs(get_parameter("robot_goal_clear_radius").as_double());
    return {should_stop, cube_distance, robot_distance};
  }

  bool stop_if_goal_reached(const RobotSnapshot & robot)
  {
    const auto condition = goal_condition(robot);
    if (!condition.should_stop) {
      return false;
    }
    goal_reached_ = true;
    publish_goal_marker();
    set_policy_enabled(false);
    RCLCPP_INFO(get_logger(),
      "High-level policy stopped: cube reached goal while robot is clear (cube=%.3fm robot=%.3fm)",
      condition.cube_distance, condition.robot_distance);
    return true;
  }

  static size_t button_index(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(),
      [](unsigned char value) {return static_cast<char>(std::tolower(value));});
    static const std::unordered_map<std::string, size_t> buttons{
      {"r1", 0}, {"l1", 1}, {"start", 2}, {"select", 3}, {"back", 3},
      {"r2", 4}, {"l2", 5}, {"f1", 6}, {"f2", 7}, {"a", 8}, {"b", 9},
      {"x", 10}, {"y", 11}, {"up", 12}, {"right", 13}, {"down", 14}, {"left", 15}};
    const auto found = buttons.find(name);
    if (found == buttons.end()) {
      throw std::runtime_error("Unknown remote button: " + name);
    }
    return found->second;
  }

  void update_remote_controls(const RobotSnapshot & robot)
  {
    const auto toggle_name = get_parameter("high_level_toggle_button").as_string();
    const bool toggle = robot.remote.pressed(button_index(toggle_name));
    if (toggle && !last_toggle_pressed_) {
      if (!get_parameter("use_high_level_policy").as_bool()) {
        set_policy_enabled(false);
        RCLCPP_INFO(get_logger(), "Remote toggle pressed, but policy is disabled by parameter");
      } else {
        set_policy_enabled(!policy_enabled_);
        RCLCPP_INFO(get_logger(), "Remote toggle pressed: command source is now %s",
          policy_enabled_ ? "PushCube high-level policy" :
          (uses_sport_mode() ? "Unitree Sport StopMove" : "joystick"));
      }
    }
    last_toggle_pressed_ = toggle;

    const auto goal_name = get_parameter("goal_set_button").as_string();
    const bool goal_pressed = robot.remote.pressed(button_index(goal_name));
    if (goal_pressed && !last_goal_pressed_) {
      if (!std::isfinite(robot.last_robot_odom)) {
        RCLCPP_WARN(get_logger(), "Cannot set goal before robot odometry is available");
      } else {
        goal_reached_ = false;
        cube_goal_enter_time_ = -std::numeric_limits<double>::infinity();
        goal_ = robot.robot_position;
        (void)set_parameter(rclcpp::Parameter("goal_xy",
          std::vector<double>{goal_[0], goal_[1]}));
        publish_goal_marker();
        RCLCPP_INFO(get_logger(), "Goal region set to [%.3f, %.3f]", goal_[0], goal_[1]);
      }
    }
    last_goal_pressed_ = goal_pressed;
  }

  void update_sport_select(const RemoteController & remote)
  {
    if (button_index(get_parameter("high_level_toggle_button").as_string()) == 3) {
      return;
    }
    const bool pressed = remote.pressed(3);
    if (pressed && !last_select_pressed_) {
      set_policy_enabled(false);
      sport_exit_requested_ = true;
      RCLCPP_INFO(get_logger(),
        "Remote SELECT pressed: high-level policy disabled and Sport StopMove sent");
    }
    last_select_pressed_ = pressed;
  }

  std::array<float, 3> joystick_command(const RemoteController & remote) const
  {
    auto scale = double_array_parameter<3>("joystick_command_scale");
    std::array<float, 3> result{remote.ly, -remote.lx, -remote.rx};
    for (size_t index = 0; index < 3; ++index) {
      result[index] = std::clamp(result[index] * scale[index],
        -max_command_[index], max_command_[index]);
    }
    return result;
  }

  double cube_bearing(const RobotSnapshot & robot) const
  {
    std::array<float, 2> cube;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      cube = cube_position_;
    }
    return normalize_angle(std::atan2(
      cube[1] - robot.robot_position[1], cube[0] - robot.robot_position[0]) - robot.yaw);
  }

  void update_recovery_toggle(const RobotSnapshot & robot)
  {
    const auto name = get_parameter("cube_recovery_toggle_button").as_string();
    const bool pressed = robot.remote.pressed(button_index(name));
    if (pressed && !last_recovery_pressed_) {
      if (cube_recovery_active_) {
        cube_recovery_active_ = false;
        command_ = {};
        if (uses_sport_mode()) {publish_sport_stop();}
      } else if (!cube_tracking_lost_) {
        RCLCPP_INFO(get_logger(), "Cube recovery ignored because tracking was not lost");
      } else {
        double last;
        {
          std::lock_guard<std::mutex> lock(data_mutex_);
          last = last_cube_state_time_;
        }
        const double bearing = cube_bearing(robot);
        recovery_direction_ = 1.0F;
        recovery_direction_ = bearing >= 0.0 ? 1.0F : -1.0F;
        recovery_last_yaw_ = robot.yaw;
        recovery_rotation_ = 0.0;
        cube_recovery_active_ = true;
        set_policy_enabled(false);
        RCLCPP_INFO(get_logger(), "Starting cube recovery (bearing=%.1f deg)",
          bearing * 180.0 / kPi);
      }
    }
    last_recovery_pressed_ = pressed;
  }

  void update_recovery_command(const RobotSnapshot & robot)
  {
    recovery_rotation_ += std::abs(normalize_angle(robot.yaw - recovery_last_yaw_));
    recovery_last_yaw_ = robot.yaw;
    if (cube_state_fresh()) {
      const double bearing = cube_bearing(robot);
      const double front = std::abs(get_parameter("cube_recovery_front_angle_deg").as_double()) *
        kPi / 180.0;
      if (std::abs(bearing) <= front) {
        cube_recovery_active_ = false;
        cube_tracking_lost_ = false;
        command_ = {};
        if (get_parameter("use_high_level_policy").as_bool() && !stop_if_goal_reached(robot)) {
          set_policy_enabled(true);
          RCLCPP_INFO(get_logger(), "Cube reacquired; restarting high-level policy");
        }
        return;
      }
    }
    const double maximum = std::abs(
      get_parameter("cube_recovery_max_rotation_deg").as_double()) * kPi / 180.0;
    if (recovery_rotation_ >= maximum) {
      cube_recovery_active_ = false;
      command_ = {};
      if (uses_sport_mode()) {publish_sport_stop();}
      RCLCPP_WARN(get_logger(), "Cube recovery reached maximum rotation");
      return;
    }
    const float angular = std::min(
      std::abs(static_cast<float>(get_parameter("cube_recovery_angular_cmd").as_double())),
      std::abs(max_command_[2]));
    command_ = {0.0F, 0.0F, recovery_direction_ * angular};
  }

  void set_policy_enabled(bool enabled)
  {
    const bool was_enabled = policy_enabled_;
    policy_enabled_ = enabled;
    next_high_level_time_ = -std::numeric_limits<double>::infinity();
    stale_ramp_start_time_ = -std::numeric_limits<double>::infinity();
    if (enabled) {
      cube_recovery_active_ = false;
      if (!was_enabled) {previous_command_ = {};}
    } else {
      command_ = {};
      if (uses_sport_mode()) {
        publish_sport_stop();
        publish_command_marker();
      }
    }
  }

  void publish_high_level_command(bool enabled)
  {
    if (!high_level_command_publisher_ ||
      control_state_.load() != fetch_interfaces::msg::ControlState::RUNNING)
    {
      return;
    }
    std_msgs::msg::Bool state;
    state.data = enabled && commands_enabled_;
    command_enabled_publisher_->publish(state);
    geometry_msgs::msg::TwistStamped command;
    command.header.stamp = now();
    command.header.frame_id = get_parameter("command_velocity_marker_frame").as_string();
    if (state.data) {
      command.twist.linear.x = command_[0];
      command.twist.linear.y = command_[1];
      command.twist.angular.z = command_[2];
    }
    high_level_command_publisher_->publish(command);
  }

  std::array<float, 3> sport_command() const
  {
    const auto scale = double_array_parameter<3>("sport_command_scale");
    return {command_[0] * scale[0], command_[1] * scale[1], command_[2] * scale[2]};
  }

  void publish_sport_request(int64_t api_id, const std::string & parameter = {})
  {
    if (!commands_enabled_ || !sport_publisher_) {
      return;
    }
    unitree_api::msg::Request request;
    request.header.identity.api_id = api_id;
    request.parameter = parameter;
    sport_publisher_->publish(request);
  }

  void publish_sport_move()
  {
    const auto sport = sport_command();
    std::ostringstream parameter;
    parameter << std::setprecision(9) << "{\"x\": " << sport[0] <<
      ", \"y\": " << sport[1] << ", \"z\": " << sport[2] << '}';
    publish_sport_request(kSportMove, parameter.str());
    sport_stop_sent_ = false;
    const int64_t every = get_parameter("sport_command_log_every_n_steps").as_int();
    if (every > 0 && counter_ % static_cast<uint64_t>(every) == 0U) {
      RCLCPP_INFO(get_logger(), "Unitree Sport Move cmd=[%.3f %.3f %.3f]",
        command_[0], command_[1], command_[2]);
    }
  }

  void publish_sport_stop()
  {
    if (!get_parameter("sport_stop_on_disable").as_bool() || sport_stop_sent_) {
      return;
    }
    publish_sport_request(kSportStopMove);
    sport_stop_sent_ = true;
  }

  void publish_velocity_markers_if_due(const RobotSnapshot & robot)
  {
    const double rate = get_parameter("velocity_marker_rate_hz").as_double();
    const double current = steady_seconds();
    if (rate <= 0.0 || current < next_marker_time_) {
      return;
    }
    next_marker_time_ = current + 1.0 / rate;
    publish_command_marker();
    publish_velocity_marker(current_marker_publisher_, "current_velocity",
      robot.base_velocity[0], robot.base_velocity[1], 1.0F, 0.45F, 0.0F);
    publish_angular_velocity_marker(current_marker_publisher_, "current_velocity",
      robot.base_velocity[0], robot.base_velocity[1], robot.base_velocity[3],
      1.0F, 0.45F, 0.0F);
  }

  void publish_command_marker()
  {
    publish_velocity_marker(command_marker_publisher_, "command_velocity",
      command_[0], command_[1], 0.0F, 0.7F, 1.0F);
    publish_angular_velocity_marker(command_marker_publisher_, "command_velocity",
      command_[0], command_[1], command_[2], 0.0F, 0.7F, 1.0F);
  }

  void publish_velocity_marker(
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr & publisher,
    const std::string & name, float x, float y, float red, float green, float blue)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = get_parameter("command_velocity_marker_frame").as_string();
    marker.ns = name;
    marker.id = 0;
    if (std::hypot(x, y) < 1.0e-3F) {
      marker.action = visualization_msgs::msg::Marker::DELETE;
      publisher->publish(marker);
      return;
    }
    const double z = get_parameter("command_velocity_marker_z_offset").as_double();
    const double scale = get_parameter("command_velocity_marker_scale").as_double();
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    geometry_msgs::msg::Point origin;
    origin.z = z;
    geometry_msgs::msg::Point end;
    end.x = x * scale;
    end.y = y * scale;
    end.z = z;
    marker.points = {origin, end};
    marker.scale.x = 0.035;
    marker.scale.y = 0.09;
    marker.scale.z = 0.12;
    marker.color.r = red;
    marker.color.g = green;
    marker.color.b = blue;
    marker.color.a = 0.9F;
    marker.lifetime = rclcpp::Duration::from_seconds(0.25);
    publisher->publish(marker);
  }

  // Draw yaw-rate magnitude vertically from the tip of the matching planar
  // velocity arrow. The arrow always points upward; its length is proportional
  // to |angular.z| in the same way the planar arrow length follows linear speed.
  void publish_angular_velocity_marker(
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr & publisher,
    const std::string & name, float x, float y, float yaw_rate,
    float red, float green, float blue)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = get_parameter("command_velocity_marker_frame").as_string();
    marker.ns = name;
    marker.id = 1;
    if (std::abs(yaw_rate) < 1.0e-3F) {
      marker.action = visualization_msgs::msg::Marker::DELETE;
      publisher->publish(marker);
      return;
    }
    const double z = get_parameter("command_velocity_marker_z_offset").as_double();
    const double scale = get_parameter("command_velocity_marker_scale").as_double();
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    geometry_msgs::msg::Point origin;
    origin.x = x * scale;
    origin.y = y * scale;
    origin.z = z;
    geometry_msgs::msg::Point end = origin;
    end.z += std::abs(yaw_rate) * scale;
    marker.points = {origin, end};
    marker.scale.x = 0.035;
    marker.scale.y = 0.09;
    marker.scale.z = 0.12;
    marker.color.r = red;
    marker.color.g = green;
    marker.color.b = blue;
    marker.color.a = 0.9F;
    marker.lifetime = rclcpp::Duration::from_seconds(0.25);
    publisher->publish(marker);
  }

  void log_command(double current, const RobotSnapshot & robot)
  {
    const double period = get_parameter("high_level_command_log_period_s").as_double();
    if (period <= 0.0 || current - last_command_log_time_ < period) {
      return;
    }
    last_command_log_time_ = current;
    std::array<float, 2> cube;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      cube = cube_position_;
    }
    RCLCPP_INFO(get_logger(),
      "High-level command enabled=%d raw=[%.3f %.3f %.3f] cmd=[%.3f %.3f %.3f] robot=[%.3f %.3f] cube=[%.3f %.3f] goal=[%.3f %.3f]",
      policy_enabled_, action_[0], action_[1], action_[2], command_[0], command_[1], command_[2],
      robot.robot_position[0], robot.robot_position[1], cube[0], cube[1], goal_[0], goal_[1]);
  }

  void warn_if_slow(double elapsed_ms)
  {
    const double threshold = get_parameter("high_level_timing_warn_threshold_ms").as_double();
    if (threshold > 0.0 && elapsed_ms >= threshold) {
      RCLCPP_WARN(get_logger(),
        "High-level slow step: total=%.2fms trt=%.2fms execute=%.2fms h2d=%.2fms d2h=%.2fms",
        elapsed_ms, last_trt_timing_.total, last_trt_timing_.execute,
        last_trt_timing_.h2d, last_trt_timing_.d2h);
    }
  }

  std::string control_mode_;
  std::string engine_path_;
  bool fake_observations_mode_{};
  std::atomic<bool> fake_cube_observation_mode_{false};
  bool commands_enabled_{};
  std::atomic<bool> stop_{false};
  std::atomic<bool> cube_tf_warning_{false};
  std::atomic<uint8_t> control_state_{fetch_interfaces::msg::ControlState::ZERO_TORQUE};
  std::thread worker_;
  std::unique_ptr<TensorRtPolicy> policy_;
  TrtTiming last_trt_timing_;

  mutable std::mutex data_mutex_;
  mutable std::recursive_mutex supervisor_mutex_;
  std::condition_variable low_state_cv_;
  RobotSnapshot robot_;
  std::deque<TimedCubeState> cube_history_;
  std::array<float, 2> cube_position_{};
  std::array<float, 2> cube_velocity_{};
  double last_cube_state_time_{-std::numeric_limits<double>::infinity()};
  std::atomic<bool> cube_stale_logged_{false};

  std::array<size_t, kJoints> joint_map_{};
  std::array<float, kJoints> default_angles_{};
  std::array<float, 4> foot_force_offset_{};
  std::array<float, 3> max_command_{};
  std::array<float, 2> goal_{};
  float goal_radius_{};
  std::array<float, kObservationSize> observation_{};
  std::array<float, kActionSize> action_{};
  std::array<float, kActionSize> command_{};
  std::array<float, kActionSize> previous_command_{};
  std::array<float, kActionSize> stale_ramp_start_command_{};
  std::mt19937_64 fake_rng_;

  bool policy_enabled_{};
  bool goal_reached_{};
  bool cube_tracking_lost_{};
  bool cube_recovery_active_{};
  bool last_toggle_pressed_{};
  bool last_goal_pressed_{};
  bool last_recovery_pressed_{};
  bool last_select_pressed_{};
  bool sport_stop_sent_{};
  bool sport_exit_requested_{};
  float recovery_direction_{1.0F};
  double recovery_last_yaw_{};
  double recovery_rotation_{};
  double next_high_level_time_{-std::numeric_limits<double>::infinity()};
  double next_marker_time_{-std::numeric_limits<double>::infinity()};
  double stale_ramp_start_time_{-std::numeric_limits<double>::infinity()};
  double cube_goal_enter_time_{-std::numeric_limits<double>::infinity()};
  double last_command_log_time_{-std::numeric_limits<double>::infinity()};
  uint64_t counter_{};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr cube_subscription_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_subscription_;
  rclcpp::Subscription<fetch_interfaces::msg::ControlState>::SharedPtr control_state_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr high_level_command_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr command_enabled_publisher_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fake_cube_state_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cube_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr command_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr current_marker_publisher_;
  rclcpp::TimerBase::SharedPtr fake_cube_timer_;
  rclcpp::TimerBase::SharedPtr goal_marker_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
};

}  // namespace fetch_policy

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<fetch_policy::HighLevelPolicyNode>());
  } catch (const std::exception & error) {
    std::fprintf(stderr, "high_level_policy_node_cpp: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
