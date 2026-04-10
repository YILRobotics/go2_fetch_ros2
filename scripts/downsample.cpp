#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <memory>
#include <string>
#include <optional>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <cstring>   // memcpy

class RGBPointCloudDownsampler : public rclcpp::Node
{
public:
  RGBPointCloudDownsampler()
  : Node("rgb_pointcloud_downsampler"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    input_topic_       = this->declare_parameter<std::string>("input_topic", "/camera/camera/depth/color/points");
    output_topic_      = this->declare_parameter<std::string>("output_topic", "/points_downsampled");
    target_frame_      = this->declare_parameter<std::string>("target_frame", "camera_link");
    publish_rate_hz_   = this->declare_parameter<double>("publish_rate_hz", 25.0);
    factor_            = this->declare_parameter<int>("downsample_factor", 40);
    tf_timeout_s_      = this->declare_parameter<double>("tf_timeout_s", 0.02);
    use_timer_         = this->declare_parameter<bool>("use_timer", true);
    do_tf_             = this->declare_parameter<bool>("do_tf", false); // turn off unless you really need it

    // Publisher QoS: small queue
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::QoS(1));

    // Subscription QoS: keep_last(1), best effort
    auto qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg)
      {
        std::lock_guard<std::mutex> lock(mtx_);
        latest_ = std::move(msg);
      });

    if (use_timer_ && publish_rate_hz_ > 0.0) {
      timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / publish_rate_hz_),
        std::bind(&RGBPointCloudDownsampler::tick, this));
    }

    RCLCPP_INFO(this->get_logger(),
      "FAST Downsampler ready. input=%s output=%s factor=%d rate=%.1f do_tf=%s target_frame=%s",
      input_topic_.c_str(), output_topic_.c_str(), factor_, publish_rate_hz_,
      do_tf_ ? "true" : "false", target_frame_.c_str());
  }

private:
  void tick()
  {
    sensor_msgs::msg::PointCloud2::SharedPtr msg;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      msg = latest_;
    }
    if (!msg) return;


    if (do_tf_) {
        sensor_msgs::msg::PointCloud2 transformed;
        if (!transformIfNeeded(*msg, transformed)) return;
        auto out = downsampleStride(transformed);
        if (out) pub_->publish(*out);
    } else {
        auto out = downsampleStride(*msg);          // ✅ no full copy
        if (out) pub_->publish(*out);
    }
  }


  bool transformIfNeeded(const sensor_msgs::msg::PointCloud2 & in,
                         sensor_msgs::msg::PointCloud2 & out)
  {
    if (target_frame_.empty() || in.header.frame_id == target_frame_) {
      out = in;
      return true;
    }

    try {
      const auto tf = tf_buffer_.lookupTransform(
        target_frame_, in.header.frame_id, in.header.stamp,
        tf2::durationFromSec(std::max(0.0, tf_timeout_s_)));

      tf2::doTransform(in, out, tf);
      out.header.frame_id = target_frame_;
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "TF %s -> %s failed: %s",
        in.header.frame_id.c_str(), target_frame_.c_str(), ex.what());
      return false;
    }
  }

  std::optional<sensor_msgs::msg::PointCloud2>
  downsampleStride(const sensor_msgs::msg::PointCloud2 & in) const
  {
    const size_t total_pts = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    if (total_pts == 0) return std::nullopt;

    if (factor_ <= 1) return in;

    if (in.point_step == 0) return std::nullopt;
    if (in.data.size() < total_pts * static_cast<size_t>(in.point_step)) {
      // malformed message
      return std::nullopt;
    }

    // output points count (floor)
    const size_t out_pts = total_pts / static_cast<size_t>(factor_);
    if (out_pts == 0) return std::nullopt;

    sensor_msgs::msg::PointCloud2 out;
    out.header       = in.header;
    out.height       = 1;
    out.width        = static_cast<uint32_t>(out_pts);
    out.fields       = in.fields;        // preserve EVERYTHING
    out.is_bigendian = in.is_bigendian;
    out.is_dense     = in.is_dense;
    out.point_step   = in.point_step;
    out.row_step     = out.point_step * out.width;

    out.data.resize(static_cast<size_t>(out.row_step));

    const uint8_t* src = in.data.data();
    uint8_t* dst = out.data.data();
    const size_t ps = static_cast<size_t>(in.point_step);

    // copy every factor-th point as a raw block
    size_t written = 0;
    for (size_t i = 0; i < total_pts; i += static_cast<size_t>(factor_)) {
      std::memcpy(dst + written * ps, src + i * ps, ps);
      ++written;
      if (written >= out_pts) break;
    }

    out.width = static_cast<uint32_t>(written);
    out.row_step = out.point_step * out.width;
    out.data.resize(static_cast<size_t>(out.row_step));

    return (written > 0) ? std::optional(out) : std::nullopt;
  }

private:
  std::string input_topic_;
  std::string output_topic_;
  std::string target_frame_;

  double publish_rate_hz_{25.0};
  int factor_{40};
  double tf_timeout_s_{0.02};
  bool use_timer_{true};
  bool do_tf_{false};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mtx_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RGBPointCloudDownsampler>();

  rclcpp::executors::MultiThreadedExecutor exec(
      rclcpp::ExecutorOptions(), 2); // 2 Threads reichen
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}