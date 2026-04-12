#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <mutex>
#include <algorithm>

class RGBPointCloudMapper : public rclcpp::Node
{
public:
  RGBPointCloudMapper()
  : Node("rgb_point_cloud_mapper"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // New params for bottom crop and normals
    // Parameters
    this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    this->declare_parameter<std::string>("rgb_topic",   "/camera/color/image_raw");
    this->declare_parameter<std::string>("info_topic",  "/camera/camera_info"); // use color intrinsics if aligned
    this->declare_parameter<std::string>("output_topic","/rgb_map/cloud");
    this->declare_parameter<std::string>("target_frame","mycobot_base"); // map frame for the output

    this->declare_parameter<double>("bottom_crop_percent", 0.0); // percent of image height to cut from bottom

    this->declare_parameter<bool>("publish_single_frame", true);
    this->declare_parameter<std::string>("single_frame_topic", "/rgb_map/cloud_frame");
    this->declare_parameter<std::string>("cropped_rgb_topic", "/rgb_map/cropped_rgb");

    this->declare_parameter<int>("downsample_step", 3);     // pixel stride
    this->declare_parameter<double>("max_range_m", 5.0);    // depth cutoff
    this->declare_parameter<double>("publish_rate_hz", 5.0);// throttle
    this->declare_parameter<bool>("accumulate", true);      // build a "map" by accumulating clouds
    // Use int64_t for ROS parameters to match rclcpp Parameter types
    this->declare_parameter<int64_t>("max_points", static_cast<int64_t>(2'000'000)); // cap accumulated points (memory protection)

    depth_topic_  = this->get_parameter("depth_topic").as_string();
    rgb_topic_    = this->get_parameter("rgb_topic").as_string();
    info_topic_   = this->get_parameter("info_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    target_frame_ = this->get_parameter("target_frame").as_string();
    bottom_crop_percent_ = std::max(0.0, this->get_parameter("bottom_crop_percent").as_double());

    publish_single_frame_ = this->get_parameter("publish_single_frame").as_bool();
    single_frame_topic_ = this->get_parameter("single_frame_topic").as_string();
    cropped_rgb_topic_ = this->get_parameter("cropped_rgb_topic").as_string();

    {
      const int64_t ds_param = this->get_parameter("downsample_step").as_int();
      downsample_step_ = static_cast<int>(std::max<int64_t>(1, ds_param));
    }
    
    max_range_m_     = std::max(0.1, this->get_parameter("max_range_m").as_double());
    publish_rate_hz_ = this->get_parameter("publish_rate_hz").as_double();
    accumulate_      = this->get_parameter("accumulate").as_bool();
    // max_points was declared as int64_t above, read it as int and cast to size_t
    max_points_      = static_cast<size_t>(this->get_parameter("max_points").as_int());

    if (publish_rate_hz_ > 0.0) {
      min_pub_interval_ = 1.0 / publish_rate_hz_;
    } else {
      min_pub_interval_ = 0.0; // no throttle
    }

    last_pub_steady_ = std::chrono::steady_clock::now();

    // Publisher
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::QoS(10));
    if (publish_single_frame_) {
      single_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(single_frame_topic_, rclcpp::QoS(10));
    }
    cropped_rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>(cropped_rgb_topic_, rclcpp::QoS(5));

    // message_filters subscribers (for time sync)
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, depth_topic_);
    rgb_sub_   = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, rgb_topic_);
    info_sub_  = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(this, info_topic_);

    using Policy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;

    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(20), *depth_sub_, *rgb_sub_, *info_sub_);
    sync_->registerCallback(std::bind(&RGBPointCloudMapper::syncedCallback, this,
                                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    RCLCPP_INFO(this->get_logger(),
      "RGBPointCloudMapper started. depth=%s rgb=%s info=%s -> %s, target_frame=%s, accumulate=%s",
      depth_topic_.c_str(), rgb_topic_.c_str(), info_topic_.c_str(), output_topic_.c_str(),
      target_frame_.c_str(), accumulate_ ? "true" : "false");
  }

private:
  static inline bool encodingIsDepth16U(const std::string & enc)
  {
    // Common encodings: "16UC1", "mono16"
    return (enc == "16UC1" || enc == "mono16");
  }

  static inline bool encodingIsDepth32F(const std::string & enc)
  {
    // Common encodings: "32FC1"
    return (enc == "32FC1");
  }

  void syncedCallback(const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
                      const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
  {
    // Throttle using steady clock (avoids ROS time source mismatches)
    if (min_pub_interval_ > 0.0) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - last_pub_steady_).count();
      if (elapsed < min_pub_interval_) {
        return;
      }
      last_pub_steady_ = now;
    }

    // Intrinsics (assumes depth is registered to the RGB camera if using color camera_info)
    const double fx = info_msg->k[0];
    const double fy = info_msg->k[4];
    const double cx = info_msg->k[2];
    const double cy = info_msg->k[5];

    if (fx <= 0.0 || fy <= 0.0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Invalid intrinsics (fx/fy <= 0).");
      return;
    }

    // Convert RGB image
    cv_bridge::CvImagePtr rgb_cv;
    try {
      // Prefer bgr8; if already bgr8, this is cheap
      rgb_cv = cv_bridge::toCvCopy(rgb_msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "cv_bridge RGB error: %s", e.what());
      return;
    }

    // Convert depth image (keep native encoding)
    cv_bridge::CvImagePtr depth_cv;
    try {
      depth_cv = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "cv_bridge depth error: %s", e.what());
      return;
    }

    const cv::Mat & rgb   = rgb_cv->image;
    const cv::Mat & depth = depth_cv->image;

    if (rgb.empty() || depth.empty()) {
      return;
    }

    // if (rgb.rows != depth.rows || rgb.cols != depth.cols) {
    //   RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    //                        "RGB and depth sizes differ (rgb=%dx%d, depth=%dx%d). "
    //                        "You need aligned depth-to-color for per-pixel RGB mapping.",
    //                        rgb.cols, rgb.rows, depth.cols, depth.rows);
    //   return;
    // }

    const int height = depth.rows;
    const int width  = depth.cols;
    const int ds = downsample_step_;

    // Compute effective height after bottom crop.
    // Accept either a fraction in [0..1] or a percentage in (1..100].
    int effective_height = height;
    if (bottom_crop_percent_ > 0.0) {
      double frac = bottom_crop_percent_;
      if (bottom_crop_percent_ > 1.0) {
        // Interpret values >1 as percent (e.g. 10 -> 10%)
        frac = bottom_crop_percent_ / 100.0;
      }
      // Clamp fraction to [0,1]
      frac = std::min(std::max(frac, 0.0), 1.0);
      const int crop_rows = static_cast<int>(std::round(height * frac));
      effective_height = std::max(1, height - crop_rows);
    }

    // Ensure we don't attempt to crop more rows than the RGB image actually has.
    if (!rgb.empty()) {
      effective_height = std::min(effective_height, rgb.rows);
    }

    const int grid_h = (effective_height + ds - 1) / ds;
    const int grid_w = (width + ds - 1) / ds;

    // Small struct to hold per-grid 3D point and color
    struct GridPt { bool valid; float x,y,z; uint32_t rgb_packed; };

    std::vector<GridPt> grid(static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w));

    const bool depth16 = (depth.type() == CV_16UC1) || encodingIsDepth16U(depth_msg->encoding);
    const bool depth32 = (depth.type() == CV_32FC1) || encodingIsDepth32F(depth_msg->encoding);

    // Publish cropped RGB image (rows [0, effective_height))
    if (!rgb.empty() && effective_height > 0) {
      const cv::Mat cropped = rgb(cv::Range(0, std::min(effective_height, rgb.rows)), cv::Range::all());
      cv_bridge::CvImage out_msg;
      out_msg.header = rgb_msg->header;
      out_msg.header.stamp = depth_msg->header.stamp;
      out_msg.encoding = "bgr8";
      out_msg.image = cropped;
      auto img_ptr = out_msg.toImageMsg();
      if (cropped_rgb_pub_) {
        cropped_rgb_pub_->publish(*img_ptr);
        RCLCPP_INFO(this->get_logger(), "Published cropped RGB image size=%dx%d",
                    cropped.cols, cropped.rows);
      }
    }

    // Populate grid, skipping bottom crop rows
    for (int gv = 0; gv < grid_h; ++gv) {
      const int v = gv * ds;
      if (v >= effective_height) { // skip rows in the cropped area
        continue;
      }
      // If rgb image is smaller than depth, guard against out-of-range row access.
      if (rgb.empty() || v >= rgb.rows) {
        continue;
      }
      const auto * rgb_row = rgb.ptr<cv::Vec3b>(v);
      for (int gu = 0; gu < grid_w; ++gu) {
        const int u = gu * ds;
        GridPt & gp = grid[gv * grid_w + gu];
        gp.valid = false;

        // Guard against accessing columns that don't exist in either depth or rgb images.
        if (u >= width || (!rgb.empty() && u >= rgb.cols)) continue;

        float z = 0.0f;
        if (depth16) {
          const uint16_t d = depth.at<uint16_t>(v, u);
          z = static_cast<float>(d) * 0.001f;
        } else if (depth32) {
          z = depth.at<float>(v, u);
        } else {
          continue;
        }

        if (!std::isfinite(z) || z <= 0.0f || z > static_cast<float>(max_range_m_)) {
          continue;
        }

        const float x = static_cast<float>((u - cx) * z / fx);
        const float y = static_cast<float>((v - cy) * z / fy);

        const cv::Vec3b & c = rgb_row[u]; // BGR
        const uint8_t r = c[2];
        const uint8_t g = c[1];
        const uint8_t b = c[0];

        const uint32_t rgb_packed =
          (static_cast<uint32_t>(r) << 16) |
          (static_cast<uint32_t>(g) <<  8) |
          (static_cast<uint32_t>(b)      );

        gp.valid = true;
        gp.x = x; gp.y = y; gp.z = z; gp.rgb_packed = rgb_packed;
      }
    }

    // Build packed point vector from grid
    std::vector<float> points;
    points.reserve(static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w) * 4);
    for (int gv = 0; gv < grid_h; ++gv) {
      for (int gu = 0; gu < grid_w; ++gu) {
        const GridPt & gp = grid[gv * grid_w + gu];
        if (!gp.valid) continue;
        float rgb_float = 0.0f;
        std::memcpy(&rgb_float, &gp.rgb_packed, sizeof(float));
        points.push_back(gp.x);
        points.push_back(gp.y);
        points.push_back(gp.z);
        points.push_back(rgb_float);
      }
    }

    if (points.empty()) {
      return;
    }

    // Create PointCloud2 in the camera frame
    sensor_msgs::msg::PointCloud2 cloud_cam;
    cloud_cam.header.stamp = depth_msg->header.stamp;
    // Prefer camera_info frame, but fall back to depth frame if empty
    cloud_cam.header.frame_id = info_msg->header.frame_id.empty()
                                  ? depth_msg->header.frame_id
                                  : info_msg->header.frame_id;

    cloud_cam.height = 1;
    cloud_cam.width  = static_cast<uint32_t>(points.size() / 4);

    cloud_cam.fields.resize(4);
    cloud_cam.fields[0].name = "x";
    cloud_cam.fields[0].offset = 0;
    cloud_cam.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_cam.fields[0].count = 1;

    cloud_cam.fields[1].name = "y";
    cloud_cam.fields[1].offset = 4;
    cloud_cam.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_cam.fields[1].count = 1;

    cloud_cam.fields[2].name = "z";
    cloud_cam.fields[2].offset = 8;
    cloud_cam.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_cam.fields[2].count = 1;

    cloud_cam.fields[3].name = "rgb";
    cloud_cam.fields[3].offset = 12;
    cloud_cam.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_cam.fields[3].count = 1;

    cloud_cam.is_bigendian = false;
    cloud_cam.point_step = 16;
    cloud_cam.row_step = cloud_cam.point_step * cloud_cam.width;
    cloud_cam.is_dense = false;

    cloud_cam.data.resize(cloud_cam.row_step);
    std::memcpy(cloud_cam.data.data(), points.data(), cloud_cam.data.size());

    // Transform into target frame
    sensor_msgs::msg::PointCloud2 cloud_target;
    try {
      // timeout is important when TF is slightly delayed
      tf_buffer_.transform(cloud_cam, cloud_target, target_frame_, tf2::durationFromSec(0.1));
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "TF transform to '%s' failed: %s", target_frame_.c_str(), e.what());
      return;
    }

    // Publish the single-frame (non-accumulated) cloud if requested
    if (publish_single_frame_ && single_cloud_pub_) {
      single_cloud_pub_->publish(cloud_target);
      RCLCPP_INFO(this->get_logger(),
                  "Published single-frame point cloud with %u points (frame=%s)",
                  cloud_target.width,
                  cloud_target.header.frame_id.c_str());
    }

    // Accumulate (map) or publish single-frame
    if (!accumulate_) {
      cloud_pub_->publish(cloud_target);
      RCLCPP_INFO(this->get_logger(), "Published point cloud (non-accumulated) with %u points",
                  cloud_target.width);
      return;
    }

    // Accumulate bytes (each point is 16 bytes)
    {
      std::lock_guard<std::mutex> lock(accum_mutex_);

      // Append new cloud points
      accumulated_data_.insert(accumulated_data_.end(),
                               cloud_target.data.begin(),
                               cloud_target.data.end());
      accumulated_points_ += cloud_target.width;

      // Enforce max_points_ cap by dropping oldest points
      if (max_points_ > 0 && accumulated_points_ > max_points_) {
        const size_t overflow_pts = accumulated_points_ - max_points_;
        const size_t bytes_to_drop = overflow_pts * static_cast<size_t>(cloud_target.point_step);

        if (bytes_to_drop >= accumulated_data_.size()) {
          accumulated_data_.clear();
          accumulated_points_ = 0;
        } else {
          accumulated_data_.erase(accumulated_data_.begin(),
                                  accumulated_data_.begin() + static_cast<long>(bytes_to_drop));
          accumulated_points_ = max_points_;
        }
      }

      sensor_msgs::msg::PointCloud2 cloud_map = cloud_target;
      cloud_map.header.stamp = depth_msg->header.stamp;
      cloud_map.header.frame_id = target_frame_;
      cloud_map.height = 1;
      cloud_map.width = static_cast<uint32_t>(accumulated_points_);
      cloud_map.point_step = 16;
      cloud_map.row_step = cloud_map.point_step * cloud_map.width;
      cloud_map.data = accumulated_data_;

      cloud_pub_->publish(cloud_map);
      RCLCPP_INFO(this->get_logger(), "Published accumulated map point cloud with %u points",
                  cloud_map.width);
    }
  }

  // Params
  std::string depth_topic_;
  std::string rgb_topic_;
  std::string info_topic_;
  std::string output_topic_;
  std::string target_frame_;
  double bottom_crop_percent_ = 0.0;
  bool publish_single_frame_ = true;
  std::string single_frame_topic_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr single_cloud_pub_;
  std::string cropped_rgb_topic_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cropped_rgb_pub_;

  int downsample_step_ = 1;
  double max_range_m_ = 5.0;
  double publish_rate_hz_ = 5.0;
  double min_pub_interval_ = 0.0;
  bool accumulate_ = true;
  size_t max_points_ = 2'000'000;

  // Throttle
  std::chrono::steady_clock::time_point last_pub_steady_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Publisher
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  // message_filters
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> info_sub_;

  using Policy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
  std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;

  // Accumulation buffer
  std::mutex accum_mutex_;
  std::vector<uint8_t> accumulated_data_;
  size_t accumulated_points_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RGBPointCloudMapper>());
  rclcpp::shutdown();
  return 0;
}