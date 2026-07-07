#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace
{
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Mat4 = Eigen::Matrix<double, 4, 4>;

double stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nanosec);
}

builtin_interfaces::msg::Time seconds_to_stamp(double seconds)
{
  builtin_interfaces::msg::Time stamp;
  const auto nanoseconds = static_cast<int64_t>(std::llround(seconds * 1e9));
  stamp.sec = static_cast<int32_t>(nanoseconds / 1000000000LL);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000LL);
  return stamp;
}

double point_segment_distance_xy(
  const Eigen::Vector2d & p, const Eigen::Vector2d & a, const Eigen::Vector2d & b)
{
  const Eigen::Vector2d ab = b - a;
  const double denominator = ab.squaredNorm();
  const double u = denominator > 1e-12 ? std::clamp((p - a).dot(ab) / denominator, 0.0, 1.0) : 0.0;
  return (p - (a + u * ab)).norm();
}

double point_segment_distance_squared_3d(
  const Eigen::Vector3d & p, const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  const Eigen::Vector3d ab = b - a;
  const double denominator = ab.squaredNorm();
  const double u = denominator > 1e-12 ? std::clamp((p - a).dot(ab) / denominator, 0.0, 1.0) : 0.0;
  return (p - (a + u * ab)).squaredNorm();
}
}  // namespace

class CubeTrackerPclNode final : public rclcpp::Node
{
public:
  CubeTrackerPclNode()
  : Node("cube_tracker_pcl_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    declare_parameters();
    read_parameters();

    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(2);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic_, sensor_qos,
      std::bind(&CubeTrackerPclNode::cloud_callback, this, std::placeholders::_1));
    yolo_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      yolo_topic_, 10,
      std::bind(&CubeTrackerPclNode::yolo_callback, this, std::placeholders::_1));
    tracked_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(tracked_points_topic_, sensor_qos);
    pcl_measurement_pub_ =
      create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(pcl_measurement_topic_, 10);
    state_pub_ = create_publisher<nav_msgs::msg::Odometry>(cube_state_topic_, 10);
    visible_pub_ = create_publisher<std_msgs::msg::Bool>(cube_visible_topic_, 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(cube_marker_topic_, 10);
    visibility_timer_ = create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&CubeTrackerPclNode::visibility_timer, this));
    if (status_log_rate_hz_ > 0.0) {
      status_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / status_log_rate_hz_),
        std::bind(&CubeTrackerPclNode::status_timer, this));
    }

    RCLCPP_INFO(
      get_logger(), "PCL cube tracker ready: cloud=%s target=%s, max_rate=%.1f Hz",
      pointcloud_topic_.c_str(), target_frame_.c_str(), processing_rate_hz_);
  }

private:
  struct Segment
  {
    Eigen::Vector3d a;
    Eigen::Vector3d b;
  };

  struct Candidate
  {
    Cloud::Ptr cloud{new Cloud};
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d min{Eigen::Vector3d::Zero()};
    Eigen::Vector3d max{Eigen::Vector3d::Zero()};
    bool merged{false};
  };

  enum class Source { Pcl, Yolo };
  struct Measurement
  {
    double time{0.0};
    Eigen::Vector2d position{Eigen::Vector2d::Zero()};
    double variance{0.0};
    double gate{0.0};
    Source source{Source::Pcl};
  };
  struct Entry
  {
    Measurement measurement;
    Vec4 state{Vec4::Zero()};
    Mat4 covariance{Mat4::Identity()};
  };

  void declare_parameters()
  {
    declare_parameter("pointcloud_topic", "/camera/depth/color/points");
    declare_parameter("yolo_measurement_topic", "/go2_fetch/yolo/cube_measurement");
    declare_parameter("pcl_measurement_topic", "/go2_fetch/pcl/cube_measurement");
    declare_parameter("tracked_points_topic", "/go2_fetch/pcl/tracked_points");
    declare_parameter("cube_state_topic", "/go2_fetch/cube_state");
    declare_parameter("cube_visible_topic", "/go2_fetch/cube_visible");
    declare_parameter("cube_marker_topic", "/go2_fetch/cube_marker");
    declare_parameter("target_frame", "odom");
    declare_parameter("processing_rate_hz", 30.0);
    declare_parameter("status_log_rate_hz", 2.0);
    declare_parameter("min_range_m", 0.12);
    declare_parameter("max_range_m", 4.0);
    declare_parameter("voxel_leaf_m", 0.012);
    declare_parameter("floor_distance_threshold_m", 0.018);
    declare_parameter("floor_max_tilt_deg", 15.0);
    declare_parameter("floor_max_iterations", 60);
    declare_parameter("cluster_tolerance_m", 0.035);
    declare_parameter("fragment_min_points", 15);
    declare_parameter("cluster_min_points", 40);
    declare_parameter("cluster_max_points", 12000);
    declare_parameter("cube_dimensions", std::vector<double>{0.16, 0.16, 0.16});
    declare_parameter("cube_dimension_min_m", 0.045);
    declare_parameter("cube_dimension_max_m", 0.28);
    declare_parameter("merged_min_visible_fraction", 0.4);
    declare_parameter("fragment_depth_tolerance_m", 0.15);
    declare_parameter("association_gate_m", 0.60);
    declare_parameter(
      "leg_frames", std::vector<std::string>{
        "FL_thigh", "FL_calf", "FL_foot", "FR_thigh", "FR_calf", "FR_foot"});
    declare_parameter("leg_exclusion_radius_m", 0.065);
    declare_parameter("leg_merge_gap_radius_m", 0.13);
    declare_parameter("process_accel_std_mps2", 1.5);
    declare_parameter("velocity_decay_tau_s", 0.25);
    declare_parameter("pcl_measurement_variance_xy", 0.0064);
    declare_parameter("yolo_measurement_variance_xy", 0.0025);
    declare_parameter("pcl_innovation_gate", 9.21);
    declare_parameter("yolo_innovation_gate", 13.82);
    declare_parameter("history_duration_s", 0.5);
    declare_parameter("detection_timeout_s", 2.2);
  }

  void read_parameters()
  {
    pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
    yolo_topic_ = get_parameter("yolo_measurement_topic").as_string();
    pcl_measurement_topic_ = get_parameter("pcl_measurement_topic").as_string();
    tracked_points_topic_ = get_parameter("tracked_points_topic").as_string();
    cube_state_topic_ = get_parameter("cube_state_topic").as_string();
    cube_visible_topic_ = get_parameter("cube_visible_topic").as_string();
    cube_marker_topic_ = get_parameter("cube_marker_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    processing_rate_hz_ = get_parameter("processing_rate_hz").as_double();
    status_log_rate_hz_ = get_parameter("status_log_rate_hz").as_double();
    min_range_ = get_parameter("min_range_m").as_double();
    max_range_ = get_parameter("max_range_m").as_double();
    voxel_leaf_ = get_parameter("voxel_leaf_m").as_double();
    floor_threshold_ = get_parameter("floor_distance_threshold_m").as_double();
    floor_tilt_rad_ = get_parameter("floor_max_tilt_deg").as_double() * 3.14159265358979323846 / 180.0;
    floor_max_iterations_ = get_parameter("floor_max_iterations").as_int();
    cluster_tolerance_ = get_parameter("cluster_tolerance_m").as_double();
    fragment_min_points_ = get_parameter("fragment_min_points").as_int();
    cluster_min_points_ = get_parameter("cluster_min_points").as_int();
    cluster_max_points_ = get_parameter("cluster_max_points").as_int();
    const auto dimensions = get_parameter("cube_dimensions").as_double_array();
    if (dimensions.size() != 3U) {
      throw std::invalid_argument("cube_dimensions must contain exactly three values");
    }
    cube_dimensions_ = Eigen::Vector3d(dimensions[0], dimensions[1], dimensions[2]);
    cube_dimension_min_ = get_parameter("cube_dimension_min_m").as_double();
    cube_dimension_max_ = get_parameter("cube_dimension_max_m").as_double();
    merged_visible_fraction_ = get_parameter("merged_min_visible_fraction").as_double();
    fragment_depth_tolerance_ = get_parameter("fragment_depth_tolerance_m").as_double();
    association_gate_ = get_parameter("association_gate_m").as_double();
    leg_frames_ = get_parameter("leg_frames").as_string_array();
    leg_radius_ = get_parameter("leg_exclusion_radius_m").as_double();
    leg_merge_radius_ = get_parameter("leg_merge_gap_radius_m").as_double();
    process_accel_std_ = get_parameter("process_accel_std_mps2").as_double();
    velocity_decay_tau_ = get_parameter("velocity_decay_tau_s").as_double();
    if (velocity_decay_tau_ <= 0.0) {
      throw std::invalid_argument("velocity_decay_tau_s must be greater than zero");
    }
    pcl_variance_ = get_parameter("pcl_measurement_variance_xy").as_double();
    yolo_variance_ = get_parameter("yolo_measurement_variance_xy").as_double();
    pcl_gate_ = get_parameter("pcl_innovation_gate").as_double();
    yolo_gate_ = get_parameter("yolo_innovation_gate").as_double();
    history_duration_ = get_parameter("history_duration_s").as_double();
    detection_timeout_ = get_parameter("detection_timeout_s").as_double();
  }

  std::optional<Eigen::Vector3d> frame_origin(
    const std::string & frame, const builtin_interfaces::msg::Time & stamp)
  {
    try {
      const auto tf = tf_buffer_.lookupTransform(target_frame_, frame, rclcpp::Time(stamp), rclcpp::Duration::from_seconds(0.02));
      return Eigen::Vector3d(
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "TF %s <- %s unavailable: %s",
        target_frame_.c_str(), frame.c_str(), error.what());
      return std::nullopt;
    }
  }

  std::vector<Segment> leg_segments(const builtin_interfaces::msg::Time & stamp)
  {
    std::vector<Segment> segments;
    for (std::size_t i = 0; i + 2 < leg_frames_.size(); i += 3) {
      const auto thigh = frame_origin(leg_frames_[i], stamp);
      const auto calf = frame_origin(leg_frames_[i + 1], stamp);
      const auto foot = frame_origin(leg_frames_[i + 2], stamp);
      if (thigh && calf && foot) {
        segments.push_back({*thigh, *calf});
        segments.push_back({*calf, *foot});
      }
    }
    return segments;
  }

  Cloud::Ptr preprocess(
    const sensor_msgs::msg::PointCloud2 & message, std::vector<Segment> & segments,
    Eigen::Vector3d & camera_origin)
  {
    geometry_msgs::msg::TransformStamped cloud_transform;
    try {
      cloud_transform = tf_buffer_.lookupTransform(
        target_frame_, message.header.frame_id, rclcpp::Time(message.header.stamp),
        rclcpp::Duration::from_seconds(0.04));
      camera_origin = Eigen::Vector3d(
        cloud_transform.transform.translation.x,
        cloud_transform.transform.translation.y,
        cloud_transform.transform.translation.z);
    } catch (const tf2::TransformException & error) {
      ++tf_failures_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Cloud transform failed: %s", error.what());
      return {};
    }

    Cloud::Ptr raw(new Cloud);
    pcl::fromROSMsg(message, *raw);
    Cloud::Ptr ranged(new Cloud);
    ranged->reserve(raw->size());
    for (const auto & p : raw->points) {
      if (!pcl::isFinite(p)) {continue;}
      // The raw cloud is still in the camera frame, so its origin is (0, 0, 0).
      const double range_squared =
        static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y +
        static_cast<double>(p.z) * p.z;
      if (range_squared >= min_range_ * min_range_ && range_squared <= max_range_ * max_range_) {
        ranged->push_back(p);
      }
    }

    Cloud::Ptr downsampled_camera(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(ranged);
    voxel.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    voxel.filter(*downsampled_camera);

    // Transform only the reduced cloud instead of every raw camera point.
    const auto & translation = cloud_transform.transform.translation;
    const auto & rotation = cloud_transform.transform.rotation;
    Eigen::Quaternionf orientation(
      static_cast<float>(rotation.w), static_cast<float>(rotation.x),
      static_cast<float>(rotation.y), static_cast<float>(rotation.z));
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.linear() = orientation.normalized().toRotationMatrix();
    transform.translation() = Eigen::Vector3f(
      static_cast<float>(translation.x), static_cast<float>(translation.y),
      static_cast<float>(translation.z));
    Cloud::Ptr downsampled(new Cloud);
    pcl::transformPointCloud(*downsampled_camera, *downsampled, transform);

    pcl::PointIndices::Ptr floor_indices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> floor;
    floor.setOptimizeCoefficients(true);
    floor.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    floor.setMethodType(pcl::SAC_RANSAC);
    floor.setAxis(Eigen::Vector3f::UnitZ());
    floor.setEpsAngle(floor_tilt_rad_);
    floor.setDistanceThreshold(floor_threshold_);
    floor.setMaxIterations(floor_max_iterations_);
    floor.setInputCloud(downsampled);
    floor.segment(*floor_indices, *coefficients);

    Cloud::Ptr no_floor(new Cloud);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(downsampled);
    extract.setIndices(floor_indices);
    extract.setNegative(true);
    extract.filter(*no_floor);

    segments = leg_segments(message.header.stamp);
    Cloud::Ptr no_robot(new Cloud);
    no_robot->reserve(no_floor->size());
    const double leg_radius_squared = leg_radius_ * leg_radius_;
    for (const auto & p : no_floor->points) {
      const Eigen::Vector3d point(p.x, p.y, p.z);
      const bool is_leg = std::any_of(segments.begin(), segments.end(), [&](const Segment & segment) {
        return point_segment_distance_squared_3d(point, segment.a, segment.b) <= leg_radius_squared;
      });
      if (!is_leg) {no_robot->push_back(p);}
    }
    no_robot->width = no_robot->size();
    no_robot->height = 1;
    return no_robot;
  }

  Candidate make_candidate(const Cloud::Ptr & cloud, bool merged = false) const
  {
    Candidate candidate;
    candidate.cloud = cloud;
    candidate.merged = merged;
    pcl::PointXYZ minimum, maximum;
    pcl::getMinMax3D(*cloud, minimum, maximum);
    candidate.min = Eigen::Vector3d(minimum.x, minimum.y, minimum.z);
    candidate.max = Eigen::Vector3d(maximum.x, maximum.y, maximum.z);

    std::vector<float> xs, ys, zs;
    xs.reserve(cloud->size()); ys.reserve(cloud->size()); zs.reserve(cloud->size());
    for (const auto & p : cloud->points) {xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z);}
    const auto median = [](std::vector<float> & values) {
      const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
      std::nth_element(values.begin(), middle, values.end());
      return static_cast<double>(*middle);
    };
    candidate.center = Eigen::Vector3d(median(xs), median(ys), median(zs));
    return candidate;
  }

  bool dimensions_valid(const Candidate & candidate) const
  {
    const Eigen::Vector3d extent = candidate.max - candidate.min;
    return extent.maxCoeff() <= cube_dimension_max_ &&
           std::max(extent.x(), extent.y()) >= cube_dimension_min_ &&
           extent.z() >= cube_dimension_min_ * 0.5;
  }

  bool gap_explained_by_leg(
    const Candidate & a, const Candidate & b, const std::vector<Segment> & segments) const
  {
    const Eigen::Vector2d ca = a.center.head<2>();
    const Eigen::Vector2d cb = b.center.head<2>();
    return std::any_of(segments.begin(), segments.end(), [&](const Segment & segment) {
      const Eigen::Vector2d midpoint = 0.5 * (segment.a.head<2>() + segment.b.head<2>());
      return point_segment_distance_xy(midpoint, ca, cb) <= leg_merge_radius_;
    });
  }

  std::vector<Candidate> candidates(
    const Cloud::Ptr & cloud, const std::vector<Segment> & segments,
    const Eigen::Vector3d & camera_origin)
  {
    last_raw_cluster_count_ = 0;
    last_merged_candidate_count_ = 0;
    if (!cloud || cloud->size() < static_cast<std::size_t>(fragment_min_points_)) {return {};}
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> extraction;
    extraction.setClusterTolerance(cluster_tolerance_);
    extraction.setMinClusterSize(fragment_min_points_);
    extraction.setMaxClusterSize(cluster_max_points_);
    extraction.setSearchMethod(tree);
    extraction.setInputCloud(cloud);
    extraction.extract(indices);
    last_raw_cluster_count_ = indices.size();

    std::vector<Candidate> fragments;
    for (const auto & index : indices) {
      Cloud::Ptr fragment(new Cloud);
      pcl::copyPointCloud(*cloud, index.indices, *fragment);
      fragments.push_back(make_candidate(fragment));
    }

    std::vector<Candidate> output;
    for (const auto & fragment : fragments) {
      const Eigen::Vector3d extent = fragment.max - fragment.min;
      const double visible_fraction = std::max(extent.x(), extent.y()) /
        std::max(cube_dimensions_.x(), cube_dimensions_.y());
      if (fragment.cloud->size() >= static_cast<std::size_t>(cluster_min_points_) &&
        visible_fraction >= merged_visible_fraction_ && dimensions_valid(fragment)) {
        output.push_back(fragment);
      }
    }

    for (std::size_t i = 0; i < fragments.size(); ++i) {
      for (std::size_t j = i + 1; j < fragments.size(); ++j) {
        const double depth_i = (fragments[i].center - camera_origin).norm();
        const double depth_j = (fragments[j].center - camera_origin).norm();
        if (std::abs(depth_i - depth_j) > fragment_depth_tolerance_ ||
          !gap_explained_by_leg(fragments[i], fragments[j], segments)) {continue;}
        Cloud::Ptr merged(new Cloud(*fragments[i].cloud));
        *merged += *fragments[j].cloud;
        Candidate candidate = make_candidate(merged, true);
        const Eigen::Vector3d extent = candidate.max - candidate.min;
        const double visible_fraction = std::max(extent.x(), extent.y()) /
          std::max(cube_dimensions_.x(), cube_dimensions_.y());
        if (merged->size() >= static_cast<std::size_t>(cluster_min_points_) &&
          visible_fraction >= merged_visible_fraction_ && dimensions_valid(candidate)) {
          output.push_back(std::move(candidate));
          ++last_merged_candidate_count_;
        }
      }
    }
    return output;
  }

  std::optional<Eigen::Vector2d> predicted_position(double time) const
  {
    if (!initialized_) {return std::nullopt;}
    Vec4 state;
    Mat4 covariance;
    double state_time;
    latest_filter_state(state, covariance, state_time);
    predict(
      state, covariance, std::max(0.0, time - state_time),
      process_accel_std_, velocity_decay_tau_);
    return state.head<2>();
  }

  std::optional<Candidate> select_candidate(
    const std::vector<Candidate> & options, const Eigen::Vector3d & camera_origin, double time) const
  {
    if (options.empty()) {return std::nullopt;}
    const auto prediction = predicted_position(time);
    double best_score = std::numeric_limits<double>::infinity();
    const Candidate * best = nullptr;
    for (const auto & candidate : options) {
      const double score = prediction ?
        (candidate.center.head<2>() - *prediction).norm() : (candidate.center - camera_origin).norm();
      if (prediction && score > association_gate_) {continue;}
      if (score < best_score) {best_score = score; best = &candidate;}
    }
    if (!best) {return std::nullopt;}
    return *best;
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    ++clouds_received_total_;
    ++clouds_received_window_;
    const double time = stamp_seconds(message->header.stamp);
    if (time <= 0.0) {return;}
    if (processing_rate_hz_ > 0.0 && last_cloud_time_ > 0.0 &&
      time - last_cloud_time_ < 1.0 / processing_rate_hz_ - 1e-4) {return;}
    last_cloud_time_ = time;
    latest_sensor_time_ = std::max(latest_sensor_time_, time);
    ++clouds_processed_total_;
    ++clouds_processed_window_;
    const auto processing_start = std::chrono::steady_clock::now();

    std::vector<Segment> segments;
    Eigen::Vector3d camera_origin;
    const auto filtered = preprocess(*message, segments, camera_origin);
    if (!filtered) {
      last_processing_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processing_start).count();
      return;
    }
    last_filtered_points_ = filtered->size();
    const auto candidate_list = candidates(filtered, segments, camera_origin);
    last_candidate_count_ = candidate_list.size();
    const auto selected = select_candidate(candidate_list, camera_origin, time);
    if (!selected) {
      last_selected_points_ = 0;
      if (initialized_) {publish_state(latest_sensor_time_);}
      last_processing_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processing_start).count();
      return;
    }

    ++clusters_selected_total_;
    last_selected_points_ = selected->cloud->size();
    publish_tracked_cloud(*selected, message->header.stamp);
    publish_pcl_measurement(*selected, message->header.stamp);
    Measurement measurement{
      time, selected->center.head<2>(), pcl_variance_, pcl_gate_, Source::Pcl};
    if (insert_measurement(measurement)) {
      ++pcl_updates_accepted_;
      last_observation_wall_ = std::chrono::steady_clock::now();
      publish_state(latest_sensor_time_);
    } else {
      ++pcl_updates_rejected_;
    }
    last_processing_ms_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - processing_start).count();
  }

  void yolo_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr message)
  {
    ++yolo_received_;
    if (message->header.frame_id != target_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "Ignoring YOLO measurement in frame '%s'; expected '%s'",
        message->header.frame_id.c_str(), target_frame_.c_str());
      return;
    }
    const double time = stamp_seconds(message->header.stamp);
    if (time <= 0.0 || !std::isfinite(message->pose.pose.position.x) ||
      !std::isfinite(message->pose.pose.position.y)) {return;}
    const double supplied_variance = 0.5 *
      (message->pose.covariance[0] + message->pose.covariance[7]);
    Measurement measurement{
      time,
      Eigen::Vector2d(message->pose.pose.position.x, message->pose.pose.position.y),
      supplied_variance > 0.0 ? supplied_variance : yolo_variance_, yolo_gate_, Source::Yolo};
    if (insert_measurement(measurement)) {
      ++yolo_updates_accepted_;
      last_observation_wall_ = std::chrono::steady_clock::now();
      if (latest_sensor_time_ <= 0.0) {latest_sensor_time_ = time;}
      publish_state(std::max(time, latest_sensor_time_));
    } else {
      ++yolo_updates_rejected_;
    }
  }

  void status_timer()
  {
    const auto now = std::chrono::steady_clock::now();
    const double window_s = std::max(
      1e-6, std::chrono::duration<double>(now - status_window_start_).count());
    const double input_hz = static_cast<double>(clouds_received_window_) / window_s;
    const double processing_hz = static_cast<double>(clouds_processed_window_) / window_s;
    if (clouds_received_total_ == 0) {
      RCLCPP_WARN(get_logger(), "PCL status: no PointCloud2 messages received on %s", pointcloud_topic_.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "PCL status: input=%.1fHz processed=%.1fHz time=%.1fms filtered=%zu "
        "raw_clusters=%zu candidates=%zu merged=%zu selected_points=%zu "
        "selected_total=%zu tf_failures=%zu "
        "ekf_pcl=%zu/%zu ekf_yolo=%zu/%zu",
        input_hz, processing_hz, last_processing_ms_, last_filtered_points_,
        last_raw_cluster_count_, last_candidate_count_, last_merged_candidate_count_,
        last_selected_points_, clusters_selected_total_, tf_failures_,
        pcl_updates_accepted_, pcl_updates_rejected_, yolo_updates_accepted_,
        yolo_updates_rejected_);
    }
    clouds_received_window_ = 0;
    clouds_processed_window_ = 0;
    status_window_start_ = now;
  }

  static void predict(
    Vec4 & state, Mat4 & covariance, double dt,
    double acceleration_std, double velocity_decay_tau)
  {
    if (dt <= 0.0) {return;}

    // A rolling cube loses speed to floor friction. Exponential damping avoids the
    // unbounded position drift of a constant-velocity model during an occlusion.
    // The exact discrete transition also makes the result independent of callback rate.
    const double velocity_decay = std::exp(-dt / velocity_decay_tau);
    const double position_gain = velocity_decay_tau * (1.0 - velocity_decay);
    Mat4 transition = Mat4::Identity();
    transition(0, 2) = position_gain; transition(1, 3) = position_gain;
    transition(2, 2) = velocity_decay; transition(3, 3) = velocity_decay;
    const double q = acceleration_std * acceleration_std;
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    Mat4 process = Mat4::Zero();
    process(0, 0) = process(1, 1) = 0.25 * dt4 * q;
    process(0, 2) = process(2, 0) = process(1, 3) = process(3, 1) = 0.5 * dt3 * q;
    process(2, 2) = process(3, 3) = dt2 * q;
    state = transition * state;
    covariance = transition * covariance * transition.transpose() + process;
  }

  bool correct(Vec4 & state, Mat4 & covariance, const Measurement & measurement) const
  {
    Eigen::Matrix<double, 2, 4> observation = Eigen::Matrix<double, 2, 4>::Zero();
    observation(0, 0) = 1.0; observation(1, 1) = 1.0;
    const Eigen::Matrix2d noise = Eigen::Matrix2d::Identity() * measurement.variance;
    const Eigen::Vector2d innovation = measurement.position - observation * state;
    const Eigen::Matrix2d innovation_covariance = observation * covariance * observation.transpose() + noise;
    const double mahalanobis = innovation.dot(innovation_covariance.ldlt().solve(innovation));
    if (!std::isfinite(mahalanobis) || mahalanobis > measurement.gate) {return false;}
    const Eigen::Matrix<double, 4, 2> gain =
      covariance * observation.transpose() * innovation_covariance.inverse();
    state += gain * innovation;
    const Mat4 identity = Mat4::Identity();
    const Mat4 residual = identity - gain * observation;
    covariance = residual * covariance * residual.transpose() + gain * noise * gain.transpose();
    return true;
  }

  void latest_filter_state(Vec4 & state, Mat4 & covariance, double & time) const
  {
    if (entries_.empty()) {
      state = anchor_state_; covariance = anchor_covariance_; time = anchor_time_;
    } else {
      state = entries_.back().state; covariance = entries_.back().covariance;
      time = entries_.back().measurement.time;
    }
  }

  bool insert_measurement(const Measurement & measurement)
  {
    if (!initialized_) {
      initialized_ = true;
      anchor_time_ = measurement.time;
      anchor_state_ << measurement.position.x(), measurement.position.y(), 0.0, 0.0;
      anchor_covariance_ = Mat4::Zero();
      anchor_covariance_(0, 0) = anchor_covariance_(1, 1) = measurement.variance;
      anchor_covariance_(2, 2) = anchor_covariance_(3, 3) = 0.5;
      return true;
    }
    if (measurement.time + 1e-6 < anchor_time_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Discarding measurement older than EKF history");
      return false;
    }

    auto position = std::upper_bound(
      entries_.begin(), entries_.end(), measurement.time,
      [](double time, const Entry & entry) {return time < entry.measurement.time;});
    const std::size_t index = static_cast<std::size_t>(std::distance(entries_.begin(), position));
    entries_.insert(position, Entry{measurement, Vec4::Zero(), Mat4::Identity()});

    Vec4 state = index == 0 ? anchor_state_ : entries_[index - 1].state;
    Mat4 covariance = index == 0 ? anchor_covariance_ : entries_[index - 1].covariance;
    double state_time = index == 0 ? anchor_time_ : entries_[index - 1].measurement.time;
    bool inserted_accepted = false;
    for (std::size_t i = index; i < entries_.size(); ++i) {
      predict(
        state, covariance, std::max(0.0, entries_[i].measurement.time - state_time),
        process_accel_std_, velocity_decay_tau_);
      const bool accepted = correct(state, covariance, entries_[i].measurement);
      if (i == index) {inserted_accepted = accepted;}
      entries_[i].state = state;
      entries_[i].covariance = covariance;
      state_time = entries_[i].measurement.time;
    }
    if (!inserted_accepted) {
      entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
      replay_from(index);
      return false;
    }
    prune_history(std::max(latest_sensor_time_, measurement.time));
    return true;
  }

  void replay_from(std::size_t index)
  {
    if (index > entries_.size()) {return;}
    Vec4 state = index == 0 ? anchor_state_ : entries_[index - 1].state;
    Mat4 covariance = index == 0 ? anchor_covariance_ : entries_[index - 1].covariance;
    double state_time = index == 0 ? anchor_time_ : entries_[index - 1].measurement.time;
    for (std::size_t i = index; i < entries_.size(); ++i) {
      predict(
        state, covariance, std::max(0.0, entries_[i].measurement.time - state_time),
        process_accel_std_, velocity_decay_tau_);
      correct(state, covariance, entries_[i].measurement);
      entries_[i].state = state; entries_[i].covariance = covariance;
      state_time = entries_[i].measurement.time;
    }
  }

  void prune_history(double latest_time)
  {
    const double cutoff = latest_time - history_duration_;
    while (!entries_.empty() && entries_.front().measurement.time < cutoff) {
      anchor_time_ = entries_.front().measurement.time;
      anchor_state_ = entries_.front().state;
      anchor_covariance_ = entries_.front().covariance;
      entries_.pop_front();
    }
  }

  void publish_tracked_cloud(const Candidate & candidate, const builtin_interfaces::msg::Time & stamp)
  {
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*candidate.cloud, output);
    output.header.frame_id = target_frame_;
    output.header.stamp = stamp;
    tracked_cloud_pub_->publish(output);
  }

  void publish_pcl_measurement(const Candidate & candidate, const builtin_interfaces::msg::Time & stamp)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped output;
    output.header.frame_id = target_frame_; output.header.stamp = stamp;
    output.pose.pose.position.x = candidate.center.x();
    output.pose.pose.position.y = candidate.center.y();
    output.pose.pose.position.z = candidate.center.z();
    output.pose.pose.orientation.w = 1.0;
    output.pose.covariance[0] = output.pose.covariance[7] = pcl_variance_;
    output.pose.covariance[14] = pcl_variance_ * 4.0;
    pcl_measurement_pub_->publish(output);
  }

  void publish_state(double time)
  {
    if (!initialized_) {return;}
    Vec4 state; Mat4 covariance; double state_time;
    latest_filter_state(state, covariance, state_time);
    predict(
      state, covariance, std::max(0.0, time - state_time),
      process_accel_std_, velocity_decay_tau_);
    nav_msgs::msg::Odometry output;
    output.header.frame_id = target_frame_;
    output.header.stamp = seconds_to_stamp(time);
    output.child_frame_id = "cube";
    output.pose.pose.position.x = state(0); output.pose.pose.position.y = state(1);
    output.pose.pose.orientation.w = 1.0;
    output.pose.covariance[0] = covariance(0, 0); output.pose.covariance[7] = covariance(1, 1);
    output.twist.twist.linear.x = state(2); output.twist.twist.linear.y = state(3);
    output.twist.covariance[0] = covariance(2, 2); output.twist.covariance[7] = covariance(3, 3);
    state_pub_->publish(output);
    publish_marker(output);
  }

  void publish_marker(const nav_msgs::msg::Odometry & state)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = state.header; marker.ns = "cube_state"; marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = state.pose.pose; marker.pose.position.z = cube_dimensions_.z() * 0.5;
    marker.scale.x = cube_dimensions_.x(); marker.scale.y = cube_dimensions_.y();
    marker.scale.z = cube_dimensions_.z();
    marker.color.r = 1.0F; marker.color.g = 1.0F; marker.color.b = 0.2F; marker.color.a = 0.75F;
    marker_pub_->publish(marker);
  }

  void visibility_timer()
  {
    const auto now = std::chrono::steady_clock::now();
    const bool visible = initialized_ && last_observation_wall_.time_since_epoch().count() != 0 &&
      std::chrono::duration<double>(now - last_observation_wall_).count() <= detection_timeout_;
    if (visible == last_visible_) {return;}
    last_visible_ = visible;
    std_msgs::msg::Bool message; message.data = visible; visible_pub_->publish(message);
    if (!visible) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = target_frame_; marker.header.stamp = now_ros();
      marker.ns = "cube_state"; marker.id = 0;
      marker.action = visualization_msgs::msg::Marker::DELETE;
      marker_pub_->publish(marker);
    }
  }

  builtin_interfaces::msg::Time now_ros()
  {
    const auto now = get_clock()->now();
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(now.nanoseconds() / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(now.nanoseconds() % 1000000000LL);
    return stamp;
  }

  std::string pointcloud_topic_, yolo_topic_, pcl_measurement_topic_, tracked_points_topic_;
  std::string cube_state_topic_, cube_visible_topic_, cube_marker_topic_, target_frame_;
  double processing_rate_hz_{30.0}, status_log_rate_hz_{2.0};
  double min_range_{0.12}, max_range_{4.0}, voxel_leaf_{0.012};
  double floor_threshold_{0.018}, floor_tilt_rad_{0.26}, cluster_tolerance_{0.035};
  int floor_max_iterations_{60};
  int fragment_min_points_{15}, cluster_min_points_{40}, cluster_max_points_{12000};
  Eigen::Vector3d cube_dimensions_{0.16, 0.16, 0.16};
  double cube_dimension_min_{0.045}, cube_dimension_max_{0.28}, merged_visible_fraction_{0.4};
  double fragment_depth_tolerance_{0.15}, association_gate_{0.60};
  std::vector<std::string> leg_frames_;
  double leg_radius_{0.065}, leg_merge_radius_{0.13};
  double process_accel_std_{1.5}, velocity_decay_tau_{0.25};
  double pcl_variance_{0.0064}, yolo_variance_{0.0025};
  double pcl_gate_{9.21}, yolo_gate_{13.82}, history_duration_{0.5}, detection_timeout_{2.2};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr yolo_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tracked_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pcl_measurement_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr visible_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr visibility_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  bool initialized_{false}, last_visible_{false};
  double anchor_time_{0.0}, latest_sensor_time_{0.0}, last_cloud_time_{0.0};
  Vec4 anchor_state_{Vec4::Zero()};
  Mat4 anchor_covariance_{Mat4::Identity()};
  std::deque<Entry> entries_;
  std::chrono::steady_clock::time_point last_observation_wall_{};
  std::chrono::steady_clock::time_point status_window_start_{std::chrono::steady_clock::now()};
  std::size_t clouds_received_total_{0}, clouds_received_window_{0};
  std::size_t clouds_processed_total_{0}, clouds_processed_window_{0};
  std::size_t tf_failures_{0}, clusters_selected_total_{0};
  std::size_t pcl_updates_accepted_{0}, pcl_updates_rejected_{0};
  std::size_t yolo_received_{0}, yolo_updates_accepted_{0}, yolo_updates_rejected_{0};
  std::size_t last_filtered_points_{0}, last_raw_cluster_count_{0};
  std::size_t last_candidate_count_{0}, last_merged_candidate_count_{0};
  std::size_t last_selected_points_{0};
  double last_processing_ms_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CubeTrackerPclNode>());
  rclcpp::shutdown();
  return 0;
}
