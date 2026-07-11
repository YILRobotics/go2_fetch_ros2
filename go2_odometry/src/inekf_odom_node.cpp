#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "inekf/InEKF.hpp"
#include "inekf/NoiseParams.hpp"
#include "inekf/RobotState.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/model.hxx"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "unitree_go/msg/low_state.hpp"

class InekfOdomNode : public rclcpp::Node
{
public:
  InekfOdomNode()
  : Node("inekf")
  {
    declare_parameters();
    read_parameters();
    load_robot_model();
    initialize_filter_state();

    lowstate_subscription_ = create_subscription<unitree_go::msg::LowState>(
      "/lowstate", rclcpp::QoS(10),
      std::bind(&InekfOdomNode::lowstate_callback, this, std::placeholders::_1));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("/go2_odometry/filtered", 1);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

private:
  struct FeetKinematics
  {
    std::array<bool, 4> contacts;
    std::array<pinocchio::SE3, 4> imu_M_foot;
    std::array<Eigen::Matrix3d, 4> normed_covariances;
  };

  void declare_parameters()
  {
    declare_parameter<std::string>("base_frame", "base");
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<double>("robot_freq", 500.0);
    declare_parameter<bool>("wait_for_all_feet_contact", true);
    declare_parameter<double>("gyroscope_noise", 0.01);
    declare_parameter<double>("accelerometer_noise", 0.1);
    declare_parameter<double>("gyroscopeBias_noise", 0.00001);
    declare_parameter<double>("accelerometerBias_noise", 0.0001);
    declare_parameter<double>("contact_noise", 0.001);
    declare_parameter<double>("joint_position_noise", 0.001);
    declare_parameter<double>("contact_velocity_noise", 0.001);
  }

  void read_parameters()
  {
    base_frame_ = get_parameter("base_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    dt_ = 1.0 / get_parameter("robot_freq").as_double();
    wait_for_all_feet_contact_ = get_parameter("wait_for_all_feet_contact").as_bool();
    joint_position_noise_ = get_parameter("joint_position_noise").as_double();
    contact_velocity_noise_ = get_parameter("contact_velocity_noise").as_double();
  }

  void load_robot_model()
  {
    const std::string unitree_share = ament_index_cpp::get_package_share_directory("unitree_description");
    const std::string urdf_path = unitree_share + "/model/go2/go2.urdf";
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    data_ = pinocchio::Data(model_);

    foot_frame_names_ = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    for (std::size_t i = 0; i < foot_frame_names_.size(); ++i)
    {
      foot_frame_ids_[i] = model_.getFrameId(foot_frame_names_[i]);
      assert(foot_frame_ids_[i] < model_.frames.size());
    }
    imu_frame_id_ = model_.getFrameId("imu");
    base_frame_id_ = model_.getFrameId(base_frame_);
    assert(imu_frame_id_ < model_.frames.size());
    assert(base_frame_id_ < model_.frames.size());

    Eigen::VectorXd q_neutral = pinocchio::neutral(model_);
    pinocchio::forwardKinematics(model_, data_, q_neutral);
    pinocchio::updateFramePlacements(model_, data_);
    const pinocchio::SE3 & o_M_imu = data_.oMf[imu_frame_id_];
    const pinocchio::SE3 & o_M_base = data_.oMf[base_frame_id_];
    imu_M_base_ = o_M_imu.actInv(o_M_base);
  }

  void initialize_filter_state()
  {
    inekf::RobotState initial_state;
    initial_state.setRotation(Eigen::Matrix3d::Identity());
    initial_state.setVelocity(Eigen::Vector3d::Zero());
    initial_state.setPosition(Eigen::Vector3d::Zero());
    initial_state.setGyroscopeBias(Eigen::Vector3d::Zero());
    initial_state.setAccelerometerBias(Eigen::Vector3d::Zero());

    inekf::NoiseParams noise_params;
    noise_params.setGyroscopeNoise(get_parameter("gyroscope_noise").as_double());
    noise_params.setAccelerometerNoise(get_parameter("accelerometer_noise").as_double());
    noise_params.setGyroscopeBiasNoise(get_parameter("gyroscopeBias_noise").as_double());
    noise_params.setAccelerometerBiasNoise(get_parameter("accelerometerBias_noise").as_double());
    noise_params.setContactNoise(get_parameter("contact_noise").as_double());

    filter_ = inekf::InEKF(initial_state, noise_params);
    filter_.setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
  }

  void lowstate_callback(const unitree_go::msg::LowState::SharedPtr msg)
  {
    Eigen::VectorXd imu_state(6);
    for (int i = 0; i < 3; ++i)
    {
      imu_state(i) = msg->imu_state.gyroscope[i];
      imu_state(i + 3) = msg->imu_state.accelerometer[i];
    }

    const FeetKinematics feet = feet_transformations(*msg);

    if (pause_)
    {
      if (!wait_for_all_feet_contact_ || all_feet_in_contact(feet.contacts))
      {
        pause_ = false;
        initialize_filter(*msg);
        RCLCPP_INFO(get_logger(), "Starting filter.");
      }
      else
      {
        RCLCPP_INFO_ONCE(get_logger(), "Waiting for all feet to touch the ground to start filter.");
        return;
      }
    }

    filter_.propagate(imu_state, dt_);

    std::vector<std::pair<int, bool>> contact_pairs;
    inekf::vectorKinematics kinematics_list;
    contact_pairs.reserve(foot_frame_names_.size());
    kinematics_list.reserve(foot_frame_names_.size());
    for (std::size_t i = 0; i < foot_frame_names_.size(); ++i)
    {
      contact_pairs.emplace_back(static_cast<int>(i), feet.contacts[i]);
      kinematics_list.emplace_back(
        static_cast<int>(i),
        feet.imu_M_foot[i].translation(),
        joint_position_noise_ * feet.normed_covariances[i],
        Eigen::Vector3d::Zero(),
        contact_velocity_noise_ * Eigen::Matrix3d::Identity());
    }

    filter_.setContacts(contact_pairs);
    filter_.correctKinematics(kinematics_list);

    Eigen::Vector3d angular_velocity;
    for (int i = 0; i < 3; ++i)
    {
      angular_velocity(i) = msg->imu_state.gyroscope[i];
    }
    publish_state(filter_.getState(), angular_velocity);
  }

  static bool all_feet_in_contact(const std::array<bool, 4> & contacts)
  {
    return contacts[0] && contacts[1] && contacts[2] && contacts[3];
  }

  void fill_pinocchio_state(
    const unitree_go::msg::LowState & state_msg, Eigen::VectorXd & q_pin,
    Eigen::VectorXd & v_pin, std::array<double, 4> & f_pin) const
  {
    q_pin = Eigen::VectorXd::Zero(model_.nq);
    v_pin = Eigen::VectorXd::Zero(model_.nv);
    q_pin(6) = 1.0;

    for (std::size_t index_urdf = 0; index_urdf < urdf_to_sdk_index_.size(); ++index_urdf)
    {
      const std::size_t index_sdk = urdf_to_sdk_index_[index_urdf];
      q_pin(static_cast<Eigen::Index>(7 + index_urdf)) = state_msg.motor_state[index_sdk].q;
      v_pin(static_cast<Eigen::Index>(6 + index_urdf)) = state_msg.motor_state[index_sdk].dq;
    }

    f_pin = {
      static_cast<double>(state_msg.foot_force[1]),
      static_cast<double>(state_msg.foot_force[0]),
      static_cast<double>(state_msg.foot_force[3]),
      static_cast<double>(state_msg.foot_force[2]),
    };
  }

  void initialize_filter(const unitree_go::msg::LowState & state_msg)
  {
    Eigen::VectorXd q;
    Eigen::VectorXd v;
    std::array<double, 4> foot_forces;
    fill_pinocchio_state(state_msg, q, v, foot_forces);

    q(3) = state_msg.imu_state.quaternion[1];
    q(4) = state_msg.imu_state.quaternion[2];
    q(5) = state_msg.imu_state.quaternion[3];
    q(6) = state_msg.imu_state.quaternion[0];
    q.segment<4>(3).normalize();

    pinocchio::forwardKinematics(model_, data_, q, v);
    pinocchio::updateFramePlacements(model_, data_);

    pinocchio::SE3 o_M_base = data_.oMf[base_frame_id_];
    Eigen::Vector3d rpy = pinocchio::rpy::matrixToRpy(o_M_base.rotation());
    rpy(2) = 0.0;
    o_M_base.rotation() = pinocchio::rpy::rpyToMatrix(rpy);

    double z_avg = 0.0;
    for (std::size_t i = 0; i < foot_frame_ids_.size(); ++i)
    {
      z_avg += data_.oMf[foot_frame_ids_[i]].translation()(2);
    }
    z_avg /= static_cast<double>(foot_frame_ids_.size());

    o_M_base.translation().head<2>().setZero();
    o_M_base.translation()(2) -= z_avg - 0.025;

    const pinocchio::SE3 o_M_imu = o_M_base.act(imu_M_base_.inverse());

    inekf::RobotState state = filter_.getState();
    state.setRotation(o_M_imu.rotation());
    state.setPosition(o_M_imu.translation());
    filter_.setState(state);
  }

  FeetKinematics feet_transformations(const unitree_go::msg::LowState & state_msg)
  {
    Eigen::VectorXd q_pin;
    Eigen::VectorXd v_pin;
    std::array<double, 4> f_pin;
    fill_pinocchio_state(state_msg, q_pin, v_pin, f_pin);

    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_);

    FeetKinematics feet;
    const pinocchio::SE3 & o_M_imu = data_.oMf[imu_frame_id_];
    for (std::size_t i = 0; i < foot_frame_ids_.size(); ++i)
    {
      feet.contacts[i] = f_pin[i] >= 18.0;
      const pinocchio::SE3 & o_M_foot = data_.oMf[foot_frame_ids_[i]];
      feet.imu_M_foot[i] = o_M_imu.actInv(o_M_foot);

      Eigen::Matrix<double, 6, Eigen::Dynamic> frame_jacobian(6, model_.nv);
      frame_jacobian.setZero();
      pinocchio::getFrameJacobian(
        model_, data_, foot_frame_ids_[i], pinocchio::LOCAL, frame_jacobian);
      const Eigen::MatrixXd Jc = frame_jacobian.topRows<3>().rightCols(12);
      feet.normed_covariances[i] = Jc * Jc.transpose();
    }
    return feet;
  }

  void publish_state(const inekf::RobotState & filter_state, const Eigen::Vector3d & twist_angular_vel)
  {
    const builtin_interfaces::msg::Time timestamp = get_clock()->now();

    const pinocchio::SE3 o_M_imu(filter_state.getRotation(), filter_state.getPosition());
    const Eigen::Vector3d v_linear_imu_world = filter_state.getX().block<3, 1>(0, 3);
    const Eigen::Vector3d v_linear_imu_local = o_M_imu.inverse().rotation() * v_linear_imu_world;
    const pinocchio::Motion v_imu_local(v_linear_imu_local, twist_angular_vel);

    const pinocchio::SE3 base_pose = o_M_imu.act(imu_M_base_);
    const pinocchio::Motion base_velocity = imu_M_base_.actInv(v_imu_local);

    Eigen::Quaterniond base_quaternion(base_pose.rotation());
    base_quaternion.normalize();

    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = timestamp;
    transform_msg.header.frame_id = odom_frame_;
    transform_msg.child_frame_id = base_frame_;
    transform_msg.transform.translation.x = base_pose.translation()(0);
    transform_msg.transform.translation.y = base_pose.translation()(1);
    transform_msg.transform.translation.z = base_pose.translation()(2);
    transform_msg.transform.rotation.x = base_quaternion.x();
    transform_msg.transform.rotation.y = base_quaternion.y();
    transform_msg.transform.rotation.z = base_quaternion.z();
    transform_msg.transform.rotation.w = base_quaternion.w();
    tf_broadcaster_->sendTransform(transform_msg);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = timestamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_;
    odom_msg.pose.pose.position.x = base_pose.translation()(0);
    odom_msg.pose.pose.position.y = base_pose.translation()(1);
    odom_msg.pose.pose.position.z = base_pose.translation()(2);
    odom_msg.pose.pose.orientation.x = base_quaternion.x();
    odom_msg.pose.pose.orientation.y = base_quaternion.y();
    odom_msg.pose.pose.orientation.z = base_quaternion.z();
    odom_msg.pose.pose.orientation.w = base_quaternion.w();
    odom_msg.twist.twist.linear.x = base_velocity.linear()(0);
    odom_msg.twist.twist.linear.y = base_velocity.linear()(1);
    odom_msg.twist.twist.linear.z = base_velocity.linear()(2);
    odom_msg.twist.twist.angular.x = base_velocity.angular()(0);
    odom_msg.twist.twist.angular.y = base_velocity.angular()(1);
    odom_msg.twist.twist.angular.z = base_velocity.angular()(2);
    odom_publisher_->publish(odom_msg);
  }

  std::string base_frame_;
  std::string odom_frame_;
  double dt_ = 0.002;
  bool wait_for_all_feet_contact_ = true;
  bool pause_ = true;
  double joint_position_noise_ = 0.001;
  double contact_velocity_noise_ = 0.001;

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::SE3 imu_M_base_;
  std::array<std::string, 4> foot_frame_names_;
  std::array<pinocchio::FrameIndex, 4> foot_frame_ids_;
  pinocchio::FrameIndex imu_frame_id_;
  pinocchio::FrameIndex base_frame_id_;

  const std::array<std::size_t, 12> urdf_to_sdk_index_ = {
    3, 4, 5,
    0, 1, 2,
    9, 10, 11,
    6, 7, 8,
  };

  inekf::InEKF filter_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<InekfOdomNode>();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  executor.remove_node(node);

  if (rclcpp::ok())
  {
    rclcpp::shutdown();
  }
  return 0;
}
