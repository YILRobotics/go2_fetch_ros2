import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue    


def generate_launch_description():


realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "enable_depth": "true" if enable_pointcloud else "false",
            "enable_color": "true",
            "depth_module.depth_profile": "640x480x15",
            "rgb_camera.color_profile": "640x480x15",
            "align_depth.enable": "false",
            "pointcloud.enable": "true" if enable_pointcloud else "false",
            "enable_infra": "false",
            "enable_infra1": "false",
            "enable_infra2": "false"
        }.items(),
        condition=use_realsense_condition,
    )




    downsample_node = Node(
        package="teleoperation",
        executable="rgb_pointcloud_downsampler_node",
        name="rgb_pointcloud_downsampler",
        output="screen",
        parameters=[
            teleop_config,
        ],
        condition=IfCondition("true" if enable_pointcloud else "false"),
    )


    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=[
            "-d",
            os.path.join(rviz_folder, "visionpro.rviz"),
        ],
    )


rgb_pointcloud_downsampler:
  ros__parameters:
    input_topic: "/camera/camera/depth/color/points"
    output_topic: "/points_downsampled"
    target_frame: "camera_lens"
    publish_rate_hz: 5.0 # (max 30 Hz or 15 Hz depending on settings in launch file)
    downsample_factor: 60



    nodes = [
        static_transform_camera_lens_realsense,
        realsense_launch,
        realsense_description,
        downsample_node,
        # dummy_pointcloud_publisher_node, 
        
        vp_streamer_node,


        robot_state_publisher_node,
        # listen_real_node,  # disabled: teleop_control now owns the serial port and publishes /joint_states

        static_transform_map_mycobot_base,
        vp_transform_publisher_node,
        static_transform_map_vp_base_origin,

        teleop_control_cpp_node,
        inverse_kinematics_node,
        
        rviz2_node,
                        
        joint_state_to_mycobot_node,
    ]

    return LaunchDescription(nodes)