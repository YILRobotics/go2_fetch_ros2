#!/home/unitree/miniconda3/envs/env_deploy/bin/python

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:

    image_topic = '/camera/color/image_raw'
    pointcloud_topic = '/camera/depth/color/points'
    use_dummy_publisher = 'false'
    use_rviz = 'true'
    camera_namespace = ''
    camera_name = 'camera'

    realsense = Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace=camera_namespace,
            name=camera_name,
            output='screen',
            parameters=[{
                'enable_color': True,
                'enable_depth': True,
                'pointcloud__neon_.enable': True,
                'pointcloud.enable': True,
                'pointcloud.stream_filter': 0,
                'pointcloud.stream_index_filter': 0,
                'pointcloud.ordered_pc': True,
                'pointcloud__neon_.ordered_pc': True,
                'align_depth.enable': True,
                'depth_module.profile': '640x480x30',
                'rgb_camera.profile': '640x480x30',
                # Fixed RealSense post-processing filters for lighter/cleaner point clouds.
                'decimation_filter.enable': True,
                'decimation_filter.filter_magnitude': 4,
                'spatial_filter.enable': True,
                'temporal_filter.enable': True,
                'enable_sync': True,
                'enable_gyro': False,
                'enable_accel': False,
                'enable_motion': False,
                'enable_infra': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'initial_reset': True,
                'pointcloud__neon_.stream_filter': 0,
                # 'pointcloud__neon_.stream_index_filter': 0,
                # 'pointcloud__neon_.allow_no_texture_points': True,
            }],
            condition=UnlessCondition(use_dummy_publisher),
        )

    return LaunchDescription([
        realsense
    ])
