#!/usr/bin/env python3

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    fetch_share = get_package_share_directory('fetch')
    default_params = os.path.join(fetch_share, 'config', 'fetch_params.yaml')

    try:
        default_rviz_config = os.path.join(
            get_package_share_directory('fetch'),
            'rviz',
            'realsense.rviz',
        )
    except PackageNotFoundError:
        default_rviz_config = ''

    params_file = default_params
    image_topic = '/camera/color/image_raw'
    pointcloud_topic = '/camera/depth/color/points'
    use_rviz = 'true'
    rviz_config = default_rviz_config
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
            'pointcloud.enable': True,
            'pointcloud.stream_filter': 2,
            'pointcloud.stream_index_filter': 0,
            'align_depth.enable': True,
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
        }],
    )

    cube_tracker = Node(
        package='fetch',
        executable='cube_tracker_node',
        name='cube_tracker_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            params_file,
            {
                'image_topic': image_topic,
                'pointcloud_topic': pointcloud_topic,
            },
        ],
    )

    rviz_with_config = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(
            PythonExpression([
                "'",
                use_rviz,
                "'",
                " == 'true' and '",
                rviz_config,
                "' != ''",
            ])
        ),
    )

    return LaunchDescription([
        realsense,
        cube_tracker,
        rviz_with_config,
    ])
