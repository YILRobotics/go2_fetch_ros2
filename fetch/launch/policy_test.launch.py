#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('fetch')
    default_params = os.path.join(package_share, 'config', 'fetch_params.yaml')

    return LaunchDescription([
        Node(
            package='fetch',
            executable='policy_node',
            name='policy_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                default_params,
            ],
        ),
    ])
