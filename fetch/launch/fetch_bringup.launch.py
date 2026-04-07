#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('fetch')
    default_params = os.path.join(package_share, 'config', 'fetch_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the YAML parameter file for all nodes.',
    )

    params_file = LaunchConfiguration('params_file')

    cube_tracker = Node(
        package='fetch',
        executable='cube_tracker_node',
        name='cube_tracker_node',
        output='screen',
        parameters=[params_file],
    )

    policy = Node(
        package='fetch',
        executable='policy_node',
        name='policy_node',
        output='screen',
        parameters=[params_file],
    )

    state_machine = Node(
        package='fetch',
        executable='state_machine_node',
        name='state_machine_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        params_file_arg,
        cube_tracker,
        policy,
        state_machine,
    ])
