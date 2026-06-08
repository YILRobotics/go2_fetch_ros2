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

    params_file = LaunchConfiguration('params_file')
    policy_stack = LaunchConfiguration('policy_stack')
    initial_mode = LaunchConfiguration('initial_mode')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    require_lowstate = LaunchConfiguration('require_lowstate')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the policy node parameter file.',
        ),
        DeclareLaunchArgument(
            'policy_stack',
            default_value='low_level',
            description='Use low_level for cmd_vel testing or low_and_high_level for cube policy.',
        ),
        DeclareLaunchArgument(
            'initial_mode',
            default_value='standup',
            description='Initial policy node mode. Use standup first, then switch to policy when ready.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Velocity command topic used when policy_stack is low_level.',
        ),
        DeclareLaunchArgument(
            'require_lowstate',
            default_value='true',
            description='If true, do not send motor targets until fresh LowState is received.',
        ),
        Node(
            package='fetch',
            executable='policy_node',
            name='policy_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                params_file,
                {
                    'policy_stack': policy_stack,
                    'initial_mode': initial_mode,
                    'cmd_vel_topic': cmd_vel_topic,
                    'require_lowstate': require_lowstate,
                },
            ],
        ),
    ])
