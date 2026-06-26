#!/home/unitree/miniconda3/envs/env_deploy/bin/python

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('fetch')
    default_params = os.path.join(package_share, 'config', 'fetch_params.yaml')

    full_state_publisher_launch_file = PathJoinSubstitution(
        [FindPackageShare("fetch"), "launch", "odometry_inekf.launch.py"]
    )

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
        TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([full_state_publisher_launch_file]),
                ),
            ],
        ),
    ])
