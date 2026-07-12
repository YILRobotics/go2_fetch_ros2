#!/home/unitree/miniconda3/envs/env_deploy/bin/python

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('fetch')
    default_params = os.path.join(package_share, 'config', 'fetch_params.yaml')
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='hierarchical_lowcmd',
        description='hierarchical_lowcmd or unitree_sport_high_level',
    )
    control_mode = LaunchConfiguration('control_mode')

    full_state_publisher_launch_file = PathJoinSubstitution(
        [FindPackageShare("fetch"), "launch", "odometry_inekf.launch.py"]
    )

    low_level_policy_node = Node(
        package='fetch_low_level',
        executable='low_level_policy_node',
        name='low_level_policy_node',
        output='screen',
        emulate_tty=True,
        parameters=[default_params, {'control_mode': control_mode}],
    )

    high_level_policy_node = Node(
        package='fetch_low_level',
        executable='high_level_policy_node_cpp',
        name='high_level_policy_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            default_params,
            {'control_mode': control_mode},
        ],
    )

    return LaunchDescription([
        control_mode_arg,
        low_level_policy_node,
        high_level_policy_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=low_level_policy_node,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(reason='low_level_policy_node exited')
                    ),
                ],
            )
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
