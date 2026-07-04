#!/home/unitree/miniconda3/envs/env_deploy/bin/python

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
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='hierarchical_lowcmd',
        description='hierarchical_lowcmd or unitree_sport_high_level',
    )
    control_mode = LaunchConfiguration('control_mode')

    cube_tracker_yolo = Node(
        package='fetch',
        executable='cube_tracker_yolo_node',
        name='cube_tracker_yolo_node',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file],
    )

    cube_tracker_pcl = Node(
        package='fetch_pcl_tracker',
        executable='cube_tracker_pcl_node',
        name='cube_tracker_pcl_node',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file],
    )

    high_level_policy = Node(
        package='fetch',
        executable='high_level_policy_node',
        name='high_level_policy_node',
        output='screen',
        parameters=[params_file, {'control_mode': control_mode}],
    )

    low_level_policy = Node(
        package='fetch_low_level',
        executable='low_level_policy_node',
        name='low_level_policy_node',
        output='screen',
        parameters=[params_file, {'control_mode': control_mode}],
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
        control_mode_arg,
        cube_tracker_yolo,
        cube_tracker_pcl,
        low_level_policy,
        high_level_policy,
        state_machine,
    ])
