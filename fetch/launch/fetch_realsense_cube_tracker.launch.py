#!/usr/bin/env python3

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the tracker parameter file.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='Image topic consumed by cube_tracker_node.',
    )
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/camera/depth/color/points',
        description='PointCloud2 topic consumed by cube_tracker_node.',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz.',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='RViz config file path. Leave empty to use default RViz.',
    )
    camera_namespace_arg = DeclareLaunchArgument(
        'camera_namespace',
        default_value='',
        description='RealSense camera namespace.',
    )
    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='camera',
        description='RealSense camera node name.',
    )
    enable_color_arg = DeclareLaunchArgument(
        'enable_color',
        default_value='true',
        description='Enable RealSense color stream.',
    )
    enable_depth_arg = DeclareLaunchArgument(
        'enable_depth',
        default_value='true',
        description='Enable RealSense depth stream.',
    )
    enable_pointcloud_arg = DeclareLaunchArgument(
        'enable_pointcloud',
        default_value='true',
        description='Enable RealSense point cloud generation.',
    )
    pointcloud_stream_filter_arg = DeclareLaunchArgument(
        'pointcloud_stream_filter',
        default_value='2',
        description='Point cloud texture source: depth(1), color(2), infrared(3).',
    )
    pointcloud_stream_index_filter_arg = DeclareLaunchArgument(
        'pointcloud_stream_index_filter',
        default_value='0',
        description='Point cloud texture stream index.',
    )
    align_depth_arg = DeclareLaunchArgument(
        'align_depth',
        default_value='true',
        description='Align depth to color stream in RealSense wrapper.',
    )
    enable_sync_arg = DeclareLaunchArgument(
        'enable_sync',
        default_value='true',
        description='Enable RealSense stream synchronization.',
    )
    enable_gyro_arg = DeclareLaunchArgument(
        'enable_gyro',
        default_value='false',
        description='Enable RealSense gyro stream.',
    )
    enable_accel_arg = DeclareLaunchArgument(
        'enable_accel',
        default_value='false',
        description='Enable RealSense accel stream.',
    )
    enable_motion_arg = DeclareLaunchArgument(
        'enable_motion',
        default_value='false',
        description='Enable RealSense DDS motion stream.',
    )
    enable_infra_arg = DeclareLaunchArgument(
        'enable_infra',
        default_value='false',
        description='Enable RealSense infra stream.',
    )
    enable_infra1_arg = DeclareLaunchArgument(
        'enable_infra1',
        default_value='false',
        description='Enable RealSense infra1 stream.',
    )
    enable_infra2_arg = DeclareLaunchArgument(
        'enable_infra2',
        default_value='false',
        description='Enable RealSense infra2 stream.',
    )

    params_file = LaunchConfiguration('params_file')
    image_topic = LaunchConfiguration('image_topic')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    camera_namespace = LaunchConfiguration('camera_namespace')
    camera_name = LaunchConfiguration('camera_name')
    enable_color = LaunchConfiguration('enable_color')
    enable_depth = LaunchConfiguration('enable_depth')
    enable_pointcloud = LaunchConfiguration('enable_pointcloud')
    pointcloud_stream_filter = LaunchConfiguration('pointcloud_stream_filter')
    pointcloud_stream_index_filter = LaunchConfiguration('pointcloud_stream_index_filter')
    align_depth = LaunchConfiguration('align_depth')
    enable_sync = LaunchConfiguration('enable_sync')
    enable_gyro = LaunchConfiguration('enable_gyro')
    enable_accel = LaunchConfiguration('enable_accel')
    enable_motion = LaunchConfiguration('enable_motion')
    enable_infra = LaunchConfiguration('enable_infra')
    enable_infra1 = LaunchConfiguration('enable_infra1')
    enable_infra2 = LaunchConfiguration('enable_infra2')

    realsense = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace=camera_namespace,
        name=camera_name,
        output='screen',
        parameters=[{
            'enable_color': ParameterValue(enable_color, value_type=bool),
            'enable_depth': ParameterValue(enable_depth, value_type=bool),
            'pointcloud.enable': ParameterValue(enable_pointcloud, value_type=bool),
            'pointcloud.stream_filter': ParameterValue(pointcloud_stream_filter, value_type=int),
            'pointcloud.stream_index_filter': ParameterValue(pointcloud_stream_index_filter, value_type=int),
            'align_depth.enable': ParameterValue(align_depth, value_type=bool),
            'enable_sync': ParameterValue(enable_sync, value_type=bool),
            'enable_gyro': ParameterValue(enable_gyro, value_type=bool),
            'enable_accel': ParameterValue(enable_accel, value_type=bool),
            'enable_motion': ParameterValue(enable_motion, value_type=bool),
            'enable_infra': ParameterValue(enable_infra, value_type=bool),
            'enable_infra1': ParameterValue(enable_infra1, value_type=bool),
            'enable_infra2': ParameterValue(enable_infra2, value_type=bool),
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
        params_file_arg,
        image_topic_arg,
        pointcloud_topic_arg,
        use_rviz_arg,
        rviz_config_arg,
        camera_namespace_arg,
        camera_name_arg,
        enable_color_arg,
        enable_depth_arg,
        enable_pointcloud_arg,
        pointcloud_stream_filter_arg,
        pointcloud_stream_index_filter_arg,
        align_depth_arg,
        enable_sync_arg,
        enable_gyro_arg,
        enable_accel_arg,
        enable_motion_arg,
        enable_infra_arg,
        enable_infra1_arg,
        enable_infra2_arg,
        realsense,
        cube_tracker,
        rviz_with_config,
    ])
