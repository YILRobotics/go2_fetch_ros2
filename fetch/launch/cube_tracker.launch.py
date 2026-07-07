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
    depth_topic = '/camera/aligned_depth_to_color/image_raw'
    camera_info_topic = '/camera/color/camera_info'
    pointcloud_topic = '/camera/depth/color/points'
    use_dummy_publisher_arg = DeclareLaunchArgument(
        'use_dummy_publisher',
        default_value='false',
        description='Use synchronized saved RGB/depth/cloud data instead of RealSense.',
    )
    use_dummy_publisher = LaunchConfiguration('use_dummy_publisher')
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
            'pointcloud.enable': False,
            'pointcloud__neon_.enable': True,
            'pointcloud__neon_.stream_filter': 2, # 2 is Color texture
            'pointcloud__neon_.stream_index_filter': 0,
            'pointcloud__neon_.ordered_pc': True,
            'pointcloud__neon_.ordered_pc': True,
            'align_depth.enable': True,
            'depth_module.profile': '640x360x30',
            'rgb_camera.profile': '640x360x30',
            # 'depth_module.depth_profile': '1280x720x30',
            # 'rgb_camera.color_profile': '1280x720x30',
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
            'color_qos': 'SENSOR_DATA',
            'depth_qos': 'SENSOR_DATA',
            'color_info_qos': 'SENSOR_DATA',
            'depth_info_qos': 'SENSOR_DATA',
            'pointcloud__neon_.pointcloud_qos': 'SENSOR_DATA',
            'initial_reset': True,
        }],
        condition=UnlessCondition(use_dummy_publisher),
    )

    dummy_realsense_publisher = Node(
        package='fetch',
        executable='dummy_image_pointcloud_publisher_node',
        name='dummy_image_pointcloud_publisher_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'image_path': os.path.join(fetch_share, 'data', 'realsense_color_box.png'),
            'pointcloud_npy_path': os.path.join(fetch_share, 'data', 'realsense_points_box.npy'),
            'image_topic': image_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'pointcloud_topic': pointcloud_topic,
            'frame_id': 'camera_color_optical_frame',
            'publish_rate_hz': 30.0,
            'max_points': 500000,
            'published_cloud_max_points': 50000,
        }],
        condition=IfCondition(use_dummy_publisher),
    )

    cube_tracker_yolo = Node(
        package='fetch',
        executable='cube_tracker_yolo_node',
        name='cube_tracker_yolo_node',
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            params_file,
            {
                'image_topic': image_topic,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'inference_width': 640,
                'inference_height': 360,
            },
        ],
    )

    cube_tracker_pcl = Node(
        package='fetch_pcl_tracker',
        executable='cube_tracker_pcl_node',
        name='cube_tracker_pcl_node',
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file, {'pointcloud_topic': pointcloud_topic}],
    )

    rviz_with_config = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
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


    full_state_publisher_launch_file = PathJoinSubstitution(
        [FindPackageShare("fetch"), "launch", "odometry_inekf.launch.py"]
    )

    base_to_camera_static_tf_launch_file = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="screen" ,
            arguments=["-0.1", "0.01", "-0.1", "0", "0.40", "0", "base", "camera_link"],
            condition=UnlessCondition(use_dummy_publisher),
        )


    return LaunchDescription([
        use_dummy_publisher_arg,
        realsense,
        dummy_realsense_publisher,
        cube_tracker_yolo,
        cube_tracker_pcl,
        # rviz_with_config,
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([full_state_publisher_launch_file]),
        # ),
        base_to_camera_static_tf_launch_file,
    ])
