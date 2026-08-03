from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Shared arguments
        DeclareLaunchArgument('model_path', default_value='/ws/best.pt'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('target_frame', default_value='b2/base_link'),

        # Configurable topic names (pixel-aligned RGB-D camera) — defaults
        # match the ugv_node defaults (rosbag-replay topic set)
        DeclareLaunchArgument('rgb_image_topic',
                              default_value='/camera_front/raw_image'),
        DeclareLaunchArgument('depth_image_topic',
                              default_value='/camera_front/realsense_front/depth/image_rect_raw'),
        DeclareLaunchArgument('camera_info_topic',
                              default_value='/camera_front/camera_info'),
        DeclareLaunchArgument('use_dummy_detections', default_value='false'),

        # GeoJSON bridge arguments
        DeclareLaunchArgument('publish_to_api', default_value='false'),
        DeclareLaunchArgument('save_to_disk', default_value='true'),
        DeclareLaunchArgument('publish_period_s', default_value='5.0'),
        DeclareLaunchArgument('output_dir', default_value='/ws/samples'),

        # UGV Perception Node
        Node(
            package='triffid_ugv_perception',
            executable='ugv_node',
            name='ugv_perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'target_frame': LaunchConfiguration('target_frame'),
                'rgb_image_topic': LaunchConfiguration('rgb_image_topic'),
                'depth_image_topic': LaunchConfiguration('depth_image_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'use_dummy_detections': LaunchConfiguration('use_dummy_detections'),
            }],
        ),

        # GeoJSON Bridge Node
        Node(
            package='triffid_ugv_perception',
            executable='geojson_bridge',
            name='geojson_bridge',
            output='screen',
            parameters=[{
                'publish_to_api': LaunchConfiguration('publish_to_api'),
                'save_to_disk': LaunchConfiguration('save_to_disk'),
                'publish_period_s': LaunchConfiguration('publish_period_s'),
                'output_dir': LaunchConfiguration('output_dir'),
            }],
        ),
    ])
