#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""Publish a saved image and a colored PointCloud2 on ROS 2 topics."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Tuple
import xml.etree.ElementTree as ET

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class DummyImagePointCloudPublisherNode(Node):
    """Republish saved capture artifacts as ROS 2 test topics."""

    def __init__(self) -> None:
        super().__init__("dummy_image_pointcloud_publisher_node")

        data_dir = Path(get_package_share_directory("fetch")) / "data"
        self.declare_parameter("image_path", str(data_dir / "realsense_color.png"))
        self.declare_parameter("pointcloud_npy_path", str(data_dir / "realsense_points.npy"))
        # Defaults intentionally match the RealSense topics so this node is a drop-in test source.
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("pointcloud_topic", "/camera/depth/color/points")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("max_points", 10000)
        # The full capture is still used to reconstruct depth. Only PointCloud2
        # publication is sampled to avoid serializing several megabytes at 30 Hz.
        self.declare_parameter("published_cloud_max_points", 50000)
        # Approximate D435 color intrinsics for the saved 640x480 captures.
        self.declare_parameter("fx", 615.0)
        self.declare_parameter("fy", 615.0)
        self.declare_parameter("cx", 319.5)
        self.declare_parameter("cy", 239.5)
        self.declare_parameter("depth_scale_m", 0.001)
        self.declare_parameter(
            "robot_description_path",
            "/home/ferdinand/unitree/src/go2_fetch_ros2/go2_description/model/go2/go2.urdf",
        )

        image_path = Path(str(self.get_parameter("image_path").value)).expanduser().resolve()
        cloud_path = Path(str(self.get_parameter("pointcloud_npy_path").value)).expanduser().resolve()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.depth_scale_m = float(self.get_parameter("depth_scale_m").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        max_points = int(self.get_parameter("max_points").value)
        published_cloud_max_points = int(
            self.get_parameter("published_cloud_max_points").value
        )
        if self.fx <= 0.0 or self.fy <= 0.0 or self.depth_scale_m <= 0.0:
            raise ValueError("fx, fy, and depth_scale_m must be positive")

        self._bridge = CvBridge()
        self._cv_image = self._load_image(image_path)
        full_points = self._load_or_create_points(cloud_path, max_points=max_points)
        self._depth_image = self._project_depth(full_points[:, :3])
        self._points = self._sample_published_cloud(
            full_points, published_cloud_max_points
        )
        self._point_fields = self._point_fields_for_points(self._points)
        self._published_once = False

        # Match RealSense and tracker subscriptions exactly: best effort, volatile,
        # and a small queue so old multi-megabyte frames cannot build up.
        self._image_pub = self.create_publisher(
            Image, self.image_topic, qos_profile_sensor_data
        )
        self._depth_pub = self.create_publisher(
            Image, self.depth_topic, qos_profile_sensor_data
        )
        self._camera_info_pub = self.create_publisher(
            CameraInfo, self.camera_info_topic, qos_profile_sensor_data
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, self.pointcloud_topic, qos_profile_sensor_data
        )
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_dummy_transforms()
        self._timer = self.create_timer(1.0 / max(publish_rate_hz, 0.1), self._on_timer)

        self.get_logger().info(
            f"Publishing synchronized dummy camera: image={self.image_topic}, "
            f"depth={self.depth_topic}, camera_info={self.camera_info_topic}, "
            f"cloud={self.pointcloud_topic}, points={len(self._points)}, "
            f"source_points={len(full_points)}, "
            f"depth_pixels={int(np.count_nonzero(self._depth_image))}"
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                self.get_logger().info(f"Loaded image from: {image_path}")
                return image

        self.get_logger().warn(
            f"Image not found/readable at {image_path}. Publishing fallback dummy image."
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            image,
            "Dummy Image",
            (140, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return image

    def _load_or_create_points(self, cloud_path: Path, max_points: int) -> np.ndarray:
        if cloud_path.exists():
            try:
                points = np.load(str(cloud_path))
                if points.ndim == 2 and points.shape[1] >= 3:
                    has_rgb = points.shape[1] >= 4
                    points = points[:, :4 if has_rgb else 3].astype(np.float32, copy=False)
                    points = points[np.isfinite(points[:, :3]).all(axis=1)]
                    points = points[: max(max_points, 1)]
                    if points.shape[0] > 0:
                        self.get_logger().info(
                            f"Loaded point cloud from: {cloud_path} ({points.shape[0]} points)"
                        )
                        return points
            except Exception as exc:
                self.get_logger().warn(f"Failed to load pointcloud NPY at {cloud_path}: {exc}")

        self.get_logger().warn(
            f"Point cloud file not usable at {cloud_path}. Publishing synthetic dummy RGB cloud."
        )
        return self._make_dummy_cloud(max_points=max_points)

    def _make_dummy_cloud(self, max_points: int) -> np.ndarray:
        n = int(np.clip(max_points, 500, 50000))
        side = int(np.sqrt(n))
        xs = np.linspace(-0.5, 0.5, side, dtype=np.float32)
        ys = np.linspace(-0.5, 0.5, side, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        zv = np.full_like(xv, 1.0, dtype=np.float32)
        red = np.clip((xv + 0.5) * 255.0, 0.0, 255.0).astype(np.uint32)
        green = np.clip((yv + 0.5) * 255.0, 0.0, 255.0).astype(np.uint32)
        blue = np.full_like(red, 192, dtype=np.uint32)
        rgb_uint32 = (red << 16) | (green << 8) | blue
        rgb_float = rgb_uint32.view(np.float32)
        return np.column_stack((xv.reshape(-1), yv.reshape(-1), zv.reshape(-1), rgb_float.reshape(-1)))

    def _point_fields_for_points(self, points: np.ndarray) -> list[PointField]:
        if points.shape[1] >= 4:
            return [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

    def _sample_published_cloud(self, points: np.ndarray, limit: int) -> np.ndarray:
        """Uniformly sample the capture while preserving its full spatial extent."""
        if limit <= 0 or points.shape[0] <= limit:
            return np.ascontiguousarray(points, dtype=np.float32)
        indices = np.linspace(0, points.shape[0] - 1, num=limit, dtype=np.int64)
        sampled = np.ascontiguousarray(points[indices], dtype=np.float32)
        self.get_logger().info(
            f"Sampled published cloud from {points.shape[0]} to {sampled.shape[0]} points"
        )
        return sampled

    def _project_depth(self, xyz: np.ndarray) -> np.ndarray:
        """Project an unorganized XYZ cloud into the RGB grid using nearest depth."""
        height, width = self._cv_image.shape[:2]
        depth = np.full(height * width, np.iinfo(np.uint16).max, dtype=np.uint16)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.0)
        points = xyz[valid].astype(np.float64, copy=False)
        if points.size == 0:
            return np.zeros((height, width), dtype=np.uint16)

        u = np.rint(self.fx * points[:, 0] / points[:, 2] + self.cx).astype(np.int64)
        v = np.rint(self.fy * points[:, 1] / points[:, 2] + self.cy).astype(np.int64)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z = u[inside], v[inside], points[inside, 2]
        raw_depth = np.clip(
            np.rint(z / self.depth_scale_m), 1, np.iinfo(np.uint16).max - 1
        ).astype(np.uint16)
        np.minimum.at(depth, v * width + u, raw_depth)
        depth[depth == np.iinfo(np.uint16).max] = 0
        return depth.reshape(height, width)

    @staticmethod
    def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def _make_transform(
        self,
        parent: str,
        child: str,
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = xyz[0]
        transform.transform.translation.y = xyz[1]
        transform.transform.translation.z = xyz[2]
        qx, qy, qz, qw = self._quaternion_from_rpy(*rpy)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        return transform

    def _publish_dummy_transforms(self) -> None:
        """Publish a fixed zero-joint Go2 tree and stationary dummy camera."""
        requested_path = Path(str(self.get_parameter("robot_description_path").value)).expanduser()
        urdf_path = requested_path
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Go2 URDF not found at {requested_path}")

        root = ET.parse(urdf_path).getroot()
        transforms = [self._make_transform("odom", "base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        for joint in root.findall("joint"):
            parent_element = joint.find("parent")
            child_element = joint.find("child")
            if parent_element is None or child_element is None:
                continue
            origin = joint.find("origin")
            xyz_text = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
            rpy_text = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
            xyz = tuple(float(value) for value in xyz_text.split())
            rpy = tuple(float(value) for value in rpy_text.split())
            transforms.append(
                self._make_transform(
                    parent_element.get("link", ""), child_element.get("link", ""), xyz, rpy
                )
            )

        # Camera mount used by this workspace, followed by the standard ROS optical rotation.
        transforms.append(
            self._make_transform("base", "camera_link", (-0.1, 0.01, -0.1), (0.0, 0.5236, 0.0))
        )
        transforms.append(
            self._make_transform(
                "camera_link",
                "camera_color_optical_frame",
                (0.0, 0.0, 0.0),
                (-math.pi / 2.0, 0.0, -math.pi / 2.0),
            )
        )
        self._static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(
            f"Published {len(transforms)} static dummy transforms from {urdf_path}"
        )

    def _build_messages(self) -> Tuple[Image, Image, CameraInfo, PointCloud2]:
        stamp = self.get_clock().now().to_msg()

        if not hasattr(self, "_cached_messages"):
            rgb_image = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2RGB)
            image_msg = self._bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            depth_msg = self._bridge.cv2_to_imgmsg(self._depth_image, encoding="16UC1")
            camera_info = CameraInfo()
            height, width = self._cv_image.shape[:2]
            camera_info.width = width
            camera_info.height = height
            camera_info.distortion_model = "plumb_bob"
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            camera_info.p = [
                self.fx, 0.0, self.cx, 0.0,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            cloud_msg = PointCloud2()
            cloud_msg.header = image_msg.header
            cloud_msg.height = 1
            cloud_msg.width = int(self._points.shape[0])
            cloud_msg.fields = self._point_fields
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = int(self._points.shape[1] * np.dtype(np.float32).itemsize)
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = bool(np.isfinite(self._points[:, :3]).all())
            cloud_msg.data = np.ascontiguousarray(self._points, dtype='<f4').tobytes()
            self._cached_messages = (image_msg, depth_msg, camera_info, cloud_msg)

        image_msg, depth_msg, camera_info, cloud_msg = self._cached_messages
        for message in (image_msg, depth_msg, camera_info, cloud_msg):
            message.header.stamp = stamp
            message.header.frame_id = self.frame_id
        return image_msg, depth_msg, camera_info, cloud_msg

    def _on_timer(self) -> None:
        image_msg, depth_msg, camera_info, cloud_msg = self._build_messages()
        self._image_pub.publish(image_msg)
        self._depth_pub.publish(depth_msg)
        self._camera_info_pub.publish(camera_info)
        self._cloud_pub.publish(cloud_msg)
        if not self._published_once:
            self._published_once = True
            self.get_logger().info(
                "Published first synchronized RGB, depth, CameraInfo, and PointCloud2 set"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DummyImagePointCloudPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
