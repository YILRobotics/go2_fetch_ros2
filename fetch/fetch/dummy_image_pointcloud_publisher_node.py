#!/usr/bin/env python3
"""Publish a saved image and a colored PointCloud2 on ROS 2 topics."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class DummyImagePointCloudPublisherNode(Node):
    """Republish saved capture artifacts as ROS 2 test topics."""

    def __init__(self) -> None:
        super().__init__("dummy_image_pointcloud_publisher_node")

        self.declare_parameter("image_path", "/home/ferdinand/unitree/go2_fetch_ros2/data/realsense_color.png")
        self.declare_parameter("pointcloud_npy_path", "/home/ferdinand/unitree/go2_fetch_ros2/data/realsense_points.npy")
        self.declare_parameter("image_topic", "/dummy/camera/color/image_raw")
        self.declare_parameter("pointcloud_topic", "/dummy/camera/depth/color/points")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("max_points", 500000)

        image_path = Path(str(self.get_parameter("image_path").value)).expanduser().resolve()
        cloud_path = Path(str(self.get_parameter("pointcloud_npy_path").value)).expanduser().resolve()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        max_points = int(self.get_parameter("max_points").value)

        self._bridge = CvBridge()
        self._cv_image = self._load_image(image_path)
        self._points = self._load_or_create_points(cloud_path, max_points=max_points)
        self._point_fields = self._point_fields_for_points(self._points)
        self._points_list = [tuple(point) for point in self._points]

        self._image_pub = self.create_publisher(Image, self.image_topic, 10)
        self._cloud_pub = self.create_publisher(PointCloud2, self.pointcloud_topic, 10)
        self._timer = self.create_timer(1.0 / max(publish_rate_hz, 0.1), self._on_timer)

        self.get_logger().info(
            f"Publishing RGB image/cloud: image_topic={self.image_topic}, "
            f"pointcloud_topic={self.pointcloud_topic}, points={len(self._points_list)}"
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

    def _build_messages(self) -> Tuple[Image, PointCloud2]:
        stamp = self.get_clock().now().to_msg()

        rgb_image = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2RGB)
        image_msg = self._bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id

        cloud_header = image_msg.header
        cloud_msg = point_cloud2.create_cloud(cloud_header, self._point_fields, self._points_list)
        return image_msg, cloud_msg

    def _on_timer(self) -> None:
        image_msg, cloud_msg = self._build_messages()
        self._image_pub.publish(image_msg)
        self._cloud_pub.publish(cloud_msg)


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
