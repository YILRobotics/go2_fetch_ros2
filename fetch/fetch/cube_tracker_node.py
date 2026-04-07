#!/usr/bin/env python3
"""Cube tracker node using YOLOE segmentation + Realsense point cloud."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool
import tf2_ros

try:
    import tf2_geometry_msgs  # noqa: F401  # Registers PointStamped conversions.
except ImportError:
    tf2_geometry_msgs = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class DetectionResult:
    position_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    frame_id: str
    distance_m: float


class CubeTrackerNode(Node):
    """Detects cube-like object and publishes filtered planar state."""

    def __init__(self) -> None:
        super().__init__('cube_tracker_node')

        self._declare_parameters()

        self.image_topic = self.get_parameter('image_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.cube_state_topic = self.get_parameter('cube_state_topic').value
        self.cube_visible_topic = self.get_parameter('cube_visible_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        self.model_path = self.get_parameter('model_path').value
        self.target_classes = set(self.get_parameter('target_classes').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.max_mask_samples = int(self.get_parameter('max_mask_samples').value)
        self.min_inlier_points = int(self.get_parameter('min_inlier_points').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.outlier_mad_scale = float(self.get_parameter('outlier_mad_scale').value)
        self.detection_timeout_s = float(self.get_parameter('detection_timeout_s').value)
        self.velocity_window_s = float(self.get_parameter('velocity_window_s').value)
        self.processing_rate_hz = float(self.get_parameter('processing_rate_hz').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.tf_timeout_s = float(self.get_parameter('tf_timeout_s').value)

        self._bridge = CvBridge()
        self._rng = np.random.default_rng(seed=1234)
        self._history: deque[Tuple[float, float, float]] = deque()

        self._latest_pair: Optional[Tuple[Image, PointCloud2]] = None
        self._latest_lock = threading.Lock()
        self._last_processed_stamp: Optional[Tuple[int, int]] = None
        self._last_detection_time: Optional[float] = None
        self._last_visible = False
        self._processing = False

        self._model = self._load_model()

        self._state_pub = self.create_publisher(Odometry, self.cube_state_topic, 10)
        self._visible_pub = self.create_publisher(Bool, self.cube_visible_topic, 10)
        self._debug_pub = self.create_publisher(Image, self.debug_image_topic, 2)

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self._image_sub = Subscriber(self, Image, self.image_topic, qos_profile=sensor_qos)
        self._pointcloud_sub = Subscriber(self, PointCloud2, self.pointcloud_topic, qos_profile=sensor_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._image_sub, self._pointcloud_sub],
            queue_size=30,
            slop=0.08,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._sync_cb)

        self._timer = self.create_timer(1.0 / self.processing_rate_hz, self._on_timer)

        self.get_logger().info(
            f'Cube tracker ready. model={self.model_path} image={self.image_topic} cloud={self.pointcloud_topic}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('pointcloud_topic', '/camera/depth/color/points')
        self.declare_parameter('cube_state_topic', '/go2_fetch/cube_state')
        self.declare_parameter('cube_visible_topic', '/go2_fetch/cube_visible')
        self.declare_parameter('debug_image_topic', '/go2_fetch/cube_debug_image')

        self.declare_parameter('model_path', 'yoloe-26l-seg.pt')
        self.declare_parameter('target_classes', ['cube', 'box', 'black cube', 'bottle'])
        self.declare_parameter('conf_threshold', 0.15)

        self.declare_parameter('max_mask_samples', 2500)
        self.declare_parameter('min_inlier_points', 30)
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('outlier_mad_scale', 3.0)

        self.declare_parameter('processing_rate_hz', 20.0)
        self.declare_parameter('detection_timeout_s', 0.35)
        self.declare_parameter('velocity_window_s', 0.6)

        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('tf_timeout_s', 0.05)

        self.declare_parameter('publish_debug_image', False)

    def _load_model(self):
        if YOLO is None:
            self.get_logger().error('ultralytics is not installed. Cube tracker cannot run YOLO inference.')
            return None

        model = YOLO(self.model_path)
        if self.target_classes:
            model.set_classes(list(self.target_classes))
        return model

    def _sync_cb(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        with self._latest_lock:
            self._latest_pair = (image_msg, cloud_msg)

    def _on_timer(self) -> None:
        if self._processing:
            return

        pair: Optional[Tuple[Image, PointCloud2]] = None
        with self._latest_lock:
            if self._latest_pair is not None:
                pair = self._latest_pair

        now_s = self.get_clock().now().nanoseconds * 1e-9

        if pair is None:
            self._update_visibility_from_timeout(now_s)
            return

        stamp = (pair[0].header.stamp.sec, pair[0].header.stamp.nanosec)
        if stamp == self._last_processed_stamp:
            self._update_visibility_from_timeout(now_s)
            return

        self._processing = True
        try:
            det = self._process_pair(pair[0], pair[1])
            self._last_processed_stamp = stamp
            if det is not None:
                self._last_detection_time = now_s
                self._publish_detection(det, stamp)
            self._update_visibility_from_timeout(now_s)
        except Exception as exc:
            self.get_logger().error(f'Cube tracker callback failed: {exc}')
        finally:
            self._processing = False

    def _process_pair(self, image_msg: Image, cloud_msg: PointCloud2) -> Optional[DetectionResult]:
        if self._model is None:
            return None

        image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        yolo_out = self._model.predict(image, verbose=False, conf=self.conf_threshold)
        if not yolo_out:
            return None

        result = yolo_out[0]
        best_idx = self._pick_best_detection(result)
        if best_idx is None:
            return None

        mask = self._extract_mask(result, best_idx, image.shape[:2])
        if mask is None:
            return None

        centroid = self._extract_filtered_centroid(mask, cloud_msg)
        if centroid is None:
            return None

        x, y, z = centroid
        point = PointStamped()
        point.header = cloud_msg.header
        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        out_frame = cloud_msg.header.frame_id
        if self.target_frame and self.target_frame != cloud_msg.header.frame_id:
            try:
                transformed = self._tf_buffer.transform(
                    point,
                    self.target_frame,
                    timeout=Duration(seconds=self.tf_timeout_s),
                )
                point = transformed
                out_frame = self.target_frame
            except Exception as exc:
                self.get_logger().warn(f'TF transform failed ({cloud_msg.header.frame_id}->{self.target_frame}): {exc}')

        pos_xy = (float(point.point.x), float(point.point.y))
        vel_xy = self._estimate_velocity(pos_xy)

        if self.publish_debug_image:
            self._publish_debug(image, result, best_idx, mask, pos_xy, vel_xy)

        distance_m = float(np.linalg.norm([point.point.x, point.point.y, point.point.z]))
        return DetectionResult(position_xy=pos_xy, velocity_xy=vel_xy, frame_id=out_frame, distance_m=distance_m)

    def _pick_best_detection(self, result) -> Optional[int]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        names = result.names if hasattr(result, 'names') else {}
        best_idx = None
        best_score = -1.0

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            if self.target_classes and label not in self.target_classes:
                continue

            xyxy = boxes.xyxy[i].detach().cpu().numpy()
            area = max(1.0, float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))
            score = conf * area
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            return best_idx

        # Fallback: highest confidence detection when no class match.
        confidences = boxes.conf.detach().cpu().numpy()
        return int(np.argmax(confidences)) if confidences.size > 0 else None

    def _extract_mask(self, result, idx: int, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        h, w = image_shape

        if result.masks is not None and len(result.masks.data) > idx:
            raw_mask = result.masks.data[idx].detach().cpu().numpy()
            if raw_mask.shape != (h, w):
                raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = raw_mask > 0.5
        else:
            boxes = result.boxes.xyxy[idx].detach().cpu().numpy().astype(np.int32)
            x1 = int(np.clip(boxes[0], 0, w - 1))
            y1 = int(np.clip(boxes[1], 0, h - 1))
            x2 = int(np.clip(boxes[2], 0, w - 1))
            y2 = int(np.clip(boxes[3], 0, h - 1))
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2 + 1, x1:x2 + 1] = True

        if int(mask.sum()) < self.min_inlier_points:
            return None
        return mask

    def _extract_filtered_centroid(self, mask: np.ndarray, cloud_msg: PointCloud2) -> Optional[np.ndarray]:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None

        if xs.size > self.max_mask_samples:
            pick = self._rng.choice(xs.size, size=self.max_mask_samples, replace=False)
            xs = xs[pick]
            ys = ys[pick]

        uvs = [(int(u), int(v)) for u, v in zip(xs.tolist(), ys.tolist())]
        points = point_cloud2.read_points_numpy(
            cloud_msg,
            field_names=['x', 'y', 'z'],
            skip_nans=False,
            uvs=uvs,
        )

        if points.size == 0:
            return None

        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.shape[0] < self.min_inlier_points:
            return None

        depth_mask = np.logical_and(points[:, 2] > self.min_depth_m, points[:, 2] < self.max_depth_m)
        points = points[depth_mask]
        if points.shape[0] < self.min_inlier_points:
            return None

        median = np.median(points, axis=0)
        mad = np.median(np.abs(points - median), axis=0)
        mad = np.maximum(mad, 1e-4)
        robust_distance = np.abs(points - median) / mad
        inliers = np.all(robust_distance < self.outlier_mad_scale, axis=1)
        points = points[inliers]

        if points.shape[0] < self.min_inlier_points:
            return None

        return np.mean(points, axis=0)

    def _estimate_velocity(self, pos_xy: Tuple[float, float]) -> Tuple[float, float]:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._history.append((now_s, pos_xy[0], pos_xy[1]))

        while self._history and (now_s - self._history[0][0]) > self.velocity_window_s:
            self._history.popleft()

        if len(self._history) < 2:
            return (0.0, 0.0)

        t = np.array([h[0] for h in self._history], dtype=np.float64)
        x = np.array([h[1] for h in self._history], dtype=np.float64)
        y = np.array([h[2] for h in self._history], dtype=np.float64)
        t = t - t[0]

        if t[-1] < 1e-3:
            return (0.0, 0.0)

        vx = float(np.polyfit(t, x, 1)[0])
        vy = float(np.polyfit(t, y, 1)[0])
        return (vx, vy)

    def _publish_detection(self, det: DetectionResult, stamp: Tuple[int, int]) -> None:
        msg = Odometry()
        msg.header.stamp.sec = stamp[0]
        msg.header.stamp.nanosec = stamp[1]
        msg.header.frame_id = det.frame_id
        msg.child_frame_id = 'cube'

        msg.pose.pose.position.x = det.position_xy[0]
        msg.pose.pose.position.y = det.position_xy[1]
        msg.pose.pose.position.z = 0.0
        msg.pose.covariance[0] = 0.01
        msg.pose.covariance[7] = 0.01

        msg.twist.twist.linear.x = det.velocity_xy[0]
        msg.twist.twist.linear.y = det.velocity_xy[1]
        msg.twist.covariance[0] = 0.04
        msg.twist.covariance[7] = 0.04

        self._state_pub.publish(msg)
        self._publish_visible(True)

    def _publish_debug(self, image, result, idx: int, mask: np.ndarray, pos_xy, vel_xy) -> None:
        draw = result.plot()
        overlay = np.zeros_like(draw)
        overlay[mask] = (0, 255, 0)
        draw = cv2.addWeighted(draw, 1.0, overlay, 0.2, 0.0)
        cv2.putText(
            draw,
            f'xy=({pos_xy[0]:.2f}, {pos_xy[1]:.2f}) vel=({vel_xy[0]:.2f}, {vel_xy[1]:.2f})',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 250, 40),
            2,
            cv2.LINE_AA,
        )
        debug_msg = self._bridge.cv2_to_imgmsg(draw, encoding='bgr8')
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self._debug_pub.publish(debug_msg)

    def _update_visibility_from_timeout(self, now_s: float) -> None:
        visible = False
        if self._last_detection_time is not None:
            visible = (now_s - self._last_detection_time) <= self.detection_timeout_s
        self._publish_visible(visible)
        if not visible:
            self._history.clear()

    def _publish_visible(self, value: bool) -> None:
        if value == self._last_visible:
            return
        self._last_visible = value
        msg = Bool()
        msg.data = value
        self._visible_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubeTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
