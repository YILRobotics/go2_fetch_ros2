#!/usr/bin/env python3
"""Cube tracker node using YOLOE segmentation + Realsense point cloud."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
import os
import shutil
import time
import traceback
from typing import Optional, Tuple
import zipfile

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
from visualization_msgs.msg import Marker
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
    position_xyz: Tuple[float, float, float]
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
        self.cube_marker_topic = self.get_parameter('cube_marker_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.mask_image_topic = self.get_parameter('mask_image_topic').value
        self.pre_yolo_image_topic = self.get_parameter('pre_yolo_image_topic').value

        self.model_path = (self.get_parameter('model_path').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.yolo_classes = [str(v) for v in self.get_parameter('yolo_classes').value]
        self.max_mask_samples = int(self.get_parameter('max_mask_samples').value)
        self.min_inlier_points = int(self.get_parameter('min_inlier_points').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.outlier_mad_scale = float(self.get_parameter('outlier_mad_scale').value)
        self.detection_timeout_s = float(self.get_parameter('detection_timeout_s').value)
        self.velocity_window_s = float(self.get_parameter('velocity_window_s').value)
        self.cube_state_low_pass_cutoff_hz = float(
            self.get_parameter('cube_state_low_pass_cutoff_hz').value
        )
        self.cube_dimensions = tuple(float(v) for v in self.get_parameter('cube_dimensions').value)
        if len(self.cube_dimensions) != 3 or any(v <= 0.0 for v in self.cube_dimensions):
            raise ValueError('cube_dimensions must contain three positive values [x, y, z]')
        self.processing_rate_hz = float(self.get_parameter('processing_rate_hz').value)
        self.status_log_rate_hz = float(self.get_parameter('status_log_rate_hz').value)
        self.pipeline_log_rate_hz = float(self.get_parameter('pipeline_log_rate_hz').value)
        self.enable_timing_log = bool(self.get_parameter('enable_timing_log').value)
        self.timing_log_rate_hz = float(self.get_parameter('timing_log_rate_hz').value)
        self.enable_hsv_pre_mask = bool(self.get_parameter('enable_hsv_pre_mask').value)
        self.publish_pre_yolo_image = bool(self.get_parameter('publish_pre_yolo_image').value)
        self.hsv_green_lower = np.array(self.get_parameter('hsv_green_lower').value, dtype=np.uint8)
        self.hsv_green_upper = np.array(self.get_parameter('hsv_green_upper').value, dtype=np.uint8)
        self.hsv_blue_lower = np.array(self.get_parameter('hsv_blue_lower').value, dtype=np.uint8)
        self.hsv_blue_upper = np.array(self.get_parameter('hsv_blue_upper').value, dtype=np.uint8)
        self.target_frame = self.get_parameter('target_frame').value
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.publish_mask_image = bool(self.get_parameter('publish_mask_image').value)
        self.tf_timeout_s = float(self.get_parameter('tf_timeout_s').value)

        self._rng = np.random.default_rng(seed=1234)
        self._bridge = CvBridge()
        self._history: deque[Tuple[float, float, float]] = deque()

        self._latest_pair: Optional[Tuple[Image, PointCloud2]] = None
        self._latest_lock = threading.Lock()
        self._last_processed_stamp: Optional[Tuple[int, int]] = None
        self._last_detection_time: Optional[float] = None
        self._last_visible = False
        self._last_marker_frame_id = ''
        self._filtered_cube_position: Optional[np.ndarray] = None
        self._last_filter_time_s: Optional[float] = None
        self._processing = False
        self._last_status_log_time = 0.0
        self._last_yolo_log_time = 0.0
        self._last_pipeline_log_time = 0.0
        self._last_timing_log_time = 0.0
        self._last_centroid_reject_reason = ''
        self._last_process_wall_time: Optional[float] = None
        self._current_processing_hz = 0.0

        self._model = self._load_model()

        self._state_pub = self.create_publisher(Odometry, self.cube_state_topic, 10)
        self._visible_pub = self.create_publisher(Bool, self.cube_visible_topic, 10)
        self._marker_pub = self.create_publisher(Marker, self.cube_marker_topic, 10)
        self._debug_pub = self.create_publisher(Image, self.debug_image_topic, 2)
        self._mask_pub = self.create_publisher(Image, self.mask_image_topic, 2)
        self._pre_yolo_pub = self.create_publisher(Image, self.pre_yolo_image_topic, 2)

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        # Use the node executor thread for TF callbacks to avoid extra thread shutdown races.
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=False)

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
            slop=0.2,
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
        self.declare_parameter('cube_marker_topic', '/go2_fetch/cube_marker')
        self.declare_parameter('debug_image_topic', '/go2_fetch/cube_debug_image')
        self.declare_parameter('mask_image_topic', '/go2_fetch/cube_mask_image')
        self.declare_parameter('pre_yolo_image_topic', '/go2_fetch/cube_pre_yolo_image')

        self.declare_parameter('model_path', '/home/ferdinand/unitree/go2_fetch_ros2/fetch/models/yoloe-26l-seg.onnx')
        self.declare_parameter('conf_threshold', 0.1)
        self.declare_parameter('yolo_classes', ['box'])

        self.declare_parameter('max_mask_samples', 2500)
        self.declare_parameter('min_inlier_points', 30)
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('outlier_mad_scale', 3.0)

        self.declare_parameter('processing_rate_hz', 20.0)
        self.declare_parameter('status_log_rate_hz', 2.0)
        self.declare_parameter('pipeline_log_rate_hz', 5.0)
        self.declare_parameter('enable_timing_log', True)
        self.declare_parameter('timing_log_rate_hz', 5.0)
        self.declare_parameter('enable_hsv_pre_mask', False)
        self.declare_parameter('publish_pre_yolo_image', True)
        self.declare_parameter('hsv_green_lower', [35, 40, 40])
        self.declare_parameter('hsv_green_upper', [90, 255, 255])
        self.declare_parameter('hsv_blue_lower', [90, 40, 40])
        self.declare_parameter('hsv_blue_upper', [135, 255, 255])
        self.declare_parameter('detection_timeout_s', 0.35)
        self.declare_parameter('velocity_window_s', 0.6)
        self.declare_parameter('cube_state_low_pass_cutoff_hz', 10.0)
        self.declare_parameter('cube_dimensions', [0.16, 0.16, 0.16])

        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('tf_timeout_s', 0.05)

        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('publish_mask_image', True)

    def _load_model(self):
        if YOLO is None:
            self.get_logger().error('ultralytics is not installed. Cube tracker cannot run YOLO inference.')
            return None

        resolved_model_path = os.path.expanduser(os.path.expandvars(str(self.model_path)))
        self.model_path = resolved_model_path
        model = YOLO(resolved_model_path)

        if self.yolo_classes:
            try:
                model.set_classes(self.yolo_classes)
                self.get_logger().info(f'YOLOE classes set to: {self.yolo_classes}')
            except Exception as exc:
                self.get_logger().warn(f'Could not set YOLOE classes: {exc}')

        return model

    def _prepare_text_embedding_asset(self) -> None:
        asset_name = 'mobileclip2_b.ts'
        model_dir = os.path.dirname(self.model_path)
        model_asset = os.path.join(model_dir, asset_name)
        home_asset = os.path.expanduser(os.path.join('~', asset_name))
        cwd_asset = os.path.abspath(asset_name)

        def is_valid_asset(path: str) -> bool:
            return os.path.isfile(path) and zipfile.is_zipfile(path)

        if not is_valid_asset(model_asset) and is_valid_asset(home_asset):
            try:
                os.makedirs(model_dir, exist_ok=True)
                shutil.copy2(home_asset, model_asset)
                self.get_logger().info(f'Copied {asset_name} to model directory: {model_asset}')
            except OSError as exc:
                self.get_logger().warn(f'Failed to copy {asset_name} into model directory: {exc}')

        if os.path.isfile(cwd_asset) and not zipfile.is_zipfile(cwd_asset):
            quarantined_path = f'{cwd_asset}.corrupt'
            try:
                os.replace(cwd_asset, quarantined_path)
                self.get_logger().warn(
                    f'Found corrupted {asset_name} at {cwd_asset}; moved to {quarantined_path}.'
                )
            except OSError as exc:
                self.get_logger().warn(
                    f'Found corrupted {asset_name} at {cwd_asset} but could not move it: {exc}'
                )

        try:
            from ultralytics.utils import SETTINGS

            SETTINGS['weights_dir'] = model_dir
        except Exception as exc:
            self.get_logger().warn(f'Failed to set ultralytics weights_dir={model_dir}: {exc}')

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
        now_wall = time.perf_counter()
        if self._last_process_wall_time is not None:
            dt = now_wall - self._last_process_wall_time
            if dt > 1e-6:
                self._current_processing_hz = 1.0 / dt
        self._last_process_wall_time = now_wall

        self._processing = True
        try:
            det = self._process_pair(pair[0], pair[1])
            self._last_processed_stamp = stamp
            if det is not None:
                self._last_detection_time = now_s
                self._publish_detection(det, stamp)
            self._update_visibility_from_timeout(now_s)
            self._log_status(now_s, det)
        except Exception as exc:
            self.get_logger().error(f'Cube tracker callback failed: {exc}\n{traceback.format_exc()}')
        finally:
            self._processing = False

    def _process_pair(self, image_msg: Image, cloud_msg: PointCloud2) -> Optional[DetectionResult]:
        if self._model is None:
            return None

        pipeline_start = time.perf_counter()
        stage_start = pipeline_start
        stage_durations_ms: list[Tuple[str, float]] = []

        def mark(stage_name: str) -> None:
            nonlocal stage_start
            now = time.perf_counter()
            stage_durations_ms.append((stage_name, (now - stage_start) * 1000.0))
            stage_start = now

        def log_timing(outcome: str) -> None:
            self._maybe_log_timing(outcome, pipeline_start, stage_durations_ms)

        image = self._image_msg_to_bgr8(image_msg)
        mark('image_msg_to_bgr8')
        if image is None:
            log_timing('unsupported_image_encoding')
            return None
        yolo_input = image
        if self.enable_hsv_pre_mask:
            yolo_input, color_mask = self._apply_hsv_green_blue_mask(image)
            mark('hsv_pre_mask')
            keep_ratio = float(np.count_nonzero(color_mask)) / float(color_mask.size)
            self._maybe_log_pipeline(f'pipeline: hsv pre-mask enabled keep_ratio={keep_ratio:.3f}')

        if self.publish_pre_yolo_image:
            self._publish_pre_yolo_image(yolo_input, image_msg)
            mark('publish_pre_yolo_image')

        yolo_out = self._model.predict(yolo_input, verbose=False, conf=self.conf_threshold)
        mark('yolo_predict')
        if not yolo_out:
            log_timing('no_yolo_output')
            return None

        result = yolo_out[0]
        self._maybe_log_yolo_detections(result)
        mark('yolo_postprocess')
        best_idx = self._pick_best_detection(result)
        mark('pick_best_detection')
        if best_idx is None:
            self._maybe_log_pipeline('pipeline: no detection selected after confidence filtering')
            log_timing('no_detection_selected')
            return None

        boxes = result.boxes
        if boxes is not None and len(boxes) > best_idx:
            names = result.names if hasattr(result, 'names') else {}
            cls_id = int(boxes.cls[best_idx].item())
            conf = float(boxes.conf[best_idx].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            self._maybe_log_pipeline(f'pipeline: selected detection label={label} conf={conf:.2f} idx={best_idx}')

        mask = self._extract_mask(result, best_idx, image.shape[:2])
        mark('extract_mask')
        if mask is None:
            self._maybe_log_pipeline('pipeline: mask rejected (too few mask pixels)')
            log_timing('mask_rejected')
            return None

        self._maybe_log_pipeline(f'pipeline: mask pixels={int(mask.sum())}')

        if self.publish_mask_image:
            self._publish_mask(mask, image_msg)
            mark('publish_mask_image')

        if self.publish_debug_image:
            self._publish_debug(image, result, mask)
            mark('publish_debug_image_2d')

        centroid = self._extract_filtered_centroid(mask, cloud_msg)
        mark('extract_filtered_centroid')
        if centroid is None:
            reason = self._last_centroid_reject_reason or 'unknown reason'
            self._maybe_log_pipeline(f'pipeline: distance unavailable, centroid rejected ({reason})')
            log_timing('centroid_rejected')
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
            mark('tf_transform')

        raw_pos_xyz = (float(point.point.x), float(point.point.y), float(point.point.z))
        pos_xyz = self._low_pass_cube_position(raw_pos_xyz)
        pos_xy = pos_xyz[:2]
        vel_xy = self._estimate_velocity(pos_xy)
        mark('estimate_velocity')

        if self.publish_debug_image:
            self._publish_debug(image, result, mask, pos_xy, vel_xy)
            mark('publish_debug_image_3d')

        distance_m = float(np.linalg.norm([point.point.x, point.point.y, point.point.z]))
        mark('finalize')

        self.get_logger().info(
            f'detected: pos=({pos_xy[0]:.2f}, {pos_xy[1]:.2f}) '
            f'vel=({vel_xy[0]:.2f}, {vel_xy[1]:.2f}) '
            f'dist={distance_m:.2f}m frame={out_frame}'
        )
        log_timing('detected')
        return DetectionResult(
            position_xy=pos_xy,
            position_xyz=pos_xyz,
            velocity_xy=vel_xy,
            frame_id=out_frame,
            distance_m=distance_m,
        )

    def _maybe_log_yolo_detections(self, result) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self.status_log_rate_hz <= 0.0:
            return
        period = 1.0 / self.status_log_rate_hz
        if (now_s - self._last_yolo_log_time) < period:
            return
        self._last_yolo_log_time = now_s

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.get_logger().info('yolo: no detections')
            return

        names = result.names if hasattr(result, 'names') else {}
        labels = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            labels.append(f'{label}:{conf:.2f}')

        self.get_logger().info(f'yolo: {len(labels)} detections -> {", ".join(labels[:6])}')

    def _maybe_log_pipeline(self, message: str) -> None:
        if self.pipeline_log_rate_hz <= 0.0:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        period = 1.0 / self.pipeline_log_rate_hz
        if (now_s - self._last_pipeline_log_time) < period:
            return
        self._last_pipeline_log_time = now_s
        self.get_logger().info(message)

    def _maybe_log_timing(
        self,
        outcome: str,
        pipeline_start_s: float,
        stage_durations_ms: list[Tuple[str, float]],
    ) -> None:
        if not self.enable_timing_log or self.timing_log_rate_hz <= 0.0:
            return

        now_s = self.get_clock().now().nanoseconds * 1e-9
        period = 1.0 / self.timing_log_rate_hz
        if (now_s - self._last_timing_log_time) < period:
            return
        self._last_timing_log_time = now_s

        total_ms = (time.perf_counter() - pipeline_start_s) * 1000.0
        stages = ', '.join(f'{name}={dt_ms:.1f}ms' for name, dt_ms in stage_durations_ms)
        if stages:
            self.get_logger().info(f'timing: outcome={outcome} total={total_ms:.1f}ms stages=[{stages}]')
        else:
            self.get_logger().info(f'timing: outcome={outcome} total={total_ms:.1f}ms')
        self.get_logger().info(
            f'current_hz: {self._current_processing_hz:.2f} (target={self.processing_rate_hz:.2f})'
        )

    def _publish_mask(self, mask: np.ndarray, image_msg: Image) -> None:
        mask_u8 = (mask.astype(np.uint8) * 255)
        self._publish_numpy_image(mask_u8, encoding='mono8', header=image_msg.header, publisher=self._mask_pub)

    def _publish_pre_yolo_image(self, image: np.ndarray, image_msg: Image) -> None:
        self._publish_numpy_image(image, encoding='bgr8', header=image_msg.header, publisher=self._pre_yolo_pub)

    def _image_msg_to_bgr8(self, image_msg: Image) -> Optional[np.ndarray]:
        encoding = str(image_msg.encoding).lower()
        if encoding not in ('bgr8', 'rgb8'):
            self.get_logger().warn(f'Unsupported image encoding: {image_msg.encoding}')
            return None

        height = int(image_msg.height)
        width = int(image_msg.width)
        if height <= 0 or width <= 0:
            self.get_logger().warn(
                f'Invalid image shape from message: width={image_msg.width}, height={image_msg.height}'
            )
            return None

        expected_row_bytes = width * 3
        step = int(image_msg.step)
        if step < expected_row_bytes:
            self.get_logger().warn(f'Invalid image step {step} for width={width} encoding={image_msg.encoding}')
            return None

        buffer = np.frombuffer(image_msg.data, dtype=np.uint8)
        expected_total_bytes = step * height
        if buffer.size < expected_total_bytes:
            self.get_logger().warn(
                f'Image payload too small: got={buffer.size} expected_at_least={expected_total_bytes}'
            )
            return None

        array = buffer[:expected_total_bytes].reshape((height, step))
        image = array[:, :expected_row_bytes].reshape((height, width, 3))
        if encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    def _publish_numpy_image(self, image: np.ndarray, encoding: str, header, publisher) -> None:
        if encoding == 'mono8':
            if image.ndim != 2:
                raise ValueError(f'mono8 expects 2D image, got shape={image.shape}')
            out = np.ascontiguousarray(image, dtype=np.uint8)
        elif encoding == 'bgr8':
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'bgr8 expects HxWx3 image, got shape={image.shape}')
            out = np.ascontiguousarray(image, dtype=np.uint8)
        else:
            raise ValueError(f'Unsupported publish encoding: {encoding}')

        msg = self._bridge.cv2_to_imgmsg(out, encoding=encoding)
        msg.header = header
        publisher.publish(msg)

    def _apply_hsv_green_blue_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.hsv_green_lower, self.hsv_green_upper)
        blue_mask = cv2.inRange(hsv, self.hsv_blue_lower, self.hsv_blue_upper)
        keep_mask = cv2.bitwise_or(green_mask, blue_mask)
        filtered = cv2.bitwise_and(image, image, mask=keep_mask)
        return filtered, keep_mask

    def _pick_best_detection(self, result) -> Optional[int]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = None
        best_score = -1.0

        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())

            xyxy = boxes.xyxy[i].detach().cpu().numpy()
            area = max(1.0, float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))
            score = conf * area
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            return best_idx

        # Fallback: highest confidence detection.
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
            self._last_centroid_reject_reason = 'mask has zero valid pixels'
            return None

        if xs.size > self.max_mask_samples:
            pick = self._rng.choice(xs.size, size=self.max_mask_samples, replace=False)
            xs = xs[pick]
            ys = ys[pick]

        width = int(cloud_msg.width)
        height = int(cloud_msg.height)
        if width <= 0 or height <= 0:
            self._last_centroid_reject_reason = f'invalid cloud dimensions width={width} height={height}'
            return None

        mask_height, mask_width = mask.shape
        if height == 1 and mask_height > 1:
            self._last_centroid_reject_reason = (
                f'point cloud is unordered ({width}x{height}); enable pointcloud.ordered_pc '
                'to map image pixels to 3D points'
            )
            return None

        # RealSense depth decimation changes the organized cloud resolution. Map
        # mask pixel centers into that lower-resolution grid before reading XYZ.
        if width != mask_width or height != mask_height:
            xs = np.floor((xs.astype(np.float64) + 0.5) * width / mask_width).astype(np.int64)
            ys = np.floor((ys.astype(np.float64) + 0.5) * height / mask_height).astype(np.int64)

        valid = np.logical_and.reduce((
            xs >= 0,
            ys >= 0,
            xs < width,
            ys < height,
        ))
        xs = xs[valid]
        ys = ys[valid]
        if xs.size == 0:
            self._last_centroid_reject_reason = 'all sampled mask pixels fell outside cloud bounds'
            return None

        # ROS 2 Humble read_points_numpy() indexes by flattened point indices.
        flat_indices = ys.astype(np.int64) * width + xs.astype(np.int64)
        flat_indices = np.unique(flat_indices)
        points = point_cloud2.read_points_numpy(
            cloud_msg,
            field_names=['x', 'y', 'z'],
            skip_nans=False,
            uvs=flat_indices,
        )

        points = np.asarray(points)
        if points.size == 0:
            self._last_centroid_reject_reason = 'read_points_numpy returned zero points'
            return None
        if points.ndim == 1:
            if points.shape[0] != 3:
                self._last_centroid_reject_reason = f'unexpected 1D points shape {points.shape}'
                return None
            points = points.reshape(1, 3)
        elif points.ndim != 2 or points.shape[1] != 3:
            points = points.reshape((-1, 3))

        if points.size == 0:
            self._last_centroid_reject_reason = 'points empty after reshape'
            return None

        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.shape[0] < self.min_inlier_points:
            self._last_centroid_reject_reason = (
                f'too few finite points: {points.shape[0]} < min_inlier_points({self.min_inlier_points})'
            )
            return None

        depth_mask = np.logical_and(points[:, 2] > self.min_depth_m, points[:, 2] < self.max_depth_m)
        points = points[depth_mask]
        if points.shape[0] < self.min_inlier_points:
            self._last_centroid_reject_reason = (
                f'too few depth-in-range points: {points.shape[0]} < min_inlier_points({self.min_inlier_points}) '
                f'for z in ({self.min_depth_m}, {self.max_depth_m})'
            )
            return None

        median = np.median(points, axis=0)
        mad = np.median(np.abs(points - median), axis=0)
        mad = np.maximum(mad, 1e-4)
        robust_distance = np.abs(points - median) / mad
        inliers = np.all(robust_distance < self.outlier_mad_scale, axis=1)
        points = points[inliers]

        if points.shape[0] < self.min_inlier_points:
            self._last_centroid_reject_reason = (
                f'too few MAD inliers: {points.shape[0]} < min_inlier_points({self.min_inlier_points}) '
                f'with scale={self.outlier_mad_scale}'
            )
            return None

        self._last_centroid_reject_reason = ''

        return np.mean(points, axis=0)

    def _low_pass_cube_position(
        self,
        position_xyz: Tuple[float, float, float],
        now_s: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        sample = np.asarray(position_xyz, dtype=np.float64)
        if now_s is None:
            now_s = self.get_clock().now().nanoseconds * 1e-9

        if (
            self.cube_state_low_pass_cutoff_hz <= 0.0
            or self._filtered_cube_position is None
            or self._last_filter_time_s is None
        ):
            filtered = sample
        else:
            dt = max(0.0, now_s - self._last_filter_time_s)
            alpha = 1.0 - np.exp(-2.0 * np.pi * self.cube_state_low_pass_cutoff_hz * dt)
            filtered = self._filtered_cube_position + alpha * (sample - self._filtered_cube_position)

        self._filtered_cube_position = filtered
        self._last_filter_time_s = now_s
        return (float(filtered[0]), float(filtered[1]), float(filtered[2]))

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
        self._publish_cube_marker(det, stamp)
        self._publish_visible(True)

    def _publish_cube_marker(self, det: DetectionResult, stamp: Tuple[int, int]) -> None:
        marker = Marker()
        marker.header.stamp.sec = stamp[0]
        marker.header.stamp.nanosec = stamp[1]
        marker.header.frame_id = det.frame_id
        marker.ns = 'cube_state'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = det.position_xyz[0]
        marker.pose.position.y = det.position_xyz[1]
        marker.pose.position.z = det.position_xyz[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.cube_dimensions[0]
        marker.scale.y = self.cube_dimensions[1]
        marker.scale.z = self.cube_dimensions[2]
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        self._marker_pub.publish(marker)
        self._last_marker_frame_id = det.frame_id

    def _delete_cube_marker(self) -> None:
        if not self._last_marker_frame_id:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._last_marker_frame_id
        marker.ns = 'cube_state'
        marker.id = 0
        marker.action = Marker.DELETE
        self._marker_pub.publish(marker)
        self._last_marker_frame_id = ''

    def _publish_debug(
        self,
        image,
        result,
        mask: np.ndarray,
        pos_xy: Optional[Tuple[float, float]] = None,
        vel_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        draw = result.plot()
        overlay = np.zeros_like(draw)
        overlay[mask] = (0, 255, 0)
        draw = cv2.addWeighted(draw, 1.0, overlay, 0.2, 0.0)
        if pos_xy is not None and vel_xy is not None:
            line = f'xy=({pos_xy[0]:.2f}, {pos_xy[1]:.2f}) vel=({vel_xy[0]:.2f}, {vel_xy[1]:.2f})'
        else:
            line = '2D detection only (no valid depth centroid)'
        cv2.putText(
            draw,
            line,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 250, 40),
            2,
            cv2.LINE_AA,
        )
        debug_msg = Image()
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self._publish_numpy_image(draw, encoding='bgr8', header=debug_msg.header, publisher=self._debug_pub)

    def _update_visibility_from_timeout(self, now_s: float) -> None:
        visible = False
        if self._last_detection_time is not None:
            visible = (now_s - self._last_detection_time) <= self.detection_timeout_s
        was_visible = self._last_visible
        self._publish_visible(visible)
        if was_visible and not visible:
            self._delete_cube_marker()
        if not visible:
            self._history.clear()
            self._filtered_cube_position = None
            self._last_filter_time_s = None

    def _publish_visible(self, value: bool) -> None:
        self._last_visible = value
        msg = Bool()
        msg.data = value
        self._visible_pub.publish(msg)

    def _log_status(self, now_s: float, det: Optional[DetectionResult]) -> None:
        if self.status_log_rate_hz <= 0.0:
            return

        period = 1.0 / self.status_log_rate_hz
        if (now_s - self._last_status_log_time) < period:
            return
        self._last_status_log_time = now_s

        if det is None:
            self.get_logger().info('cube status: nothing found')
            return

        self.get_logger().info(
            f'cube status: pos=({det.position_xy[0]:.3f}, {det.position_xy[1]:.3f}) '
            f'vel=({det.velocity_xy[0]:.3f}, {det.velocity_xy[1]:.3f}) '
            f'dist={det.distance_m:.3f} frame={det.frame_id}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubeTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            # launch can already shut down the context during SIGINT; avoid double-shutdown RCLError.
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
