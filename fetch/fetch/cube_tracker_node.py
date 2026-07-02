#!/home/unitree/miniconda3/envs/env_deploy/bin/python

"""Cube tracker using reduced-resolution YOLO segmentation and aligned depth."""

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
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
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


@dataclass
class WorkerResult:
    detection: Optional[DetectionResult]
    stamp: Tuple[int, int]
    error: Optional[str] = None


class CubeTrackerNode(Node):
    """Detects cube-like object and publishes filtered planar state."""

    def __init__(self) -> None:
        super().__init__('cube_tracker_node')

        self._declare_parameters()

        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.cube_state_topic = self.get_parameter('cube_state_topic').value
        self.cube_visible_topic = self.get_parameter('cube_visible_topic').value
        self.cube_marker_topic = self.get_parameter('cube_marker_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.mask_image_topic = self.get_parameter('mask_image_topic').value
        self.pre_yolo_image_topic = self.get_parameter('pre_yolo_image_topic').value

        self.model_path = (self.get_parameter('model_path').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.yolo_classes = [str(v) for v in self.get_parameter('yolo_classes').value]
        self.inference_width = int(self.get_parameter('inference_width').value)
        self.inference_height = int(self.get_parameter('inference_height').value)
        if self.inference_width <= 0 or self.inference_height <= 0:
            raise ValueError('inference_width and inference_height must be positive')
        self.depth_scale_m = float(self.get_parameter('depth_scale_m').value)
        if self.depth_scale_m <= 0.0:
            raise ValueError('depth_scale_m must be positive')
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
        if self.processing_rate_hz <= 0.0:
            raise ValueError('processing_rate_hz must be positive')
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
        self.cube_position_offset_xyz = tuple(float(v) for v in self.get_parameter('cube_position_offset_xyz').value)
        self.tf_timeout_s = float(self.get_parameter('tf_timeout_s').value)
        self.camera_info_max_age_s = float(self.get_parameter('camera_info_max_age_s').value)
        self.camera_info_stamp_tolerance_s = float(
            self.get_parameter('camera_info_stamp_tolerance_s').value
        )
        self.camera_info_aspect_tolerance = float(
            self.get_parameter('camera_info_aspect_tolerance').value
        )
        self.inference_timeout_s = float(self.get_parameter('inference_timeout_s').value)
        self.input_stall_warn_s = float(self.get_parameter('input_stall_warn_s').value)
        self.mask_erode_iterations = int(self.get_parameter('mask_erode_iterations').value)
        self.mask_erode_kernel_size = int(self.get_parameter('mask_erode_kernel_size').value)
        self.central_mask_keep_ratio = float(self.get_parameter('central_mask_keep_ratio').value)
        self.max_pose_jump_m = float(self.get_parameter('max_pose_jump_m').value)
        self.max_pose_speed_mps = float(self.get_parameter('max_pose_speed_mps').value)

        self._rng = np.random.default_rng(seed=1234)
        self._bridge = CvBridge()
        self._history: deque[Tuple[float, float, float]] = deque()

        self._latest_pair: Optional[Tuple[Image, Image]] = None
        self._camera_info: Optional[CameraInfo] = None
        self._camera_info_received_wall_s: Optional[float] = None
        self._latest_lock = threading.Lock()
        self._last_processed_stamp: Optional[Tuple[int, int]] = None
        self._last_submitted_stamp: Optional[Tuple[int, int]] = None
        self._last_sync_wall_s: Optional[float] = None
        self._last_sync_stamp: Optional[Tuple[int, int]] = None
        self._last_sync_stamp_change_wall_s: Optional[float] = None
        self._last_detection_time: Optional[float] = None
        self._last_visible = False
        self._last_marker_frame_id = ''
        self._filtered_cube_position: Optional[np.ndarray] = None
        self._last_filter_time_s: Optional[float] = None
        self._worker_condition = threading.Condition()
        self._worker_pending: Optional[Tuple[Image, Image, Tuple[int, int]]] = None
        self._worker_result: Optional[WorkerResult] = None
        self._worker_stop = False
        self._inference_started_wall_s: Optional[float] = None
        self._inference_count = 0
        self._last_inference_duration_s: Optional[float] = None
        self._last_status_log_time = 0.0
        self._last_yolo_log_time = 0.0
        self._last_pipeline_log_time = 0.0
        self._last_timing_log_time = 0.0
        self._last_health_log_wall_s = 0.0
        self._watchdog_fired = False
        self._last_centroid_reject_reason = ''
        self._last_process_wall_time: Optional[float] = None
        self._current_processing_hz = 0.0

        self._model = self._load_model()

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name='cube-tracker-inference',
            daemon=True,
        )
        self._worker_thread.start()

        self._state_pub = self.create_publisher(Odometry, self.cube_state_topic, 10)
        self._visible_pub = self.create_publisher(Bool, self.cube_visible_topic, 10)
        self._marker_pub = self.create_publisher(Marker, self.cube_marker_topic, 10)
        self._debug_pub = self.create_publisher(CompressedImage, self.debug_image_topic, 2)
        self._mask_pub = self.create_publisher(CompressedImage, self.mask_image_topic, 2)
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
        self._depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_cb,
            sensor_qos,
        )
        self._sync = ApproximateTimeSynchronizer(
            [self._image_sub, self._depth_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._sync_cb)

        # A steady clock keeps health checks and the inference watchdog alive even
        # if ROS/simulation time pauses.
        self._timer = self.create_timer(
            1.0 / self.processing_rate_hz,
            self._on_timer,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        self.get_logger().info(
            f'Cube tracker ready. model={self.model_path} image={self.image_topic} '
            f'depth={self.depth_topic} inference={self.inference_width}x{self.inference_height}'
        )

    def _declare_parameters(self) -> None:
        # Input and output topics.
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        # Aligned depth uses the color optical geometry.
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('cube_state_topic', '/go2_fetch/cube_state')
        self.declare_parameter('cube_visible_topic', '/go2_fetch/cube_visible')
        self.declare_parameter('cube_marker_topic', '/go2_fetch/cube_marker')
        self.declare_parameter('debug_image_topic', '/go2_fetch/cube_debug_image/compressed')
        self.declare_parameter('mask_image_topic', '/go2_fetch/cube_mask_image/compressed')
        self.declare_parameter('pre_yolo_image_topic', '/go2_fetch/cube_pre_yolo_image')

        # Detector model.
        self.declare_parameter('model_path', '/home/ferdinand/unitree/go2_fetch_ros2/fetch/models/yoloe-26l-seg.onnx')
        self.declare_parameter('conf_threshold', 0.1)
        self.declare_parameter('yolo_classes', ['box'])
        self.declare_parameter('inference_width', 640)
        self.declare_parameter('inference_height', 360)
        self.declare_parameter('depth_scale_m', 0.001)

        # Processing and diagnostics.
        self.declare_parameter('processing_rate_hz', 20.0)
        self.declare_parameter('status_log_rate_hz', 2.0)
        self.declare_parameter('pipeline_log_rate_hz', 5.0)
        self.declare_parameter('enable_timing_log', True)
        self.declare_parameter('timing_log_rate_hz', 5.0)
        self.declare_parameter('inference_timeout_s', 5.0)
        self.declare_parameter('input_stall_warn_s', 2.0)

        # Image and mask preprocessing.
        self.declare_parameter('enable_hsv_pre_mask', False)
        self.declare_parameter('publish_pre_yolo_image', True)
        self.declare_parameter('hsv_green_lower', [35, 40, 40])
        self.declare_parameter('hsv_green_upper', [90, 255, 255])
        self.declare_parameter('hsv_blue_lower', [90, 40, 40])
        self.declare_parameter('hsv_blue_upper', [135, 255, 255])
        self.declare_parameter('mask_erode_iterations', 1)
        self.declare_parameter('mask_erode_kernel_size', 5)
        self.declare_parameter('central_mask_keep_ratio', 0.3)

        # Point-cloud extraction and filtering.
        self.declare_parameter('max_mask_samples', 2500)
        self.declare_parameter('min_inlier_points', 30)
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('outlier_mad_scale', 3.0)

        # Tracking and smoothing.
        self.declare_parameter('detection_timeout_s', 0.35)
        self.declare_parameter('velocity_window_s', 0.6)
        self.declare_parameter('cube_state_low_pass_cutoff_hz', 2.0)
        self.declare_parameter('max_pose_jump_m', 0.30)
        self.declare_parameter('max_pose_speed_mps', 1.0)

        # Coordinate frame and cube geometry.
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('tf_timeout_s', 0.05)
        self.declare_parameter('camera_info_max_age_s', 5.0)
        self.declare_parameter('camera_info_stamp_tolerance_s', 1.0)
        self.declare_parameter('camera_info_aspect_tolerance', 0.02)
        self.declare_parameter('cube_dimensions', [0.16, 0.16, 0.16])
        self.declare_parameter('cube_position_offset_xyz', [0.0, 0.0, 0.0])

        # Debug visualization.
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

    def _camera_info_cb(self, camera_info_msg: CameraInfo) -> None:
        self._camera_info = camera_info_msg
        self._camera_info_received_wall_s = time.monotonic()

    def _sync_cb(self, image_msg: Image, depth_msg: Image) -> None:
        now_wall_s = time.monotonic()
        stamp = (image_msg.header.stamp.sec, image_msg.header.stamp.nanosec)
        with self._latest_lock:
            self._latest_pair = (image_msg, depth_msg)
            self._last_sync_wall_s = now_wall_s
            if stamp != self._last_sync_stamp:
                self._last_sync_stamp = stamp
                self._last_sync_stamp_change_wall_s = now_wall_s

    def _on_timer(self) -> None:
        self._consume_worker_result()

        now_s = self.get_clock().now().nanoseconds * 1e-9
        now_wall_s = time.monotonic()
        self._update_visibility_from_timeout(now_s)
        self._check_inference_watchdog(now_wall_s)
        self._log_health(now_wall_s)

        pair: Optional[Tuple[Image, Image]] = None
        with self._latest_lock:
            if self._latest_pair is not None:
                pair = self._latest_pair

        if pair is None:
            return

        stamp = (pair[0].header.stamp.sec, pair[0].header.stamp.nanosec)
        if stamp == self._last_submitted_stamp:
            return
        now_wall = time.perf_counter()
        if self._last_process_wall_time is not None:
            dt = now_wall - self._last_process_wall_time
            if dt > 1e-6:
                self._current_processing_hz = 1.0 / dt
        self._last_process_wall_time = now_wall

        with self._worker_condition:
            if self._worker_pending is not None or self._inference_started_wall_s is not None:
                return
            self._last_submitted_stamp = stamp
            self._worker_pending = (pair[0], pair[1], stamp)
            self._worker_condition.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._worker_condition:
                self._worker_condition.wait_for(
                    lambda: self._worker_pending is not None or self._worker_stop
                )
                if self._worker_stop:
                    return
                image_msg, depth_msg, stamp = self._worker_pending
                self._worker_pending = None
                self._inference_started_wall_s = time.monotonic()

            self.get_logger().info(
                f'inference start: frame={stamp[0]}.{stamp[1]:09d} count={self._inference_count + 1}'
            )
            error = None
            detection = None
            try:
                detection = self._process_pair(image_msg, depth_msg)
            except Exception as exc:
                error = f'{exc}\n{traceback.format_exc()}'

            finished_wall_s = time.monotonic()
            with self._worker_condition:
                started_wall_s = self._inference_started_wall_s
                self._inference_started_wall_s = None
                self._inference_count += 1
                if started_wall_s is not None:
                    self._last_inference_duration_s = finished_wall_s - started_wall_s
                self._worker_result = WorkerResult(detection, stamp, error)

            self.get_logger().info(
                f'inference end: frame={stamp[0]}.{stamp[1]:09d} '
                f'duration={self._last_inference_duration_s or 0.0:.3f}s'
            )

    def _consume_worker_result(self) -> None:
        with self._worker_condition:
            result = self._worker_result
            self._worker_result = None
        if result is None:
            return

        self._last_processed_stamp = result.stamp
        if result.error is not None:
            self.get_logger().error(f'Cube tracker worker failed: {result.error}')
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if result.detection is not None:
            self._last_detection_time = now_s
            self._publish_detection(result.detection, result.stamp)
        self._update_visibility_from_timeout(now_s)
        self._log_status(now_s, result.detection)

    def _check_inference_watchdog(self, now_wall_s: float) -> None:
        with self._worker_condition:
            started_wall_s = self._inference_started_wall_s
        if (
            self.inference_timeout_s <= 0.0
            or started_wall_s is None
            or self._watchdog_fired
        ):
            return
        elapsed_s = now_wall_s - started_wall_s
        if elapsed_s <= self.inference_timeout_s:
            return
        self._watchdog_fired = True
        self.get_logger().fatal(
            f'Inference stalled for {elapsed_s:.2f}s (limit={self.inference_timeout_s:.2f}s); '
            'exiting so the launch supervisor can restart the node.'
        )
        os._exit(70)

    def _log_health(self, now_wall_s: float) -> None:
        period_s = 1.0 / self.status_log_rate_hz if self.status_log_rate_hz > 0.0 else 1.0
        if now_wall_s - self._last_health_log_wall_s < period_s:
            return
        self._last_health_log_wall_s = now_wall_s
        with self._latest_lock:
            sync_age = (
                None if self._last_sync_wall_s is None else now_wall_s - self._last_sync_wall_s
            )
            stamp_age = (
                None
                if self._last_sync_stamp_change_wall_s is None
                else now_wall_s - self._last_sync_stamp_change_wall_s
            )
        with self._worker_condition:
            inference_age = (
                None if self._inference_started_wall_s is None
                else now_wall_s - self._inference_started_wall_s
            )
        if sync_age is None or (
            self.input_stall_warn_s > 0.0 and sync_age > self.input_stall_warn_s
        ):
            age_text = 'never' if sync_age is None else f'{sync_age:.2f}s'
            self.get_logger().warn(f'health: no recent synchronized color/depth pair; age={age_text}')
        elif self.input_stall_warn_s > 0.0 and (
            stamp_age is None or stamp_age > self.input_stall_warn_s
        ):
            age_text = 'never' if stamp_age is None else f'{stamp_age:.2f}s'
            self.get_logger().warn(f'health: synchronized frame timestamp is not advancing; age={age_text}')
        elif inference_age is not None:
            self.get_logger().info(
                f'health: executor alive, inference active for {inference_age:.2f}s, '
                f'completed={self._inference_count}'
            )
        else:
            duration_text = (
                'n/a' if self._last_inference_duration_s is None
                else f'{self._last_inference_duration_s:.3f}s'
            )
            self.get_logger().info(
                f'health: executor alive, sync_age={sync_age:.2f}s, '
                f'last_inference={duration_text}, completed={self._inference_count}'
            )

    def _process_pair(self, image_msg: Image, depth_msg: Image) -> Optional[DetectionResult]:
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
        yolo_input = cv2.resize(
            image,
            (self.inference_width, self.inference_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mark('resize_for_yolo')
        if self.enable_hsv_pre_mask:
            yolo_input, color_mask = self._apply_hsv_green_blue_mask(yolo_input)
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

        inference_mask = self._extract_mask(result, best_idx, yolo_input.shape[:2])
        mark('extract_mask')
        if inference_mask is None:
            self._maybe_log_pipeline('pipeline: mask rejected (too few mask pixels)')
            log_timing('mask_rejected')
            return None

        self._maybe_log_pipeline(f'pipeline: inference mask pixels={int(inference_mask.sum())}')

        camera_mask = None
        if self.publish_mask_image or self.publish_debug_image:
            camera_mask = cv2.resize(
                inference_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            mark('resize_mask_to_camera')

        if self.publish_mask_image:
            self._publish_mask(camera_mask, image_msg)
            mark('publish_mask_image')

        if self.publish_debug_image:
            self._publish_debug(image, result, camera_mask)
            mark('publish_debug_image_2d')

        # Keep depth sampling in the small inference grid. Pixel coordinates are
        # mapped directly to the aligned-depth resolution during deprojection.
        depth_mask = self._erode_mask_for_depth(inference_mask)
        centroid = self._extract_filtered_centroid(depth_mask, depth_msg)
        mark('extract_filtered_centroid')
        if centroid is None:
            reason = self._last_centroid_reject_reason or 'unknown reason'
            self._maybe_log_pipeline(f'pipeline: distance unavailable, centroid rejected ({reason})')
            log_timing('centroid_rejected')
            return None

        x, y, z = centroid
        point = PointStamped()
        point.header = depth_msg.header
        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        out_frame = depth_msg.header.frame_id
        if self.target_frame and self.target_frame != depth_msg.header.frame_id:
            try:
                transformed = self._tf_buffer.transform(
                    point,
                    self.target_frame,
                    timeout=Duration(seconds=self.tf_timeout_s),
                )
                point = transformed
                out_frame = self.target_frame
            except Exception as exc:
                self.get_logger().warn(f'TF transform failed ({depth_msg.header.frame_id}->{self.target_frame}): {exc}')
                log_timing('tf_transform_failed')
                return None
            mark('tf_transform')

        now_filter_s = self.get_clock().now().nanoseconds * 1e-9
        raw_pos_array = np.asarray(
            [float(point.point.x), float(point.point.y), float(point.point.z)],
            dtype=np.float64,
        )

        if not self._is_reasonable_pose_jump(raw_pos_array, now_filter_s):
            self._maybe_log_pipeline('pipeline: rejected unreasonable pose jump')
            log_timing('pose_jump_rejected')
            return None

        raw_pos_xyz = (float(raw_pos_array[0]), float(raw_pos_array[1]), float(raw_pos_array[2]))
        filtered_pos_xyz = self._low_pass_cube_position(raw_pos_xyz, now_s=now_filter_s)
        pos_xyz = tuple(
            float(filtered_pos_xyz[i] + self.cube_position_offset_xyz[i]) for i in range(3)
        )
        pos_xy = pos_xyz[:2]
        vel_xy = self._estimate_velocity(pos_xy)
        mark('estimate_velocity')

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
        encoded, buffer = cv2.imencode(
            '.jpg', mask_u8, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not encoded:
            self.get_logger().warn('Failed to JPEG-encode mask image')
            return
        msg = CompressedImage()
        msg.header = image_msg.header
        msg.format = 'jpeg'
        msg.data = buffer.tobytes()
        self._mask_pub.publish(msg)

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
            polygon = None
            if hasattr(result.masks, 'xy') and len(result.masks.xy) > idx:
                polygon = np.asarray(result.masks.xy[idx], dtype=np.float32)

            if polygon is not None and polygon.ndim == 2 and polygon.shape[0] >= 3:
                points = np.rint(polygon).astype(np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                mask_u8 = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_u8, [points], 1)
                mask = mask_u8.astype(bool)
            else:
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

    def _erode_mask_for_depth(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_erode_iterations <= 0:
            return mask

        kernel_size = max(1, int(self.mask_erode_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=self.mask_erode_iterations,
        ).astype(bool)

        # If the cube is far away, erosion may remove too many pixels.
        # In that case, keep the original mask.
        if int(eroded.sum()) < self.min_inlier_points:
            return mask

        return eroded


    def _keep_central_mask_pixels(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if xs.size == 0:
            return xs, ys

        keep_ratio = float(np.clip(self.central_mask_keep_ratio, 0.1, 1.0))
        if keep_ratio >= 1.0:
            return xs, ys

        cx = np.median(xs)
        cy = np.median(ys)
        r = np.sqrt((xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2)
        threshold = np.quantile(r, keep_ratio)

        keep = r <= threshold
        if np.count_nonzero(keep) < self.min_inlier_points:
            return xs, ys

        return xs[keep], ys[keep]

    def _depth_array(self, depth_msg: Image) -> Optional[Tuple[np.ndarray, float]]:
        encoding = str(depth_msg.encoding).lower()
        if encoding in ('16uc1', 'mono16'):
            dtype = np.dtype(np.uint16)
            scale = self.depth_scale_m
        elif encoding == '32fc1':
            dtype = np.dtype(np.float32)
            scale = 1.0
        else:
            self._last_centroid_reject_reason = f'unsupported depth encoding {depth_msg.encoding}'
            return None

        dtype = dtype.newbyteorder('>' if depth_msg.is_bigendian else '<')
        width = int(depth_msg.width)
        height = int(depth_msg.height)
        row_items = int(depth_msg.step) // dtype.itemsize
        if width <= 0 or height <= 0 or row_items < width:
            self._last_centroid_reject_reason = (
                f'invalid depth layout width={width} height={height} step={depth_msg.step}'
            )
            return None

        expected_items = row_items * height
        values = np.frombuffer(depth_msg.data, dtype=dtype, count=expected_items)
        if values.size != expected_items:
            self._last_centroid_reject_reason = (
                f'depth payload has {values.size} values, expected {expected_items}'
            )
            return None

        return values.reshape(height, row_items)[:, :width], scale

    def _extract_filtered_centroid(self, mask: np.ndarray, depth_msg: Image) -> Optional[np.ndarray]:
        camera_info = self._camera_info
        if camera_info is None:
            self._last_centroid_reject_reason = 'waiting for aligned-depth camera_info'
            return None

        now_wall_s = time.monotonic()
        if self._camera_info_received_wall_s is None or (
            self.camera_info_max_age_s > 0.0
            and now_wall_s - self._camera_info_received_wall_s > self.camera_info_max_age_s
        ):
            self._last_centroid_reject_reason = 'camera_info is stale'
            return None
        if (
            camera_info.header.frame_id
            and depth_msg.header.frame_id
            and camera_info.header.frame_id != depth_msg.header.frame_id
        ):
            self._last_centroid_reject_reason = (
                f'camera_info frame {camera_info.header.frame_id} does not match depth frame '
                f'{depth_msg.header.frame_id}'
            )
            return None
        info_stamp_s = camera_info.header.stamp.sec + camera_info.header.stamp.nanosec * 1e-9
        depth_stamp_s = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        if (
            self.camera_info_stamp_tolerance_s > 0.0
            and info_stamp_s > 0.0
            and depth_stamp_s > 0.0
            and abs(depth_stamp_s - info_stamp_s) > self.camera_info_stamp_tolerance_s
        ):
            self._last_centroid_reject_reason = (
                f'camera_info timestamp differs from depth by '
                f'{abs(depth_stamp_s - info_stamp_s):.3f}s'
            )
            return None

        depth_result = self._depth_array(depth_msg)
        if depth_result is None:
            return None
        depth, depth_scale = depth_result

        ys, xs = np.where(mask)
        if xs.size == 0:
            self._last_centroid_reject_reason = 'mask has zero valid pixels'
            return None

        xs, ys = self._keep_central_mask_pixels(xs, ys)

        if xs.size > self.max_mask_samples:
            pick = self._rng.choice(xs.size, size=self.max_mask_samples, replace=False)
            xs = xs[pick]
            ys = ys[pick]

        height, width = depth.shape
        if width <= 0 or height <= 0:
            self._last_centroid_reject_reason = f'invalid depth dimensions width={width} height={height}'
            return None

        mask_height, mask_width = mask.shape
        # Map the full camera mask into the aligned-depth grid. This also handles
        # a decimation filter changing the aligned depth resolution.
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
            self._last_centroid_reject_reason = 'all sampled mask pixels fell outside depth bounds'
            return None

        flat_indices = np.unique(ys.astype(np.int64) * width + xs.astype(np.int64))
        ys, xs = np.divmod(flat_indices, width)
        z = depth[ys, xs].astype(np.float64) * depth_scale

        info_width = int(camera_info.width) or width
        info_height = int(camera_info.height) or height
        if info_width <= 0 or info_height <= 0:
            self._last_centroid_reject_reason = (
                f'invalid camera_info dimensions {info_width}x{info_height}'
            )
            return None
        aspect_error = abs((width / height) / (info_width / info_height) - 1.0)
        if aspect_error > self.camera_info_aspect_tolerance:
            self._last_centroid_reject_reason = (
                f'camera_info aspect ratio does not match depth: info={info_width}x{info_height} '
                f'depth={width}x{height}'
            )
            return None
        scale_x = width / info_width
        scale_y = height / info_height
        fx = float(camera_info.k[0]) * scale_x
        fy = float(camera_info.k[4]) * scale_y
        cx = float(camera_info.k[2]) * scale_x
        cy = float(camera_info.k[5]) * scale_y
        if fx <= 0.0 or fy <= 0.0:
            self._last_centroid_reject_reason = f'invalid camera intrinsics fx={fx} fy={fy}'
            return None

        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        points = np.column_stack((x, y, z))

        valid_depth = np.logical_and.reduce((
            np.isfinite(points).all(axis=1),
            points[:, 2] > self.min_depth_m,
            points[:, 2] < self.max_depth_m,
        ))
        points = points[valid_depth]
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

        z_median = np.median(points[:, 2])
        z_mad = np.median(np.abs(points[:, 2] - z_median))
        z_mad = max(float(z_mad), 1e-4)

        depth_cluster = np.abs(points[:, 2] - z_median) < 2.5 * z_mad
        points = points[depth_cluster]

        if points.shape[0] < self.min_inlier_points:
            self._last_centroid_reject_reason = (
                f'too few depth-cluster inliers: {points.shape[0]} < '
                f'min_inlier_points({self.min_inlier_points})'
            )
            return None

        self._last_centroid_reject_reason = ''

        return np.median(points, axis=0)

    def _is_reasonable_pose_jump(
        self,
        sample_xyz: np.ndarray,
        now_s: float,
    ) -> bool:
        if self._filtered_cube_position is None or self._last_filter_time_s is None:
            return True

        dt = max(1e-3, now_s - self._last_filter_time_s)
        planar_jump = float(np.linalg.norm(sample_xyz[:2] - self._filtered_cube_position[:2]))

        allowed_jump = self.max_pose_jump_m + self.max_pose_speed_mps * dt
        return planar_jump <= allowed_jump

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
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.cube_dimensions[0]
        marker.scale.y = self.cube_dimensions[1]
        marker.scale.z = self.cube_dimensions[2]
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 0.75
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
    ) -> None:
        draw = result.plot()
        if draw.shape[:2] != image.shape[:2]:
            draw = cv2.resize(draw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = np.zeros_like(draw)
        overlay[mask] = (0, 255, 0)
        draw = cv2.addWeighted(draw, 1.0, overlay, 0.2, 0.0)
        line = 'YOLO detection'
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
        encoded, buffer = cv2.imencode(
            '.jpg',
            np.ascontiguousarray(draw, dtype=np.uint8),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not encoded:
            self.get_logger().warn('Failed to JPEG-encode debug image')
            return
        debug_msg = CompressedImage()
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        debug_msg.format = 'jpeg'
        debug_msg.data = buffer.tobytes()
        self._debug_pub.publish(debug_msg)

    def _update_visibility_from_timeout(self, now_s: float) -> None:
        visible = False
        if self._last_detection_time is not None:
            visible = (now_s - self._last_detection_time) <= self.detection_timeout_s
        was_visible = self._last_visible
        self._publish_visible(visible)
        if was_visible and not visible:
            self._delete_cube_marker()
        if not visible:
            with self._worker_condition:
                inference_active = self._inference_started_wall_s is not None
            if not inference_active:
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

    def destroy_node(self):
        with self._worker_condition:
            self._worker_stop = True
            self._worker_condition.notify_all()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        return super().destroy_node()


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
