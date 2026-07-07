#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""High-level ROS 2 policy supervisor for the Go2 fetch task."""

from __future__ import annotations

import math
import json
import csv
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import tensorrt as trt
import torch
from geometry_msgs.msg import Point, TwistStamped
from std_msgs.msg import Bool
from fetch_interfaces.msg import ControlState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
import tf2_ros
from visualization_msgs.msg import Marker

from fetch.deploy_real_utils import (
    KeyMap,
    RemoteController,
    get_gravity_orientation,
)
from fetch.policy_observation import (
    TimedCubeState,
    build_pushcube_observation,
    reorder_and_correct_foot_force,
    select_cube_state,
)

CONTROL_MODE_HIERARCHICAL_LOWCMD = "hierarchical_lowcmd"
CONTROL_MODE_UNITREE_SPORT_HIGH_LEVEL = "unitree_sport_high_level"
SPORT_API_ID_STOPMOVE = 1003
SPORT_API_ID_MOVE = 1008


@dataclass(frozen=True)
class SupervisorConfig:
    lowstate_topic: str
    leg_joint2motor_idx: tuple[int, ...]
    default_angles: np.ndarray
    max_cmd: np.ndarray
    num_actions: int


class TensorRTPolicy:
    """Run a single-input, single-output TensorRT policy on CUDA."""

    _TORCH_DTYPES = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }

    def __init__(self, engine_path: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is required to run TensorRT policy engine: {engine_path}"
            )

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        input_names = []
        output_names = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"TensorRT policy must have one input and one output; "
                f"found {len(input_names)} inputs and {len(output_names)} outputs in {engine_path}"
            )

        self._input_name = input_names[0]
        self._output_name = output_names[0]
        self._input_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        self._output_shape = tuple(self._engine.get_tensor_shape(self._output_name))
        if any(dimension < 0 for dimension in self._input_shape + self._output_shape):
            raise RuntimeError(
                f"Dynamic TensorRT policy shapes are not supported: {engine_path}"
            )

        input_trt_dtype = self._engine.get_tensor_dtype(self._input_name)
        output_trt_dtype = self._engine.get_tensor_dtype(self._output_name)
        try:
            input_torch_dtype = self._TORCH_DTYPES[input_trt_dtype]
            output_torch_dtype = self._TORCH_DTYPES[output_trt_dtype]
        except KeyError as error:
            raise RuntimeError(
                f"Unsupported TensorRT policy data type in {engine_path}: {error.args[0]}"
            ) from error

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                f"Failed to create TensorRT execution context: {engine_path}"
            )
        self._input = torch.empty(
            self._input_shape, dtype=input_torch_dtype, device="cuda"
        )
        self._output = torch.empty(
            self._output_shape, dtype=output_torch_dtype, device="cuda"
        )
        self._host_input = torch.empty(
            self._input_shape,
            dtype=input_torch_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._host_output = torch.empty(
            self._output_shape,
            dtype=output_torch_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._host_input_numpy = self._host_input.numpy()
        self._host_output_numpy = self._host_output.numpy()
        self._stream = torch.cuda.Stream()
        self._event_start = torch.cuda.Event(enable_timing=True)
        self._event_after_h2d = torch.cuda.Event(enable_timing=True)
        self._event_after_execute = torch.cuda.Event(enable_timing=True)
        self._event_after_d2h = torch.cuda.Event(enable_timing=True)
        self.last_timing_ms = {
            "host_input": 0.0,
            "h2d": 0.0,
            "enqueue": 0.0,
            "execute": 0.0,
            "d2h": 0.0,
            "sync_wait": 0.0,
            "total": 0.0,
        }
        input_address_set = self._context.set_tensor_address(
            self._input_name, self._input.data_ptr()
        )
        output_address_set = self._context.set_tensor_address(
            self._output_name, self._output.data_ptr()
        )
        if not input_address_set or not output_address_set:
            raise RuntimeError(
                f"Failed to bind TensorRT policy buffers: {engine_path}"
            )

        warmup_input = np.zeros(self._input_shape, dtype=self._host_input_numpy.dtype)
        for _ in range(50):
            self.infer(warmup_input)
        print(f"TensorRT warm-up finished: {engine_path}", flush=True)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def infer(self, observation: np.ndarray) -> np.ndarray:
        call_start = time.perf_counter()
        observation = np.asarray(observation)
        if observation.shape == self._input_shape[1:]:
            observation = observation.reshape(self._input_shape)
        if observation.shape != self._input_shape:
            raise ValueError(
                f"TensorRT policy expected input shape {self._input_shape}, "
                f"got {observation.shape}"
            )

        np.copyto(self._host_input_numpy, observation, casting="unsafe")
        host_input_done = time.perf_counter()
        with torch.cuda.stream(self._stream):
            self._event_start.record(self._stream)
            self._input.copy_(self._host_input, non_blocking=True)
            self._event_after_h2d.record(self._stream)
            enqueue_start = time.perf_counter()
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT policy inference failed")
            enqueue_done = time.perf_counter()
            self._event_after_execute.record(self._stream)
            self._host_output.copy_(self._output, non_blocking=True)
            self._event_after_d2h.record(self._stream)
        queued_done = time.perf_counter()
        self._stream.synchronize()
        sync_done = time.perf_counter()
        self.last_timing_ms = {
            "host_input": (host_input_done - call_start) * 1000.0,
            "h2d": self._event_start.elapsed_time(self._event_after_h2d),
            "enqueue": (enqueue_done - enqueue_start) * 1000.0,
            "execute": self._event_after_h2d.elapsed_time(
                self._event_after_execute
            ),
            "d2h": self._event_after_execute.elapsed_time(self._event_after_d2h),
            "sync_wait": (sync_done - queued_done) * 1000.0,
            "total": (sync_done - call_start) * 1000.0,
        }
        return self._host_output_numpy.reshape(-1)


class HighLevelPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("high_level_policy_node")
        self._declare_parameters()
        self.config = self._config_from_parameters()
        self.control_mode = self._control_mode_from_parameter()
        self.remote_controller = RemoteController()
        self.base_lin_vel_input = [0, 0, 0, 0]
        self.sport_request_publisher = None
        self.SportRequestRos = None
        self._stop_event = threading.Event()
        self._shutdown_requested = threading.Event()
        self._worker_thread = None
        self.fake_observations_mode = bool(self.get_parameter("fake_observations_mode").value)
        self.fake_cube_observation_mode = bool(
            self.get_parameter("fake_cube_observation_mode").value
        )
        self.use_high_level_policy = False
        self.add_on_set_parameters_callback(self._parameter_callback)
        self.commands_enabled = (
            bool(self.get_parameter("send_commands").value)
            and not self.fake_observations_mode
        )
        self._fake_rng = np.random.default_rng(
            int(self.get_parameter("fake_observation_seed").value)
        )

        self.get_logger().info("1] -> Configuration file loaded")
        self._initialize_controller_state()
        self._initialize_fake_cube_publisher()

        if self.fake_observations_mode:
            self.get_logger().warn(
                "FAKE OBSERVATION MODE ENABLED: DDS is not initialized and robot commands are disabled."
            )
        else:
            self._load_ros_interfaces()
            self._initialize_ros_io()
            self._initialize_robot_interfaces()

        if not self.commands_enabled:
            self.get_logger().warn("Robot command output is disabled.")
        if self.fake_cube_observation_mode and not self.fake_observations_mode:
            self.get_logger().warn(
                "Fake cube observation enabled: robot observations remain real, "
                "but cube position/velocity will come from parameters."
            )

        if bool(self.get_parameter("start_policy_on_startup").value):
            worker_target = self._run_high_level_supervisor_sequence
            self._worker_thread = threading.Thread(
                target=self._run_worker,
                args=(worker_target,),
                daemon=True,
            )
            self._worker_thread.start()

    def _run_worker(self, worker_target) -> None:
        try:
            worker_target()
        finally:
            self._shutdown_requested.set()

    def _declare_parameters(self) -> None:
        # Startup, operating mode, and command gates.
        self.declare_parameter("start_policy_on_startup", True)
        self.declare_parameter("control_mode", CONTROL_MODE_HIERARCHICAL_LOWCMD)
        self.declare_parameter("send_commands", False)
        self.declare_parameter("use_high_level_policy", True)

        # Remote controls.
        self.declare_parameter("high_level_toggle_button", "X")
        self.declare_parameter("goal_set_button", "Y")
        self.declare_parameter("cube_recovery_toggle_button", "B")
        # Unitree remote axes -> [forward, lateral, yaw]. Axis signs preserve
        # the mapping used by the original combined policy node.
        self.declare_parameter("joystick_command_scale", [1.0, 1.0, 1.0])

        # Goal definition and completion condition.
        self.declare_parameter("goal_xy", [0.0, 0.0])
        self.declare_parameter("goal_radius", 0.2)
        self.declare_parameter("cube_goal_stop_radius", 0.3)
        self.declare_parameter("cube_goal_hold_s", 0.6)
        self.declare_parameter("robot_goal_clear_radius", 0.35)

        # Cube-loss recovery rotation.
        self.declare_parameter("cube_state_timeout_s", 0.5)
        self.declare_parameter("cube_target_age_s", 0.065)
        self.declare_parameter("cube_stale_stop_ramp_s", 1.0)
        self.declare_parameter("cube_recovery_angular_cmd", 0.5)
        self.declare_parameter("cube_recovery_front_angle_deg", 20.0)
        self.declare_parameter("cube_recovery_max_rotation_deg", 360.0)

        # Fake observation modes.
        self.declare_parameter("fake_observations_mode", True)
        self.declare_parameter("fake_observation_seed", 0)
        self.declare_parameter("fake_observation_min", -1.0)
        self.declare_parameter("fake_observation_max", 1.0)
        self.declare_parameter("fake_log_every_n_steps", 100)
        self.declare_parameter("fake_cube_observation_mode", False)
        self.declare_parameter("fake_cube_position_xy", [0.8, 0.0])
        self.declare_parameter("fake_cube_velocity_xy", [0.0, 0.0])
        self.declare_parameter("fake_cube_publish_period_s", 0.05)

        # ROS topics and coordinate frames.
        self.declare_parameter("kalman_odom_topic", "/go2_odometry/filtered")
        self.declare_parameter("cube_state_topic", "/go2_fetch/cube_state")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("sport_request_topic", "/api/sport/request")
        self.declare_parameter("policy_world_frame", "odom")
        self.declare_parameter("lf_foot_frame", "FL_foot")
        self.declare_parameter("lf_foot_tf_timeout_s", 0.02)
        self.declare_parameter("cube_state_tf_timeout_s", 0.05)
        self.declare_parameter("robot_twist_in_body_frame", True)

        # RViz visualization.
        self.declare_parameter("cube_marker_topic", "/go2_fetch/cube_marker")
        self.declare_parameter("cube_dimensions", [0.16, 0.16, 0.16])
        self.declare_parameter("goal_marker_topic", "/go2_fetch/goal_marker")
        self.declare_parameter("goal_marker_publish_period_s", 0.2)
        self.declare_parameter("command_velocity_marker_topic", "/go2_fetch/command_velocity_marker")
        self.declare_parameter("current_velocity_marker_topic", "/go2_fetch/current_velocity_marker")
        self.declare_parameter("command_velocity_marker_frame", "base")
        self.declare_parameter("command_velocity_marker_z_offset", 0.25)
        self.declare_parameter("command_velocity_marker_scale", 0.25)
        self.declare_parameter("velocity_marker_rate_hz", 15.0)

        # Policy models and inference rates.
        self.declare_parameter(
            "high_level_policy_path",
            "logs/rsl_rl/unitree_go2_pushcube_4l/2026-05-15_02-52-05_cam_6/exported/policy.engine",
        )
        self.declare_parameter("high_level_rate_hz", 15.384615)
        self.declare_parameter("high_level_num_obs", 52)

        # Unitree Sport high-level control.
        self.declare_parameter("sport_move_publish_rate_hz", 15.0)
        self.declare_parameter("sport_stop_on_disable", True)
        self.declare_parameter("sport_command_log_every_n_steps", 50)
        self.declare_parameter("sport_command_scale", [-1.0, 1.0, 1.0])

        # High-level observation shape and command limits.
        self.declare_parameter("num_actions", 12)
        self.declare_parameter("max_cmd", [0.6, 0.4, 0.8])

        # Joint state fields included in the high-level observation.
        self.declare_parameter("leg_joint2motor_idx", [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
        self.declare_parameter("default_angles", [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
        self.declare_parameter("foot_force_offset", [4.0, 0.0, 5.0, 5.0])
        self.declare_parameter("foot_force_clip_max", 150.0)
        self.declare_parameter("foot_force_scale", 100.0)

        # Diagnostics and analysis output.
        self.declare_parameter("high_level_command_log_period_s", 1.0)
        self.declare_parameter("plot_on_exit", False)
        self.declare_parameter("analysis_pdf_path", "analyse_robot.png")
        self.declare_parameter("observation_csv_path", "")

    def _control_mode_from_parameter(self) -> str:
        control_mode = str(self.get_parameter("control_mode").value)
        valid_modes = {
            CONTROL_MODE_HIERARCHICAL_LOWCMD,
            CONTROL_MODE_UNITREE_SPORT_HIGH_LEVEL,
        }
        if control_mode not in valid_modes:
            valid = ", ".join(sorted(valid_modes))
            raise ValueError(f"control_mode must be one of: {valid}")
        return control_mode

    def _uses_unitree_sport_high_level(self) -> bool:
        return self.control_mode == CONTROL_MODE_UNITREE_SPORT_HIGH_LEVEL

    def _parameter_callback(self, parameters) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name == "fake_cube_observation_mode":
                if parameter.type_ != Parameter.Type.BOOL:
                    return SetParametersResult(
                        successful=False,
                        reason="fake_cube_observation_mode must be a Boolean",
                    )
                self.fake_cube_observation_mode = bool(parameter.value)
                if self.fake_cube_observation_mode:
                    self._apply_fake_cube_observation()
                    self._publish_fake_cube_state()
                source = "fake cube parameters" if parameter.value else "cube_state_topic"
                self.get_logger().info(f"Cube observation source changed to: {source}")
                continue

            if parameter.name != "use_high_level_policy":
                continue
            if parameter.type_ != Parameter.Type.BOOL:
                return SetParametersResult(
                    successful=False,
                    reason="use_high_level_policy must be a Boolean",
                )

            self._next_high_level_time = -math.inf
            if not bool(parameter.value):
                self._set_high_level_policy_enabled(False)
                self.get_logger().info(
                    "High-level policy disabled by parameter"
                )
            else:
                self.get_logger().info(
                    "High-level policy allowed by parameter; press the remote toggle button to enable it"
                )

        return SetParametersResult(successful=True)

    def _config_from_parameters(self) -> SupervisorConfig:
        return SupervisorConfig(
            lowstate_topic=str(self.get_parameter("lowstate_topic").value),
            leg_joint2motor_idx=tuple(
                int(value)
                for value in self.get_parameter("leg_joint2motor_idx").value
            ),
            default_angles=np.array(self.get_parameter("default_angles").value, dtype=np.float32),
            max_cmd=np.array(self.get_parameter("max_cmd").value, dtype=np.float32),
            num_actions=int(self.get_parameter("num_actions").value),
        )

    def _load_ros_interfaces(self) -> None:
        try:
            from unitree_go.msg import LowState as LowStateRos

            self.LowStateRos = LowStateRos
            if self._uses_unitree_sport_high_level():
                from unitree_api.msg import Request as SportRequestRos

                self.SportRequestRos = SportRequestRos
        except Exception as exc:
            self.LowStateRos = None
            self.SportRequestRos = None
            raise RuntimeError(f"unitree_go ROS messages are required in real mode: {exc}") from exc

    def _initialize_ros_io(self) -> None:
        self.create_subscription(Odometry, self.get_parameter("kalman_odom_topic").value, self._kalman_odom_callback, 10)
        self.create_subscription(Odometry, self.get_parameter("cube_state_topic").value, self._cube_state_callback, 10)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.high_level_cmd_publisher = self.create_publisher(
            TwistStamped, "/go2_fetch/high_level_cmd", 10
        )
        command_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.high_level_cmd_enabled_publisher = self.create_publisher(
            Bool, "/go2_fetch/high_level_cmd_enabled", command_state_qos
        )
        self.control_state_subscription = self.create_subscription(
            ControlState,
            "/go2_fetch/control_state",
            self._control_state_callback,
            command_state_qos,
        )

    def _control_state_callback(self, msg: ControlState) -> None:
        self.control_state = int(msg.state)

    def _publish_high_level_command(self, enabled: bool) -> None:
        if self.control_state != ControlState.RUNNING:
            return
        enabled_msg = Bool()
        enabled = bool(enabled and self.commands_enabled)
        enabled_msg.data = enabled
        self.high_level_cmd_enabled_publisher.publish(enabled_msg)
        command_msg = TwistStamped()
        command_msg.header.stamp = self.get_clock().now().to_msg()
        command_msg.header.frame_id = str(
            self.get_parameter("command_velocity_marker_frame").value
        )
        if enabled:
            command_msg.twist.linear.x = float(self.cmd[0])
            command_msg.twist.linear.y = float(self.cmd[1])
            command_msg.twist.angular.z = float(self.cmd[2])
        self.high_level_cmd_publisher.publish(command_msg)

    def _initialize_fake_cube_publisher(self) -> None:
        self.fake_cube_state_publisher = self.create_publisher(
            Odometry,
            self.get_parameter("cube_state_topic").value,
            10,
        )
        self.fake_cube_marker_publisher = self.create_publisher(
            Marker,
            self.get_parameter("cube_marker_topic").value,
            10,
        )
        self.goal_marker_publisher = self.create_publisher(
            Marker,
            self.get_parameter("goal_marker_topic").value,
            10,
        )
        self.command_velocity_marker_publisher = self.create_publisher(
            Marker,
            self.get_parameter("command_velocity_marker_topic").value,
            10,
        )
        self.current_velocity_marker_publisher = self.create_publisher(
            Marker,
            self.get_parameter("current_velocity_marker_topic").value,
            10,
        )
        self.fake_cube_publish_timer = self.create_timer(
            float(self.get_parameter("fake_cube_publish_period_s").value),
            self._publish_fake_cube_state,
        )
        self.goal_marker_publish_timer = self.create_timer(
            float(self.get_parameter("goal_marker_publish_period_s").value),
            self._publish_goal_marker,
        )

    def _initialize_controller_state(self) -> None:
        self.get_logger().info("3] ---> Loading high-level policy")
        high_level_path = self._resolve_policy_path("high_level_policy_path")
        self.high_level_policy = TensorRTPolicy(high_level_path)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.high_level_action = np.zeros(3, dtype=np.float32)
        self.previous_clamped_command = np.zeros(3, dtype=np.float32)
        self.high_level_obs = np.zeros(
            int(self.get_parameter("high_level_num_obs").value), dtype=np.float32
        )
        self.robot_pos_xy = np.zeros(2, dtype=np.float32)
        self.robot_lin_vel_world_xy = np.zeros(2, dtype=np.float32)
        self.robot_yaw = 0.0
        self.cube_pos_xy = np.zeros(2, dtype=np.float32)
        self.cube_lin_vel_xy = np.zeros(2, dtype=np.float32)
        self._cube_state_history = deque(maxlen=256)
        self._selected_cube_state = None
        self.robot_quaternion_world_from_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_cube_state_time = -math.inf
        self._cube_state_stale_logged = False
        self._cube_state_tf_warned = False
        if self.fake_cube_observation_mode:
            self._apply_fake_cube_observation()
        self.goal_xy = np.array(self.get_parameter("goal_xy").value, dtype=np.float32)
        self.goal_radius = float(self.get_parameter("goal_radius").value)
        self._goal_condition_reached = False
        self._cube_goal_enter_time = -math.inf
        self._last_robot_odom_time = -math.inf
        self._next_high_level_time = -math.inf
        self._next_velocity_marker_time = -math.inf
        self._last_high_level_toggle_pressed = False
        self._last_goal_set_button_pressed = False
        self._last_cube_recovery_toggle_pressed = False
        self._cube_tracking_lost = False
        self._stale_cube_ramp_start_time = -math.inf
        self._stale_cube_ramp_start_cmd = np.zeros(3, dtype=np.float32)
        self._cube_recovery_active = False
        self._cube_recovery_direction = 1.0
        self._cube_recovery_last_yaw = self.robot_yaw
        self._cube_recovery_rotation_rad = 0.0
        self._last_select_stop_pressed = False
        self._sport_stop_sent = False
        self._last_high_level_command_log_time = -math.inf
        self._analysis_plot_lock = threading.Lock()
        self._analysis_plot_attempted = False

        self.counter = 0

        self.L_base_vel_cmd_input_1 = []
        self.L_base_vel_cmd_input_2 = []
        self.L_base_vel_cmd_input_3 = []
        self.L_base_lin_vel_input_1 = []
        self.L_base_lin_vel_input_2 = []
        self.L_base_lin_vel_input_3 = []
        self.L_base_lin_vel_kalman_input_1 = []
        self.L_base_lin_vel_kalman_input_2 = []
        self.L_base_lin_vel_kalman_input_3 = []
        self.L_base_lin_vel_kalman_input_4 = []
        self.L_base_ang_vel_input_1 = []
        self.L_base_ang_vel_input_2 = []
        self.L_base_ang_vel_input_3 = []
        self.L_foot_force_fr = []
        self.L_foot_force_fl = []
        self.L_foot_force_rr = []
        self.L_foot_force_rl = []

    def _resolve_policy_path(self, parameter_name: str) -> Path:
        path = Path(str(self.get_parameter(parameter_name).value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{parameter_name} does not exist: {path}")
        if path.suffix != ".engine":
            raise ValueError(f"{parameter_name} must point to a .engine file: {path}")
        return path

    def _initialize_robot_interfaces(self) -> None:
        self.get_logger().info("4] ----> Initializing ROS 2 Unitree topics")
        self.sport_request_publisher = None
        if self._uses_unitree_sport_high_level():
            self.sport_request_publisher = self.create_publisher(
                self.SportRequestRos,
                self.get_parameter("sport_request_topic").value,
                10,
            )
        self.lowstate_subscriber = self.create_subscription(
            self.LowStateRos,
            self.config.lowstate_topic,
            self._low_state_go_handler,
            10,
        )
        self.low_state = self.LowStateRos()
        self.control_state = ControlState.ZERO_TORQUE


    def wait_for_low_state(self) -> None:
        while self.low_state.tick == 0 and not self._stop_event.is_set():
            time.sleep(0.01)
        self.get_logger().info("         Connected to robot")

    def _low_state_go_handler(self, msg) -> None:
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def _kalman_odom_callback(self, msg: Odometry) -> None:
        self._last_robot_odom_time = time.monotonic()
        self.base_lin_vel_input[0] = msg.twist.twist.linear.x
        self.base_lin_vel_input[1] = msg.twist.twist.linear.y
        self.base_lin_vel_input[2] = msg.twist.twist.linear.z
        self.base_lin_vel_input[3] = msg.twist.twist.angular.z
        self.robot_pos_xy[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        orientation = msg.pose.pose.orientation
        self.robot_quaternion_world_from_base[:] = [
            orientation.w, orientation.x, orientation.y, orientation.z
        ]
        self.robot_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        velocity_xy = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y], dtype=np.float32
        )
        if bool(self.get_parameter("robot_twist_in_body_frame").value):
            cos_yaw = math.cos(self.robot_yaw)
            sin_yaw = math.sin(self.robot_yaw)
            velocity_xy = np.array(
                [
                    cos_yaw * velocity_xy[0] - sin_yaw * velocity_xy[1],
                    sin_yaw * velocity_xy[0] + cos_yaw * velocity_xy[1],
                ],
                dtype=np.float32,
            )
        self.robot_lin_vel_world_xy[:] = velocity_xy

    def _cube_state_callback(self, msg: Odometry) -> None:
        if self.fake_cube_observation_mode:
            return
        transformed = self._transform_cube_state_to_policy_frame(msg)
        if transformed is None:
            return
        pos_xy, vel_xy = transformed
        now = time.monotonic()
        sample_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        if sample_time <= 0.0:
            sample_time = self.get_clock().now().nanoseconds * 1.0e-9
        self.cube_pos_xy[:] = pos_xy
        self.cube_lin_vel_xy[:] = vel_xy
        state = TimedCubeState(sample_time, pos_xy.copy(), vel_xy.copy())
        if not self._cube_state_history or sample_time >= self._cube_state_history[-1].stamp_s:
            self._cube_state_history.append(state)
        self._last_cube_state_time = now
        self._cube_state_stale_logged = False

    def _transform_cube_state_to_policy_frame(
        self, msg: Odometry
    ) -> tuple[np.ndarray, np.ndarray] | None:
        source_frame = msg.header.frame_id
        target_frame = str(self.get_parameter("policy_world_frame").value)
        pos_xy = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32
        )
        vel_xy = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y], dtype=np.float32
        )
        if not source_frame or source_frame == target_frame:
            self._cube_state_tf_warned = False
            return pos_xy, vel_xy

        timeout = Duration(
            seconds=float(self.get_parameter("cube_state_tf_timeout_s").value)
        )
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(msg.header.stamp),
                timeout=timeout,
            )
        except Exception as exc:
            if not self._cube_state_tf_warned:
                self.get_logger().warn(
                    f"Cube state TF {target_frame} <- {source_frame} unavailable; "
                    f"discarding cube state: {exc}"
                )
                self._cube_state_tf_warned = True
            return None

        self._cube_state_tf_warned = False
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotated_pos = np.array(
            [
                cos_yaw * pos_xy[0] - sin_yaw * pos_xy[1],
                sin_yaw * pos_xy[0] + cos_yaw * pos_xy[1],
            ],
            dtype=np.float32,
        )
        rotated_pos += np.array([translation.x, translation.y], dtype=np.float32)
        rotated_vel = np.array(
            [
                cos_yaw * vel_xy[0] - sin_yaw * vel_xy[1],
                sin_yaw * vel_xy[0] + cos_yaw * vel_xy[1],
            ],
            dtype=np.float32,
        )
        return rotated_pos, rotated_vel

    def _apply_fake_cube_observation(self) -> None:
        self.cube_pos_xy[:] = self._xy_parameter("fake_cube_position_xy")
        self.cube_lin_vel_xy[:] = self._xy_parameter("fake_cube_velocity_xy")

    def _publish_fake_cube_state(self) -> None:
        if not self.fake_cube_observation_mode:
            return
        self._apply_fake_cube_observation()
        self._last_cube_state_time = time.monotonic()

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("policy_world_frame").value)
        msg.child_frame_id = "cube"

        msg.pose.pose.position.x = float(self.cube_pos_xy[0])
        msg.pose.pose.position.y = float(self.cube_pos_xy[1])
        msg.pose.pose.position.z = 0.0
        msg.pose.covariance[0] = 0.01
        msg.pose.covariance[7] = 0.01

        msg.twist.twist.linear.x = float(self.cube_lin_vel_xy[0])
        msg.twist.twist.linear.y = float(self.cube_lin_vel_xy[1])
        msg.twist.covariance[0] = 0.04
        msg.twist.covariance[7] = 0.04

        self.fake_cube_state_publisher.publish(msg)
        self._publish_fake_cube_marker(msg.header.stamp, msg.header.frame_id)

    def _publish_fake_cube_marker(self, stamp, frame_id: str) -> None:
        dimensions = list(self.get_parameter("cube_dimensions").value)
        if len(dimensions) != 3:
            raise ValueError("cube_dimensions must contain exactly three values [x, y, z]")

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "cube_state"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.cube_pos_xy[0])
        marker.pose.position.y = float(self.cube_pos_xy[1])
        marker.pose.position.z = float(dimensions[2]) / 2.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(dimensions[0])
        marker.scale.y = float(dimensions[1])
        marker.scale.z = float(dimensions[2])
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        self.fake_cube_marker_publisher.publish(marker)

    def _publish_goal_marker(self) -> None:
        if not self._goal_condition_reached:
            if self._cube_state_is_current():
                self._goal_condition_reached = self._high_level_goal_stop_condition()[0]
            else:
                self._cube_goal_enter_time = -math.inf
        radius = abs(float(self.get_parameter("cube_goal_stop_radius").value))
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(self.get_parameter("policy_world_frame").value)
        marker.ns = "goal_region"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal_xy[0])
        marker.pose.position.y = float(self.goal_xy[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = 0.01
        marker.color.r = 0.2 if self._goal_condition_reached else 1.0
        marker.color.g = 1.0 if self._goal_condition_reached else 0.0
        marker.color.b = 0.2
        marker.color.a = 0.7
        self.goal_marker_publisher.publish(marker)

    def _publish_command_velocity_marker(self) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(
            self.get_parameter("command_velocity_marker_frame").value
        )
        marker.ns = "command_velocity"
        marker.id = 0

        vx = float(self.cmd[0])
        vy = float(self.cmd[1])
        command_speed = math.hypot(vx, vy)
        if command_speed < 1e-3:
            marker.action = Marker.DELETE
            self.command_velocity_marker_publisher.publish(marker)
            return

        z_offset = float(self.get_parameter("command_velocity_marker_z_offset").value)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            Point(x=0.0, y=0.0, z=z_offset),
            Point(
                x=vx * float(self.get_parameter("command_velocity_marker_scale").value),
                y=vy * float(self.get_parameter("command_velocity_marker_scale").value),
                z=z_offset,
            ),
        ]
        marker.scale.x = 0.035
        marker.scale.y = 0.09
        marker.scale.z = 0.12
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 0.9
        marker.lifetime = Duration(seconds=0.25).to_msg()
        self.command_velocity_marker_publisher.publish(marker)

    def _publish_current_velocity_marker(self) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(
            self.get_parameter("command_velocity_marker_frame").value
        )
        marker.ns = "current_velocity"
        marker.id = 0

        vx = float(self.base_lin_vel_input[0])
        vy = float(self.base_lin_vel_input[1])
        current_speed = math.hypot(vx, vy)
        if current_speed < 1e-3:
            marker.action = Marker.DELETE
            self.current_velocity_marker_publisher.publish(marker)
            return

        z_offset = float(self.get_parameter("command_velocity_marker_z_offset").value)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            Point(x=0.0, y=0.0, z=z_offset),
            Point(
                x=vx * float(self.get_parameter("command_velocity_marker_scale").value),
                y=vy * float(self.get_parameter("command_velocity_marker_scale").value),
                z=z_offset,
            ),
        ]
        marker.scale.x = 0.035
        marker.scale.y = 0.09
        marker.scale.z = 0.12
        marker.color.r = 1.0
        marker.color.g = 0.45
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.lifetime = Duration(seconds=0.25).to_msg()
        self.current_velocity_marker_publisher.publish(marker)

    def _publish_velocity_markers_if_due(self) -> None:
        rate_hz = float(self.get_parameter("velocity_marker_rate_hz").value)
        if rate_hz <= 0.0:
            return
        now = time.monotonic()
        if now < self._next_velocity_marker_time:
            return
        self._next_velocity_marker_time = now + 1.0 / rate_hz
        self._publish_command_velocity_marker()
        self._publish_current_velocity_marker()

    def _xy_parameter(self, name: str) -> np.ndarray:
        values = list(self.get_parameter(name).value)
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values [x, y]")
        return np.array(values, dtype=np.float32)







    def _publish_sport_request(self, api_id: int, parameter: dict | None = None) -> None:
        if not self.commands_enabled:
            return
        if self.sport_request_publisher is None or self.SportRequestRos is None:
            return

        request = self.SportRequestRos()
        request.header.identity.api_id = int(api_id)
        if parameter is not None:
            request.parameter = json.dumps(parameter)
        self.sport_request_publisher.publish(request)

    def _sport_command(self) -> np.ndarray:
        scale = np.array(self.get_parameter("sport_command_scale").value, dtype=np.float32)
        if scale.size != 3:
            raise ValueError("sport_command_scale must contain exactly three values [x, y, yaw]")
        return (self.cmd * scale).astype(np.float32)

    def _publish_sport_move(self) -> None:
        sport_cmd = self._sport_command()
        self._publish_sport_request(
            SPORT_API_ID_MOVE,
            {
                "x": float(sport_cmd[0]),
                "y": float(sport_cmd[1]),
                "z": float(sport_cmd[2]),
            },
        )
        self._sport_stop_sent = False
        self._log_sport_command()

    def _publish_sport_stop_move(self) -> None:
        if not bool(self.get_parameter("sport_stop_on_disable").value):
            return
        if self._sport_stop_sent:
            return
        self._publish_sport_request(SPORT_API_ID_STOPMOVE)
        self._sport_stop_sent = True

    def _set_high_level_policy_enabled(self, enabled: bool) -> None:
        was_enabled = self.use_high_level_policy
        self.use_high_level_policy = bool(enabled)
        self._next_high_level_time = -math.inf
        self._stale_cube_ramp_start_time = -math.inf
        if self.use_high_level_policy:
            self._cube_recovery_active = False
            if not was_enabled:
                self.previous_clamped_command[:] = 0.0
        if not self.use_high_level_policy:
            self.cmd[:] = 0.0
            if self._uses_unitree_sport_high_level():
                self._publish_sport_stop_move()
                self._publish_command_velocity_marker()

    def _real_cube_state_is_fresh(self) -> bool:
        if self.fake_observations_mode or self.fake_cube_observation_mode:
            return True

        timeout_s = float(self.get_parameter("cube_state_timeout_s").value)
        if timeout_s <= 0.0:
            return True

        age_s = time.monotonic() - self._last_cube_state_time
        if age_s <= timeout_s:
            return True

        if not self._cube_state_stale_logged:
            self.get_logger().warn(
                "No fresh cube_state_topic messages; disabling high-level policy commands "
                f"until cube tracking resumes. age_s={age_s:.3f} timeout_s={timeout_s:.3f}"
            )
            self._cube_state_stale_logged = True
        return False

    def _stop_for_stale_cube_state(self) -> None:
        self._cube_goal_enter_time = -math.inf
        self._cube_tracking_lost = True
        ramp_start_cmd = self.cmd.copy()
        self._set_high_level_policy_enabled(False)
        ramp_s = max(
            float(self.get_parameter("cube_stale_stop_ramp_s").value), 0.0
        )
        if not self._uses_unitree_sport_high_level() and ramp_s > 0.0:
            self._stale_cube_ramp_start_cmd[:] = ramp_start_cmd
            self._stale_cube_ramp_start_time = time.monotonic()
            self.cmd[:] = ramp_start_cmd
        self._publish_command_velocity_marker()
        if self._uses_unitree_sport_high_level():
            self._publish_sport_stop_move()

    def _high_level_goal_stop_condition(self) -> tuple[bool, float, float]:
        now = time.monotonic()
        cube_distance = float(np.linalg.norm(self.cube_pos_xy - self.goal_xy))
        robot_distance = float(np.linalg.norm(self.robot_pos_xy - self.goal_xy))
        cube_stop_radius = abs(
            float(self.get_parameter("cube_goal_stop_radius").value)
        )
        robot_clear_radius = abs(
            float(self.get_parameter("robot_goal_clear_radius").value)
        )
        cube_in_goal = cube_distance <= cube_stop_radius
        if cube_in_goal:
            if not math.isfinite(self._cube_goal_enter_time):
                self._cube_goal_enter_time = now
        else:
            self._cube_goal_enter_time = -math.inf
        hold_s = max(float(self.get_parameter("cube_goal_hold_s").value), 0.0)
        cube_hold_complete = (
            cube_in_goal and now - self._cube_goal_enter_time >= hold_s
        )
        should_stop = (
            cube_hold_complete
            and robot_distance > robot_clear_radius
        )
        return should_stop, cube_distance, robot_distance

    def _stop_high_level_if_goal_reached(self) -> bool:
        should_stop, cube_distance, robot_distance = (
            self._high_level_goal_stop_condition()
        )
        if not should_stop:
            return False
        self._goal_condition_reached = True
        self._publish_goal_marker()
        self._set_high_level_policy_enabled(False)
        self.get_logger().info(
            "High-level policy stopped: cube reached goal while robot base is clear "
            f"cube_goal_distance={cube_distance:.3f}m "
            f"robot_goal_distance={robot_distance:.3f}m "
            f"cube_hold={float(self.get_parameter('cube_goal_hold_s').value):.3f}s"
        )
        return True

    def _log_sport_command(self) -> None:
        log_every = int(self.get_parameter("sport_command_log_every_n_steps").value)
        if log_every <= 0 or self.counter % log_every != 0:
            return
        self.get_logger().info(
            "Unitree Sport Move "
            f"step={self.counter} "
            f"cmd={np.array2string(self.cmd, precision=3, suppress_small=True)}"
        )


    def run_fake_policy_step(self):
        observation_min = float(self.get_parameter("fake_observation_min").value)
        observation_max = float(self.get_parameter("fake_observation_max").value)
        if observation_min > observation_max:
            observation_min, observation_max = observation_max, observation_min

        self.counter += 1
        self.high_level_obs = self._fake_rng.uniform(
            observation_min,
            observation_max,
            size=int(self.get_parameter("high_level_num_obs").value),
        ).astype(np.float32)
        if self.use_high_level_policy:
            self.high_level_action = self.high_level_policy.infer(self.high_level_obs)
        else:
            self.high_level_action = self._fake_rng.uniform(
                observation_min, observation_max, size=3
            ).astype(np.float32)
        self.cmd[:] = np.clip(
            self.high_level_action[:3],
            -self.config.max_cmd,
            self.config.max_cmd,
        )
        self._publish_velocity_markers_if_due()

        log_every = max(
            1,
            int(self.get_parameter("fake_log_every_n_steps").value),
        )
        if self.counter % log_every == 0:
            self.get_logger().info(
                f"Fake high-level policy step {self.counter}: "
                f"high_obs={self.high_level_obs.size} "
                f"high_action={self.high_level_action[:3]} "
                "commands_sent=false"
            )

        return self.high_level_obs





    def run_unitree_sport_high_level_step(self):
        self.counter += 1
        self._update_sport_select_stop_from_remote()
        self._update_high_level_toggle_from_remote()
        self._update_goal_from_remote()
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)

        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = qj_obs - self.config.default_angles

        if self.use_high_level_policy:
            if not self._real_cube_state_is_fresh():
                self._stop_for_stale_cube_state()
            elif self._stop_high_level_if_goal_reached():
                pass
            else:
                self._cube_tracking_lost = False
                self._update_high_level_command(ang_vel, gravity_orientation, qj_obs, dqj_obs)
                self._publish_sport_move()
        self._update_cube_recovery_toggle_from_remote()
        if self._cube_recovery_active:
            self._update_cube_recovery_command()
            if self._cube_recovery_active:
                self._publish_sport_move()
            else:
                self._publish_sport_stop_move()
        elif not self.use_high_level_policy:
            self.cmd[:] = 0.0
            self._publish_sport_stop_move()

        self._publish_velocity_markers_if_due()
        self._record_policy_command_inputs(ang_vel)
        return self.high_level_obs

    def _update_high_level_command(
        self, ang_vel, gravity_orientation, qj_obs, dqj_obs
    ) -> None:
        now = time.monotonic()
        high_period = 1.0 / max(
            float(self.get_parameter("high_level_rate_hz").value), 1e-6
        )
        if now >= self._next_high_level_time:
            self.high_level_obs = self._build_high_level_observation(
                ang_vel, gravity_orientation, qj_obs, dqj_obs
            )
            self.high_level_action = self.high_level_policy.infer(
                self.high_level_obs
            )
            if self.high_level_action.size != 3:
                raise RuntimeError(
                    f"High-level policy returned {self.high_level_action.size} actions; expected 3"
                )
            if not math.isfinite(self._next_high_level_time):
                self._next_high_level_time = now + high_period
            else:
                while self._next_high_level_time <= now:
                    self._next_high_level_time += high_period

        self.cmd[:] = np.clip(
            self.high_level_action[:3],
            -self.config.max_cmd,
            self.config.max_cmd,
        )
        self.previous_clamped_command[:] = self.cmd
        self._log_high_level_command(now)

    def _log_high_level_command(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        period_s = float(self.get_parameter("high_level_command_log_period_s").value)
        if period_s <= 0.0 or now - self._last_high_level_command_log_time < period_s:
            return
        self._last_high_level_command_log_time = now
        self.get_logger().info(
            "High-level command "
            f"enabled={self.use_high_level_policy} "
            f"frame={self.get_parameter('policy_world_frame').value} "
            f"raw_action={np.array2string(self.high_level_action[:3], precision=3, suppress_small=True)} "
            f"cmd={np.array2string(self.cmd, precision=3, suppress_small=True)} "
            f"sport_cmd={np.array2string(self._sport_command(), precision=3, suppress_small=True)} "
            f"robot_xy={np.array2string(self.robot_pos_xy, precision=3, suppress_small=True)} "
            f"cube_xy={np.array2string(self.cube_pos_xy, precision=3, suppress_small=True)} "
            f"goal_xy={np.array2string(self.goal_xy, precision=3, suppress_small=True)} "
            f"cube_to_goal={np.array2string(self.goal_xy - self.cube_pos_xy, precision=3, suppress_small=True)}"
        )

    def _record_policy_command_inputs(self, ang_vel) -> None:
        self.L_base_vel_cmd_input_1.append(self.cmd[0])
        self.L_base_vel_cmd_input_2.append(self.cmd[1])
        self.L_base_vel_cmd_input_3.append(self.cmd[2])
        self.L_base_lin_vel_input_1.append(self.robot_lin_vel_world_xy[0])
        self.L_base_lin_vel_input_2.append(self.robot_lin_vel_world_xy[1])
        self.L_base_lin_vel_input_3.append(self.base_lin_vel_input[2])
        self.L_base_lin_vel_kalman_input_1.append(self.base_lin_vel_input[0])
        self.L_base_lin_vel_kalman_input_2.append(self.base_lin_vel_input[1])
        self.L_base_lin_vel_kalman_input_3.append(self.base_lin_vel_input[2])
        self.L_base_lin_vel_kalman_input_4.append(self.base_lin_vel_input[3])
        self.L_base_ang_vel_input_1.append(ang_vel[0])
        self.L_base_ang_vel_input_2.append(ang_vel[1])
        self.L_base_ang_vel_input_3.append(ang_vel[2])
        foot_force = self.low_state.foot_force
        self.L_foot_force_fr.append(foot_force[0])
        self.L_foot_force_fl.append(foot_force[1])
        self.L_foot_force_rr.append(foot_force[2])
        self.L_foot_force_rl.append(foot_force[3])


    def _update_high_level_toggle_from_remote(self) -> None:
        button_name = str(self.get_parameter("high_level_toggle_button").value)
        button_index = self._remote_button_index(button_name)
        pressed = self.remote_controller.button[button_index] == 1
        if pressed != self._last_high_level_toggle_pressed:
            self.get_logger().info(
                f"Remote {button_name.upper()} state changed to {int(pressed)} "
                f"button_index={button_index} key_mask=0x{self.remote_controller.keys:04x}"
            )
        if pressed and not self._last_high_level_toggle_pressed:
            if not bool(self.get_parameter("use_high_level_policy").value):
                self._set_high_level_policy_enabled(False)
                self.get_logger().info(
                    f"Remote {button_name.upper()} pressed, but high-level policy is disabled by parameter"
                )
                self._last_high_level_toggle_pressed = pressed
                return
            self._set_high_level_policy_enabled(not self.use_high_level_policy)
            if self._uses_unitree_sport_high_level():
                source = (
                    "PushCube high-level policy"
                    if self.use_high_level_policy
                    else "Unitree Sport StopMove"
                )
            else:
                source = (
                    "PushCube high-level policy"
                    if self.use_high_level_policy
                    else "joystick"
                )
            self.get_logger().info(
                f"Remote {button_name.upper()} pressed: velocity command source is now {source}"
            )
        self._last_high_level_toggle_pressed = pressed

    def _joystick_velocity_command(self) -> np.ndarray:
        """Convert the Unitree remote sticks to the low-level velocity command."""
        command = np.array(
            [
                self.remote_controller.ly,
                -self.remote_controller.lx,
                -self.remote_controller.rx,
            ],
            dtype=np.float32,
        )
        scale = np.asarray(
            self.get_parameter("joystick_command_scale").value,
            dtype=np.float32,
        )
        if scale.size != 3:
            raise ValueError(
                "joystick_command_scale must contain [forward, lateral, yaw]"
            )
        command *= scale
        return np.clip(command, -self.config.max_cmd, self.config.max_cmd)

    def _update_goal_from_remote(self) -> None:
        button_name = str(self.get_parameter("goal_set_button").value)
        button_index = self._remote_button_index(button_name)
        pressed = self.remote_controller.button[button_index] == 1
        if pressed and not self._last_goal_set_button_pressed:
            if not math.isfinite(self._last_robot_odom_time):
                self.get_logger().warn(
                    f"Remote {button_name.upper()} pressed: cannot set goal before robot odometry is available"
                )
            else:
                self._goal_condition_reached = False
                self._cube_goal_enter_time = -math.inf
                self.goal_xy[:] = self.robot_pos_xy
                goal_values = [float(self.goal_xy[0]), float(self.goal_xy[1])]
                result = self.set_parameters(
                    [
                        Parameter(
                            "goal_xy",
                            Parameter.Type.DOUBLE_ARRAY,
                            goal_values,
                        )
                    ]
                )[0]
                if not result.successful:
                    self.get_logger().warn(
                        "Goal was updated internally, but the goal_xy parameter "
                        f"update failed: {result.reason}"
                    )
                self._publish_goal_marker()
                self.get_logger().info(
                    f"Remote {button_name.upper()} pressed: goal region set "
                    f"frame={self.get_parameter('policy_world_frame').value} "
                    f"goal_xy=[{goal_values[0]:.3f}, {goal_values[1]:.3f}] "
                    f"cube_radius={float(self.get_parameter('cube_goal_stop_radius').value):.3f}m "
                    f"robot_clear_radius={float(self.get_parameter('robot_goal_clear_radius').value):.3f}m"
                )
        self._last_goal_set_button_pressed = pressed

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _cube_bearing_in_robot_frame(self) -> float:
        cube_delta = self.cube_pos_xy - self.robot_pos_xy
        cube_world_bearing = math.atan2(float(cube_delta[1]), float(cube_delta[0]))
        return self._normalize_angle(cube_world_bearing - self.robot_yaw)

    def _cube_state_is_current(self) -> bool:
        if self.fake_observations_mode or self.fake_cube_observation_mode:
            return True
        timeout_s = float(self.get_parameter("cube_state_timeout_s").value)
        return timeout_s <= 0.0 or time.monotonic() - self._last_cube_state_time <= timeout_s

    def _update_cube_recovery_toggle_from_remote(self) -> None:
        button_name = str(self.get_parameter("cube_recovery_toggle_button").value)
        button_index = self._remote_button_index(button_name)
        pressed = self.remote_controller.button[button_index] == 1
        if pressed and not self._last_cube_recovery_toggle_pressed:
            if self._cube_recovery_active:
                self._cube_recovery_active = False
                self.cmd[:] = 0.0
                if self._uses_unitree_sport_high_level():
                    self._publish_sport_stop_move()
                self.get_logger().info(
                    f"Remote {button_name.upper()} pressed: cube recovery rotation cancelled"
                )
            elif not self._cube_tracking_lost:
                self.get_logger().info(
                    f"Remote {button_name.upper()} pressed: cube recovery ignored because tracking was not lost"
                )
            elif not math.isfinite(self._last_cube_state_time):
                self.get_logger().warn(
                    f"Remote {button_name.upper()} pressed: cube recovery cannot start without a previous cube position"
                )
            else:
                bearing = self._cube_bearing_in_robot_frame()
                self._cube_recovery_direction = 1.0 if bearing >= 0.0 else -1.0
                self._cube_recovery_last_yaw = self.robot_yaw
                self._cube_recovery_rotation_rad = 0.0
                self._cube_recovery_active = True
                self._set_high_level_policy_enabled(False)
                self.get_logger().info(
                    f"Remote {button_name.upper()} pressed: starting cube recovery "
                    f"bearing_deg={math.degrees(bearing):.1f} "
                    f"direction={'left' if self._cube_recovery_direction > 0.0 else 'right'}"
                )
        self._last_cube_recovery_toggle_pressed = pressed

    def _update_cube_recovery_command(self) -> None:
        yaw_delta = self._normalize_angle(self.robot_yaw - self._cube_recovery_last_yaw)
        self._cube_recovery_rotation_rad += abs(yaw_delta)
        self._cube_recovery_last_yaw = self.robot_yaw

        if self._cube_state_is_current():
            bearing = self._cube_bearing_in_robot_frame()
            front_angle_rad = math.radians(
                abs(float(self.get_parameter("cube_recovery_front_angle_deg").value))
            )
            if abs(bearing) <= front_angle_rad:
                self._cube_recovery_active = False
                self._cube_tracking_lost = False
                self.cmd[:] = 0.0
                if bool(self.get_parameter("use_high_level_policy").value):
                    if not self._stop_high_level_if_goal_reached():
                        self._set_high_level_policy_enabled(True)
                        self.get_logger().info(
                            "Cube reacquired in front of robot; restarting high-level policy"
                        )
                else:
                    self.get_logger().info(
                        "Cube reacquired in front of robot; high-level policy remains disabled by parameter"
                    )
                return

        max_rotation_rad = math.radians(
            abs(float(self.get_parameter("cube_recovery_max_rotation_deg").value))
        )
        if self._cube_recovery_rotation_rad >= max_rotation_rad:
            self._cube_recovery_active = False
            self.cmd[:] = 0.0
            if self._uses_unitree_sport_high_level():
                self._publish_sport_stop_move()
            self.get_logger().warn(
                "Cube recovery stopped after reaching the maximum rotation "
                f"({math.degrees(self._cube_recovery_rotation_rad):.1f} deg)"
            )
            return

        angular_cmd = abs(float(self.get_parameter("cube_recovery_angular_cmd").value))
        angular_limit = abs(float(self.config.max_cmd[2]))
        self.cmd[:] = [
            0.0,
            0.0,
            self._cube_recovery_direction * min(angular_cmd, angular_limit),
        ]

    def _update_sport_select_stop_from_remote(self) -> None:
        if not self._uses_unitree_sport_high_level():
            return
        if self._remote_button_index(str(self.get_parameter("high_level_toggle_button").value)) == KeyMap.select:
            return

        pressed = self.remote_controller.button[KeyMap.select] == 1
        if pressed != self._last_select_stop_pressed:
            self.get_logger().info(
                f"Remote SELECT state changed to {int(pressed)} "
                f"key_mask=0x{self.remote_controller.keys:04x}"
            )
        if pressed and not self._last_select_stop_pressed:
            self._set_high_level_policy_enabled(False)
            self.get_logger().info(
                "Remote SELECT pressed: high-level policy disabled and Unitree Sport StopMove sent"
            )
        self._last_select_stop_pressed = pressed

    @staticmethod
    def _remote_button_index(button_name: str) -> int:
        aliases = {
            "r1": KeyMap.R1,
            "l1": KeyMap.L1,
            "start": KeyMap.start,
            "select": KeyMap.select,
            "back": KeyMap.select,
            "r2": KeyMap.R2,
            "l2": KeyMap.L2,
            "f1": KeyMap.F1,
            "f2": KeyMap.F2,
            "a": KeyMap.A,
            "b": KeyMap.B,
            "x": KeyMap.X,
            "y": KeyMap.Y,
            "up": KeyMap.up,
            "right": KeyMap.right,
            "down": KeyMap.down,
            "left": KeyMap.left,
        }
        key = button_name.strip().lower()
        if key not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(f"Unknown high-level toggle button '{button_name}'. Valid buttons: {valid}")
        return aliases[key]

    def _build_high_level_observation(
        self, ang_vel, gravity_orientation, qj_obs, dqj_obs
    ) -> np.ndarray:
        if self.fake_cube_observation_mode:
            self._apply_fake_cube_observation()
            cube_position = self.cube_pos_xy.copy()
            cube_velocity = self.cube_lin_vel_xy.copy()
            cube_stamp_s = None
        else:
            target_stamp = (
                self.get_clock().now().nanoseconds * 1.0e-9
                - float(self.get_parameter("cube_target_age_s").value)
            )
            state = select_cube_state(self._cube_state_history, target_stamp)
            if state is None:
                raise RuntimeError("No timestamped cube state is available")
            self._selected_cube_state = state
            cube_position = state.position_world_xy
            cube_velocity = state.velocity_world_xy
            cube_stamp_s = state.stamp_s
        lf_foot_xy = self._lookup_lf_foot_xy(cube_stamp_s)

        obs = build_pushcube_observation(
            angular_velocity_base=ang_vel,
            projected_gravity=gravity_orientation,
            joint_position_relative=qj_obs,
            joint_velocity=dqj_obs,
            previous_clamped_command=self.previous_clamped_command,
            robot_position_world_xy=self.robot_pos_xy,
            robot_velocity_world_xy=self.robot_lin_vel_world_xy,
            cube_position_world_xy=cube_position,
            cube_velocity_world_xy=cube_velocity,
            goal_position_world_xy=self.goal_xy,
            goal_radius=self.goal_radius,
            lf_foot_position_world_xy=lf_foot_xy,
            foot_force=self._corrected_foot_force(),
            foot_force_scale=float(self.get_parameter("foot_force_scale").value),
            quaternion_world_from_base_wxyz=np.asarray(
                self.low_state.imu_state.quaternion, dtype=np.float32
            ),
        )
        self._require_observation_size(
            obs, int(self.get_parameter("high_level_num_obs").value), "High-level"
        )
        self._append_observation_csv(obs)
        return obs

    def _corrected_foot_force(self) -> np.ndarray:
        foot_force_offset = np.asarray(
            self.get_parameter("foot_force_offset").value, dtype=np.float32
        )
        if foot_force_offset.size != 4:
            raise ValueError("foot_force_offset must contain exactly four values")
        foot_force_clip_max = float(
            self.get_parameter("foot_force_clip_max").value
        )
        if foot_force_clip_max <= 0.0:
            raise ValueError("foot_force_clip_max must be greater than zero")
        foot_force = np.asarray(self.low_state.foot_force, dtype=np.float32)
        corrected_foot_force = reorder_and_correct_foot_force(
            foot_force, foot_force_offset
        )
        return np.clip(corrected_foot_force, 0.0, foot_force_clip_max)

    def _append_observation_csv(self, observation: np.ndarray) -> None:
        path_value = str(self.get_parameter("observation_csv_path").value).strip()
        if not path_value:
            return
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(["ros_time_s", *[f"obs_{index}" for index in range(52)]])
            writer.writerow(
                [self.get_clock().now().nanoseconds * 1.0e-9, *observation.tolist()]
            )

    def _lookup_lf_foot_xy(self, stamp_s: float | None = None) -> np.ndarray:
        world_frame = str(self.get_parameter("policy_world_frame").value)
        foot_frame = str(self.get_parameter("lf_foot_frame").value)
        timeout = Duration(
            seconds=float(self.get_parameter("lf_foot_tf_timeout_s").value)
        )
        try:
            lookup_time = (
                Time()
                if stamp_s is None
                else Time(nanoseconds=int(stamp_s * 1.0e9))
            )
            transform = self.tf_buffer.lookup_transform(
                world_frame, foot_frame, lookup_time, timeout=timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"Required TF transform {world_frame} -> {foot_frame} is unavailable: {exc}"
            ) from exc

        translation = transform.transform.translation
        return np.array([translation.x, translation.y], dtype=np.float32)

    @staticmethod
    def _require_observation_size(observation, expected_size: int, label: str) -> None:
        if observation.size != expected_size:
            raise RuntimeError(
                f"{label} observation has {observation.size} values; expected {expected_size}"
            )

    def run_high_level_supervisor_step(self) -> None:
        """Compute and publish only the latency-insensitive command source."""
        self.counter += 1
        self._update_high_level_toggle_from_remote()
        self._update_goal_from_remote()
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        gravity_orientation = get_gravity_orientation(
            self.low_state.imu_state.quaternion
        )
        for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq
        qj_obs = self.qj - np.asarray(self.config.default_angles, dtype=np.float32)
        dqj_obs = self.dqj

        command_enabled = False
        if self.use_high_level_policy:
            if not self._real_cube_state_is_fresh():
                self._stop_for_stale_cube_state()
            elif not self._stop_high_level_if_goal_reached():
                self._cube_tracking_lost = False
                self._update_high_level_command(
                    ang_vel, gravity_orientation, qj_obs, dqj_obs
                )
                command_enabled = True

        self._update_cube_recovery_toggle_from_remote()
        if self._cube_recovery_active:
            self._update_cube_recovery_command()
            command_enabled = self._cube_recovery_active
        elif not self.use_high_level_policy:
            ramp_s = max(
                float(self.get_parameter("cube_stale_stop_ramp_s").value), 0.0
            )
            ramp_elapsed_s = time.monotonic() - self._stale_cube_ramp_start_time
            if math.isfinite(self._stale_cube_ramp_start_time) and ramp_elapsed_s < ramp_s:
                remaining = 1.0 - ramp_elapsed_s / ramp_s
                self.cmd[:] = self._stale_cube_ramp_start_cmd * remaining
            else:
                self._stale_cube_ramp_start_time = -math.inf
                # Manual fallback: keep the low-level locomotion policy enabled
                # and feed it the latest remote-stick command. Centered sticks
                # publish a valid zero velocity, so the policy keeps balancing.
                self.cmd[:] = self._joystick_velocity_command()
            command_enabled = True

        self._publish_high_level_command(command_enabled)
        self._publish_velocity_markers_if_due()
        self._record_policy_command_inputs(ang_vel)

    def _run_high_level_supervisor_sequence(self) -> None:
        if self.fake_observations_mode:
            self._run_fake_observation_sequence()
            return
        if self._uses_unitree_sport_high_level():
            self._run_unitree_sport_high_level_sequence()
            return
        try:
            self.wait_for_low_state()
            period_s = 1.0 / max(
                float(self.get_parameter("high_level_rate_hz").value), 1e-6
            )
            next_deadline = time.perf_counter()
            self.Liste_t = []
            while not self._stop_event.is_set():
                next_deadline += period_s
                if self.control_state == ControlState.RUNNING:
                    self.run_high_level_supervisor_step()
                remaining_s = next_deadline - time.perf_counter()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
                else:
                    self.get_logger().warn(
                        f"High-level 15 Hz deadline missed by {-remaining_s * 1000.0:.2f} ms"
                    )
                    next_deadline = time.perf_counter()
        except Exception as exc:
            self.get_logger().error(f"High-level supervisor stopped: {exc}")
        finally:
            if self.control_state == ControlState.RUNNING:
                self.cmd[:] = 0.0
                self._publish_high_level_command(False)
            self._plot_analysis_if_enabled()


    def _run_unitree_sport_high_level_sequence(self) -> None:
        try:
            self.wait_for_low_state()
            self.get_logger().info("8] --------> High-level policy is running")
            self.get_logger().info(
                "             Unitree Sport mode remains responsible for low-level locomotion"
            )
            self.get_logger().info("             ###############################################")
            self.get_logger().info("             # Press 'SELECT' to stop Sport commands       #")
            self.get_logger().info("             ###############################################")

            loop_period = 1.0 / max(
                float(self.get_parameter("sport_move_publish_rate_hz").value),
                1e-6,
            )
            self.Liste_t = []
            time_ms = 0
            next_deadline = time.perf_counter()
            while not self._stop_event.is_set():
                next_deadline += loop_period
                self.run_unitree_sport_high_level_step()
                self.Liste_t.append(time_ms)
                time_ms += int(loop_period * 1000)
                if (
                    self.remote_controller.button[KeyMap.select] == 1
                    and self._remote_button_index(
                        str(self.get_parameter("high_level_toggle_button").value)
                    )
                    != KeyMap.select
                ):
                    self.get_logger().info(
                        "Remote SELECT pressed: stopping Unitree Sport high-level loop"
                    )
                    self._publish_sport_stop_move()
                    break
                remaining_s = next_deadline - time.perf_counter()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
                else:
                    self.get_logger().warn(
                        f"Sport policy deadline missed by {-remaining_s * 1000.0:.2f} ms"
                    )
                    next_deadline = time.perf_counter()

            self._plot_analysis_if_enabled()
        except Exception as exc:
            self.get_logger().error(
                f"High-level Unitree Sport policy stopped because of an error: {exc}"
            )
            self._plot_analysis_if_enabled()
        finally:
            self._publish_sport_stop_move()

    def _run_fake_observation_sequence(self) -> None:
        try:
            self.get_logger().info(
                "Fake observation policy loop started. No DDS connection or robot commands will be used."
            )
            loop_period = 1.0 / max(
                float(self.get_parameter("high_level_rate_hz").value), 1e-6
            )
            while not self._stop_event.is_set():
                self.run_fake_policy_step()
                time.sleep(loop_period)
        except Exception as exc:
            self.get_logger().error(
                f"Fake observation policy loop stopped because of an error: {exc}"
            )

    def _plot_analysis_if_enabled(self) -> None:
        if not bool(self.get_parameter("plot_on_exit").value):
            return
        with self._analysis_plot_lock:
            if self._analysis_plot_attempted:
                return
            self._analysis_plot_attempted = True
            try:
                self._plot_analysis()
            except Exception as exc:
                if rclpy.ok():
                    self.get_logger().error(f"Failed to save analysis plot: {exc}")

    def _plot_analysis(self) -> None:
        import matplotlib.pyplot as plt

        self.get_logger().info("10] ----------> Visualizing data")
        fig = plt.figure(figsize=(40, 30))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :])
        ax1.plot(self.L_base_vel_cmd_input_1, label="L_base_vel_cmd_input_1")
        ax1.plot(self.L_base_lin_vel_input_1, label="L_base_lin_vel_input_1")
        ax1.plot(self.L_base_lin_vel_kalman_input_1, label="L_base_lin_vel_kalman_input_1")
        ax1.legend()
        ax1.set_title("Vx")
        ax2.plot(self.L_base_vel_cmd_input_2, label="L_base_vel_cmd_input_2")
        ax2.plot(self.L_base_lin_vel_input_2, label="L_base_lin_vel_input_2")
        ax2.plot(self.L_base_lin_vel_kalman_input_2, label="L_base_lin_vel_kalman_input_2")
        ax2.legend()
        ax2.set_title("Vy")
        ax3.plot(self.L_base_lin_vel_input_3, label="L_base_lin_vel_input_3")
        ax3.plot(self.L_base_lin_vel_kalman_input_3, label="L_base_lin_vel_kalman_input_3")
        ax3.legend()
        ax3.set_title("Vz")
        ax4.plot(self.L_base_vel_cmd_input_3, label="L_base_vel_cmd_input_3")
        ax4.plot(self.L_base_ang_vel_input_1, label="L_base_ang_vel_input_1")
        ax4.plot(self.L_base_lin_vel_kalman_input_4, label="L_base_lin_vel_kalman_input_4")
        ax4.legend()
        ax4.set_title("Wz")
        ax5.plot(self.L_foot_force_fr, label="FR")
        ax5.plot(self.L_foot_force_fl, label="FL")
        ax5.plot(self.L_foot_force_rr, label="RR")
        ax5.plot(self.L_foot_force_rl, label="RL")
        ax5.legend()
        ax5.set_title("Foot force sensors")
        ax5.set_xlabel("Policy step")
        ax5.set_ylabel("Raw sensor value")
        plt.tight_layout()
        output_path = Path(
            str(self.get_parameter("analysis_pdf_path").value)
        ).expanduser()
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        try:
            fig.savefig(output_path)
        finally:
            plt.close(fig)
        print(f"Analysis visualization finished: {output_path}", flush=True)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._uses_unitree_sport_high_level():
            self._publish_sport_stop_move()
        self._plot_analysis_if_enabled()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HighLevelPolicyNode()
    try:
        while rclpy.ok() and not node._shutdown_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        if "Unable to convert call argument to Python object" not in str(exc):
            raise
        if rclpy.ok():
            node.get_logger().warn(
                "Ignoring ROS message conversion error during shutdown."
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("High-level policy node shutdown completed cleanly.", flush=True)


if __name__ == "__main__":
    main()
