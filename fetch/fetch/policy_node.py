#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""ROS 2 node version of Deploy_SimToReal_RL_Go2/deploy_real."""

from __future__ import annotations

import math
import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point
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
    DeployRealConfig,
    KeyMap,
    RemoteController,
    compute_go2_lowcmd_crc,
    create_zero_cmd,
    get_gravity_orientation,
    init_cmd_go,
    set_motor_cmd_gains,
    set_motor_cmd_position,
    set_motor_cmd_torque,
    set_motor_cmd_velocity,
)

CONTROL_MODE_HIERARCHICAL_LOWCMD = "hierarchical_lowcmd"
CONTROL_MODE_UNITREE_SPORT_HIGH_LEVEL = "unitree_sport_high_level"
SPORT_API_ID_STOPMOVE = 1003
SPORT_API_ID_MOVE = 1008


class RealPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self._declare_parameters()
        self.config = self._config_from_parameters()
        self.control_mode = self._control_mode_from_parameter()
        # self.project_root = Path(self.get_parameter("project_root").value).expanduser()
        # sdk_paths = self._array_parameter(
        #     "unitree_sdk_paths",
        #     Parameter.Type.STRING_ARRAY,
        # )
        # sdk_paths.insert(0, str(self.project_root))
        # add_unitree_sdk_paths(sdk_paths)

        self.remote_controller = RemoteController()
        self.base_lin_vel_input = [0, 0, 0, 0]
        self.low_state_msg = None
        self.lowcmd_publisher_ = None
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
            worker_target = (
                self._run_fake_observation_sequence
                if self.fake_observations_mode
                else self._run_deploy_sequence
            )
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
        self.declare_parameter("auto_switch_to_low_level", False)
        self.declare_parameter("send_commands", False)
        self.declare_parameter("use_high_level_policy", True)
        self.declare_parameter("wait_for_start_button", True)
        self.declare_parameter("wait_for_a_button", True)

        # DDS connection.
        self.declare_parameter("network_interface", "")
        self.declare_parameter("dds_domain_id", 0)

        # Remote controls.
        self.declare_parameter("high_level_toggle_button", "X")
        self.declare_parameter("goal_set_button", "Y")
        self.declare_parameter("cube_recovery_toggle_button", "B")

        # Goal definition and completion condition.
        self.declare_parameter("goal_xy", [0.0, 0.0])
        self.declare_parameter("goal_radius", 0.2)
        self.declare_parameter("cube_goal_stop_radius", 0.3)
        self.declare_parameter("cube_goal_hold_s", 0.6)
        self.declare_parameter("robot_goal_clear_radius", 0.35)

        # Cube-loss recovery rotation.
        self.declare_parameter("cube_state_timeout_s", 0.5)
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
        self.declare_parameter("kalman_odom_topic", "/odometry/filtered")
        self.declare_parameter("cube_state_topic", "/go2_fetch/cube_state")
        self.declare_parameter("inekf_lowstate_topic", "/inekf_lowstate")
        self.declare_parameter("lowcmd_topic", "/lowcmd")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("sportstate_topic", "/sportmodestate")
        self.declare_parameter("sport_request_topic", "/api/sport/request")
        self.declare_parameter("policy_world_frame", "odom")
        self.declare_parameter("lf_foot_frame", "FL_foot")
        self.declare_parameter("lf_foot_tf_timeout_s", 0.02)
        self.declare_parameter("cube_state_tf_timeout_s", 0.05)
        self.declare_parameter("robot_twist_in_body_frame", True)
        self.declare_parameter("lowstate_publish_period_s", 0.005)

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

        # Policy models and inference rates.
        self.declare_parameter(
            "high_level_policy_path",
            "logs/rsl_rl/unitree_go2_pushcube_4l/2026-05-15_02-52-05_cam_6/exported/policy.pt",
        )
        self.declare_parameter(
            "low_level_policy_path",
            "logs/rsl_rl/unitree_go2_velocity_4l/2026-04-05_12-01-56_walk_2/exported/policy.pt",
        )
        self.declare_parameter("model_loop_period_s", 0.02)
        self.declare_parameter("high_level_rate_hz", 15.0)
        self.declare_parameter("high_level_num_obs", 52)
        self.declare_parameter("low_level_num_obs", 49)

        # Unitree Sport high-level control.
        self.declare_parameter("sport_move_publish_rate_hz", 15.0)
        self.declare_parameter("sport_stop_on_disable", True)
        self.declare_parameter("sport_command_log_every_n_steps", 50)
        self.declare_parameter("sport_command_scale", [-1.0, 1.0, 1.0])

        # Low-level policy interface and scaling.
        self.declare_parameter("control_dt", 0.005)
        self.declare_parameter("msg_type", "go")
        self.declare_parameter("imu_type", "pelvis")
        self.declare_parameter("ang_vel_scale", 0.2)
        self.declare_parameter("dof_pos_scale", 1.0)
        self.declare_parameter("dof_vel_scale", 0.05)
        self.declare_parameter("action_scale", 0.25)
        self.declare_parameter("cmd_scale", [0.8, 0.8, 1.0])
        self.declare_parameter("num_actions", 12)
        self.declare_parameter("num_obs", 45)
        self.declare_parameter("max_cmd", [1.0, 1.0, 1.0])

        # Leg motor mapping, gains, and limits.
        self.declare_parameter("leg_joint2motor_idx", [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
        self.declare_parameter("default_angles", [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
        self.declare_parameter("kps", [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        self.declare_parameter("kds", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.declare_parameter("torque_limits", [23.7, 23.7, 35.55, 23.7, 23.7, 35.55, 23.7, 23.7, 35.55, 23.7, 23.7, 35.55])
        self.declare_parameter("weak_motor", Parameter.Type.INTEGER_ARRAY)

        # Optional arm and waist motors.
        self.declare_parameter(
            "arm_waist_joint2motor_idx",
            Parameter.Type.INTEGER_ARRAY,
        )
        self.declare_parameter(
            "arm_waist_kps",
            Parameter.Type.DOUBLE_ARRAY,
        )
        self.declare_parameter(
            "arm_waist_kds",
            Parameter.Type.DOUBLE_ARRAY,
        )
        self.declare_parameter(
            "arm_waist_target",
            Parameter.Type.DOUBLE_ARRAY,
        )
        self.declare_parameter("foot_force_offset", [4.0, 0.0, 5.0, 5.0])
        self.declare_parameter("foot_force_clip_max", 150.0)
        self.declare_parameter("foot_force_scale", 100.0)

        # Diagnostics and analysis output.
        self.declare_parameter("startup_sleep_s", 0.001)
        self.declare_parameter("motor_log_every_n_steps", 50)
        self.declare_parameter("cycle_timing_log_every_n_steps", 50)
        self.declare_parameter("profile_timing_every_n_steps", 50)
        self.declare_parameter("torque_limit_log_period_s", 1.0)
        self.declare_parameter("high_level_command_log_period_s", 1.0)
        self.declare_parameter("plot_on_exit", False)
        self.declare_parameter("analysis_pdf_path", "analyse_robot.png")

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

    def _array_parameter(
        self,
        name: str,
        parameter_type: Parameter.Type,
        default: list | None = None,
    ) -> list:
        fallback = Parameter(name, parameter_type, [] if default is None else list(default))
        return list(self.get_parameter_or(name, fallback).value)

    def _config_from_parameters(self) -> DeployRealConfig:
        return DeployRealConfig(
            control_dt=float(self.get_parameter("control_dt").value),
            msg_type=str(self.get_parameter("msg_type").value),
            imu_type=str(self.get_parameter("imu_type").value),
            weak_motor=self._array_parameter("weak_motor", Parameter.Type.INTEGER_ARRAY),
            lowcmd_topic=str(self.get_parameter("lowcmd_topic").value),
            lowstate_topic=str(self.get_parameter("lowstate_topic").value),
            leg_joint2motor_idx=list(self.get_parameter("leg_joint2motor_idx").value),
            kps=list(self.get_parameter("kps").value),
            kds=list(self.get_parameter("kds").value),
            torque_limits=list(self.get_parameter("torque_limits").value),
            default_angles=np.array(self.get_parameter("default_angles").value, dtype=np.float32),
            arm_waist_joint2motor_idx=self._array_parameter(
                "arm_waist_joint2motor_idx",
                Parameter.Type.INTEGER_ARRAY,
            ),
            arm_waist_kps=self._array_parameter(
                "arm_waist_kps",
                Parameter.Type.DOUBLE_ARRAY,
            ),
            arm_waist_kds=self._array_parameter(
                "arm_waist_kds",
                Parameter.Type.DOUBLE_ARRAY,
            ),
            arm_waist_target=np.array(
                self._array_parameter("arm_waist_target", Parameter.Type.DOUBLE_ARRAY),
                dtype=np.float32,
            ),
            ang_vel_scale=float(self.get_parameter("ang_vel_scale").value),
            dof_pos_scale=float(self.get_parameter("dof_pos_scale").value),
            dof_vel_scale=float(self.get_parameter("dof_vel_scale").value),
            action_scale=float(self.get_parameter("action_scale").value),
            cmd_scale=np.array(self.get_parameter("cmd_scale").value, dtype=np.float32),
            max_cmd=np.array(self.get_parameter("max_cmd").value, dtype=np.float32),
            num_actions=int(self.get_parameter("num_actions").value),
            num_obs=int(self.get_parameter("num_obs").value),
        )

    def _load_ros_interfaces(self) -> None:
        try:
            from unitree_go.msg import LowCmd as LowCmdRos
            from unitree_go.msg import LowState as LowStateRos
            from unitree_go.msg import SportModeState as SportModeStateRos

            self.LowCmdRos = LowCmdRos
            self.LowStateRos = LowStateRos
            self.SportModeStateRos = SportModeStateRos
            if self._uses_unitree_sport_high_level():
                from unitree_api.msg import Request as SportRequestRos

                self.SportRequestRos = SportRequestRos
        except Exception as exc:
            self.LowCmdRos = None
            self.LowStateRos = None
            self.SportModeStateRos = None
            self.SportRequestRos = None
            raise RuntimeError(f"unitree_go ROS messages are required in real mode: {exc}") from exc

    def _initialize_ros_io(self) -> None:
        self.create_subscription(Odometry, self.get_parameter("kalman_odom_topic").value, self._kalman_odom_callback, 10)
        self.create_subscription(Odometry, self.get_parameter("cube_state_topic").value, self._cube_state_callback, 10)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.lowstate_publisher = self.create_publisher(
            self.LowStateRos,
            self.get_parameter("inekf_lowstate_topic").value,
            10,
        )

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
        if self._uses_unitree_sport_high_level():
            self.get_logger().info("3] ---> Loading high-level policy")
        else:
            self.get_logger().info("3] ---> Loading high-level and low-level policies")
        high_level_path = self._resolve_policy_path("high_level_policy_path")
        self.high_level_policy = torch.jit.load(high_level_path).eval()
        self.low_level_policy = None
        if not self._uses_unitree_sport_high_level():
            low_level_path = self._resolve_policy_path("low_level_policy_path")
            self.low_level_policy = torch.jit.load(low_level_path).eval()

        self.default_isaac = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        self.base_lin_vel = np.array([0, 0, 0])
        self.cmd = np.zeros(3, dtype=np.float32)
        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.high_level_action = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_isaac.copy()
        self.high_level_obs = np.zeros(
            int(self.get_parameter("high_level_num_obs").value), dtype=np.float32
        )
        self.obs = np.zeros(
            int(self.get_parameter("low_level_num_obs").value), dtype=np.float32
        )
        self.robot_pos_xy = np.zeros(2, dtype=np.float32)
        self.robot_lin_vel_world_xy = np.zeros(2, dtype=np.float32)
        self.robot_yaw = 0.0
        self.cube_pos_xy = np.zeros(2, dtype=np.float32)
        self.cube_lin_vel_xy = np.zeros(2, dtype=np.float32)
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
        self._last_high_level_toggle_pressed = False
        self._last_goal_set_button_pressed = False
        self._last_cube_recovery_toggle_pressed = False
        self._cube_tracking_lost = False
        self._cube_recovery_active = False
        self._cube_recovery_direction = 1.0
        self._cube_recovery_last_yaw = self.robot_yaw
        self._cube_recovery_rotation_rad = 0.0
        self._last_select_stop_pressed = False
        self._sport_stop_sent = False
        self._last_high_level_command_log_time = -math.inf
        self._last_torque_limit_log_time = -math.inf
        self._torque_limit_events = {}
        self.policy_cycle_times_ms = []
        self._last_send_cmd_timing_ms = (0.0, 0.0, 0.0)
        self._analysis_plot_lock = threading.Lock()
        self._analysis_plot_attempted = False

        self.dt = 0.002
        self.startPos = [0.0] * 12
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.firstRun = True
        self.counter = 0

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1, -1.5, 0.1, 1, -1.5]

        window_size = 20
        self.vx_window = [0] * window_size
        self.vy_window = [0] * window_size
        self.vz_window = [0] * window_size

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
        return path

    def _initialize_robot_interfaces(self) -> None:
        self.get_logger().info("4] ----> Initializing ROS 2 Unitree topics")
        self.lowcmd_publisher_ = None
        self.sport_request_publisher = None
        if self._uses_unitree_sport_high_level():
            self.sport_request_publisher = self.create_publisher(
                self.SportRequestRos,
                self.get_parameter("sport_request_topic").value,
                10,
            )
        else:
            self.lowcmd_publisher_ = self.create_publisher(
                self.LowCmdRos,
                self.config.lowcmd_topic,
                10,
            )
        self.lowstate_subscriber = self.create_subscription(
            self.LowStateRos,
            self.config.lowstate_topic,
            self._low_state_go_handler,
            10,
        )
        self.sportstate_subscriber = self.create_subscription(
            self.SportModeStateRos,
            self.get_parameter("sportstate_topic").value,
            self._sport_state_message_handler,
            10,
        )

        self.low_cmd = None
        self.low_state = self.LowStateRos()
        if not self._uses_unitree_sport_high_level():
            self.low_cmd = self.LowCmdRos()
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def init_low_level_mode(self) -> None:
        if not bool(self.get_parameter("auto_switch_to_low_level").value):
            return

        raise RuntimeError(
            "auto_switch_to_low_level is unsupported in ROS-topic mode; "
            "set it to false and switch the robot with a separate Unitree tool"
        )

    def wait_for_low_state(self) -> None:
        while self.low_state.tick == 0 and not self._stop_event.is_set():
            time.sleep(self.config.control_dt)
        self.get_logger().info("         Connected to robot")

    def _low_state_go_handler(self, msg) -> None:
        self.low_state = msg
        self.low_state_msg = msg
        self.remote_controller.set(self.low_state.wireless_remote)
        self.lowstate_publisher.publish(msg)

    def _sport_state_message_handler(self, sport_state_msg) -> None:
        self.velocity = sport_state_msg.velocity

    def _kalman_odom_callback(self, msg: Odometry) -> None:
        self._last_robot_odom_time = time.monotonic()
        self.base_lin_vel_input[0] = msg.twist.twist.linear.x
        self.base_lin_vel_input[1] = msg.twist.twist.linear.y
        self.base_lin_vel_input[2] = msg.twist.twist.linear.z
        self.base_lin_vel_input[3] = msg.twist.twist.angular.z
        self.robot_pos_xy[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        orientation = msg.pose.pose.orientation
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
        transformed_state = self._transform_cube_state_to_policy_frame(msg)
        if transformed_state is None:
            return
        pos_xy, vel_xy = transformed_state
        self.cube_pos_xy[:] = pos_xy
        self.cube_lin_vel_xy[:] = vel_xy
        self._last_cube_state_time = time.monotonic()
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
                target_frame, source_frame, Time(), timeout=timeout
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
        rotated_vel = np.array(
            [
                cos_yaw * vel_xy[0] - sin_yaw * vel_xy[1],
                sin_yaw * vel_xy[0] + cos_yaw * vel_xy[1],
            ],
            dtype=np.float32,
        )
        rotated_pos += np.array([translation.x, translation.y], dtype=np.float32)
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
        marker.pose.position.z = 0.01
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = 0.02
        marker.color.r = 0.0 if self._goal_condition_reached else 1.0
        marker.color.g = 1.0 if self._goal_condition_reached else 0.0
        marker.color.b = 0.0
        marker.color.a = 0.45
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

    def _xy_parameter(self, name: str) -> np.ndarray:
        values = list(self.get_parameter(name).value)
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values [x, y]")
        return np.array(values, dtype=np.float32)

    def _publish_lowstate(self) -> None:
        if self.lowstate_publisher is None or self.low_state_msg is None:
            return
        self.lowstate_publisher.publish(self.low_state_msg)

    def _apply_torque_limits(self, cmd) -> None:
        limits = getattr(self.config, "torque_limits", None)
        if limits is None:
            return
        for i, limit in enumerate(limits):
            if i >= len(cmd.motor_cmd):
                break
            limit = abs(float(limit))
            self._limit_motor_command_torque(cmd.motor_cmd[i], i, limit)

    def _limit_motor_command_torque(self, motor_cmd, motor_index: int, limit: float) -> None:
        if self.low_state is None or motor_index >= len(self.low_state.motor_state):
            requested_torque = float(motor_cmd.tau)
            clamped_torque = float(np.clip(requested_torque, -limit, limit))
            if clamped_torque != requested_torque:
                self._record_torque_limit(
                    motor_index, requested_torque, clamped_torque
                )
            set_motor_cmd_torque(motor_cmd, clamped_torque)
            return

        motor_state = self.low_state.motor_state[motor_index]
        q_actual = float(motor_state.q)
        dq_actual = float(motor_state.dq)
        q_target = float(motor_cmd.q)
        dq_target = float(getattr(motor_cmd, "dq", getattr(motor_cmd, "qd", 0.0)))
        kp = float(motor_cmd.kp)
        kd = float(motor_cmd.kd)
        tau = float(motor_cmd.tau)

        estimated_torque = (
            kp * (q_target - q_actual)
            + kd * (dq_target - dq_actual)
            + tau
        )
        clamped_torque = float(np.clip(estimated_torque, -limit, limit))
        if clamped_torque == estimated_torque:
            return

        self._record_torque_limit(motor_index, estimated_torque, clamped_torque)

        # Prefer changing q because this node uses position control. If kp is
        # zero, fall back to dq, then tau.
        if abs(kp) > 1e-6:
            q_limited = q_actual + (
                clamped_torque - kd * (dq_target - dq_actual) - tau
            ) / kp
            set_motor_cmd_position(motor_cmd, q_limited)
        elif abs(kd) > 1e-6:
            dq_limited = dq_actual + (clamped_torque - tau) / kd
            set_motor_cmd_velocity(motor_cmd, dq_limited)
        else:
            set_motor_cmd_torque(motor_cmd, clamped_torque)

    def _record_torque_limit(
        self, motor_index: int, requested_torque: float, clamped_torque: float
    ) -> None:
        previous = self._torque_limit_events.get(motor_index)
        hit_count = 1 if previous is None else previous[2] + 1
        self._torque_limit_events[motor_index] = (
            requested_torque,
            clamped_torque,
            hit_count,
        )

        now = time.monotonic()
        log_period = max(
            0.0, float(self.get_parameter("torque_limit_log_period_s").value)
        )
        if now - self._last_torque_limit_log_time < log_period:
            return

        event_text = "; ".join(
            f"motor {index}: {requested:.2f} -> {clamped:.2f} Nm "
            f"({count} hits)"
            for index, (requested, clamped, count) in sorted(
                self._torque_limit_events.items()
            )
        )
        print(f"\033[38;5;208mTorque limits reached: {event_text}\033[0m")
        self._torque_limit_events.clear()
        self._last_torque_limit_log_time = now

    def send_cmd(self, cmd) -> None:
        if not self.commands_enabled:
            return
        if self.lowcmd_publisher_ is None:
            return
        timing_start = time.perf_counter()
        self._apply_torque_limits(cmd)
        torque_limits_done = time.perf_counter()
        cmd.crc = compute_go2_lowcmd_crc(cmd)
        crc_done = time.perf_counter()
        self.lowcmd_publisher_.publish(cmd)
        publish_done = time.perf_counter()
        self._last_send_cmd_timing_ms = (
            (torque_limits_done - timing_start) * 1000.0,
            (crc_done - torque_limits_done) * 1000.0,
            (publish_done - crc_done) * 1000.0,
        )

    def _send_safe_hold_command(self) -> None:
        if (
            not self.commands_enabled
            or self._uses_unitree_sport_high_level()
            or self.lowcmd_publisher_ is None
            or self.low_cmd is None
            or self.low_state is None
            or not rclpy.ok()
        ):
            return

        try:
            for motor_idx in range(12):
                q_actual = float(self.low_state.motor_state[motor_idx].q)
                if not math.isfinite(q_actual):
                    return
                set_motor_cmd_position(self.low_cmd.motor_cmd[motor_idx], q_actual)
                set_motor_cmd_velocity(self.low_cmd.motor_cmd[motor_idx], 0.0)
                set_motor_cmd_gains(self.low_cmd.motor_cmd[motor_idx], 20.0, 3.0)
                set_motor_cmd_torque(self.low_cmd.motor_cmd[motor_idx], 0.0)
            self.send_cmd(self.low_cmd)
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warn(f"Failed to publish safe hold command: {exc}")

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
        self.use_high_level_policy = bool(enabled)
        self._next_high_level_time = -math.inf
        if self.use_high_level_policy:
            self._cube_recovery_active = False
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
        self._set_high_level_policy_enabled(False)
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

    def _log_policy_motor_outputs(self) -> None:
        log_every = int(self.get_parameter("motor_log_every_n_steps").value)
        if log_every <= 0 or self.counter % log_every != 0:
            return

        motor_indices = self.config.leg_joint2motor_idx
        q_targets = np.array(
            [self.low_cmd.motor_cmd[motor_idx].q for motor_idx in motor_indices],
            dtype=np.float32,
        )
        kp_values = np.array(
            [self.low_cmd.motor_cmd[motor_idx].kp for motor_idx in motor_indices],
            dtype=np.float32,
        )
        kd_values = np.array(
            [self.low_cmd.motor_cmd[motor_idx].kd for motor_idx in motor_indices],
            dtype=np.float32,
        )
        tau_values = np.array(
            [self.low_cmd.motor_cmd[motor_idx].tau for motor_idx in motor_indices],
            dtype=np.float32,
        )
        self.get_logger().info(
            "Policy motor output "
            f"step={self.counter} "
            # f"cmd={np.array2string(self.cmd, precision=3, suppress_small=True)} "
            # f"action={np.array2string(self.action, precision=3, suppress_small=True)} "
            # f"q_target={np.array2string(q_targets, precision=3, suppress_small=True)} "
            # f"kp={np.array2string(kp_values, precision=2, suppress_small=True)} "
            # f"kd={np.array2string(kd_values, precision=2, suppress_small=True)} "
            # f"tau={np.array2string(tau_values, precision=3, suppress_small=True)}"
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
            with torch.inference_mode():
                self.high_level_action = (
                    self.high_level_policy(torch.from_numpy(self.high_level_obs).unsqueeze(0))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
        else:
            self.high_level_action = self._fake_rng.uniform(
                observation_min, observation_max, size=3
            ).astype(np.float32)
        self.cmd[:] = np.clip(
            self.high_level_action[:3],
            -self.config.max_cmd,
            self.config.max_cmd,
        )
        self._publish_command_velocity_marker()
        self._publish_current_velocity_marker()

        if self._uses_unitree_sport_high_level():
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

        self.obs = self._fake_rng.uniform(
            observation_min,
            observation_max,
            size=int(self.get_parameter("low_level_num_obs").value),
        ).astype(np.float32)
        self.obs[6:9] = self.high_level_action[:3]
        with torch.inference_mode():
            self.action = (
                self.low_level_policy(torch.from_numpy(self.obs).unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        log_every = max(
            1,
            int(self.get_parameter("fake_log_every_n_steps").value),
        )
        if self.counter % log_every == 0:
            self.get_logger().info(
                f"Fake policy step {self.counter}: "
                f"high_obs={self.high_level_obs.size} high_action={self.high_level_action[:3]} "
                f"low_obs={self.obs.size} low_action=[{self.action.min():.3f}, {self.action.max():.3f}] "
                "commands_sent=false"
            )

        return self.obs

    def zero_torque_state(self) -> None:
        self.get_logger().info("5] -----> Zero-torque state is active")
        self.get_logger().info("          ##################################################")
        self.get_logger().info("          # Waiting for START button to stand up robot     #")
        self.get_logger().info("          ##################################################")
        while (
            bool(self.get_parameter("wait_for_start_button").value)
            and self.remote_controller.button[KeyMap.start] != 1
            and not self._stop_event.is_set()
        ):
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self) -> None:
        self.get_logger().info("6] ------> Robot is moving to the default pose")
        done = False

        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

        self.count = 0
        while not done and not self._stop_event.is_set():
            self.count += 1
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            if self.percent_1 < 1:
                for i in range(12):
                    set_motor_cmd_position(
                        self.low_cmd.motor_cmd[i],
                        (1 - self.percent_1) * self.startPos[i]
                        + self.percent_1 * self._targetPos_1[i],
                    )
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    set_motor_cmd_gains(self.low_cmd.motor_cmd[i], 60.0, 5.0)
                    set_motor_cmd_torque(self.low_cmd.motor_cmd[i], 0.0)

            if self.percent_1 == 1 and self.percent_2 <= 1:
                self.percent_2 += 1.0 / self.duration_2
                self.percent_2 = min(self.percent_2, 1)
                for i in range(12):
                    set_motor_cmd_position(
                        self.low_cmd.motor_cmd[i],
                        (1 - self.percent_2) * self._targetPos_1[i]
                        + self.percent_2 * self._targetPos_2[i],
                    )
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    set_motor_cmd_gains(self.low_cmd.motor_cmd[i], 60.0, 5.0)
                    set_motor_cmd_torque(self.low_cmd.motor_cmd[i], 0.0)

            if self.percent_1 == 1 and self.percent_2 == 1 and self.percent_3 < 1:
                self.percent_3 += 1.0 / self.duration_3
                self.percent_3 = min(self.percent_3, 1)
                for i in range(12):
                    set_motor_cmd_position(
                        self.low_cmd.motor_cmd[i], self._targetPos_2[i]
                    )
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    set_motor_cmd_gains(self.low_cmd.motor_cmd[i], 60.0, 5.0)
                    set_motor_cmd_torque(self.low_cmd.motor_cmd[i], 0.0)

            self.send_cmd(self.low_cmd)
            if self.percent_3 == 1.0 or self.count == 2500000000:
                done = True
            time.sleep(float(self.get_parameter("startup_sleep_s").value))

        self.get_logger().info("7] -------> Robot is standing")
        self.get_logger().info("            ###########################################")
        self.get_logger().info("            # Press 'A' to start the model            #")
        self.get_logger().info("            ###########################################")

        while (
            bool(self.get_parameter("wait_for_a_button").value)
            and self.remote_controller.button[KeyMap.A] != 1
            and not self._stop_event.is_set()
        ):
            default = self.config.default_angles
            for policy_idx, motor_idx in enumerate(self.config.leg_joint2motor_idx):
                set_motor_cmd_position(
                    self.low_cmd.motor_cmd[motor_idx], default[policy_idx]
                )
                set_motor_cmd_velocity(self.low_cmd.motor_cmd[motor_idx], 0)
                set_motor_cmd_gains(self.low_cmd.motor_cmd[motor_idx], 60.0, 5.0)
                set_motor_cmd_torque(self.low_cmd.motor_cmd[motor_idx], 0.0)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_ground(self) -> None:
        percent = 0
        pos_init = []
        for k in range(12):
            pos_init.append(self.low_state.motor_state[k].q)
        while percent != 1 and not self._stop_event.is_set():
            percent += 1.0 / 300
            percent = min(percent, 1)
            lying_pose = [0, 1.36, -2.65, 0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
            for i in range(12):
                set_motor_cmd_position(
                    self.low_cmd.motor_cmd[i],
                    (1 - percent) * pos_init[i] + percent * lying_pose[i],
                )
                set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                set_motor_cmd_gains(self.low_cmd.motor_cmd[i], 60.0, 5.0)
                set_motor_cmd_torque(self.low_cmd.motor_cmd[i], 0.0)
            self.send_cmd(self.low_cmd)
            time.sleep(0.002)
        self.get_logger().info("9] ---------> Robot is lying down")

    def run_policy_step(self):
        step_start = time.perf_counter()
        self.counter += 1
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
        default_joint = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        qj_obs = qj_obs - default_joint
        input_done = time.perf_counter()

        if self.use_high_level_policy:
            if not self._real_cube_state_is_fresh():
                self._stop_for_stale_cube_state()
            elif self._stop_high_level_if_goal_reached():
                pass
            else:
                self._cube_tracking_lost = False
                self._update_high_level_command(ang_vel, gravity_orientation, qj_obs, dqj_obs)
        self._update_cube_recovery_toggle_from_remote()
        if self._cube_recovery_active:
            self._update_cube_recovery_command()
        elif not self.use_high_level_policy:
            self.cmd[:] = self._joystick_velocity_command()
        command_done = time.perf_counter()

        self._publish_command_velocity_marker()
        self._publish_current_velocity_marker()
        markers_done = time.perf_counter()

        self.obs = np.concatenate(
            [
                ang_vel * self.config.ang_vel_scale,
                gravity_orientation,
                self.cmd,
                qj_obs * self.config.dof_pos_scale,
                dqj_obs * self.config.dof_vel_scale,
                self._scaled_foot_force(),
                self.action,
            ]
        ).astype(np.float32)
        self._require_observation_size(
            self.obs, int(self.get_parameter("low_level_num_obs").value), "Low-level"
        )
        observation_done = time.perf_counter()
        with torch.inference_mode():
            self.action = (
                self.low_level_policy(torch.from_numpy(self.obs).unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        inference_done = time.perf_counter()
        if self.action.size != self.config.num_actions:
            raise RuntimeError(
                f"Low-level policy returned {self.action.size} actions; expected {self.config.num_actions}"
            )

        target_dof_pos = self.action * self.config.action_scale + default_joint
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            set_motor_cmd_position(
                self.low_cmd.motor_cmd[motor_idx], target_dof_pos[i]
            )
            set_motor_cmd_velocity(self.low_cmd.motor_cmd[motor_idx], 0)
            set_motor_cmd_gains(
                self.low_cmd.motor_cmd[motor_idx],
                self.config.kps[i],
                self.config.kds[i],
            )
            set_motor_cmd_torque(self.low_cmd.motor_cmd[motor_idx], 0.0)
        motor_command_done = time.perf_counter()
        self.send_cmd(self.low_cmd)
        send_done = time.perf_counter()
        self._log_policy_motor_outputs()

        self._record_policy_command_inputs(ang_vel)
        record_done = time.perf_counter()

        profile_every = int(
            self.get_parameter("profile_timing_every_n_steps").value
        )
        if profile_every > 0 and self.counter % profile_every == 0:
            torque_ms, crc_ms, publish_ms = self._last_send_cmd_timing_ms
            print(
                "Policy step profile "
                f"step={self.counter} "
                f"input={(input_done - step_start) * 1000.0:.2f}ms "
                f"command={(command_done - input_done) * 1000.0:.2f}ms "
                f"markers={(markers_done - command_done) * 1000.0:.2f}ms "
                f"observation={(observation_done - markers_done) * 1000.0:.2f}ms "
                f"inference={(inference_done - observation_done) * 1000.0:.2f}ms "
                f"motor_build={(motor_command_done - inference_done) * 1000.0:.2f}ms "
                f"send_total={(send_done - motor_command_done) * 1000.0:.2f}ms "
                f"torque_limit={torque_ms:.2f}ms "
                f"crc={crc_ms:.2f}ms "
                f"publish={publish_ms:.2f}ms "
                f"record={(record_done - send_done) * 1000.0:.2f}ms "
                f"total={(record_done - step_start) * 1000.0:.2f}ms",
                flush=True,
            )

        return self.obs

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
        default_joint = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        qj_obs = qj_obs - default_joint

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

        self._publish_command_velocity_marker()
        self._publish_current_velocity_marker()
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
            with torch.inference_mode():
                self.high_level_action = (
                    self.high_level_policy(
                        torch.from_numpy(self.high_level_obs).unsqueeze(0)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
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

    def _joystick_velocity_command(self) -> np.ndarray:
        command = np.array(
            [
                self.remote_controller.ly,
                -self.remote_controller.lx,
                -self.remote_controller.rx,
            ],
            dtype=np.float32,
        )
        command *= self.config.cmd_scale
        return np.clip(command, -self.config.max_cmd, self.config.max_cmd)

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
        lf_foot_xy = self._lookup_lf_foot_xy()
        if self.fake_cube_observation_mode:
            self._apply_fake_cube_observation()
        obs = np.concatenate(
            [
                np.asarray(ang_vel, dtype=np.float32) * 0.2,
                np.asarray(gravity_orientation, dtype=np.float32),
                qj_obs,
                dqj_obs * 0.05,
                self.high_level_action,
                self.robot_pos_xy,
                self.robot_lin_vel_world_xy,
                self.cube_pos_xy,
                self.cube_lin_vel_xy,
                self.goal_xy,
                np.array([self.goal_radius], dtype=np.float32),
                self.goal_xy - self.cube_pos_xy,
                self.cube_pos_xy - lf_foot_xy,
                self._scaled_foot_force(),
            ]
        ).astype(np.float32)
        self._require_observation_size(
            obs, int(self.get_parameter("high_level_num_obs").value), "High-level"
        )
        return obs

    def _scaled_foot_force(self) -> np.ndarray:
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
        foot_force_scale = float(self.get_parameter("foot_force_scale").value)
        if foot_force_scale <= 0.0:
            raise ValueError("foot_force_scale must be greater than zero")
        foot_force = np.asarray(self.low_state.foot_force, dtype=np.float32)
        corrected_foot_force = np.clip(
            foot_force - foot_force_offset,
            0.0,
            foot_force_clip_max,
        )
        # Unitree: [FR, FL, RR, RL]; Isaac Lab policy: [FL, FR, RL, RR].
        return corrected_foot_force[[1, 0, 3, 2]] / foot_force_scale

    def _lookup_lf_foot_xy(self) -> np.ndarray:
        world_frame = str(self.get_parameter("policy_world_frame").value)
        foot_frame = str(self.get_parameter("lf_foot_frame").value)
        timeout = Duration(
            seconds=float(self.get_parameter("lf_foot_tf_timeout_s").value)
        )
        try:
            transform = self.tf_buffer.lookup_transform(
                world_frame, foot_frame, Time(), timeout=timeout
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

    def _update_velocity_window(self, window, value):
        temp = 0
        for k in range(len(window)):
            temp += window[k]
        temp += value * 5
        temp = temp / (len(window) + 5)
        for k in range(len(window) - 1):
            window[k] = window[k + 1]
        window[len(window) - 1] = temp
        return temp

    def _run_deploy_sequence(self) -> None:
        if self._uses_unitree_sport_high_level():
            self._run_unitree_sport_high_level_sequence()
            return

        try:
            self.init_low_level_mode()
            self.wait_for_low_state()
            self.zero_torque_state()
            self.move_to_default_pos()
            self.get_logger().info("8] --------> Model is running")
            self.get_logger().info("             ###############################################")
            self.get_logger().info("             # Press 'SELECT' to stop the model            #")
            self.get_logger().info("             ###############################################")

            loop_period_s = float(
                self.get_parameter("model_loop_period_s").value
            )
            if loop_period_s <= 0.0:
                raise ValueError("model_loop_period_s must be greater than zero")
            next_cycle_deadline = time.perf_counter()
            time_ms = 0
            self.Liste_t = []
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                self.run_policy_step()
                self.Liste_t.append(time_ms)
                time_ms += int(round(loop_period_s * 1000.0))
                if self.remote_controller.button[KeyMap.select] == 1:
                    self.get_logger().info(
                        "Remote SELECT pressed: stopping low-level policy loop and moving to ground"
                    )
                    self.move_to_ground()
                    break

                next_cycle_deadline += loop_period_s
                remaining_s = next_cycle_deadline - time.perf_counter()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
                else:
                    next_cycle_deadline = time.perf_counter()

                cycle_time_ms = (time.perf_counter() - cycle_start) * 1000.0
                self.policy_cycle_times_ms.append(cycle_time_ms)
                log_every = int(
                    self.get_parameter("cycle_timing_log_every_n_steps").value
                )
                if log_every > 0 and self.counter % log_every == 0:
                    recent_times = self.policy_cycle_times_ms[-log_every:]
                    print(
                        "Policy cycle time: "
                        f"avg={np.mean(recent_times):.2f} ms, "
                        f"min={np.min(recent_times):.2f} ms, "
                        f"max={np.max(recent_times):.2f} ms "
                        f"({len(recent_times)} cycles)"
                    )

            self._plot_analysis_if_enabled()
        except Exception as exc:
            self._stop_event.set()
            self._send_safe_hold_command()
            if rclpy.ok():
                self.get_logger().error(
                    f"Policy deployment stopped because of an error: {exc}"
                )
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
            while not self._stop_event.is_set():
                self.run_unitree_sport_high_level_step()
                time.sleep(loop_period)
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
            loop_period = float(self.get_parameter("model_loop_period_s").value)
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
        else:
            self._send_safe_hold_command()
        self._plot_analysis_if_enabled()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RealPolicyNode()
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
        print("Policy node shutdown completed cleanly.", flush=True)


if __name__ == "__main__":
    main()
