#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""ROS 2 node version of Deploy_SimToReal_RL_Go2/deploy_real."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
import tf2_ros

from fetch.deploy_real_utils import (
    DeployRealConfig,
    KeyMap,
    RemoteController,
    add_unitree_sdk_paths,
    copy_low_state_dds_to_ros,
    create_zero_cmd,
    get_gravity_orientation,
    init_cmd_go,
    set_motor_cmd_velocity,
)


class RealPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self._declare_parameters()
        self.config = self._config_from_parameters()
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
        self._stop_event = threading.Event()
        self._worker_thread = None
        self.fake_observations_mode = bool(self.get_parameter("fake_observations_mode").value)
        self.use_high_level_policy = bool(
            self.get_parameter("use_high_level_policy").value
        )
        self.add_on_set_parameters_callback(self._parameter_callback)
        self.commands_enabled = (
            bool(self.get_parameter("send_commands").value)
            and not self.fake_observations_mode
        )
        self._fake_rng = np.random.default_rng(
            int(self.get_parameter("fake_observation_seed").value)
        )

        self.get_logger().info("1] -> LE FICHIER DE CONFIG A BIEN ETE CHARGE")
        self._initialize_controller_state()

        if self.fake_observations_mode:
            self.get_logger().warn(
                "FAKE OBSERVATION MODE ENABLED: DDS is not initialized and robot commands are disabled."
            )
        else:
            self._load_unitree_sdk()
            self._load_ros_interfaces()
            self._initialize_ros_io()
            self._initialize_dds()
            self._initialize_robot_interfaces()

        if not self.commands_enabled:
            self.get_logger().warn("Robot command output is disabled.")

        if bool(self.get_parameter("start_policy_on_startup").value):
            worker_target = (
                self._run_fake_observation_sequence
                if self.fake_observations_mode
                else self._run_deploy_sequence
            )
            self._worker_thread = threading.Thread(target=worker_target, daemon=True)
            self._worker_thread.start()

    def _declare_parameters(self) -> None:
        self.declare_parameter("network_interface", "")
        self.declare_parameter("dds_domain_id", 0)
        # self.declare_parameter("project_root", "/home/ferdinand/unitree/Deploy_SimToReal_RL_Go2")
        # self.declare_parameter(
        #     "unitree_sdk_paths",
        #     Parameter.Type.STRING_ARRAY,
        # )
        self.declare_parameter("start_policy_on_startup", True)
        self.declare_parameter("fake_observations_mode", True)
        self.declare_parameter("send_commands", False)
        self.declare_parameter("use_high_level_policy", True)
        self.declare_parameter("fake_observation_seed", 0)
        self.declare_parameter("fake_observation_min", -1.0)
        self.declare_parameter("fake_observation_max", 1.0)
        self.declare_parameter("fake_log_every_n_steps", 100)
        self.declare_parameter("auto_switch_to_low_level", True)
        self.declare_parameter("wait_for_start_button", True)
        self.declare_parameter("wait_for_a_button", True)
        self.declare_parameter("plot_on_exit", False)
        self.declare_parameter("analysis_pdf_path", "analyse_robot.pdf")
        self.declare_parameter("kalman_odom_topic", "/odometry/filtered")
        self.declare_parameter("cube_state_topic", "/go2_fetch/cube_state")
        self.declare_parameter("goal_xy", [0.0, 0.0])
        self.declare_parameter("goal_radius", 0.2)
        self.declare_parameter("policy_world_frame", "odom")
        self.declare_parameter("lf_foot_frame", "FL_foot")
        self.declare_parameter("lf_foot_tf_timeout_s", 0.02)
        self.declare_parameter("robot_twist_in_body_frame", True)
        self.declare_parameter("inekf_lowstate_topic", "/inekf_lowstate")
        self.declare_parameter("lowstate_publish_period_s", 0.005)
        self.declare_parameter("model_loop_period_s", 0.02)
        self.declare_parameter("startup_sleep_s", 0.001)

        self.declare_parameter("control_dt", 0.005)
        self.declare_parameter("msg_type", "go")
        self.declare_parameter("imu_type", "pelvis")
        self.declare_parameter(
            "weak_motor",
            Parameter.Type.INTEGER_ARRAY,
        )
        self.declare_parameter("lowcmd_topic", "rt/lowcmd")
        self.declare_parameter("lowstate_topic", "rt/lowstate")
        self.declare_parameter("sportstate_topic", "rt/sportmodestate")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_base_dir", "/home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl")
        self.declare_parameter(
            "high_level_policy_path",
            "logs/rsl_rl/unitree_go2_pushcube_4l/2026-05-15_02-52-05_cam_6/exported/policy.pt",
        )
        self.declare_parameter(
            "low_level_policy_path",
            "logs/rsl_rl/unitree_go2_velocity_4l/2026-04-05_12-01-56_walk_2/exported/policy.pt",
        )
        self.declare_parameter("high_level_rate_hz", 15.0)
        self.declare_parameter("high_level_num_obs", 48)
        self.declare_parameter("low_level_num_obs", 45)
        self.declare_parameter("leg_joint2motor_idx", [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
        self.declare_parameter("default_angles", [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
        self.declare_parameter("kps", [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        self.declare_parameter("kds", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
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
        self.declare_parameter("ang_vel_scale", 0.2)
        self.declare_parameter("dof_pos_scale", 1.0)
        self.declare_parameter("dof_vel_scale", 0.05)
        self.declare_parameter("action_scale", 0.25)
        self.declare_parameter("cmd_scale", [0.8, 0.8, 1.0])
        self.declare_parameter("num_actions", 12)
        self.declare_parameter("num_obs", 45)
        self.declare_parameter("max_cmd", [1.0, 1.0, 1.0])

    def _parameter_callback(self, parameters) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name != "use_high_level_policy":
                continue
            if parameter.type_ != Parameter.Type.BOOL:
                return SetParametersResult(
                    successful=False,
                    reason="use_high_level_policy must be a Boolean",
                )

            self.use_high_level_policy = bool(parameter.value)
            self._next_high_level_time = -math.inf
            source = "PushCube high-level policy" if parameter.value else "joystick"
            self.get_logger().info(
                f"Low-level velocity command source changed to: {source}"
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
            policy_path=str(self.get_parameter("policy_path").value),
            leg_joint2motor_idx=list(self.get_parameter("leg_joint2motor_idx").value),
            kps=list(self.get_parameter("kps").value),
            kds=list(self.get_parameter("kds").value),
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

    def _load_unitree_sdk(self) -> None:
        from unitree_sdk2_python.unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2_python.unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2_python.unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
        from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from unitree_sdk2_python.unitree_sdk2py.utils.crc import CRC

        self.MotionSwitcherClient = MotionSwitcherClient
        self.ChannelFactoryInitialize = ChannelFactoryInitialize
        self.ChannelPublisher = ChannelPublisher
        self.ChannelSubscriber = ChannelSubscriber
        self.SportClient = SportClient
        self.LowCmdDefault = unitree_go_msg_dds__LowCmd_
        self.LowStateDefault = unitree_go_msg_dds__LowState_
        self.LowCmdGo = LowCmdGo
        self.LowStateGo = LowStateGo
        self.SportModeState = SportModeState_
        self.CRC = CRC

    def _load_ros_interfaces(self) -> None:
        try:
            from unitree_go.msg import LowState as LowStateRos

            self.LowStateRos = LowStateRos
        except Exception as exc:
            self.LowStateRos = None
            self.get_logger().warn(f"unitree_go.msg.LowState is not available; /inekf_lowstate disabled: {exc}")

    def _initialize_ros_io(self) -> None:
        self.create_subscription(Odometry, self.get_parameter("kalman_odom_topic").value, self._kalman_odom_callback, 10)
        self.create_subscription(Odometry, self.get_parameter("cube_state_topic").value, self._cube_state_callback, 10)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.lowstate_publisher = None
        if self.LowStateRos is not None:
            self.lowstate_publisher = self.create_publisher(
                self.LowStateRos,
                self.get_parameter("inekf_lowstate_topic").value,
                10,
            )
            self.create_timer(float(self.get_parameter("lowstate_publish_period_s").value), self._publish_lowstate)

    def _initialize_dds(self) -> None:
        domain_id = int(self.get_parameter("dds_domain_id").value)
        net = str(self.get_parameter("network_interface").value).strip()
        if net:
            self.ChannelFactoryInitialize(domain_id, net)
        else:
            self.ChannelFactoryInitialize(domain_id)
        self.get_logger().info("2] --> LE CHANNELFACTORY A ETE CREE")

    def _initialize_controller_state(self) -> None:
        self.get_logger().info("3] ---> CHARGEMENT DES POLITIQUES HIGH-LEVEL ET LOW-LEVEL")
        high_level_path = self._resolve_policy_path("high_level_policy_path")
        low_level_path = self._resolve_policy_path("low_level_policy_path")
        self.high_level_policy = torch.jit.load(high_level_path).eval()
        self.low_level_policy = torch.jit.load(low_level_path).eval()

        self.defaut_isaac = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        self.base_lin_vel = np.array([0, 0, 0])
        self.cmd = np.zeros(3, dtype=np.float32)
        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.high_level_action = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.target_dof_pos = self.defaut_isaac.copy()
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
        self.goal_xy = np.array(self.get_parameter("goal_xy").value, dtype=np.float32)
        self.goal_radius = float(self.get_parameter("goal_radius").value)
        self._next_high_level_time = -math.inf

        self.dt = 0.002
        self.startPos = [0.0] * 12
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.firstRun = True
        self.counter = 0

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1, -1.5, 0.1, 1, -1.5]
        self._targetPos_3 = self._targetPos_2

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

    def _resolve_policy_path(self, parameter_name: str) -> Path:
        path = Path(str(self.get_parameter(parameter_name).value)).expanduser()
        if not path.is_absolute():
            path = Path(self.get_parameter("policy_base_dir").value).expanduser() / path
        if not path.is_file():
            raise FileNotFoundError(f"{parameter_name} does not exist: {path}")
        return path

    def _initialize_robot_interfaces(self) -> None:
        self.get_logger().info("4] ----> INITIALISATION DES CHANNELS")
        self.lowcmd_publisher_ = self.ChannelPublisher(self.config.lowcmd_topic, self.LowCmdGo)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = self.ChannelSubscriber(self.config.lowstate_topic, self.LowStateGo)
        self.lowstate_subscriber.Init(self._low_state_go_handler, 10)
        self.sportstate_subscriber = self.ChannelSubscriber(self.get_parameter("sportstate_topic").value, self.SportModeState)
        self.sportstate_subscriber.Init(self._sport_state_message_handler, 10)

        self.low_cmd = self.LowCmdDefault()
        self.low_state = self.LowStateDefault()
        self.wait_for_low_state()
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def init_low_level_mode(self) -> None:
        if not bool(self.get_parameter("auto_switch_to_low_level").value):
            return

        self.sc = self.SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = self.MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"] and not self._stop_event.is_set():
            self.sc.StandDown()
            self.msc.ReleaseMode()
            self.get_logger().info(
                "3] ---> LE ROBOT EST EN POSITION ALLONGE ET LE MODE HAUT NIVEAU EST RELACHE -> PASSAGE EN BAS NIVEAU"
            )
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def wait_for_low_state(self) -> None:
        while self.low_state.tick == 0 and not self._stop_event.is_set():
            time.sleep(self.config.control_dt)
        self.get_logger().info("         Connecte au robot")

    def _low_state_go_handler(self, msg) -> None:
        self.low_state = msg
        self.low_state_msg = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def _sport_state_message_handler(self, sport_state_msg) -> None:
        self.velocity = sport_state_msg.velocity

    def _kalman_odom_callback(self, msg: Odometry) -> None:
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
        self.cube_pos_xy[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.cube_lin_vel_xy[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def _publish_lowstate(self) -> None:
        if self.lowstate_publisher is None or self.low_state_msg is None:
            return
        ros_msg = self.LowStateRos()
        copy_low_state_dds_to_ros(self.low_state_msg, ros_msg)
        self.lowstate_publisher.publish(ros_msg)

    def send_cmd(self, cmd) -> None:
        if not self.commands_enabled:
            return
        cmd.crc = self.CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

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
        self.get_logger().info("5] -----> LE ZERO TORQUE STATE EST EN COURS")
        self.get_logger().info("          ##################################################")
        self.get_logger().info("          # EN ATTENTE DU BOUTON START POUR LEVER LE ROBOT #")
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
        self.get_logger().info("6] ------> LE ROBOT SE DEPLACE VERS LA DEFAULT POSE")
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
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if self.percent_1 == 1 and self.percent_2 <= 1:
                self.percent_2 += 1.0 / self.duration_2
                self.percent_2 = min(self.percent_2, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if self.percent_1 == 1 and self.percent_2 == 1 and self.percent_3 < 1:
                self.percent_3 += 1.0 / self.duration_3
                self.percent_3 = min(self.percent_3, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = self._targetPos_2[i]
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if self.percent_1 == 1 and self.percent_2 == 1 and self.percent_3 == 1 and self.percent_4 <= 1:
                self.percent_4 += 1.0 / self.duration_4
                self.percent_4 = min(self.percent_4, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self._targetPos_2[i] + self.percent_4 * self._targetPos_3[i]
                    set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            self.send_cmd(self.low_cmd)
            if self.percent_4 == 1.0 or self.count == 2500000000:
                done = True
            time.sleep(float(self.get_parameter("startup_sleep_s").value))

        self.get_logger().info("7] -------> LE ROBOT SE MAINTIENT DEBOUT")
        self.get_logger().info("            ###########################################")
        self.get_logger().info("            # APPUYEZ SUR 'A' POUR DEMARRER LE MODELE #")
        self.get_logger().info("            ###########################################")

        while (
            bool(self.get_parameter("wait_for_a_button").value)
            and self.remote_controller.button[KeyMap.A] != 1
            and not self._stop_event.is_set()
        ):
            default = self.config.default_angles
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = default[i]
                set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                self.low_cmd.motor_cmd[i].kp = 60
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
                self.send_cmd(self.low_cmd)
                time.sleep(0.002)

    def move_to_ground(self) -> None:
        percent = 0
        pos_init = []
        for k in range(12):
            pos_init.append(self.low_state.motor_state[k].q)
        while percent != 1 and not self._stop_event.is_set():
            percent += 1.0 / 300
            percent = min(percent, 1)
            couche = [0, 1.36, -2.65, 0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - percent) * pos_init[i] + percent * couche[i]
                set_motor_cmd_velocity(self.low_cmd.motor_cmd[i], 0)
                self.low_cmd.motor_cmd[i].kp = 60
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
                self.send_cmd(self.low_cmd)
            time.sleep(0.002)
        self.get_logger().info("9] ---------> LE ROBOT EST ALLONGE")

    def run_policy_step(self):
        self.counter += 1
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)

        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        defaut_joint = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        qj_obs = qj_obs - defaut_joint

        if self.use_high_level_policy:
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

            self.cmd[:] = self.high_level_action
        else:
            self.cmd[:] = self._joystick_velocity_command()

        self.obs = np.concatenate(
            [
                ang_vel * self.config.ang_vel_scale,
                gravity_orientation,
                self.cmd,
                qj_obs * self.config.dof_pos_scale,
                dqj_obs * self.config.dof_vel_scale,
                self.action,
            ]
        ).astype(np.float32)
        self._require_observation_size(
            self.obs, int(self.get_parameter("low_level_num_obs").value), "Low-level"
        )
        with torch.inference_mode():
            self.action = (
                self.low_level_policy(torch.from_numpy(self.obs).unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        if self.action.size != self.config.num_actions:
            raise RuntimeError(
                f"Low-level policy returned {self.action.size} actions; expected {self.config.num_actions}"
            )

        target_dof_pos = self.action * self.config.action_scale + defaut_joint
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            set_motor_cmd_velocity(self.low_cmd.motor_cmd[motor_idx], 0)
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

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

        return self.obs

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

    def _build_high_level_observation(
        self, ang_vel, gravity_orientation, qj_obs, dqj_obs
    ) -> np.ndarray:
        lf_foot_xy = self._lookup_lf_foot_xy()
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
            ]
        ).astype(np.float32)
        self._require_observation_size(
            obs, int(self.get_parameter("high_level_num_obs").value), "High-level"
        )
        return obs

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
        try:
            self.init_low_level_mode()
            self.zero_torque_state()
            self.move_to_default_pos()
            self.get_logger().info("8] --------> LE MODELE EST LANCE")
            self.get_logger().info("             ###############################################")
            self.get_logger().info("             # APPUYEZ SUR 'SELECT' POUR ARRETER LE MODELE #")
            self.get_logger().info("             ###############################################")

            time_ms = 0
            self.Liste_t = []
            while not self._stop_event.is_set():
                self.run_policy_step()
                time.sleep(float(self.get_parameter("model_loop_period_s").value))
                self.Liste_t.append(time_ms)
                time_ms += 20
                if self.remote_controller.button[KeyMap.select] == 1:
                    self.move_to_ground()
                    break

            if bool(self.get_parameter("plot_on_exit").value):
                self._plot_analysis()
        except Exception as exc:
            self.get_logger().error(f"Policy deployment stopped because of an error: {exc}")

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

    def _plot_analysis(self) -> None:
        import matplotlib.pyplot as plt

        self.get_logger().info("10] ----------> VISUALISATION DES DONNEES EN COURS")
        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
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
        plt.tight_layout()
        plt.savefig(self.get_parameter("analysis_pdf_path").value)
        plt.show()

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
