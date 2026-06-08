#!/usr/bin/env python3
"""ROS 2 node version of Deploy_SimToReal_RL_Go2/deploy_real."""

from __future__ import annotations

import threading
import time
from math import pi
from pathlib import Path

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter

from fetch.deploy_real_utils import (
    DeployRealConfig,
    KeyMap,
    RemoteController,
    add_unitree_sdk_paths,
    compute_velocity,
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
        self.project_root = Path(self.get_parameter("project_root").value).expanduser()
        sdk_paths = self._array_parameter(
            "unitree_sdk_paths",
            Parameter.Type.STRING_ARRAY,
        )
        sdk_paths.insert(0, str(self.project_root))
        add_unitree_sdk_paths(sdk_paths)

        self.remote_controller = RemoteController()
        self.base_lin_vel_input = [0, 0, 0, 0]
        self.low_state_msg = None
        self._stop_event = threading.Event()
        self._worker_thread = None
        self.fake_observations_mode = bool(self.get_parameter("fake_observations_mode").value)
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
        self.declare_parameter("project_root", "/home/ferdinand/unitree/Deploy_SimToReal_RL_Go2")
        self.declare_parameter(
            "unitree_sdk_paths",
            Parameter.Type.STRING_ARRAY,
        )
        self.declare_parameter("start_policy_on_startup", True)
        self.declare_parameter("fake_observations_mode", True)
        self.declare_parameter("send_commands", False)
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
        self.declare_parameter("inekf_lowstate_topic", "/inekf_lowstate")
        self.declare_parameter("lowstate_publish_period_s", 0.005)
        self.declare_parameter("model_loop_period_s", 0.025)
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
        self.declare_parameter("policy_path", "policy_rough.pt")
        self.declare_parameter("policy_base_dir", "/home/ferdinand/unitree/Deploy_SimToReal_RL_Go2/pre_train")
        self.declare_parameter("leg_joint2motor_idx", [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
        self.declare_parameter("default_angles", [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
        self.declare_parameter("kps", [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        self.declare_parameter("kds", [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
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
        self.declare_parameter("ang_vel_scale", 0.25)
        self.declare_parameter("dof_pos_scale", 1.0)
        self.declare_parameter("dof_vel_scale", 0.05)
        self.declare_parameter("action_scale", 0.25)
        self.declare_parameter("cmd_scale", [0.8, 0.8, 1.0])
        self.declare_parameter("num_actions", 12)
        self.declare_parameter("num_obs", 52)
        self.declare_parameter("max_cmd", [1.0, 1.0, 1.0])

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
        self.get_logger().info("3] ---> CHARGEMENT DE LA POLITIQUE")
        policy_path = Path(self.config.policy_path).expanduser()
        if not policy_path.is_absolute():
            policy_path = Path(self.get_parameter("policy_base_dir").value).expanduser() / policy_path
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

        self.defaut_isaac = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        self.base_lin_vel = np.array([0, 0, 0])
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.target_dof_pos = self.defaut_isaac.copy()
        self.obs = np.zeros(self.config.num_obs, dtype=np.float32)

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
        self.obs = self._fake_rng.uniform(
            observation_min,
            observation_max,
            size=self.config.num_obs,
        ).astype(np.float32)

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        with torch.inference_mode():
            self.action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()

        log_every = max(
            1,
            int(self.get_parameter("fake_log_every_n_steps").value),
        )
        if self.counter % log_every == 0:
            self.get_logger().info(
                f"Fake policy step {self.counter}: "
                f"obs=[{self.obs.min():.3f}, {self.obs.max():.3f}] "
                f"action=[{self.action.min():.3f}, {self.action.max():.3f}] "
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

        theta1 = []
        theta2 = []
        theta3 = []
        thetav1 = []
        thetav2 = []
        thetav3 = []
        theta1.append(-self.low_state.motor_state[1].q + 1.5708)
        theta2.append(-self.low_state.motor_state[2].q - 1.7 + pi / 2)
        theta3.append(-self.low_state.motor_state[0].q)
        thetav1.append(self.low_state.motor_state[1].dq)
        thetav2.append(self.low_state.motor_state[2].dq)
        thetav3.append(-self.low_state.motor_state[0].dq)
        theta1.append(-self.low_state.motor_state[4].q + 1.5708)
        theta2.append(-self.low_state.motor_state[5].q - 1.7 + pi / 2)
        theta3.append(-self.low_state.motor_state[3].q)
        thetav1.append(self.low_state.motor_state[4].dq)
        thetav2.append(self.low_state.motor_state[5].dq)
        thetav3.append(-self.low_state.motor_state[3].dq)
        theta1.append(-self.low_state.motor_state[7].q + 1.5708)
        theta2.append(-self.low_state.motor_state[8].q - 1.7 + pi / 2)
        theta3.append(-self.low_state.motor_state[6].q)
        thetav1.append(self.low_state.motor_state[7].dq)
        thetav2.append(self.low_state.motor_state[8].dq)
        thetav3.append(-self.low_state.motor_state[6].dq)
        theta1.append(-self.low_state.motor_state[10].q + 1.5708)
        theta2.append(-self.low_state.motor_state[11].q - 1.7 + pi / 2)
        theta3.append(-self.low_state.motor_state[9].q)
        thetav1.append(self.low_state.motor_state[10].dq)
        thetav2.append(self.low_state.motor_state[11].dq)
        thetav3.append(-self.low_state.motor_state[9].dq)
        foot = self.low_state.foot_force

        vx_calc, vy_calc, vz_calc = compute_velocity(theta1, theta2, theta3, thetav1, thetav2, thetav3, foot)
        vx = self._update_velocity_window(self.vx_window, vx_calc)
        vy = self._update_velocity_window(self.vy_window, vy_calc)
        vz = self._update_velocity_window(self.vz_window, vz_calc)

        f0 = self.low_state.foot_force[1] / 100
        f1 = self.low_state.foot_force[3] / 100
        f2 = self.low_state.foot_force[0] / 100
        f3 = self.low_state.foot_force[2] / 100

        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)

        self.cmd[0] = round(self.remote_controller.ly, 1)
        self.cmd[1] = round(self.remote_controller.lx * -1, 1)
        self.cmd[2] = round(self.remote_controller.rx * -1, 1)

        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        defaut_joint = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        qj_obs = qj_obs - defaut_joint

        num_actions = self.config.num_actions
        self.obs[:4] = [f0, f1, f2, f3]
        self.obs[4:7] = [self.base_lin_vel_input[0], self.base_lin_vel_input[1], self.base_lin_vel_input[2]]
        self.obs[7:10] = ang_vel
        self.obs[10:13] = gravity_orientation
        self.obs[13:16] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[16 : 16 + num_actions] = qj_obs
        self.obs[16 + num_actions : 16 + num_actions * 2] = dqj_obs
        self.obs[16 + num_actions * 2 : 16 + num_actions * 3] = self.action

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        with torch.inference_mode():
            self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        target_dof_pos = self.action * self.config.action_scale + defaut_joint
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            set_motor_cmd_velocity(self.low_cmd.motor_cmd[motor_idx], 0)
            self.low_cmd.motor_cmd[motor_idx].kp = 40
            self.low_cmd.motor_cmd[motor_idx].kd = 0.5
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

        self.L_base_vel_cmd_input_1.append(self.obs[13])
        self.L_base_vel_cmd_input_2.append(self.obs[14])
        self.L_base_vel_cmd_input_3.append(self.obs[15])
        self.L_base_lin_vel_input_1.append(vx * 2)
        self.L_base_lin_vel_input_2.append(vy * 2)
        self.L_base_lin_vel_input_3.append(vz * 2)
        self.L_base_lin_vel_kalman_input_1.append(self.base_lin_vel_input[0])
        self.L_base_lin_vel_kalman_input_2.append(self.base_lin_vel_input[1])
        self.L_base_lin_vel_kalman_input_3.append(self.base_lin_vel_input[2])
        self.L_base_lin_vel_kalman_input_4.append(self.base_lin_vel_input[3])
        self.L_base_ang_vel_input_1.append(self.obs[3])
        self.L_base_ang_vel_input_2.append(self.obs[4])
        self.L_base_ang_vel_input_3.append(self.obs[5])

        return self.obs

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
                time_ms += 25
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
