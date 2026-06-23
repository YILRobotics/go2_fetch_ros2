#!/home/unitree/miniconda3/envs/env_deploy/bin/python

"""Policy rollout node: high-level push policy + low-level locomotion policy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    import torch
except ImportError:
    torch = None


def _resolve_policy_path(path_raw: str) -> Path:
    path = Path(path_raw).expanduser().resolve()
    if path.name.startswith('model_') and path.suffix == '.pt':
        exported_candidate = path.parent / 'exported' / 'policy.pt'
        if exported_candidate.is_file():
            return exported_candidate
    return path


def _resolve_torch_device(device_name: str) -> str:
    normalized = str(device_name).strip().lower()
    if normalized == 'gpu':
        return 'cuda'
    return normalized


def _initialize_go2_lowcmd(cmd) -> None:
    if hasattr(cmd, 'head') and len(cmd.head) >= 2:
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
    if hasattr(cmd, 'level_flag'):
        cmd.level_flag = 0xFF
    if hasattr(cmd, 'gpio'):
        cmd.gpio = 0

    for motor_cmd in getattr(cmd, 'motor_cmd', []):
        if hasattr(motor_cmd, 'mode'):
            motor_cmd.mode = 0x0A
        if hasattr(motor_cmd, 'q'):
            motor_cmd.q = 2.146e9
        if hasattr(motor_cmd, 'dq'):
            motor_cmd.dq = 16000.0
        if hasattr(motor_cmd, 'kp'):
            motor_cmd.kp = 0.0
        if hasattr(motor_cmd, 'kd'):
            motor_cmd.kd = 0.0
        if hasattr(motor_cmd, 'tau'):
            motor_cmd.tau = 0.0


def _safe_normalize_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _rotation_matrix_from_wxyz(q_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = _safe_normalize_quat_wxyz(np.array(q_wxyz, dtype=np.float64))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _projected_gravity_body_from_wxyz(q_wxyz: Sequence[float]) -> np.ndarray:
    rot_bw = _rotation_matrix_from_wxyz(q_wxyz)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    g_body = rot_bw.T @ g_world
    return g_body.astype(np.float32)


def _yaw_from_xyzw(q_xyzw: Sequence[float]) -> float:
    x, y, z, w = q_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


class PolicyNode(Node):
    """Runs the push policy stack and emits low-level motor targets."""

    def __init__(self) -> None:
        super().__init__('policy_node')
        self._declare_parameters()

        if torch is None:
            raise RuntimeError('PyTorch is required for policy rollout but is not installed.')

        self.mode_topic = self.get_parameter('mode_topic').value
        self.lowstate_topic = self.get_parameter('lowstate_topic').value
        self.lowcmd_topic = self.get_parameter('lowcmd_topic').value
        self.robot_odom_topic = self.get_parameter('robot_odom_topic').value
        self.cube_state_topic = self.get_parameter('cube_state_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.policy_stack = self.get_parameter('policy_stack').value.strip().lower()
        self.mode = self.get_parameter('initial_mode').value.strip().lower()
        self.cmd_vel_timeout_s = float(self.get_parameter('cmd_vel_timeout_s').value)
        self.require_lowstate = bool(self.get_parameter('require_lowstate').value)
        self.lowstate_timeout_s = float(self.get_parameter('lowstate_timeout_s').value)

        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.high_level_rate_hz = float(self.get_parameter('high_level_rate_hz').value)
        self.standup_duration_s = float(self.get_parameter('standup_duration_s').value)
        self.search_spin_rate = float(self.get_parameter('search_spin_rate').value)

        self.action_scale = float(self.get_parameter('action_scale').value)
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.tau_ff = float(self.get_parameter('tau_ff').value)

        self.goal_xy = np.array(self.get_parameter('goal_xy').value, dtype=np.float32)
        self.goal_radius = float(self.get_parameter('goal_radius').value)

        self.max_lin_x = float(self.get_parameter('max_lin_x').value)
        self.max_lin_y = float(self.get_parameter('max_lin_y').value)
        self.max_yaw = float(self.get_parameter('max_yaw').value)

        self.lf_foot_xy_offset = np.array(self.get_parameter('lf_foot_xy_offset').value, dtype=np.float32)

        self.default_joint_angles = np.array(self.get_parameter('default_joint_angles').value, dtype=np.float32)
        self.policy_to_motor = np.array(self.get_parameter('policy_to_motor_map').value, dtype=np.int32)

        self.device_name = _resolve_torch_device(self.get_parameter('torch_device').value)
        self.device = torch.device(self.device_name)

        self.ang_vel_scale = float(self.get_parameter('ang_vel_scale').value)
        self.dof_pos_scale = float(self.get_parameter('dof_pos_scale').value)
        self.dof_vel_scale = float(self.get_parameter('dof_vel_scale').value)
        self.cmd_scale = np.array(self.get_parameter('cmd_scale').value, dtype=np.float32)
        self.max_cmd = np.array(self.get_parameter('max_cmd').value, dtype=np.float32)
        self.foot_force_scale = float(self.get_parameter('foot_force_scale').value)
        self.low_level_num_obs = int(self.get_parameter('low_level_num_obs').value)

        high_level_path = _resolve_policy_path(self.get_parameter('high_level_policy_path').value)
        low_level_path = _resolve_policy_path(self.get_parameter('low_level_policy_path').value)
        self._high_level_policy = None
        if self._uses_high_level_policy():
            self._high_level_policy = self._load_torchscript(high_level_path, 'high-level')
        self._low_level_policy = self._load_torchscript(low_level_path, 'low-level')

        self._mode_start_time = self.get_clock().now().nanoseconds * 1e-9
        self._standup_start_policy_joints: Optional[np.ndarray] = None

        self._last_high_level_time = -1.0
        self._last_high_action = np.zeros(3, dtype=np.float32)
        self._last_low_action = np.zeros(12, dtype=np.float32)
        self._cmd_vel_action = np.zeros(3, dtype=np.float32)
        self._last_cmd_vel_time: Optional[float] = None
        self._last_lowstate_time: Optional[float] = None

        self._motor_q = np.zeros(12, dtype=np.float32)
        self._motor_dq = np.zeros(12, dtype=np.float32)
        self._foot_force = np.zeros(4, dtype=np.float32)
        self._imu_gyro = np.zeros(3, dtype=np.float32)
        self._imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._robot_pos_xy = np.zeros(2, dtype=np.float32)
        self._robot_vel_xyz = np.zeros(3, dtype=np.float32)
        self._robot_odom_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self._cube_pos_xy = np.zeros(2, dtype=np.float32)
        self._cube_vel_xy = np.zeros(2, dtype=np.float32)

        self._lowcmd_pub = None
        self._lowcmd_msg_cls = None
        self._lowstate_msg_cls = None
        self._load_unitree_interfaces()

        self._joint_target_pub = self.create_publisher(Float32MultiArray, 'go2_fetch/joint_targets', 10)
        self._command_pub = self.create_publisher(TwistStamped, 'go2_fetch/policy_cmd', 10)

        self.create_subscription(String, self.mode_topic, self._mode_cb, 10)
        self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_vel_cb, 10)
        self.create_subscription(Odometry, self.robot_odom_topic, self._robot_odom_cb, 20)
        self.create_subscription(Odometry, self.cube_state_topic, self._cube_state_cb, 20)

        if self._lowstate_msg_cls is not None:
            self.create_subscription(self._lowstate_msg_cls, self.lowstate_topic, self._lowstate_cb, 20)
        else:
            self.get_logger().warn(
                'unitree_go.msg.LowState not available. Motor-state observations stay zero and lowcmd output is disabled.'
            )

        self._control_timer = self.create_timer(1.0 / self.control_rate_hz, self._control_step)
        self.get_logger().info(
            f'Policy node ready. stack={self.policy_stack} high={high_level_path} low={low_level_path} '
            f'mode={self.mode} mode_topic={self.mode_topic} cmd_vel_topic={self.cmd_vel_topic}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('mode_topic', '/go2_fetch/mode')
        self.declare_parameter('lowstate_topic', '/lowstate')
        self.declare_parameter('lowcmd_topic', '/lowcmd')
        self.declare_parameter('robot_odom_topic', '/lio_sam_ros2/mapping/odometry')
        self.declare_parameter('cube_state_topic', '/go2_fetch/cube_state')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('policy_stack', 'low_and_high_level')
        self.declare_parameter('initial_mode', 'standup')
        self.declare_parameter('cmd_vel_timeout_s', 0.5)
        self.declare_parameter('require_lowstate', True)
        self.declare_parameter('lowstate_timeout_s', 0.25)

        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('high_level_rate_hz', 15.0)
        self.declare_parameter('standup_duration_s', 2.5)
        self.declare_parameter('search_spin_rate', 0.30)

        self.declare_parameter('action_scale', 0.25)
        self.declare_parameter('kp', 40.0)
        self.declare_parameter('kd', 0.5)
        self.declare_parameter('tau_ff', 0.0)
        self.declare_parameter('ang_vel_scale', 1.0)
        self.declare_parameter('dof_pos_scale', 1.0)
        self.declare_parameter('dof_vel_scale', 0.05)
        self.declare_parameter('cmd_scale', [0.8, 0.8, 1.0])
        self.declare_parameter('max_cmd', [1.0, 1.0, 1.0])
        self.declare_parameter('foot_force_scale', 0.01)
        self.declare_parameter('low_level_num_obs', 45)

        self.declare_parameter('goal_xy', [0.0, 0.0])
        self.declare_parameter('goal_radius', 0.2)

        self.declare_parameter('max_lin_x', 0.5)
        self.declare_parameter('max_lin_y', 0.5)
        self.declare_parameter('max_yaw', 0.25)

        self.declare_parameter('lf_foot_xy_offset', [0.22, 0.10])
        self.declare_parameter('default_joint_angles', [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
        self.declare_parameter('policy_to_motor_map', [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])

        self.declare_parameter('high_level_policy_path', '')
        self.declare_parameter('low_level_policy_path', '')
        self.declare_parameter('torch_device', 'cpu')

    def _load_torchscript(self, path: Path, name: str):
        if not path.is_file():
            raise FileNotFoundError(f'{name} policy file does not exist: {path}')

        policy = torch.jit.load(str(path), map_location=self.device)
        policy.eval()
        return policy

    def _load_unitree_interfaces(self) -> None:
        try:
            from unitree_go.msg import LowCmd, LowState

            self._lowstate_msg_cls = LowState
            self._lowcmd_msg_cls = LowCmd
            self._lowcmd_pub = self.create_publisher(LowCmd, self.lowcmd_topic, 10)
            self.get_logger().info('Using unitree_go LowState/LowCmd interfaces.')
        except Exception as exc:
            self._lowstate_msg_cls = None
            self._lowcmd_msg_cls = None
            self._lowcmd_pub = None
            self.get_logger().warn(f'Could not import unitree_go interfaces: {exc}')

    def _uses_high_level_policy(self) -> bool:
        return self.policy_stack in ('low_and_high_level', 'high_and_low', 'hierarchical')

    def _mode_cb(self, msg: String) -> None:
        new_mode = msg.data.strip().lower()
        if new_mode == self.mode:
            return

        self.mode = new_mode
        self._mode_start_time = self.get_clock().now().nanoseconds * 1e-9
        self._standup_start_policy_joints = None
        self.get_logger().info(f'Mode switched to: {self.mode}')

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._cmd_vel_action[:] = [
            float(np.clip(msg.linear.x, -self.max_lin_x, self.max_lin_x)),
            float(np.clip(msg.linear.y, -self.max_lin_y, self.max_lin_y)),
            float(np.clip(msg.angular.z, -self.max_yaw, self.max_yaw)),
        ]
        self._last_cmd_vel_time = self.get_clock().now().nanoseconds * 1e-9

    def _robot_odom_cb(self, msg: Odometry) -> None:
        self._robot_pos_xy[0] = float(msg.pose.pose.position.x)
        self._robot_pos_xy[1] = float(msg.pose.pose.position.y)

        self._robot_vel_xyz[0] = float(msg.twist.twist.linear.x)
        self._robot_vel_xyz[1] = float(msg.twist.twist.linear.y)
        self._robot_vel_xyz[2] = float(msg.twist.twist.linear.z)

        self._robot_odom_quat_xyzw[:] = [
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
            float(msg.pose.pose.orientation.w),
        ]

    def _cube_state_cb(self, msg: Odometry) -> None:
        self._cube_pos_xy[0] = float(msg.pose.pose.position.x)
        self._cube_pos_xy[1] = float(msg.pose.pose.position.y)

        self._cube_vel_xy[0] = float(msg.twist.twist.linear.x)
        self._cube_vel_xy[1] = float(msg.twist.twist.linear.y)

    def _lowstate_cb(self, msg) -> None:
        try:
            self._last_lowstate_time = self.get_clock().now().nanoseconds * 1e-9
            motor_state = msg.motor_state
            for i in range(min(12, len(motor_state))):
                self._motor_q[i] = float(motor_state[i].q)
                self._motor_dq[i] = float(motor_state[i].dq)

            if hasattr(msg, 'foot_force'):
                for i in range(min(4, len(msg.foot_force))):
                    self._foot_force[i] = float(msg.foot_force[i])

            imu = msg.imu_state
            self._imu_gyro[:] = [float(imu.gyroscope[0]), float(imu.gyroscope[1]), float(imu.gyroscope[2])]
            self._imu_quat_wxyz[:] = [
                float(imu.quaternion[0]),
                float(imu.quaternion[1]),
                float(imu.quaternion[2]),
                float(imu.quaternion[3]),
            ]
        except Exception as exc:
            self.get_logger().error(f'Failed to parse LowState: {exc}')

    def _control_step(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9

        if self.require_lowstate and not self._has_fresh_lowstate(now_s):
            self.get_logger().warn(
                'Waiting for fresh LowState before sending motor targets.',
                throttle_duration_sec=2.0,
            )
            return

        if self.mode == 'standup':
            target_policy_joints = self._standup_targets(now_s)
            self._publish_motor_targets(target_policy_joints)
            return

        if self.mode == 'search':
            high_action = np.array([0.0, 0.0, self.search_spin_rate], dtype=np.float32)
        elif self.mode == 'policy':
            if self._uses_high_level_policy():
                high_action = self._run_high_level_policy(now_s)
            else:
                high_action = self._get_cmd_vel_action(now_s)
        else:
            target_policy_joints = self.default_joint_angles.copy()
            self._publish_motor_targets(target_policy_joints)
            return

        low_action = self._run_low_level_policy(high_action)
        target_policy_joints = low_action * self.action_scale + self.default_joint_angles
        self._publish_motor_targets(target_policy_joints)
        self._publish_debug_command(high_action)

    def _has_fresh_lowstate(self, now_s: float) -> bool:
        if self._last_lowstate_time is None:
            return False
        return now_s - self._last_lowstate_time <= self.lowstate_timeout_s

    def _standup_targets(self, now_s: float) -> np.ndarray:
        current_policy_joints = self._motor_q[self.policy_to_motor]
        if self._standup_start_policy_joints is None:
            self._standup_start_policy_joints = current_policy_joints.copy()

        elapsed = now_s - self._mode_start_time
        alpha = np.clip(elapsed / max(self.standup_duration_s, 1e-3), 0.0, 1.0)
        return (1.0 - alpha) * self._standup_start_policy_joints + alpha * self.default_joint_angles

    def _run_high_level_policy(self, now_s: float) -> np.ndarray:
        if self._high_level_policy is None:
            raise RuntimeError('High-level policy requested, but policy_stack is not configured for it.')

        update_period = 1.0 / max(self.high_level_rate_hz, 1e-3)
        if now_s - self._last_high_level_time < update_period:
            return self._last_high_action.copy()

        obs = self._build_high_level_obs()
        obs_tensor = torch.from_numpy(obs).to(device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self._high_level_policy(obs_tensor).detach().cpu().numpy().reshape(-1)

        action = np.asarray(action[:3], dtype=np.float32)
        action[0] = float(np.clip(action[0], -self.max_lin_x, self.max_lin_x))
        action[1] = float(np.clip(action[1], -self.max_lin_y, self.max_lin_y))
        action[2] = float(np.clip(action[2], -self.max_yaw, self.max_yaw))

        self._last_high_action = action
        self._last_high_level_time = now_s
        return action.copy()

    def _get_cmd_vel_action(self, now_s: float) -> np.ndarray:
        if self._last_cmd_vel_time is None:
            return np.zeros(3, dtype=np.float32)
        if now_s - self._last_cmd_vel_time > self.cmd_vel_timeout_s:
            return np.zeros(3, dtype=np.float32)
        return self._cmd_vel_action.copy()

    def _run_low_level_policy(self, high_cmd: np.ndarray) -> np.ndarray:
        low_obs = self._build_low_level_obs(high_cmd)
        low_obs_tensor = torch.from_numpy(low_obs).to(device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self._low_level_policy(low_obs_tensor).detach().cpu().numpy().reshape(-1)

        action = np.asarray(action[:12], dtype=np.float32)
        self._last_low_action = action.copy()
        return action

    def _build_high_level_obs(self) -> np.ndarray:
        joint_pos_policy = self._motor_q[self.policy_to_motor]
        joint_vel_policy = self._motor_dq[self.policy_to_motor]

        base_ang_vel = self._imu_gyro * 0.2
        projected_gravity = _projected_gravity_body_from_wxyz(self._imu_quat_wxyz)

        yaw = _yaw_from_xyzw(self._robot_odom_quat_xyzw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        lf_foot_xy = self._robot_pos_xy + np.array(
            [
                cos_yaw * self.lf_foot_xy_offset[0] - sin_yaw * self.lf_foot_xy_offset[1],
                sin_yaw * self.lf_foot_xy_offset[0] + cos_yaw * self.lf_foot_xy_offset[1],
            ],
            dtype=np.float32,
        )

        cube_to_goal = self.goal_xy - self._cube_pos_xy
        lf_foot_to_cube = self._cube_pos_xy - lf_foot_xy

        obs_parts = [
            base_ang_vel,
            projected_gravity,
            joint_pos_policy - self.default_joint_angles,
            joint_vel_policy * 0.05,
            self._last_high_action,
            self._robot_pos_xy,
            self._robot_vel_xyz[:2],
            self._cube_pos_xy,
            self._cube_vel_xy,
            self.goal_xy,
            np.array([self.goal_radius], dtype=np.float32),
            cube_to_goal,
            lf_foot_to_cube,
        ]

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        return obs

    def _build_low_level_obs(self, high_cmd: np.ndarray) -> np.ndarray:
        joint_pos_policy = self._motor_q[self.policy_to_motor]
        joint_vel_policy = self._motor_dq[self.policy_to_motor]

        obs_parts = [
            self._imu_gyro * self.ang_vel_scale,
            _projected_gravity_body_from_wxyz(self._imu_quat_wxyz),
            np.asarray(high_cmd, dtype=np.float32),
            (joint_pos_policy - self.default_joint_angles) * self.dof_pos_scale,
            joint_vel_policy * self.dof_vel_scale,
            self._last_low_action,
        ]

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        if obs.size != self.low_level_num_obs:
            self.get_logger().warn(
                f'Low-level observation has {obs.size} values, expected {self.low_level_num_obs}.',
                throttle_duration_sec=2.0,
            )
        return obs

    def _publish_motor_targets(self, target_policy_joints: np.ndarray) -> None:
        target_policy_joints = np.asarray(target_policy_joints, dtype=np.float32)
        target_motor = np.zeros(12, dtype=np.float32)

        for policy_idx, motor_idx in enumerate(self.policy_to_motor):
            if 0 <= motor_idx < 12:
                target_motor[motor_idx] = target_policy_joints[policy_idx]

        self._publish_joint_debug(target_motor)

        if self._lowcmd_pub is None or self._lowcmd_msg_cls is None:
            return

        cmd = self._lowcmd_msg_cls()
        if not hasattr(cmd, 'motor_cmd'):
            self.get_logger().error('LowCmd message does not contain motor_cmd field.')
            return

        _initialize_go2_lowcmd(cmd)
        n_motors = min(12, len(cmd.motor_cmd))
        for i in range(n_motors):
            m = cmd.motor_cmd[i]
            if hasattr(m, 'q'):
                m.q = float(target_motor[i])
            if hasattr(m, 'dq'):
                m.dq = 0.0
            elif hasattr(m, 'qd'):
                m.qd = 0.0
            if hasattr(m, 'kp'):
                m.kp = self.kp
            if hasattr(m, 'kd'):
                m.kd = self.kd
            if hasattr(m, 'tau'):
                m.tau = self.tau_ff

        self._lowcmd_pub.publish(cmd)

    def _publish_joint_debug(self, targets: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = targets.astype(np.float32).tolist()
        self._joint_target_pub.publish(msg)

    def _publish_debug_command(self, cmd: np.ndarray) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(cmd[0])
        msg.twist.linear.y = float(cmd[1])
        msg.twist.angular.z = float(cmd[2])
        self._command_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
