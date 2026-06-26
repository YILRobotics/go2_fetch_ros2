#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""Helpers shared by the real Go2 deployment ROS node."""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
from typing import Sequence

import numpy as np


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16
        self.keys = 0

    def set(self, data):
        keys = struct.unpack("<H", data[2:4])[0]
        self.keys = keys
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


@dataclass
class DeployRealConfig:
    control_dt: float
    msg_type: str
    imu_type: str
    weak_motor: list[int]
    lowcmd_topic: str
    lowstate_topic: str
    policy_path: str
    leg_joint2motor_idx: list[int]
    kps: list[float]
    kds: list[float]
    torque_limits: list[float]
    default_angles: np.ndarray
    arm_waist_joint2motor_idx: list[int]
    arm_waist_kps: list[float]
    arm_waist_kds: list[float]
    arm_waist_target: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: np.ndarray
    max_cmd: np.ndarray
    num_actions: int
    num_obs: int


def add_unitree_sdk_paths(extra_paths: Sequence[str] | None = None) -> None:
    candidate_paths = [
        ".",
        "..",
        Path("/home/ferdinand/unitree/Deploy_SimToReal_RL_Go2"),
        Path("/home/ferdinand/unitree/Deploy_SimToReal_RL_Go2/unitree_sdk2_python"),
        Path("/home/ferdinand/unitree"),
        Path("/home/ferdinand/unitree/unitree_sdk2_python"),
    ]
    if extra_paths:
        candidate_paths.extend(Path(path).expanduser() for path in extra_paths if path)

    for path in candidate_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)


def create_zero_cmd(cmd) -> None:
    for motor_cmd in cmd.motor_cmd:
        set_motor_cmd_position(motor_cmd, 0.0)
        set_motor_cmd_velocity(motor_cmd, 0)
        set_motor_cmd_gains(motor_cmd, 0.0, 0.0)
        set_motor_cmd_torque(motor_cmd, 0.0)


def create_damping_cmd(cmd) -> None:
    for motor_cmd in cmd.motor_cmd:
        set_motor_cmd_position(motor_cmd, 0.0)
        set_motor_cmd_velocity(motor_cmd, 0)
        set_motor_cmd_gains(motor_cmd, 0.0, 8.0)
        set_motor_cmd_torque(motor_cmd, 0.0)


def init_cmd_go(cmd, weak_motor: Sequence[int] | None = None) -> None:
    weak_motor = weak_motor or []
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    pos_stop = 2.146e9
    vel_stop = 16000.0
    for i, motor_cmd in enumerate(cmd.motor_cmd):
        motor_cmd.mode = 1 if i in weak_motor else 0x0A
        set_motor_cmd_position(motor_cmd, pos_stop)
        set_motor_cmd_velocity(motor_cmd, vel_stop)
        set_motor_cmd_gains(motor_cmd, 0.0, 0.0)
        set_motor_cmd_torque(motor_cmd, 0.0)


def compute_go2_lowcmd_crc(cmd) -> int:
    pack_fmt = "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I"
    data = []
    data.extend(cmd.head)
    data.append(cmd.level_flag)
    data.append(cmd.frame_reserve)
    data.extend(cmd.sn)
    data.extend(cmd.version)
    data.append(cmd.bandwidth)

    for motor_cmd in cmd.motor_cmd:
        data.append(motor_cmd.mode)
        data.append(motor_cmd.q)
        data.append(motor_cmd.dq)
        data.append(motor_cmd.tau)
        data.append(motor_cmd.kp)
        data.append(motor_cmd.kd)
        data.extend(motor_cmd.reserve)

    data.append(cmd.bms_cmd.off)
    data.extend(cmd.bms_cmd.reserve)
    data.extend(cmd.wireless_remote)
    data.extend(cmd.led)
    data.extend(cmd.fan)
    data.append(cmd.gpio)
    data.append(cmd.reserve)
    data.append(cmd.crc)

    packed = struct.pack(pack_fmt, *data)
    words = []
    for i in range((len(packed) >> 2) - 1):
        words.append(
            (packed[i * 4 + 3] << 24)
            | (packed[i * 4 + 2] << 16)
            | (packed[i * 4 + 1] << 8)
            | packed[i * 4]
        )
    return _crc32_words(words)


def _crc32_words(words: Sequence[int]) -> int:
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for word in words:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if word & bit:
                crc ^= polynomial
            bit >>= 1
    return crc & 0xFFFFFFFF


def set_motor_cmd_position(motor_cmd, value: float) -> None:
    motor_cmd.q = float(value)


def set_motor_cmd_velocity(motor_cmd, value: float) -> None:
    value = float(value)
    if hasattr(motor_cmd, "dq"):
        motor_cmd.dq = value
    else:
        motor_cmd.qd = value


def set_motor_cmd_gains(motor_cmd, kp: float, kd: float) -> None:
    motor_cmd.kp = float(kp)
    motor_cmd.kd = float(kd)


def set_motor_cmd_torque(motor_cmd, value: float) -> None:
    motor_cmd.tau = float(value)


def get_gravity_orientation(quaternion) -> np.ndarray:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = -1 + 2 * (qx * qx + qy * qy)
    return gravity_orientation


def compute_velocity(theta1, theta2, theta3, thetav1, thetav2, thetav3, foot):
    vitessex = 0
    vitessey = 0
    vitessez = 0
    l1 = 0.21
    l2 = 0.23
    num_pate = 0

    for k in range(4):
        if foot[k] < 20:
            pate_au_sol = 0
        else:
            pate_au_sol = 1
            num_pate += 1

        vitessex += pate_au_sol * (
            (-l1 * sin(theta1[k]) - l2 * sin(theta1[k] + theta2[k])) * -thetav1[k]
            + (-l2 * sin(theta1[k] + theta2[k])) * -thetav2[k]
        )
        vitessey += pate_au_sol * (
            (l1 * cos(theta1[k]) + l2 * cos(theta1[k] + theta2[k])) * sin(theta3[k]) * -thetav1[k]
            + (l2 * cos(theta1[k] + theta2[k]) * sin(theta3[k])) * -thetav2[k]
            + (l1 * sin(theta1[k]) + l2 * sin(theta1[k] + theta2[k])) * cos(theta3[k]) * thetav3[k]
        )
        vitessez += pate_au_sol * (
            -l1 * cos(theta1[k]) * thetav1[k]
            - l2 * cos(theta1[k] + theta2[k]) * (thetav1[k] + thetav2[k])
        )

    if num_pate > 0:
        vitessex = vitessex / num_pate
        vitessey = vitessey / num_pate
        vitessez = vitessez / num_pate

    return vitessex, vitessey, vitessez


def copy_low_state_dds_to_ros(dds_msg, ros_msg) -> None:
    ros_msg.head = list(dds_msg.head)
    ros_msg.level_flag = dds_msg.level_flag
    ros_msg.frame_reserve = dds_msg.frame_reserve
    ros_msg.sn = list(dds_msg.sn)
    ros_msg.version = list(dds_msg.version)
    ros_msg.bandwidth = dds_msg.bandwidth
    ros_msg.tick = dds_msg.tick
    ros_msg.wireless_remote = list(dds_msg.wireless_remote)
    ros_msg.bit_flag = dds_msg.bit_flag
    ros_msg.adc_reel = dds_msg.adc_reel
    ros_msg.temperature_ntc1 = dds_msg.temperature_ntc1
    ros_msg.temperature_ntc2 = dds_msg.temperature_ntc2
    ros_msg.power_v = dds_msg.power_v
    ros_msg.power_a = dds_msg.power_a
    ros_msg.fan_frequency = list(dds_msg.fan_frequency)
    ros_msg.reserve = dds_msg.reserve
    ros_msg.crc = dds_msg.crc

    ros_msg.imu_state.quaternion = list(dds_msg.imu_state.quaternion)
    ros_msg.imu_state.gyroscope = list(dds_msg.imu_state.gyroscope)
    ros_msg.imu_state.accelerometer = list(dds_msg.imu_state.accelerometer)
    ros_msg.imu_state.rpy = list(dds_msg.imu_state.rpy)
    ros_msg.imu_state.temperature = dds_msg.imu_state.temperature

    for i in range(min(len(dds_msg.motor_state), len(ros_msg.motor_state))):
        ros_msg.motor_state[i].mode = dds_msg.motor_state[i].mode
        ros_msg.motor_state[i].q = dds_msg.motor_state[i].q
        ros_msg.motor_state[i].dq = dds_msg.motor_state[i].dq
        ros_msg.motor_state[i].ddq = dds_msg.motor_state[i].ddq
        ros_msg.motor_state[i].tau_est = dds_msg.motor_state[i].tau_est
        ros_msg.motor_state[i].q_raw = dds_msg.motor_state[i].q_raw
        ros_msg.motor_state[i].dq_raw = dds_msg.motor_state[i].dq_raw
        ros_msg.motor_state[i].ddq_raw = dds_msg.motor_state[i].ddq_raw
        ros_msg.motor_state[i].temperature = dds_msg.motor_state[i].temperature
        ros_msg.motor_state[i].lost = dds_msg.motor_state[i].lost
        ros_msg.motor_state[i].reserve = list(dds_msg.motor_state[i].reserve)

    ros_msg.bms_state.version_high = dds_msg.bms_state.version_high
    ros_msg.bms_state.version_low = dds_msg.bms_state.version_low
    ros_msg.bms_state.status = dds_msg.bms_state.status
    ros_msg.bms_state.soc = dds_msg.bms_state.soc
    ros_msg.bms_state.current = dds_msg.bms_state.current
    ros_msg.bms_state.cycle = dds_msg.bms_state.cycle
    ros_msg.bms_state.bq_ntc = list(dds_msg.bms_state.bq_ntc)
    ros_msg.bms_state.mcu_ntc = list(dds_msg.bms_state.mcu_ntc)
    ros_msg.bms_state.cell_vol = list(dds_msg.bms_state.cell_vol)

    ros_msg.foot_force = list(dds_msg.foot_force)
    ros_msg.foot_force_est = list(dds_msg.foot_force_est)
