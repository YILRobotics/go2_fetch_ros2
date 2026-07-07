"""Pure helpers for the PushCube high-level policy observation contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimedCubeState:
    stamp_s: float
    position_world_xy: np.ndarray
    velocity_world_xy: np.ndarray


def reorder_and_correct_foot_force(raw_force, offset) -> np.ndarray:
    raw_force = np.asarray(raw_force, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    if raw_force.shape != (4,) or offset.shape != (4,):
        raise ValueError("foot force and offset must each contain four values")
    # Unitree [FR, FL, RR, RL] -> Isaac Lab [FL, FR, RL, RR].
    return (raw_force - offset)[[1, 0, 3, 2]]


def normalize_quaternion_wxyz(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite (w, x, y, z) values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def world_vector_to_base_xy(vector_world_xy, quaternion_world_from_base_wxyz) -> np.ndarray:
    """Apply the inverse full-quaternion rotation and return its base-frame XY."""
    qw, qx, qy, qz = normalize_quaternion_wxyz(quaternion_world_from_base_wxyz)
    rotation_world_from_base = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    vector_world = np.array(
        [float(vector_world_xy[0]), float(vector_world_xy[1]), 0.0], dtype=np.float64
    )
    return (rotation_world_from_base.T @ vector_world)[:2].astype(np.float32)


def select_cube_state(history, target_stamp_s: float) -> TimedCubeState | None:
    """Return the newest state at/before the target, or the oldest startup state."""
    if not history:
        return None
    eligible = [state for state in history if state.stamp_s <= target_stamp_s]
    return eligible[-1] if eligible else history[0]


def build_pushcube_observation(
    *,
    angular_velocity_base,
    projected_gravity,
    joint_position_relative,
    joint_velocity,
    previous_clamped_command,
    robot_position_world_xy,
    robot_velocity_world_xy,
    cube_position_world_xy,
    cube_velocity_world_xy,
    goal_position_world_xy,
    goal_radius: float,
    lf_foot_position_world_xy,
    foot_force,
    quaternion_world_from_base_wxyz,
    foot_force_scale: float = 100.0,
) -> np.ndarray:
    """Build the exact 52-value post-ff_5_2 high-level policy observation."""
    robot_goal_xy = np.asarray(robot_position_world_xy, dtype=np.float32) - goal_position_world_xy
    cube_goal_frame_xy = np.asarray(cube_position_world_xy, dtype=np.float32) - goal_position_world_xy
    cube_to_goal_world = np.asarray(goal_position_world_xy, dtype=np.float32) - cube_position_world_xy
    foot_to_cube_world = np.asarray(cube_position_world_xy, dtype=np.float32) - lf_foot_position_world_xy

    if foot_force_scale <= 0.0:
        raise ValueError("foot_force_scale must be greater than zero")
    terms = [
        np.clip(angular_velocity_base, -100.0, 100.0) * 0.2,
        np.clip(projected_gravity, -100.0, 100.0),
        np.clip(joint_position_relative, -100.0, 100.0),
        np.clip(joint_velocity, -100.0, 100.0) * 0.05,
        np.clip(previous_clamped_command, -100.0, 100.0),
        np.clip(robot_goal_xy, -100.0, 100.0) * 0.5,
        np.clip(robot_velocity_world_xy, -100.0, 100.0),
        np.clip(cube_goal_frame_xy, -100.0, 100.0) * 0.5,
        np.clip(world_vector_to_base_xy(cube_velocity_world_xy, quaternion_world_from_base_wxyz), -100.0, 100.0),
        np.zeros(2, dtype=np.float32),
        np.clip(np.array([goal_radius], dtype=np.float32), -100.0, 100.0),
        np.clip(world_vector_to_base_xy(cube_to_goal_world, quaternion_world_from_base_wxyz), -100.0, 100.0) * 0.5,
        np.clip(world_vector_to_base_xy(foot_to_cube_world, quaternion_world_from_base_wxyz), -100.0, 100.0),
        np.clip(foot_force, 0.0, 150.0) / float(foot_force_scale),
    ]
    observation = np.concatenate(terms).astype(np.float32)
    if observation.shape != (52,):
        raise ValueError(f"PushCube observation has shape {observation.shape}; expected (52,)")
    if not np.all(np.isfinite(observation)):
        raise ValueError("PushCube observation contains non-finite values")
    return observation
