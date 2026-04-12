from __future__ import annotations

import colorsys
import math
import random
from typing import TYPE_CHECKING

import torch
from pxr import Gf, Sdf, UsdPhysics, UsdShade

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sensors import ContactSensor
from unitree_rl_lab.tasks import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_rigid_body_mass_simple(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str = "add",
    min_mass: float = 1e-6,
):
    """Function-style mass randomization compatible with EventManager direct calls.

    This mirrors the common startup use-case of IsaacLab's mass randomization term.
    """
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses()
    default_mass = asset.data.default_mass

    # Reset selected entries to defaults before applying new randomization.
    masses[env_ids[:, None], body_ids] = default_mass[env_ids[:, None], body_ids].clone()

    low, high = mass_distribution_params
    samples = math_utils.sample_uniform(low, high, (len(env_ids), len(body_ids)), device="cpu")

    if operation == "add":
        masses[env_ids[:, None], body_ids] = masses[env_ids[:, None], body_ids] + samples
    elif operation == "scale":
        masses[env_ids[:, None], body_ids] = masses[env_ids[:, None], body_ids] * samples
    elif operation == "abs":
        masses[env_ids[:, None], body_ids] = samples
    else:
        raise ValueError(f"Unsupported mass randomization operation: {operation}")

    masses = torch.clamp(masses, min=min_mass)
    asset.root_physx_view.set_masses(masses, env_ids)


def randomize_floor_friction_per_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    static_friction_range: tuple[float, float] = (0.55, 1.25),
    dynamic_friction_range: tuple[float, float] = (0.45, 1.05),
    restitution_range: tuple[float, float] = (0.0, 0.02),
    terrain_material_prim_path: str = "/World/ground/terrain/physicsMaterial",
):
    """Randomize global terrain friction each reset call.

    Note: In this setup the terrain physics material is shared across environments, so
    the sampled values apply globally for the simulation step after each reset.
    """
    del env_ids

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(terrain_material_prim_path)
    if not prim or not prim.IsValid():
        return

    material_api = UsdPhysics.MaterialAPI(prim)
    if not material_api:
        material_api = UsdPhysics.MaterialAPI.Apply(prim)

    static_friction = float(
        math_utils.sample_uniform(
            static_friction_range[0],
            static_friction_range[1],
            (1,),
            device=env.device,
        )[0].item()
    )
    dynamic_friction = float(
        math_utils.sample_uniform(
            dynamic_friction_range[0],
            dynamic_friction_range[1],
            (1,),
            device=env.device,
        )[0].item()
    )
    restitution = float(
        math_utils.sample_uniform(
            restitution_range[0],
            restitution_range[1],
            (1,),
            device=env.device,
        )[0].item()
    )

    dynamic_friction = min(dynamic_friction, static_friction)

    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(restitution)


def set_cube_and_goal_matching_env_colors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    palette_size: int = 12,
    brightness_range: tuple[float, float] = (0.92, 1.08),
    saturation_range: tuple[float, float] = (0.90, 1.10),
    random_seed: int | None = None,
):
    """Assign one color per environment to both cube and goal marker.

    Colors come from a compact high-contrast palette and are randomly assigned to env ids.
    Brightness/saturation are lightly jittered per environment for extra separation.
    """
    stage = get_current_stage()
    if stage is None:
        return

    num_envs = int(env.scene.num_envs)
    if num_envs <= 0:
        return

    if env_ids is None:
        env_indices = torch.arange(num_envs, dtype=torch.long, device="cpu")
    else:
        env_indices = env_ids.to(device="cpu", dtype=torch.long).flatten()

    # High-contrast base colors (fewer colors with larger differences).
    high_contrast_palette = [
        (0.95, 0.20, 0.20),  # red
        (0.12, 0.55, 0.95),  # blue
        (0.95, 0.65, 0.15),  # orange
        (0.20, 0.78, 0.25),  # green
        (0.80, 0.22, 0.88),  # magenta
        (0.12, 0.82, 0.82),  # cyan
        (0.95, 0.92, 0.18),  # yellow
        (0.58, 0.34, 0.95),  # violet
        (0.95, 0.35, 0.60),  # pink
        (0.34, 0.86, 0.58),  # mint
        (0.80, 0.45, 0.18),  # amber-brown
        (0.25, 0.90, 0.55),  # lime
    ]
    palette_size = max(1, min(int(palette_size), len(high_contrast_palette)))
    palette = high_contrast_palette[:palette_size]

    env_list = env_indices.tolist()
    assign_rng = random.Random(random_seed)
    shuffled_envs = list(env_list)
    assign_rng.shuffle(shuffled_envs)

    env_to_color: dict[int, tuple[float, float, float]] = {}
    for idx, env_idx in enumerate(shuffled_envs):
        base_color = palette[idx % palette_size]
        seed_base = int(random_seed) if random_seed is not None else assign_rng.randint(0, 2**31 - 1)
        rng = random.Random(seed_base * 9973 + env_idx)

        # Small per-env saturation/brightness jitter in HSV space.
        h, s, v = colorsys.rgb_to_hsv(*base_color)
        sat = max(0.0, min(1.0, s * rng.uniform(saturation_range[0], saturation_range[1])))
        val = max(0.0, min(1.0, v * rng.uniform(brightness_range[0], brightness_range[1])))
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        env_to_color[env_idx] = (r, g, b)

    for env_idx in env_list:
        color = env_to_color[env_idx]

        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        for prim_name in ("Cube", "GoalArea"):
            shader_prim = stage.GetPrimAtPath(f"{env.scene.env_ns}/env_{env_idx}/{prim_name}/geometry/material/Shader")
            if not shader_prim or not shader_prim.IsValid():
                continue
            shader = UsdShade.Shader(shader_prim)
            diffuse_input = shader.GetInput("diffuseColor")
            if not diffuse_input:
                diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            diffuse_input.Set(color_vec)


def _goal_xy_world(env: ManagerBasedRLEnv, env_ids: torch.Tensor, goal_xy: tuple[float, float]) -> torch.Tensor:
    goal_xy_tensor = _cached_goal_xy_vector(env, goal_xy)
    return _scene_env_origins_xy(env)[env_ids, :] + goal_xy_tensor


def _scene_env_origins_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env XY origins matching cloned env Xforms used by {ENV_REGEX_NS} assets.

    With replicate_physics=False, InteractiveScene keeps cloned-grid origins in
    `_default_env_origins`, while `scene.env_origins` may come from terrain
    patch assignment. Push task assets (robot/cube/goal marker) are cloned under
    env Xforms, so we anchor them to `_default_env_origins` when available.
    """
    default_env_origins = getattr(env.scene, "_default_env_origins", None)
    if default_env_origins is not None:
        return default_env_origins[:, :2]
    return env.scene.env_origins[:, :2]


def _cached_env_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)
    cached = getattr(env, "_push_cached_env_ids", None)
    if cached is None or cached.shape[0] != env.num_envs or cached.device != device:
        cached = torch.arange(env.num_envs, device=device, dtype=torch.long)
        env._push_cached_env_ids = cached
    return cached


def _cached_goal_xy_vector(env: ManagerBasedRLEnv, goal_xy: tuple[float, float]) -> torch.Tensor:
    device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)
    goal_xy_key = (float(goal_xy[0]), float(goal_xy[1]))
    cache_key = getattr(env, "_push_goal_xy_vector_key", None)
    cached = getattr(env, "_push_goal_xy_vector", None)
    if cached is None or cache_key != (goal_xy_key, device.type, device.index) or cached.device != device:
        cached = torch.tensor(goal_xy_key, device=device, dtype=torch.float32)
        env._push_goal_xy_vector = cached
        env._push_goal_xy_vector_key = (goal_xy_key, device.type, device.index)
    return cached


def _cached_goal_radius_obs(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)
    radius = float(goal_radius)
    cache_key = getattr(env, "_push_goal_radius_obs_key", None)
    cached = getattr(env, "_push_goal_radius_obs", None)
    expected_key = (radius, env.num_envs, device.type, device.index)
    if cached is None or cache_key != expected_key or cached.device != device:
        cached = torch.full((env.num_envs, 1), radius, device=device)
        env._push_goal_radius_obs = cached
        env._push_goal_radius_obs_key = expected_key
    return cached


def _cube_goal_distance(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    goal_xy: tuple[float, float],
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    env_ids = _cached_env_ids(env)
    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)
    cube_xy_w = cube.data.root_pos_w[:, :2]
    return torch.linalg.norm(cube_xy_w - goal_xy_w, dim=1)


def _env_step_time_s(env: ManagerBasedRLEnv) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None:
        return max(float(step_dt), 1e-6)
    cfg = getattr(env, "cfg", None)
    if cfg is not None and getattr(cfg, "sim", None) is not None and hasattr(cfg, "decimation"):
        return max(float(cfg.sim.dt) * float(cfg.decimation), 1e-6)
    return 1.0


def _goal_hold_success_mask(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    goal_xy: tuple[float, float],
    goal_radius: float,
    cube_speed_threshold: float,
    hold_time_s: float,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)
    in_goal_and_slow = torch.logical_and(dist <= goal_radius, speed <= cube_speed_threshold)

    required_steps = max(1, int(math.ceil(float(hold_time_s) / _env_step_time_s(env))))

    if (
        not hasattr(env, "_push_goal_hold_counter")
        or env._push_goal_hold_counter.shape[0] != env.num_envs
    ):
        env._push_goal_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._push_goal_hold_last_step = -1

    current_step = int(env.common_step_counter)
    if getattr(env, "_push_goal_hold_last_step", -1) != current_step:
        reset_mask = env.episode_length_buf == 0
        env._push_goal_hold_counter[reset_mask] = 0
        env._push_goal_hold_counter = torch.where(
            in_goal_and_slow,
            env._push_goal_hold_counter + 1,
            torch.zeros_like(env._push_goal_hold_counter),
        )
        env._push_goal_hold_last_step = current_step

    return env._push_goal_hold_counter >= required_steps


def _curriculum_alpha(env: ManagerBasedRLEnv, transition_steps: int) -> float:
    if transition_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(env.common_step_counter) / float(transition_steps)))


def _symmetric_curriculum_range(limit_range: tuple[float, float], initial_abs: float, alpha: float) -> list[float]:
    # Keep command ranges symmetric around zero while ramping from a small start envelope.
    limit_abs = min(abs(float(limit_range[0])), abs(float(limit_range[1])))
    start_abs = max(0.0, min(float(initial_abs), limit_abs))
    current_abs = start_abs + (limit_abs - start_abs) * float(alpha)
    return [-current_abs, current_abs]


def command_velocity_envelope_stepwise_curriculum(
    env,
    env_ids,
    step_size: int = 5000,
    lin_vel_increment: float = 0.05,
    ang_vel_increment: float = 0.02,
    initial_lin_vel_abs: float = 0.05,
    initial_ang_vel_abs: float = 0.02,
    limit_lin_vel_x: float = 0.6,
    limit_lin_vel_y: float = 0.5,
    limit_ang_vel_z: float = 0.3,
    scale_back_vel: float = 1.0,
    scale_side_vel: float = 1.0,
):
    """Increase command velocity limits in steps after every step_size steps.

    Supports asymmetric envelopes:
    - backward speed can be scaled via ``scale_back_vel`` for x-command
    - lateral speed can be scaled via ``scale_side_vel`` for y-command
    """
    del env_ids
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Number of increments so far
    n_increments = int(env.common_step_counter // step_size)
    # Compute new abs limits
    lin_vel_x_abs = min(initial_lin_vel_abs + n_increments * lin_vel_increment, limit_lin_vel_x)
    lin_vel_y_abs = min(initial_lin_vel_abs + n_increments * lin_vel_increment, limit_lin_vel_y)
    ang_vel_abs = min(initial_ang_vel_abs + n_increments * ang_vel_increment, limit_ang_vel_z)
    back_scale = max(0.0, float(scale_back_vel))
    side_scale = max(0.0, float(scale_side_vel))

    # Set ranges (supports asymmetric backward / lateral scaling).
    ranges.lin_vel_x = [-lin_vel_x_abs * back_scale, lin_vel_x_abs]
    ranges.lin_vel_y = [-lin_vel_y_abs * side_scale, lin_vel_y_abs * side_scale]
    ranges.ang_vel_z = [-ang_vel_abs, ang_vel_abs]

    # For logging: return the current increment
    return lin_vel_x_abs


def curriculum_common_step_counter(env, env_ids):
    """Expose per-environment step progress for logger backends (TensorBoard/W&B)."""
    del env_ids
    return float(env.common_step_counter)


def curriculum_goal_reward_alpha(
    env,
    env_ids,
    transition_steps: int = 50_000,
):
    """Expose current goal curriculum alpha for logger backends (TensorBoard/W&B)."""
    del env_ids
    if transition_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(env.common_step_counter) / float(transition_steps)))


def _corrupt_xy_observation(
    env: ManagerBasedRLEnv,
    obs_xy: torch.Tensor,
    state_prefix: str,
    reset_mask: torch.Tensor,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    delay_steps: int = 0,
    spike_prob: float = 0.0,
    spike_std: float = 0.0,
) -> torch.Tensor:
    delay_steps = max(0, int(delay_steps))
    history_len = delay_steps + 1

    hist_name = f"{state_prefix}_hist"
    prev_name = f"{state_prefix}_prev"
    last_step_name = f"{state_prefix}_last_step"

    if (
        not hasattr(env, hist_name)
        or getattr(env, hist_name).shape != (env.num_envs, history_len, obs_xy.shape[1])
    ):
        setattr(env, hist_name, obs_xy.unsqueeze(1).repeat(1, history_len, 1).clone())
        setattr(env, prev_name, obs_xy.clone())
        setattr(env, last_step_name, -1)

    hist = getattr(env, hist_name)
    prev_obs = getattr(env, prev_name)
    current_step = int(env.common_step_counter)

    if getattr(env, last_step_name) != current_step:
        hist[reset_mask] = obs_xy[reset_mask].unsqueeze(1).repeat(1, history_len, 1)
        prev_obs[reset_mask] = obs_xy[reset_mask]

        if history_len > 1:
            hist[:, 1:, :] = hist[:, :-1, :].clone()
        hist[:, 0, :] = obs_xy

        delayed_obs = hist[:, delay_steps, :].clone()

        if noise_std > 0.0:
            delayed_obs = delayed_obs + torch.randn_like(delayed_obs) * float(noise_std)

        if spike_prob > 0.0 and spike_std > 0.0:
            spike_mask = (torch.rand(env.num_envs, device=env.device) < float(spike_prob)).unsqueeze(1)
            spike_noise = torch.randn_like(delayed_obs) * float(spike_std)
            delayed_obs = torch.where(spike_mask, delayed_obs + spike_noise, delayed_obs)

        if dropout_prob > 0.0:
            dropout_mask = (torch.rand(env.num_envs, device=env.device) < float(dropout_prob)).unsqueeze(1)
            delayed_obs = torch.where(dropout_mask, prev_obs, delayed_obs)

        delayed_obs[reset_mask] = obs_xy[reset_mask]
        prev_obs[:] = delayed_obs
        setattr(env, hist_name, hist)
        setattr(env, prev_name, prev_obs)
        setattr(env, last_step_name, current_step)

    return getattr(env, prev_name)


def cube_position_xy(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    delay_steps: int = 0,
    spike_prob: float = 0.0,
    spike_std: float = 0.0,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_xy = cube.data.root_pos_w[:, :2] - _scene_env_origins_xy(env)
    reset_mask = env.episode_length_buf == 0
    return _corrupt_xy_observation(
        env=env,
        obs_xy=cube_xy,
        state_prefix="_push_cube_pos_obs",
        reset_mask=reset_mask,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
        delay_steps=delay_steps,
        spike_prob=spike_prob,
        spike_std=spike_std,
    )

# Cube's linear velocity in XY plane. From previous position and velocity and current position 
def cube_linear_velocity_xy(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    delay_steps: int = 0,
    spike_prob: float = 0.0,
    spike_std: float = 0.0,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_xy = cube.data.root_pos_w[:, :2]

    if (
        not hasattr(env, "_push_prev_cube_pos_xy_obs")
        or env._push_prev_cube_pos_xy_obs.shape != cube_xy.shape
    ):
        env._push_prev_cube_pos_xy_obs = cube_xy.clone()
        env._push_prev_cube_vel_xy_obs = torch.zeros_like(cube_xy)
        env._push_cube_vel_obs_last_step = -1

    current_step = int(env.common_step_counter)
    if getattr(env, "_push_cube_vel_obs_last_step", -1) != current_step:
        dt = _env_step_time_s(env) # high level policy dt (~15hz)
        reset_mask = env.episode_length_buf == 0
        env._push_prev_cube_pos_xy_obs[reset_mask] = cube_xy[reset_mask]

        vel_xy = (cube_xy - env._push_prev_cube_pos_xy_obs) / dt
        vel_xy[reset_mask] = 0.0

        env._push_prev_cube_vel_xy_obs = vel_xy
        env._push_prev_cube_pos_xy_obs = cube_xy.clone()
        env._push_cube_vel_obs_last_step = current_step

    return _corrupt_xy_observation(
        env=env,
        obs_xy=env._push_prev_cube_vel_xy_obs,
        state_prefix="_push_cube_vel_obs",
        reset_mask=env.episode_length_buf == 0,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
        delay_steps=delay_steps,
        spike_prob=spike_prob,
        spike_std=spike_std,
    )


def goal_position_xy(env: ManagerBasedRLEnv, goal_xy: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    goal_xy_tensor = _cached_goal_xy_vector(env, goal_xy)
    return goal_xy_tensor.unsqueeze(0).expand(env.num_envs, -1)


def goal_radius_obs(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    return _cached_goal_radius_obs(env, goal_radius)


def cube_to_goal_vector_xy(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    return goal_position_xy(env, goal_xy=goal_xy) - cube_position_xy(env, cube_cfg=cube_cfg)


def left_front_foot_to_cube_vector_xy(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot: Articulation = env.scene[foot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    foot_xy = robot.data.body_pos_w[:, foot_cfg.body_ids, :2].mean(dim=1)
    cube_xy = cube.data.root_pos_w[:, :2]
    return cube_xy - foot_xy


def robot_position_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, :2] - _scene_env_origins_xy(env)


def robot_linear_velocity_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_lin_vel_w[:, :2]

# This function gives a positive reward for the cube getting closer to the goal and a negative reward for moving away. 
# The current distance from the cube to the goal is stored and compared to the previous step's distance to compute the progress.
def cube_to_goal_progress_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    transition_steps: int = 50_000,
) -> torch.Tensor:
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    if not hasattr(env, "_push_prev_cube_goal_dist"):
        env._push_prev_cube_goal_dist = dist.clone()

    reset_mask = env.episode_length_buf == 0
    env._push_prev_cube_goal_dist[reset_mask] = dist[reset_mask]

    progress = env._push_prev_cube_goal_dist - dist
    env._push_prev_cube_goal_dist[:] = dist

    return _curriculum_alpha(env, transition_steps) * progress


def success_bonus_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    hold_time_s: float = 1.0,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    is_success = _goal_hold_success_mask(
        env=env,
        cube_cfg=cube_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        cube_speed_threshold=cube_speed_threshold,
        hold_time_s=hold_time_s,
    )
    return _curriculum_alpha(env, transition_steps) * is_success.float()


def success_trigger_reward(
    # """Compute a one-shot success reward when the task first transitions from unsuccessful to successful.

    # This reward is emitted only on the **False -> True** edge of the success condition
    # (per environment), not continuously while success remains true. The success condition
    # is defined by `_goal_hold_success_mask(...)` (cube within goal radius, below speed
    # threshold, and held for the required duration).

    # The function stores per-environment internal state on `env` to detect transitions:

    # - `_push_prev_success_mask`: previous-step success boolean mask.
    # - `_push_success_trigger_flag`: current-step one-shot trigger mask.
    # - `_push_success_trigger_last_step`: guard to avoid recomputation within the same global step.

    # On episode reset (`env.episode_length_buf == 0`), previous success is cleared so a new
    # success transition can be detected in the new episode.

    # The returned reward is scaled by curriculum factor `_curriculum_alpha(env, transition_steps)`.

    # Args:
    #     env: Manager-based RL environment containing scene tensors and step counters.
    #     cube_cfg: Scene entity config used to locate cube state (default: `"cube"`).
    #     goal_xy: 2D world-frame goal position `(x, y)`.
    #     goal_radius: Distance threshold for considering cube at goal.
    #     cube_speed_threshold: Maximum cube speed to count as settled.
    #     hold_time_s: Required continuous success duration before success is true.
    #     transition_steps: Number of steps over which curriculum scaling ramps.

    # Returns:
    #     torch.Tensor: Float tensor of shape `(num_envs,)` with values in `{0.0, alpha}`,
    #     where `alpha = _curriculum_alpha(env, transition_steps)`. Non-zero only on the
    #     first step success becomes true for each environment.
    # """
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    hold_time_s: float = 1.0,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    """One-shot success reward on the False->True transition of the success condition."""
    is_success = _goal_hold_success_mask(
        env=env,
        cube_cfg=cube_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        cube_speed_threshold=cube_speed_threshold,
        hold_time_s=hold_time_s,
    )

    trigger = _one_shot_bool_trigger(env=env, mask=is_success, state_prefix="_push_success_trigger")
    return _curriculum_alpha(env, transition_steps) * trigger.float()


def success_trigger_reward_robot_outsid_goal(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot.*"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    cube_in_goal_additional_margin: float = 0.15,
    hold_time_s: float = 1.0,
    robot_speed_threshold: float = 0.10,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    """One-shot success reward matching the full termination success condition."""
    is_success = cube_goal_reached_robot_outsid_goal(
        env=env,
        cube_cfg=cube_cfg,
        foot_cfg=foot_cfg,
        robot_cfg=robot_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        cube_speed_threshold=cube_speed_threshold,
        cube_in_goal_additional_margin=cube_in_goal_additional_margin,
        hold_time_s=hold_time_s,
        robot_speed_threshold=robot_speed_threshold,
    )
    trigger = _one_shot_bool_trigger(env=env, mask=is_success, state_prefix="_push_success_trigger_robot_outside")
    return _curriculum_alpha(env, transition_steps) * trigger.float()


def _one_shot_bool_trigger(
    env: ManagerBasedRLEnv,
    mask: torch.Tensor,
    state_prefix: str,
) -> torch.Tensor:
    prev_name = f"{state_prefix}_prev"
    flag_name = f"{state_prefix}_flag"
    last_step_name = f"{state_prefix}_last_step"

    if not hasattr(env, prev_name) or getattr(env, prev_name).shape[0] != env.num_envs:
        setattr(env, prev_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        setattr(env, flag_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        setattr(env, last_step_name, -1)

    current_step = int(env.common_step_counter)
    if getattr(env, last_step_name, -1) != current_step:
        prev_mask = getattr(env, prev_name)
        reset_mask = env.episode_length_buf == 0
        prev_mask[reset_mask] = False
        trigger = torch.logical_and(~prev_mask, mask)
        setattr(env, flag_name, trigger)
        setattr(env, prev_name, mask.clone())
        setattr(env, last_step_name, current_step)

    return getattr(env, flag_name)


def cube_settled_in_goal_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    hold_time_s: float = 1.0,
    vel_std: float = 0.12,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)
    in_goal = _goal_hold_success_mask(
        env=env,
        cube_cfg=cube_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        cube_speed_threshold=cube_speed_threshold,
        hold_time_s=hold_time_s,
    )
    settle_score = torch.exp(-torch.square(speed) / (vel_std * vel_std))
    return _curriculum_alpha(env, transition_steps) * in_goal.float() * settle_score


def cube_in_goal_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    in_goal = dist <= goal_radius
    return _curriculum_alpha(env, transition_steps) * in_goal.float()


def goal_hold_progress_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    hold_time_s: float = 1.0,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    # Ensure hold counter state is updated for this step.
    _goal_hold_success_mask(
        env=env,
        cube_cfg=cube_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        cube_speed_threshold=cube_speed_threshold,
        hold_time_s=hold_time_s,
    )

    required_steps = max(1, int(math.ceil(float(hold_time_s) / _env_step_time_s(env))))
    hold_counter = env._push_goal_hold_counter

    cube: RigidObject = env.scene[cube_cfg.name]
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)
    in_goal_and_slow = torch.logical_and(dist <= goal_radius, speed <= cube_speed_threshold)
    not_yet_done = hold_counter < required_steps

    return _curriculum_alpha(env, transition_steps) * torch.logical_and(in_goal_and_slow, not_yet_done).float()


def cube_exit_goal_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    in_goal = dist <= goal_radius

    if (
        not hasattr(env, "_push_prev_in_goal")
        or env._push_prev_in_goal.shape[0] != env.num_envs
    ):
        env._push_prev_in_goal = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._push_had_goal_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._push_goal_exit_last_step = -1

    current_step = int(env.common_step_counter)
    if getattr(env, "_push_goal_exit_last_step", -1) != current_step:
        reset_mask = env.episode_length_buf == 0
        env._push_prev_in_goal[reset_mask] = False
        env._push_had_goal_contact[reset_mask] = False

        just_exited = torch.logical_and(env._push_prev_in_goal, ~in_goal)
        exit_after_being_inside = torch.logical_and(just_exited, env._push_had_goal_contact)
        env._push_goal_exit_flag = exit_after_being_inside

        env._push_had_goal_contact = torch.logical_or(env._push_had_goal_contact, in_goal)
        env._push_prev_in_goal = in_goal.clone()
        env._push_goal_exit_last_step = current_step

    return _curriculum_alpha(env, transition_steps) * env._push_goal_exit_flag.float()


def robot_in_goal_area_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    margin: float = 0.05,
    transition_steps: int = 50_000,
    start_offset_steps: int = 0,
) -> torch.Tensor:
    """Penalty indicator when robot base enters goal area (with optional margin)."""
    robot: Articulation = env.scene[robot_cfg.name]
    env_ids = _cached_env_ids(env)
    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)
    robot_xy_w = robot.data.root_pos_w[:, :2]
    robot_goal_dist = torch.linalg.norm(robot_xy_w - goal_xy_w, dim=1)
    is_inside = robot_goal_dist <= (goal_radius + margin)

    current_steps = float(env.common_step_counter)
    offset = max(0.0, float(start_offset_steps))
    if current_steps < offset:
        schedule = 0.0
    else:
        shifted_steps = current_steps - offset
        if transition_steps <= 0:
            schedule = 1.0
        else:
            schedule = min(1.0, max(0.0, shifted_steps / float(transition_steps)))
    return is_inside.float() * schedule


def robot_stop_after_goal_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_in_goal_additional_margin: float = 0.15,
    vel_std: float = 0.08,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    """Reward low robot speed once the cube is clearly inside the goal."""
    robot: Articulation = env.scene[robot_cfg.name]
    robot_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    effective_goal_radius = max(1e-3, goal_radius - cube_in_goal_additional_margin)
    cube_dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    cube_in_goal = cube_dist <= effective_goal_radius
    stop_score = torch.exp(-torch.square(robot_speed) / (vel_std * vel_std))
    return _curriculum_alpha(env, transition_steps) * cube_in_goal.float() * stop_score


def robot_to_cube_approach_progress_reward(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="FL_foot.*"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    cube_far_distance: float = 0.7,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    vec = left_front_foot_to_cube_vector_xy(env, foot_cfg=foot_cfg, cube_cfg=cube_cfg)
    dist = torch.linalg.norm(vec, dim=1)

    if not hasattr(env, "_push_prev_foot_cube_dist"):
        env._push_prev_foot_cube_dist = dist.clone()

    reset_mask = env.episode_length_buf == 0
    env._push_prev_foot_cube_dist[reset_mask] = dist[reset_mask]

    progress = env._push_prev_foot_cube_dist - dist
    env._push_prev_foot_cube_dist[:] = dist

    cube_goal_dist = _cube_goal_distance(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    gate = (cube_goal_dist > cube_far_distance).float()
    return (1.0 - _curriculum_alpha(env, transition_steps)) * gate * progress

# This function gives a positive reward for the cube moving in the direction of the goal. It computes
# the velocity of the cube in the XY plane and projects it onto the direction vector from the cube to
# the goal. Only positive projections (moving towards the goal) are rewarded.
def push_direction_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    goal_margin: float = 0.0,
    transition_steps: int = 50_000,
    speed_threshold: float = 0.05,
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_goal_vec = cube_to_goal_vector_xy(env, cube_cfg=cube_cfg, goal_xy=goal_xy)
    cube_goal_dir = cube_goal_vec / (torch.linalg.norm(cube_goal_vec, dim=1, keepdim=True) + 1e-6)
    cube_vel_xy = cube.data.root_lin_vel_w[:, :2]
    toward_goal_speed = torch.sum(cube_vel_xy * cube_goal_dir, dim=1)
    cube_speed = torch.linalg.norm(cube_vel_xy, dim=1)
    moving_mask = (cube_speed > speed_threshold).float()
    cube_goal_dist = torch.linalg.norm(cube_goal_vec, dim=1)
    outside_goal_mask = (cube_goal_dist > (goal_radius + goal_margin)).float()
    return _curriculum_alpha(env, transition_steps) * moving_mask * outside_goal_mask * torch.clamp(toward_goal_speed, min=0.0)


def forward_push_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    cube_speed_threshold: float = 0.03,
    forward_speed_threshold: float = 0.05,
    max_robot_cube_distance: float = 0.8,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    """Reward pushing while moving forward: robot forward speed + cube motion + proximity gate."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    # Prefer body-frame forward velocity if available; otherwise fall back to world-frame x velocity.
    robot_forward_speed = (
        robot.data.root_lin_vel_b[:, 0]
        if hasattr(robot.data, "root_lin_vel_b")
        else robot.data.root_lin_vel_w[:, 0]
    )
    cube_speed = torch.linalg.norm(cube.data.root_lin_vel_w[:, :2], dim=1)

    robot_xy = robot.data.root_pos_w[:, :2]
    cube_xy = cube.data.root_pos_w[:, :2]
    robot_cube_dist = torch.linalg.norm(robot_xy - cube_xy, dim=1)

    cube_moving_mask = (cube_speed > cube_speed_threshold).float()
    forward_mask = (robot_forward_speed > forward_speed_threshold).float()
    proximity_mask = (robot_cube_dist < max_robot_cube_distance).float()

    forward_push_intensity = torch.clamp(robot_forward_speed, min=0.0) * torch.clamp(cube_speed, min=0.0)
    return _curriculum_alpha(env, transition_steps) * cube_moving_mask * forward_mask * proximity_mask * forward_push_intensity


def backward_body_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    deadzone: float = 0.02,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    """Penalize backward base velocity in robot body frame (x < 0).

    A small deadzone avoids punishing tiny estimator/noise jitter around zero.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    backward_speed = torch.clamp(-robot.data.root_lin_vel_b[:, 0] - deadzone, min=0.0)
    return _curriculum_alpha(env, transition_steps) * backward_speed


def cube_to_nearest_foot_distance_penalty(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot.*"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    max_distance: float = 0.35,
    transition_steps: int = 50_000,
) -> torch.Tensor:
    robot: Articulation = env.scene[foot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    foot_xy = robot.data.body_pos_w[:, foot_cfg.body_ids, :2]
    cube_xy = cube.data.root_pos_w[:, :2].unsqueeze(1)
    nearest_foot_dist = torch.linalg.norm(foot_xy - cube_xy, dim=2).min(dim=1).values
    excess_dist = torch.clamp(nearest_foot_dist - max_distance, min=0.0)
    return _curriculum_alpha(env, transition_steps) * excess_dist


def cube_goal_reached(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    hold_time_s: float = 1.0,
) -> torch.Tensor:
    return _goal_hold_success_mask(
        env=env,
        cube_cfg=cube_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        cube_speed_threshold=cube_speed_threshold,
        hold_time_s=hold_time_s,
    )


def cube_goal_reached_robot_outsid_goal(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot.*"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_speed_threshold: float = 0.05,
    cube_in_goal_additional_margin: float = 0.15,
    hold_time_s: float = 1.0,
    robot_speed_threshold: float = 0.10,
) -> torch.Tensor:
    cube_success = _goal_hold_success_mask(
        env=env,
        cube_cfg=cube_cfg,
        goal_xy=goal_xy,
        goal_radius=goal_radius - cube_in_goal_additional_margin,
        cube_speed_threshold=cube_speed_threshold,
        hold_time_s=hold_time_s,
    )

    robot: Articulation = env.scene[foot_cfg.name]
    env_ids = _cached_env_ids(env)
    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)
    foot_xy_w = robot.data.body_pos_w[:, foot_cfg.body_ids, :2]
    foot_goal_dist = torch.linalg.norm(foot_xy_w - goal_xy_w.unsqueeze(1), dim=2)
    all_feet_outside_goal = torch.all(foot_goal_dist > goal_radius, dim=1)

    robot_root: Articulation = env.scene[robot_cfg.name]
    robot_speed = torch.linalg.norm(robot_root.data.root_lin_vel_w[:, :2], dim=1)
    robot_slow = robot_speed <= robot_speed_threshold

    return torch.logical_and(torch.logical_and(cube_success, all_feet_outside_goal), robot_slow)


def reset_robot_and_cube_uniform_around_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_spawn_radius_range: tuple[float, float] = (0.8, 2.0),
    cube_height: float = 0.10,
    cube_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    robot_spawn_radius_range: tuple[float, float] = (0.4, 1.2),
    robot_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    robot_velocity_range: dict[str, tuple[float, float]] | None = None,
):
    if robot_velocity_range is None:
        robot_velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    num_resets = len(env_ids)
    device = env.device

    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)

    # Randomize cube in an annulus around the goal and keep it outside the goal region.
    cube_min_radius = max(cube_spawn_radius_range[0], goal_radius)
    cube_max_radius = max(cube_spawn_radius_range[1], cube_min_radius + 1e-3)
    cube_radius = math_utils.sample_uniform(cube_min_radius, cube_max_radius, (num_resets,), device=device)
    cube_angle = math_utils.sample_uniform(-torch.pi, torch.pi, (num_resets,), device=device)
    cube_xy = goal_xy_w + torch.stack((cube_radius * torch.cos(cube_angle), cube_radius * torch.sin(cube_angle)), dim=1)

    cube_yaw = math_utils.sample_uniform(cube_yaw_range[0], cube_yaw_range[1], (num_resets,), device=device)
    cube_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_resets, device=device),
        torch.zeros(num_resets, device=device),
        cube_yaw,
    )
    cube_pos = torch.cat((cube_xy, torch.full((num_resets, 1), cube_height, device=device)), dim=1)
    cube_vel = torch.zeros((num_resets, 6), device=device)
    cube.write_root_pose_to_sim(torch.cat((cube_pos, cube_quat), dim=1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(cube_vel, env_ids=env_ids)

    # Randomize robot pose around the cube and keep a minimum spacing to avoid overlaps.
    robot_min_radius = robot_spawn_radius_range[0]
    robot_max_radius = max(robot_spawn_radius_range[1], robot_min_radius + 1e-3)
    robot_radius = math_utils.sample_uniform(robot_min_radius, robot_max_radius, (num_resets,), device=device)
    robot_angle = math_utils.sample_uniform(-torch.pi, torch.pi, (num_resets,), device=device)
    robot_xy = cube_xy + torch.stack((robot_radius * torch.cos(robot_angle), robot_radius * torch.sin(robot_angle)), dim=1)

    robot_root_state = robot.data.default_root_state[env_ids].clone()
    robot_yaw = math_utils.sample_uniform(robot_yaw_range[0], robot_yaw_range[1], (num_resets,), device=device)
    robot_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_resets, device=device),
        torch.zeros(num_resets, device=device),
        robot_yaw,
    )

    robot_pos = torch.cat((robot_xy, robot_root_state[:, 2:3]), dim=1)

    range_list = [robot_velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(range_list, device=device)
    robot_vel = math_utils.sample_uniform(
        vel_ranges[:, 0],
        vel_ranges[:, 1],
        (num_resets, 6),
        device=device,
    )

    robot.write_root_pose_to_sim(torch.cat((robot_pos, robot_quat), dim=1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_vel, env_ids=env_ids)


def respawn_cube_uniform_around_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_spawn_radius_range: tuple[float, float] = (0.8, 2.0),
    cube_height: float = 0.10,
    cube_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
):
    """Respawn only the cube while keeping the robot state unchanged."""
    cube: RigidObject = env.scene[cube_cfg.name]

    num_resets = len(env_ids)
    device = env.device
    goal_xy_w = _goal_xy_world(env, env_ids, goal_xy)

    cube_min_radius = max(cube_spawn_radius_range[0], goal_radius)
    cube_max_radius = max(cube_spawn_radius_range[1], cube_min_radius + 1e-3)
    cube_radius = math_utils.sample_uniform(cube_min_radius, cube_max_radius, (num_resets,), device=device)
    cube_angle = math_utils.sample_uniform(-torch.pi, torch.pi, (num_resets,), device=device)
    cube_xy = goal_xy_w + torch.stack((cube_radius * torch.cos(cube_angle), cube_radius * torch.sin(cube_angle)), dim=1)

    cube_yaw = math_utils.sample_uniform(cube_yaw_range[0], cube_yaw_range[1], (num_resets,), device=device)
    cube_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_resets, device=device),
        torch.zeros(num_resets, device=device),
        cube_yaw,
    )
    cube_pos = torch.cat((cube_xy, torch.full((num_resets, 1), cube_height, device=device)), dim=1)
    cube_vel = torch.zeros((num_resets, 6), device=device)
    cube.write_root_pose_to_sim(torch.cat((cube_pos, cube_quat), dim=1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(cube_vel, env_ids=env_ids)


def no_op_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    del env, env_ids


def reset_push_episode_by_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_xy: tuple[float, float] = (0.0, 0.0),
    goal_radius: float = 0.35,
    cube_spawn_radius_range: tuple[float, float] = (0.8, 2.0),
    cube_height: float = 0.10,
    cube_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    robot_spawn_radius_range: tuple[float, float] = (0.4, 1.2),
    robot_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    robot_velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (1.0, 1.0),
    joint_velocity_range: tuple[float, float] = (-1.0, 1.0),
):
    """Branch push-task reset behavior by termination cause.

    Success: keep the robot where it is and only respawn the cube.
    Failure/timeout: fully reset robot pose, cube pose, and joints.
    """
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if env_ids.numel() == 0:
        return

    success_mask = env.termination_manager.get_term("success")[env_ids]
    success_env_ids = env_ids[success_mask]
    failure_env_ids = env_ids[~success_mask]

    if success_env_ids.numel() > 0:
        respawn_cube_uniform_around_goal(
            env=env,
            env_ids=success_env_ids,
            cube_cfg=cube_cfg,
            goal_xy=goal_xy,
            goal_radius=goal_radius,
            cube_spawn_radius_range=cube_spawn_radius_range,
            cube_height=cube_height,
            cube_yaw_range=cube_yaw_range,
        )

    if failure_env_ids.numel() > 0:
        reset_robot_and_cube_uniform_around_goal(
            env=env,
            env_ids=failure_env_ids,
            cube_cfg=cube_cfg,
            robot_cfg=robot_cfg,
            goal_xy=goal_xy,
            goal_radius=goal_radius,
            cube_spawn_radius_range=cube_spawn_radius_range,
            cube_height=cube_height,
            cube_yaw_range=cube_yaw_range,
            robot_spawn_radius_range=robot_spawn_radius_range,
            robot_yaw_range=robot_yaw_range,
            robot_velocity_range=robot_velocity_range,
        )
        mdp.reset_joints_by_scale(
            env=env,
            env_ids=failure_env_ids,
            position_range=joint_position_range,
            velocity_range=joint_velocity_range,
            asset_cfg=robot_cfg,
        )
