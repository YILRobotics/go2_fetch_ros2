import os
import random
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from unitree_rl_lab.assets.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks import mdp

from . import push_mdp
from .velocity_4l_env_cfg import RobotEnvCfg as LowLevel4LEnvCfg

SIM_DT = 0.005
GOAL_XY = (0.0, 0.0)
GOAL_RADIUS_M = 0.2

HIGH_LEVEL_POLICY_HZ = 15.0

# Curriculum step parameters
CMD_CURRICULUM_STEP_SIZE = 500  # Number of steps before each increment ((env.common_step_counter) = total_steps/num_envs)
CMD_CURRICULUM_LIN_VEL_INCREMENT = 0.05  # Linear velocity increment per step
CMD_CURRICULUM_ANG_VEL_INCREMENT = 0.02  # Angular velocity increment per step
CMD_INIT_LIN_VEL_ABS = 0.05 # Initial value
CMD_INIT_ANG_VEL_ABS = 0.02
CMD_LIMIT_LIN_VEL_X_ABS = 0.5 # Final limit
CMD_LIMIT_LIN_VEL_Y_ABS = 0.5
CMD_LIMIT_ANG_VEL_Z_ABS = 0.25

SCALE_BACK_VEL = 1.0 # used to reduce use of backward vel but working as good. 
SCALE_SIDE_VEL = 1.0 # used to reduce use of side vel but working as good

TRANSITION_STEPS = 8000 # number of common steps (total steps / num_envs) over which to linearly transition the reward from dense to sparse.

SUCCESS_CUBE_SPEED_THRESHOLD = 0.05
SUCCESS_HOLD_TIME_S = 0.6
SUCCESS_CUBE_IN_GOAL_ADDITIONAL_MARGIN = 0.05
SUCCESS_ROBOT_SPEED_THRESHOLD = 0.15

CUBE_POS_OBS_NOISE_STD = 0.015 # m
CUBE_VEL_OBS_NOISE_STD = 0.08 # m/s
CUBE_POS_OBS_DROPOUT_PROB = 0.05 
CUBE_VEL_OBS_DROPOUT_PROB = 0.08
CUBE_POS_OBS_DELAY_STEPS = 1
CUBE_VEL_OBS_DELAY_STEPS = 1
CUBE_POS_OBS_SPIKE_PROB = 0.01
CUBE_VEL_OBS_SPIKE_PROB = 0.01
CUBE_POS_OBS_SPIKE_STD = 0.05
CUBE_VEL_OBS_SPIKE_STD = 0.15

def _hz_to_decimation(policy_hz: float, sim_dt: float) -> int:
    return max(1, int(round(1.0 / (sim_dt * policy_hz))))


LOW_LEVEL_ENV_CFG = LowLevel4LEnvCfg() # 50 hz
LOW_LEVEL_TRAINED_HZ = 1.0 / (LOW_LEVEL_ENV_CFG.sim.dt * LOW_LEVEL_ENV_CFG.decimation)
LOW_LEVEL_POLICY_HZ = float(os.getenv("GO2_PUSH_LOW_LEVEL_HZ", f"{LOW_LEVEL_TRAINED_HZ:.6f}"))

HIGH_LEVEL_DECIMATION = _hz_to_decimation(HIGH_LEVEL_POLICY_HZ, SIM_DT)
LOW_LEVEL_DECIMATION = _hz_to_decimation(LOW_LEVEL_POLICY_HZ, SIM_DT)


def _resolve_low_level_policy_path() -> str:
    def _normalize_and_validate(path: Path) -> str:
        # If a training checkpoint is passed, map to exported TorchScript policy if available.
        if path.name.startswith("model_") and path.suffix == ".pt":
            exported_candidate = path.parent / "exported" / "policy.pt"
            if exported_candidate.is_file():
                path = exported_candidate
            else:
                raise FileNotFoundError(
                    f"Received checkpoint '{path}'. The push low-level policy requires a TorchScript file "
                    f"at '{exported_candidate}'. Export it first (or point to an existing exported/policy.pt)."
                )

        if not path.is_file():
            raise FileNotFoundError(f"Low-level policy file not found: {path}")

        path_str = path.as_posix()
        if not path_str.endswith("/exported/policy.pt"):
            raise ValueError(
                "Low-level policy must be a TorchScript export named exported/policy.pt, "
                f"got: {path}"
            )

        # Enforce using the 4-leg velocity policy family for push low-level walking.
        if "unitree_go2_velocity" not in path_str:
            raise ValueError(
                "Low-level policy must come from the Unitree-Go2-Velocity-4L training family "
                f"(expected path containing 'unitree_go2_velocity'), got: {path}"
            )

        return str(path)

    policy_override = os.getenv("GO2_PUSH_LOW_LEVEL_POLICY_PATH", "").strip()
    if policy_override:
        return _normalize_and_validate(Path(policy_override).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[4]
    candidates = sorted(repo_root.glob("logs/rsl_rl/unitree_go2_velocity/*/exported/policy.pt"))
    if candidates:
        return _normalize_and_validate(candidates[-1].resolve())

    return _normalize_and_validate(
        (repo_root / "logs/rsl_rl/unitree_go2_velocity/2026-03-25_23-05-55_e30_allterain/exported/policy.pt").resolve()
    )


LOW_LEVEL_POLICY_PATH = _resolve_low_level_policy_path()


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=10.0,
    num_rows=100, # 65
    num_cols=100, # 65
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.02), noise_step=0.01, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1,
        #     slope_range=(0.0, 0.1),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1,
        #     slope_range=(0.0, 0.4),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
    },
)


# PUSH_MIXED_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=10.0,
#     num_rows=8,
#     num_cols=3,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     sub_terrains={
#         # flat
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.34),
#         # slanted
#         "sloped": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.33,
#             slope_range=(0.0, 0.005),
#             platform_width=2.2,
#             border_width=0.2,
#         ),
#         # slightly uneven
#         "slightly_uneven": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.33, # means approximately 1-2 patches of noise per 1 patch of slope
#             # Keep values aligned with vertical_scale=0.005 to avoid zero height-step quantization.
#             noise_range=(0.001, 0.005),
#             noise_step=0.005, # multiple of vertical_scale to avoid zero height-step quantization
#             border_width=0.2,
#         ),
#     },
# )


@configclass
class LowLevelObsCfg(ObsGroup):
    """Low-level observation terms expected by the pretrained 4L locomotion policy."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene for push task with a Go2 robot and a rigid cube target object."""
    replicate_physics = False

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.80,
            dynamic_friction=0.65,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.10), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.095, 0.095, 0.095),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(density=60.0), # Mass is calculated from here. Range for EVA or polyurethane is 20-120 
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.65, restitution=0.0),
        ),
    )

    goal_area = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalArea",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=GOAL_RADIUS_M,
            height=0.004,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Events for randomization and reset."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup", # startup: called once at the beginning of training. reset: called at every env reset.
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=push_mdp.randomize_rigid_body_mass_simple,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    cube_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.75, 0.95),
            "dynamic_friction_range": (0.75, 0.95),
            "restitution_range": (0.0, 0.1),
            "make_consistent": True,
            "num_buckets": 32,
        },
    )

    cube_mass_variation = EventTerm(
        func=push_mdp.randomize_rigid_body_mass_simple,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (0.5, 1.50), # min. and max. percent of original
            "operation": "scale",
        },
    )
    
    cube_size_variation = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "scale_range": (0.6, 1.8),
        },
    )

    cube_goal_env_colors = EventTerm(
        func=push_mdp.set_cube_and_goal_matching_env_colors,
        mode="startup",
        params={
            "palette_size": 12,
            "brightness_range": (0.3, 1.2),
            "saturation_range": (0.5, 2.0),
            "random_seed": int(os.getenv("GO2_PUSH_COLOR_SEED", "42")),
        },
    )

    floor_friction_per_reset = EventTerm(
        func=push_mdp.randomize_floor_friction_per_reset,
        mode="reset",
        params={
            "static_friction_range": (0.65, 0.95),
            "dynamic_friction_range": (0.55, 0.85),
            "restitution_range": (0.02, 0.08),
            "terrain_material_prim_path": "/World/ground/terrain/physicsMaterial",
        },
    )

    reset_robot_and_cube = EventTerm(
        func=push_mdp.reset_robot_and_cube_uniform_around_goal,
        mode="reset",
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "cube_spawn_radius_range": (0.8, 2.5), # min should be > goal radius to avoid spawns inside the goal
            "cube_height": 0.12, # cube spawn height 
            "cube_yaw_range": (-3.14, 3.14), 
            "robot_spawn_radius_range": (0.3, 2.5), # robot is spawned in this radius around the CUBE. Min should be big enough to avoid initial robot-cube peneration. 
            "robot_yaw_range": (-3.14, 3.14),
            "robot_velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class CommandsCfg:
    """Low-level velocity command consumed by the pretrained locomotion policy."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=False, # Show velocity arrow over the robot
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-CMD_INIT_LIN_VEL_ABS*SCALE_BACK_VEL, CMD_INIT_LIN_VEL_ABS), # forward / backward
            lin_vel_y=(-CMD_INIT_LIN_VEL_ABS*SCALE_SIDE_VEL, CMD_INIT_LIN_VEL_ABS*SCALE_SIDE_VEL), # right / left
            ang_vel_z=(-CMD_INIT_ANG_VEL_ABS, CMD_INIT_ANG_VEL_ABS),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-CMD_LIMIT_LIN_VEL_X_ABS*SCALE_BACK_VEL, CMD_LIMIT_LIN_VEL_X_ABS),
            lin_vel_y=(-CMD_LIMIT_LIN_VEL_Y_ABS*SCALE_SIDE_VEL, CMD_LIMIT_LIN_VEL_Y_ABS*SCALE_SIDE_VEL),
            ang_vel_z=(-CMD_LIMIT_ANG_VEL_Z_ABS, CMD_LIMIT_ANG_VEL_Z_ABS),
        ),
    )


@configclass
class ActionsCfg:
    """High-level action term that outputs cmd_vel and runs the pretrained low-level policy."""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=LOW_LEVEL_DECIMATION,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.JointPositionAction,
        low_level_observations=LowLevelObsCfg(),
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Observation specification for push policy and critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        previous_action = ObsTerm(func=mdp.last_action)

        robot_pos_xy = ObsTerm(func=push_mdp.robot_position_xy)
        robot_lin_vel_xy = ObsTerm(func=push_mdp.robot_linear_velocity_xy)
        cube_pos_xy = ObsTerm(
            func=push_mdp.cube_position_xy,
            params={
                "noise_std": CUBE_POS_OBS_NOISE_STD,
                "dropout_prob": CUBE_POS_OBS_DROPOUT_PROB,
                "delay_steps": CUBE_POS_OBS_DELAY_STEPS,
                "spike_prob": CUBE_POS_OBS_SPIKE_PROB,
                "spike_std": CUBE_POS_OBS_SPIKE_STD,
            },
        )
        cube_lin_vel_xy = ObsTerm(
            func=push_mdp.cube_linear_velocity_xy,
            params={
                "noise_std": CUBE_VEL_OBS_NOISE_STD,
                "dropout_prob": CUBE_VEL_OBS_DROPOUT_PROB,
                "delay_steps": CUBE_VEL_OBS_DELAY_STEPS,
                "spike_prob": CUBE_VEL_OBS_SPIKE_PROB,
                "spike_std": CUBE_VEL_OBS_SPIKE_STD,
            },
        )
        goal_pos_xy = ObsTerm(func=push_mdp.goal_position_xy, params={"goal_xy": GOAL_XY})
        goal_radius = ObsTerm(func=push_mdp.goal_radius_obs, params={"goal_radius": GOAL_RADIUS_M})
        cube_to_goal_xy = ObsTerm(func=push_mdp.cube_to_goal_vector_xy, params={"goal_xy": GOAL_XY})
        lf_foot_to_cube_xy = ObsTerm(
            func=push_mdp.left_front_foot_to_cube_vector_xy,
            params={
                "foot_cfg": SceneEntityCfg("robot", body_names="FL_foot.*"),
                "cube_cfg": SceneEntityCfg("cube"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        previous_action = ObsTerm(func=mdp.last_action)

        robot_pos_xy = ObsTerm(func=push_mdp.robot_position_xy)
        robot_lin_vel_xy = ObsTerm(func=push_mdp.robot_linear_velocity_xy)
        cube_pos_xy = ObsTerm(
            func=push_mdp.cube_position_xy,
            params={
                "noise_std": CUBE_POS_OBS_NOISE_STD,
                "dropout_prob": CUBE_POS_OBS_DROPOUT_PROB,
                "delay_steps": CUBE_POS_OBS_DELAY_STEPS,
                "spike_prob": CUBE_POS_OBS_SPIKE_PROB,
                "spike_std": CUBE_POS_OBS_SPIKE_STD,
            },
        )
        cube_lin_vel_xy = ObsTerm(
            func=push_mdp.cube_linear_velocity_xy,
            params={
                "noise_std": CUBE_VEL_OBS_NOISE_STD,
                "dropout_prob": CUBE_VEL_OBS_DROPOUT_PROB,
                "delay_steps": CUBE_VEL_OBS_DELAY_STEPS,
                "spike_prob": CUBE_VEL_OBS_SPIKE_PROB,
                "spike_std": CUBE_VEL_OBS_SPIKE_STD,
            },
        )
        goal_pos_xy = ObsTerm(func=push_mdp.goal_position_xy, params={"goal_xy": GOAL_XY})
        goal_radius = ObsTerm(func=push_mdp.goal_radius_obs, params={"goal_radius": GOAL_RADIUS_M})
        cube_to_goal_xy = ObsTerm(func=push_mdp.cube_to_goal_vector_xy, params={"goal_xy": GOAL_XY})
        lf_foot_to_cube_xy = ObsTerm(
            func=push_mdp.left_front_foot_to_cube_vector_xy,
            params={
                "foot_cfg": SceneEntityCfg("robot", body_names="FL_foot.*"),
                "cube_cfg": SceneEntityCfg("cube"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Push task rewards with curriculum from approach to goal pushing."""

    #### REWARDS ####

    cube_to_goal_progress = RewTerm(
        func=push_mdp.cube_to_goal_progress_reward,
        weight=28.0,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "goal_xy": GOAL_XY,
            "transition_steps": TRANSITION_STEPS/2, # number of common steps (total steps / num_envs) over which to linearly transition the reward from dense to sparse.
        },
    )
    
    # # One time reward
    # success_bonus = RewTerm(
    #     func=push_mdp.success_bonus_reward,
    #     weight=40.0,
    #     params={
    #         "cube_cfg": SceneEntityCfg("cube"),
    #         "goal_xy": GOAL_XY,
    #         "goal_radius": GOAL_RADIUS_M,
    #         "cube_speed_threshold": 0.0,
    #         "hold_time_s": HOLD_TIME_S,
    #         "transition_steps": TRANSITION_STEPS,
    #     },
    # )
    
    # One time reward
    success_bonus_pretrigger = RewTerm(
        func=push_mdp.success_trigger_reward_robot_outsid_goal,
        weight=50.0,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot.*"),
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "cube_speed_threshold": SUCCESS_CUBE_SPEED_THRESHOLD,
            "cube_in_goal_additional_margin": SUCCESS_CUBE_IN_GOAL_ADDITIONAL_MARGIN,
            "hold_time_s": SUCCESS_HOLD_TIME_S,
            "robot_speed_threshold": SUCCESS_ROBOT_SPEED_THRESHOLD,
            "transition_steps": TRANSITION_STEPS/2,
        },
    )
    
    # cube_settled_in_goal = RewTerm(
    #     func=push_mdp.cube_settled_in_goal_reward,
    #     weight=8.0,
    #     params={
    #         "cube_cfg": SceneEntityCfg("cube"),
    #         "goal_xy": GOAL_XY,
    #         "goal_radius": GOAL_RADIUS_M,
    #         "cube_speed_threshold": 0.05,
    #         "hold_time_s": HOLD_TIME_S,
    #         "vel_std": 0.1,
    #         "transition_steps": TRANSITION_STEPS,
    #     },
    # )

    robot_stop_after_goal = RewTerm(
        func=push_mdp.robot_stop_after_goal_reward,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cube_cfg": SceneEntityCfg("cube"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "cube_in_goal_additional_margin": SUCCESS_CUBE_IN_GOAL_ADDITIONAL_MARGIN,
            "vel_std": 0.05,
            "transition_steps": TRANSITION_STEPS,
        },
    )

    # Continous reward
    # cube_in_goal = RewTerm(
    #     func=push_mdp.cube_in_goal_reward,
    #     weight=10.0,
    #     params={
    #         "cube_cfg": SceneEntityCfg("cube"),
    #         "goal_xy": GOAL_XY,
    #         "goal_radius": GOAL_RADIUS_M,
    #         "transition_steps": TRANSITION_STEPS/2,
    #     },
    # )

    # goal_hold_progress = RewTerm(
    #     func=push_mdp.goal_hold_progress_reward,
    #     weight=2.0,
    #     params={
    #         "cube_cfg": SceneEntityCfg("cube"),
    #         "goal_xy": GOAL_XY,
    #         "goal_radius": GOAL_RADIUS_M,
    #         "cube_speed_threshold": 0.05,
    #         "hold_time_s": HOLD_TIME_S,
    #         "transition_steps": TRANSITION_STEPS,
    #     },
    # )
    
    robot_to_cube_approach = RewTerm(
        func=push_mdp.robot_to_cube_approach_progress_reward,
        weight=1.5,
        params={
            "foot_cfg": SceneEntityCfg("robot", body_names="FL_foot.*"),
            "cube_cfg": SceneEntityCfg("cube"),
            "goal_xy": GOAL_XY,
            "cube_far_distance": 0.7,
            "transition_steps": TRANSITION_STEPS, # this transitions down.
        },
    )

    # just active when cube is outside the goal
    push_direction = RewTerm(
        func=push_mdp.push_direction_reward,
        weight=2.5,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "goal_margin": 0.0,
            "transition_steps": TRANSITION_STEPS,
            "speed_threshold": 0.05,
        },
    )

    # # rewards when robot forward moving, cube is moving and robot is close to the cube. This reward encourages the robot to move forward while pushing the cube, but only when it's effectively pushing (cube is moving) and in the right direction (robot close to cube).   
    # forward_push = RewTerm(
    #     func=push_mdp.forward_push_reward,
    #     weight=0.1,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "cube_cfg": SceneEntityCfg("cube"),
    #         "cube_speed_threshold": 0.03,
    #         "forward_speed_threshold": 0.05,
    #         "max_robot_cube_distance": 0.8,
    #         "transition_steps": TRANSITION_STEPS,
    #     },
    # )
    
    #### PENATLIES ####
    
    # until max_distance 0 penatly, after, linearly increasing penalty.
    cube_to_leg_distance_penalty = RewTerm( 
        func=push_mdp.cube_to_nearest_foot_distance_penalty,
        weight=-4.4,
        params={
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot.*"),
            "cube_cfg": SceneEntityCfg("cube"),
            "max_distance": 0.2,
            "transition_steps": TRANSITION_STEPS,
        },
    )
    
    goal_exit_penalty = RewTerm(
        func=push_mdp.cube_exit_goal_penalty,
        weight=-15.0,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "transition_steps": TRANSITION_STEPS,
        },
    )

    # backward_body_velocity_penalty = RewTerm(
    #     func=push_mdp.backward_body_velocity_penalty,
    #     weight=-0.001,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "deadzone": 0.04,
    #         "transition_steps": TRANSITION_STEPS,
    #     },
    # )

    robot_in_goal_area = RewTerm(
        func=push_mdp.robot_in_goal_area_penalty,
        weight=-8.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "margin": 0.35, # robot radius from base
            "transition_steps": TRANSITION_STEPS,
            "start_offset_steps": 0,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # Keep some stabilizing penalties from locomotion tasks.
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.12)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)


@configclass
class TerminationsCfg:
    """Termination conditions for push task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # success = DoneTerm(
    #     func=push_mdp.cube_goal_reached,
    #     params={
    #         "cube_cfg": SceneEntityCfg("cube"),
    #         "goal_xy": GOAL_XY,
    #         "goal_radius": GOAL_RADIUS_M,
    #         "cube_speed_threshold": 0.05,
    #         "hold_time_s": 1.0,
    #     },
    # )

    success = DoneTerm(
        func=push_mdp.cube_goal_reached_robot_outsid_goal,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot.*"),
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_xy": GOAL_XY,
            "goal_radius": GOAL_RADIUS_M,
            "cube_speed_threshold": SUCCESS_CUBE_SPEED_THRESHOLD,
            "cube_in_goal_additional_margin": SUCCESS_CUBE_IN_GOAL_ADDITIONAL_MARGIN,  # cube should be a bit more inside the goal area
            "hold_time_s": SUCCESS_HOLD_TIME_S,
            "robot_speed_threshold": SUCCESS_ROBOT_SPEED_THRESHOLD,
        },
    )
    
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for push command magnitudes (stepwise increments)."""

    # Overrides Settings set in CommandsCfg
    command_velocity_envelope = CurrTerm(
        func=push_mdp.command_velocity_envelope_stepwise_curriculum,
        params={
            "step_size": CMD_CURRICULUM_STEP_SIZE,
            "lin_vel_increment": CMD_CURRICULUM_LIN_VEL_INCREMENT,
            "ang_vel_increment": CMD_CURRICULUM_ANG_VEL_INCREMENT,
            "initial_lin_vel_abs": CMD_INIT_LIN_VEL_ABS,
            "initial_ang_vel_abs": CMD_INIT_ANG_VEL_ABS,
            "limit_lin_vel_x": CMD_LIMIT_LIN_VEL_X_ABS,
            "limit_lin_vel_y": CMD_LIMIT_LIN_VEL_Y_ABS,
            "limit_ang_vel_z": CMD_LIMIT_ANG_VEL_Z_ABS,
            "scale_back_vel": SCALE_BACK_VEL,
            "scale_side_vel": SCALE_SIDE_VEL,
        },
    )
    
    common_step_counter = CurrTerm(
        func=push_mdp.curriculum_common_step_counter,
        params={},
    )
    
    goal_reward_alpha = CurrTerm(
        func=push_mdp.curriculum_goal_reward_alpha,
        params={
            "transition_steps": TRANSITION_STEPS,
        },
    )


@configclass
class RobotPushEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 high-level push task using a pretrained low-level locomotion policy."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=6144, env_spacing=10.0) # 4096 # distance (in meters) between neighboring environment origins (goal centers) in the world layout.

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = HIGH_LEVEL_DECIMATION
        self.episode_length_s = 25.0 # timeout time in seconds, used by mdp.time_out termination condition

        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPushPlayEnvCfg(RobotPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        reset_mode = os.getenv("GO2_PUSH_PLAY_RESET_MODE", "standard").strip().lower()
        if reset_mode == "success_keep_robot":
            self.events.reset_robot_and_cube.func = push_mdp.reset_push_episode_by_termination
            self.events.reset_robot_and_cube.params["joint_position_range"] = (1.0, 1.0)
            self.events.reset_robot_and_cube.params["joint_velocity_range"] = (-1.0, 1.0)
            self.events.reset_robot_joints.func = push_mdp.no_op_reset
            self.events.reset_robot_joints.params = {}
