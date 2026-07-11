# Real-Inference Changes Since `2026-07-02_22-18-29_ff_5_2`

## Scope

`2026-07-02_22-18-29_ff_5_2` is a PushCube training-run directory, not a Git commit. This summary therefore uses commit `4fa919f` (2026-06-30), the last repository commit before the run, as the code baseline. It covers commits `ae67137` (2026-07-05) and `499b51c` (2026-07-07).

The changes improve the high-level PushCube policy and its readiness for real inference. No files under `deploy/` or `isaacsim_extensions/` changed in this period, so the real-robot inference runtime itself was not updated.

## Changes Relevant to Real Inference

### Safer high-level commands

- High-level velocity commands are now clamped before being sent to the low-level locomotion policy.
- Final command limits are `x = 0.6 m/s`, `y = 0.4 m/s`, and `yaw = 0.8 rad/s`.
- Training starts with limits of `0.1`, `0.1`, and `0.1`, then increases them over the first half of the transition curriculum.
- Play/inference disables this command-limit curriculum and immediately uses the final limits.
- The policy's previous-action observation now uses the bounded command actually sent to the low-level policy, rather than the unbounded network output.

### Observation corrections and frame consistency

- Cube velocity now comes directly from the simulator's synchronized rigid-body velocity instead of being estimated from position differences at the high-level control rate.
- Cube velocity, cube-to-goal direction, and front-foot-to-cube direction are transformed from world coordinates into the robot base frame.
- Observation scales and explicit clipping were added to policy and critic terms.
- Position/velocity noise, dropout, spikes, and delay settings were retuned.
- Observation corruption is introduced gradually during training. Play/inference uses the final observation configuration immediately, while random corruption remains disabled in the play environment.

These changes make the high-level input/output contract more suitable for deployment, but the real perception pipeline must reproduce the same base-frame transforms, scaling, ordering, clipping, and timing.

### Domain randomization changes

- Ground friction randomization changed to:
  - static friction: `0.35-0.75`
  - dynamic friction: `0.35-0.60`
- Robot mass randomization changed from `0.50-1.50x` to `0.25-1.25x` of the original mass.

These settings change the sim-to-real robustness distribution and should be considered when comparing newer policies with `ff_5_2`.

### Policy behavior and reward shaping

- Spawn positions were adjusted to less extreme near/far corner configurations.
- Reward transition length increased from `8,000` to `10,000` common steps.
- Added robot-cube-goal alignment guidance.
- Added penalties for sideways body motion and incorrect heading toward the cube.
- Approach progress now uses the nearest front foot instead of only the front-left foot.
- Timeout penalty now scales with remaining cube-to-goal distance.
- Failure termination, timeout, and per-step time penalties are handled separately.
- Several existing reward weights and safety distances were retuned.
- Directional push reward now preserves negative reward when the cube moves away from the goal.

These are training-policy changes; policies trained before them, including `ff_5_2`, do not gain the new behavior unless retrained.

### Recording and diagnosis

- Push play now records environment 0 to CSV, including:
  - foot normal forces
  - bounded high-level velocity commands
  - cube position and velocity observations
  - all cached policy and critic observation terms
- Recordings are stored under the run's `recordings/play/` directory.
- `scripts/plot_recording.py` replaces the old foot-force-only plotter and separates 50 Hz low-level data from 15.38 Hz high-level data.
- Recorded videos receive task-specific, timestamped filenames.

This provides a better way to compare simulation signals with signals observed during real inference and to detect frame, scale, timing, or command-saturation mismatches.

## Compatibility Notes

- `ff_5_2/model_2499.pt` was trained with the older observation/action semantics. It should be evaluated with the matching configuration rather than assumed compatible with the new observation layout.
- A policy trained after these changes expects the revised observation preprocessing and bounded command behavior.
- The deployment side still needs explicit verification against the new high-level contract; no deployment implementation was changed during this period.

## Exact Real-Inference Contract for the New Policy

This section describes the current PushCube policy contract represented by `push_env_cfg.py` and the post-`ff_5_2` deployment configuration, for example `2026-07-07_10-07-42_ff_5_19/params/deploy.yaml`. Always ship the `deploy.yaml` exported with the exact model; do not combine an old YAML with a new ONNX model.

### Frame assignment: not all XY values are in the base frame

The observation vector intentionally mixes world-frame and base-frame values.

| Observation | Frame expected by new model | Scaling | Required computation |
|---|---|---:|---|
| `robot_pos_xy` | World/local-map frame | `0.5` | Robot world XY minus the chosen environment/map origin |
| `robot_lin_vel_xy` | World frame | `1.0` | Robot linear velocity in world XY |
| `cube_pos_xy` | World/local-map frame | `0.5` | Cube world XY minus the same environment/map origin |
| `cube_lin_vel_xy` | **Robot base frame** | `1.0` | Rotate cube world velocity into the current robot base frame |
| `goal_pos_xy` | World/local-map frame | `1.0` | Goal XY relative to the same map origin; training uses `(0, 0)` |
| `cube_to_goal_xy` | **Robot base frame** | `0.5` | Form `goal_world - cube_world`, then rotate the vector into base frame |
| `lf_foot_to_cube_xy` | **Robot base frame** | `1.0` | Form `cube_world - front_left_foot_world`, then rotate into base frame |

Therefore, `cube_pos_xy` itself is **still world-frame**. The terms changed to base frame are exactly:

1. cube linear velocity;
2. cube-to-goal vector;
3. front-left-foot-to-cube vector.

`robot_pos_xy`, `robot_lin_vel_xy`, `cube_pos_xy`, and `goal_pos_xy` remain world-frame. Do not rotate those four values into the base frame for this model.

### Exact world-to-base transform

Isaac Lab stores `root_quat_w` as a quaternion that rotates a base-frame vector into the world frame, in `(w, x, y, z)` order. The policy code performs the inverse full-quaternion rotation:

```text
v_base_3d = inverse(q_world_from_base) * [v_world_x, v_world_y, 0]
observation_xy = [v_base_3d.x, v_base_3d.y]
```

Use the robot orientation at the same sample time as the perception data. For exact simulation parity, use the full orientation, including roll and pitch, not only yaw. In the ROS runtime, cube, goal, robot, and front-left-foot positions are expressed in the policy world frame (`odom`), so the base-frame terms must use the frame-consistent odometry quaternion (`odom_from_base`, stored as `robot_quaternion_world_from_base`) rather than the raw Unitree IMU quaternion. Confirm the quaternion component order in the real-robot library before applying the transform.

All world quantities must use one consistent right-handed map frame. Define a fixed map origin and express robot, cube, front-left foot, and goal in it. To reproduce training exactly, place the goal at map `(0, 0)`. If the real localization system uses another origin, subtract the goal/map-origin offset consistently from `robot_pos_xy`, `cube_pos_xy`, and `goal_pos_xy`; relative vectors are translation invariant.

### Exact 52-value policy observation order

Construct one flat vector in this order. The index ranges are zero-based and inclusive.

| Indices | Count | Term | Input before scaling |
|---:|---:|---|---|
| `0-2` | 3 | `base_ang_vel` | IMU angular velocity in base frame |
| `3-5` | 3 | `projected_gravity` | World gravity direction projected into base frame |
| `6-17` | 12 | `joint_pos_rel` | Joint position minus default joint position, in training joint order |
| `18-29` | 12 | `joint_vel_rel` | Joint velocity in training joint order |
| `30-32` | 3 | `previous_action` | Previous **clamped high-level** command `[vx, vy, yaw_rate]` |
| `33-34` | 2 | `robot_pos_xy` | World/map-frame robot XY |
| `35-36` | 2 | `robot_lin_vel_xy` | World-frame robot XY velocity |
| `37-38` | 2 | `cube_pos_xy` | World/map-frame cube XY |
| `39-40` | 2 | `cube_lin_vel_xy` | Base-frame cube XY velocity |
| `41-42` | 2 | `goal_pos_xy` | World/map-frame goal XY, normally `[0, 0]` |
| `43` | 1 | `goal_radius` | `0.2` m |
| `44-45` | 2 | `cube_to_goal_xy` | Base-frame cube-to-goal vector |
| `46-47` | 2 | `lf_foot_to_cube_xy` | Base-frame front-left-foot-to-cube vector |
| `48-51` | 4 | `foot_force` | Per-foot compressive force in the exact training foot order |

Apply clipping first and scaling second, matching Isaac Lab and the repository's C++ `ObservationManager`:

- `base_ang_vel *= 0.2`;
- `joint_vel_rel *= 0.05`;
- `robot_pos_xy *= 0.5`;
- `cube_pos_xy *= 0.5`;
- `cube_to_goal_xy *= 0.5`;
- `foot_force = clamp(force, 0, 150) * 0.01`;
- all other listed terms use scale `1.0`;
- all non-foot-force terms have the broad clip `[-100, 100]` before scaling.

The deployed low-level locomotion policy `2026-06-30_11-11-40_walk_ff_5` uses the same `0.01` foot-force scale in its exported deploy YAML. In ROS config, that means both high-level and low-level `foot_force_scale` values should be `100.0`.

Do not reorder joints or feet based on names at runtime unless the resulting order exactly matches the training vector. The exported joint SDK mapping is `[3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]`; the real code must apply the model's accompanying mapping exactly. Verify the four hardware foot-force indices against the simulation body order rather than assuming that Unitree SDK order and Isaac body order are identical.

### Cube position, velocity, delay, and dropout state

The new training code no longer estimates cube velocity with `(position_now - position_previous) / policy_dt`. It reads synchronized simulator rigid-body velocity and then rotates it into the base frame. Real inference should provide a timestamped world-frame cube velocity from the tracker, preferably from a filtered estimator, and rotate that velocity into base frame. If velocity must be differentiated from positions, differentiate using the real timestamps, filter it, and then rotate it; do not hard-code a nominal derivative interval when frames can arrive late.

Training applies a one-high-level-step delay (`65 ms`) independently to cube position and cube velocity before adding synthetic noise/dropout/spikes. The same delayed cube-position sample is reused to construct `cube_pos_xy`, `cube_to_goal_xy`, and `lf_foot_to_cube_xy` during a policy step.

For real inference:

- use one timestamp-consistent cube state for all four cube-related terms;
- account for perception latency so the effective latency is close to the trained `65 ms` delay;
- do not blindly add another `65 ms` buffer if the camera/tracker already has approximately that latency;
- normally do not add artificial noise, spikes, or dropout to real measurements because the sensor already supplies those errors;
- on a missing cube detection, hold the previous accepted cube observation, matching training dropout behavior;
- on reset/startup, initialize history from the first valid measurement rather than zeros.

The final training corruption distribution was position noise `0.08 m`, position dropout `0.04`, position spike probability/std `0.04/0.1 m`, velocity noise `0.15 m/s`, velocity dropout `0.04`, and velocity spike probability/std `0.05/0.4 m/s`. These values describe robustness training, not mandatory extra noise to inject on the robot.

### High-level output and low-level policy execution

Run the high-level ONNX model every `0.065 s` (approximately `15.38 Hz`, resulting from 13 simulation steps at `0.005 s`). Its three outputs are body-frame velocity commands in this order:

```text
[linear_x, linear_y, angular_z]
```

Clamp the raw model output component-wise before storing it or passing it to locomotion:

```text
linear_x  = clamp(output[0], -0.6, 0.6) m/s
linear_y  = clamp(output[1], -0.4, 0.4) m/s
angular_z = clamp(output[2], -0.8, 0.8) rad/s
```

Use these clamped values for both:

1. the command sent to the low-level velocity policy; and
2. `previous_action` at indices `30-32` on the next high-level step.

Initialize `previous_action` to `[0, 0, 0]` on startup/reset. Hold the latest clamped command between high-level updates while the low-level locomotion policy runs at `50 Hz`. The low-level model and its observation preprocessing must remain the exact 4-leg velocity policy referenced by the training run.

### What is missing in the current C++ real-inference runtime

The current `deploy/` runtime supports locomotion observations such as IMU, joints, joystick commands, last action, and foot force. It does **not** currently implement the PushCube high-level runtime. Loading the new PushCube `policy.onnx` and `deploy.yaml` directly will not work because:

- `ArticulationData` has no world robot position/velocity, cube state, goal state, or front-left-foot pose;
- no real perception/localization interface supplies cube and goal measurements;
- the C++ observation registry has no implementations for `previous_action`, `robot_pos_xy`, `robot_lin_vel_xy`, `cube_pos_xy`, `cube_lin_vel_xy`, `goal_pos_xy`, `goal_radius`, `cube_to_goal_xy`, or `lf_foot_to_cube_xy` as required by this YAML (`last_action` exists, but the YAML term is named `previous_action`);
- the C++ action registry has no `pre_trained_policy_action` implementation that clamps high-level outputs and runs/commands the nested low-level policy;
- there is no high-level/low-level dual-rate scheduler;
- there is no cube observation delay/hold-last state;
- the exported Python action configuration refers to a TorchScript low-level policy path, while the C++ runtime expects ONNX execution and needs an explicit second model/session.

To deploy the new model, the real code must add all of those components, then validate the generated 52-value observation against a simulation CSV from `scripts/rsl_rl/play.py` for the same static state and trajectory. A safe acceptance test is component-by-component agreement for frame, sign, unit, scale, ordering, timing, previous-action state, and output clamping before enabling motion.

## Files Changed

- `source/unitree_rl_lab/unitree_rl_lab/tasks/mdp/actions.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/mdp/observations.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/push_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/push_mdp.py`
- `scripts/rsl_rl/play.py`
- `scripts/rsl_rl/train.py`
- `scripts/plot_recording.py`
- `README.md`
