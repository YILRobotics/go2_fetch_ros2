# fetch

ROS 2 Humble package for Go2 push-cube deployment:

1. `cube_tracker_node`: RealSense + reduced-resolution YOLOE segmentation + aligned-depth filtering, publishes cube planar state.
2. `low_level_policy_node` (package `fetch_low_level`): C++ startup state machine, low-level TensorRT inference, and `/lowcmd` output.
3. `high_level_policy_node`: Python high-level TensorRT policy, goal/cube recovery, Sport mode, and visualization.
4. `state_machine_node`: Controls task-level modes.

## Architecture

- `cube_tracker_node`
  - Subscribes:
    - `/camera/color/image_raw` (`sensor_msgs/Image`)
    - `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`)
    - `/camera/color/camera_info` (`sensor_msgs/CameraInfo`)
  - Resizes the wide camera frame to 640x360 for YOLOE, maps the mask back to the camera frame, and deprojects only sampled aligned-depth pixels before MAD filtering.
  - Publishes:
    - `/go2_fetch/cube_state` (`nav_msgs/Odometry`):
      - `pose.pose.position.x/y`: cube XY on floor frame
      - `twist.twist.linear.x/y`: cube XY velocity
    - `/go2_fetch/cube_visible` (`std_msgs/Bool`)
    - `/go2_fetch/cube_debug_image` (`sensor_msgs/Image`, optional)
  - Processing timer: 15 Hz (configurable)

- `low_level_policy_node`
  - Subscribes to `/lowstate`, `/go2_fetch/high_level_cmd`, and `/go2_fetch/high_level_cmd_enabled`.
  - Owns START/A/SELECT, low-level inference, torque limiting, CRC, `/lowcmd`, `/inekf_lowstate`, control state, and timing.

- `high_level_policy_node`
  - Subscribes:
    - `/odometry/filtered` (`nav_msgs/Odometry`) for Kalman linear velocity
    - `/lowstate` (`unitree_go/msg/LowState`)
    - `/sportmodestate` (`unitree_go/msg/SportModeState`)
  - Loads the high-level TensorRT policy and handles X/Y/B command-source controls.
  - Publishes:
    - `/go2_fetch/high_level_cmd` (`geometry_msgs/msg/TwistStamped`)
    - `/go2_fetch/high_level_cmd_enabled` (`std_msgs/msg/Bool`)

- `state_machine_node`
  - Subscribes:
    - `/go2_fetch/cube_visible` (`std_msgs/Bool`)
  - Publishes:
    - `/go2_fetch/mode` (`std_msgs/String`) with values: `standup`, `policy`, `search`
    - `/go2_fetch/state` (`std_msgs/String`)
    - `/go2_fetch/cube_tracking_lost` (`std_msgs/Bool`)
  - Behavior:
    - Start in `standup`
    - Move to `policy` when cube is visible
    - Move to `search` when cube is lost
    - Return to `policy` after stable reacquisition

## Configuration

All main parameters are in:

- `config/fetch_params.yaml`

Important fields:

- model paths in their respective node sections
- control mode (`control_mode`):
  - `hierarchical_lowcmd`: current full stack, publishes `/lowcmd`
  - `unitree_sport_high_level`: only runs the high-level policy and sends Unitree Sport `Move` requests on `/api/sport/request`
- Unitree ROS 2 topics (`lowstate_topic`, `lowcmd_topic`, `sport_request_topic`)
- safe policy test mode (`fake_observations_mode`, `send_commands`)
- runtime velocity-command source (`use_high_level_policy`)
- low-level startup and safety controls in the `low_level_policy_node` section
- tracker thresholds (confidence, depth range, outlier filtering)
- FSM timeouts (`cube_lost_timeout_s`, `cube_reacquire_hold_s`)

## Policy Model Files

`high_level_policy_node.py` loads the high-level TensorRT `.engine`. The C++
`low_level_policy_node` independently loads the low-level TensorRT `.engine`.

## PushCube-4L Policy Interface

The two policy nodes implement the hierarchical `Unitree-Go2-PushCube-4L`
stack from:

```text
/home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl
```

The PushCube-4L task uses a hierarchical policy stack:

```text
high-level push policy: 48 observations -> 3 velocity commands
low-level 4L velocity policy: 45 observations -> 12 joint actions
```

The high-level output is not sent directly to the motors. It becomes the velocity command input for the low-level locomotion policy.

### High-Level Push Policy

The high-level PushCube policy input is 48 values:

```text
0:3    base angular velocity * 0.2
3:6    projected gravity
6:18   joint positions relative to default
18:30  joint velocities * 0.05
30:33  previous high-level action
33:35  robot XY position
35:37  robot XY linear velocity
37:39  cube XY position
39:41  cube XY velocity
41:43  goal XY position
43:44  goal radius
44:46  cube-to-goal XY vector
46:48  left-front-foot-to-cube XY vector
```

The high-level PushCube policy output is 3 values:

```text
base_lin_vel_x
base_lin_vel_y
base_ang_vel_z
```

These 3 values are the command for the low-level locomotion policy.

### Low-Level 4L Velocity Policy

The low-level velocity policy input is 45 values:

```text
0:3    base angular velocity * 0.2
3:6    projected gravity
6:9    velocity command from high-level policy
9:21   joint positions relative to default
21:33  joint velocities * 0.05
33:45  previous low-level action
```

The low-level output is 12 joint actions:

```text
target_joint = action * 0.25 + default_joint_position
```

### IsaacLab Generic Velocity Policy Reference

TheoBounac's IsaacLab `velocity_env_cfg.py` is here:

```text
https://github.com/TheoBounac/IsaacLab/blob/6fee5b7f970009b83c11bcb7264afdca97106dfe/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
```

That generic IsaacLab velocity locomotion environment uses these policy observation terms, in order:

```text
base_lin_vel
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
height_scan
```

For a 12-joint Go2-style robot, the non-height-scan part is:

```text
base_lin_vel        3
base_ang_vel        3
projected_gravity   3
velocity_commands   3
joint_pos_rel      12
joint_vel_rel      12
last_action        12
```

The height scan comes from a yaw-aligned ray grid:

```text
resolution = 0.1
size = [1.6, 1.0]
```

That usually gives about 187 height values, so the full observation for a 12-joint robot is roughly:

```text
3 + 3 + 3 + 3 + 12 + 12 + 12 + 187 = 235 values
```

The generic IsaacLab velocity action is also different from the PushCube ROS node:

```text
target_joint = action * 0.5 + default_joint_position
```

The custom `Unitree-Go2-Velocity-4L` policy used by the PushCube stack is not this generic height-scan policy. In this repository's 4L config, the low-level policy removes `base_lin_vel` and `height_scan`, uses `base_ang_vel * 0.2`, and uses:

```text
45 observations -> 12 joint actions
target_joint = action * 0.25 + default_joint_position
```

### Current Node Interface

The node now loads both TorchScript policies, builds the 48-value high-level
observation at 15 Hz, and passes its 3-value output into the 45-value low-level
observation at 50 Hz. The 12 low-level actions become joint position targets
using scale `0.25`, the exported default offsets, `Kp=25`, and `Kd=0.5`.

Robot and cube positions, cube-to-goal, and foot-to-cube must use the same world
frame. The default tracker target frame is therefore `odom`; `goal_xy` must also
be configured in that frame.

#### Cube And Foot Observations

- Fake mode uses random observations and does not require cube or foot TF data.
- Real mode reads the left-front-foot position from the `odom -> FL_foot` TF.
- Real mode stops deployment if the required transform is unavailable.
- `policy_world_frame`, `lf_foot_frame`, and `lf_foot_tf_timeout_s` configure the lookup.
- No approximate left-front-foot offset is used.

The real robot deployment needs these signals:

```text
IMU gyroscope
projected gravity from IMU orientation
joint positions
joint velocities
previous high-level action
previous low-level action
robot XY position
robot XY velocity
cube XY position
cube XY velocity
goal XY position
goal radius
left-front-foot-to-cube XY vector
```

Foot force is not part of the current PushCube-4L trained policy interface. Adding foot force is only useful if the policy is retrained with foot force in Isaac Sim. For sim-to-real transfer, matching the trained observation layout exactly is more important than adding extra sensors.

## Build

```bash
cd /home/ferdinand/unitree/go2_fetch_ros2
colcon build --packages-select fetch
source install/setup.bash
```

## Run

```bash
ros2 launch fetch fetch_bringup.launch.py
```

RealSense + cube tracker + RViz:

```bash
ros2 launch fetch fetch_realsense_cube_tracker.launch.py
```

If your RealSense topics come up under `/camera/camera/...`, override tracker topics:

```bash
ros2 launch fetch fetch_realsense_cube_tracker.launch.py \
  image_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/camera/color/camera_info
```

Custom parameter file:

```bash
ros2 launch fetch fetch_bringup.launch.py \
  params_file:=/absolute/path/to/your_params.yaml
```

Run random fake observations without sending robot commands:

```bash
ros2 launch fetch policy_test.launch.py \
  fake_observations_mode:=true
```

### Low-Level Mode Switch

`policy_node` uses ROS 2 `unitree_go` topics for command and state I/O. Do not
initialize `unitree_sdk2py` DDS inside the same Python process as `rclpy`; the
two CycloneDDS users can conflict during topic creation.

Because of that, `policy_node` does not call `MotionSwitcherClient` directly.
If the robot is still in a high-level Unitree mode, release that mode in a
separate terminal before launching the policy:

```bash
conda activate env_deploy
cd ~/fetch_ws

python src/go2_fetch_ros2/fetch/fetch/switch_to_low_level.py --interface enP8p1s0
```

Then launch the policy in another command:

```bash
ros2 launch fetch policy_odom.launch.py
```

In simple terms: the script above puts the robot down, releases the built-in
Unitree high-level controller, then exits. The ROS policy node can then publish
low-level motor commands on `/lowcmd`.

### Restore High-Level Mode

After stopping `policy_node`, switch the robot back to a Unitree high-level mode
from a separate terminal:

```bash
conda activate env_deploy
cd ~/fetch_ws

python src/go2_fetch_ros2/fetch/fetch/restore_high_level.py --interface enP8p1s0
```

Use this after `Ctrl+C` stops the policy launch. For Go2, the default mode is
`ai`. To request a different mode:

```bash
python src/go2_fetch_ros2/fetch/fetch/restore_high_level.py --interface enP8p1s0 --mode normal
```

If the robot is lying down and should stand after switching back, add
`--stand-up`. The script only calls `RecoveryStand()` if the mode switch
succeeds:

```bash
python src/go2_fetch_ros2/fetch/fetch/restore_high_level.py --interface enP8p1s0 --stand-up
```

If `SelectMode` fails but you still want to try `RecoveryStand()`, add
`--force-stand-up`:

```bash
python src/go2_fetch_ros2/fetch/fetch/restore_high_level.py --interface enP8p1s0 --stand-up --force-stand-up
```

After the next package build, the same helpers are also available as ROS package
commands:

```bash
ros2 run fetch switch_to_low_level --interface enP8p1s0
ros2 run fetch restore_high_level --interface enP8p1s0 --stand-up
```

## Runtime Commands

Publish a Unitree Sport API request at 10 Hz:

```bash
ros2 topic pub /api/sport/request unitree_api/msg/Request \
  "{header: {identity: {api_id: 1008}}, parameter: '{\"x\": 0.0, \"y\": 0.0, \"z\": 0.5}'}" \
  -r 10
```

The low-level policy command source is controlled by
`use_high_level_policy`:

```text
true   48-value PushCube policy generates [vx, vy, wz]
false  Unitree joystick generates [vx, vy, wz]
```

Read the current mode:

```bash
ros2 param get /policy_node use_high_level_policy
```

Disable the high-level policy and use the joystick immediately:

```bash
ros2 param set /policy_node use_high_level_policy false
```

Re-enable the high-level PushCube policy immediately:

```bash
ros2 param set /policy_node use_high_level_policy true
```

This parameter can be changed while the node is running. It is separate from
`fake_observations_mode`, which switches the entire node to random observations,
skips DDS and TF setup, and disables robot command output.

## Notes

- `ultralytics`, `torch`, `opencv`, and `cv_bridge` are required for tracker/policy inference.
- The Unitree SDK Python package is required only for separate mode-switching tools, not inside `policy_node`.
- `unitree_go` ROS messages are required for `/lowcmd`, `/lf/lowstate`, `/sportmodestate`, and `/inekf_lowstate`.
