# fetch

ROS 2 Humble package with three nodes for Go2 push-cube deployment:

1. `cube_tracker_node`: Realsense + YOLOE segmentation + pointcloud filtering, publishes cube planar state.
2. `policy_node`: ROS 2 port of `Deploy_SimToReal_RL_Go2/deploy_real`, including Unitree DDS control and Kalman odometry input.
3. `state_machine_node`: Controls modes (`standup -> policy -> search`) and handles cube-loss recovery.

## Architecture

- `cube_tracker_node`
  - Subscribes:
    - `/camera/color/image_raw` (`sensor_msgs/Image`)
    - `/camera/depth/color/points` (`sensor_msgs/PointCloud2`)
  - Runs YOLOE segmentation (class-filtered), samples mask pixels in pointcloud, rejects outliers with MAD filtering.
  - Publishes:
    - `/go2_fetch/cube_state` (`nav_msgs/Odometry`):
      - `pose.pose.position.x/y`: cube XY on floor frame
      - `twist.twist.linear.x/y`: cube XY velocity
    - `/go2_fetch/cube_visible` (`std_msgs/Bool`)
    - `/go2_fetch/cube_debug_image` (`sensor_msgs/Image`, optional)
  - Processing timer: 20 Hz (configurable)

- `policy_node`
  - Subscribes:
    - `/odometry/filtered` (`nav_msgs/Odometry`) for Kalman linear velocity
    - Unitree DDS `rt/lowstate` and `rt/sportmodestate`
  - Loads the original `policy_rough.pt` TorchScript policy.
  - Preserves the remote sequence: START to stand, A to run, SELECT to lower the robot.
  - Publishes:
    - Unitree DDS `rt/lowcmd`
    - `/inekf_lowstate` (`unitree_go/msg/LowState`) for the estimator

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

- model path (`policy_base_dir`, `policy_path`)
- Unitree DDS setup (`network_interface`, `dds_domain_id`)
- Unitree topics (`lowstate_topic`, `lowcmd_topic`, `sportstate_topic`)
- safe policy test mode (`fake_observations_mode`, `send_commands`)
- startup controls (`wait_for_start_button`, `wait_for_a_button`)
- tracker thresholds (confidence, depth range, outlier filtering)
- FSM timeouts (`cube_lost_timeout_s`, `cube_reacquire_hold_s`)

## PushCube-4L Policy Interface

The current `policy_node.py` is a ROS 2 port of `Deploy_SimToReal_RL_Go2/deploy_real`.
It matches the older direct deployment policy style:

```text
52 observations -> 12 joint actions
```

That is different from the `Unitree-Go2-PushCube-4L` task in:

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

The generic IsaacLab velocity action is also different from the current ROS node:

```text
target_joint = action * 0.5 + default_joint_position
```

The custom `Unitree-Go2-Velocity-4L` policy used by the PushCube stack is not this generic height-scan policy. In this repository's 4L config, the low-level policy removes `base_lin_vel` and `height_scan`, uses `base_ang_vel * 0.2`, and uses:

```text
45 observations -> 12 joint actions
target_joint = action * 0.25 + default_joint_position
```

### Current Node Interface

The current `policy_node.py` builds a 52-value observation:

```text
0:4    foot forces
4:7    base linear velocity
7:10   IMU angular velocity
10:13  projected gravity
13:16  joystick command
16:28  joint positions relative to default
28:40  joint velocities
40:52  previous action
```

It then sends 12 joint actions directly to the robot.

This means the current node is not compatible with the PushCube-4L high-level or low-level policies.

### What Needs To Change For PushCube-4L

To run the PushCube-4L policy stack on the real robot, `policy_node.py` should be changed to:

1. Load two TorchScript policies:
   - high-level PushCube policy
   - low-level 4L velocity policy
2. Build the 48-value high-level observation.
3. Run the high-level policy at about 15 Hz.
4. Use the 3-value high-level output as the low-level velocity command.
5. Build the 45-value low-level observation.
6. Run the low-level policy at about 50 Hz.
7. Convert the 12 low-level actions into joint position targets.
8. Send those joint targets to the robot.

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
  pointcloud_topic:=/camera/camera/depth/color/points
```

Custom parameter file:

```bash
ros2 launch fetch fetch_bringup.launch.py \
  params_file:=/absolute/path/to/your_params.yaml
```

Run random fake observations without connecting to DDS or sending robot commands:

```bash
ros2 launch fetch policy_test.launch.py \
  fake_observations_mode:=true
```

## Notes

- `ultralytics`, `torch`, `opencv`, and `cv_bridge` are required for tracker/policy inference.
- The Unitree SDK Python package is required for DDS command/state communication.
- `unitree_go` ROS messages are required only for publishing `/inekf_lowstate`.
