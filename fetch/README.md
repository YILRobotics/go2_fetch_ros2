# fetch

ROS 2 Humble package with three nodes for Go2 push-cube deployment:

1. `cube_tracker_node`: Realsense + YOLOE segmentation + pointcloud filtering, publishes cube planar state.
2. `policy_node`: Runs high-level push policy and low-level locomotion policy, publishes low-level motor commands.
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
    - `/go2_fetch/mode` (`std_msgs/String`)
    - `/go2_fetch/cube_state` (`nav_msgs/Odometry`)
    - `/lio_sam_ros2/mapping/odometry` (`nav_msgs/Odometry`)
    - `/lowstate` (`unitree_go/msg/LowState`) when available
  - Loads two TorchScript policies:
    - High-level push policy (`model_*.pt` paths auto-resolved to `exported/policy.pt` when present)
    - Low-level walk policy (`.../exported/policy.pt`)
  - High-level observation is aligned to your push config terms (`base_ang_vel`, `projected_gravity`, joints, robot/cube/goal terms).
  - Publishes:
    - `/lowcmd` (`unitree_go/msg/LowCmd`) when Unitree ROS messages are available
    - `go2_fetch/joint_targets` (`std_msgs/Float32MultiArray`) debug target joints
    - `go2_fetch/policy_cmd` (`geometry_msgs/TwistStamped`) debug velocity command

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

- model paths (`high_level_policy_path`, `low_level_policy_path`)
- Unitree topics (`lowstate_topic`, `lowcmd_topic`)
- tracker thresholds (confidence, depth range, outlier filtering)
- FSM timeouts (`cube_lost_timeout_s`, `cube_reacquire_hold_s`)

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

## Notes

- `ultralytics`, `torch`, `opencv`, and `cv_bridge` are required for tracker/policy inference.
- For real Go2 low-level command output, `unitree_go` ROS message package must be installed and sourced.
- If `unitree_go` messages are not present, the policy node still publishes debug joint targets and velocity command topics.
