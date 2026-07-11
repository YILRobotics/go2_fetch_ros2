# Odometry C++ InEKF Node Changes

This note documents the final odometry change: the InEKF ROS wrapper was moved
from Python to C++ while preserving the original estimator behavior.

## Files Changed

- `go2_odometry/src/inekf_odom_node.cpp`
- `go2_odometry/CMakeLists.txt`
- `go2_odometry/package.xml`
- `go2_odometry/launch/go2_inekf_odometry.launch.py`
- `fetch/launch/odometry_inekf.launch.py`
- `go2_odometry/scripts/inekf_odom.py`

## Behavior Changes

- Launch files now run the C++ executable `inekf_odom`.
- The original Python wrapper is still installed as `inekf_odom_py` for
  fallback/debug use.
- The C++ node calls the native `inekf` C++ API directly, avoiding
  Boost.Python conversion overhead in the high-rate odometry loop.
- EKF propagation uses the original fixed timestep:

```yaml
dt: 1.0 / robot_freq
robot_freq: 500.0
```

- The measured callback timing experiment was removed because `/lowstate` is
  published steadily near 500 Hz, while Python callback scheduling sometimes
  produced large wall-clock gaps and skipped EKF propagation.

## Preserved Estimator Behavior

- Subscribes to `/lowstate`.
- Publishes `/go2_odometry/filtered`.
- Broadcasts `odom -> base`.
- Waits for all feet to contact the ground before starting when
  `wait_for_all_feet_contact` is enabled.
- Uses the same hard foot contact threshold: reordered foot force `>= 18`.
- Uses the same IMU initialization, zero initial yaw, foot-height base z
  initialization, foot-pose correction, and zero contact velocity input.

## Rebuild Notes

Rebuild the ROS package so the new C++ executable is installed:

```bash
cd ~/fetch_ws
colcon build --packages-select go2_odometry
source install/setup.bash
```

If `inekf` is not already installed in the active environment, rebuild/install
that dependency first.

## Patch File

The corresponding patch is:

```text
patches/go2-odometry-z-stability.patch
```
