# Unitree Mode Switching

`policy_node` uses ROS 2 `unitree_go` topics for command and state I/O. Do not
initialize `unitree_sdk2py` DDS inside the same Python process as `rclpy`; the
two CycloneDDS users can conflict during topic creation.

Use these helper scripts from a separate terminal before and after running the
policy.

## Switch To Low-Level Mode

Run this before launching `policy_node`:

```bash
conda activate env_deploy
cd ~/fetch_ws

python src/go2_fetch_ros2/fetch/fetch/switch_to_low_level.py --interface enP8p1s0
```

This puts the robot down, releases the built-in Unitree high-level controller,
and exits. After that, the ROS policy node can publish low-level motor commands
on `/lowcmd`.

Then launch the policy:

```bash
ros2 launch fetch policy_odom.launch.py
```

## Restore High-Level Mode

After stopping the policy launch with `Ctrl+C`, run:

```bash
conda activate env_deploy
cd ~/fetch_ws

python src/go2_fetch_ros2/fetch/fetch/restore_high_level.py --interface enP8p1s0
```

The default restore mode is `ai`, which is the mode used in the local Unitree
SDK motion-switcher example.

To request a different mode:

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

Common return codes seen locally:

```text
7004  robot rejected the requested high-level mode
3102  client send failure; the Sport API request did not reach a matching service
```

## After Building The Package

After the next package build, the same helpers are available as ROS package
commands:

```bash
ros2 run fetch switch_to_low_level --interface enP8p1s0
ros2 run fetch restore_high_level --interface enP8p1s0 --stand-up
```

## Options

Both scripts accept:

```text
--interface enP8p1s0
--domain-id 0
--timeout 5.0
```

`restore_high_level.py` also accepts:

```text
--mode ai
--stand-up
--stand-up-delay 1.0
```
