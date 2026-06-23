# ROS 2 topics over Wi-Fi

This setup forwards selected ROS 2 topics from the Jetson Docker container to
the workstation over Wi-Fi. It uses `zenoh-bridge-ros2dds` because version
1.9.0 is already installed for ARM64 on the Jetson and AMD64 on the workstation.

## Network layout

```text
Go2 / RealSense network
        |
Jetson eth0: 192.168.123.18
        |
humble_fetch (Docker host networking)
        |
Zenoh bridge on Jetson
        |
Jetson wlan0: 192.168.11.8
        |
Wi-Fi / TCP port 7447
        |
Workstation eno1: 192.168.11.2
        |
Local Zenoh bridge and ROS 2 Humble
```

Docker must continue using host networking:

```bash
ssh unitree@192.168.11.8
docker inspect humble_fetch --format 'network={{.HostConfig.NetworkMode}} running={{.State.Running}}'
```

Expected:

```text
network=host running=true
```

## Forwarded topics

The bridge currently permits publishers and subscribers matching:

- `/camera/**` — RealSense camera topics
- `/go2_fetch/**` — fetch package topics
- `/rgb_map/**` — processed RGB-D outputs
- `/points_downsampled` — downsampled point cloud

Unitree command topics such as `/lowcmd` are deliberately not forwarded.

The filter is stored in:

- Workstation: `/home/ferdinand/unitree/zenoh_fetch_bridge.json5`
- Jetson host: `/home/unitree/zenoh_fetch_bridge.json5`
- Inside Docker: `/home/unitree/host/zenoh_fetch_bridge.json5`

## Start the Jetson bridge

Run on the Jetson host:

```bash
ssh unitree@192.168.11.8

docker exec -d \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  -e 'CYCLONEDDS_URI=<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" multicast="default"/></Interfaces></General></Domain></CycloneDDS>' \
  humble_fetch bash -c \
  'exec zenoh-bridge-ros2dds \
    -c /home/unitree/host/zenoh_fetch_bridge.json5 \
    -l tcp/192.168.11.8:7447 \
    --no-multicast-scouting router \
    > /home/unitree/host/zenoh_fetch_bridge.log 2>&1'
```

Check it:

```bash
docker exec humble_fetch pgrep -af zenoh-bridge-ros2dds
tail -f /home/unitree/zenoh_fetch_bridge.log
```

## Start the workstation bridge

Run on the workstation:

```bash
tmux new-session -d -s zenoh_fetch_bridge \
  "env ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 \
  CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"eno1\" multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>' \
  zenoh-bridge-ros2dds \
  -c /home/ferdinand/unitree/zenoh_fetch_bridge.json5 \
  -e tcp/192.168.11.8:7447 \
  --no-multicast-scouting client"
```

View its output:

```bash
tmux attach -t zenoh_fetch_bridge
```

Detach without stopping it using `Ctrl-b`, then `d`.

## Start RealSense

Inside `humble_fetch`:

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  align_depth.enable:=true
```

Start the required fetch launch file in another container terminal. The bridge
discovers matching topics automatically; it does not need to be restarted.

## Verify on the workstation

```bash
source /opt/ros/humble/setup.bash

ros2 topic list | grep -E 'camera|go2_fetch|rgb_map|points_downsampled'
ros2 topic hz /camera/color/image_raw
ros2 topic echo /go2_fetch/cube_state --once
```

If no matching nodes are running inside Docker, no matching topics will appear.

## Stop the bridges

Workstation:

```bash
tmux kill-session -t zenoh_fetch_bridge
```

Jetson:

```bash
ssh unitree@192.168.11.8
docker exec humble_fetch pkill -f zenoh-bridge-ros2dds
```

## Persistence

The bridges survive terminal closure, but they do not restart automatically
after the workstation, Jetson, or Docker container reboots. Repeat the two start
sections after a reboot.

