# Simple Summary

You can use internet and robot ROS2 at the same time.

Best setup:

```text
PC internet:      eno1 / 192.168.11.x
PC robot ROS2:    eno2 / 192.168.123.222
Robot ROS2 LAN:   enP8p1s0 / 192.168.123.18
Robot internet:   WiFi, or internet shared from PC
```

## What Works

### PC has internet on one cable and robot ROS2 on another cable

This works.

```text
Internet traffic -> PC eno1
Robot ROS2       -> PC eno2
```

Why it works: Linux can use different network interfaces for different traffic. Normal internet goes to the default route on `eno1`. ROS2 multicast and robot traffic are forced to `eno2`.

### Robot uses WiFi for internet and LAN for ROS2

This should work.

```text
Robot internet -> WiFi
Robot ROS2     -> enP8p1s0
```

Why it works: internet and ROS2 do not need to use the same interface. The robot can keep its default route on WiFi while DDS multicast for ROS2 stays on `enP8p1s0`.

### Robot gets internet through the PC LAN cable

This can work, but needs extra setup.

The PC must share internet from `eno1` to `eno2` using routing/NAT. This is more complex than using robot WiFi, but possible.

## What Does Not Work

### Only using WiFi for native Go2 ROS2 topics

This did not work for the native Go2 topics.

Why: the Go2 DDS graph is on the internal robot LAN `192.168.123.x`, not only on the WiFi network. When CycloneDDS was forced to WiFi, the robot only saw local topics like `/rosout` and `/parameter_events`.

### Letting DDS multicast choose the wrong interface

This does not work reliably.

DDS discovery uses multicast such as `239.255.0.1`. If that multicast goes through the internet interface instead of the robot LAN, the PC will not discover the robot topics.

Correct rule:

```text
Internet route:       eno1 or WiFi
ROS2 multicast route: robot LAN interface
```

---

# ROS2 Go2 Multi-Machine Discovery Fix

## Current Status

The PC can now see the Unitree Go2 ROS2 topics over the robot LAN.

Final working command AND RECOMENDED TO ADD TO YOUR `.bashrc` FILE on the PC:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

then try 
```bash
ros2 topic list # or ros2 topic list --no-daemon
```

The PC now sees robot topics such as:

```text
/lowstate
/sportmodestate
/utlidar/cloud
/wirelesscontroller
/api/sport/request
```

## Final Network Setup

```text
PC internet interface:     eno1 = 192.168.11.2
PC robot LAN interface:   eno2 = 192.168.123.222
Robot internal interface: enP8p1s0 = 192.168.123.18
```

Normal internet stays on `eno1`.

Robot ROS2/DDS traffic uses `eno2`.

## Original Problem

The PC could ping or SSH to the robot, but ROS2 discovery only showed local topics:

```text
/parameter_events
/rosout
```

The robot itself could see all Go2 topics locally.

## Root Causes

### 1. The Go2 DDS graph is on the internal robot LAN

The native Go2 topics are available through the robot internal network:

```text
192.168.123.x
```

The robot interface for that network is:

```text
enP8p1s0 = 192.168.123.18
```

The PC needed to join that network using:

```text
eno2 = 192.168.123.222/24
```

### 2. DDS multicast had to use the robot LAN

ROS2 DDS discovery uses multicast, especially:

```text
239.255.0.1
```

The PC needed multicast routed through `eno2`:

```bash
sudo ip route replace 239.255.0.0/16 dev eno2
```

Verify on the PC:

```bash
ip route get 239.255.0.1
```

Expected:

```text
multicast 239.255.0.1 dev eno2 src 192.168.123.222
```

The robot also needed multicast routed through `enP8p1s0`:

```bash
sudo ip route replace 239.255.0.0/16 dev enP8p1s0
```

Verify on the robot:

```bash
ip route get 239.255.0.1
```

Expected:

```text
multicast 239.255.0.1 dev enP8p1s0 src 192.168.123.18
```

### 3. PC firewall blocked incoming DDS UDP

UFW was active on the PC and default incoming traffic was denied.

ROS2/DDS needs incoming UDP from the robot.

This fixed it on the PC:

```bash
sudo ufw allow in on eno2 from 192.168.123.0/24 proto udp
```

This only allows UDP on the robot LAN interface `eno2` from the robot subnet.

It does not open the internet interface `eno1`.

### 4. The ROS2 daemon cached stale settings

The alias was:

```bash
alias rtl='ros2 topic list'
```

That uses the ROS2 daemon/cache.

If the daemon was started with old settings, such as a bad `CYCLONEDDS_URI` or wrong interface, it can keep showing only:

```text
/parameter_events
/rosout
```

Using this bypasses the daemon and reads DDS directly:

```bash
ros2 topic list --no-daemon
```

Recommended alias:

```bash
alias rtl='ros2 topic list --no-daemon'
```

## Recommended PC Bash Settings

Add or keep this in `~/.bashrc` on the PC:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

Make sure there is no old `CYCLONEDDS_URI` line using `eno1`.

Recommended `~/.bash_aliases` entry:

```bash
alias rtl='ros2 topic list --no-daemon'
```

Reload after editing:

```bash
source ~/.bashrc
source ~/.bash_aliases
```

## Useful Checks

Check internet still uses `eno1`:

```bash
ip route get 8.8.8.8
```

Expected:

```text
8.8.8.8 via 192.168.11.1 dev eno1
```

Check ROS2 multicast uses `eno2`:

```bash
ip route get 239.255.0.1
```

Expected:

```text
multicast 239.255.0.1 dev eno2 src 192.168.123.222
```

Check robot is reachable:

```bash
ping 192.168.123.18
ssh unitree@192.168.123.18
```

Check topics directly:

```bash
ros2 topic list --no-daemon
```

## Persistence Notes

The UFW rule is persistent.

The `CYCLONEDDS_URI` export is persistent only if added to `~/.bashrc`.

The multicast route commands are temporary unless made persistent with NetworkManager or another startup method:

```bash
sudo ip route replace 239.255.0.0/16 dev eno2
sudo ip route replace 239.255.0.0/16 dev enP8p1s0
```

The first command is for the PC.

The second command is for the robot.

## Simple Explanation

```text
The robot topics live on 192.168.123.x.
The PC must use eno2 for robot ROS2.
DDS multicast must go through eno2.
The PC firewall must allow UDP from the robot LAN.
Use --no-daemon to avoid stale ROS2 daemon cache.
```
