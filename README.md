# go2_fetch_ros2
ROS2 packages for the Unitree Go2 robot fetch project. Running a RL policy to push back a play cube


### Always disable conda env (with cdd as shortcut) when using ROS2!


## Connect to Unitree with SSH

```bash
ssh unitree@192.168.123.18 # password 123, then choose (1) ROS2 foxy
```

## Realsense Camera connect to PC

```bash
lsusb -t

sudo dmesg -w # check port live
```

```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
```


Information: 
https://forum.mybotshop.de/t/unitree-go2-openmanipulator-realsense-d435i-realsense-d405-mid360-lidar-ros-foxy/1007 


<br>

## Unitree Onboard Computer Specifications

## 1. Core Processor (CPU)
* **Model:** ARMv8 Processor (6-Core)
* **Architecture:** aarch64 (64-bit)
* **Clock Speed:** Max 1510 MHz (1.5 GHz)
* **Hardware Type:** Likely an **NVIDIA Jetson Xavier NX** module (consistent with 6-core ARMv8 and ~8GB RAM specs).

## 2. Memory (RAM)
* **Total Capacity:** 7.2 GiB (Effective 8GB Physical RAM)
* **Currently Used:** ~831 MiB
* **Available:** ~6.1 GiB
* **Swap Space:** 3.6 GiB configured

## 3. Storage (Disk)
* **Main Drive:** 234 GB NVMe SSD (`/dev/nvme0n1p1`)
* **Used Space:** 22 GB (10%)
* **Available Space:** 203 GB
* **Note:** The use of an NVMe drive indicates this is an upgraded or "Edu" version of the Unitree controller, as base models often use slower eMMC storage.

## 4. Graphics & AI (GPU)
* **Status:** `nvidia-smi` not found. 
* **Note:** On Jetson devices, `nvidia-smi` is often absent. To check GPU usage on this hardware, use:
    ```bash
    sudo tegrastats
    ```
* **Package Info:** `nvidia-jetpack` was not found in the apt-cache, suggesting the OS may have been custom-flashed or the repositories are not currently pointed to the NVIDIA servers.

## 5. Operating System
* **User:** unitree
* **Hostname:** ubuntu
* **Environment:** ROS 2 (Foxy) and ROS 1 (Noetic) installed via FishROS.



Your robot is equipped with an 8GB Xavier NX or a similar Jetson module. You have plenty of storage (203GB free) and RAM (6GB available), which is more than enough to run the SLAM and LiDAR nodes you have active.

However, the "package not found" for Jetpack suggests that if you need to compile GPU-accelerated code (like custom CUDA kernels), you might need to fix your source lists or rely on the pre-installed libraries in /usr/local/cuda-11.4/.

