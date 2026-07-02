# go2_fetch_ros2

ROS 2 packages for the Unitree Go2 robot fetch project. This project runs an RL policy to push back a play cube.

> Always disable the Conda environment when using ROS 2.
>
> Shortcut note: use `cdd` if that is configured on your machine.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Connect to Unitree](#connect-to-unitree)
- [ROS 2 Network Setup](#ros-2-network-setup)
- [TensorRT Policy Engines](#tensorrt-policy-engines)
- [Robot Motion Commands](#robot-motion-commands)
- [Realsense Camera](#realsense-camera)
- [Foxglove and RViz2](#foxglove-and-rviz2)
- [Useful Commands](#useful-commands)
- [Troubleshooting](#troubleshooting)
- [Information](#information)
- [Unitree Onboard Computer Specifications](#unitree-onboard-computer-specifications)

## Environment Setup

### Conda and ROS 2 Conflicts

Disable Conda auto-activation to avoid conflicts with ROS 2:

```bash
conda config --set auto_activate false
conda config --set auto_activate_base false
```

When using VS Code, reload your terminal cleanly:

```bash
exec bash
```

This is cleaner than sourcing `~/.bashrc`. Verify your Python environment:

```bash
echo $PATH
which python3  # Should output: /usr/bin/python3
```

Ensure Conda is not in your `PATH`, as it interferes with ROS 2.

### Conda Environment for Deploying RL Policy with ROS 2

```bash
python3 --version # Check system Python version. Here: Python 3.10.12.
conda create -n env_deploy python=3.10.12
conda activate env_deploy
```

```bash
pip install --upgrade pip
```

### Install PyTorch

Install a CUDA-enabled PyTorch 2.7.0 build for CUDA 12.8. CUDA 13.0 is present here, but 12.8 is still OK:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### Install Extra Requirements

```bash
pip install --no-cache-dir \
  matplotlib==3.8.4 \
  scipy==1.15.3 \
  PyYAML==6.0.1 \
  psutil \
  requests \
  colorama \
  typeguard \
  importlib-metadata \
  pytz
```

### Install Unitree SDK2 Python

`Unitree_sdk2py` is a Python library that enables direct communication with Unitree robots. It plays a crucial role in this project, allowing the system to collect sensor data from the robot and send velocity and motor commands in real time.

Clone the repository using Git:

```bash
cd ~/unitree
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

### Install Things for `go2_odometry`

```bash
pip install "empy==3.3.4" "catkin-pkg==1.1.0" "lark==1.1.1" colcon-common-extensions

conda install -c conda-forge pinocchio -y
```

### Installed Packages

```bash
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
pip install git+https://github.com/openai/CLIP.git
python3 -m pip uninstall -y numpy
```

### Installed Things

```bash
sudo apt install v4l-utils
sudo apt install ffmpeg
/usr/bin/python3 -m pip install --user pyrealsense2

pip install --user onnx>=1.12.0,<2.0.0
pip install --user onnxruntime-gpu
pip install --user onnxslim
```

### Compressed Image Transport Plugin

For compressed images:

```bash
sudo apt install ros-humble-image-transport-plugins
```

## Connect to Unitree

### Ethernet

```bash
ssh unitree@192.168.123.18 # password 123
```

### Wi-Fi

```bash
ssh unitree@192.168.11.8  # password 123
```

copy policy model:

```bash
cd src/go2_fetch_ros2/fetch/models

rsync -avzP ferdinand@192.168.11.2:/home/ferdinand/fetchrobot/ferdinand/go2_fetch_rl/logs/rsl_rl/unitree_go2_velocity_4l/2026-06-30_11-11-40_walk_ff_5/exported/policy.pt .
```

## ROS 2 Network Setup

### Run ROS 2 on Computer

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

### On Robot

```bash
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>enP8p1s0</NetworkInterfaceAddress><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>"
```

### Final Working Command

Recommended to add to your `.bashrc` file on the PC:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

### ROS 2 Environment Configuration

```bash
unset CYCLONEDDS_URI
echo "$CYCLONEDDS_URI"  # should print empty line
```

## TensorRT Policy Engines

### Make Engine File from .onnx file example

```bash
cd /home/unitree/fetch_ws/src/go2_fetch_ros2/fetch/models/unitree_go2_velocity_4l/2026-04-05_12-01-56_walk_2

/usr/src/tensorrt/bin/trtexec \
  --onnx=policy.onnx \
  --saveEngine=policy.engine \
  --fp16
```


A TensorRT .engine is a serialized optimized execution plan. TensorRT does not provide the same one-line model interface,
so the added TensorRTPolicy wrapper must perform those missing tasks:

- Deserialize the .engine file.
- Find its input and output tensor names.
- Read expected shapes and data types.
- Validate that it has one input and one output.
- Create a TensorRT execution context.
- Allocate CUDA input/output memory using Torch tensors.
- Bind those memory addresses to TensorRT.
- Copy each NumPy observation to GPU memory.
- Execute inference on a CUDA stream.
- Wait for inference to finish.
- Copy the result back to NumPy.
- Report useful errors for wrong shapes, unsupported types, missing CUDA, or failed inference.


Check speed:
```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=policy.engine
```

Walking policy:
  GPU compute: 0.052 ms
  End-to-end TensorRT latency: 0.069 ms

  Your node reports 16–29 ms, so TensorRT itself is not the bottleneck. Over 99% of “inference” time is in the Python
    wrapper:

  - NumPy → Torch conversion
  - Torch CPU → CUDA copy
  - CUDA stream synchronization
  - CUDA → Torch CPU copy
  - Torch → NumPy conversion
  - Possible GPU contention with camera inference



[policy_node-1] Policy cycle time: avg=32.99 ms, min=27.01 ms, max=41.63 ms (50 cycles)
[policy_node-1] Policy step profile step=650 input=0.20ms command=0.08ms markers=0.01ms observation=0.13ms inference=30.77ms trt_host_input=0.03ms trt_h2d=9.15ms trt_enqueue=3.50ms trt_execute=7.03ms trt_d2h=7.24ms trt_sync_wait=3.43ms trt_total=30.68ms motor_build=0.19ms send_total=1.31ms torque_limit=0.28ms crc=0.73ms publish=0.29ms record=0.05ms total=32.74ms

## Robot Motion Commands

### Make Robot Move

```bash
ros2 topic pub /api/sport/request unitree_api/msg/Request "{header: {identity: {api_id: 1008}}, parameter: '{\"x\": 0.0, \"y\": 0.0, \"z\": 0.5}'}" -r 10
```

### Stand Down

```bash
ros2 topic pub --once /api/sport/request unitree_api/msg/Request \
"{header: {identity: {api_id: 1005}}, parameter: ''}"
```

### Other Commands

```bash
ros2 launch fetch policy_test.launch.py
ros2 topic pub --once /go2_fetch/mode std_msgs/msg/String "{data: policy}"

ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.03, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10
```

## Realsense Camera

PYTHONPATH="$PYTHONPATH:/usr/lib/python3/dist-packages" ros2 run fetch go2_camera_node

### Realsense Camera Setup

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo mv 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and replug the camera, then verify:

```bash
realsense-viewer
```

### Realsense Camera Connected to PC

```bash
lsusb -t

sudo dmesg -w # check port live
```

```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
```

### Start Realsense Camera

```bash
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  -r __ns:=/realsense \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p pointcloud.enable:=true \
  -p pointcloud.stream_filter:=2 \
  -p pointcloud.stream_index_filter:=0 \
  -p align_depth.enable:=true \
  -p decimation_filter.enable:=true \
  -p decimation_filter.filter_magnitude:=4 \
  -p spatial_filter.enable:=true \
  -p temporal_filter.enable:=true \
  -p enable_sync:=true
```

### Start Realsense Camera on Jetson

```bash
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  -r __ns:=/realsense \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p pointcloud__neon_.enable:=true \
  -p pointcloud.stream_filter:=2 \
  -p pointcloud.stream_index_filter:=0 \
  -p align_depth.enable:=true \
  -p decimation_filter.enable:=true \
  -p decimation_filter.filter_magnitude:=4 \
  -p spatial_filter.enable:=true \
  -p temporal_filter.enable:=true \
  -p enable_sync:=true
```

### Stream Realsense with FFmpeg from Terminal

```bash
ffplay /dev/video4
```

## Foxglove and RViz2

### Foxglove SSH Tunnel

```bash
ssh -L 8765:localhost:8765 unitree@192.168.11.10
```

### Start Foxglove

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

### Foxglove Commands

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml

ros2 launch foxglove_bridge foxglove_bridge_launch.xml topic_whitelist:="['/rosout', '/utlidar/.*']"

ros2 run foxglove_bridge foxglove_bridge --ros-args --params-file /home/unitree/fetch_ws/src/go2_fetch_ros2/fetch/config/foxglove_config.yaml
```

### Repeated Foxglove Notes from Earlier

For Foxglove SSH tunnel:

```bash
ssh -L 8765:localhost:8765 unitree@192.168.11.10
```

Start Foxglove:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

### RViz2

```bash
rviz2 -d /home/ferdinand/unitree/src/go2_fetch_ros2/fetch/rviz/realsense.rviz
```

## Useful Commands

### Record ROS 2 Bags

```bash
cd rosbags
ros2 bag record -e "(/go2_fetch/.*|/go2_odometry/filtered|/tf|/camera/depth/color/points|/camera/color/image_raw/compressed|/robot_description|/tf_static|/lf/lowstate)" -x ".*compressedDepth.*"

ros2 bag record -e "(/go2_fetch/.*|/go2_odometry/filtered|/tf|/robot_description|/tf_static|/lf/lowstate)"
```

Now the cause is clear: rosbag’s default 100 MiB cache fills after about 7.3 seconds—exactly when cube
  tracking stops. Flushing that cache causes enough DDS/I/O pressure that the tracker’s best-effort camera
  subscriptions stop receiving complete messages.

  Try direct writing with no cache:

  ros2 bag record --max-cache-size 0 \
    -e "(/go2_fetch/.*|/go2_odometry/filtered|/tf|/camera/depth/color/points|/camera/color/image_raw/
    compressed|/robot_description|/tf_static|/lf/lowstate)" \
    -x ".*compressedDepth.*"
    

### Check Disk Usage

```bash
du -h --max-depth=1
```

## Troubleshooting

### Conda and ROS 2 `libstdc++` Fix

If you are inside a Conda environment, pre-load the system `libstdc++` or `rclpy` may complain about `GLIBCXX`.

When you run ROS 2 Python nodes from a Conda environment, Conda provides its own `libstdc++.so.6`. ROS 2 Humble wheels such as `rclpy` were built against the system `libstdc++`, which has newer `GLIBCXX` symbols. Tell the dynamic loader to prefer the system runtime to prevent `GLIBCXX_*` errors. This is the cleanest fix when Conda and ROS 2 need to coexist.

Make sure to always pre-load the system `libstdc++`, or add this to `.bashrc`:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Maybe also needed because ROS does not use Python from the library:

```bash
export PYTHONPATH="$(python -c 'import site; print(site.getsitepackages()[0])'):$PYTHONPATH"
```

### Build Problem in Docker with Conda

```bash
colcon build --symlink-install --cmake-args -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/home/unitree/miniforge3/envs/go2_env_go2/lib -L/home/unitree/miniforge3/envs/go2_env_go2/lib" -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/home/unitree/miniforge3/envs/go2_env_go2/lib -L/home/unitree/miniforge3/envs/go2_env_go2/lib" -DPYTHON_EXECUTABLE=/home/unitree/miniforge3/envs/go2_env_go2/bin/python3
```

### Killing ROS 2 Processes

Not really necessary.

```bash
cd ~/unitree
pkill -9 -f ros2
pkill -9 -f ros2-daemon
pkill -9 -f _ros2_daemon
ros2 daemon stop
ros2 daemon start
unset PYTHONPATH
unset ROS_DOMAIN_ID
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash
which python3
ros2 topic list
source install/setup.bash
```

Stop specific ROS 2 processes:

```bash
pkill -f "ros2 topic echo"
```

## Information

- https://forum.mybotshop.de/t/unitree-go2-openmanipulator-realsense-d435i-realsense-d405-mid360-lidar-ros-foxy/1007
- https://techshare.co.jp/faq/unitree/unitree-go2_pc_lan.html
- https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2

<br>

## Unitree Onboard Computer Specifications

### 1. Core Processor (CPU)

- **Model:** ARMv8 Processor, 6-core
- **Architecture:** aarch64, 64-bit
- **Clock Speed:** Max 1510 MHz, 1.5 GHz
- **Hardware Type:** Likely an **NVIDIA Jetson Xavier NX** module, consistent with 6-core ARMv8 and about 8 GB RAM specs.

### 2. Memory (RAM)

- **Total Capacity:** 7.2 GiB, effective 8 GB physical RAM
- **Currently Used:** about 831 MiB
- **Available:** about 6.1 GiB
- **Swap Space:** 3.6 GiB configured

### 3. Storage (Disk)

- **Main Drive:** 234 GB NVMe SSD, `/dev/nvme0n1p1`
- **Used Space:** 22 GB, 10%
- **Available Space:** 203 GB
- **Note:** The use of an NVMe drive indicates this is an upgraded or "Edu" version of the Unitree controller, as base models often use slower eMMC storage.

### 4. Graphics and AI (GPU)

- **Status:** `nvidia-smi` not found.
- **Note:** On Jetson devices, `nvidia-smi` is often absent. To check GPU usage on this hardware, use:

  ```bash
  sudo tegrastats
  ```

- **Package Info:** `nvidia-jetpack` was not found in the apt-cache, suggesting the OS may have been custom-flashed or the repositories are not currently pointed to the NVIDIA servers.

### 5. Operating System

- **User:** unitree
- **Hostname:** ubuntu
- **Environment:** ROS 2 Foxy and ROS 1 Noetic installed via FishROS.

Your robot is equipped with an 8 GB Xavier NX or a similar Jetson module. You have plenty of storage, 203 GB free, and RAM, 6 GB available, which is more than enough to run the SLAM and LiDAR nodes you have active.

However, the "package not found" for Jetpack suggests that if you need to compile GPU-accelerated code, such as custom CUDA kernels, you might need to fix your source lists or rely on the pre-installed libraries in `/usr/local/cuda-11.4/`.
