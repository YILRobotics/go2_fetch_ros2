# go2_fetch_ros2
ROS2 packages for the Unitree Go2 robot fetch project. Running a RL policy to push back a play cube


### Always disable conda env (with cdd as shortcut) when using ROS2!


## Connect to Unitree with SSH

Ethernet 
```bash
ssh unitree@192.168.123.18 # password 123
```

Wifi
```bash
ssh unitree@192.168.11.8  # password 123
```

run ros2 on computer
```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

make robot move:

```bash
ros2 topic pub /api/sport/request unitree_api/msg/Request "{header: {identity: {api_id: 1008}}, parameter: '{\"x\": 0.0, \"y\": 0.0, \"z\": 0.5}'}" -r 10
```


stand down 
```bash
ros2 topic pub --once /api/sport/request unitree_api/msg/Request \
"{header: {identity: {api_id: 1005}}, parameter: ''}"
```

others
```bash
ros2 launch fetch policy_test.launch.py
ros2 topic pub --once /go2_fetch/mode std_msgs/msg/String "{data: policy}"

ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.03, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10
```

## Make conda env for deploying RL policy with ROS2

```bash
python3 --version # Check systems python version (here Python 3.10.12)
conda create -n env_deploy python=3.10.12
conda activate env_deploy
```

```bash
pip install --upgrade pip
```

Install a CUDA-enabled PyTorch 2.7.0 build for CUDA 12.8 (here CUDA 13.0 but still 12.8 is still ok):
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Install extra stuff and stuff that is required
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

`Unitree_sdk2py` is a Python library that enables direct communication with Unitree robots.
It plays a crucial role in this project, allowing the system to collect sensor data from the robot and send velocity and motor commands in real time.

Clone the repository using Git :
```bash
cd ~/unitree
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

Install things for go2_odometry
```bash
pip install "empy==3.3.4" "catkin-pkg==1.1.0" "lark==1.1.1" colcon-common-extensions

conda install -c conda-forge pinocchio -y
```


**If you are inside a Conda env, pre-load the system libstdc++** or `rclpy` may complain about `GLIBCXX`. When you run ROS 2 Python nodes from a **Conda environment**, Conda provides its own `libstdc++.so.6`. ROS 2 Humble’s wheels (`rclpy`, others) were built against the **system** libstdc++ (newer GLIBCXX symbols). It is necessary to tell the dynamic loader to **prefer the system runtime**, preventing the `GLIBCXX_*` family of errors. It’s the cleanest fix when you want Conda + ROS 2 to coexist.
So, make sure to always pre-load the system libstdc++ (or add it to the `.bashrc` file):
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

maybe also because ros doesnt use python from library:   
```bash
export PYTHONPATH="$(python -c 'import site; print(site.getsitepackages()[0])'):$PYTHONPATH"
```



build problem in docker with conda 

colcon build --symlink-install --cmake-args -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/home/unitree/miniforge3/envs/go2_env_go2/lib -L/home/unitree/miniforge3/envs/go2_env_go2/lib" -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/home/unitree/miniforge3/envs/go2_env_go2/lib -L/home/unitree/miniforge3/envs/go2_env_go2/lib" -DPYTHON_EXECUTABLE=/home/unitree/miniforge3/envs/go2_env_go2/bin/python3

## Realsense Camera

start realsense camera

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

On Jetson: 
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

for foxglove ssh tunnel

ssh -L 8765:localhost:8765 unitree@192.168.11.10

start foxglove

ros2 launch foxglove_bridge foxglove_bridge_launch.xml


### Final working command AND RECOMENDED TO ADD TO YOUR `.bashrc` FILE on the PC:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

## Realsense Camera connect to PC

```bash
lsusb -t

sudo dmesg -w # check port live
```

```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
```

## Conda and ROS2 Conflicts

Disable conda auto-activation to avoid conflicts with ROS2:

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

Ensure conda is not in your `PATH`, as it interferes with ROS2.
### Killing ROS2 Processes (Not really neccesary)

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

Stop specific ROS2 processes:
```bash
pkill -f "ros2 topic echo"
```

### Installed Packages

```bash
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
pip install git+https://github.com/openai/CLIP.git
python3 -m pip uninstall -y numpy
```

### ROS2 Environment Configuration

```bash
unset CYCLONEDDS_URI
echo "$CYCLONEDDS_URI"  # should print empty line
```

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

Stream realsense with ffmpeg from terminal
```bash
ffplay /dev/video4
```

### Installed things 

```bash
sudo apt install v4l-utils 
sudo apt install ffmpeg
/usr/bin/python3 -m pip install --user pyrealsense2

pip install --user onnx>=1.12.0,<2.0.0
pip install --user onnxruntime-gpu
pip install --user onnxslim
```

### Information

* https://forum.mybotshop.de/t/unitree-go2-openmanipulator-realsense-d435i-realsense-d405-mid360-lidar-ros-foxy/1007
* https://techshare.co.jp/faq/unitree/unitree-go2_pc_lan.html
* https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2


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

