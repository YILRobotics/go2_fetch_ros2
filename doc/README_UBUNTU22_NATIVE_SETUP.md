# Native Jetson Ubuntu 22.04 setup: ROS 2 Humble, Conda, Go2 fetch and RealSense

This guide reproduces the Docker development environment directly on Ubuntu
22.04. It installs ROS 2 through `apt`, keeps machine-learning packages in a
Conda environment, and builds the ROS workspace in `~/fetch_ws`.

Nothing in this guide requires Docker.

## 1. Confirm the machine type

```bash
lsb_release -ds
uname -m
nvidia-smi
python3 --version
```

Confirmed on `unitree-jetson-payload`:

```text
Ubuntu 22.04.5 LTS
aarch64
NVIDIA Jetson Orin
CUDA driver capability 12.6
Python 3.10.12
```

This is an ARM64 Jetson. Do not install PyTorch from the x86_64 `cu128` index.
A Jetson must use a PyTorch build matching its exact JetPack/L4T release.

`nvidia-smi` reports the maximum CUDA level supported by the driver. Check the
installed CUDA Toolkit and JetPack separately:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack 2>/dev/null || true
nvcc --version
```

## 2. Install ROS 2 Humble

Follow the official ROS instructions if the repository format changes:
[ROS 2 Humble Ubuntu installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

```bash
sudo apt update
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe

sudo curl -sSL \
  https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-dev-tools \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-rosidl-generator-dds-idl
```

Test ROS:

```bash
source /opt/ros/humble/setup.bash
ros2 --help
```

Do not install ROS 2 itself with Conda or pip. The system installation in
`/opt/ros/humble` supplies the correct C++ libraries and Python bindings.

## 3. Install system dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  pkg-config \
  python3-dev \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  libyaml-cpp-dev \
  libeigen3-dev \
  ros-humble-realsense2-camera \
  ros-humble-realsense2-camera-msgs \
  ros-humble-librealsense2
```

If the InEKF build requires Pinocchio and it is available in the configured
repositories:

```bash
sudo apt install -y python3-pinocchio
```

Ubuntu 22.04 normally provides CMake 3.22, which is sufficient for InEKF:

```bash
cmake --version
```

## 4. Use the existing Miniconda installation

Conda is already installed at `/home/unitree/miniconda3`. Do not install
Miniforge or a second copy of Conda.

Initialize it in the current terminal:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
conda info --base
conda env list
```

Expected base path:

```text
/home/unitree/miniconda3
```

## 5. Create the Conda environment

First test whether `go2_rl` already has Python 3.10 and working CUDA PyTorch:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate go2_rl

python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Choose exactly one of the following creation methods.

### Option A: clone the working Jetson environment

Use this option only when `go2_rl` reports Python 3.10 and
`CUDA available: True`:

```bash
conda deactivate
conda create -n go2_env_go2 --clone go2_rl
conda activate go2_env_go2
```

### Option B: create a clean environment

Use this when `go2_rl` has the wrong Python version or no working PyTorch:

```bash
conda deactivate
conda create -n go2_env_go2 python=3.10 pip -y
conda activate go2_env_go2
```

Do not run both options, and do not overwrite an existing `go2_env_go2`
environment without inspecting it first.

Confirm the selected environment:

```bash
python --version
which python
```

Install the Python packages needed by ROS builds and the fetch project:

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  "numpy==1.26.4" \
  "empy==3.3.4" \
  "catkin-pkg==1.1.0" \
  "lark-parser==0.12.0" \
  "PyYAML==6.0.1" \
  "matplotlib==3.8.4" \
  "scipy==1.15.3" \
  psutil \
  requests \
  colorama \
  typeguard \
  importlib-metadata \
  pytz
```

Keep NumPy on 1.26.4. Pinocchio and other ROS C++ Python bindings may fail or
segfault when loaded with NumPy 2.x.

## 6. Make the Conda environment use system ROS

Activate Conda first, then source ROS:

```bash
conda activate go2_env_go2
source /opt/ros/humble/setup.bash
```

Verify that the Conda Python can import ROS:

```bash
python - <<'PY'
import sys
import rclpy
import yaml
import numpy

print("Python:", sys.executable)
print("rclpy:", rclpy.__file__)
print("NumPy:", numpy.__version__)
PY
```

If `rclpy` cannot be found, use:

```bash
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
```

## 7. Install Jetson-compatible CUDA PyTorch

> **STOP — ARM64 Jetson:** never use the PyTorch `cu128` index from
> `download.pytorch.org` on this machine. It contains x86_64 packages. The
> literal `...` shown in abbreviated shell examples is explanatory notation,
> not a valid pip requirement and must never be copied into a command.

If Option A cloned a working `go2_rl`, PyTorch is already present. If Option B
created a clean environment, confirm the Jetson release and CUDA runtime:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack 2>/dev/null || true
nvcc --version
ldconfig -p | grep -E 'libcudart|libcudnn|libcupti' || true
```

This machine reports L4T `R36.4.7`, Python 3.10, and CUDA driver capability
12.6. The absence of `nvcc` means the CUDA compiler/toolkit is not installed;
it does not prevent a prebuilt PyTorch wheel from using installed CUDA runtime
libraries.

Install the CUDA runtime extras that the Jetson PyTorch wheel may load at
import time:

```bash
sudo apt update
sudo apt install -y libcudnn9-cuda-12 cuda-cupti-12-6
sudo ldconfig
```

If `cuda-cupti-12-6` is already installed, this command is harmless. If the
package is missing from apt, check available package names with:

```bash
apt-cache search cupti | grep -E 'cupti|cuda'
```

The Jetson AI Lab `jp6/cu126` ARM64 index provides the compatible stable pair
PyTorch 2.8.0 and torchvision 0.23.0. Install the exact ARM64 wheels by URL so
pip cannot accidentally select an x86_64 or CPU-only package:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate go2_env_go2

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.26.4" pillow \
  filelock typing-extensions sympy networkx jinja2 fsspec

python -m pip install --no-cache-dir \
  "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
  "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
```

These are community-maintained Jetson ARM64 wheels, not the x86_64 wheels from
`download.pytorch.org`. Do not substitute a wheel for another CUDA/JetPack
release.

After installing or cloning PyTorch, activate `go2_env_go2` and verify it:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate go2_env_go2

# If you named the environment differently, for example env_deploy,
# activate that environment instead and confirm it:
echo "$CONDA_DEFAULT_ENV"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")

print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(2000, 2000, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Result device:", y.device)
print("GPU calculation: OK")
PY
```

## 7.1 Install Ultralytics

Install a NumPy-1.x-compatible OpenCV before Ultralytics. This prevents a newer
OpenCV dependency from upgrading NumPy to 2.x and breaking ROS/Pinocchio Python
bindings.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate go2_env_go2

python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84"
python -m pip install ultralytics
python -m pip check
```

Verify Ultralytics without downloading or running a model:

```bash
python - <<'PY'
import torch
import torchvision
import ultralytics

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## 8. Create the workspace

```bash
mkdir -p "$HOME/fetch_ws/src"
cd "$HOME/fetch_ws/src"
```

Clone the public dependencies:

```bash
git clone https://github.com/unitreerobotics/unitree_ros2.git
git clone https://github.com/inria-paris-robotics-lab/go2_description.git
git clone https://github.com/inria-paris-robotics-lab/invariant-ekf.git
git clone https://github.com/inria-paris-robotics-lab/go2_odometry.git
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
```

Place the project repository at:

```text
~/fetch_ws/src/go2_fetch_ros2
```

Do not create a second nested workspace under `src`.

## 9. Install the Unitree Python SDK

The SDK pins `cyclonedds==0.10.2`. Its Python build expects a conventional DDS
prefix containing `include`, `lib`, and `bin`.

```bash
conda activate go2_env_go2
source /opt/ros/humble/setup.bash

mkdir -p "$HOME/cyclonedds-prefix"/{include,lib,bin}
ln -sfn /opt/ros/humble/include/dds "$HOME/cyclonedds-prefix/include/dds"

DDS_LIBRARY="$(find /opt/ros/humble -name libddsc.so -print -quit)"
test -n "$DDS_LIBRARY"
ln -sf "$(dirname "$DDS_LIBRARY")"/libddsc.so* "$HOME/cyclonedds-prefix/lib/"

export CYCLONEDDS_HOME="$HOME/cyclonedds-prefix"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME:/opt/ros/humble:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$(dirname "$DDS_LIBRARY"):${LD_LIBRARY_PATH:-}"

python -m pip install --no-build-isolation "cyclonedds==0.10.2"
python -m pip install --no-build-isolation -e "$HOME/fetch_ws/src/unitree_sdk2_python"
```

Verify:

```bash
python - <<'PY'
import cyclonedds
import unitree_sdk2py
print("CycloneDDS Python: OK")
print("Unitree SDK Python: OK")
PY
```

## 10. Resolve ROS package dependencies

Run once per machine:

```bash
sudo rosdep init 2>/dev/null || true
rosdep update
```

Install workspace dependencies:

```bash
conda activate go2_env_go2
source /opt/ros/humble/setup.bash
cd "$HOME/fetch_ws"

rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

## 11. Build InEKF

First check its required CMake version:

```bash
head -n 10 "$HOME/fetch_ws/src/invariant-ekf/CMakeLists.txt"
cmake --version
```

Build and install it only if it is not already provided by the workspace:

```bash
conda activate go2_env_go2
source /opt/ros/humble/setup.bash

cd "$HOME/fetch_ws/src/invariant-ekf"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$(which python)"
cmake --build build -j"$(nproc)"
sudo cmake --install build
sudo ldconfig
```

Do not change `cmake_minimum_required` unless CMake reports an actual version
error.

## 12. Build the ROS workspace

```bash
conda activate go2_env_go2
source /opt/ros/humble/setup.bash

cd "$HOME/fetch_ws"
colcon build --symlink-install
```

Source it:

```bash
source "$HOME/fetch_ws/install/setup.bash"
```

Sanity checks:

```bash
ros2 pkg list | grep -E '^(fetch|go2_|unitree_)'
ros2 pkg executables fetch
```

## 13. RealSense permissions and launch

Add the native user to the camera-related groups:

```bash
sudo usermod -aG video,plugdev "$USER"
```

Log out and back in after changing groups. Check the camera:

```bash
lsusb -t
```

The D435i should report `5000M`, not `480M`.

Launch a bandwidth-conscious profile:

```bash
conda activate go2_env_go2
source /opt/ros/humble/setup.bash
source "$HOME/fetch_ws/install/setup.bash"

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  rgb_camera.color_profile:=640,480,15 \
  enable_depth:=true \
  depth_module.depth_profile:=640,480,15 \
  pointcloud.enable:=true \
  align_depth.enable:=true
```

Check locally:

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_rect_raw
```

Use RViz or `rqt_image_view` for images. Avoid printing full raw images with
`ros2 topic echo`.

## 14. Run odometry and fetch

Open a new terminal for each launch and prepare it with:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate go2_env_go2
source /opt/ros/humble/setup.bash
source "$HOME/fetch_ws/install/setup.bash"
```

Test fake odometry:

```bash
ros2 launch go2_odometry go2_odometry_switch.launch.py \
  odom_type:=fake base_height:=0.30
```

Run the real filter:

```bash
ros2 launch go2_odometry go2_odometry_switch.launch.py \
  odom_type:=use_full_odom
```

List the fetch launch files before selecting one:

```bash
find "$HOME/fetch_ws/src/go2_fetch_ros2/fetch/launch" \
  -maxdepth 1 -name '*.launch.py' -printf '%f\n'
```

For example:

```bash
ros2 launch fetch launch_cube_tracker_with_odom.launch.py
```

Verify outputs:

```bash
ros2 topic list | grep -E 'camera|go2_fetch|odometry|lowstate'
ros2 topic echo /go2_odometry/filtered --once
ros2 topic echo /go2_fetch/cube_state --once
```

## 15. ROS networking

Use the same domain on every participating machine:

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

For a computer directly connected to the Go2 network, select the interface that
has a `192.168.123.x` address:

```bash
ip -brief address
```

Example for interface `eno2`:

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2" multicast="default"/></Interfaces></General></Domain></CycloneDDS>'
```

For the existing routed Wi-Fi setup through the Jetson, use the instructions in
[README_WIFI_ROS2_BRIDGE.md](README_WIFI_ROS2_BRIDGE.md). Raw color, depth,
aligned depth, and point clouds together can saturate Wi-Fi; use compression or
rate limits for camera data.

## 16. Convenient shell function

Add this to `~/.bash_aliases` after the workspace builds successfully:

```bash
go2env() {
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate go2_env_go2
  source /opt/ros/humble/setup.bash
  source "$HOME/fetch_ws/install/setup.bash"
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
}
```

Reload and activate:

```bash
source ~/.bashrc
go2env
```

## Troubleshooting

### `GLIBCXX_* not found`

Conda is loading an incompatible `libstdc++.so.6`. On this ARM64 Jetson:

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6
```

Use this only when the error occurs; do not set it globally without need.

### `ImportError: libcudnn.so.9` when importing Torch

The selected Jetson PyTorch wheel requires the cuDNN 9 runtime. Prefer the
native ARM64 system package when the NVIDIA apt repository provides it:

```bash
apt-cache policy libcudnn9-cuda-12
```

If that command shows a candidate version:

```bash
sudo apt install -y libcudnn9-cuda-12
sudo ldconfig
```

If no apt candidate exists, install the matching ARM64 runtime inside the
active Conda environment instead:

```bash
python -m pip install "nvidia-cudnn-cu12==9.10.2.21"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
```

Test `import torch` after using one method. Do not install both system and pip
cuDNN unless there is a specific reason.

### `ImportError: libcupti.so.12` when importing Torch

The selected Jetson PyTorch wheel also needs CUPTI, the CUDA profiling/runtime
support library. On Jetson it may not be installed by default, especially when
`nvcc` is also missing.

First check whether the library exists:

```bash
sudo find /usr/local /usr/lib -name 'libcupti.so*' 2>/dev/null
```

If nothing is printed, install CUPTI:

```bash
sudo apt update
sudo apt install -y cuda-cupti-12-6
sudo ldconfig
```

Then verify it appears:

```bash
sudo find /usr/local /usr/lib -name 'libcupti.so*' 2>/dev/null
```

If it appears under `/usr/local/cuda-12.6/extras/CUPTI/lib64`, expose that path
before running Python:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/extras/CUPTI/lib64:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
```

Then run the PyTorch CUDA test again.

If a requested Conda environment does not exist, `conda activate` reports an
error and leaves the previously active environment selected. Always confirm:

```bash
echo "$CONDA_DEFAULT_ENV"
which python
```

### NumPy 2.x or Pinocchio import failure

```bash
python -m pip install --force-reinstall "numpy==1.26.4"
```

### ROS CLI shows stale topics

```bash
ros2 daemon stop
ROS2CLI_NO_DAEMON=1 ros2 topic list
```

### `cyclonedds==0.10.2` cannot locate CycloneDDS

Confirm all three prefix directories exist:

```bash
test -f "$HOME/cyclonedds-prefix/include/dds/dds.h" && echo header_OK
test -e "$HOME/cyclonedds-prefix/lib/libddsc.so" && echo library_OK
test -d "$HOME/cyclonedds-prefix/bin" && echo bin_OK
echo "$CYCLONEDDS_HOME"
```

### RealSense topic exists but raw images are slow over Wi-Fi

Test the topic on the camera machine first:

```bash
ros2 topic hz /camera/color/image_raw
```

If it works locally but not remotely, the camera and USB are working. Reduce the
forwarded frame rate or use compressed image transport; do not forward every raw
camera representation and point cloud at full rate.
