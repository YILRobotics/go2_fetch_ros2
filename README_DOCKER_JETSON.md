# Jetson Docker and ROS 2 quick guide

The Jetson is available at `192.168.11.8`. Its login user is `unitree`.

## Connect to the Jetson

From a terminal on this computer:

```bash
ssh unitree@192.168.11.8
```

Enter the Jetson password when SSH asks for it.

## Start the deployment container

Check if its started already

```bash
docker ps --filter name=humble_fetch
```

After logging in to the Jetson:

```bash
docker start humble_fetch
```

If `humble_fetch` has not been created yet, create it from the configured image:

```bash
docker run -d \
  --name humble_fetch \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  --ipc host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu:/opt/jetson-libs:/opt/jetson-libs/tegra \
  -v /usr/local/cuda-11.4:/usr/local/cuda-11.4:ro \
  -v /usr/lib/aarch64-linux-gnu:/opt/jetson-libs:ro \
  -v /home/unitree:/home/unitree/host \
  humble_dev:deploy sleep infinity
```

`--network host` is important for ROS 2 DDS discovery. The bind mount makes the
Jetson host directory available inside the container at `/home/unitree/host`.
The two read-only library mounts provide the JetPack 5.1.1 CUDA 11.4 and cuDNN
libraries required by NVIDIA's Jetson PyTorch build.

## Enter the container as the correct user

```bash
docker exec -it --user unitree humble_fetch bash
```

Check the user and environment:

```bash
whoami
echo "$HOME"
python --version
conda info --envs
```

Expected values are `unitree`, `/home/unitree`, Python 3.8.10, and an active
environment named `env_deploy`. ROS Humble's system Python remains Python 3.10.
If Conda is not active in an unusual non-interactive shell:

```bash
source /home/unitree/miniforge3/etc/profile.d/conda.sh
conda activate env_deploy
```

Check PyTorch and Jetson CUDA access:

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device")'
```

## View ROS 2 topics inside the container

Enter the container, then run:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
ros2 daemon stop
ros2 topic list
```

Do not source `/home/unitree/host/cyclonedds_ws/install/setup.bash` in this
container: that host workspace was built for ROS Foxy, while the container uses
ROS Humble.

Useful topic commands:

```bash
ros2 topic list -t
ros2 topic info /TOPIC_NAME --verbose
ros2 topic echo /TOPIC_NAME
ros2 topic hz /TOPIC_NAME
```

For example, if the robot publishes the sport mode state:

```bash
ros2 topic echo /sportmodestate
ros2 topic echo /lf/lowstate
```

## View ROS 2 topics directly on this computer

The computer and Jetson must be on the same network, multicast must be allowed,
and both sides must use the same `ROS_DOMAIN_ID`.

On this computer:

```bash
source /opt/ros/humble/setup.bash
source /home/ferdinand/unitree/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
ros2 daemon stop
ros2 topic list
```

If topics do not appear, identify the interface connected to the robot or Jetson:

```bash
ip -br address
```

Then force CycloneDDS to use it, replacing `INTERFACE_NAME`:

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="INTERFACE_NAME" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
ros2 daemon stop
ros2 topic list
```

## Stop or inspect the container

```bash
docker ps -a --filter name=humble_fetch
docker logs humble_fetch
docker stop humble_fetch
```
