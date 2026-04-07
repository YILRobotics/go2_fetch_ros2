# go2_custom_description
Unitree GO2 description package for ROS 2.

## ROS 2 Humble
Build in a ROS 2 workspace with `colcon`:

```bash
source /opt/ros/humble/setup.bash
cd <your_ws>
colcon build --packages-select go2_custom_description go2_rviz
source install/setup.bash
```

Publish robot state:

```bash
ros2 launch go2_custom_description robot.launch.py
```

Open RViz with the provided config:

```bash
ros2 launch go2_rviz rviz.launch.py
```

By default `robot.launch.py` loads `urdf/go2_custom_description.urdf`.
If you want the xacro model, launch with:

```bash
ros2 launch go2_custom_description robot.launch.py description_file:=xacro/robot.xacro
```

