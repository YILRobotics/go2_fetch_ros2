 
 Here is a concise summary of the issue, the fix, and how to manually handle the mode switching now.

### The Core Problem

Your application was crashing (`DDS_RETCODE_BAD_PARAMETER`) because it tried to run **two separate DDS systems inside a single Python process**:

1. **ROS 2 (`rclpy`)**, which initializes CycloneDDS for robot state and odometry.
2. **The Unitree SDK (`unitree_sdk2py`)**, which tried to initialize CycloneDDS a second time to send commands.

Because they conflicted instantly, the program crashed before any RL model logic or robot safety modes even mattered.

---

### The Fix (Option 1)

To eliminate the conflict, the `policy_node` was converted to use **pure ROS 2 communication**.

* **What changed:** Instead of sending commands directly through the Unitree SDK, the node now publishes a standard ROS 2 message (`unitree_go/msg/LowCmd`) and subscribes to ROS 2 state topics (`/lf/lowstate`).
* **What stays the same:** The core functionality—loading your PyTorch (`.pt`) models, processing observations, and calculating the torque/joint commands—remains exactly identical to the original repository.


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 ---------------------------------------------
 ---------------------------------------------
 ---------------------------------------------
 ---------------------------------------------





 policy_nod: Multicast Ports: discovery 7400 data 7401
  1782378249.508292 [0] policy_nod: failed to increase socket receive buffer size to 1048576 bytes, continuing with 425984 bytes
  1782378249.508318 [0] policy_nod: interface enP8p1s0: transmit port 52992
  (env_deploy) unitree@unitree-jetson-payload:~/fetch_ws$
 
 
• The /tmp/cdds.LOG is useful: your network interface is fine.
 
 
  It found and selected:
 
 
  enP8p1s0 udp/192.168.123.18
  selected interfaces: enP8p1s0
 
 
  So this is not caused by a wrong interface, and it is not because the Unitree is in lock mode. The crash still happens before robot mode/lock matters.
 

  The cyclonedds==0.10.5 warning is probably relevant. unitree-sdk2py explicitly wants:


  cyclonedds==0.10.2


  So I would revert:


  pip install --force-reinstall cyclonedds==0.10.2


  Then check:


  python -c "import importlib.metadata as m; print(m.version('cyclonedds'))"
  ldd ~/miniconda3/envs/env_deploy/lib/python3.10/site-packages/cyclonedds/_clayer*.so | grep libddsc


  The bad part is still this if it appears:


  /opt/ros/humble/lib/aarch64-linux-gnu/libddsc.so.0


  That means Unitree’s Python binding is using ROS Humble’s CycloneDDS C library. That can cause this DDS_RETCODE_BAD_PARAMETER at topic creation.


  One quick isolation test: try launching with fake observations:


  ros2 launch fetch policy_odom.launch.py fake_observations_mode:=true


  If that starts, it confirms the rest of the node/model is OK and only Unitree DDS init is failing.


  Also, your launch is still using build/fetch/fetch/policy_node.py, so source-only edits won’t show until you rebuild or run from source. I won’t run
  build, per your instruction.
 
 
› what else should i do
 
 
• I’ll check whether another ROS 2 interface/package is already defining rt/lowcmd or LowCmd in a way that conflicts with the Unitree SDK topic. This is
  read-only and should tell us the next concrete step.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• The next distinction is important: if a standalone Unitree DDS publisher can create rt/lowcmd, the failure is caused by mixing ROS 2/rclpy and Unitree
  SDK DDS in the same process. If standalo

  tree@unitree-jetson-payload:~/fetch_ws$
 
 
• Your version is back to 0.10.2, which matches Unitree’s Python package. I also reproduced the exact failure with a minimal script: Unitree DDS alone
  works, but rclpy.create_node(...) first and then ChannelPublisher('rt/lowcmd', LowCmd_) fails with the same DDS_RETCODE_BAD_PARAMETER.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• I tested the reverse order too. If Unitree DDS initializes first, ROS 2 node creation fails instead. So this is an in-process CycloneDDS conflict
  between rclpy/rmw_cyclonedds_cpp and unitree_sdk2py.core.channel, not robot lock mode and not the NIC.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• Do one of these next. The clean fix is option 1.
 
 
  Option 1: make policy_node use ROS 2 Unitree topics, not unitree_sdk2py DDS directly.
 
 
  Your process cannot safely mix:
 
 
  rclpy / rmw_cyclonedds_cpp
  unitree_sdk2py.core.channel / cyclonedds Python
 
 
  I reproduced it with a minimal test:
 
 
  Unitree DDS alone: works
  rclpy node + Unitree DDS in same process: same BAD_PARAMETER crash
  Unitree DDS first + rclpy node after: ROS node creation fails
 
 
  So the next code change should be: remove direct ChannelPublisher/ChannelSubscriber use from policy_node, and publish/subscribe using ROS 2 messages
  instead:
 
 
  from unitree_go.msg import LowCmd, LowState, SportModeState
 
 
  Use ROS topics like:
 
 
  /lowcmd
  /lowstate or /lf/lowstate
  /sportmodestate
 
 
  This matches the Unitree ROS 2 examples and avoids the DDS conflict.
 
 
  Option 2: split Unitree DDS into a separate non-ROS process.
 
 
  Keep policy_node as ROS 2 only, and have a separate Python process using unitree_sdk2py only. Communicate between them with something non-DDS, like
  UDP/local socket. More moving parts, but it avoids mixing the two DDS APIs in one process.
 
 
  Do not focus on lock mode. Lock mode can block control later, but your crash happens while creating the DDS topic object, before robot mode matters.
 
 
  Also, your cyclonedds==0.10.2 is correct for unitree-sdk2py. The ldd line pointing to /opt/ros/humble/.../libddsc.so.0 is not the main issue anymore;
  the minimal tests show the real issue is mixing ROS 2 CycloneDDS and Unitree Python CycloneDDS in the same process.
 
 
› implement nuber 1
 
 
• I’ll convert policy_node to 


mplement nuber 1
 
 
• I’ll convert policy_node to use ROS 2 unitree_go publishers/subscribers for low-level I/O instead of unitree_sdk2py channels, keeping the policy/
  control logic intact. I’ll only edit source files, not build.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• The implementation needs a ROS-message CRC path because the existing Unitree CRC helper only accepts CycloneDDS IDL objects. I’m adding a small ROS
  LowCmd CRC helper using the same packing as Unitree’s C++ example, then wiring policy_node to ROS publishers/subscribers.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• There’s one behavior change to call out: with direct Unitree SDK removed from this ROS node, auto_switch_to_low_level can’t use MotionSwitcherClient
  inside this process anymore. I’ll make it log a clear warning and continue; you’ll need to put the robot in low-level mode using a separate Unitree
  tool/process if required.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• The code now uses ROS 2 messages for Unitree I/O. I’m checking the configured topic names against the rest of this workspace, because rt/lowstate may
  not be the ROS topic that your odometry is actually receiving.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• The odometry node in this launch already listens on /lf/lowstate, and the Unitree ROS 2 examples use /lowcmd for Go2 commands. I’m updating the source
  config defaults to those ROS topics so the converted node subscribes where data is already flowing.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• The source edits are in place. I’m doing syntax checks and a targeted search for any remaining direct Unitree SDK channel use in the source policy
  node.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• Implemented option 1 in source only. I did not run a build.
 
 
  Changed:
 
 
  - src/go2_fetch_ros2/fetch/fetch/policy_node.py:69 no longer imports/uses unitree_sdk2py DDS channels.
  - It now uses ROS 2 unitree_go.msg.LowCmd, LowState, and SportModeState.
  - send_cmd() now publishes ROS LowCmd and computes CRC locally.
  - src/go2_fetch_ros2/fetch/config/fetch_params.yaml:124 now defaults to:
      - lowcmd_topic: /lowcmd
      - lowstate_topic: /lf/lowstate
      - sportstate_topic: /sportmodestate
 
 
  - src/go2_fetch_ros2/fetch/fetch/deploy_real_utils.py:128 now has compute_go2_lowcmd_crc() for ROS messages.
 
 
  Important behavior change:
  auto_switch_to_low_level no longer uses MotionSwitcherClient inside policy_node, because that would reintroduce the same DDS conflict. The node now
  logs a warning; switch the robot to low-level mode with a separate Unitree tool/process if needed.
 
 
  Verification:
 
 
  python3 -m py_compile src/go2_fetch_ros2/fetch/fetch/policy_node.py src/go2_fetch_ros2/fetch/fetch/deploy_real_utils.py
 
 
  passed.
 
 
  Since you told me not to build, your current ros2 launch fetch policy_odom.launch.py will still use the old installed/build copy until you rebuild or
  otherwise run from source.
 
 
› but is it ok to change beacuse i used this repo https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2 and maybe its not the same functionalyt, also
  explein the problemsimplena
 
 
• Searched the web for TheoBounac Deploy_SimToReal_RL_Go

 / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / 
  otherwise run from source.
 
 
› but is it ok to change beacuse i used this repo https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2 and maybe its not the same functionalyt, also
  explein the problemsimplena
 
 
• Searched the web for TheoBounac Deploy_SimToReal_RL_Go2 GitHub
 
• Searched the web for https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• Yes, it is OK to change it, because your current app is not exactly the original repo anymore.
 
 
  The original repo says deployment is via the Unitree SDK and also uses ROS 2/Kalman filtering for state feedback. That matches the design, but your ROS
  node put both of these inside one Python process:
 
 
  1. ROS 2 node: rclpy
  2. Unitree SDK DDS: unitree_sdk2py.ChannelPublisher
 
 
  That is the problem.
 
 
  Simple explanation:
 
 
  rclpy already starts CycloneDDS for ROS 2.
  Then unitree_sdk2py tries to start/use CycloneDDS again inside the same Python process.
  They conflict.
  So the program crashes before it even talks to the robot.
 
 
  That means this is not about the RL model, and not about the robot lock mode. The crash happens before control starts.
 
 
  What I changed keeps the same high-level behavior:
 
 
  - policy still loads the same .pt models
  - observations still come from lowstate/odom/cube state
  - commands are still LowCmd
  - CRC is still computed
  - robot command output still respects send_commands
 
 
  But the transport changed:
 
 
  - Before: direct Unitree SDK DDS inside policy_node
  - Now: ROS 2 unitree_go/msg/LowCmd and LowState
 
 
  This is safer for your setup because your launch is already ROS 2 based, and your odometry nodes already receive /lf/lowstate.
 
 
  The only functionality that is not the same is automatic mode switching with MotionSwitcherClient. I disabled that inside policy_node because using it
  would bring back the same DDS conflict. If needed, switch the robot to low-level mode with a separate Unitree script/process before launching the
  policy.
 
 
  Source: the repo README describes deployment using Unitree SDK and ROS 2/Kalman together, but your failure is specifically from mixing two CycloneDDS
  users in one Python process. See the repo overview here: https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2
 
 
› how
    The only functionality that is 

    tic mode switching with MotionSwitcherClient. I disabled that inside policy_node because using it
  would bring back the same DDS conflict. If needed, switch the robot to low-level mode with a separate Unitree script/process before launching the
  policy.
 
 
  Source: the repo README describes deployment using Unitree SDK and ROS 2/Kalman together, but your failure is specifically from mixing two CycloneDDS
  users in one Python process. See the repo overview here: https://github.com/TheoBounac/Deploy_SimToReal_RL_Go2
 
 
› how
    The only functionality that is not the same is automatic mode switching with MotionSwitcherClient. I disabled that inside policy_node because using
  it would bring back the
    same DDS conflict. If needed, switch the robot to low-level mode with a separate Unitree script/process before launching the policy.
 
 
• I’ll check the local Unitree SDK examples/client signatures so the command I give you matches what is installed on this machine.
 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
• You do it in a separate terminal before launching policy_node.
 
 
  The old code did this automatically:
 
 
  1. Connect to Unitree SDK.
  2. Ask robot what high-level mode is active.
  3. Call StandDown().
  4. Call ReleaseMode().
  5. Then the low-level controller can publish /lowcmd.
 
 
  Now that must be a separate step.
 
 
  Use this command:
 
 
  conda activate env_deploy
  cd ~/fetch_ws
 
 
  PYTHONPATH=~/fetch_ws/src/unitree_sdk2_python:$PYTHONPATH \
  python -c "
  import time
  from unitree_sdk2py.core.channel import ChannelFactoryInitialize
  from unitree_sdk2py.go2.sport.sport_client import SportClient
  from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
 
 
  ChannelFactoryInitialize(0, 'enP8p1s0')
 
 
  sc = SportClient()
  sc.SetTimeout(5.0)
  sc.Init()
 
 ...