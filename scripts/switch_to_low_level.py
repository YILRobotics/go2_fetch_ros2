#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""Release Unitree high-level mode before running the ROS policy node."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_unitree_sdk_path() -> None:
    workspace_sdk = Path(__file__).resolve().parents[3] / "unitree_sdk2_python"
    if workspace_sdk.is_dir():
        sys.path.insert(0, str(workspace_sdk))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Put the robot down and release Unitree high-level mode."
    )
    parser.add_argument("--interface", default="enP8p1s0", help="Robot network interface")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--timeout", type=float, default=5.0, help="RPC timeout in seconds")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between mode calls")
    parser.add_argument(
        "--lidar",
        choices=("off", "on", "unchanged"),
        default="off",
        help="UTLiDAR state requested before releasing high-level mode (default: off)",
    )
    args = parser.parse_args()

    _add_unitree_sdk_path()

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    ChannelFactoryInitialize(args.domain_id, args.interface)

    if args.lidar != "unchanged":
        lidar_switch = ChannelPublisher("rt/utlidar/switch", String_)
        lidar_switch.Init()
        lidar_command = std_msgs_msg_dds__String_()
        lidar_command.data = args.lidar.upper()
        # This command has no acknowledgement. Repeating it reduces the chance
        # that a newly discovered DDS reader misses the first sample.
        for _ in range(3):
            lidar_switch.Write(lidar_command)
            time.sleep(0.1)
        print(f"UTLiDAR switch command sent: {lidar_command.data}")

    sport = SportClient()
    sport.SetTimeout(args.timeout)
    sport.Init()

    motion = MotionSwitcherClient()
    motion.SetTimeout(args.timeout)
    motion.Init()

    status, mode = motion.CheckMode()
    print("current mode:", status, mode)

    while mode and mode.get("name"):
        print("standing down and releasing high-level mode:", mode)
        stand_down_ret = sport.StandDown()
        print("StandDown ret:", stand_down_ret)
        time.sleep(args.sleep)

        release_ret, _ = motion.ReleaseMode()
        print("ReleaseMode ret:", release_ret)
        time.sleep(args.sleep)

        status, mode = motion.CheckMode()
        print("current mode:", status, mode)

    print("high-level mode released; low-level control should be available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
