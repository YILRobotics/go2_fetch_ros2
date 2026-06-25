#!/home/unitree/miniconda3/envs/env_deploy/bin/python
"""Restore Unitree high-level mode after stopping the ROS policy node."""

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
        description="Select a Unitree high-level mode, optionally standing up afterward."
    )
    parser.add_argument("--interface", default="enP8p1s0", help="Robot network interface")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--mode", default="ai", help="High-level mode to select")
    parser.add_argument("--timeout", type=float, default=5.0, help="RPC timeout in seconds")
    parser.add_argument(
        "--stand-up",
        action="store_true",
        help="Call SportClient.RecoveryStand after selecting the mode",
    )
    parser.add_argument(
        "--force-stand-up",
        action="store_true",
        help="Try RecoveryStand even if SelectMode fails",
    )
    parser.add_argument(
        "--stand-up-delay",
        type=float,
        default=1.0,
        help="Delay before optional RecoveryStand call",
    )
    args = parser.parse_args()

    _add_unitree_sdk_path()

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(args.domain_id, args.interface)

    motion = MotionSwitcherClient()
    motion.SetTimeout(args.timeout)
    motion.Init()

    status, mode = motion.CheckMode()
    print("before:", status, mode)

    ret, _ = motion.SelectMode(args.mode)
    print(f"SelectMode {args.mode} ret:", ret)

    status, mode = motion.CheckMode()
    print("after:", status, mode)

    if ret != 0:
        print(
            "mode selection failed. The robot may not support this mode in its "
            "current state/firmware."
        )

    if args.stand_up:
        if ret != 0 and not args.force_stand_up:
            print(
                "skipping RecoveryStand because SelectMode failed. "
                "Use --force-stand-up to try it anyway."
            )
            return ret

        from unitree_sdk2py.go2.sport.sport_client import SportClient

        time.sleep(args.stand_up_delay)
        sport = SportClient()
        sport.SetTimeout(args.timeout)
        sport.Init()
        print("RecoveryStand ret:", sport.RecoveryStand())

    return 0 if ret == 0 else ret


if __name__ == "__main__":
    raise SystemExit(main())
