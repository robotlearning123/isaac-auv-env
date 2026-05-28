#!/usr/bin/env python3
"""Standalone entry point for the OceanScale ROS2 bridge.

Runs the bridge with a default BlueROV2 Heavy simulation, accepting
configuration from the command line.

Usage::

    python scripts/ros2_bridge.py
    python scripts/ros2_bridge.py --namespace /rov --rate 50 --vehicle bluerov2_heavy
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OceanScale ROS2 bridge — streams sim observations to ROS2 topics",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="/oceanscale",
        help="ROS2 namespace (default: /oceanscale)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Simulation step rate in Hz (default: 100.0)",
    )
    parser.add_argument(
        "--env-idx",
        type=int,
        default=0,
        dest="env_idx",
        help="Which parallel env to expose (default: 0)",
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        default=True,
        dest="use_sim_time",
        help="Publish sim time as ROS clock (default: true)",
    )
    parser.add_argument(
        "--no-sim-time",
        action="store_false",
        dest="use_sim_time",
        help="Disable publishing sim time as ROS clock",
    )
    parser.add_argument(
        "--vehicle",
        type=str,
        default="bluerov2_heavy",
        dest="vehicle_name",
        help="Vehicle name (default: bluerov2_heavy)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Lazy imports so --help works without rclpy
    from oceanscale.ros2.bridge import OceanROS2BridgeConfig, run_bridge
    from oceanscale.sim import OceanSimConfig
    from oceanscale.vehicles import BlueROV2Heavy

    vehicle_map = {"bluerov2_heavy": BlueROV2Heavy}
    vehicle_cls = vehicle_map.get(args.vehicle_name, BlueROV2Heavy)
    sim_config = OceanSimConfig(vehicle=vehicle_cls())
    config = OceanROS2BridgeConfig(
        sim_config=sim_config,
        namespace=args.namespace,
        rate_hz=args.rate,
        env_idx=args.env_idx,
        use_sim_time=args.use_sim_time,
        vehicle_name=args.vehicle_name,
    )

    print(
        f"Starting OceanScale ROS2 bridge: "
        f"ns={config.namespace}, rate={config.rate_hz} Hz, "
        f"env_idx={config.env_idx}, vehicle={config.vehicle_name}",
        file=sys.stderr,
    )

    run_bridge(config)


if __name__ == "__main__":
    main()
