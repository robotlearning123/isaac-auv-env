"""OceanScale ROS2 bridge — optional dependency.

Provides message conversion and bridge node for streaming OceanSim
observations into the ROS2 ecosystem.

Requires: ``pip install oceanscale[ros2]`` (or ``rclpy`` on PYTHONPATH).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "OceanROS2Bridge",
    "OceanROS2BridgeConfig",
    "imu_from_obs",
    "odometry_from_obs",
    "pose_from_obs",
    "pressure_from_obs",
    "run_bridge",
    "twist_from_obs",
    "wrench_from_action",
]


def _check_rclpy() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "rclpy is required for oceanscale.ros2. Install with: pip install oceanscale[ros2]"
        ) from None


def __getattr__(name: str) -> Any:
    _lazy = {
        "OceanROS2Bridge": "oceanscale.ros2.bridge",
        "OceanROS2BridgeConfig": "oceanscale.ros2.bridge",
        "run_bridge": "oceanscale.ros2.bridge",
        "pose_from_obs": "oceanscale.ros2.messages",
        "twist_from_obs": "oceanscale.ros2.messages",
        "odometry_from_obs": "oceanscale.ros2.messages",
        "imu_from_obs": "oceanscale.ros2.messages",
        "pressure_from_obs": "oceanscale.ros2.messages",
        "wrench_from_action": "oceanscale.ros2.messages",
    }
    if name in _lazy:
        _check_rclpy()
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'oceanscale.ros2' has no attribute {name!r}")
