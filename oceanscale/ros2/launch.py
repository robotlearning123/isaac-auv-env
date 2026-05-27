"""ROS2 launch utilities for the OceanScale bridge node.

Provides both a standard ``generate_launch_description()`` for use with
``ros2 launch`` and a convenience :func:`launch_bridge` for programmatic
use from Python.

All ROS2 / launch imports are deferred so the module is importable without
a ROS2 installation.

Usage::

    # From the command line
    ros2 launch oceanscale_ros2 bridge_launch.py namespace:=/rov rate_hz:=50

    # Programmatically
    from oceanscale.ros2.launch import launch_bridge
    launch_bridge(namespace="/rov", rate_hz=50.0)
"""

from __future__ import annotations

from typing import Any


def generate_launch_description() -> Any:
    """Return a ROS2 LaunchDescription for the OceanScale bridge node.

    Launch arguments:
        namespace     — ROS2 namespace (default: ``/oceanscale``)
        rate_hz       — Simulation step rate (default: ``100.0``)
        env_idx       — Which parallel env to expose (default: ``0``)
        use_sim_time  — Publish sim time as ROS clock (default: ``true``)
        vehicle_name  — Vehicle identifier (default: ``bluerov2_heavy``)
    """
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node

    namespace = LaunchConfiguration("namespace")
    rate_hz = LaunchConfiguration("rate_hz")
    env_idx = LaunchConfiguration("env_idx")
    use_sim_time = LaunchConfiguration("use_sim_time")
    vehicle_name = LaunchConfiguration("vehicle_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="/oceanscale",
                description="ROS2 namespace for the bridge node",
            ),
            DeclareLaunchArgument(
                "rate_hz",
                default_value="100.0",
                description="Simulation step rate in Hz",
            ),
            DeclareLaunchArgument(
                "env_idx",
                default_value="0",
                description="Which parallel environment to expose on ROS2",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Publish simulation time as the ROS clock",
            ),
            DeclareLaunchArgument(
                "vehicle_name",
                default_value="bluerov2_heavy",
                description="Vehicle identifier",
            ),
            Node(
                package="oceanscale",
                executable="ros2_bridge",
                name="oceanscale_bridge",
                namespace=namespace,
                parameters=[
                    {"rate_hz": rate_hz},
                    {"env_idx": env_idx},
                    {"use_sim_time": use_sim_time},
                    {"vehicle_name": vehicle_name},
                ],
                output="screen",
            ),
        ]
    )


def launch_bridge(**kwargs: Any) -> None:
    """Launch the OceanScale bridge node programmatically.

    Creates a launch description with the given kwargs and runs it in the
    current process. Accepts the same keyword arguments as
    :class:`~oceanscale.ros2.bridge.OceanROS2BridgeConfig`.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`~oceanscale.ros2.bridge.run_bridge`.
        Common keys: ``namespace``, ``rate_hz``, ``env_idx``,
        ``use_sim_time``, ``vehicle_name``, ``sim_config``.
    """
    from oceanscale.ros2.bridge import run_bridge

    run_bridge(**kwargs)
