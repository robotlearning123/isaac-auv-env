"""ROS2 bridge node for OceanScale GPU underwater simulation.

Exposes OceanSim as a ROS2 node, publishing vehicle state, sensor data,
sim clock, and TF transforms while subscribing to twist commands and
emergency stop signals.

All ROS2 imports are lazy — the module imports cleanly without rclpy
installed and only fails at runtime when you actually create a bridge.

Usage::

    from oceanscale.ros2.bridge import run_bridge, OceanROS2BridgeConfig
    from oceanscale.vehicles import BlueROV2Heavy
    from oceanscale.sim import OceanSimConfig

    run_bridge(OceanROS2BridgeConfig(
        sim_config=OceanSimConfig(vehicle=BlueROV2Heavy()),
    ))
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Any


def _check_rclpy() -> None:
    """Raise ImportError with actionable message if rclpy is missing."""
    try:
        import rclpy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "rclpy is required for the ROS2 bridge. "
            "Install with: pip install rclpy   (or via your ROS2 workspace)"
        ) from e


@dataclass
class OceanROS2BridgeConfig:
    """Configuration for the OceanScale ROS2 bridge node."""

    sim_config: Any  # OceanSimConfig — typed as Any to avoid import at module level
    namespace: str = "/oceanscale"
    rate_hz: float = 100.0
    env_idx: int = 0
    use_sim_time: bool = True
    vehicle_name: str = "bluerov2_heavy"
    init_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)


class OceanROS2Bridge:
    """ROS2 node that bridges OceanSim to the ROS2 ecosystem.

    Publishes:
        - Vehicle state (pose, twist) on ``<ns>/state``
        - Sensor data on ``<ns>/sensors``
        - Sim clock on ``/clock``
        - TF transforms (odom -> base_link)

    Subscribes:
        - Twist commands on ``<ns>/cmd_vel``
        - Emergency stop on ``<ns>/e_stop``

    Args:
        config: Bridge configuration including sim config, namespace, rate.
    """

    def __init__(self, config: OceanROS2BridgeConfig) -> None:
        _check_rclpy()

        import rclpy
        from rclpy.node import Node

        from oceanscale.ros2.publishers import (
            ClockPublisher,
            SensorPublisher,
            TfPublisher,
            VehicleStatePublisher,
        )
        from oceanscale.ros2.subscribers import (
            EmergencyStopSubscriber,
            TwistCommandSubscriber,
        )
        from oceanscale.sim import OceanSim

        # Initialize rclpy if not already done
        if not rclpy.ok():
            rclpy.init()

        self._config = config
        self._env_idx = config.env_idx
        self._rate_hz = config.rate_hz
        self._ns = config.namespace.rstrip("/")

        # Create sim
        self._sim = OceanSim(config.sim_config)
        self._obs: dict[str, Any] = {}
        self._e_stopped = False

        # ROS2 node — use a plain name, not a subclass
        self._node = Node("oceanscale_bridge", namespace=self._ns.lstrip("/"))
        if config.use_sim_time:
            self._node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])

        # Publishers
        self._state_pub = VehicleStatePublisher(self._node, self._ns)
        self._sensor_pub = SensorPublisher(self._node, self._ns)
        self._clock_pub = ClockPublisher(self._node)
        self._tf_pub = TfPublisher(self._node)

        # Subscribers (absolute topics matching publisher convention)
        self._cmd_sub = TwistCommandSubscriber(self._node, topic=f"{self._ns}/cmd_vel")
        self._estop_sub = EmergencyStopSubscriber(self._node, topic=f"{self._ns}/emergency_stop")

        # Timer for main loop
        period = 1.0 / config.rate_hz
        self._timer = self._node.create_timer(period, self._step_callback)

        # Graceful shutdown
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._node.get_logger().info(
            f"OceanScale bridge started: ns={self._ns}, "
            f"rate={config.rate_hz} Hz, env_idx={config.env_idx}"
        )

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self._node.get_logger().info(f"Received signal {signum}, shutting down...")
        self._shutdown = True

    def _step_callback(self) -> None:
        """Main simulation loop — called at rate_hz by ROS2 timer."""
        if self._shutdown or self._e_stopped:
            return

        # Get command from subscriber
        action = self._cmd_sub.get_action()  # np.ndarray (6,) in [-1, 1]

        # Check emergency stop
        if self._estop_sub.is_stopped():
            self._e_stopped = True
            self._node.get_logger().warn("Emergency stop activated!")
            return

        # Step simulation
        self._obs = self._sim.step(action)

        # Publish all data
        idx = self._env_idx
        self._state_pub.publish(self._obs, env_idx=idx)
        sensors = self._obs.get("sensors", {})
        self._sensor_pub.publish(sensors, env_idx=idx, sim_time=self._sim.time)
        self._clock_pub.publish(self._sim.time)
        self._tf_pub.publish(self._obs, env_idx=idx)

    def reset(self) -> dict[str, Any]:
        """Reset the simulation to initial conditions.

        Returns:
            Initial observation dict from OceanSim.reset().
        """
        self._obs = self._sim.reset()
        self._e_stopped = False
        self._node.get_logger().info("Simulation reset")
        return self._obs

    def spin(self) -> None:
        """Spin the node, processing callbacks until shutdown."""
        import rclpy

        try:
            while rclpy.ok() and not self._shutdown:
                rclpy.spin_once(self._node, timeout_sec=0.0)
        finally:
            self.destroy()

    def destroy(self) -> None:
        """Clean up ROS2 resources and close the simulation."""
        self._node.get_logger().info("Destroying OceanScale bridge")
        self._sim.close()
        self._node.destroy_node()


def run_bridge(
    config: OceanROS2BridgeConfig | None = None,
    *,
    sim_config: Any | None = None,
    namespace: str = "/oceanscale",
    rate_hz: float = 100.0,
    env_idx: int = 0,
    use_sim_time: bool = True,
) -> None:
    """Create and spin an OceanROS2Bridge node.

    Accepts either a pre-built config or individual kwargs. When called
    with no arguments, creates a default BlueROV2 Heavy bridge.

    Args:
        config: Full bridge config (takes precedence over kwargs).
        sim_config: OceanSimConfig instance (used if config is None).
        namespace: ROS2 namespace.
        rate_hz: Simulation step rate.
        env_idx: Which parallel env to expose on ROS2.
        use_sim_time: Publish sim time as ROS clock.
    """
    _check_rclpy()

    import rclpy

    if config is None:
        if sim_config is None:
            from oceanscale.sim import OceanSimConfig
            from oceanscale.vehicles import BlueROV2Heavy

            sim_config = OceanSimConfig(vehicle=BlueROV2Heavy())
        config = OceanROS2BridgeConfig(
            sim_config=sim_config,
            namespace=namespace,
            rate_hz=rate_hz,
            env_idx=env_idx,
            use_sim_time=use_sim_time,
        )

    bridge = OceanROS2Bridge(config)
    try:
        bridge.spin()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy()
        if rclpy.ok():
            rclpy.shutdown()
