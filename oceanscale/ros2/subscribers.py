"""ROS2 bridge subscribers for feeding external commands into OceanSim.

Provides subscriber classes that accept ROS2 messages and store them as numpy
arrays for consumption by the simulation loop. All classes are thread-safe.

Gracefully handles missing rclpy — imports succeed but construction raises
if rclpy is not available.

Usage:
    node = rclpy.create_node("ocean_bridge")
    cmd_sub = TwistCommandSubscriber(node)
    rclpy.spin(node)  # or spin in background thread
"""

from __future__ import annotations

import importlib.util
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from rclpy.node import Node

# rclpy is optional — allow import without it installed.
_HAS_RCLPY = importlib.util.find_spec("rclpy") is not None

ZEROS_6 = np.zeros(6, dtype=np.float32)


def _require_rclpy() -> None:
    if not _HAS_RCLPY:
        raise RuntimeError(
            "rclpy is not installed. Install it with: pip install rclpy  "
            "(or: sudo apt install ros-${ROS_DISTRO}-rclpy)"
        )


# ---------------------------------------------------------------------------
# Protocols for type-checking without hard dependency on ROS2 message types
# ---------------------------------------------------------------------------


@runtime_checkable
class _TwistLinear(Protocol):
    x: float
    y: float
    z: float


@runtime_checkable
class _TwistAngular(Protocol):
    x: float
    y: float
    z: float


@runtime_checkable
class _Twist(Protocol):
    linear: _TwistLinear
    angular: _TwistAngular


@runtime_checkable
class _TwistStamped(Protocol):
    twist: _Twist


@runtime_checkable
class _Float32MultiArray(Protocol):
    data: tuple[float, ...]


@runtime_checkable
class _Point(Protocol):
    x: float
    y: float
    z: float


@runtime_checkable
class _Quaternion(Protocol):
    x: float
    y: float
    z: float
    w: float


@runtime_checkable
class _Pose(Protocol):
    position: _Point
    orientation: _Quaternion


@runtime_checkable
class _PoseStamped(Protocol):
    pose: _Pose


@runtime_checkable
class _BoolMsg(Protocol):
    data: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_array(
    arr: NDArray[np.float32], lo: float = -1.0, hi: float = 1.0
) -> NDArray[np.float32]:
    """Clamp array values in-place and return."""
    np.clip(arr, lo, hi, out=arr)
    return arr


class _Timestamped:
    """Mixin that stores a message receipt timestamp."""

    def __init__(self) -> None:
        self._stamp: float = 0.0

    @property
    def stamp(self) -> float:
        return self._stamp

    def _touch(self) -> None:
        self._stamp = time.monotonic()


# ---------------------------------------------------------------------------
# TwistCommandSubscriber
# ---------------------------------------------------------------------------


class TwistCommandSubscriber(_Timestamped):
    """Subscribe to geometry_msgs/Twist on ``~/cmd_vel``.

    Maps the standard 6-DOF twist into an action array compatible with
    OceanSim's action space: ``[surge, sway, heave, roll, pitch, yaw]``,
    each clamped to ``[-1, 1]``.

    Thread-safe: the ROS2 callback and :meth:`get_action` may run on
    different threads.
    """

    def __init__(self, node: Node, topic: str = "~/cmd_vel") -> None:
        _require_rclpy()
        super().__init__()
        self._lock = threading.Lock()
        self._action: NDArray[np.float32] = ZEROS_6.copy()

        from geometry_msgs.msg import Twist

        self._sub = node.create_subscription(Twist, topic, self._callback, 10)

    def _callback(self, msg: Any) -> None:
        action = np.array(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ],
            dtype=np.float32,
        )
        _clamp_array(action)
        with self._lock:
            self._action = action
            self._touch()

    def get_action(self) -> NDArray[np.float32]:
        """Return the latest action as ``(6,)`` float32 array."""
        with self._lock:
            return self._action.copy()

    def __repr__(self) -> str:
        return f"TwistCommandSubscriber(topic='{self._sub.topic}')"


# ---------------------------------------------------------------------------
# ThrusterCommandSubscriber
# ---------------------------------------------------------------------------


class ThrusterCommandSubscriber(_Timestamped):
    """Subscribe to std_msgs/Float32MultiArray on ``~/thruster_cmd``.

    Accepts raw thruster commands of arbitrary dimensionality
    ``(n_thrusters,)``. Values are **not** clamped — the caller is
    responsible for valid ranges.

    Thread-safe.
    """

    def __init__(self, node: Node, topic: str = "~/thruster_cmd") -> None:
        _require_rclpy()
        super().__init__()
        self._lock = threading.Lock()
        self._cmd: NDArray[np.float32] | None = None

        from std_msgs.msg import Float32MultiArray

        self._sub = node.create_subscription(Float32MultiArray, topic, self._callback, 10)

    def _callback(self, msg: Any) -> None:
        cmd = np.array(msg.data, dtype=np.float32)
        with self._lock:
            self._cmd = cmd
            self._touch()

    def get_thruster_cmd(self) -> NDArray[np.float32] | None:
        """Return latest thruster command or ``None`` if none received."""
        with self._lock:
            return self._cmd.copy() if self._cmd is not None else None

    def __repr__(self) -> str:
        return f"ThrusterCommandSubscriber(topic='{self._sub.topic}')"


# ---------------------------------------------------------------------------
# WaypointSubscriber
# ---------------------------------------------------------------------------


class WaypointSubscriber(_Timestamped):
    """Subscribe to geometry_msgs/PoseStamped on ``~/waypoint``.

    Stores the target position as ``(3,)`` and orientation as quaternion
    ``(4,)`` in ``(x, y, z, w)`` order.

    Thread-safe.
    """

    def __init__(self, node: Node, topic: str = "~/waypoint") -> None:
        _require_rclpy()
        super().__init__()
        self._lock = threading.Lock()
        self._position: NDArray[np.float32] | None = None
        self._orientation: NDArray[np.float32] | None = None

        from geometry_msgs.msg import PoseStamped

        self._sub = node.create_subscription(PoseStamped, topic, self._callback, 10)

    def _callback(self, msg: Any) -> None:
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        with self._lock:
            self._position = pos
            self._orientation = quat
            self._touch()

    def get_waypoint(self) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        """Return ``(position, quaternion)`` or ``None`` if no waypoint received."""
        with self._lock:
            if self._position is None:
                return None
            return self._position.copy(), self._orientation.copy()  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return f"WaypointSubscriber(topic='{self._sub.topic}')"


# ---------------------------------------------------------------------------
# EmergencyStopSubscriber
# ---------------------------------------------------------------------------


class EmergencyStopSubscriber(_Timestamped):
    """Subscribe to std_msgs/Bool on ``~/emergency_stop``.

    When the latest message is ``True``, :meth:`is_stopped` returns
    ``True`` and callers should emit a zero-action wrench.

    Thread-safe.
    """

    def __init__(self, node: Node, topic: str = "~/emergency_stop") -> None:
        _require_rclpy()
        super().__init__()
        self._lock = threading.Lock()
        self._stopped: bool = False

        from std_msgs.msg import Bool

        self._sub = node.create_subscription(Bool, topic, self._callback, 10)

    def _callback(self, msg: Any) -> None:
        with self._lock:
            self._stopped = bool(msg.data)
            self._touch()

    def is_stopped(self) -> bool:
        """Return ``True`` if an emergency stop is active."""
        with self._lock:
            return self._stopped

    def __repr__(self) -> str:
        return f"EmergencyStopSubscriber(topic='{self._sub.topic}')"
