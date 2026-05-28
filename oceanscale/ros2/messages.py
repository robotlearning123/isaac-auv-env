"""ROS2 message conversion helpers for OceanSim observations.

All functions accept numpy arrays (from OceanSim) and return ROS2 message
objects. Quaternion convention: (x, y, z, w) in both OceanSim and ROS2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geometry_msgs.msg import PoseStamped, TwistStamped, Wrench
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import FluidPressure, Imu

# Physical constants for hydrostatic pressure.
_RHO_WATER = 1025.0  # kg/m^3 (seawater)
_G = 9.81  # m/s^2
_P0 = 101325.0  # Pa (atmospheric at surface)


def _stamp() -> object:
    """Return a zero-valued builtin_interfaces/Time stamp."""
    from builtin_interfaces.msg import Time

    return Time()


def pose_from_obs(obs: dict, frame_id: str = "odom") -> PoseStamped:
    """Convert OceanSim observation to geometry_msgs/PoseStamped.

    Parameters
    ----------
    obs : dict
        Must contain ``"position"`` (3,) and ``"orientation"`` (4,) as
        numpy arrays. Orientation is (x, y, z, w).
    frame_id : str
        Header frame ID.

    Returns
    -------
    PoseStamped
    """
    from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion

    pos = np.asarray(obs["position"], dtype=np.float64).ravel()[:3]
    quat = np.asarray(obs["orientation"], dtype=np.float64).ravel()[:4]

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = _stamp()
    msg.pose = Pose(
        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
        ),
    )
    return msg


def twist_from_obs(obs: dict, frame_id: str = "base_link") -> TwistStamped:
    """Convert OceanSim observation to geometry_msgs/TwistStamped.

    Parameters
    ----------
    obs : dict
        Must contain ``"linear_velocity"`` (3,) and ``"angular_velocity"``
        (4,) as numpy arrays.
    frame_id : str
        Header frame ID.

    Returns
    -------
    TwistStamped
    """
    from geometry_msgs.msg import (
        Twist,
        TwistStamped,
        Vector3,
    )

    lin = np.asarray(obs["linear_velocity"], dtype=np.float64).ravel()[:3]
    ang = np.asarray(obs["angular_velocity"], dtype=np.float64).ravel()[:3]

    msg = TwistStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = _stamp()
    msg.twist = Twist(
        linear=Vector3(x=float(lin[0]), y=float(lin[1]), z=float(lin[2])),
        angular=Vector3(x=float(ang[0]), y=float(ang[1]), z=float(ang[2])),
    )
    return msg


def odometry_from_obs(
    obs: dict,
    frame_id: str = "odom",
    child_frame_id: str = "base_link",
) -> Odometry:
    """Convert OceanSim observation to nav_msgs/Odometry.

    Parameters
    ----------
    obs : dict
        Must contain ``position``, ``orientation``, ``linear_velocity``,
        ``angular_velocity``.
    frame_id : str
        Odometry frame ID.
    child_frame_id : str
        Body frame ID.

    Returns
    -------
    Odometry
    """
    from geometry_msgs.msg import (
        Point,
        Pose,
        PoseWithCovariance,
        Quaternion,
        Twist,
        TwistWithCovariance,
        Vector3,
    )
    from nav_msgs.msg import Odometry

    pos = np.asarray(obs["position"], dtype=np.float64).ravel()[:3]
    quat = np.asarray(obs["orientation"], dtype=np.float64).ravel()[:4]
    lin = np.asarray(obs["linear_velocity"], dtype=np.float64).ravel()[:3]
    ang = np.asarray(obs["angular_velocity"], dtype=np.float64).ravel()[:3]

    msg = Odometry()
    msg.header.frame_id = frame_id
    msg.header.stamp = _stamp()
    msg.child_frame_id = child_frame_id

    msg.pose = PoseWithCovariance(
        pose=Pose(
            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            orientation=Quaternion(
                x=float(quat[0]),
                y=float(quat[1]),
                z=float(quat[2]),
                w=float(quat[3]),
            ),
        )
    )

    msg.twist = TwistWithCovariance(
        twist=Twist(
            linear=Vector3(x=float(lin[0]), y=float(lin[1]), z=float(lin[2])),
            angular=Vector3(x=float(ang[0]), y=float(ang[1]), z=float(ang[2])),
        )
    )

    return msg


def imu_from_obs(obs: dict, frame_id: str = "imu_link") -> Imu:
    """Convert OceanSim observation to sensor_msgs/Imu.

    Populates orientation, angular velocity, and linear acceleration
    (set to zero if not present in the observation).

    Parameters
    ----------
    obs : dict
        Must contain ``orientation``, ``angular_velocity``.  May contain
        ``linear_acceleration``.
    frame_id : str
        IMU frame ID.

    Returns
    -------
    Imu
    """
    from geometry_msgs.msg import Quaternion, Vector3
    from sensor_msgs.msg import Imu

    quat = np.asarray(obs["orientation"], dtype=np.float64).ravel()[:4]
    ang = np.asarray(obs["angular_velocity"], dtype=np.float64).ravel()[:3]

    acc = np.zeros(3, dtype=np.float64)
    if "linear_acceleration" in obs:
        acc = np.asarray(obs["linear_acceleration"], dtype=np.float64).ravel()[:3]

    msg = Imu()
    msg.header.frame_id = frame_id
    msg.header.stamp = _stamp()

    msg.orientation = Quaternion(
        x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
    )
    msg.angular_velocity = Vector3(x=float(ang[0]), y=float(ang[1]), z=float(ang[2]))
    msg.linear_acceleration = Vector3(x=float(acc[0]), y=float(acc[1]), z=float(acc[2]))

    return msg


def pressure_from_obs(
    obs: dict,
    frame_id: str = "pressure_link",
) -> FluidPressure:
    """Compute hydrostatic pressure from depth in the observation.

    Uses P = P0 + rho * g * h where h = -z (NED: depth is negative z).

    Parameters
    ----------
    obs : dict
        Must contain ``"position"`` (3,) in NED frame (z < 0 underwater).
    frame_id : str
        Pressure sensor frame ID.

    Returns
    -------
    FluidPressure
    """
    from sensor_msgs.msg import FluidPressure

    pos = np.asarray(obs["position"], dtype=np.float64).ravel()[:3]
    depth = -float(pos[2])  # NED: z negative underwater

    pressure_pa = _P0 + _RHO_WATER * _G * depth

    msg = FluidPressure()
    msg.header.frame_id = frame_id
    msg.header.stamp = _stamp()
    msg.fluid_pressure = float(pressure_pa)
    msg.variance = 0.0

    return msg


def wrench_from_action(
    action: np.ndarray,
    frame_id: str = "base_link",
) -> Wrench:
    """Convert a 6-dim action vector to geometry_msgs/Wrench.

    Action order: [surge, sway, heave, roll, pitch, yaw] in [-1, 1].

    Parameters
    ----------
    action : np.ndarray
        Shape (6,) normalized wrench command.
    frame_id : str
        Unused but kept for API symmetry / future TF lookups.

    Returns
    -------
    Wrench
    """
    from geometry_msgs.msg import Vector3, Wrench

    a = np.asarray(action, dtype=np.float64).ravel()[:6]

    return Wrench(
        force=Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2])),
        torque=Vector3(x=float(a[3]), y=float(a[4]), z=float(a[5])),
    )
