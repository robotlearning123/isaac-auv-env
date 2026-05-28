"""ROS2 bridge publishers for OceanSim observation data.

Lazy-imports rclpy so the module can be imported without a ROS2 installation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import rclpy.node
    import tf2_ros

# ---------------------------------------------------------------------------
# Lazy rclpy / ROS2 message imports
# ---------------------------------------------------------------------------

_rclpy = None
_msgs_nav = None
_msgs_geo = None
_msgs_sensor = None
_msgs_rosgraph = None
_msgs_tf2 = None
_tf2_ros = None


def _ensure_rclpy():
    """Import rclpy and message packages on first use."""
    global _rclpy, _msgs_nav, _msgs_geo, _msgs_sensor, _msgs_rosgraph, _msgs_tf2, _tf2_ros
    if _rclpy is not None:
        return

    import rclpy
    import rclpy.clock
    import rclpy.node
    import tf2_ros
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import (
        PoseStamped,
        Quaternion,
        TransformStamped,
        Twist,
        TwistStamped,
        Vector3,
    )
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import FluidPressure, Imu, MagneticField, Range

    _rclpy = rclpy
    _tf2_ros = tf2_ros
    _msgs_nav = {"Odometry": Odometry}
    _msgs_geo = {
        "PoseStamped": PoseStamped,
        "Quaternion": Quaternion,
        "TransformStamped": TransformStamped,
        "Twist": Twist,
        "TwistStamped": TwistStamped,
        "Vector3": Vector3,
        "Time": Time,
    }
    _msgs_sensor = {"FluidPressure": FluidPressure, "Imu": Imu, "MagneticField": MagneticField, "Range": Range}
    _msgs_rosgraph = {"Clock": Clock}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_G = 9.80665  # m/s^2 standard gravity
_WATER_DENSITY = 1025.0  # kg/m^3 seawater
_ATM_PRESSURE = 101325.0  # Pa


def _depth_to_pressure(depth_m: float) -> float:
    """Convert depth (metres below surface, positive-down) to absolute pressure."""
    return _ATM_PRESSURE + _WATER_DENSITY * _G * depth_m


def _stamp_from_sim_time(node: rclpy.node.Node, sim_time: float):
    """Build a ROS2 Time message from a float simulation timestamp."""
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    msg = _msgs_geo["Time"]()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


# ---------------------------------------------------------------------------
# VehicleStatePublisher
# ---------------------------------------------------------------------------


class VehicleStatePublisher:
    """Publishes vehicle pose, twist, IMU, and pressure from an OceanSim obs dict.

    Topics (relative to *namespace*):
        ~/odom    — nav_msgs/Odometry
        ~/pose    — geometry_msgs/PoseStamped
        ~/twist   — geometry_msgs/TwistStamped
        ~/imu     — sensor_msgs/Imu
        ~/pressure — sensor_msgs/FluidPressure
    """

    def __init__(
        self,
        node: rclpy.node.Node,
        namespace: str = "/oceanscale/vehicle",
        frame_id: str = "base_link",
    ) -> None:
        _ensure_rclpy()
        self._node = node
        self._namespace = namespace.rstrip("/")
        self._frame_id = frame_id

        Odometry = _msgs_nav["Odometry"]
        PoseStamped = _msgs_geo["PoseStamped"]
        TwistStamped = _msgs_geo["TwistStamped"]
        FluidPressure = _msgs_sensor["FluidPressure"]
        Imu = _msgs_sensor["Imu"]

        self._pub_odom = node.create_publisher(Odometry, f"{self._namespace}/odom", 10)
        self._pub_pose = node.create_publisher(PoseStamped, f"{self._namespace}/pose", 10)
        self._pub_twist = node.create_publisher(TwistStamped, f"{self._namespace}/twist", 10)
        self._pub_imu = node.create_publisher(Imu, f"{self._namespace}/imu", 10)
        self._pub_pressure = node.create_publisher(FluidPressure, f"{self._namespace}/pressure", 10)

    # ---- helpers ----

    def _header(self, stamp):
        from std_msgs.msg import Header

        return Header(stamp=stamp, frame_id=self._frame_id)

    def _odom_frame(self) -> str:
        return f"{self._namespace}/odom"

    # ---- publish ----

    def publish(self, obs: dict[str, Any], env_idx: int = 0) -> None:
        """Publish all vehicle-state topics from *obs* for environment *env_idx*."""
        _ensure_rclpy()
        sim_time: float = float(obs["time"])
        stamp = _stamp_from_sim_time(self._node, sim_time)
        header = self._header(stamp)

        pos = np.asarray(obs["position"][env_idx], dtype=np.float64)  # (3,)
        quat = np.asarray(obs["orientation"][env_idx], dtype=np.float64)  # (x,y,z,w)
        lin_vel = np.asarray(obs["linear_velocity"][env_idx], dtype=np.float64)  # body
        ang_vel = np.asarray(obs["angular_velocity"][env_idx], dtype=np.float64)  # body

        # -- Odometry --
        odom = _msgs_nav["Odometry"]()
        odom.header = header
        odom.child_frame_id = self._frame_id
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.x = float(quat[0])
        odom.pose.pose.orientation.y = float(quat[1])
        odom.pose.pose.orientation.z = float(quat[2])
        odom.pose.pose.orientation.w = float(quat[3])
        odom.twist.twist.linear.x = float(lin_vel[0])
        odom.twist.twist.linear.y = float(lin_vel[1])
        odom.twist.twist.linear.z = float(lin_vel[2])
        odom.twist.twist.angular.x = float(ang_vel[0])
        odom.twist.twist.angular.y = float(ang_vel[1])
        odom.twist.twist.angular.z = float(ang_vel[2])
        self._pub_odom.publish(odom)

        # -- PoseStamped --
        pose_msg = _msgs_geo["PoseStamped"]()
        pose_msg.header = header
        pose_msg.pose = odom.pose.pose
        self._pub_pose.publish(pose_msg)

        # -- TwistStamped --
        twist_msg = _msgs_geo["TwistStamped"]()
        twist_msg.header = header
        twist_msg.twist = odom.twist.twist
        self._pub_twist.publish(twist_msg)

        # -- Imu --
        imu = _msgs_sensor["Imu"]()
        imu.header = header
        imu.orientation = odom.pose.pose.orientation
        imu.angular_velocity = odom.twist.twist.angular
        imu.linear_acceleration.x = 0.0
        imu.linear_acceleration.y = 0.0
        imu.linear_acceleration.z = _G  # gravity in body frame (simplified)
        self._pub_imu.publish(imu)

        # -- FluidPressure (depth-derived) --
        depth = -float(pos[2])  # positive-down depth
        pressure_msg = _msgs_sensor["FluidPressure"]()
        pressure_msg.header = header
        pressure_msg.fluid_pressure = _depth_to_pressure(max(depth, 0.0))
        pressure_msg.variance = 0.0
        self._pub_pressure.publish(pressure_msg)


# ---------------------------------------------------------------------------
# SensorPublisher
# ---------------------------------------------------------------------------


class SensorPublisher:
    """Publishes sensor data from an OceanSim sensors dict.

    Topics (relative to *namespace*):
        ~/dvl          — geometry_msgs/TwistStamped
        ~/sonar        — sensor_msgs/Range
        ~/magnetometer — sensor_msgs/MagneticField
    """

    def __init__(
        self,
        node: rclpy.node.Node,
        namespace: str = "/oceanscale/vehicle",
        frame_id: str = "base_link",
    ) -> None:
        _ensure_rclpy()
        self._node = node
        self._namespace = namespace.rstrip("/")
        self._frame_id = frame_id

        TwistStamped = _msgs_geo["TwistStamped"]
        MagneticField = _msgs_sensor["MagneticField"]
        Range = _msgs_sensor["Range"]

        self._pub_dvl = node.create_publisher(TwistStamped, f"{self._namespace}/dvl", 10)
        self._pub_sonar = node.create_publisher(Range, f"{self._namespace}/sonar", 10)
        self._pub_mag = node.create_publisher(MagneticField, f"{self._namespace}/magnetometer", 10)

    def _header(self, stamp):
        from std_msgs.msg import Header

        return Header(stamp=stamp, frame_id=self._frame_id)

    def publish(self, sensors: dict[str, Any], env_idx: int = 0, sim_time: float = 0.0) -> None:
        """Publish sensor topics from *sensors* dict for environment *env_idx*."""
        _ensure_rclpy()
        stamp = _stamp_from_sim_time(self._node, sim_time)
        header = self._header(stamp)

        # -- DVL --
        dvl = sensors.get("dvl")
        if dvl is not None:
            vel = np.asarray(dvl[env_idx] if hasattr(dvl, "__getitem__") else dvl, dtype=np.float64)
            msg = _msgs_geo["TwistStamped"]()
            msg.header = header
            msg.twist.linear.x = float(vel[0])
            msg.twist.linear.y = float(vel[1])
            msg.twist.linear.z = float(vel[2])
            self._pub_dvl.publish(msg)

        # -- Sonar --
        sonar = sensors.get("sonar")
        if sonar is not None:
            rng = np.asarray(
                sonar[env_idx] if hasattr(sonar, "__getitem__") else sonar, dtype=np.float64
            )
            msg = _msgs_sensor["Range"]()
            msg.header = header
            msg.radiation_type = 0  # ULTRASOUND
            msg.field_of_view = 0.3  # ~17 deg typical
            msg.min_range = 0.1
            msg.max_range = 100.0
            msg.range = float(rng.flat[0]) if rng.ndim > 0 else float(rng)
            self._pub_sonar.publish(msg)

        # -- Magnetometer --
        mag = sensors.get("magnetometer")
        if mag is not None:
            field = np.asarray(
                mag[env_idx] if hasattr(mag, "__getitem__") else mag, dtype=np.float64
            )
            msg = _msgs_sensor["MagneticField"]()
            msg.header = header
            msg.magnetic_field.x = float(field[0])
            msg.magnetic_field.y = float(field[1])
            msg.magnetic_field.z = float(field[2])
            self._pub_mag.publish(msg)


# ---------------------------------------------------------------------------
# ClockPublisher
# ---------------------------------------------------------------------------


class ClockPublisher:
    """Publishes rosgraph_msgs/Clock for simulation time synchronisation."""

    def __init__(self, node: rclpy.node.Node, topic: str = "/clock") -> None:
        _ensure_rclpy()
        self._node = node
        Clock = _msgs_rosgraph["Clock"]
        self._pub = node.create_publisher(Clock, topic, 10)

    def publish(self, sim_time: float) -> None:
        """Publish current simulation time as a Clock message."""
        _ensure_rclpy()
        msg = _msgs_rosgraph["Clock"]()
        msg.clock = _stamp_from_sim_time(self._node, sim_time)
        self._pub.publish(msg)


# ---------------------------------------------------------------------------
# TfPublisher
# ---------------------------------------------------------------------------


class TfPublisher:
    """Publishes the odom -> base_link TF2 transform."""

    def __init__(self, node: rclpy.node.Node, parent_frame: str = "odom") -> None:
        _ensure_rclpy()
        self._node = node
        self._parent_frame = parent_frame
        self._broadcaster: tf2_ros.TransformBroadcaster = _tf2_ros.TransformBroadcaster(node)

    def publish(
        self, obs: dict[str, Any], env_idx: int = 0, child_frame: str = "base_link"
    ) -> None:
        """Broadcast odom -> base_link transform from *obs*."""
        _ensure_rclpy()
        sim_time: float = float(obs["time"])
        stamp = _stamp_from_sim_time(self._node, sim_time)

        pos = np.asarray(obs["position"][env_idx], dtype=np.float64)
        quat = np.asarray(obs["orientation"][env_idx], dtype=np.float64)

        t = _msgs_geo["TransformStamped"]()
        t.header.stamp = stamp
        t.header.frame_id = self._parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self._broadcaster.sendTransform(t)


__all__ = [
    "ClockPublisher",
    "SensorPublisher",
    "TfPublisher",
    "VehicleStatePublisher",
]
