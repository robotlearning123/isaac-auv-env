"""Tests for oceanscale.ros2 bridge — mock-based, no rclpy required.

Covers:
    - messages.py: pose_from_obs, twist_from_obs, odometry_from_obs,
      imu_from_obs, pressure_from_obs, wrench_from_action
    - subscribers.py: TwistCommandSubscriber, ThrusterCommandSubscriber,
      WaypointSubscriber, EmergencyStopSubscriber
    - __init__.py: lazy import gate
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock ROS2 message base class
# ---------------------------------------------------------------------------


class _Header:
    """Mock std_msgs/Header with stamp and frame_id."""

    def __init__(self) -> None:
        self.stamp = None
        self.frame_id = ""


class _Msg:
    """Generic mock ROS2 message.

    When called with no args (e.g. ``PoseStamped()``), creates an object
    with a ``header`` sub-object.  When called with kwargs (e.g.
    ``Point(x=1, y=2, z=3)``), stores them as attributes.
    """

    def __init__(self, **kwargs: object) -> None:
        self.header = _Header()
        for k, v in kwargs.items():
            setattr(self, k, v)


# Concrete mock classes — all share _Msg behaviour.
MockPoint = _Msg
MockQuaternion = _Msg
MockVector3 = _Msg
MockPose = _Msg
MockPoseStamped = _Msg
MockTwist = _Msg
MockTwistStamped = _Msg
MockPoseWithCovariance = _Msg
MockTwistWithCovariance = _Msg
MockOdometry = _Msg
MockImu = _Msg
MockFluidPressure = _Msg
MockWrench = _Msg
MockTime = _Msg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_ros2_msgs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject mock ROS2 message modules into sys.modules.

    This allows ``from geometry_msgs.msg import Point`` etc. to succeed
    without a real ROS2 installation.
    """
    # geometry_msgs.msg
    geom = MagicMock()
    for name, cls in [
        ("Point", MockPoint),
        ("Pose", MockPose),
        ("PoseStamped", MockPoseStamped),
        ("PoseWithCovariance", MockPoseWithCovariance),
        ("Quaternion", MockQuaternion),
        ("Twist", MockTwist),
        ("TwistStamped", MockTwistStamped),
        ("TwistWithCovariance", MockTwistWithCovariance),
        ("Vector3", MockVector3),
        ("Wrench", MockWrench),
    ]:
        setattr(geom, name, cls)

    # nav_msgs.msg
    nav = MagicMock()
    nav.Odometry = MockOdometry

    # sensor_msgs.msg
    sensor = MagicMock()
    sensor.FluidPressure = MockFluidPressure
    sensor.Imu = MockImu

    # builtin_interfaces.msg
    builtin = MagicMock()
    builtin.Time = MockTime

    # std_msgs.msg
    std = MagicMock()
    std.Float32MultiArray = _Msg
    std.Bool = _Msg

    monkeypatch.setitem(sys.modules, "geometry_msgs", MagicMock(msg=geom))
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geom)
    monkeypatch.setitem(sys.modules, "nav_msgs", MagicMock(msg=nav))
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav)
    monkeypatch.setitem(sys.modules, "sensor_msgs", MagicMock(msg=sensor))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor)
    monkeypatch.setitem(sys.modules, "builtin_interfaces", MagicMock(msg=builtin))
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", builtin)
    monkeypatch.setitem(sys.modules, "std_msgs", MagicMock(msg=std))
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std)


@pytest.fixture()
def _mock_rclpy(monkeypatch: pytest.MonkeyPatch, _mock_ros2_msgs: None) -> None:
    """Mock rclpy so subscriber constructors pass _require_rclpy().

    Strategy: let the module load naturally (rclpy not installed, so
    _HAS_RCLPY starts as False), then patch _HAS_RCLPY to True.
    The _mock_ros2_msgs fixture provides geometry_msgs etc. which the
    subscriber constructors need for their local ``from ... import``.
    """
    import oceanscale.ros2.subscribers as sub_mod

    monkeypatch.setattr(sub_mod, "_HAS_RCLPY", True)


@pytest.fixture()
def obs() -> dict:
    """Sample OceanSim observation dict with all fields."""
    return {
        "position": np.array([1.0, 2.0, -5.0]),
        "orientation": np.array([0.1, 0.2, 0.3, 0.9]),
        "linear_velocity": np.array([0.5, -0.3, 0.1]),
        "angular_velocity": np.array([0.01, -0.02, 0.03]),
        "linear_acceleration": np.array([0.1, 0.2, 9.8]),
    }


# ---------------------------------------------------------------------------
# messages.py — pure numpy -> mock message conversions
# ---------------------------------------------------------------------------


class TestPoseFromObs:
    """Tests for pose_from_obs."""

    def test_position_extraction(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pose_from_obs

        msg = pose_from_obs(obs)
        assert msg.pose.position.x == pytest.approx(1.0)
        assert msg.pose.position.y == pytest.approx(2.0)
        assert msg.pose.position.z == pytest.approx(-5.0)

    def test_orientation_extraction(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pose_from_obs

        msg = pose_from_obs(obs)
        assert msg.pose.orientation.x == pytest.approx(0.1)
        assert msg.pose.orientation.y == pytest.approx(0.2)
        assert msg.pose.orientation.z == pytest.approx(0.3)
        assert msg.pose.orientation.w == pytest.approx(0.9)

    def test_default_frame_id(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pose_from_obs

        msg = pose_from_obs(obs)
        assert msg.header.frame_id == "odom"

    def test_custom_frame_id(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pose_from_obs

        msg = pose_from_obs(obs, frame_id="map")
        assert msg.header.frame_id == "map"

    def test_numpy_float64_cast(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pose_from_obs

        obs_int = {
            "position": np.array([1, 2, 3]),
            "orientation": np.array([0, 0, 0, 1]),
        }
        msg = pose_from_obs(obs_int)
        assert isinstance(msg.pose.position.x, float)


class TestTwistFromObs:
    """Tests for twist_from_obs."""

    def test_linear_velocity(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import twist_from_obs

        msg = twist_from_obs(obs)
        assert msg.twist.linear.x == pytest.approx(0.5)
        assert msg.twist.linear.y == pytest.approx(-0.3)
        assert msg.twist.linear.z == pytest.approx(0.1)

    def test_angular_velocity(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import twist_from_obs

        msg = twist_from_obs(obs)
        assert msg.twist.angular.x == pytest.approx(0.01)
        assert msg.twist.angular.y == pytest.approx(-0.02)
        assert msg.twist.angular.z == pytest.approx(0.03)

    def test_default_frame_id(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import twist_from_obs

        msg = twist_from_obs(obs)
        assert msg.header.frame_id == "base_link"

    def test_custom_frame_id(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import twist_from_obs

        msg = twist_from_obs(obs, frame_id="odom")
        assert msg.header.frame_id == "odom"


class TestOdometryFromObs:
    """Tests for odometry_from_obs."""

    def test_position(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import odometry_from_obs

        msg = odometry_from_obs(obs)
        pos = msg.pose.pose.position
        assert pos.x == pytest.approx(1.0)
        assert pos.y == pytest.approx(2.0)
        assert pos.z == pytest.approx(-5.0)

    def test_orientation(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import odometry_from_obs

        msg = odometry_from_obs(obs)
        quat = msg.pose.pose.orientation
        assert quat.x == pytest.approx(0.1)
        assert quat.y == pytest.approx(0.2)
        assert quat.z == pytest.approx(0.3)
        assert quat.w == pytest.approx(0.9)

    def test_twist(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import odometry_from_obs

        msg = odometry_from_obs(obs)
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        assert lin.x == pytest.approx(0.5)
        assert ang.z == pytest.approx(0.03)

    def test_frame_ids(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import odometry_from_obs

        msg = odometry_from_obs(obs, frame_id="world", child_frame_id="body")
        assert msg.header.frame_id == "world"
        assert msg.child_frame_id == "body"

    def test_default_frame_ids(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import odometry_from_obs

        msg = odometry_from_obs(obs)
        assert msg.header.frame_id == "odom"
        assert msg.child_frame_id == "base_link"


class TestImuFromObs:
    """Tests for imu_from_obs."""

    def test_orientation(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import imu_from_obs

        msg = imu_from_obs(obs)
        assert msg.orientation.x == pytest.approx(0.1)
        assert msg.orientation.w == pytest.approx(0.9)

    def test_angular_velocity(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import imu_from_obs

        msg = imu_from_obs(obs)
        assert msg.angular_velocity.x == pytest.approx(0.01)
        assert msg.angular_velocity.z == pytest.approx(0.03)

    def test_with_linear_acceleration(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import imu_from_obs

        msg = imu_from_obs(obs)
        assert msg.linear_acceleration.x == pytest.approx(0.1)
        assert msg.linear_acceleration.y == pytest.approx(0.2)
        assert msg.linear_acceleration.z == pytest.approx(9.8)

    def test_without_linear_acceleration(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import imu_from_obs

        obs_no_acc = {
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
        }
        msg = imu_from_obs(obs_no_acc)
        assert msg.linear_acceleration.x == pytest.approx(0.0)
        assert msg.linear_acceleration.y == pytest.approx(0.0)
        assert msg.linear_acceleration.z == pytest.approx(0.0)

    def test_default_frame_id(self, obs: dict, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import imu_from_obs

        msg = imu_from_obs(obs)
        assert msg.header.frame_id == "imu_link"


class TestPressureFromObs:
    """Tests for pressure_from_obs — hydrostatic P = P0 + rho*g*h."""

    def test_at_surface(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pressure_from_obs

        obs_surface = {"position": np.array([0.0, 0.0, 0.0])}
        msg = pressure_from_obs(obs_surface)
        # h = -z = 0, so P = P0 = 101325
        assert msg.fluid_pressure == pytest.approx(101325.0)

    def test_underwater(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pressure_from_obs

        obs_5m = {"position": np.array([0.0, 0.0, -5.0])}
        msg = pressure_from_obs(obs_5m)
        # h = -(-5) = 5m
        expected = 101325.0 + 1025.0 * 9.81 * 5.0
        assert msg.fluid_pressure == pytest.approx(expected)

    def test_10m_depth(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pressure_from_obs

        obs_10m = {"position": np.array([3.0, 4.0, -10.0])}
        msg = pressure_from_obs(obs_10m)
        expected = 101325.0 + 1025.0 * 9.81 * 10.0
        assert msg.fluid_pressure == pytest.approx(expected)

    def test_above_surface(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pressure_from_obs

        obs_above = {"position": np.array([0.0, 0.0, 2.0])}
        msg = pressure_from_obs(obs_above)
        # h = -2 (negative depth — above surface)
        expected = 101325.0 + 1025.0 * 9.81 * (-2.0)
        assert msg.fluid_pressure == pytest.approx(expected)

    def test_variance_is_zero(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pressure_from_obs

        obs_surface = {"position": np.array([0.0, 0.0, 0.0])}
        msg = pressure_from_obs(obs_surface)
        assert msg.variance == pytest.approx(0.0)

    def test_default_frame_id(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import pressure_from_obs

        obs_surface = {"position": np.array([0.0, 0.0, 0.0])}
        msg = pressure_from_obs(obs_surface)
        assert msg.header.frame_id == "pressure_link"


class TestWrenchFromAction:
    """Tests for wrench_from_action."""

    def test_force_components(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import wrench_from_action

        action = np.array([0.5, -0.3, 0.8, 0.0, 0.0, 0.0])
        msg = wrench_from_action(action)
        assert msg.force.x == pytest.approx(0.5)
        assert msg.force.y == pytest.approx(-0.3)
        assert msg.force.z == pytest.approx(0.8)

    def test_torque_components(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import wrench_from_action

        action = np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3])
        msg = wrench_from_action(action)
        assert msg.torque.x == pytest.approx(0.1)
        assert msg.torque.y == pytest.approx(-0.2)
        assert msg.torque.z == pytest.approx(0.3)

    def test_full_6dof(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import wrench_from_action

        action = np.array([1.0, -1.0, 0.5, 0.3, -0.3, 0.9])
        msg = wrench_from_action(action)
        assert msg.force.x == pytest.approx(1.0)
        assert msg.force.y == pytest.approx(-1.0)
        assert msg.force.z == pytest.approx(0.5)
        assert msg.torque.x == pytest.approx(0.3)
        assert msg.torque.y == pytest.approx(-0.3)
        assert msg.torque.z == pytest.approx(0.9)

    def test_boundary_values(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import wrench_from_action

        action = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        msg = wrench_from_action(action)
        assert msg.force.x == pytest.approx(1.0)
        assert msg.torque.x == pytest.approx(-1.0)

    def test_from_list(self, _mock_ros2_msgs: None) -> None:
        from oceanscale.ros2.messages import wrench_from_action

        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg = wrench_from_action(action)
        assert msg.force.x == pytest.approx(0.1)
        assert msg.torque.z == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# subscribers.py — ROS2 subscriber wrappers
# ---------------------------------------------------------------------------


def _make_twist_msg(
    lx: float = 0.0,
    ly: float = 0.0,
    lz: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 0.0,
) -> _Msg:
    """Create a mock geometry_msgs/Twist message."""
    return _Msg(
        linear=_Msg(x=lx, y=ly, z=lz),
        angular=_Msg(x=ax, y=ay, z=az),
    )


def _make_pose_stamped_msg(
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    ox: float = 0.0,
    oy: float = 0.0,
    oz: float = 0.0,
    ow: float = 1.0,
) -> _Msg:
    """Create a mock geometry_msgs/PoseStamped message."""
    return _Msg(
        pose=_Msg(
            position=_Msg(x=px, y=py, z=pz),
            orientation=_Msg(x=ox, y=oy, z=oz, w=ow),
        ),
    )


def _make_bool_msg(data: bool) -> _Msg:
    """Create a mock std_msgs/Bool message."""
    return _Msg(data=data)


def _make_float32_multi_array(*values: float) -> _Msg:
    """Create a mock std_msgs/Float32MultiArray message."""
    return _Msg(data=list(values))


class TestTwistCommandSubscriber:
    """Tests for TwistCommandSubscriber."""

    def test_initial_action_is_zeros(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        action = sub.get_action()
        np.testing.assert_array_equal(action, np.zeros(6, dtype=np.float32))

    def test_callback_updates_action(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        msg = _make_twist_msg(lx=0.5, ly=-0.3, lz=0.1, ax=0.2, ay=-0.1, az=0.4)
        sub._callback(msg)

        action = sub.get_action()
        expected = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.4], dtype=np.float32)
        np.testing.assert_array_almost_equal(action, expected)

    def test_action_ordering(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        """Verify action is [surge, sway, heave, roll, pitch, yaw]."""
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        msg = _make_twist_msg(lx=0.1, ly=0.2, lz=0.3, ax=0.4, ay=0.5, az=0.6)
        sub._callback(msg)

        action = sub.get_action()
        np.testing.assert_array_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_stamp_updates_on_callback(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        assert sub.stamp == pytest.approx(0.0)
        sub._callback(_make_twist_msg(lx=1.0))
        assert sub.stamp > 0.0

    def test_repr(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        node = MagicMock()
        sub = TwistCommandSubscriber(node)
        assert "TwistCommandSubscriber" in repr(sub)


class TestTwistClamping:
    """Verify values are clamped to [-1, 1]."""

    def test_positive_clamp(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        msg = _make_twist_msg(lx=5.0, ly=10.0, lz=-3.0, ax=2.0, ay=0.5, az=-8.0)
        sub._callback(msg)

        action = sub.get_action()
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        assert action[0] == pytest.approx(1.0)  # 5.0 clamped
        assert action[1] == pytest.approx(1.0)  # 10.0 clamped
        assert action[2] == pytest.approx(-1.0)  # -3.0 clamped
        assert action[3] == pytest.approx(1.0)  # 2.0 clamped
        assert action[4] == pytest.approx(0.5)  # within range
        assert action[5] == pytest.approx(-1.0)  # -8.0 clamped

    def test_already_in_range(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        msg = _make_twist_msg(lx=0.5, ly=-0.5, lz=0.0, ax=1.0, ay=-1.0, az=0.3)
        sub._callback(msg)

        action = sub.get_action()
        expected = np.array([0.5, -0.5, 0.0, 1.0, -1.0, 0.3], dtype=np.float32)
        np.testing.assert_array_almost_equal(action, expected)


class TestEmergencyStopSubscriber:
    """Tests for EmergencyStopSubscriber."""

    def test_initial_state_not_stopped(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import EmergencyStopSubscriber

        sub = EmergencyStopSubscriber(MagicMock())
        assert sub.is_stopped() is False

    def test_stop_true(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import EmergencyStopSubscriber

        sub = EmergencyStopSubscriber(MagicMock())
        sub._callback(_make_bool_msg(True))
        assert sub.is_stopped() is True

    def test_stop_false(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import EmergencyStopSubscriber

        sub = EmergencyStopSubscriber(MagicMock())
        sub._callback(_make_bool_msg(True))
        sub._callback(_make_bool_msg(False))
        assert sub.is_stopped() is False

    def test_stamp_updates(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import EmergencyStopSubscriber

        sub = EmergencyStopSubscriber(MagicMock())
        sub._callback(_make_bool_msg(True))
        assert sub.stamp > 0.0


class TestWaypointSubscriber:
    """Tests for WaypointSubscriber."""

    def test_initial_waypoint_is_none(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import WaypointSubscriber

        sub = WaypointSubscriber(MagicMock())
        assert sub.get_waypoint() is None

    def test_callback_stores_position_and_orientation(
        self, _mock_ros2_msgs: None, _mock_rclpy: None
    ) -> None:
        from oceanscale.ros2.subscribers import WaypointSubscriber

        sub = WaypointSubscriber(MagicMock())
        msg = _make_pose_stamped_msg(px=1.0, py=2.0, pz=3.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
        sub._callback(msg)

        result = sub.get_waypoint()
        assert result is not None
        pos, quat = result
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(quat, [0.0, 0.0, 0.0, 1.0])

    def test_waypoint_overwrite(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import WaypointSubscriber

        sub = WaypointSubscriber(MagicMock())
        sub._callback(_make_pose_stamped_msg(px=1.0, py=2.0, pz=3.0))
        sub._callback(_make_pose_stamped_msg(px=4.0, py=5.0, pz=6.0))

        result = sub.get_waypoint()
        assert result is not None
        pos, _ = result
        np.testing.assert_array_almost_equal(pos, [4.0, 5.0, 6.0])

    def test_stamp_updates(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import WaypointSubscriber

        sub = WaypointSubscriber(MagicMock())
        assert sub.stamp == pytest.approx(0.0)
        sub._callback(_make_pose_stamped_msg(px=1.0))
        assert sub.stamp > 0.0


class TestThrusterCommandSubscriber:
    """Tests for ThrusterCommandSubscriber."""

    def test_initial_cmd_is_none(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import ThrusterCommandSubscriber

        sub = ThrusterCommandSubscriber(MagicMock())
        assert sub.get_thruster_cmd() is None

    def test_callback_stores_command(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import ThrusterCommandSubscriber

        sub = ThrusterCommandSubscriber(MagicMock())
        msg = _make_float32_multi_array(0.5, -0.3, 0.8, 0.1)
        sub._callback(msg)

        cmd = sub.get_thruster_cmd()
        assert cmd is not None
        np.testing.assert_array_almost_equal(cmd, [0.5, -0.3, 0.8, 0.1])

    def test_no_clamping(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        """Thruster commands are NOT clamped — caller's responsibility."""
        from oceanscale.ros2.subscribers import ThrusterCommandSubscriber

        sub = ThrusterCommandSubscriber(MagicMock())
        msg = _make_float32_multi_array(5.0, -3.0, 10.0)
        sub._callback(msg)

        cmd = sub.get_thruster_cmd()
        assert cmd is not None
        np.testing.assert_array_almost_equal(cmd, [5.0, -3.0, 10.0])

    def test_cmd_overwrite(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        from oceanscale.ros2.subscribers import ThrusterCommandSubscriber

        sub = ThrusterCommandSubscriber(MagicMock())
        sub._callback(_make_float32_multi_array(1.0, 2.0))
        sub._callback(_make_float32_multi_array(3.0, 4.0, 5.0))

        cmd = sub.get_thruster_cmd()
        assert cmd is not None
        np.testing.assert_array_almost_equal(cmd, [3.0, 4.0, 5.0])


class TestThreadSafety:
    """Verify lock-based thread safety for subscriber state."""

    def test_twist_concurrent_write_read(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        """TwistCommandSubscriber: no race between callback and get_action."""
        from oceanscale.ros2.subscribers import TwistCommandSubscriber

        sub = TwistCommandSubscriber(MagicMock())
        errors: list[str] = []

        def writer() -> None:
            for i in range(500):
                msg = _make_twist_msg(lx=float(i % 3) * 0.5)
                try:
                    sub._callback(msg)
                except Exception as exc:
                    errors.append(f"writer: {exc}")

        def reader() -> None:
            for _ in range(500):
                try:
                    action = sub.get_action()
                    assert action.shape == (6,)
                except Exception as exc:
                    errors.append(f"reader: {exc}")

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_emergency_stop_concurrent(self, _mock_ros2_msgs: None, _mock_rclpy: None) -> None:
        """EmergencyStopSubscriber: no race between callback and is_stopped."""
        from oceanscale.ros2.subscribers import EmergencyStopSubscriber

        sub = EmergencyStopSubscriber(MagicMock())
        errors: list[str] = []

        def writer() -> None:
            for i in range(500):
                try:
                    sub._callback(_make_bool_msg(i % 2 == 0))
                except Exception as exc:
                    errors.append(f"writer: {exc}")

        def reader() -> None:
            for _ in range(500):
                try:
                    _ = sub.is_stopped()
                except Exception as exc:
                    errors.append(f"reader: {exc}")

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# __init__.py — lazy import gate
# ---------------------------------------------------------------------------


class TestLazyImport:
    """Tests for __init__.py __getattr__ gating."""

    def test_lazy_import_raises_without_rclpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Accessing a lazy name without rclpy raises ImportError."""
        # Ensure rclpy is NOT importable for _check_rclpy
        monkeypatch.delitem(sys.modules, "rclpy", raising=False)

        # Reload the module so __getattr__ is fresh
        import oceanscale.ros2 as ros2_mod

        # __getattr__ should be called for any name in _lazy
        with pytest.raises(ImportError, match="rclpy is required"):
            ros2_mod.pose_from_obs  # noqa: B018

    def test_lazy_import_raises_for_bridge_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bridge names also require rclpy."""
        monkeypatch.delitem(sys.modules, "rclpy", raising=False)

        import oceanscale.ros2 as ros2_mod

        with pytest.raises(ImportError, match="rclpy is required"):
            ros2_mod.OceanROS2Bridge  # noqa: B018

    def test_unknown_attribute_raises_attribute_error(
        self, _mock_ros2_msgs: None, _mock_rclpy: None
    ) -> None:
        """Accessing a non-existent name raises AttributeError."""
        import oceanscale.ros2 as ros2_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            ros2_mod.nonexistent_function  # noqa: B018
