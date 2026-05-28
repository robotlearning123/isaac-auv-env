"""Tests for the MarineGym compatibility layer.

Verifies all ported modules: math, thruster, controller, sensor, hover task config.
"""

import math

import pytest
import torch

from oceanscale.marinegym_compat import (
    euler_to_quaternion,
    normalize,
    off_diag,
    quat_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    symexp,
    symlog,
)
from oceanscale.marinegym_compat.controller import (
    AttitudeController,
    LeePositionController,
    RateController,
    compute_parameters,
)
from oceanscale.marinegym_compat.hover_task import ACT_DIM, OBS_DIM, MarineGymHoverCfg
from oceanscale.marinegym_compat.sensor import (
    FisheyeCameraCfg,
    PinholeCameraCfg,
    orientation_from_view,
    load_camera_cfg_from_dict,
)
from oceanscale.marinegym_compat.thruster import RotorConfig, T200Thruster
from oceanscale.marinegym_compat.transforms import (
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_quat():
    torch.manual_seed(42)
    euler = torch.rand(64, 3) * torch.tensor([math.pi, math.pi / 2 - 0.1, math.pi])
    return euler_to_quaternion(euler), euler


@pytest.fixture
def rotor_cfg():
    return RotorConfig(
        force_constants=torch.tensor([4.4e-7] * 6),
        moment_constants=torch.zeros(6),
        max_rotation_velocities=torch.tensor([3900.0] * 6),
        time_constants=torch.tensor([0.02] * 6),
        directions=torch.zeros(6),
        num_rotors=6,
    )


@pytest.fixture
def uav_params():
    return {
        "name": "BlueROVHeavy",
        "mass": 11.5,
        "inertia": {"xx": 0.16, "yy": 0.16, "zz": 0.32},
        "rotor_configuration": {
            "num_rotors": 8,
            "directions": [1, -1, 1, -1, 1, -1, 1, -1],
            "time_constants": [0.43] * 8,
            "force_constants": [4.4e-7] * 8,
            "moment_constants": [1.4e-7] * 8,
            "max_rotation_velocities": [3900.0] * 8,
            "rotor_angles": [0, 1.57, 3.14, 4.71, 0, 1.57, 3.14, 4.71],
            "arm_lengths": [0.2] * 8,
        },
    }


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


class TestMath:
    def test_euler_quaternion_roundtrip(self, batch_quat):
        q, euler = batch_quat
        euler2 = quaternion_to_euler(q)
        assert (euler - euler2).abs().max() < 1e-4

    def test_quat_rotate_inverse(self, batch_quat):
        q, _ = batch_quat
        v = torch.randn(64, 3)
        v_back = quat_rotate_inverse(q, quat_rotate(q, v))
        assert (v - v_back).abs().max() < 1e-4

    def test_quat_mul_identity(self, batch_quat):
        q, _ = batch_quat
        q_inv = q.clone()
        q_inv[:, 1:] *= -1
        q_id = quat_mul(q, q_inv)
        assert q_id[:, 0].abs().sub(1.0).abs().max() < 1e-4

    def test_rotation_matrix_orthogonal(self, batch_quat):
        q, _ = batch_quat
        R = quaternion_to_rotation_matrix(q)
        I = torch.bmm(R.transpose(-2, -1), R)
        assert (I - torch.eye(3)).abs().max() < 1e-4

    def test_rotation_matrix_det(self, batch_quat):
        q, _ = batch_quat
        R = quaternion_to_rotation_matrix(q)
        dets = torch.linalg.det(R)
        assert (dets - 1.0).abs().max() < 1e-4

    def test_euler_rotmat_roundtrip(self, batch_quat):
        _, euler = batch_quat
        R = euler_to_rotation_matrix(euler)
        euler2 = rotation_matrix_to_euler(R)
        assert (euler - euler2).abs().max() < 1e-4

    def test_symlog_symexp_inverse(self):
        x = torch.randn(50)
        assert (x - symexp(symlog(x))).abs().max() < 1e-5

    def test_off_diag(self):
        A = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        od = off_diag(A)
        assert od.shape == (3, 2)

    def test_normalize(self):
        x = torch.randn(32, 3)
        xn = normalize(x)
        assert (xn.norm(dim=-1) - 1.0).abs().max() < 1e-5

    def test_quat_axis(self, batch_quat):
        q, _ = batch_quat
        assert quat_axis(q, 0).shape == (64, 3)
        assert quat_axis(q, 2).shape == (64, 3)


# ---------------------------------------------------------------------------
# Thruster
# ---------------------------------------------------------------------------


class TestThruster:
    def test_t200_positive_thrust(self, rotor_cfg):
        t200 = T200Thruster(rotor_cfg, dt=0.02)
        t, _, _, _ = t200(torch.ones(6) * 0.5, torch.zeros(6), torch.zeros(6))
        assert (t > 0).all()

    def test_t200_dead_zone(self, rotor_cfg):
        t200 = T200Thruster(rotor_cfg, dt=0.02)
        _, _, _, rpm = t200(torch.ones(6) * 0.01, torch.zeros(6), torch.zeros(6))
        assert (rpm.abs() < 1.0).all()

    def test_t200_batch(self, rotor_cfg):
        t200 = T200Thruster(rotor_cfg, dt=0.02)
        t, m, nt, nr = t200(torch.randn(32, 6), torch.zeros(32, 6), torch.zeros(32, 6))
        assert t.shape == (32, 6)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class TestController:
    def test_lee_position_controller(self, uav_params):
        lee = LeePositionController(g=9.81, uav_params=uav_params)
        rs = torch.zeros(1, 13)
        rs[0, :3] = torch.tensor([1.0, 0.5, 2.0])
        rs[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cmd = lee.compute(rs, target_pos=torch.zeros(1, 3))
        assert cmd.shape == (1, 8)

    def test_attitude_controller(self, uav_params):
        att = AttitudeController(g=9.81, uav_params=uav_params)
        rs = torch.zeros(1, 13)
        rs[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cmd = att.compute(rs, target_thrust=torch.ones(1, 1) * 10.0)
        assert cmd.shape == (1, 8)

    def test_rate_controller(self, uav_params):
        rate = RateController(g=9.81, uav_params=uav_params)
        rs = torch.zeros(1, 13)
        rs[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cmd = rate.compute(rs, target_rate=torch.zeros(1, 3), target_thrust=torch.ones(1, 1) * 10.0)
        assert cmd.shape == (1, 8)


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------


class TestSensor:
    def test_orientation_from_view(self):
        q = orientation_from_view([0, 0, 5], [0, 0, 0])
        assert len(q) == 4
        assert q[0] > 0  # w > 0 for forward-looking camera

    def test_pinhole_defaults(self):
        cfg = PinholeCameraCfg()
        assert cfg.resolution == (640, 480)
        assert cfg.projection_type == "pinhole"

    def test_fisheye_defaults(self):
        cfg = FisheyeCameraCfg()
        assert cfg.projection_type == "fisheye_polynomial"

    def test_load_from_dict(self):
        cfg = load_camera_cfg_from_dict({"projection_type": "pinhole", "resolution": [320, 240]})
        assert cfg.resolution == (320, 240)


# ---------------------------------------------------------------------------
# Hover task config
# ---------------------------------------------------------------------------


class TestHoverTask:
    def test_dimensions(self):
        cfg = MarineGymHoverCfg()
        assert cfg.compute_obs_dim() == OBS_DIM
        assert OBS_DIM == 32
        assert ACT_DIM == 6

    def test_config_defaults(self):
        cfg = MarineGymHoverCfg()
        assert cfg.num_envs == 64
        assert cfg.episode_length_s == 10.0
        assert cfg.time_encoding is True

    def test_config_no_time_encoding(self):
        cfg = MarineGymHoverCfg(time_encoding=False)
        assert cfg.compute_obs_dim() == 28  # 32 - 4
