"""Tests for oceanscale.sensors — IMU, Pressure, DVL stubs."""

import torch
import pytest

from oceanscale.sensors import DVLSensor, IMUSensor, PressureSensor


# Use CPU for tests (no GPU required in CI)
DEVICE = "cpu"
N_ENVS = 4


def _make_state(n: int, device: str = DEVICE):
    """Create plausible body state tensors."""
    torch.manual_seed(42)
    return {
        "position": torch.randn(n, 3, device=device),
        "velocity": torch.randn(n, 3, device=device),
        "quaternion": _random_unit_quat(n, device),
        "angular_velocity": torch.randn(n, 3, device=device),
    }


def _random_unit_quat(n: int, device: str) -> torch.Tensor:
    q = torch.randn(n, 4, device=device)
    return q / q.norm(dim=-1, keepdim=True)


class TestIMU:
    def test_creation(self):
        sensor = IMUSensor(N_ENVS, device=DEVICE)
        assert sensor.dim == 6

    def test_shape(self):
        s = _make_state(N_ENVS)
        out = IMUSensor(N_ENVS, device=DEVICE).read(**s)
        assert out.shape == (N_ENVS, 6)

    def test_deterministic(self):
        s = _make_state(N_ENVS)
        sensor = IMUSensor(N_ENVS, device=DEVICE, noise_std=0.0)
        a = sensor.read(**s)
        b = sensor.read(**s)
        assert torch.allclose(a, b)

    def test_noisy_varies(self):
        s = _make_state(N_ENVS)
        sensor = IMUSensor(N_ENVS, device=DEVICE, noise_std=0.1)
        a = sensor.read(**s)
        b = sensor.read(**s)
        assert not torch.allclose(a, b)


class TestPressure:
    def test_creation(self):
        sensor = PressureSensor(N_ENVS, device=DEVICE)
        assert sensor.dim == 1

    def test_shape(self):
        s = _make_state(N_ENVS)
        out = PressureSensor(N_ENVS, device=DEVICE).read(
            s["position"], s["velocity"], s["quaternion"]
        )
        assert out.shape == (N_ENVS, 1)

    def test_depth_positive(self):
        pos = torch.tensor([[0, 0, -5.0], [0, 0, 2.0]])
        sensor = PressureSensor(2, device=DEVICE, noise_std=0.0)
        out = sensor.read(pos, torch.zeros(2, 3), torch.zeros(2, 4))
        assert out[0].item() == pytest.approx(5.0)
        assert out[1].item() == pytest.approx(0.0)  # above surface clamped

    def test_deterministic(self):
        s = _make_state(N_ENVS)
        sensor = PressureSensor(N_ENVS, device=DEVICE, noise_std=0.0)
        a = sensor.read(s["position"], s["velocity"], s["quaternion"])
        b = sensor.read(s["position"], s["velocity"], s["quaternion"])
        assert torch.allclose(a, b)

    def test_noisy_varies(self):
        s = _make_state(N_ENVS)
        sensor = PressureSensor(N_ENVS, device=DEVICE, noise_std=0.1)
        a = sensor.read(s["position"], s["velocity"], s["quaternion"])
        b = sensor.read(s["position"], s["velocity"], s["quaternion"])
        assert not torch.allclose(a, b)


class TestDVL:
    def test_creation(self):
        sensor = DVLSensor(N_ENVS, device=DEVICE)
        assert sensor.dim == 3

    def test_shape(self):
        s = _make_state(N_ENVS)
        out = DVLSensor(N_ENVS, device=DEVICE).read(**s)
        assert out.shape == (N_ENVS, 3)

    def test_deterministic(self):
        s = _make_state(N_ENVS)
        sensor = DVLSensor(N_ENVS, device=DEVICE, noise_std=0.0)
        a = sensor.read(**s)
        b = sensor.read(**s)
        assert torch.allclose(a, b)

    def test_noisy_varies(self):
        s = _make_state(N_ENVS)
        sensor = DVLSensor(N_ENVS, device=DEVICE, noise_std=0.1)
        a = sensor.read(**s)
        b = sensor.read(**s)
        assert not torch.allclose(a, b)

    def test_identity_quat_gives_world_vel(self):
        n = 2
        pos = torch.zeros(n, 3)
        vel = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        quat = torch.tensor([[0, 0, 0, 1.0]] * n)  # identity rotation
        sensor = DVLSensor(n, device=DEVICE, noise_std=0.0)
        out = sensor.read(pos, vel, quat)
        assert torch.allclose(out, vel, atol=1e-6)
