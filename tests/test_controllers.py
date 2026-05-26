"""Tests for the controllers module — PID, Lee position, configs, pretrained."""

from pathlib import Path

import numpy as np
import pytest

from oceanscale.controllers.pid import (
    CascadedPIDController,
    PIDConfig,
    PIDController,
)
from oceanscale.controllers.lee_position import (
    LeePositionConfig,
    LeePositionController,
)


class TestPIDConfig:
    def test_rexrov_position_preset(self):
        c = PIDConfig.rexrov_position()
        assert c.kp.shape == (6,)
        assert c.kp[0] == 3300.0
        assert c.ki.sum() == 0.0

    def test_rexrov_nonlinear_preset(self):
        c = PIDConfig.rexrov_nonlinear()
        assert c.ki[0] == 200.0

    def test_bluerov2_hover_preset(self):
        c = PIDConfig.bluerov2_hover()
        assert c.kp[0] == pytest.approx(1.1)
        assert c.kp[2] == pytest.approx(0.25)

    def test_cascaded_presets(self):
        p = PIDConfig.rexrov_cascaded_position()
        v = PIDConfig.rexrov_cascaded_velocity()
        assert p.sat is not None
        assert v.sat is not None


class TestPIDController:
    def test_zero_error_zero_output(self):
        ctrl = PIDController(PIDConfig.rexrov_position())
        out = ctrl.compute(np.zeros(6, dtype=np.float32))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_proportional_response(self):
        ctrl = PIDController(PIDConfig.rexrov_position())
        error = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
        out = ctrl.compute(error)
        assert out[0] > 0

    def test_integral_accumulation(self):
        cfg = PIDConfig(
            kp=np.zeros(6, dtype=np.float32),
            ki=np.ones(6, dtype=np.float32) * 100,
            kd=np.zeros(6, dtype=np.float32),
        )
        ctrl = PIDController(cfg, dt=0.01)
        error = np.ones(6, dtype=np.float32)
        out1 = ctrl.compute(error)
        out2 = ctrl.compute(error)
        assert out2[0] > out1[0]

    def test_reset(self):
        ctrl = PIDController(PIDConfig.rexrov_position())
        ctrl.compute(np.ones(6, dtype=np.float32))
        ctrl.reset()
        assert ctrl._integral.sum() == 0.0

    def test_saturation(self):
        ctrl = PIDController(PIDConfig.rexrov_cascaded_position())
        error = np.ones(6, dtype=np.float32) * 1000
        out = ctrl.compute(error)
        assert np.all(np.abs(out) <= ctrl.config.sat + 1e-6)


class TestCascadedPID:
    def test_default_construction(self):
        ctrl = CascadedPIDController()
        assert ctrl.pos_pid is not None
        assert ctrl.vel_pid is not None

    def test_compute(self):
        ctrl = CascadedPIDController()
        out = ctrl.compute(
            np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
            np.zeros(6, dtype=np.float32),
        )
        assert out.shape == (6,)

    def test_reset(self):
        ctrl = CascadedPIDController()
        ctrl.compute(np.ones(6, dtype=np.float32), np.zeros(6, dtype=np.float32))
        ctrl.reset()
        assert ctrl.pos_pid._integral.sum() == 0.0


class TestLeePositionController:
    def test_default_config(self):
        c = LeePositionConfig.bluerov_heavy()
        assert c.position_gain.shape == (3,)
        assert c.position_gain[0] == 6.0

    def test_compute_at_target(self):
        ctrl = LeePositionController()
        out = ctrl.compute(
            position=np.array([1, 0, 0], dtype=np.float32),
            velocity=np.zeros(6, dtype=np.float32),
            orientation_quat=np.array([0, 0, 0, 1], dtype=np.float32),
            angular_velocity=np.zeros(6, dtype=np.float32),
            target_position=np.array([1, 0, 0], dtype=np.float32),
        )
        assert out.shape == (4,)
        assert abs(out[0]) < 1.0

    def test_compute_with_error(self):
        ctrl = LeePositionController()
        out = ctrl.compute(
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(6, dtype=np.float32),
            orientation_quat=np.array([0, 0, 0, 1], dtype=np.float32),
            angular_velocity=np.zeros(6, dtype=np.float32),
            target_position=np.array([1, 0, 0], dtype=np.float32),
        )
        assert out.shape == (4,)


class TestConfigs:
    def test_rexrov_pid_yaml_exists(self):
        path = Path(__file__).parent.parent / "oceanscale/controllers/configs/rexrov_pid.yaml"
        assert path.exists()

    def test_bluerov_lee_yaml_exists(self):
        path = Path(__file__).parent.parent / "oceanscale/controllers/configs/bluerov_lee.yaml"
        assert path.exists()


class TestTrainingConfigs:
    def test_hover_config_exists(self):
        path = Path(__file__).parent.parent / "oceanscale/training/configs/hover_bluerov.yaml"
        assert path.exists()

    def test_poshold_config_exists(self):
        path = Path(__file__).parent.parent / "oceanscale/training/configs/poshold_warpauv_ppo.yaml"
        assert path.exists()


class TestPretrainedModels:
    def test_warpauv_weights_exist(self):
        path = Path(__file__).parent.parent / "oceanscale/controllers/pretrained/warpauv_poshold_ppo.pt"
        assert path.exists()
        assert path.stat().st_size > 100_000

    def test_warpauv_weights_loadable(self):
        import torch
        path = Path(__file__).parent.parent / "oceanscale/controllers/pretrained/warpauv_poshold_ppo.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_provenance_exists(self):
        path = Path(__file__).parent.parent / "oceanscale/controllers/pretrained/PROVENANCE.md"
        assert path.exists()
        content = path.read_text()
        assert "isaac-auv-env" in content
