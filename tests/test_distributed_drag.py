"""Tests for per-link distributed drag model."""

import numpy as np
import pytest
import warp as wp

from oceanscale.hydro.distributed_drag import (
    DistributedDrag,
    DistributedDragConfig,
)

wp.init()


@pytest.fixture
def model():
    return DistributedDrag(
        DistributedDragConfig(water_density=1025.0, water_velocity=(0.0, 0.0, 0.0)),
        device="cuda:0",
    )


def _ones(n, val=0.0):
    return np.full((n, 3), val, dtype=np.float32)


def _single_link(model, vx=0.0, vy=0.0, vz=0.0, cd=1.0, area=0.1):
    pos = np.zeros((1, 3), dtype=np.float32)
    vel = np.array([[vx, vy, vz]], dtype=np.float32)
    angvel = np.zeros((1, 3), dtype=np.float32)
    cds = np.full((1, 3), cd, dtype=np.float32)
    areas = np.full((1, 3), area, dtype=np.float32)
    return model.compute(pos, vel, angvel, cds, areas)


class TestDragBasics:
    def test_zero_velocity_no_drag(self, model):
        forces, torques = _single_link(model, vx=0.0, vy=0.0, vz=0.0)
        np.testing.assert_allclose(forces[0], [0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(torques[0], [0, 0, 0], atol=1e-8)

    def test_positive_vx_gives_negative_fx(self, model):
        forces, _ = _single_link(model, vx=1.0)
        assert forces[0, 0] < 0.0

    def test_quadratic_scaling(self, model):
        f1, _ = _single_link(model, vx=1.0, cd=1.0, area=0.1)
        f2, _ = _single_link(model, vx=2.0, cd=1.0, area=0.1)
        ratio = abs(f2[0, 0]) / abs(f1[0, 0])
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_drag_proportional_to_area(self, model):
        f1, _ = _single_link(model, vx=1.0, area=0.1)
        f2, _ = _single_link(model, vx=1.0, area=0.2)
        ratio = abs(f2[0, 0]) / abs(f1[0, 0])
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_known_value(self, model):
        forces, _ = _single_link(model, vx=1.0, cd=1.0, area=0.1)
        expected = -0.5 * 1025.0 * 1.0 * 0.1 * 1.0
        assert forces[0, 0] == pytest.approx(expected, rel=1e-4)


class TestCurrentSubtraction:
    def test_current_reduces_drag(self):
        model_still = DistributedDrag(
            DistributedDragConfig(water_velocity=(0, 0, 0)), device="cuda:0"
        )
        model_current = DistributedDrag(
            DistributedDragConfig(water_velocity=(0.5, 0, 0)), device="cuda:0"
        )
        f_still, _ = _single_link(model_still, vx=1.0)
        f_current, _ = _single_link(model_current, vx=1.0)
        assert abs(f_current[0, 0]) < abs(f_still[0, 0])

    def test_moving_with_current_no_drag(self):
        model = DistributedDrag(
            DistributedDragConfig(water_velocity=(1.0, 0, 0)), device="cuda:0"
        )
        forces, _ = _single_link(model, vx=1.0)
        assert forces[0, 0] == pytest.approx(0.0, abs=1e-4)


class TestMultiLink:
    def test_independent_links(self, model):
        n = 4
        pos = np.zeros((n, 3), dtype=np.float32)
        vel = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
        ], dtype=np.float32)
        angvel = np.zeros((n, 3), dtype=np.float32)
        cds = np.ones((n, 3), dtype=np.float32)
        areas = np.full((n, 3), 0.1, dtype=np.float32)
        forces, _ = model.compute(pos, vel, angvel, cds, areas)

        assert forces[0, 0] < 0.0
        assert forces[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert forces[1, 1] < 0.0
        assert forces[1, 0] == pytest.approx(0.0, abs=1e-6)
        assert forces[2, 2] < 0.0
        np.testing.assert_allclose(forces[3], [0, 0, 0], atol=1e-6)

    def test_batch_consistency(self, model):
        f_single, _ = _single_link(model, vx=2.0, cd=0.8, area=0.05)

        pos = np.zeros((3, 3), dtype=np.float32)
        vel = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0]], dtype=np.float32)
        angvel = np.zeros((3, 3), dtype=np.float32)
        cds = np.full((3, 3), 0.8, dtype=np.float32)
        areas = np.full((3, 3), 0.05, dtype=np.float32)
        forces, _ = model.compute(pos, vel, angvel, cds, areas)

        for i in range(3):
            assert forces[i, 0] == pytest.approx(f_single[0, 0], rel=1e-4)
