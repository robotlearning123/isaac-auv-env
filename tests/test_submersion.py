"""Tests for partial submersion model."""

import numpy as np
import pytest
import warp as wp

from oceanscale.hydro.submersion import (
    PartialSubmersion,
    PartialSubmersionConfig,
)

wp.init()


@pytest.fixture
def model():
    return PartialSubmersion(
        PartialSubmersionConfig(water_surface_z=0.0, water_density=1025.0, gravity=9.81, drag_coeff=1.0),
        device="cuda:0",
    )


def _single(model, z, vz=0.0, h=1.0, vol=0.01):
    pos = np.array([[0.0, 0.0, z]], dtype=np.float32)
    vel = np.array([[0.0, 0.0, vz]], dtype=np.float32)
    heights = np.array([h], dtype=np.float32)
    volumes = np.array([vol], dtype=np.float32)
    return model.compute(pos, vel, heights, volumes)


class TestSubmersionFraction:
    def test_fully_submerged(self, model):
        buoy, drag, _friction = _single(model, z=-5.0)
        expected_buoy = 1025.0 * 9.81 * 0.01
        assert buoy[0, 2] == pytest.approx(expected_buoy, rel=1e-4)

    def test_fully_above_water(self, model):
        buoy, drag, _friction = _single(model, z=5.0)
        assert buoy[0, 2] == pytest.approx(0.0, abs=1e-6)

    def test_half_submerged(self, model):
        buoy, _, _friction = _single(model, z=0.0, h=2.0)
        full_buoy = 1025.0 * 9.81 * 0.01
        assert buoy[0, 2] == pytest.approx(0.5 * full_buoy, rel=1e-4)

    def test_just_touching_surface(self, model):
        buoy, _, _friction = _single(model, z=-0.5, h=1.0)
        assert buoy[0, 2] == pytest.approx(1025.0 * 9.81 * 0.01, rel=1e-4)

    def test_buoyancy_is_upward(self, model):
        buoy, _, _friction = _single(model, z=-2.0)
        assert buoy[0, 2] > 0.0
        assert buoy[0, 0] == pytest.approx(0.0, abs=1e-8)
        assert buoy[0, 1] == pytest.approx(0.0, abs=1e-8)


class TestSubmersionDrag:
    def test_drag_when_submerged(self, model):
        _, drag, _friction = _single(model, z=-5.0, vz=1.0)
        assert drag[0, 2] < 0.0

    def test_no_drag_above_water(self, model):
        _, drag, _friction = _single(model, z=5.0, vz=1.0)
        assert drag[0, 2] == pytest.approx(0.0, abs=1e-6)

    def test_drag_scales_with_submersion(self, model):
        _, drag_full, _ = _single(model, z=-5.0, vz=1.0, h=1.0)
        _, drag_half, _ = _single(model, z=0.0, vz=1.0, h=2.0)
        assert abs(drag_half[0, 2]) < abs(drag_full[0, 2])
        assert abs(drag_half[0, 2]) == pytest.approx(0.5 * abs(drag_full[0, 2]), rel=0.1)

    def test_zero_velocity_no_drag(self, model):
        _, drag, _friction = _single(model, z=-5.0, vz=0.0)
        np.testing.assert_allclose(drag[0], [0.0, 0.0, 0.0], atol=1e-8)


class TestSubmersionBatch:
    def test_multiple_links_at_different_depths(self, model):
        positions = np.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ], dtype=np.float32)
        velocities = np.zeros((3, 3), dtype=np.float32)
        heights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        volumes = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        buoy, drag, _friction = model.compute(positions, velocities, heights, volumes)

        assert buoy[0, 2] > 0.0
        assert buoy[1, 2] > 0.0
        assert buoy[2, 2] == pytest.approx(0.0, abs=1e-6)
        assert buoy[0, 2] > buoy[1, 2]
