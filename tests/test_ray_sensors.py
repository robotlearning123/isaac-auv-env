"""Tests for ray-casting sensors (RayDVL, RaySonar) using wp.Mesh."""

import math

import numpy as np
import pytest
import warp as wp

wp.init()

from oceanscale.sensors.ray_dvl import RayDVL
from oceanscale.sensors.ray_sonar import RaySonar


def _make_flat_plane(y: float = 0.0, size: float = 100.0) -> wp.Mesh:
    """Create a flat plane mesh at given y height."""
    pts = np.array([
        [-size, y, -size], [size, y, -size], [size, y, size], [-size, y, size]
    ], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(pts, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(idx, dtype=wp.int32, device="cuda:0"),
    )


def _make_box(center: np.ndarray, half_ext: float = 1.0) -> wp.Mesh:
    """Create a box mesh centered at the given position."""
    cx, cy, cz = center
    h = half_ext
    pts = np.array([
        [cx - h, cy - h, cz - h], [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h], [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h], [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h], [cx - h, cy + h, cz + h],
    ], dtype=np.float32)
    idx = np.array([
        0, 2, 1, 0, 3, 2,  # back
        4, 5, 6, 4, 6, 7,  # front
        0, 1, 5, 0, 5, 4,  # bottom
        2, 3, 7, 2, 7, 6,  # top
        0, 4, 7, 0, 7, 3,  # left
        1, 2, 6, 1, 6, 5,  # right
    ], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(pts, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(idx, dtype=wp.int32, device="cuda:0"),
    )


# ── DVL tests ──

class TestRayDVL:
    def test_flat_seabed_altitude(self):
        mesh = _make_flat_plane(y=0.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=200.0)
        pos = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        result = dvl.measure(pos)
        expected_altitude = 10.0
        assert abs(result["altitude"] - expected_altitude) < 0.5

    def test_beam_count(self):
        mesh = _make_flat_plane(y=0.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=200.0)
        pos = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        result = dvl.measure(pos)
        assert result["beam_ranges"].shape == (4,)
        assert result["normals"].shape == (4, 3)

    def test_valid_flag(self):
        mesh = _make_flat_plane(y=0.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=200.0)
        pos = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        result = dvl.measure(pos)
        assert result["valid"] is True

    def test_out_of_range(self):
        mesh = _make_flat_plane(y=0.0, size=1.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=50.0)
        pos = np.array([0.0, 1000.0, 0.0], dtype=np.float32)
        result = dvl.measure(pos)
        assert result["altitude"] == 50.0
        assert result["valid"] is False

    def test_closer_means_shorter_range(self):
        mesh = _make_flat_plane(y=0.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=200.0)
        r_high = dvl.measure(np.array([0, 20, 0], dtype=np.float32))
        r_low = dvl.measure(np.array([0, 5, 0], dtype=np.float32))
        assert r_low["altitude"] < r_high["altitude"]

    def test_beam_ranges_consistent_with_geometry(self):
        mesh = _make_flat_plane(y=0.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=200.0)
        height = 10.0
        pos = np.array([0.0, height, 0.0], dtype=np.float32)
        result = dvl.measure(pos)
        cos_angle = math.cos(math.radians(30.0))
        expected_range = height / cos_angle
        for r in result["beam_ranges"]:
            assert abs(r - expected_range) < 0.5, f"Range {r} != expected {expected_range}"

    def test_tilted_orientation(self):
        mesh = _make_flat_plane(y=0.0)
        dvl = RayDVL(mesh, n_beams=4, beam_angle=30.0, max_range=200.0)
        pos = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        r_identity = dvl.measure(pos, orientation=np.array([0, 0, 0, 1], dtype=np.float32))
        pitch_15 = math.radians(15.0) / 2.0
        r_tilted = dvl.measure(pos, orientation=np.array([math.sin(pitch_15), 0, 0, math.cos(pitch_15)], dtype=np.float32))
        assert not np.allclose(r_identity["beam_ranges"], r_tilted["beam_ranges"], atol=0.1)


# ── Sonar tests ──

class TestRaySonar:
    def test_ray_count(self):
        mesh = _make_flat_plane(y=0.0)
        sonar = RaySonar(mesh, n_rays=32, fov=90.0, max_range=50.0)
        pos = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        ranges = sonar.scan(pos)
        assert ranges.shape == (32,)

    def test_empty_scene_max_range(self):
        mesh = _make_flat_plane(y=-100.0, size=1.0)
        sonar = RaySonar(mesh, n_rays=16, fov=90.0, max_range=50.0)
        pos = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        ranges = sonar.scan(pos)
        np.testing.assert_array_equal(ranges, 50.0)

    def test_obstacle_detection(self):
        box = _make_box(center=np.array([0.0, 0.0, 10.0]), half_ext=2.0)
        sonar = RaySonar(box, n_rays=64, fov=90.0, max_range=50.0)
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ranges = sonar.scan(pos)
        center_rays = ranges[28:36]
        assert np.any(center_rays < 50.0), "Sonar should detect box in center rays"
        assert np.any(center_rays < 15.0), "Box at z=10, sonar should see it at ~8m"

    def test_closer_obstacle_shorter_range(self):
        box_near = _make_box(center=np.array([0.0, 0.0, 5.0]), half_ext=1.0)
        box_far = _make_box(center=np.array([0.0, 0.0, 20.0]), half_ext=1.0)
        sonar_near = RaySonar(box_near, n_rays=16, fov=30.0, max_range=50.0)
        sonar_far = RaySonar(box_far, n_rays=16, fov=30.0, max_range=50.0)
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        r_near = sonar_near.scan(pos)
        r_far = sonar_far.scan(pos)
        assert np.min(r_near) < np.min(r_far)

    def test_all_ranges_finite(self):
        mesh = _make_flat_plane(y=0.0)
        sonar = RaySonar(mesh, n_rays=64, fov=90.0, max_range=50.0)
        pos = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        ranges = sonar.scan(pos)
        assert np.all(np.isfinite(ranges))

    def test_ranges_positive(self):
        box = _make_box(center=np.array([0.0, 0.0, 5.0]), half_ext=1.0)
        sonar = RaySonar(box, n_rays=16, fov=60.0, max_range=50.0)
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ranges = sonar.scan(pos)
        assert np.all(ranges > 0.0)
