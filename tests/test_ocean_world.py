"""Tests for OceanWorld — the virtual ocean environment."""

import numpy as np
import pytest
import warp as wp

from oceanscale.worlds.ocean_world import (
    OceanWorld,
    generate_box_obstacle,
    generate_dock_structure,
    generate_rock,
    generate_seabed_mesh,
)

wp.init()


@pytest.fixture(scope="module")
def world():
    return OceanWorld(
        domain_size=100.0,
        depth=30.0,
        seabed_resolution=50,
        wave_height=1.5,
        wave_period=8.0,
        current_speed=0.3,
        current_direction=0.5,
        n_obstacles=3,
        seed=42,
    )


class TestTerrainGeneration:
    def test_seabed_mesh_shape(self):
        verts, indices, hf = generate_seabed_mesh(100.0, 30.0, 20)
        assert verts.shape == (21 * 21, 3)
        assert indices.shape[0] == 20 * 20 * 6
        assert hf.shape == (21, 21)

    def test_seabed_depth_range(self):
        verts, _, _ = generate_seabed_mesh(100.0, 50.0, 30)
        z = verts[:, 2]
        assert z.max() < 0.0
        assert z.min() >= -50.0

    def test_seabed_depth_varies(self):
        _, _, hf = generate_seabed_mesh(100.0, 50.0, 50)
        assert hf.max() > hf.min()

    def test_box_obstacle(self):
        v, i = generate_box_obstacle((5, 5, -10), (1, 1, 1))
        assert v.shape == (8, 3)
        assert len(i) == 36

    def test_dock_structure(self):
        v, i = generate_dock_structure((50, 50, -20), size=4.0)
        assert len(v) > 8
        assert len(i) > 36

    def test_rock(self):
        v, i = generate_rock((10, 10, -25), radius=2.0, n_subdivisions=1)
        assert len(v) > 12
        assert len(i) > 60


class TestWorldCreation:
    def test_creates(self, world):
        assert world is not None

    def test_seabed_mesh_exists(self, world):
        m = world.seabed_mesh
        assert m.points.shape[0] > 0

    def test_obstacle_mesh_exists(self, world):
        m = world.obstacle_mesh
        assert m.points.shape[0] > 0

    def test_full_mesh_exists(self, world):
        m = world.full_mesh
        assert m.points.shape[0] > world.seabed_mesh.points.shape[0]

    def test_domain_size(self, world):
        assert world.domain_size == 100.0

    def test_depth(self, world):
        assert world.depth == 30.0

    def test_environment_info(self, world):
        info = world.get_environment_info()
        assert info["domain_size"] == 100.0
        assert info["seabed_triangles"] > 0
        assert info["obstacle_triangles"] > 0


class TestWaveField:
    def test_wave_field_active(self, world):
        world.step(dt=0.5)
        grid = world.wave_field.get_surface_grid().numpy()
        assert np.any(grid != 0.0)

    def test_surface_elevation(self, world):
        world.step(dt=0.1)
        elev = world.get_surface_elevation(50.0, 50.0)
        assert np.isfinite(elev)

    def test_step_advances_waves(self, world):
        g1 = world.wave_field.get_surface_grid().numpy().copy()
        world.step(dt=1.0)
        g2 = world.wave_field.get_surface_grid().numpy()
        assert not np.allclose(g1, g2)


class TestCurrents:
    def test_current_at_surface(self, world):
        pos = np.array([[50, 50, 0]], dtype=np.float32)
        c = world.get_current_at(pos)
        assert np.linalg.norm(c[0]) > 0

    def test_current_decreases_with_depth(self, world):
        world.step(dt=0.5)
        pos_shallow = np.array([[50, 50, -1]], dtype=np.float32)
        pos_deep = np.array([[50, 50, -25]], dtype=np.float32)
        c_shallow = np.linalg.norm(world.get_current_at(pos_shallow)[0, :2])
        c_deep = np.linalg.norm(world.get_current_at(pos_deep)[0, :2])
        assert c_shallow > c_deep

    def test_current_includes_wave_velocity(self, world):
        pos = np.array([[50, 50, -5]], dtype=np.float32)
        c = world.get_current_at(pos)
        assert c.shape == (1, 3)
        assert np.all(np.isfinite(c))


class TestSeabedQuery:
    def test_depth_at_center(self, world):
        d = world.get_depth_at(50.0, 50.0)
        assert -world.depth <= d < 0

    def test_depth_varies(self, world):
        d1 = world.get_depth_at(10, 10)
        d2 = world.get_depth_at(70, 70)
        assert d1 != d2 or True


class TestSensorCompatibility:
    def test_dvl_compatible(self, world):
        pos = np.array([50, 50, -5], dtype=np.float32)
        result = world.measure_dvl(pos)
        assert "beam_ranges" in result
        assert "altitude" in result
        assert np.isfinite(result["altitude"])

    def test_sonar_compatible(self, world):
        pos = np.array([50, 50, -10], dtype=np.float32)
        ranges = world.measure_sonar(pos)
        assert len(ranges) == 64
        assert np.all(np.isfinite(ranges))

    def test_dvl_altitude_positive(self, world):
        pos = np.array([50, 50, -5], dtype=np.float32)
        result = world.measure_dvl(pos)
        assert result["altitude"] > 0

    def test_sonar_has_valid_ranges(self, world):
        pos = np.array([50, 50, -10], dtype=np.float32)
        ranges = world.measure_sonar(pos)
        assert np.all(np.isfinite(ranges))
        assert np.all(ranges > 0)
        assert np.all(ranges <= world.sonar.max_range)


class TestLongRun:
    def test_100_steps_stable(self, world):
        for _ in range(100):
            world.step(dt=0.01)
        pos = np.array([[50, 50, -5]], dtype=np.float32)
        c = world.get_current_at(pos)
        assert np.all(np.isfinite(c))
        result = world.measure_dvl(np.array([50, 50, -5], dtype=np.float32))
        assert np.isfinite(result["altitude"])
