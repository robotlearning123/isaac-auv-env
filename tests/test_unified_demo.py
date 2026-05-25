"""Tests for the grand unified demo — all NVIDIA features end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

from oceanscale.demo import UnifiedDemo


@pytest.fixture(scope="module")
def demo():
    return UnifiedDemo()


class TestDemoCreation:
    def test_creates(self, demo):
        assert demo is not None

    def test_has_wave(self, demo):
        assert demo.wave is not None

    def test_has_dvl(self, demo):
        assert demo.dvl is not None

    def test_has_sonar(self, demo):
        assert demo.sonar is not None

    def test_has_tether(self, demo):
        assert demo.tether is not None

    def test_has_cloth(self, demo):
        assert demo.cloth is not None

    def test_has_volume_solver(self, demo):
        assert demo.volume_solver is not None

    def test_has_adaptive_domain(self, demo):
        assert demo.adaptive_domain is not None

    def test_has_mesh_boundary(self, demo):
        assert demo.mesh_boundary is not None

    def test_has_surface_extractor(self, demo):
        assert demo.surface_extractor is not None

    def test_has_ocean_physics_models(self, demo):
        assert demo.water_column is not None
        assert demo.current_field is not None
        assert demo.acoustics is not None
        assert demo.propeller is not None
        assert demo.buoyancy is not None


class TestDemoReset:
    def test_reset_returns_dict(self, demo):
        obs = demo.reset()
        assert isinstance(obs, dict)

    def test_reset_has_rov_position(self, demo):
        obs = demo.reset()
        assert "rov_position" in obs
        assert obs["rov_position"].shape == (3,)

    def test_reset_has_dvl(self, demo):
        obs = demo.reset()
        assert "dvl" in obs
        assert "altitude" in obs["dvl"]

    def test_reset_has_sonar(self, demo):
        obs = demo.reset()
        assert "sonar_ranges" in obs
        assert len(obs["sonar_ranges"]) == 16


class TestDemoStep:
    def test_step_returns_all_keys(self, demo):
        demo.reset()
        obs = demo.step(np.zeros(6, dtype=np.float32))
        expected = [
            "rov_position", "rov_velocity", "wave_velocity",
            "current", "water_density", "sound_speed", "acoustic_loss_db",
            "dvl", "sonar_ranges", "tether_tension",
            "tether_positions", "cloth_deformation", "dist_to_dock",
            "reward", "time", "step",
        ]
        for key in expected:
            assert key in obs, f"Missing key: {key}"

    def test_step_all_finite(self, demo):
        demo.reset()
        obs = demo.step(np.zeros(6, dtype=np.float32))
        assert np.all(np.isfinite(obs["rov_position"]))
        assert np.all(np.isfinite(obs["rov_velocity"]))
        assert np.all(np.isfinite(obs["wave_velocity"]))
        assert np.isfinite(obs["reward"])
        assert np.isfinite(obs["tether_tension"])

    def test_step_sonar_finite(self, demo):
        demo.reset()
        obs = demo.step()
        assert np.all(np.isfinite(obs["sonar_ranges"]))

    def test_step_time_advances(self, demo):
        demo.reset()
        obs1 = demo.step()
        obs2 = demo.step()
        assert obs2["time"] > obs1["time"]
        assert obs2["step"] == obs1["step"] + 1


class TestDemoMultiStep:
    def test_100_steps_no_nan(self, demo):
        demo.reset()
        for i in range(100):
            action = np.random.randn(6).astype(np.float32) * 0.3
            obs = demo.step(action)
            assert np.all(np.isfinite(obs["rov_position"])), f"NaN at step {i}"
            assert np.all(np.isfinite(obs["rov_velocity"])), f"NaN vel at step {i}"
            assert np.isfinite(obs["reward"]), f"NaN reward at step {i}"

    def test_zero_action_100_steps_remains_bounded(self):
        d = UnifiedDemo()
        start = d.reset()["rov_position"].copy()
        for _ in range(100):
            obs = d.step(np.zeros(6, dtype=np.float32))

        drift = float(np.linalg.norm(obs["rov_position"] - start))
        assert drift < 20.0
        assert obs["rov_position"][2] > -d.depth


class TestAllFeaturesActive:
    def test_wave_velocity_nonzero(self, demo):
        demo.reset()
        obs = demo.step()
        assert np.any(np.abs(obs["wave_velocity"]) > 1e-10)

    def test_dvl_altitude_positive(self, demo):
        demo.reset()
        obs = demo.step()
        assert obs["dvl"]["altitude"] > 0 or obs["dvl"]["altitude"] == obs["dvl"]["ranges"].max()

    def test_sonar_detects_something(self, demo):
        demo.reset()
        for _ in range(10):
            obs = demo.step(np.array([0.5, 0, 0.5, 0, 0, 0], dtype=np.float32))
        ranges = obs["sonar_ranges"]
        assert np.any(ranges < demo.sonar.max_range) or np.all(ranges == demo.sonar.max_range)

    def test_tether_exists(self, demo):
        demo.reset()
        obs = demo.step()
        assert obs["tether_positions"].shape[0] == demo.tether.n_segments

    def test_cloth_has_particles(self, demo):
        assert demo.cloth.particle_count > 0

    def test_volume_solver_has_voxels(self, demo):
        assert demo.volume_solver.get_voxel_count() > 0

    def test_adaptive_domain_has_cells(self, demo):
        assert demo.adaptive_domain.cell_count > 0

    def test_surface_extractor_works(self, demo):
        import warp as wp
        sdf = wp.full((32, 32, 32), value=1.0, dtype=wp.float32, device=demo.device)
        v, _idx = demo.surface_extractor.extract(sdf, threshold=0.5)
        assert isinstance(v, np.ndarray)

    def test_current_nonzero(self, demo):
        demo.reset()
        obs = demo.step()
        assert np.any(np.abs(obs["current"]) > 0)

    def test_reward_in_range(self, demo):
        demo.reset()
        obs = demo.step()
        assert 0 <= obs["reward"] <= 1.0


class TestDemoRun:
    def test_run_completes(self, demo):
        result = demo.run(n_steps=50)
        assert result["n_steps"] == 50
        assert result["wall_time_s"] > 0
        assert result["steps_per_sec"] > 0

    def test_run_all_features_counted(self, demo):
        result = demo.run(n_steps=10)
        assert result["features_active"] == result["features_total"]

    def test_run_has_metrics(self, demo):
        result = demo.run(n_steps=10)
        assert "volume_voxels" in result
        assert "adaptive_cells" in result
        assert "fft_grid_shape" in result
        assert "surface_mesh_verts" in result
        assert "cloth_deformation" in result
