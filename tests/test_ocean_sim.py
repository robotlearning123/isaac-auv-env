"""Tests for the unified OceanSim orchestrator."""

import numpy as np
import pytest
import warp as wp

from oceanscale.sim import OceanConfig, OceanSim, OceanSimConfig, SensorMount
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy, BlueROV2MarineGym

wp.init()


class TestConstruction:
    def test_default_bluerov(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        assert sim.n_envs == 2
        assert sim.time == 0.0
        sim.close()

    def test_marinegym_bluerov(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2MarineGym(), n_envs=1))
        assert sim.n_envs == 1
        sim.close()

    def test_custom_init_pos(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            init_pos=(3.0, 4.0, -10.0),
            n_envs=1,
        ))
        obs = sim.reset()
        np.testing.assert_allclose(obs["position"][0], [3.0, 4.0, -10.0], atol=0.01)
        sim.close()

    def test_with_ocean_config(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            ocean=OceanConfig(current_speed=0.3, wave_height=1.0),
            n_envs=2,
        ))
        assert sim._current_field is not None
        assert sim._wave_field is not None
        sim.close()

    def test_no_ocean_modules_when_zero(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=1))
        assert sim._current_field is None
        assert sim._wave_field is None
        sim.close()


class TestReset:
    def test_reset_returns_dict(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=4))
        obs = sim.reset()
        assert isinstance(obs, dict)
        assert "position" in obs
        assert "orientation" in obs
        assert "linear_velocity" in obs
        assert "angular_velocity" in obs
        sim.close()

    def test_reset_shapes(self):
        n = 8
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=n))
        obs = sim.reset()
        assert obs["position"].shape == (n, 3)
        assert obs["orientation"].shape == (n, 4)
        assert obs["linear_velocity"].shape == (n, 3)
        assert obs["angular_velocity"].shape == (n, 3)
        sim.close()

    def test_reset_zeroes_velocity(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        sim.reset()
        sim.step(np.ones(6))
        obs = sim.reset()
        np.testing.assert_allclose(obs["linear_velocity"], 0.0, atol=1e-6)
        sim.close()

    def test_reset_restores_position(self):
        init = (1.0, 2.0, -3.0)
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(), init_pos=init, n_envs=2,
        ))
        sim.reset()
        for _ in range(50):
            sim.step(np.array([1, 0, 0, 0, 0, 0]))
        obs = sim.reset()
        np.testing.assert_allclose(obs["position"][0], list(init), atol=0.01)
        sim.close()

    def test_partial_reset(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(), init_pos=(0, 0, -5), n_envs=4,
        ))
        sim.reset()
        for _ in range(20):
            sim.step(np.ones(6))
        obs_before = sim._observe()
        _pos_env0_before = obs_before["position"][0].copy()
        sim.reset_envs(np.array([0, 2]))
        obs_after = sim._observe()
        np.testing.assert_allclose(obs_after["position"][0], [0, 0, -5], atol=0.1)
        sim.close()


class TestStep:
    def test_step_returns_dict(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        sim.reset()
        obs = sim.step(np.zeros(6))
        assert isinstance(obs, dict)
        assert "position" in obs
        sim.close()

    def test_step_advances_time(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=1, dt=0.01))
        sim.reset()
        sim.step(np.zeros(6))
        assert abs(sim.time - 0.01) < 1e-9
        sim.step(np.zeros(6))
        assert abs(sim.time - 0.02) < 1e-9
        sim.close()

    def test_step_increments_count(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        sim.reset()
        sim.step(np.zeros(6))
        obs = sim.step(np.zeros(6))
        assert obs["step_count"][0] == 2
        sim.close()

    def test_nonzero_action_moves_vehicle(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=1))
        obs0 = sim.reset()
        p0 = obs0["position"][0].copy()
        for _ in range(200):
            sim.step(np.array([1, 0, 0, 0, 0, 0]))
        obs1 = sim._observe()
        p1 = obs1["position"][0]
        assert np.linalg.norm(p1 - p0) > 0.001

    def test_zero_action_restoring_force(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=1))
        sim.reset()
        for _ in range(100):
            obs = sim.step(np.zeros(6))
        assert np.all(np.isfinite(obs["position"]))
        sim.close()

    def test_batched_action(self):
        n = 4
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=n))
        sim.reset()
        actions = np.random.uniform(-1, 1, (n, 6)).astype(np.float32)
        obs = sim.step(actions)
        assert obs["position"].shape == (n, 3)
        sim.close()

    def test_single_action_broadcasts(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=4))
        sim.reset()
        obs = sim.step(np.array([0.5, 0, 0, 0, 0, 0]))
        assert obs["position"].shape == (4, 3)
        sim.close()

    def test_finite_after_many_steps(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        sim.reset()
        for _ in range(200):
            obs = sim.step(np.random.uniform(-0.5, 0.5, 6).astype(np.float32))
        assert np.all(np.isfinite(obs["position"]))
        assert np.all(np.isfinite(obs["linear_velocity"]))
        sim.close()


class TestOceanModules:
    def test_current_field_affects_motion(self):
        sim_still = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            ocean=OceanConfig(current_speed=0.0),
            n_envs=1,
        ))
        sim_current = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            ocean=OceanConfig(current_speed=2.0),
            current_drag_coeff=20.0,
            n_envs=1,
        ))
        sim_still.reset()
        sim_current.reset()
        for _ in range(500):
            sim_still.step(np.zeros(6))
            sim_current.step(np.zeros(6))
        p_still = sim_still._observe()["position"][0]
        p_current = sim_current._observe()["position"][0]
        assert not np.allclose(p_still, p_current, atol=0.01)
        sim_still.close()
        sim_current.close()

    def test_water_column_initialized(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            ocean=OceanConfig(surface_temp=25.0, bottom_temp=2.0),
            n_envs=1,
        ))
        assert sim._water_column is not None
        density = sim._water_column.density(10.0)
        assert 1020 < density < 1030
        sim.close()


class TestSensors:
    @pytest.fixture
    def seabed_mesh(self):
        verts = np.array([
            [-50, -50, -30], [50, -50, -30], [-50, 50, -30],
            [50, 50, -30],
        ], dtype=np.float32)
        indices = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
        return wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device="cuda:0"),
            indices=wp.array(indices, dtype=wp.int32, device="cuda:0"),
        )

    def test_magnetometer_attached(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            sensors=[SensorMount(type="magnetometer")],
            n_envs=1,
        ))
        obs = sim.reset()
        assert "sensors" in obs
        assert "magnetometer" in obs["sensors"]
        assert obs["sensors"]["magnetometer"].shape == (3,)
        sim.close()

    def test_dvl_with_seabed(self, seabed_mesh):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            sensors=[SensorMount(type="dvl")],
            seabed_mesh=seabed_mesh,
            n_envs=1,
        ))
        obs = sim.reset()
        assert "sensors" in obs
        assert "dvl" in obs["sensors"]
        sim.close()

    def test_no_dvl_without_mesh(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            sensors=[SensorMount(type="dvl")],
            n_envs=1,
        ))
        obs = sim.reset()
        assert "sensors" not in obs or "dvl" not in obs.get("sensors", {})
        sim.close()
