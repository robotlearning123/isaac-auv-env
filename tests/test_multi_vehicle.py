"""Tests for multi-vehicle fleet simulation."""

import numpy as np
import pytest
import warp as wp

from oceanscale.sim import (
    MultiVehicleConfig,
    OceanSim,
    OceanSimConfig,
    VehicleSpec,
)
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy, BlueROV2MarineGym

wp.init()


def _make_two_vehicle_sim(n_envs: int = 1) -> OceanSim:
    return OceanSim(MultiVehicleConfig(
        vehicles={
            "alpha": VehicleSpec(BlueROV2Heavy(), init_pos=(0.0, 0.0, -5.0)),
            "beta": VehicleSpec(BlueROV2Heavy(), init_pos=(5.0, 0.0, -5.0)),
        },
        n_envs=n_envs,
    ))


class TestMultiVehicleConstruction:
    def test_two_vehicles(self):
        sim = _make_two_vehicle_sim(n_envs=2)
        assert sim._is_multi
        assert len(sim._fleet) == 2
        assert "alpha" in sim._fleet
        assert "beta" in sim._fleet
        sim.close()

    def test_reset_returns_nested_dict(self):
        sim = _make_two_vehicle_sim()
        obs = sim.reset()
        assert "alpha" in obs
        assert "beta" in obs
        for vehicle_obs in obs.values():
            assert "position" in vehicle_obs
            assert "orientation" in vehicle_obs
        sim.close()

    def test_dict_step(self):
        sim = _make_two_vehicle_sim()
        sim.reset()
        obs = sim.step({
            "alpha": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
            "beta": np.zeros(6, dtype=np.float32),
        })
        assert "alpha" in obs
        assert "beta" in obs
        assert obs["alpha"]["position"].shape == (1, 3)
        sim.close()

    def test_mixed_vehicle_types(self):
        sim = OceanSim(MultiVehicleConfig(
            vehicles={
                "rov": VehicleSpec(BlueROV2Heavy(), init_pos=(0, 0, -5)),
                "rov_mg": VehicleSpec(BlueROV2MarineGym(), init_pos=(3, 0, -5)),
            },
            n_envs=1,
        ))
        obs = sim.reset()
        assert "rov" in obs
        assert "rov_mg" in obs
        sim.close()

    def test_non_dict_action_raises(self):
        sim = _make_two_vehicle_sim()
        sim.reset()
        with pytest.raises(TypeError, match="dict"):
            sim.step(np.zeros(6))
        sim.close()

    def test_batched_envs(self):
        sim = _make_two_vehicle_sim(n_envs=4)
        obs = sim.reset()
        assert obs["alpha"]["position"].shape == (4, 3)
        obs = sim.step({
            "alpha": np.random.uniform(-1, 1, (4, 6)).astype(np.float32),
            "beta": np.zeros((4, 6), dtype=np.float32),
        })
        assert obs["alpha"]["position"].shape == (4, 3)
        sim.close()


class TestIndependentMotion:
    def test_independent_actions_produce_independent_motion(self):
        sim = _make_two_vehicle_sim()
        sim.reset()
        for _ in range(200):
            sim.step({
                "alpha": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
                "beta": np.zeros(6, dtype=np.float32),
            })
        obs = sim._observe()
        p_a = obs["alpha"]["position"][0]
        p_b = obs["beta"]["position"][0]
        # alpha moved forward
        assert np.linalg.norm(p_a - np.array([0, 0, -5])) > 0.01
        # beta stayed put
        assert np.linalg.norm(p_b - np.array([5, 0, -5])) < 0.1
        sim.close()

    def test_opposite_actions(self):
        sim = OceanSim(MultiVehicleConfig(
            vehicles={
                "left": VehicleSpec(BlueROV2Heavy(), init_pos=(0, 0, -5)),
                "right": VehicleSpec(BlueROV2Heavy(), init_pos=(0, 0, -5)),
            },
            n_envs=1,
        ))
        sim.reset()
        for _ in range(200):
            sim.step({
                "left": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
                "right": np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32),
            })
        obs = sim._observe()
        p_left = obs["left"]["position"][0]
        p_right = obs["right"]["position"][0]
        # They diverged in x
        assert p_left[0] > p_right[0]
        sim.close()


class TestResetIndependence:
    def test_reset_one_vehicle_doesnt_affect_other(self):
        sim = _make_two_vehicle_sim()
        sim.reset()
        for _ in range(200):
            sim.step({
                "alpha": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
                "beta": np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
            })
        obs_before = sim._observe()
        p_b_before = obs_before["beta"]["position"][0].copy()

        sim.reset_vehicle("alpha")
        obs_after = sim._observe()

        # alpha back to init
        np.testing.assert_allclose(
            obs_after["alpha"]["position"][0], [0, 0, -5], atol=0.1,
        )
        # beta unaffected
        np.testing.assert_allclose(
            obs_after["beta"]["position"][0], p_b_before, atol=0.01,
        )
        sim.close()

    def test_full_reset_both_vehicles(self):
        sim = _make_two_vehicle_sim()
        sim.reset()
        for _ in range(100):
            sim.step({
                "alpha": np.ones(6, dtype=np.float32),
                "beta": np.ones(6, dtype=np.float32),
            })
        obs = sim.reset()
        for vehicle_obs in obs.values():
            np.testing.assert_allclose(
                vehicle_obs["linear_velocity"], 0.0, atol=1e-5,
            )
        sim.close()


class TestAddVehicle:
    def test_add_vehicle_after_construction(self):
        sim = OceanSim(MultiVehicleConfig(
            vehicles={"alpha": VehicleSpec(BlueROV2Heavy())},
            n_envs=1,
        ))
        assert len(sim._fleet) == 1
        sim.add_vehicle("beta", BlueROV2Heavy(), init_pos=(5, 0, -5))
        assert len(sim._fleet) == 2
        sim.reset()
        obs = sim.step({
            "alpha": np.zeros(6, dtype=np.float32),
            "beta": np.zeros(6, dtype=np.float32),
        })
        assert "alpha" in obs
        assert "beta" in obs
        sim.close()

    def test_add_vehicle_on_single_raises(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=1))
        with pytest.raises(RuntimeError, match="MultiVehicleConfig"):
            sim.add_vehicle("x", BlueROV2Heavy())
        sim.close()


class TestBackwardCompat:
    def test_single_vehicle_still_works(self):
        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        assert not sim._is_multi
        obs = sim.reset()
        assert obs["position"].shape == (2, 3)
        obs = sim.step(np.zeros(6, dtype=np.float32))
        assert obs["position"].shape == (2, 3)
        sim.close()

    def test_single_vehicle_step_torch(self):
        import torch

        sim = OceanSim(OceanSimConfig(vehicle=BlueROV2Heavy(), n_envs=2))
        sim.reset()
        obs = sim.step_torch(torch.zeros(6))
        assert obs["position"].shape[0] == 2
        sim.close()

    def test_single_reset_restores_position(self):
        init = (1.0, 2.0, -3.0)
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(), init_pos=init, n_envs=1,
        ))
        sim.reset()
        for _ in range(50):
            sim.step(np.array([1, 0, 0, 0, 0, 0], dtype=np.float32))
        obs = sim.reset()
        np.testing.assert_allclose(obs["position"][0], list(init), atol=0.01)
        sim.close()
