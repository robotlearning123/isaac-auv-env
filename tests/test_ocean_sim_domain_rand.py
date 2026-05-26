"""Tests for OceanSim domain randomization."""

import numpy as np
import warp as wp

from oceanscale.sim import OceanSim, OceanSimConfig
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy

wp.init()


def _mass_values(sim: OceanSim) -> np.ndarray:
    return sim.tier1.mass_arr.numpy()[: sim.n_envs].copy()


class TestCoefficientsDifferAcrossEnvs:
    def test_mass_differs_after_randomization(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=16,
            use_domain_randomization=True,
        ))
        mass_before = _mass_values(sim)
        assert np.allclose(mass_before, mass_before[0])

        sim.reset()
        mass_after = _mass_values(sim)
        assert not np.allclose(mass_after, mass_after[0]), (
            "Mass should differ across envs after randomization"
        )
        sim.close()

    def test_damping_differs_after_randomization(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=16,
            use_domain_randomization=True,
        ))
        sim.reset()
        d_lin = sim.tier1.d_lin_lin.numpy()[: sim.n_envs].copy()
        assert not np.allclose(d_lin, d_lin[0]), (
            "Linear damping should differ across envs"
        )
        sim.close()

    def test_no_randomization_when_disabled(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=16,
            use_domain_randomization=False,
        ))
        sim.reset()
        mass = _mass_values(sim)
        assert np.allclose(mass, mass[0]), (
            "Mass should be uniform when domain randomization is disabled"
        )
        sim.close()

    def test_custom_ranges_respected(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=64,
            use_domain_randomization=True,
            randomization_ranges={"mass": 0.05},
        ))
        sim.reset()
        mass = _mass_values(sim)
        base_mass = BlueROV2Heavy().mass
        deviation = np.abs(mass - base_mass) / base_mass
        assert np.all(deviation <= 0.06), (
            f"Mass deviation {deviation.max():.3f} exceeds 5% range"
        )
        sim.close()


class TestSeedReproducibility:
    def test_same_seed_same_result(self):
        cfg = OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=8,
            use_domain_randomization=True,
            randomization_seed=42,
        )
        sim1 = OceanSim(cfg)
        sim1.reset()
        mass1 = _mass_values(sim1)
        sim1.close()

        sim2 = OceanSim(cfg)
        sim2.reset()
        mass2 = _mass_values(sim2)
        sim2.close()

        np.testing.assert_array_equal(mass1, mass2)

    def test_different_seeds_different_result(self):
        sim1 = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=8,
            use_domain_randomization=True,
            randomization_seed=1,
        ))
        sim1.reset()
        mass1 = _mass_values(sim1)
        sim1.close()

        sim2 = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=8,
            use_domain_randomization=True,
            randomization_seed=2,
        ))
        sim2.reset()
        mass2 = _mass_values(sim2)
        sim2.close()

        assert not np.allclose(mass1, mass2)

    def test_consecutive_resets_differ(self):
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=8,
            use_domain_randomization=True,
            randomization_seed=99,
        ))
        sim.reset()
        mass1 = _mass_values(sim)
        sim.reset()
        mass2 = _mass_values(sim)
        sim.close()

        assert not np.allclose(mass1, mass2), (
            "Consecutive resets should produce different randomization"
        )


class TestPartialReset:
    def test_reset_only_specified_envs(self):
        n = 8
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=n,
            use_domain_randomization=True,
            randomization_seed=10,
        ))
        sim.reset()
        mass_before = _mass_values(sim)

        target_ids = np.array([0, 3, 7])
        sim.reset_envs(target_ids)
        mass_after = _mass_values(sim)
        sim.close()

        unchanged_ids = np.array([i for i in range(n) if i not in target_ids])
        np.testing.assert_array_equal(mass_after[unchanged_ids], mass_before[unchanged_ids])
        assert not np.allclose(mass_after[target_ids], mass_before[target_ids])

    def test_partial_reset_preserves_other_envs_coefficients(self):
        n = 4
        sim = OceanSim(OceanSimConfig(
            vehicle=BlueROV2Heavy(),
            n_envs=n,
            use_domain_randomization=True,
        ))
        sim.reset()
        d_quad_before = sim.tier1.d_quad_lin.numpy()[:n].copy()

        sim.reset_envs(np.array([1]))
        d_quad_after = sim.tier1.d_quad_lin.numpy()[:n].copy()
        sim.close()

        np.testing.assert_array_equal(d_quad_after[0], d_quad_before[0])
        np.testing.assert_array_equal(d_quad_after[2], d_quad_before[2])
        np.testing.assert_array_equal(d_quad_after[3], d_quad_before[3])
