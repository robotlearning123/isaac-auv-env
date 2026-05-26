"""Tests for domain randomization config and sampling."""

import numpy as np

from oceanscale.training.domain_rand import (
    DomainRandomizationConfig,
    sample_randomized_params,
)


class TestDomainRandomizationConfig:
    def test_default_no_randomization(self):
        cfg = DomainRandomizationConfig.no_randomization()
        assert not cfg.enabled

    def test_warpauv_defaults(self):
        cfg = DomainRandomizationConfig.warpauv_defaults()
        assert cfg.enabled
        assert cfg.volume_range[0] < cfg.volume_range[1]
        assert cfg.com_to_cob_offset_radius == 0.05

    def test_bluerov2_defaults(self):
        cfg = DomainRandomizationConfig.bluerov2_defaults()
        assert cfg.enabled
        assert cfg.mass_range[0] < cfg.mass_range[1]


class TestSampling:
    def test_no_randomization_returns_constants(self):
        cfg = DomainRandomizationConfig.no_randomization()
        params = sample_randomized_params(cfg, n_envs=4)
        assert np.all(params["drag_scale"] == 1.0)
        assert np.all(params["thruster_noise"] == 0.0)

    def test_sample_shape(self):
        cfg = DomainRandomizationConfig.bluerov2_defaults()
        params = sample_randomized_params(cfg, n_envs=16)
        assert params["volume"].shape == (16,)
        assert params["mass"].shape == (16,)
        assert params["com_to_cob_offset"].shape == (16, 3)
        assert params["drag_scale"].shape == (16,)

    def test_volume_within_range(self):
        cfg = DomainRandomizationConfig.bluerov2_defaults()
        params = sample_randomized_params(cfg, n_envs=1000)
        assert np.all(params["volume"] >= cfg.volume_range[0])
        assert np.all(params["volume"] <= cfg.volume_range[1])

    def test_com_offset_within_radius(self):
        cfg = DomainRandomizationConfig.warpauv_defaults()
        params = sample_randomized_params(cfg, n_envs=1000)
        norms = np.linalg.norm(params["com_to_cob_offset"], axis=1)
        assert np.all(norms <= cfg.com_to_cob_offset_radius + 1e-6)

    def test_reproducible_with_seed(self):
        cfg = DomainRandomizationConfig.bluerov2_defaults()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = sample_randomized_params(cfg, n_envs=8, rng=rng1)
        p2 = sample_randomized_params(cfg, n_envs=8, rng=rng2)
        np.testing.assert_array_equal(p1["volume"], p2["volume"])

    def test_different_seeds_differ(self):
        cfg = DomainRandomizationConfig.bluerov2_defaults()
        p1 = sample_randomized_params(cfg, n_envs=100, rng=np.random.default_rng(1))
        p2 = sample_randomized_params(cfg, n_envs=100, rng=np.random.default_rng(2))
        assert not np.array_equal(p1["volume"], p2["volume"])
