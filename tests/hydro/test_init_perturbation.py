"""Test random initial position perturbation in ROVEnv."""

import numpy as np
import pytest


def _make_env(n_envs=8, **kwargs):
    from oceanscale.rov_env import ROVEnv
    return ROVEnv(n_envs=n_envs, device="cpu", sensor_noise_std=0.0, **kwargs)


def test_reset_with_noise_varies_across_envs():
    env = _make_env(init_pos_noise_std=1.0, init_yaw_noise_std=0.5)
    obs, _ = env.reset(seed=42)
    positions = env.state_curr.body_q.numpy()[:, :3]
    assert not np.allclose(positions, env.target_pos, atol=0.01), \
        "Positions should differ from target with noise enabled"
    env.close()


def test_reset_no_noise_deterministic():
    env = _make_env(init_pos_noise_std=0.0, init_yaw_noise_std=0.0)
    obs1, _ = env.reset(seed=42)
    pos1 = env.state_curr.body_q.numpy()[:, :3].copy()
    obs2, _ = env.reset(seed=42)
    pos2 = env.state_curr.body_q.numpy()[:, :3].copy()
    assert np.allclose(pos1, pos2, atol=1e-6), \
        "Deterministic reset should reproduce same positions"
    assert np.allclose(pos1, env.target_pos, atol=1e-6), \
        "No noise: all envs at target"
    env.close()


def test_mean_radius_approx_noise_std():
    env = _make_env(n_envs=512, init_pos_noise_std=1.0, init_yaw_noise_std=0.0)
    env.reset(seed=123)
    positions = env.state_curr.body_q.numpy()[:, :3]
    radii = np.linalg.norm(positions - env.target_pos, axis=1)
    # Uniform[-1,1]^3 has mean radius ~ sqrt(3) * uniform_1d_mean ~ 0.96
    assert 0.3 < radii.mean() < 2.0, f"Mean radius {radii.mean():.3f} unexpected for std=1.0"
    env.close()


def test_seed_reproducibility():
    env = _make_env(init_pos_noise_std=0.5, init_yaw_noise_std=0.3)
    obs1, _ = env.reset(seed=99)
    pos1 = env.state_curr.body_q.numpy()[:, :3].copy()
    obs2, _ = env.reset(seed=99)
    pos2 = env.state_curr.body_q.numpy()[:, :3].copy()
    assert np.allclose(pos1, pos2, atol=1e-6), \
        "Same seed should give same perturbation"
    env.close()
