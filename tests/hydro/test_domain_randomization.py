"""Tests for Tier1.randomize_coeffs() domain randomization."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from oceanscale.hydro import RandomizationRanges, Tier1

wp.init()


@pytest.fixture
def tier1_64():
    t = Tier1(n_envs=64, n_thrusters=8, device="cuda")
    t.set_coeffs(
        added_mass=(5.5, 12.7, 14.57, 0.12, 0.12, 0.12),
        d_lin=(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
        d_quad=(18.18, 21.66, 36.99, 1.55, 1.55, 1.55),
        mass=11.4,
        volume=0.0113459,
        coBM=0.01,
    )
    return t


@pytest.mark.gpu
def test_randomize_all_envs(tier1_64) -> None:
    """All envs get different coefficients after randomization."""
    rng = np.random.default_rng(42)
    ma_before = tier1_64.M_A_lin.numpy().copy()
    tier1_64.randomize_coeffs(rng=rng)
    ma_after = tier1_64.M_A_lin.numpy()
    # Should differ from uniform base
    assert not np.allclose(ma_after, ma_before)
    # Each env should differ
    assert not np.allclose(ma_after[0], ma_after[1])


@pytest.mark.gpu
def test_randomize_subset_envs(tier1_64) -> None:
    """Only specified env_ids are randomized; others unchanged."""
    rng = np.random.default_rng(123)
    ma_before = tier1_64.M_A_lin.numpy().copy()
    ids = [0, 5, 10]
    tier1_64.randomize_coeffs(env_ids=ids, rng=rng)
    ma_after = tier1_64.M_A_lin.numpy()
    # Modified envs should differ
    for i in ids:
        assert not np.allclose(ma_after[i], ma_before[i])
    # Unmodified envs should be identical
    for i in range(64):
        if i not in ids:
            np.testing.assert_array_equal(ma_after[i], ma_before[i])


@pytest.mark.gpu
def test_randomize_respects_ranges(tier1_64) -> None:
    """Coefficients stay within expected bounds."""
    base_mass = 11.4
    ranges = RandomizationRanges(
        mass=0.20, added_mass=0.25, d_lin=0.30, d_quad=0.30, volume=0.15, coBM=0.50
    )
    # Run many times to check bounds
    for seed in range(50):
        rng = np.random.default_rng(seed)
        tier1_64.randomize_coeffs(ranges=ranges, rng=rng)
        m = tier1_64.mass_arr.numpy()
        assert np.all(m >= base_mass * (1 - ranges.mass))
        assert np.all(m <= base_mass * (1 + ranges.mass))
        v = tier1_64.volume_arr.numpy()
        assert np.all(v > 0), "volume must stay positive"


@pytest.mark.gpu
def test_randomize_reproducible(tier1_64) -> None:
    """Same RNG seed produces same results."""
    rng1 = np.random.default_rng(999)
    tier1_64.randomize_coeffs(rng=rng1)
    ma1 = tier1_64.M_A_lin.numpy().copy()
    m1 = tier1_64.mass_arr.numpy().copy()

    # Reset to base
    tier1_64.set_coeffs(
        added_mass=(5.5, 12.7, 14.57, 0.12, 0.12, 0.12),
        d_lin=(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
        d_quad=(18.18, 21.66, 36.99, 1.55, 1.55, 1.55),
        mass=11.4,
        volume=0.0113459,
        coBM=0.01,
    )

    rng2 = np.random.default_rng(999)
    tier1_64.randomize_coeffs(rng=rng2)
    ma2 = tier1_64.M_A_lin.numpy()
    m2 = tier1_64.mass_arr.numpy()

    np.testing.assert_array_equal(ma1, ma2)
    np.testing.assert_array_equal(m1, m2)


@pytest.mark.gpu
def test_randomize_damping_and_added_mass(tier1_64) -> None:
    """All 6 DOF of damping and added-mass are independently randomized."""
    rng = np.random.default_rng(7)
    tier1_64.randomize_coeffs(rng=rng)
    ma_lin = tier1_64.M_A_lin.numpy()
    ma_ang = tier1_64.M_A_ang.numpy()
    dll = tier1_64.d_lin_lin.numpy()
    dla = tier1_64.d_lin_ang.numpy()
    for i in range(tier1_64.n_envs):
        assert not np.allclose(ma_lin[i, 0], ma_lin[i, 1])
        assert not np.allclose(ma_ang[i, 0], ma_ang[i, 1])
        assert not np.allclose(dll[i, 0], dll[i, 1])
        assert not np.allclose(dla[i, 0], dla[i, 1])


@pytest.mark.gpu
def test_wrench_after_randomization(tier1_64) -> None:
    """Compute wrench still produces finite output after randomization."""
    rng = np.random.default_rng(42)
    tier1_64.randomize_coeffs(rng=rng)

    n = tier1_64.n_envs
    nu = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda")
    quat = wp.array(
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
        dtype=wp.quatf,
        device="cuda",
    )
    u_cmd = wp.zeros((n, 8), dtype=wp.float32, device="cuda")

    tier1_64.compute_wrench(nu, quat, u_cmd, dt=1.0 / 240.0)
    wp.synchronize()
    w = tier1_64.wrench_buf.numpy()
    assert np.all(np.isfinite(w))


@pytest.mark.gpu
def test_zero_range_no_change(tier1_64) -> None:
    """Ranges all zero => coefficients unchanged."""
    ranges = RandomizationRanges(
        mass=0.0, added_mass=0.0, d_lin=0.0, d_quad=0.0, volume=0.0, coBM=0.0, current_speed_max=0.0
    )
    ma_before = tier1_64.M_A_lin.numpy().copy()
    m_before = tier1_64.mass_arr.numpy().copy()

    rng = np.random.default_rng(42)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng)

    np.testing.assert_array_equal(tier1_64.M_A_lin.numpy(), ma_before)
    np.testing.assert_array_equal(tier1_64.mass_arr.numpy(), m_before)
    assert tier1_64.current_vec_arr is None


# ── current_speed domain randomization tests ──


@pytest.mark.gpu
def test_current_speed_populated(tier1_64) -> None:
    """current_vec_arr is populated after randomize_coeffs with current_speed_max > 0."""
    rng = np.random.default_rng(42)
    ranges = RandomizationRanges(current_speed_max=0.5)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng)
    assert tier1_64.current_vec_arr is not None
    cv = tier1_64.current_vec_arr.numpy()
    assert cv.shape == (64, 3)
    assert np.all(np.isfinite(cv))


@pytest.mark.gpu
def test_current_speed_bounds(tier1_64) -> None:
    """Current speed per env stays in [0, max]."""
    max_speed = 0.5
    ranges = RandomizationRanges(current_speed_max=max_speed)
    for seed in range(50):
        rng = np.random.default_rng(seed)
        tier1_64.randomize_coeffs(ranges=ranges, rng=rng)
        cv = tier1_64.current_vec_arr.numpy()
        speeds = np.linalg.norm(cv, axis=-1)
        assert np.all(speeds >= 0)
        assert np.all(speeds <= max_speed + 1e-6)


@pytest.mark.gpu
def test_current_direction_unit(tier1_64) -> None:
    """Direction is a proper unit vector (norm == speed)."""
    rng = np.random.default_rng(7)
    ranges = RandomizationRanges(current_speed_max=0.5)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng)
    cv = tier1_64.current_vec_arr.numpy()
    speeds = np.linalg.norm(cv, axis=-1)
    nonzero = speeds > 1e-6
    if np.any(nonzero):
        directions = cv[nonzero] / speeds[nonzero, np.newaxis]
        dir_norms = np.linalg.norm(directions, axis=-1)
        np.testing.assert_allclose(dir_norms, 1.0, atol=1e-5)


@pytest.mark.gpu
def test_current_per_env_unique(tier1_64) -> None:
    """Each env gets a different current vector."""
    rng = np.random.default_rng(42)
    ranges = RandomizationRanges(current_speed_max=0.5)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng)
    cv = tier1_64.current_vec_arr.numpy()
    assert not np.allclose(cv[0], cv[1])
    assert not np.allclose(cv[0], cv[2])


@pytest.mark.gpu
def test_current_subset_envs(tier1_64) -> None:
    """Only specified env_ids get current randomized."""
    rng = np.random.default_rng(123)
    ranges = RandomizationRanges(current_speed_max=0.5)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng)
    cv_before = tier1_64.current_vec_arr.numpy().copy()

    ids = [0, 5, 10]
    tier1_64.randomize_coeffs(env_ids=ids, ranges=ranges, rng=rng)
    cv_after = tier1_64.current_vec_arr.numpy()

    for i in ids:
        assert not np.allclose(cv_after[i], cv_before[i])
    for i in range(64):
        if i not in ids:
            np.testing.assert_array_equal(cv_after[i], cv_before[i])


@pytest.mark.gpu
def test_current_reproducible(tier1_64) -> None:
    """Same RNG seed produces same current vectors."""
    rng1 = np.random.default_rng(999)
    ranges = RandomizationRanges(current_speed_max=0.5)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng1)
    cv1 = tier1_64.current_vec_arr.numpy().copy()

    rng2 = np.random.default_rng(999)
    tier1_64.randomize_coeffs(ranges=ranges, rng=rng2)
    cv2 = tier1_64.current_vec_arr.numpy()

    np.testing.assert_array_equal(cv1, cv2)
