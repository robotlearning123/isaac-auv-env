"""Tier-1 smoke tests — verify each kernel launches without error and produces
non-NaN output. Real unit tests with numerical verification follow in
subsequent test files (test_added_mass.py, test_damping.py, etc.)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tier1():
    import warp as wp

    from oceanscale.hydro import Tier1

    wp.init()
    n_envs = 64
    n_thrusters = 8
    t = Tier1(n_envs=n_envs, n_thrusters=n_thrusters, device="cuda")
    # BlueROV basic coefs (from MarineGym BlueROV.yaml, NOT Heavy)
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
def test_tier1_import() -> None:
    from oceanscale.hydro import Tier1

    assert Tier1 is not None


@pytest.mark.gpu
def test_tier1_construct() -> None:
    import warp as wp

    from oceanscale.hydro import Tier1

    wp.init()
    t = Tier1(n_envs=8, n_thrusters=8, device="cuda")
    assert t.n_envs == 8
    assert t.wrench_buf.shape == (8,)


@pytest.mark.gpu
def test_tier1_zero_wrench(tier1) -> None:
    import warp as wp

    tier1.zero_wrench()
    wp.synchronize()
    w = tier1.wrench_buf.numpy()
    assert w.shape == (64, 6)
    assert np.allclose(w, 0.0)


@pytest.mark.gpu
def test_tier1_compute_wrench_runs(tier1) -> None:
    """Smoke test — all kernels launch without error, wrench is finite."""
    import warp as wp

    n = tier1.n_envs
    # Zero state: rest body, identity orientation, no commands
    nu = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda")
    quat = wp.array(
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
        dtype=wp.quatf,
        device="cuda",
    )
    u_cmd = wp.zeros((n, 8), dtype=wp.float32, device="cuda")

    tier1.compute_wrench(nu, quat, u_cmd, dt=1.0 / 240.0)
    wp.synchronize()
    w = tier1.wrench_buf.numpy()
    assert w.shape == (n, 6)
    assert np.all(np.isfinite(w)), f"got non-finite values: {w[~np.isfinite(w).any(axis=1)]}"


@pytest.mark.gpu
def test_tier1_neutral_buoyancy_at_rest(tier1) -> None:
    """At rest, identity orientation, near-neutral buoyancy: wrench should be small.

    For BlueROV basic: mass=11.4 kg, volume=0.0113459 m³, ρ=1025 → W=111.8 N, B=114.1 N.
    B-W ≈ +2.3 N (slightly positive → vehicle floats up). Body z-down convention.
    """
    import warp as wp

    n = tier1.n_envs
    nu = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda")
    quat = wp.array(
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
        dtype=wp.quatf,
        device="cuda",
    )
    u_cmd = wp.zeros((n, 8), dtype=wp.float32, device="cuda")

    # Step twice to settle the EMA filter (first step has nu_prev=0, nu_dot will be 0 anyway)
    for _ in range(2):
        tier1.compute_wrench(nu, quat, u_cmd, dt=1.0 / 240.0)
    wp.synchronize()
    w = tier1.wrench_buf.numpy()

    # For identity quaternion, body frame = world frame.
    # World gravity pulls -z (down), buoyancy pushes +z (up).
    # Net world force = (0, 0, B - W).  For BlueROV basic at ρ_salt=1025:
    #   W = m·g = 11.4 · 9.81 = 111.83 N (downward)
    #   B = ρ·V·g = 1025 · 0.0113459 · 9.81 = 114.06 N (upward)
    #   Net = B - W ≈ +2.23 N upward
    # Kernel layout: z_body = quat_rotate_inv(I, (0,0,-1)) = (0,0,-1)
    #                f_body = (W - B) · z_body = (-2.23) · (0,0,-1) = (0, 0, +2.23)
    # Expected: f_body[2] ≈ +2.23 N.
    assert np.allclose(w[:, 0], w[0, 0]), "all envs should be identical"
    fz = w[0, 2]
    W = 11.4 * 9.81
    B = 1025.0 * 0.0113459 * 9.81
    expected = B - W  # positive: floats up
    (
        np.testing.assert_allclose(fz, expected, rtol=0.05),
        (f"net z-force {fz} should be approx B-W={expected:.3f} (positive = floats up)"),
    )
