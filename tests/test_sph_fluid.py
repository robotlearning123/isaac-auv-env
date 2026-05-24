"""Tests for SPHSolver."""

import numpy as np
import pytest
import warp as wp

from oceanscale.fluid.grid import SPHSolver

wp.init()


def test_sph_creation():
    """Create solver, verify particle arrays."""
    n = 100
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.1,
        domain=(1.0, 1.0, 1.0),
        device="cuda",
    )
    assert solver.n == n
    assert solver.pos.shape[0] == n
    assert solver.vel.shape[0] == n
    assert solver.density.shape[0] == n
    assert solver.force.shape[0] == n
    assert solver.h == 0.1


def test_sph_step():
    """Isolated particles under gravity: COM must follow symplectic Euler free-fall."""
    n = 8
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.05,
        domain=(1.0, 1.0, 1.0),
        rest_density=1000.0,
        stiffness=2000.0,
        viscosity=200.0,
        gravity=-9.81,
        device="cuda",
    )

    # place particles well-separated (spacing > 2*h) so no SPH forces act
    pos_np = np.array(
        [
            [0.2, 0.8, 0.2],
            [0.8, 0.8, 0.2],
            [0.2, 0.8, 0.8],
            [0.8, 0.8, 0.8],
            [0.2, 0.6, 0.2],
            [0.8, 0.6, 0.2],
            [0.2, 0.6, 0.8],
            [0.8, 0.6, 0.8],
        ],
        dtype=np.float32,
    )
    vel_np = np.zeros((n, 3), dtype=np.float32)
    wp.copy(solver.pos, wp.array(pos_np, dtype=wp.vec3, device="cuda"))
    wp.copy(solver.vel, wp.array(vel_np, dtype=wp.vec3, device="cuda"))

    wp.synchronize_device("cuda")
    y_before = float(solver.pos.numpy()[:, 1].mean())

    dt = 0.005
    n_steps = 10
    for _ in range(n_steps):
        solver.step(dt=dt)

    wp.synchronize_device("cuda")
    y_after = float(solver.pos.numpy()[:, 1].mean())

    assert y_after < y_before, f"Particles did not fall: before={y_before:.4f} after={y_after:.4f}"

    # symplectic Euler free-fall from rest: dy = g * dt^2 * n*(n+1)/2
    g = 9.81
    expected_dy = -g * dt * dt * n_steps * (n_steps + 1) / 2.0
    actual_dy = y_after - y_before
    relative_error = abs(actual_dy - expected_dy) / abs(expected_dy)
    assert relative_error < 0.01, (
        f"COM displacement {actual_dy:.6f} too far from free-fall {expected_dy:.6f} "
        f"(error {relative_error:.1%})"
    )


def test_density_computation():
    """Static particles: center density > corner density, all positive."""
    n = 27  # 3x3x3 grid
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.25,
        domain=(1.0, 1.0, 1.0),
        rest_density=1000.0,
        device="cuda",
    )

    # place particles on a 3x3x3 grid in [0.3, 0.7]
    spacing = 0.15
    offset = 0.35
    pos_np = np.zeros((n, 3), dtype=np.float32)
    idx = 0
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                pos_np[idx] = [offset + ix * spacing, offset + iy * spacing, offset + iz * spacing]
                idx += 1

    wp.copy(solver.pos, wp.array(pos_np, dtype=wp.vec3, device="cuda"))
    wp.copy(solver.vel, wp.zeros(n, dtype=wp.vec3, device="cuda"))

    solver.build_spatial_hash()
    solver.compute_density()
    wp.synchronize_device("cuda")

    density = solver.density.numpy()

    assert np.all(density > 0), f"Some densities are non-positive: {density.min()}"

    # center (idx 13) has 18 neighbors within h=0.25; corners have ~6.
    # center density must be strictly higher than any corner.
    corner_indices = [0, 2, 6, 8, 18, 20, 24, 26]
    corner_density_mean = float(np.mean(density[corner_indices]))
    assert density[13] > corner_density_mean, (
        f"Center density ({density[13]:.2f}) not higher than corner mean ({corner_density_mean:.2f})"
    )
    # quantitative: center should be at least 50% denser than average corner
    assert density[13] > 1.5 * corner_density_mean, (
        f"Center/corner density ratio too low: {density[13] / corner_density_mean:.2f}x"
    )


def test_sph_conservation():
    """Particle count and total mass unchanged; all particles remain in domain."""
    n = 27  # 3x3x3 grid
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.25,
        domain=(1.0, 1.0, 1.0),
        rest_density=1000.0,
        stiffness=500.0,
        viscosity=100.0,
        gravity=-9.81,
        device="cuda",
    )

    spacing = 0.15
    offset = 0.35
    pos_np = np.zeros((n, 3), dtype=np.float32)
    idx = 0
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                pos_np[idx] = [offset + ix * spacing, offset + iy * spacing, offset + iz * spacing]
                idx += 1
    vel_np = np.zeros((n, 3), dtype=np.float32)
    wp.copy(solver.pos, wp.array(pos_np, dtype=wp.vec3, device="cuda"))
    wp.copy(solver.vel, wp.array(vel_np, dtype=wp.vec3, device="cuda"))

    total_mass_before = solver.n * solver.mass

    dt = 0.0002
    for _ in range(10):
        solver.step(dt=dt)

    wp.synchronize_device("cuda")

    total_mass_after = solver.n * solver.mass
    assert total_mass_after == total_mass_before, "Total particle mass changed"

    # all particles must remain inside the domain (boundary reflection keeps them)
    pos = solver.pos.numpy()
    assert np.all(pos >= 0.0), f"Particle escaped domain (min={pos.min():.4f})"
    assert np.all(pos[:, 0] <= solver.domain[0]), "Particle escaped domain in x"
    assert np.all(pos[:, 1] <= solver.domain[1]), "Particle escaped domain in y"
    assert np.all(pos[:, 2] <= solver.domain[2]), "Particle escaped domain in z"


@pytest.mark.xfail(
    reason="known bug: grid.py:675 pressure term non-symmetric (divides by rho_j only)"
)
def test_sph_force_symmetry():
    """Two particles with different densities must exert equal and opposite forces."""
    n = 2
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.25,
        domain=(1.0, 1.0, 1.0),
        rest_density=1000.0,
        stiffness=2000.0,
        viscosity=200.0,
        gravity=0.0,
        device="cuda",
    )

    pos_np = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]], dtype=np.float32)
    wp.copy(solver.pos, wp.array(pos_np, dtype=wp.vec3, device="cuda"))
    wp.copy(solver.vel, wp.zeros(n, dtype=wp.vec3, device="cuda"))

    # Asymmetric densities → p_term uses only rho_j, so forces are not opposite
    density_np = np.array([1500.0, 800.0], dtype=np.float32)
    wp.copy(solver.density, wp.array(density_np, dtype=wp.float32, device="cuda"))

    solver.compute_forces()
    wp.synchronize_device("cuda")

    f = solver.force.numpy()
    net_force = f[0] + f[1]
    np.testing.assert_allclose(
        net_force,
        [0.0, 0.0, 0.0],
        atol=100.0,
        err_msg=f"Newton's third law violated: F_0={f[0]}, F_1={f[1]}, net={net_force}",
    )


@pytest.mark.xfail(reason="known bug: grid.py:675-677 asymmetric pressure + extra 1/dist")
def test_sph_momentum_conservation():
    """Total momentum must be conserved in a closed system (no gravity)."""
    n = 6
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.25,
        domain=(1.0, 1.0, 1.0),
        rest_density=1000.0,
        stiffness=2000.0,
        viscosity=100.0,
        gravity=0.0,
        device="cuda",
    )

    # Asymmetric cluster → non-uniform density → asymmetric pressure forces
    pos_np = np.array(
        [
            [0.45, 0.50, 0.50],
            [0.55, 0.50, 0.50],
            [0.50, 0.45, 0.50],
            [0.50, 0.55, 0.50],
            [0.47, 0.47, 0.50],
            [0.53, 0.53, 0.50],
        ],
        dtype=np.float32,
    )

    vel_np = np.zeros((n, 3), dtype=np.float32)
    vel_np[0] = [0.5, 0.1, 0.0]
    vel_np[1] = [-0.3, 0.0, -0.2]

    wp.copy(solver.pos, wp.array(pos_np, dtype=wp.vec3, device="cuda"))
    wp.copy(solver.vel, wp.array(vel_np, dtype=wp.vec3, device="cuda"))

    wp.synchronize_device("cuda")
    momentum_before = solver.vel.numpy().sum(axis=0) * solver.mass

    dt = 0.001
    for _ in range(10):
        solver.step(dt)

    wp.synchronize_device("cuda")
    momentum_after = solver.vel.numpy().sum(axis=0) * solver.mass

    np.testing.assert_allclose(
        momentum_after,
        momentum_before,
        rtol=0.01,
        err_msg=f"Momentum drift: before={momentum_before}, after={momentum_after}",
    )
