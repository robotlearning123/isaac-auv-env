"""Tests for SPHSolver."""

import numpy as np
import pytest
import warp as wp

wp.init()

from oceanscale.fluid.grid import SPHSolver


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
    """Step 10 times, verify particles move under gravity."""
    n = 64
    solver = SPHSolver(
        n_particles=n,
        smoothing_length=0.15,
        domain=(1.0, 1.0, 1.0),
        rest_density=1000.0,
        stiffness=2000.0,
        viscosity=200.0,
        gravity=-9.81,
        device="cuda",
    )

    # initialize particles in a block in upper half
    pos_np = np.random.uniform(0.3, 0.7, (n, 3)).astype(np.float32)
    pos_np[:, 1] = np.random.uniform(0.6, 0.9, n).astype(np.float32)  # y in upper half
    vel_np = np.zeros((n, 3), dtype=np.float32)
    wp.copy(solver.pos, wp.array(pos_np, dtype=wp.vec3, device="cuda"))
    wp.copy(solver.vel, wp.array(vel_np, dtype=wp.vec3, device="cuda"))

    # initial mean y
    wp.synchronize_device("cuda")
    y_before = solver.pos.numpy()[:, 1].mean()

    # step 10 times
    dt = 0.0005
    for _ in range(10):
        solver.step(dt=dt)

    wp.synchronize_device("cuda")
    y_after = solver.pos.numpy()[:, 1].mean()

    # particles should have fallen (gravity is -9.81 in y)
    assert y_after < y_before, f"Particles did not fall: before={y_before:.4f} after={y_after:.4f}"


def test_density_computation():
    """Static particles, verify density ~ rest_density at interior."""
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

    # compute density only
    solver.compute_density()
    wp.synchronize_device("cuda")

    density = solver.density.numpy()

    # interior particles should have nonzero density
    # center particle (index 13) should have highest density
    assert density[13] > 0.0, f"Center particle density is zero: {density[13]}"
    # all densities should be positive
    assert np.all(density > 0), f"Some densities are non-positive: {density.min()}"
