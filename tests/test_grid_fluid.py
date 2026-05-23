"""Tests for GridFluidSolver."""

import numpy as np
import pytest
import warp as wp

wp.init()

from oceanscale.fluid.grid import GridFluidSolver, _compute_divergence


def test_grid_creation():
    """Create solver, verify grid shapes."""
    solver = GridFluidSolver(grid_res=16, domain_size=1.0, device="cuda")
    assert solver.n == 16 * 16 * 16
    assert solver.u.shape[0] == solver.n
    assert solver.v.shape[0] == solver.n
    assert solver.w.shape[0] == solver.n
    assert solver.p.shape[0] == solver.n
    assert solver.density.shape[0] == solver.n
    assert solver.dx == pytest.approx(1.0 / 16.0)


def test_advect():
    """Add velocity source, advect, verify density moves."""
    solver = GridFluidSolver(grid_res=16, domain_size=1.0, device="cuda")

    nx = solver.nx
    ny = solver.ny
    nz = solver.nz

    # add density blob at center
    density_np = np.zeros(solver.n, dtype=np.float32)
    indices = np.arange(solver.n)
    ix_arr = indices % nx
    iy_arr = (indices // nx) % ny
    iz_arr = indices // (nx * ny)
    dist = np.sqrt((ix_arr - 8) ** 2 + (iy_arr - 8) ** 2 + (iz_arr - 8) ** 2)
    mask = dist < 3.0
    density_np[mask] = 1.0 - dist[mask] / 3.0
    wp.copy(solver.density, wp.array(density_np, dtype=wp.float32, device="cuda"))

    # set a velocity field pointing in +x
    u_np = np.ones(solver.n, dtype=np.float32) * 2.0
    v_np = np.zeros(solver.n, dtype=np.float32)
    w_np = np.zeros(solver.n, dtype=np.float32)
    wp.copy(solver.u, wp.array(u_np, dtype=wp.float32, device="cuda"))
    wp.copy(solver.v, wp.array(v_np, dtype=wp.float32, device="cuda"))
    wp.copy(solver.w, wp.array(w_np, dtype=wp.float32, device="cuda"))

    # compute center of mass before
    wp.synchronize_device("cuda")
    density_before = solver.density.numpy()
    total_before = density_before.sum()
    com_x_before = (density_before * ix_arr).sum() / max(total_before, 1e-6)

    # advect
    solver.advect(dt=0.01)

    wp.synchronize_device("cuda")
    density_after = solver.density.numpy()
    total_after = density_after.sum()
    com_x_after = (density_after * ix_arr).sum() / max(total_after, 1e-6)

    assert com_x_after > com_x_before, (
        f"COM did not move in +x: before={com_x_before:.3f} after={com_x_after:.3f}"
    )


def test_pressure_solve():
    """Divergence-free check: sum of |div| should decrease after pressure solve."""
    solver = GridFluidSolver(grid_res=16, domain_size=1.0, device="cuda")

    n = solver.n

    # add random velocity to create divergence
    u_np = np.random.uniform(-1, 1, n).astype(np.float32)
    v_np = np.random.uniform(-1, 1, n).astype(np.float32)
    w_np = np.random.uniform(-1, 1, n).astype(np.float32)
    wp.copy(solver.u, wp.array(u_np, dtype=wp.float32, device="cuda"))
    wp.copy(solver.v, wp.array(v_np, dtype=wp.float32, device="cuda"))
    wp.copy(solver.w, wp.array(w_np, dtype=wp.float32, device="cuda"))

    # compute initial divergence
    div_before = wp.zeros(n, dtype=wp.float32, device="cuda")
    wp.launch(
        _compute_divergence,
        dim=n,
        inputs=[solver.u, solver.v, solver.w, div_before, solver.nx, solver.ny, solver.nz, solver.dx],
        device="cuda",
    )
    wp.synchronize_device("cuda")
    div_before_sum = float(np.abs(div_before.numpy()).sum())

    # pressure solve + project
    solver.pressure_solve(iterations=50)
    solver.project()

    # compute divergence after
    div_after = wp.zeros(n, dtype=wp.float32, device="cuda")
    wp.launch(
        _compute_divergence,
        dim=n,
        inputs=[solver.u, solver.v, solver.w, div_after, solver.nx, solver.ny, solver.nz, solver.dx],
        device="cuda",
    )
    wp.synchronize_device("cuda")
    div_after_sum = float(np.abs(div_after.numpy()).sum())

    assert div_after_sum < div_before_sum, (
        f"Divergence did not decrease: before={div_before_sum:.4f} after={div_after_sum:.4f}"
    )


def test_sample_velocity():
    """Sample at known positions, verify interpolation."""
    solver = GridFluidSolver(grid_res=16, domain_size=1.0, device="cuda")

    n = solver.n
    nx = solver.nx

    # set u = grid_index, v = 0, w = 0
    indices = np.arange(n)
    ix_arr = indices % nx
    u_np = ix_arr.astype(np.float32)
    v_np = np.zeros(n, dtype=np.float32)
    w_np = np.zeros(n, dtype=np.float32)
    wp.copy(solver.u, wp.array(u_np, dtype=wp.float32, device="cuda"))
    wp.copy(solver.v, wp.array(v_np, dtype=wp.float32, device="cuda"))
    wp.copy(solver.w, wp.array(w_np, dtype=wp.float32, device="cuda"))

    # sample at position (dx*5, dx*5, dx*5) -> grid coord 5,5,5 -> u ~ 5.0
    dx = solver.dx
    sample_pos = np.array([[5.0 * dx, 5.0 * dx, 5.0 * dx]], dtype=np.float32)
    positions = wp.array(sample_pos, dtype=wp.vec3, device="cuda")
    vel = solver.sample_velocity_at(positions)
    wp.synchronize_device("cuda")
    result = vel.numpy()[0]

    assert abs(result[0] - 5.0) < 0.5, f"Expected u~5.0, got {result[0]:.3f}"
    assert abs(result[1]) < 0.1, f"Expected v~0.0, got {result[1]:.3f}"
    assert abs(result[2]) < 0.1, f"Expected w~0.0, got {result[2]:.3f}"


def test_64_grid_10_steps():
    """Run 10 steps on 64^3 grid, verify no crash."""
    solver = GridFluidSolver(grid_res=64, domain_size=1.0, viscosity=0.001, device="cuda")

    solver.add_source(position=(32.0, 32.0, 32.0), velocity=(1.0, 0.0, 0.0), radius=5.0)

    dt = 0.001
    for _ in range(10):
        solver.step(dt=dt)

    wp.synchronize_device("cuda")

    u_np = solver.u.numpy()
    assert np.abs(u_np).max() > 0.0, "Velocity field is all zeros after 10 steps"
