"""Tier-1 determinism — same input produces identical output across repeated runs.

W2 gate requirement: bit-identical (or near-identical within float32 tolerance)
output across 3 consecutive runs. Each kernel writes to its own thread index
via atomic_add (no inter-thread contention), so results must be bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from oceanscale.hydro import Tier1

wp.init()

_COEFFS = dict(
    added_mass=(5.5, 12.7, 14.57, 0.12, 0.12, 0.12),
    d_lin=(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
    d_quad=(18.18, 21.66, 36.99, 1.55, 1.55, 1.55),
    mass=11.4,
    volume=0.0113459,
    coBM=0.01,
)


def _run_pipeline(n_envs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = Tier1(n_envs=n_envs, n_thrusters=8, device="cuda")
    t.set_coeffs(**_COEFFS)

    nu_np = rng.normal(0, 0.5, (n_envs, 6)).astype(np.float32)
    nu = wp.array(nu_np, dtype=wp.spatial_vectorf, device="cuda")

    q_np = np.tile([0.0, 0.0, 0.0, 1.0], (n_envs, 1)).astype(np.float32)
    q_np[:, 0] = rng.normal(0, 0.05, n_envs).astype(np.float32)
    q_np /= np.linalg.norm(q_np, axis=1, keepdims=True)
    quat = wp.array(q_np, dtype=wp.quatf, device="cuda")

    u_np = rng.uniform(-0.5, 0.5, (n_envs, 8)).astype(np.float32)
    u_cmd = wp.array(u_np, dtype=wp.float32, device="cuda")

    t.compute_wrench(nu, quat, u_cmd, dt=1.0 / 240.0)
    wp.synchronize()
    return t.wrench_buf.numpy().copy()


@pytest.mark.gpu
def test_determinism_full_pipeline() -> None:
    """3 runs with identical inputs produce bit-identical wrench output."""
    n_envs = 256
    results = [_run_pipeline(n_envs, seed=42) for _ in range(3)]

    np.testing.assert_array_equal(results[0], results[1], err_msg="run 0 != run 1")
    np.testing.assert_array_equal(results[1], results[2], err_msg="run 1 != run 2")


@pytest.mark.gpu
def test_determinism_individual_kernels() -> None:
    """Each Tier-1 kernel produces identical output across 3 launches."""
    from oceanscale.hydro.tier1_kernels import (
        tier1_added_mass,
        tier1_coriolis_a,
        tier1_damping,
        tier1_restoring,
        tier1_thruster_alloc,
    )

    n = 64
    rng = np.random.default_rng(123)
    nu_np = rng.normal(0, 1.0, (n, 6)).astype(np.float32)
    ma_lin_np = np.tile(_COEFFS["added_mass"][:3], (n, 1))
    ma_ang_np = np.tile(_COEFFS["added_mass"][3:], (n, 1))
    dll_np = np.tile(_COEFFS["d_lin"][:3], (n, 1))
    dla_np = np.tile(_COEFFS["d_lin"][3:], (n, 1))
    dql_np = np.tile(_COEFFS["d_quad"][:3], (n, 1))
    dqa_np = np.tile(_COEFFS["d_quad"][3:], (n, 1))
    q_np = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    u_cmd_np = rng.uniform(-0.5, 0.5, (n, 8)).astype(np.float32)
    T_np = np.zeros((n, 6, 8), dtype=np.float32)
    for k in range(min(6, 8)):
        T_np[:, k, k] = 1.0

    # Pre-allocate GPU arrays (inputs are identical across runs)
    nu_w = wp.array(nu_np, dtype=wp.spatial_vectorf, device="cuda")
    ma_lin_w = wp.array(ma_lin_np, dtype=wp.vec3f, device="cuda")
    ma_ang_w = wp.array(ma_ang_np, dtype=wp.vec3f, device="cuda")
    dll_w = wp.array(dll_np, dtype=wp.vec3f, device="cuda")
    dla_w = wp.array(dla_np, dtype=wp.vec3f, device="cuda")
    dql_w = wp.array(dql_np, dtype=wp.vec3f, device="cuda")
    dqa_w = wp.array(dqa_np, dtype=wp.vec3f, device="cuda")
    quat_w = wp.array(q_np, dtype=wp.quatf, device="cuda")
    mass_w = wp.array(np.full(n, 11.4, dtype=np.float32), dtype=wp.float32, device="cuda")
    vol_w = wp.array(np.full(n, 0.0113459, dtype=np.float32), dtype=wp.float32, device="cuda")
    cobm_w = wp.array(np.full(n, 0.01, dtype=np.float32), dtype=wp.float32, device="cuda")
    u_cmd_w = wp.array(u_cmd_np, dtype=wp.float32, device="cuda")
    u_prev_w = wp.zeros((n, 8), dtype=wp.float32, device="cuda")
    T_w = wp.array(T_np, dtype=wp.float32, device="cuda")

    def _run_three(kernel, inputs):
        outs = []
        for _ in range(3):
            wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda")
            wp.launch(kernel, dim=n, inputs=[*inputs, wrench], device="cuda")
            wp.synchronize()
            outs.append(wrench.numpy().copy())
        return outs

    # added_mass — source: tier1_kernels.py:17-33
    for name, kernel, inputs in [
        ("added_mass", tier1_added_mass, [nu_w, ma_lin_w, ma_ang_w]),
        ("damping", tier1_damping, [nu_w, dll_w, dla_w, dql_w, dqa_w]),
        ("coriolis_a", tier1_coriolis_a, [nu_w, ma_lin_w, ma_ang_w]),
        (
            "restoring",
            tier1_restoring,
            [
                quat_w,
                mass_w,
                vol_w,
                cobm_w,
                1025.0,
                9.81,
            ],
        ),
    ]:
        outs = _run_three(kernel, inputs)
        np.testing.assert_array_equal(outs[0], outs[1], err_msg=f"{name}: run 0 != run 1")
        np.testing.assert_array_equal(outs[1], outs[2], err_msg=f"{name}: run 1 != run 2")

    # thruster_alloc needs u_eff_out as extra output — source: tier1_kernels.py:147-196
    outs_ta = []
    for _ in range(3):
        wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda")
        u_out = wp.zeros((n, 8), dtype=wp.float32, device="cuda")
        wp.launch(
            tier1_thruster_alloc,
            dim=n,
            inputs=[
                u_cmd_w,
                u_prev_w,
                u_out,
                T_w,
                51.5,
                0.05,
                0.1,
                1.0 / 240.0,
                wrench,
            ],
            device="cuda",
        )
        wp.synchronize()
        outs_ta.append(wrench.numpy().copy())
    np.testing.assert_array_equal(outs_ta[0], outs_ta[1], err_msg="thruster_alloc: run 0 != run 1")
    np.testing.assert_array_equal(outs_ta[1], outs_ta[2], err_msg="thruster_alloc: run 1 != run 2")
