"""T2.8 — Tier-1 Fossen throughput benchmark.

Asserts >= 100k env-steps/s at 8192 envs on RTX 5090
per IMPLEMENTATION_PLAN.md W2 gate §T2.8.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp

from oceanscale.hydro import Tier1

wp.init()

_N_ENVS = [64, 512, 8192]
_WARMUP = 10
_STEPS = 200


def _make_tier1(n_envs: int) -> Tier1:
    t = Tier1(n_envs=n_envs, n_thrusters=8, device="cuda")
    t.set_coeffs(
        added_mass=(5.5, 12.7, 14.57, 0.12, 0.12, 0.12),
        d_lin=(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
        d_quad=(18.18, 21.66, 36.99, 1.55, 1.55, 1.55),
        mass=11.4,
        volume=0.0113459,
        coBM=0.01,
    )
    return t


def _make_inputs(n_envs: int, rng: np.random.Generator):
    nu = wp.array(
        rng.normal(0, 0.5, (n_envs, 6)).astype(np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    q_np = np.tile([0.0, 0.0, 0.0, 1.0], (n_envs, 1)).astype(np.float32)
    q_np[:, 0] = rng.normal(0, 0.05, n_envs).astype(np.float32)
    q_np /= np.linalg.norm(q_np, axis=1, keepdims=True)
    quat = wp.array(q_np, dtype=wp.quatf, device="cuda")
    u_cmd = wp.array(
        rng.uniform(-0.5, 0.5, (n_envs, 8)).astype(np.float32),
        dtype=wp.float32,
        device="cuda",
    )
    return nu, quat, u_cmd


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.parametrize("n_envs", _N_ENVS)
def test_tier1_throughput(n_envs: int) -> None:
    t = _make_tier1(n_envs)
    rng = np.random.default_rng(0)
    nu, quat, u_cmd = _make_inputs(n_envs, rng)
    dt = 1.0 / 240.0

    for _ in range(_WARMUP):
        t.compute_wrench(nu, quat, u_cmd, dt=dt)
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(_STEPS):
        t.compute_wrench(nu, quat, u_cmd, dt=dt)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    rate = n_envs * _STEPS / elapsed
    print(
        f"\n  n_envs={n_envs:5d}  steps={_STEPS}  elapsed={elapsed:.3f}s  env-steps/s={rate:,.0f}"
    )

    if n_envs == 8192:
        assert rate >= 100_000, f"Throughput {rate:,.0f} env-steps/s < 100k gate at n_envs=8192"
