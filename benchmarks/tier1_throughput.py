"""T2.8 — Tier-1 throughput benchmark at N=8192 envs on RTX 5090.

Gate: ≥ 100,000 env-steps/s.

Replicate(N) BlueROV-like body, attach Tier1 with full 6-kernel pipeline,
inject wrench, step Newton SolverSemiImplicit. Measure env-steps/s.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp


def _make_template():
    import newton

    t = newton.ModelBuilder()
    t.add_body()
    t.add_shape_sphere(0, radius=0.15)
    t.joint_q = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    t.joint_qd = [0.0] * 6
    return t


def bench(n_envs: int, n_steps: int = 100) -> dict[str, float]:
    import newton

    from oceanscale.hydro import Tier1

    wp.init()
    template = _make_template()
    scene = newton.ModelBuilder()
    scene.replicate(template, world_count=n_envs, spacing=(0.0, 0.0, 0.0))
    model = scene.finalize(device="cuda")

    state_a, state_b = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)

    solver = newton.solvers.SolverSemiImplicit(model)

    # Tier-1 — BlueROV basic coefs
    tier1 = Tier1(n_envs=n_envs, n_thrusters=8, device="cuda")
    tier1.set_coeffs(
        added_mass=(5.5, 12.7, 14.57, 0.12, 0.12, 0.12),
        d_lin=(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
        d_quad=(18.18, 21.66, 36.99, 1.55, 1.55, 1.55),
        mass=11.4,
        volume=0.0113459,
        coBM=0.01,
    )

    # Build nu / quat / u_cmd buffers we'll feed each step
    nu = wp.zeros(n_envs, dtype=wp.spatial_vectorf, device="cuda")
    quat = wp.array(
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_envs, 1)),
        dtype=wp.quatf,
        device="cuda",
    )
    u_cmd = wp.zeros((n_envs, 8), dtype=wp.float32, device="cuda")

    dt = 1.0 / 240.0

    # warmup: 5 steps to JIT-compile + cache modules
    for _ in range(5):
        tier1.compute_wrench(nu, quat, u_cmd, dt=dt)
        tier1.write_to_body_f(state_a.body_f)
        solver.step(state_a, state_b, control, None, dt)
        state_a, state_b = state_b, state_a
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        tier1.compute_wrench(nu, quat, u_cmd, dt=dt)
        tier1.write_to_body_f(state_a.body_f)
        solver.step(state_a, state_b, control, None, dt)
        state_a, state_b = state_b, state_a
    wp.synchronize()
    wall = time.perf_counter() - t0

    return {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "wall_s": wall,
        "step_rate_hz": n_steps / wall,
        "env_steps_per_sec": (n_steps * n_envs) / wall,
        "wall_per_step_ms": wall / n_steps * 1000,
    }


def main() -> None:
    print("=" * 78)
    print("Tier-1 Fossen throughput — RTX 5090, Newton SolverSemiImplicit + 6 Warp kernels")
    print("=" * 78)
    print(f"{'N':>6}  {'steps/s':>10}  {'ms/step':>9}  {'M env-steps/s':>14}  {'wall (s)':>10}")
    print("-" * 78)
    target = 100_000  # env-steps/s gate per goals/W2 §2
    for n in (64, 256, 1024, 4096, 8192):
        try:
            r = bench(n_envs=n, n_steps=200)
            marker = "✓" if r["env_steps_per_sec"] >= target else " "
            print(
                f"{r['n_envs']:>6}  {r['step_rate_hz']:>10.1f}  "
                f"{r['wall_per_step_ms']:>9.3f}  "
                f"{r['env_steps_per_sec'] / 1e6:>14.3f}  "
                f"{r['wall_s']:>10.3f}  {marker}"
            )
        except Exception as e:
            print(f"{n:>6}  FAILED: {type(e).__name__}: {e}")
    print()
    print(f"Target: ≥ {target / 1000:.0f}k env-steps/s @ N=8192 (per goals/W2 §2)")


if __name__ == "__main__":
    main()
