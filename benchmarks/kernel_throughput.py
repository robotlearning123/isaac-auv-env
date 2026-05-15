"""Throughput baseline benchmarks — measure real status on this hardware.

Tests:
1. Warp drag kernel — pure GPU kernel, no Newton overhead
2. Newton SemiImplicit — free bodies in vacuum at various scales
3. Newton MuJoCo-Warp — same task, primary Newton solver
"""
from __future__ import annotations

import time

import numpy as np


def warp_drag_throughput(n_envs: int = 8192, n_steps: int = 1000) -> dict[str, float]:
    """Pure Warp kernel throughput — 1D drag, n_envs in parallel."""
    import warp as wp

    wp.init()

    @wp.kernel
    def drag(v: wp.array(dtype=wp.float32), c: wp.float32, f: wp.array(dtype=wp.float32)) -> None:
        i = wp.tid()
        vi = v[i]
        f[i] = -c * wp.abs(vi) * vi

    v = wp.array(np.random.normal(0, 1, n_envs).astype(np.float32), device="cuda")
    f = wp.zeros(n_envs, dtype=wp.float32, device="cuda")

    # warmup
    for _ in range(20):
        wp.launch(drag, dim=n_envs, inputs=[v, 0.5, f], device="cuda")
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        wp.launch(drag, dim=n_envs, inputs=[v, 0.5, f], device="cuda")
    wp.synchronize()
    dt = time.perf_counter() - t0

    steps_per_sec = n_steps / dt
    env_steps_per_sec = steps_per_sec * n_envs
    return {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "wall_s": dt,
        "step_rate_hz": steps_per_sec,
        "env_step_rate_per_sec": env_steps_per_sec,
    }


def newton_throughput(
    n_envs: int = 8192,
    n_steps: int = 100,
    solver_name: str = "SolverSemiImplicit",
) -> dict[str, float]:
    """Newton rigid-body step throughput — n_envs free spheres falling.

    Note: each body is independent (no joints between them). Equivalent to
    n_envs parallel envs, which is the AUV training workload pattern.
    """
    import newton

    builder = newton.ModelBuilder()
    for _ in range(n_envs):
        b = builder.add_body()
        builder.add_shape_sphere(b, radius=0.1)
    builder.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * n_envs
    builder.joint_qd = [0.0] * (n_envs * 6)

    model = builder.finalize(device="cuda")
    state_a, state_b = model.state(), model.state()
    control = model.control()

    solver_cls = getattr(newton.solvers, solver_name)
    solver = solver_cls(model)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)

    dt = 1.0 / 240.0

    # warmup (compiles kernels)
    import warp as wp
    for _ in range(5):
        solver.step(state_a, state_b, control, None, dt)
        state_a, state_b = state_b, state_a
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        solver.step(state_a, state_b, control, None, dt)
        state_a, state_b = state_b, state_a
    wp.synchronize()
    wall = time.perf_counter() - t0

    return {
        "solver": solver_name,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "wall_s": wall,
        "step_rate_hz": n_steps / wall,
        "env_steps_per_sec": (n_steps * n_envs) / wall,
        "wall_per_step_ms": wall / n_steps * 1000,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Throughput benchmarks — RTX 5090, Newton 1.2, Warp 1.13")
    print("=" * 70)
    print()

    print("1) Warp drag kernel — pure GPU, no physics engine overhead")
    print("-" * 70)
    for n in (1024, 8192, 65536, 262144):
        r = warp_drag_throughput(n_envs=n, n_steps=500)
        print(f"   n_envs={n:>6}  → {r['step_rate_hz']:>10.1f} kernel-launches/s  "
              f"= {r['env_step_rate_per_sec'] / 1e6:>7.2f} M env-steps/s")
    print()

    print("2) Newton SolverSemiImplicit — free bodies in vacuum")
    print("-" * 70)
    for n in (64, 256, 1024, 4096, 8192):
        try:
            r = newton_throughput(n_envs=n, n_steps=200, solver_name="SolverSemiImplicit")
            print(f"   n_envs={n:>5}  → {r['step_rate_hz']:>8.1f} steps/s  "
                  f"({r['wall_per_step_ms']:>5.2f} ms/step)  "
                  f"= {r['env_steps_per_sec'] / 1e6:>6.3f} M env-steps/s")
        except Exception as e:
            print(f"   n_envs={n:>5}  FAILED: {type(e).__name__}: {e}")
    print()

    print("3) Newton SolverMuJoCo (MuJoCo-Warp, the primary)")
    print("-" * 70)
    for n in (64, 256, 1024, 4096, 8192):
        try:
            r = newton_throughput(n_envs=n, n_steps=200, solver_name="SolverMuJoCo")
            print(f"   n_envs={n:>5}  → {r['step_rate_hz']:>8.1f} steps/s  "
                  f"({r['wall_per_step_ms']:>5.2f} ms/step)  "
                  f"= {r['env_steps_per_sec'] / 1e6:>6.3f} M env-steps/s")
        except Exception as e:
            print(f"   n_envs={n:>5}  FAILED: {type(e).__name__}: {e}")
    print()
    print("Reference targets (from STACK.md):")
    print("  Tier-0 station-keep on RTX 5090:  >= 600k FPS (=0.6 M env-steps/s)")
    print("  Tier-1 pipe-follow w/ sonar:      >=  80k FPS")
