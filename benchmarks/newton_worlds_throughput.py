"""T1.2: Re-benchmark Newton throughput with ModelBuilder.replicate(world_count=N).

Supersedes the broken numbers from benchmarks/kernel_throughput.py for Newton
solvers — the original measurement put N bodies in world=-1, causing
SolverMuJoCo to OOM at N >= 256. With replicate(N), each body lives in its
own world and MuJoCo handles it as N tiny independent worlds.

Goal: measure (steps/s, ms/step, env-steps/s) for both SolverSemiImplicit
and SolverMuJoCo at N in {64, 256, 1024, 4096, 8192} on RTX 5090.
"""

from __future__ import annotations

import time

import warp as wp


def _make_template():
    import newton

    t = newton.ModelBuilder()
    t.add_body()
    t.add_shape_sphere(0, radius=0.1)
    t.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    t.joint_qd = [0.0] * 6
    return t


def bench(solver_name: str, n_worlds: int, n_steps: int = 200) -> dict[str, float]:
    """Build N-world model, step n_steps, return timing dict."""
    import newton

    template = _make_template()
    if solver_name == "SolverMuJoCo":
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)

    scene = newton.ModelBuilder()
    scene.replicate(template, world_count=n_worlds, spacing=(0.0, 0.0, 0.0))
    model = scene.finalize(device="cuda")

    state_a, state_b = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)

    solver_cls = getattr(newton.solvers, solver_name)
    solver = solver_cls(model)

    dt = 1.0 / 240.0

    # warmup (JIT compile)
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
        "n_worlds": n_worlds,
        "n_steps": n_steps,
        "wall_s": wall,
        "step_rate_hz": n_steps / wall,
        "env_steps_per_sec": (n_steps * n_worlds) / wall,
        "wall_per_step_ms": wall / n_steps * 1000,
    }


def main() -> None:
    print("=" * 78)
    print("Newton throughput with replicate(world_count=N) — RTX 5090, Newton 1.2")
    print("=" * 78)
    print()

    sizes = (64, 256, 1024, 4096, 8192)

    for solver in ("SolverSemiImplicit", "SolverMuJoCo"):
        print(f"\n{solver}")
        print("-" * 78)
        print(f"{'N':>6}  {'steps/s':>10}  {'ms/step':>8}  {'M env-steps/s':>14}  {'wall (s)':>10}")
        for n in sizes:
            try:
                r = bench(solver, n_worlds=n, n_steps=200)
                print(
                    f"{r['n_worlds']:>6}  {r['step_rate_hz']:>10.1f}  "
                    f"{r['wall_per_step_ms']:>8.2f}  "
                    f"{r['env_steps_per_sec'] / 1e6:>14.3f}  "
                    f"{r['wall_s']:>10.3f}"
                )
            except Exception as e:
                print(f"{n:>6}  FAILED: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
