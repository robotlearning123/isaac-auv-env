"""Newton 1.2.0 SolverImplicitMPM fluid simulation benchmark.

Scenarios:
  - Dam break: fluid column collapsing under gravity
  - Drop splash: fluid sphere dropping onto ground plane

Particle counts: 1K, 10K, 50K, 100K

Measures: steps/sec, GPU memory, time per step.
Runs on RTX 5090 (CUDA 12.8+).

API reference: newton/examples/mpm/example_mpm_granular.py, example_mpm_viscous.py
"""

import subprocess
import time

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

wp.init()


def get_gpu_memory_mb():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip().split("\n")[0])


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def build_dam_break(builder, num_particles, voxel_size, density=1000.0):
    """Column of fluid particles (dam break)."""
    # Compute grid dimensions from target particle count
    side = int(np.ceil(num_particles ** (1.0 / 3.0)))
    dim_x = side
    dim_y = side
    dim_z = side

    cell_size = voxel_size / 3.0  # ~3 particles per cell
    cell_volume = cell_size ** 3
    mass = cell_volume * density
    radius = cell_size * 0.5

    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=cell_size,
        cell_y=cell_size,
        cell_z=cell_size,
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )
    actual = builder.particle_count
    return actual


def build_drop_splash(builder, num_particles, voxel_size, density=1000.0):
    """Sphere of fluid particles dropping from height."""
    cell_size = voxel_size / 3.0
    cell_volume = cell_size ** 3
    mass = cell_volume * density
    radius = cell_size * 0.5

    sphere_radius = (num_particles * cell_volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
    center = np.array([0.0, 0.0, sphere_radius + 1.0])

    # Generate random points in sphere
    rng = np.random.default_rng(42)
    points = []
    while len(points) < num_particles:
        batch = rng.uniform(-1, 1, (num_particles * 3, 3))
        batch = batch[np.linalg.norm(batch, axis=1) <= 1.0]
        for p in batch:
            if len(points) >= num_particles:
                break
            points.append(center + p * sphere_radius)

    points = np.array(points[:num_particles], dtype=np.float32)
    # Jitter
    points += (rng.random(points.shape) - 0.5) * 2.0 * radius

    builder.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[mass] * len(points),
        radius=[radius] * len(points),
    )
    actual = builder.particle_count
    return actual


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(scenario_name, build_fn, num_particles, voxel_size, num_steps=50):
    """Build scene, run solver, measure performance."""
    print(f"\n{'='*70}")
    print(f"  {scenario_name} | target={num_particles} particles")
    print(f"{'='*70}")

    mem_before = get_gpu_memory_mb()

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    SolverImplicitMPM.register_custom_attributes(builder)

    actual_particles = build_fn(builder, num_particles, voxel_size)
    print(f"  Actual particles: {actual_particles}")

    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

    model = builder.finalize()
    model.set_gravity([0.0, 0.0, -9.81])

    # Fluid-like material: low friction, moderate viscosity, low yield
    model.mpm.viscosity.fill_(50.0)
    model.mpm.friction.fill_(0.0)
    model.mpm.tensile_yield_ratio.fill_(1.0)
    model.mpm.young_modulus.fill_(1.0e6)
    model.mpm.yield_pressure.fill_(1.0e4)

    config = SolverImplicitMPM.Config()
    config.voxel_size = voxel_size
    config.tolerance = 1.0e-4
    config.max_iterations = 100
    config.strain_basis = "P0"
    config.velocity_basis = "Q1"
    config.transfer_scheme = "apic"

    solver = SolverImplicitMPM(model, config)
    state_0 = model.state()
    state_1 = model.state()

    # Warmup (JIT compile)
    print("  Warmup (JIT compile + 3 steps)...")
    t0 = time.perf_counter()
    for _ in range(3):
        solver.step(state_0, state_1, None, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    wp.synchronize_device()
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup: {warmup_time:.2f}s")

    mem_after_init = get_gpu_memory_mb()

    # Timed run
    dt = 1.0 / 240.0
    print(f"  Running {num_steps} steps (dt={dt:.4f}s)...")
    t0 = time.perf_counter()
    for i in range(num_steps):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0
    wp.synchronize_device()
    elapsed = time.perf_counter() - t0

    mem_after_run = get_gpu_memory_mb()

    steps_per_sec = num_steps / elapsed
    ms_per_step = (elapsed / num_steps) * 1000.0
    mem_used = mem_after_run - mem_before

    print(f"\n  RESULTS:")
    print(f"    Particles:         {actual_particles}")
    print(f"    Voxel size:        {voxel_size}")
    print(f"    Steps:             {num_steps}")
    print(f"    Total time:        {elapsed:.3f}s")
    print(f"    Steps/sec:         {steps_per_sec:.1f}")
    print(f"    ms/step:           {ms_per_step:.2f}")
    print(f"    GPU memory delta:  {mem_used:.0f} MB")
    print(f"    GPU memory total:  {mem_after_run:.0f} MB")

    # Check particles didn't explode
    positions = state_0.particle_q.numpy()
    max_z = np.max(positions[:, 2])
    min_z = np.min(positions[:, 2])
    mean_z = np.mean(positions[:, 2])
    print(f"    Position range Z:  [{min_z:.3f}, {max_z:.3f}] (mean={mean_z:.3f})")

    # Cleanup
    del solver, state_0, state_1, model
    wp.synchronize_device()

    return {
        "scenario": scenario_name,
        "target_particles": num_particles,
        "actual_particles": actual_particles,
        "voxel_size": voxel_size,
        "steps": num_steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps_per_sec,
        "ms_per_step": ms_per_step,
        "gpu_mem_delta_mb": mem_used,
        "gpu_mem_total_mb": mem_after_run,
        "warmup_s": warmup_time,
        "z_range": (min_z, max_z, mean_z),
    }


def main():
    print("Newton 1.2.0 SolverImplicitMPM Fluid Benchmark")
    print(f"Warp version: {wp.__version__}")
    print(f"Newton version: {newton.__version__}")
    device = wp.get_device()
    print(f"Device: {device.name} ({device.arch})")

    scenarios = [
        ("Dam Break", build_dam_break),
        ("Drop Splash", build_drop_splash),
    ]

    particle_counts = [1_000, 10_000, 50_000, 100_000]
    results = []

    for scenario_name, build_fn in scenarios:
        for n in particle_counts:
            # Scale voxel size with particle count for reasonable grid sizes
            if n <= 1_000:
                voxel_size = 0.1
            elif n <= 10_000:
                voxel_size = 0.05
            elif n <= 50_000:
                voxel_size = 0.03
            else:
                voxel_size = 0.02

            num_steps = 50
            try:
                result = run_benchmark(scenario_name, build_fn, n, voxel_size, num_steps)
                results.append(result)
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "scenario": scenario_name,
                    "target_particles": n,
                    "actual_particles": 0,
                    "error": str(e),
                })

    # Summary table
    print("\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Scenario':<15} {'Particles':>10} {'Voxel':>8} {'Steps/s':>10} {'ms/step':>10} {'GPU MB':>10} {'Status':>10}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"  {r['scenario']:<15} {r['target_particles']:>10} {'---':>8} {'---':>10} {'---':>10} {'---':>10} {'FAIL':>10}")
        else:
            print(
                f"  {r['scenario']:<15} {r['actual_particles']:>10} "
                f"{r['voxel_size']:>8.3f} {r['steps_per_sec']:>10.1f} "
                f"{r['ms_per_step']:>10.2f} {r['gpu_mem_delta_mb']:>10.0f} {'OK':>10}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
