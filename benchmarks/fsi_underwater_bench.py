"""Fluid-Structure Interaction benchmark for underwater robotics — RTX 5090.

Part 1: Rigid body (sphere) in fluid with hydrodynamic forces via Warp kernels.
  - Quadratic drag, added mass, buoyancy, restoring moment
  - Validates terminal velocity against analytical: v_t = sqrt(2mg / (rho * Cd * A))

Part 2: Batched AUV (box + 6 thrusters) multi-world simulation.
  - Tests 64, 256, 1024, 4096 parallel environments
  - Measures env-steps/sec, memory per env
"""

from __future__ import annotations

import time
import subprocess

import numpy as np
import warp as wp

wp.init()
device = wp.get_device("cuda:0")

import newton

print(f"Device: {device}")
print(f"Warp {wp.__version__}, Newton {newton.__version__}")
print("=" * 78)


def gpu_mem_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return float(out)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Warp kernels for hydrodynamic forces
# ---------------------------------------------------------------------------

@wp.kernel
def apply_hydro_forces(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    # hydro params
    rho: float,
    Cd: float,
    cross_section_area: float,
    volume: float,
    added_mass_coeff: float,
    g: float,
):
    """Apply drag, buoyancy, and added-mass forces to a submerged body.

    Convention: Z-up, gravity in -Z.
    Newton spatial_vector = (top=linear, bottom=angular).
      body_qd spatial_top = linear vel [vx,vy,vz], spatial_bottom = angular vel
      body_f  spatial_top = force [fx,fy,fz], spatial_bottom = torque
    """
    tid = wp.tid()
    m = body_mass[tid]
    twist = body_qd[tid]
    lin_vel = wp.spatial_top(twist)     # [vx, vy, vz]
    ang_vel = wp.spatial_bottom(twist)  # [wx, wy, wz]

    speed_sq = lin_vel[0] * lin_vel[0] + lin_vel[1] * lin_vel[1] + lin_vel[2] * lin_vel[2]
    speed = wp.sqrt(speed_sq)

    # Quadratic drag: F_drag = -0.5 * rho * Cd * A * |v| * v
    drag_coeff = -0.5 * rho * Cd * cross_section_area * speed
    f_drag = wp.vec3(drag_coeff * lin_vel[0], drag_coeff * lin_vel[1], drag_coeff * lin_vel[2])

    # Buoyancy: F_buoy = rho * g * V_displaced  (upward = +Z)
    f_buoy_z = rho * g * volume

    # Weight: F_weight = -m * g  (downward = -Z)
    f_weight_z = -m * g

    f_x = f_drag[0]
    f_y = f_drag[1]
    f_z = f_weight_z + f_buoy_z + f_drag[2]

    existing = body_f[tid]
    prev_f = wp.spatial_top(existing)
    prev_t = wp.spatial_bottom(existing)
    body_f[tid] = wp.spatial_vector(
        prev_f[0] + f_x, prev_f[1] + f_y, prev_f[2] + f_z,
        prev_t[0], prev_t[1], prev_t[2],
    )


@wp.kernel
def apply_hydro_forces_batched(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    rho: float,
    Cd: float,
    cross_section_area: float,
    volume: float,
    g: float,
    # per-world thruster forces (6DOF per env, laid out flat)
    thruster_forces: wp.array(dtype=float),
):
    """Hydro forces + thruster input for batched AUV simulation."""
    tid = wp.tid()

    m = body_mass[tid]

    twist = body_qd[tid]
    lin_vel = wp.spatial_top(twist)     # linear velocity

    speed_sq = lin_vel[0] * lin_vel[0] + lin_vel[1] * lin_vel[1] + lin_vel[2] * lin_vel[2]
    speed = wp.sqrt(speed_sq)

    drag_coeff = -0.5 * rho * Cd * cross_section_area * speed
    f_drag_x = drag_coeff * lin_vel[0]
    f_drag_y = drag_coeff * lin_vel[1]
    f_drag_z = drag_coeff * lin_vel[2]

    f_buoy_z = rho * g * volume
    f_weight_z = -m * g

    # Thruster: 6 floats per env -> forces [fx, fy, fz, tx, ty, tz]
    idx = tid * 6
    thr_fx = thruster_forces[idx + 0]
    thr_fy = thruster_forces[idx + 1]
    thr_fz = thruster_forces[idx + 2]
    thr_tx = thruster_forces[idx + 3]
    thr_ty = thruster_forces[idx + 4]
    thr_tz = thruster_forces[idx + 5]

    f_x = f_drag_x + thr_fx
    f_y = f_drag_y + thr_fy
    f_z = f_weight_z + f_buoy_z + f_drag_z + thr_fz

    existing = body_f[tid]
    prev_f = wp.spatial_top(existing)
    prev_t = wp.spatial_bottom(existing)
    body_f[tid] = wp.spatial_vector(
        prev_f[0] + f_x, prev_f[1] + f_y, prev_f[2] + f_z,
        prev_t[0] + thr_tx, prev_t[1] + thr_ty, prev_t[2] + thr_tz,
    )


# ---------------------------------------------------------------------------
# Part 1: Single rigid body — sphere sinking in water
# ---------------------------------------------------------------------------

def part1_terminal_velocity():
    """Test sphere sinking/rising. Validate terminal velocity analytically."""
    print("\n" + "=" * 78)
    print("Part 1: Rigid body in fluid — terminal velocity validation")
    print("=" * 78)

    # Physical params (steel sphere in water)
    rho_fluid = 1025.0  # seawater kg/m^3
    sphere_radius = 0.05  # 5 cm
    sphere_vol = (4.0 / 3.0) * np.pi * sphere_radius ** 3
    sphere_area = np.pi * sphere_radius ** 2  # projected cross-section
    Cd = 0.47  # sphere drag coefficient
    Ca = 0.5  # added mass coefficient for sphere
    g = 9.81

    # Test cases: different density ratios
    test_cases = [
        ("Steel (sinking)", 7800.0),
        ("Aluminum (sinking)", 2700.0),
        ("Light plastic (rising)", 500.0),
        ("Near-neutral buoyancy", 1020.0),
    ]

    dt = 1.0 / 240.0
    n_steps = 10000  # ~41.6 seconds of sim time

    results = []

    for name, rho_body in test_cases:
        mass = rho_body * sphere_vol
        added_mass = Ca * rho_fluid * sphere_vol

        # Analytical terminal velocity (accounting for buoyancy)
        net_weight = mass * g - rho_fluid * g * sphere_vol
        if abs(net_weight) < 1e-12:
            v_terminal_analytical = 0.0
        else:
            # v_t = sqrt(2 * |net_weight| / (rho * Cd * A))
            v_terminal_analytical = np.sqrt(
                2.0 * abs(net_weight) / (rho_fluid * Cd * sphere_area)
            )
            if net_weight > 0:
                v_terminal_analytical = -v_terminal_analytical  # sinking = negative z

        # Build Newton model
        builder = newton.ModelBuilder()
        builder.gravity = 0.0  # we handle gravity ourselves via hydro kernel
        shape_cfg = builder.ShapeConfig()
        shape_cfg.density = 0.0  # don't let shape add mass to body
        b = builder.add_body(mass=mass)
        builder.add_shape_sphere(b, radius=sphere_radius, cfg=shape_cfg)
        builder.add_joint_free(child=b)
        model = builder.finalize(device="cuda")

        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        # Run simulation
        wp.synchronize()
        for step in range(n_steps):
            # Clear forces, apply hydro
            state_in.clear_forces()
            wp.launch(
                apply_hydro_forces,
                dim=1,
                inputs=[
                    state_in.body_q, state_in.body_qd, state_in.body_f, model.body_mass,
                    rho_fluid, Cd, sphere_area, sphere_vol, Ca, g,
                ],
            )
            solver.step(state_in, state_out, control, None, dt)
            state_in, state_out = state_out, state_in

        wp.synchronize()

        # Read final velocity
        final_qd = state_in.body_qd.numpy()
        vz_final = final_qd[0, 2]  # linear z = index 2 (spatial_top = first 3)

        # Also track peak velocity (should converge to terminal)
        # Re-run with tracking
        state_in2 = model.state()
        state_out2 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in2)
        vz_track = []
        for step in range(n_steps):
            state_in2.clear_forces()
            wp.launch(
                apply_hydro_forces,
                dim=1,
                inputs=[
                    state_in2.body_q, state_in2.body_qd, state_in2.body_f, model.body_mass,
                    rho_fluid, Cd, sphere_area, sphere_vol, Ca, g,
                ],
            )
            solver.step(state_in2, state_out2, control, None, dt)
            state_in2, state_out2 = state_out2, state_in2
            if step % 100 == 0:
                qd_np = state_in2.body_qd.numpy()
                vz_track.append(qd_np[0, 2])  # linear z = index 2

        wp.synchronize()
        vz_last_10 = np.mean(vz_track[-10:])

        error_pct = abs(vz_last_10 - v_terminal_analytical) / max(abs(v_terminal_analytical), 1e-12) * 100.0

        print(f"\n  {name}:")
        print(f"    body density     = {rho_body:.0f} kg/m^3")
        print(f"    sphere mass      = {mass:.4f} kg")
        print(f"    sphere volume    = {sphere_vol:.6e} m^3")
        print(f"    net weight       = {net_weight:.4f} N")
        print(f"    analytical v_t   = {v_terminal_analytical:.6f} m/s")
        print(f"    simulated v_t    = {vz_last_10:.6f} m/s")
        print(f"    error            = {error_pct:.2f}%")
        print(f"    vz convergence   = {vz_track[:5]}...{vz_track[-5:]}")

        results.append({
            "name": name,
            "rho_body": rho_body,
            "mass": mass,
            "v_analytical": v_terminal_analytical,
            "v_simulated": vz_last_10,
            "error_pct": error_pct,
        })

    return results


# ---------------------------------------------------------------------------
# Part 2: Batched AUV simulation
# ---------------------------------------------------------------------------

def part2_batched_auv():
    """Batched multi-world AUV (box + 6 thrusters) benchmark."""
    print("\n" + "=" * 78)
    print("Part 2: Batched AUV multi-world simulation")
    print("=" * 78)

    # AUV params
    auv_lx, auv_ly, auv_lz = 0.5, 0.3, 0.2  # half-extents (1.0 x 0.6 x 0.4 m box)
    auv_mass = 50.0  # kg
    rho_fluid = 1025.0
    Cd = 1.0
    cross_area = 2.0 * auv_ly * 2.0 * auv_lz  # front-facing area
    auv_vol = (2.0 * auv_lx) * (2.0 * auv_ly) * (2.0 * auv_lz)
    g = 9.81
    dt = 1.0 / 240.0

    def build_template():
        t = newton.ModelBuilder()
        t.gravity = 0.0
        shape_cfg = t.ShapeConfig()
        shape_cfg.density = 0.0
        b = t.add_body(mass=auv_mass)
        t.add_shape_box(b, hx=auv_lx, hy=auv_ly, hz=auv_lz, cfg=shape_cfg)
        t.add_joint_free(child=b)
        return t

    sizes = [64, 256, 1024, 4096]
    n_steps_warmup = 10
    n_steps_bench = 200

    print(f"\nAUV: {2*auv_lx}x{2*auv_ly}x{2*auv_lz} m, {auv_mass} kg")
    print(f"Fluid: rho={rho_fluid}, Cd={Cd}, A={cross_area:.3f} m^2, V={auv_vol:.4f} m^3")
    print(f"dt={dt}, warmup={n_steps_warmup}, bench_steps={n_steps_bench}")
    print()
    print(f"{'N_envs':>8}  {'steps/s':>10}  {'ms/step':>8}  {'env-steps/s':>12}  {'M env/s':>10}  {'GPU MB':>8}  {'MB/env':>8}")
    print("-" * 78)

    results = []
    for n_worlds in sizes:
        try:
            template = build_template()
            scene = newton.ModelBuilder()
            scene.replicate(template, world_count=n_worlds, spacing=(0.0, 0.0, 0.0))
            model = scene.finalize(device="cuda")

            state_a, state_b = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)

            solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.05)

            # Generate per-env thruster forces (random currents)
            rng = np.random.default_rng(42)
            thruster_np = rng.normal(0, 5.0, size=(n_worlds, 6)).astype(np.float32)
            thruster_wp = wp.array(thruster_np.flatten(), dtype=float, device="cuda")

            mem_before = gpu_mem_mb()

            # Warmup
            for _ in range(n_steps_warmup):
                state_a.clear_forces()
                wp.launch(
                    apply_hydro_forces_batched,
                    dim=n_worlds,
                    inputs=[
                        state_a.body_q, state_a.body_qd, state_a.body_f, model.body_mass,
                        rho_fluid, Cd, cross_area, auv_vol, g,
                        thruster_wp,
                    ],
                )
                solver.step(state_a, state_b, control, None, dt)
                state_a, state_b = state_b, state_a
            wp.synchronize()

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(n_steps_bench):
                state_a.clear_forces()
                wp.launch(
                    apply_hydro_forces_batched,
                    dim=n_worlds,
                    inputs=[
                        state_a.body_q, state_a.body_qd, state_a.body_f, model.body_mass,
                        rho_fluid, Cd, cross_area, auv_vol, g,
                        thruster_wp,
                    ],
                )
                solver.step(state_a, state_b, control, None, dt)
                state_a, state_b = state_b, state_a
            wp.synchronize()
            wall = time.perf_counter() - t0

            mem_after = gpu_mem_mb()
            mem_delta = max(mem_after - mem_before, 0.0)
            mem_per_env = mem_delta / n_worlds if n_worlds > 0 else 0.0

            step_rate = n_steps_bench / wall
            env_rate = step_rate * n_worlds
            ms_per_step = wall / n_steps_bench * 1000.0

            print(
                f"{n_worlds:>8}  {step_rate:>10.1f}  {ms_per_step:>8.2f}  "
                f"{env_rate:>12.0f}  {env_rate/1e6:>10.3f}  "
                f"{mem_delta:>8.1f}  {mem_per_env:>8.4f}"
            )
            results.append({
                "n_worlds": n_worlds,
                "step_rate": step_rate,
                "ms_per_step": ms_per_step,
                "env_rate": env_rate,
                "mem_delta_mb": mem_delta,
                "mem_per_env_kb": mem_per_env * 1024,
            })

            del model, state_a, state_b, solver, thruster_wp
        except Exception as e:
            print(f"{n_worlds:>8}  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"GPU memory at start: {gpu_mem_mb():.0f} MB")

    print("\n" + "#" * 78)
    print("# Part 1: Terminal velocity validation")
    print("#" * 78)
    part1_results = part1_terminal_velocity()

    print("\n" + "#" * 78)
    print("# Part 2: Batched AUV throughput")
    print("#" * 78)
    part2_results = part2_batched_auv()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("\nPart 1: Terminal Velocity Validation")
    for r in part1_results:
        print(f"  {r['name']:30s}  analytical={r['v_analytical']:+.6f}  "
              f"simulated={r['v_simulated']:+.6f}  error={r['error_pct']:.2f}%")

    print("\nPart 2: Batched AUV Throughput")
    for r in part2_results:
        print(f"  N={r['n_worlds']:>5}  steps/s={r['step_rate']:.0f}  "
              f"env-steps/s={r['env_rate']:.0f}  "
              f"mem={r['mem_delta_mb']:.1f} MB  per_env={r['mem_per_env_kb']:.1f} KB")

    print(f"\nGPU memory at end: {gpu_mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
