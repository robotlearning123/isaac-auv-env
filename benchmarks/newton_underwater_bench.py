"""
Newton Underwater Benchmark — RTX 5090
=======================================
Consolidated from newton_solver_bench.py + newton_bridge_bench.py.

Scenarios:
  rigid         — Falling spheres (SemiImplicit), sweeps body count
  contact       — Box stack contact detection, sweeps stack height
  multiworld    — Batched multi-world (SemiImplicit + MuJoCo), sweeps world count
  underwater    — Single sphere underwater dynamics (buoyancy + drag + added mass)
  fsi           — Multi-body fluid-structure coupling in turbulent flow
  solver_compare — SemiImplicit vs XPBD vs MuJoCo with hydro forces
  api_survey    — Newton 1.x API introspection
  all           — Run everything

Usage:
  uv run python benchmarks/newton_underwater_bench.py --scenario all
  uv run python benchmarks/newton_underwater_bench.py --scenario rigid
  uv run python benchmarks/newton_underwater_bench.py --scenario underwater,fsi
"""

import argparse
import gc
import sys
import time
import traceback

import numpy as np
import warp as wp

wp.init()
DEVICE = wp.get_device("cuda:0")

import newton
from newton import ModelBuilder

RHO_WATER = 1000.0
GRAVITY = 9.81
DT_RIGID = 1.0 / 60.0
DT_HYDRO = 0.01


# ── GPU memory helper ────────────────────────────────────────────────────


def gpu_mem_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return float(out)
    except Exception:
        return -1.0


# ── Warp hydro kernels ───────────────────────────────────────────────────


@wp.kernel
def compute_hydro_forces_sphere(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_qd_prev: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    body_count: int,
    radius: float,
    mass: float,
    rho: float,
    g: float,
    dt: float,
    Cd: float,
    Ca: float,
):
    i = wp.tid()
    if i >= body_count:
        return

    t = body_q[i]
    pos = wp.transform_get_translation(t)
    vz = pos[2]

    vel = body_qd[i]
    lin_vel = wp.vec3f(vel[0], vel[1], vel[2])

    V_sphere = (4.0 / 3.0) * 3.14159265 * radius * radius * radius
    A_cross = 3.14159265 * radius * radius

    if vz - radius >= 0.0:
        V_sub = 0.0
    elif vz + radius <= 0.0:
        V_sub = V_sphere
    else:
        cap_h = radius - vz
        V_sub = 3.14159265 * cap_h * cap_h * (3.0 * radius - cap_h) / 3.0

    F_buoy = rho * g * V_sub

    speed = wp.length(lin_vel)
    F_drag = wp.vec3f(0.0)
    if speed > 1e-8:
        drag_mag = 0.5 * rho * Cd * A_cross * speed * speed
        F_drag = -drag_mag * (lin_vel / speed)

    tau_am = 0.5
    F_added = -Ca * rho * V_sphere * lin_vel / tau_am

    fx = F_drag[0] + F_added[0]
    fy = F_drag[1] + F_added[1]
    fz = F_buoy + F_drag[2] + F_added[2]

    body_f[i] = wp.spatial_vectorf(fx, fy, fz, 0.0, 0.0, 0.0)


@wp.kernel
def compute_hydro_forces_box(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_qd_prev: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    body_count: int,
    hx: float,
    hy: float,
    hz: float,
    mass: float,
    rho: float,
    g: float,
    dt: float,
    Cd: float,
    Ca: float,
    current_vx: float,
    current_vy: float,
    current_vz: float,
):
    i = wp.tid()
    if i >= body_count:
        return

    t = body_q[i]
    pos = wp.transform_get_translation(t)

    vel = body_qd[i]
    lin_vel = wp.vec3f(vel[0], vel[1], vel[2])

    V_box = hx * hy * hz * 8.0
    A_front = hy * hz * 4.0
    A_side = hx * hz * 4.0
    A_top = hx * hy * 4.0

    z_min = pos[2] - hz
    z_max = pos[2] + hz
    if z_max <= 0.0:
        V_sub = V_box
    elif z_min >= 0.0:
        V_sub = 0.0
    else:
        frac = -z_min / (z_max - z_min)
        V_sub = V_box * frac

    F_buoy = rho * g * V_sub

    current = wp.vec3f(current_vx, current_vy, current_vz)
    rel_vel = lin_vel - current
    speed = wp.length(rel_vel)

    F_drag = wp.vec3f(0.0)
    if speed > 1e-8:
        rel_dir = rel_vel / speed
        drag_x = 0.5 * rho * Cd * A_front * speed * speed * abs(rel_dir[0])
        drag_y = 0.5 * rho * Cd * A_side * speed * speed * abs(rel_dir[1])
        drag_z = 0.5 * rho * Cd * A_top * speed * speed * abs(rel_dir[2])
        F_drag = -wp.vec3f(
            drag_x * rel_dir[0],
            drag_y * rel_dir[1],
            drag_z * rel_dir[2],
        )

    tau_am = 0.5
    F_added = -Ca * rho * V_box * (lin_vel - current) / tau_am

    r_bg = wp.vec3f(0.0, 0.0, hz * 0.3)
    z_hat = wp.vec3f(0.0, 0.0, 1.0)
    restoring = F_buoy * wp.cross(r_bg, z_hat)

    fx = F_drag[0] + F_added[0]
    fy = F_drag[1] + F_added[1]
    fz = F_buoy + F_drag[2] + F_added[2]

    body_f[i] = wp.spatial_vectorf(fx, fy, fz, restoring[0], restoring[1], restoring[2])


@wp.kernel
def compute_turbulent_flow_forces(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    body_count: int,
    hx: float,
    hy: float,
    hz: float,
    mass: float,
    rho: float,
    g: float,
    Cd: float,
    base_current_vx: float,
    base_current_vy: float,
    turbulence_scale: float,
    seed: int,
):
    i = wp.tid()
    if i >= body_count:
        return

    t = body_q[i]
    pos = wp.transform_get_translation(t)

    vel = body_qd[i]
    lin_vel = wp.vec3f(vel[0], vel[1], vel[2])

    V_box = hx * hy * hz * 8.0
    V_sub = V_box

    F_buoy = rho * g * V_sub

    turb_x = turbulence_scale * wp.sin(float(seed) + pos[0] * 2.0 + pos[1] * 3.0)
    turb_y = turbulence_scale * wp.cos(float(seed) + pos[1] * 2.0 + pos[2] * 3.0)
    turb_z = turbulence_scale * 0.3 * wp.sin(float(seed) * 0.7 + pos[2] * 2.0)

    current = wp.vec3f(base_current_vx + turb_x, base_current_vy + turb_y, turb_z)
    rel_vel = lin_vel - current
    speed = wp.length(rel_vel)

    A_cross = hx * hz * 4.0
    F_drag = wp.vec3f(0.0)
    if speed > 1e-8:
        drag_mag = 0.5 * rho * Cd * A_cross * speed * speed
        F_drag = -drag_mag * (rel_vel / speed)

    fx = F_drag[0]
    fy = F_drag[1]
    fz = F_buoy + F_drag[2]

    body_f[i] = wp.spatial_vectorf(fx, fy, fz, 0.0, 0.0, 0.0)


@wp.kernel
def copy_qd(
    src: wp.array(dtype=wp.spatial_vectorf),
    dst: wp.array(dtype=wp.spatial_vectorf),
    count: int,
):
    i = wp.tid()
    if i >= count:
        return
    dst[i] = src[i]


# ── Scene builders ───────────────────────────────────────────────────────


def build_sphere_scene(n_bodies):
    builder = ModelBuilder()
    builder.add_ground_plane()
    builder.gravity = -9.81
    cols = int(np.ceil(np.sqrt(n_bodies)))
    spacing = 2.0
    for i in range(n_bodies):
        row = i // cols
        col = i % cols
        body = builder.add_body(
            mass=1.0,
            xform=wp.transform(wp.vec3(col * spacing, 0.0, 1.0 + row * spacing), wp.quat_identity()),
        )
        builder.add_shape_sphere(body, radius=0.5)
        builder.add_joint_free(child=body)
    return builder


def build_multiworld_template(bodies_per_world=20):
    template = ModelBuilder()
    template.gravity = -9.81
    for i in range(bodies_per_world):
        row = i // 5
        col = i % 5
        body = template.add_body(
            mass=1.0,
            xform=wp.transform(wp.vec3(col * 1.0, 0.0, 1.0 + row * 1.0), wp.quat_identity()),
        )
        template.add_shape_sphere(body, radius=0.3)
        template.add_joint_free(child=body)
    return template


def build_multiworld_scene(num_worlds, bodies_per_world=20):
    builder = ModelBuilder()
    builder.gravity = -9.81
    template = build_multiworld_template(bodies_per_world)
    for w in range(num_worlds):
        builder.begin_world(label=f"world_{w}")
        builder.add_builder(template)
        builder.add_ground_plane()
        builder.end_world()
    return builder


def build_auv_model():
    hx, hy, hz = 0.25, 0.15, 0.15
    mass = 10.0
    builder = ModelBuilder()
    builder.add_body(
        mass=mass,
        xform=wp.transform((0.0, 0.0, -1.0), wp.quat_identity()),
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(body=0, hx=hx, hy=hy, hz=hz, cfg=cfg)
    return builder, hx, hy, hz, mass


def build_multi_body_model(n_bodies, hx=0.1, hy=0.1, hz=0.1):
    mass = hx * hy * hz * 8.0 * RHO_WATER
    builder = ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    side = int(np.ceil(n_bodies ** (1.0 / 3.0)))
    idx = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if idx >= n_bodies:
                    break
                x = (ix - side / 2.0) * 1.0
                y = (iy - side / 2.0) * 1.0
                z = -2.0 + iz * 0.5
                builder.add_body(
                    mass=mass,
                    xform=wp.transform((x, y, z), wp.quat_identity()),
                )
                builder.add_shape_box(body=idx, hx=hx, hy=hy, hz=hz, cfg=cfg)
                idx += 1
    return builder.finalize(), hx, hy, hz, mass


def build_sphere_model(radius=0.1, mass=5.0):
    builder = ModelBuilder()
    builder.add_body(
        mass=mass,
        xform=wp.transform((0.0, 0.0, -1.0), wp.quat_identity()),
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_sphere(body=0, radius=radius, cfg=cfg)
    return builder.finalize()


# ── Scenario: rigid ──────────────────────────────────────────────────────


def bench_rigid():
    print("\n" + "=" * 70)
    print("SCENARIO: rigid — Falling spheres (SemiImplicit)")
    print("=" * 70)

    dt = DT_RIGID
    num_steps = 1000

    for n_bodies in [100, 500, 1000, 5000]:
        gc.collect()
        try:
            builder = build_sphere_scene(n_bodies)
            model = builder.finalize(device=DEVICE)
            solver = newton.solvers.SolverSemiImplicit(model)
            state_in = model.state()
            state_out = model.state()
            control = model.control()
            contacts = model.contacts()

            mem_before = gpu_mem_mb()

            for _ in range(10):
                contacts = model.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, dt)
                state_in, state_out = state_out, state_in

            wp.synchronize_device(DEVICE)
            t0 = time.perf_counter()
            for _ in range(num_steps):
                contacts = model.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, dt)
                state_in, state_out = state_out, state_in
            wp.synchronize_device(DEVICE)
            elapsed = t0 and time.perf_counter() - t0

            mem_after = gpu_mem_mb()
            print(
                f"  N={n_bodies:>5}  |  "
                f"total={elapsed:.3f}s  |  "
                f"{num_steps / elapsed:.0f} steps/s  |  "
                f"GPU mem: {mem_before:.0f}->{mem_after:.0f} MB  |  "
                f"delta={mem_after - mem_before:.0f} MB"
            )
            del model, solver, state_in, state_out, control, contacts, builder
        except Exception as e:
            print(f"  N={n_bodies:>5}  |  FAILED: {e}")
            traceback.print_exc()


# ── Scenario: contact ────────────────────────────────────────────────────


def bench_contact():
    print("\n" + "=" * 70)
    print("SCENARIO: contact — Box stack contact detection")
    print("=" * 70)

    for stack_height in [5, 10, 20, 50]:
        gc.collect()
        try:
            builder = ModelBuilder()
            builder.add_ground_plane()
            builder.gravity = -9.81

            for stack_idx in range(2):
                x_offset = stack_idx * 4.0
                for layer in range(stack_height):
                    body = builder.add_body(
                        mass=1.0,
                        xform=wp.transform(
                            wp.vec3(x_offset, 0.0, 0.5 + layer * 1.0),
                            wp.quat_identity(),
                        ),
                    )
                    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
                    builder.add_joint_free(child=body)

            n_bodies = stack_height * 2
            model = builder.finalize(device=DEVICE)
            state = model.state()
            contacts = model.contacts()

            wp.synchronize_device(DEVICE)
            t0 = time.perf_counter()
            contacts = model.collide(state, contacts)
            wp.synchronize_device(DEVICE)
            elapsed = time.perf_counter() - t0

            n_contacts = 0
            try:
                rc = contacts.rigid_contact_count
                if hasattr(rc, "numpy"):
                    arr = rc.numpy()
                else:
                    arr = rc
                n_contacts = int(arr[0]) if hasattr(arr, "__getitem__") else int(arr)
            except Exception:
                n_contacts = -1

            print(
                f"  stack={stack_height:>2} (bodies={n_bodies:>3})  |  "
                f"collide={elapsed:.4f}s  |  "
                f"contacts={n_contacts}"
            )
            del model, state, contacts, builder
        except Exception as e:
            print(f"  stack={stack_height:>2}  |  FAILED: {e}")
            traceback.print_exc()


# ── Scenario: multiworld ─────────────────────────────────────────────────


def bench_multiworld():
    print("\n" + "=" * 70)
    print("SCENARIO: multiworld — Batched simulation (SemiImplicit + MuJoCo)")
    print("=" * 70)

    dt = DT_RIGID
    bodies_per_world = 20
    num_steps = 500

    for solver_name in ["SemiImplicit", "MuJoCo"]:
        print(f"\n  --- {solver_name} ---")
        for num_worlds in [1, 16, 64, 256]:
            gc.collect()
            try:
                builder = build_multiworld_scene(num_worlds, bodies_per_world)
                model = builder.finalize(
                    device=DEVICE,
                    skip_validation_worlds=True,
                    skip_all_validations=True,
                )
                if solver_name == "SemiImplicit":
                    solver = newton.solvers.SolverSemiImplicit(model)
                else:
                    solver = newton.solvers.SolverMuJoCo(model)

                state_in = model.state()
                state_out = model.state()
                control = model.control()
                contacts = model.contacts()
                total_bodies = bodies_per_world * num_worlds

                for _ in range(5):
                    if solver_name == "SemiImplicit":
                        contacts = model.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, dt)
                    state_in, state_out = state_out, state_in

                wp.synchronize_device(DEVICE)
                t0 = time.perf_counter()
                for _ in range(num_steps):
                    if solver_name == "SemiImplicit":
                        contacts = model.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, dt)
                    state_in, state_out = state_out, state_in
                wp.synchronize_device(DEVICE)
                elapsed = time.perf_counter() - t0

                steps_per_sec = num_steps / elapsed
                envs_per_sec = steps_per_sec * num_worlds
                mem = gpu_mem_mb()

                print(
                    f"  worlds={num_worlds:>3}  bodies={total_bodies:>5}  |  "
                    f"total={elapsed:.3f}s  |  "
                    f"{steps_per_sec:.0f} steps/s  |  "
                    f"{envs_per_sec:.0f} envs/s  |  "
                    f"GPU mem={mem:.0f} MB"
                )
                del model, solver, state_in, state_out, control, contacts, builder
            except Exception as e:
                print(f"  worlds={num_worlds:>3}  |  FAILED: {e}")
                traceback.print_exc()


# ── Scenario: underwater ──────────────────────────────────────────────────


def bench_underwater():
    print("\n" + "=" * 70)
    print("SCENARIO: underwater — Single sphere dynamics (buoyancy + drag + added mass)")
    print("=" * 70)

    radius = 0.1
    V_sphere = (4.0 / 3.0) * np.pi * radius ** 3
    mass = 500.0 * V_sphere
    density = mass / V_sphere
    Cd = 1.0
    Ca = 0.3
    n_steps = 1000
    dt = DT_HYDRO

    print(f"  Sphere: r={radius}m, m={mass:.4f}kg, V={V_sphere:.6f}m^3, rho_body={density:.1f}kg/m^3")
    print(f"  Water: rho={RHO_WATER}kg/m^3, density ratio={density/RHO_WATER:.2f}")
    print(f"  Sim: {n_steps} steps, dt={dt}s, total={n_steps*dt:.1f}s")

    model = build_sphere_model(radius, mass)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverSemiImplicit(model)
    contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=0)
    qd_prev = wp.clone(state_0.body_qd)

    pos_history = np.zeros((n_steps, 3))
    vel_history = np.zeros((n_steps, 6))
    force_history = np.zeros((n_steps, 6))

    wp.synchronize()
    t_start = time.perf_counter()

    for step in range(n_steps):
        state_0.clear_forces()
        wp.launch(
            compute_hydro_forces_sphere,
            dim=model.body_count,
            inputs=[
                state_0.body_q, state_0.body_qd, qd_prev, state_0.body_f,
                model.body_count, radius, mass, RHO_WATER, GRAVITY, dt, Cd, Ca,
            ],
        )
        solver.step(state_0, state_1, control, contacts, dt)

        q_np = state_1.body_q.numpy()
        qd_np = state_1.body_qd.numpy()
        f_np = state_0.body_f.numpy()
        pos_history[step] = q_np[0, :3]
        vel_history[step] = qd_np[0]
        force_history[step] = f_np[0]

        wp.launch(copy_qd, dim=model.body_count, inputs=[state_0.body_qd, qd_prev, model.body_count])
        state_0, state_1 = state_1, state_0

    wp.synchronize()
    elapsed = time.perf_counter() - t_start

    A_cross = np.pi * radius ** 2
    net_upward = RHO_WATER * GRAVITY * V_sphere - mass * GRAVITY
    if net_upward > 0:
        v_terminal = np.sqrt(2.0 * net_upward / (RHO_WATER * Cd * A_cross))
        print(f"  Buoyant body, net upward: {net_upward:.4f} N, terminal rise vel: {v_terminal:.4f} m/s")
    else:
        print(f"  Negatively buoyant, net downward: {-net_upward:.4f} N")

    final_pos = pos_history[-1]
    final_vel = vel_history[-1]
    eq_ratio = density / RHO_WATER

    print(f"  Wall: {elapsed:.3f}s, Steps/s: {n_steps / elapsed:.1f}")
    print(f"  Final pos: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"  Final vel_z: {final_vel[2]:.6f} m/s")
    print(f"  Z trajectory: start={pos_history[0, 2]:.4f}, min={pos_history[:, 2].min():.4f}, "
          f"max={pos_history[:, 2].max():.4f}, final={final_pos[2]:.4f}")
    print(f"  Equilibrium submersion ratio: {eq_ratio:.2f} (expected ~0.5)")

    print()
    for t_idx in [0, 50, 100, 250, 500, 750, 999]:
        z = pos_history[t_idx, 2]
        vz = vel_history[t_idx, 2]
        fz = force_history[t_idx, 2]
        print(f"  t={t_idx*dt:6.2f}s  z={z:8.4f}m  vz={vz:8.4f}m/s  Fz={fz:8.3f}N")


# ── Scenario: fsi ─────────────────────────────────────────────────────────


def bench_fsi():
    print("\n" + "=" * 70)
    print("SCENARIO: fsi — Fluid-structure coupling (turbulent flow)")
    print("=" * 70)

    Cd = 1.0
    base_current_vx = 0.5
    base_current_vy = 0.2
    turbulence_scale = 0.3
    dt = DT_HYDRO
    n_steps = 100

    results = []
    for n_bodies in [10, 100, 1000]:
        print(f"\n  --- N={n_bodies} bodies ---")
        hx, hy, hz = 0.1, 0.1, 0.1
        model, hx, hy, hz, mass = build_multi_body_model(n_bodies, hx, hy, hz)
        body_count = model.body_count
        print(f"    Actual bodies: {body_count}")

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver = newton.solvers.SolverSemiImplicit(model)
        contacts = newton.Contacts(rigid_contact_max=n_bodies * 4, soft_contact_max=0)
        collision_pipeline = newton.CollisionPipeline(model)

        for _ in range(5):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()

        t_start = time.perf_counter()
        for step in range(n_steps):
            state_0.clear_forces()
            wp.launch(
                compute_turbulent_flow_forces,
                dim=body_count,
                inputs=[
                    state_0.body_q, state_0.body_qd, state_0.body_f,
                    body_count, hx, hy, hz, mass,
                    RHO_WATER, GRAVITY, Cd,
                    base_current_vx, base_current_vy, turbulence_scale, step,
                ],
            )
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        elapsed = time.perf_counter() - t_start

        steps_per_sec = n_steps / elapsed
        print(f"    Wall: {elapsed:.3f}s, Steps/s: {steps_per_sec:.1f}")
        results.append({"n_bodies": body_count, "steps_per_sec": steps_per_sec, "elapsed": elapsed})

    print("\n  --- Scaling Summary ---")
    print(f"  {'Bodies':>8} {'Steps/s':>10} {'Wall(s)':>10}")
    for r in results:
        print(f"  {r['n_bodies']:>8} {r['steps_per_sec']:>10.1f} {r['elapsed']:>10.3f}")


# ── Scenario: solver_compare ──────────────────────────────────────────────


def bench_solver_compare():
    print("\n" + "=" * 70)
    print("SCENARIO: solver_compare — SemiImplicit vs XPBD vs MuJoCo (256 worlds)")
    print("=" * 70)

    n_worlds = 256
    auv_builder, hx, hy, hz, mass = build_auv_model()
    dt = DT_HYDRO
    Cd = 1.2
    Ca = 0.5

    scene = ModelBuilder()
    scene.add_ground_plane()
    for i in range(n_worlds):
        offset_x = (i % 16) * 2.0
        offset_y = (i // 16) * 2.0
        scene.add_world(
            auv_builder,
            xform=wp.transform((offset_x, offset_y, 0.0), wp.quat_identity()),
        )
    model = scene.finalize()
    body_count = model.body_count
    print(f"  Config: {n_worlds} worlds, {body_count} bodies")

    solver_configs = [
        ("SemiImplicit", lambda m: newton.solvers.SolverSemiImplicit(m), 1000),
        ("XPBD", lambda m: newton.solvers.SolverXPBD(m, iterations=4), 1000),
        ("MuJoCo", lambda m: newton.solvers.SolverMuJoCo(m), 200),
    ]

    results = {}
    for solver_name, make_solver, n_steps in solver_configs:
        print(f"\n  --- {solver_name} ({n_steps} steps) ---")
        try:
            solver = make_solver(model)
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=0)
            qd_prev = wp.zeros(body_count, dtype=wp.spatial_vectorf)

            for _ in range(10):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0
            wp.synchronize()

            state_0 = model.state()
            state_1 = model.state()

            t_start = time.perf_counter()
            for step in range(n_steps):
                state_0.clear_forces()
                wp.launch(
                    compute_hydro_forces_box,
                    dim=body_count,
                    inputs=[
                        state_0.body_q, state_0.body_qd, qd_prev, state_0.body_f,
                        body_count, hx, hy, hz, mass,
                        RHO_WATER, GRAVITY, dt, Cd, Ca,
                        0.0, 0.0, 0.0,
                    ],
                )
                solver.step(state_0, state_1, control, contacts, dt)
                wp.launch(copy_qd, dim=body_count, inputs=[state_0.body_qd, qd_prev, body_count])
                state_0, state_1 = state_1, state_0
            wp.synchronize()
            elapsed = time.perf_counter() - t_start

            qd_final = state_0.body_qd.numpy()
            ke = float(0.5 * mass * np.sum(qd_final[:, :3] ** 2))
            q_final = state_0.body_q.numpy()
            pe = float(mass * GRAVITY * np.sum(q_final[:, :3]))

            results[solver_name] = {
                "elapsed": elapsed,
                "steps_per_sec": n_steps / elapsed,
                "KE": ke, "PE": pe, "total_energy": ke + pe,
            }
            print(f"    Wall: {elapsed:.3f}s, Steps/s: {n_steps / elapsed:.1f}")
            print(f"    KE={ke:.4f}, PE={pe:.4f}, Total={ke+pe:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[solver_name] = {"error": str(e)}

    print("\n  --- Summary ---")
    print(f"  {'Solver':>16} {'Steps/s':>10} {'Wall(s)':>10} {'KE':>12} {'PE':>12} {'Total':>12}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:>16} FAILED: {r['error']}")
        else:
            print(f"  {name:>16} {r['steps_per_sec']:>10.1f} {r['elapsed']:>10.3f} "
                  f"{r['KE']:>12.4f} {r['PE']:>12.4f} {r['total_energy']:>12.4f}")


# ── Scenario: api_survey ──────────────────────────────────────────────────


def bench_api_survey():
    print("\n" + "=" * 70)
    print("SCENARIO: api_survey — Newton API introspection")
    print("=" * 70)

    print("\nTop-level classes:")
    for name in sorted(newton.__all__):
        obj = getattr(newton, name)
        if isinstance(obj, type):
            print(f"  {name} (class)")
        else:
            print(f"  {name} ({type(obj).__name__})")

    print("\nSolvers:")
    for name in [
        "SolverSemiImplicit", "SolverXPBD", "SolverVBD", "SolverMuJoCo",
        "SolverFeatherstone", "SolverKamino", "SolverImplicitMPM", "SolverStyle3D",
    ]:
        cls = getattr(newton.solvers, name, None)
        if cls is not None:
            doc = cls.__doc__.split("\n")[0].strip() if cls.__doc__ else "no doc"
            print(f"  {name}: {doc[:120]}")

    print("\nSubmodules:")
    for mod_name in [
        "geometry", "math", "selection", "sensors", "usd",
        "utils", "viewer", "actuators", "ik",
    ]:
        mod = getattr(newton, mod_name, None)
        if mod is not None:
            items = [x for x in dir(mod) if not x.startswith("_")]
            print(f"  newton.{mod_name}: {len(items)} items — {', '.join(items[:12])}{'...' if len(items) > 12 else ''}")

    print("\nKey geometry types:")
    for t in ["GeoType", "ShapeFlags", "BodyFlags", "JointType"]:
        obj = getattr(newton, t, None)
        if obj is not None:
            members = [m for m in dir(obj) if not m.startswith("_")]
            print(f"  {t}: {members}")


# ── Main ──────────────────────────────────────────────────────────────────

ALL_SCENARIOS = ["rigid", "contact", "multiworld", "underwater", "fsi", "solver_compare", "api_survey"]

SCENARIO_FNS = {
    "rigid": bench_rigid,
    "contact": bench_contact,
    "multiworld": bench_multiworld,
    "underwater": bench_underwater,
    "fsi": bench_fsi,
    "solver_compare": bench_solver_compare,
    "api_survey": bench_api_survey,
}


def main():
    parser = argparse.ArgumentParser(description="Newton Underwater Benchmark — RTX 5090")
    parser.add_argument(
        "--scenario",
        default="all",
        help=f"Comma-separated list of scenarios to run. Options: {', '.join(ALL_SCENARIOS)}, all (default: all)",
    )
    args = parser.parse_args()

    print("Newton Underwater Benchmark — RTX 5090")
    print(f"Warp {wp.__version__}, Newton {newton.__version__}")
    print(f"Device: {DEVICE}")
    print(f"Python {sys.version}")
    print("=" * 70)

    if args.scenario == "all":
        scenarios = ALL_SCENARIOS
    else:
        scenarios = [s.strip() for s in args.scenario.split(",")]
        for s in scenarios:
            if s not in SCENARIO_FNS:
                parser.error(f"Unknown scenario '{s}'. Choose from: {', '.join(ALL_SCENARIOS)}, all")

    t_total = time.perf_counter()
    for scenario in scenarios:
        SCENARIO_FNS[scenario]()
    total = time.perf_counter() - t_total

    print()
    print("=" * 70)
    print(f"TOTAL BENCHMARK TIME: {total:.2f}s (scenarios: {', '.join(scenarios)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
