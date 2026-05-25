"""
Newton 1.2.0 Physics Solver Benchmark — RTX 5090
Tests: rigid body sim, contact detection, batched (multi-world) simulation, API survey

Notes on Newton 1.2.0 API (verified by inspection):
  - builder.gravity is a float (scalar), default -9.81
  - builder.up_vector default is (0,0,1) = Z-up
  - Gravity vector = up_vector * gravity_scalar => (0, 0, -9.81)
  - add_body(xform=wp.transform(p, q)) sets initial pose
  - Multi-world: begin_world() / add_builder(template) / end_world()
  - replicate is a @staticmethod taking (builder, world_count, spacing)
"""

import gc
import time
import traceback

import numpy as np
import warp as wp

wp.init()
device = wp.get_device("cuda:0")
print(f"Device: {device}")
print(f"Warp {wp.__version__}, CUDA available: {wp.is_cuda_available()}")

import newton  # noqa: E402

print(f"Newton {newton.__version__}")
print("=" * 70)


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


def build_sphere_scene(n_bodies):
    """Build a scene with n_bodies falling spheres, Z-up."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    # gravity = -9.81 (float), up_vector = (0,0,1) => gravity acts in -Z
    builder.gravity = -9.81

    cols = int(np.ceil(np.sqrt(n_bodies)))
    spacing = 2.0
    for i in range(n_bodies):
        row = i // cols
        col = i % cols
        x = col * spacing
        y = 0.0
        z = 1.0 + row * spacing
        body = builder.add_body(
            mass=1.0,
            xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
        )
        builder.add_shape_sphere(body, radius=0.5)
        builder.add_joint_free(child=body)

    return builder


# ── Section 1: Rigid body simulation — falling spheres ──────────────────
print("\n" + "=" * 70)
print("BENCHMARK 1: Rigid Body Simulation (falling spheres, SemiImplicit)")
print("=" * 70)

DT = 1.0 / 60.0
NUM_STEPS = 1000

for n_bodies in [100, 500, 1000, 5000]:
    gc.collect()
    try:
        builder = build_sphere_scene(n_bodies)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSemiImplicit(model)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()

        mem_before = gpu_mem_mb()

        # Warmup
        for _ in range(10):
            contacts = model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in

        # Timed run
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(NUM_STEPS):
            contacts = model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize_device(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        steps_per_sec = NUM_STEPS / elapsed
        mem_after = gpu_mem_mb()

        print(
            f"  N={n_bodies:>5}  |  "
            f"total={elapsed:.3f}s  |  "
            f"{steps_per_sec:.0f} steps/s  |  "
            f"GPU mem: {mem_before:.0f}->{mem_after:.0f} MB  |  "
            f"delta={mem_after - mem_before:.0f} MB"
        )

        del model, solver, state_in, state_out, control, contacts, builder
    except Exception as e:
        print(f"  N={n_bodies:>5}  |  FAILED: {e}")
        traceback.print_exc()


# ── Section 2: Contact detection — box stacks ───────────────────────────
print("\n" + "=" * 70)
print("BENCHMARK 2: Contact Detection (box stacks)")
print("=" * 70)

for stack_height in [5, 10, 20, 50]:
    gc.collect()
    try:
        builder = newton.ModelBuilder()
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
        model = builder.finalize(device=device)
        state = model.state()
        contacts = model.contacts()

        wp.synchronize_device(device)
        t0 = time.perf_counter()
        contacts = model.collide(state, contacts)
        wp.synchronize_device(device)
        t1 = time.perf_counter()

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
            f"collide={t1 - t0:.4f}s  |  "
            f"contacts={n_contacts}"
        )

        del model, state, contacts, builder
    except Exception as e:
        print(f"  stack={stack_height:>2}  |  FAILED: {e}")
        traceback.print_exc()


# ── Section 3: Batched simulation (multi-world via begin_world/add_builder) ─
print("\n" + "=" * 70)
print("BENCHMARK 3: Batched Simulation (multi-world, SemiImplicit)")
print("=" * 70)

BODIES_PER_WORLD = 20


def build_template():
    """Build a single-world template with BODIES_PER_WORLD spheres."""
    template = newton.ModelBuilder()
    template.gravity = -9.81
    for i in range(BODIES_PER_WORLD):
        row = i // 5
        col = i % 5
        body = template.add_body(
            mass=1.0,
            xform=wp.transform(
                wp.vec3(col * 1.0, 0.0, 1.0 + row * 1.0),
                wp.quat_identity(),
            ),
        )
        template.add_shape_sphere(body, radius=0.3)
        template.add_joint_free(child=body)
    return template


def build_multiworld_scene(num_worlds, include_ground=True):
    """Build multi-world scene using begin_world / add_builder / end_world."""
    builder = newton.ModelBuilder()
    builder.gravity = -9.81
    template = build_template()

    for w in range(num_worlds):
        builder.begin_world(label=f"world_{w}")
        builder.add_builder(template)
        if include_ground:
            builder.add_ground_plane()
        builder.end_world()

    return builder


for num_worlds in [1, 16, 64, 256]:
    gc.collect()
    try:
        builder = build_multiworld_scene(num_worlds)
        model = builder.finalize(
            device=device,
            skip_validation_worlds=True,
            skip_all_validations=True,
        )
        solver = newton.solvers.SolverSemiImplicit(model)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()

        total_bodies = BODIES_PER_WORLD * num_worlds

        # Warmup
        for _ in range(5):
            contacts = model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in

        NUM_BATCH_STEPS = 500
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(NUM_BATCH_STEPS):
            contacts = model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize_device(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        steps_per_sec = NUM_BATCH_STEPS / elapsed
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


# ── Section 3b: Batched simulation with MuJoCo solver ───────────────────
print("\n" + "=" * 70)
print("BENCHMARK 3b: Batched Simulation (multi-world, MuJoCo solver)")
print("=" * 70)

for num_worlds in [1, 16, 64, 256]:
    gc.collect()
    try:
        builder = build_multiworld_scene(num_worlds)
        model = builder.finalize(
            device=device,
            skip_validation_worlds=True,
            skip_all_validations=True,
        )
        solver = newton.solvers.SolverMuJoCo(model)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()

        total_bodies = BODIES_PER_WORLD * num_worlds

        # Warmup
        for _ in range(5):
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in

        NUM_BATCH_STEPS = 500
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(NUM_BATCH_STEPS):
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize_device(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        steps_per_sec = NUM_BATCH_STEPS / elapsed
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


# ── Section 4: API Survey ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("API SURVEY: Newton 1.2.0 available modules, classes, solvers")
print("=" * 70)

print("\nTop-level classes:")
for name in sorted(newton.__all__):
    obj = getattr(newton, name)
    kind = type(obj).__name__
    if isinstance(obj, type):
        print(f"  {name} (class)")
    else:
        print(f"  {name} ({kind})")

print("\nSolvers:")
for name in [
    "SolverSemiImplicit",
    "SolverXPBD",
    "SolverVBD",
    "SolverMuJoCo",
    "SolverFeatherstone",
    "SolverKamino",
    "SolverImplicitMPM",
    "SolverStyle3D",
]:
    cls = getattr(newton.solvers, name, None)
    if cls is not None:
        doc = cls.__doc__.split("\n")[0].strip() if cls.__doc__ else "no doc"
        print(f"  {name}: {doc[:120]}")

print("\nSubmodules:")
for mod_name in [
    "geometry",
    "math",
    "selection",
    "sensors",
    "usd",
    "utils",
    "viewer",
    "actuators",
    "ik",
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

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
