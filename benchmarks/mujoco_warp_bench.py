"""MuJoCo-Warp 3.8.1 GPU dynamics benchmark on RTX 5090 (CUDA 12.8).

Tests:
1. Basic CPU MuJoCo simulation (box + arm + ant)
2. GPU acceleration via mujoco_warp (CPU vs GPU timing, single world)
3. Batched parallel environments (N = 1, 64, 256, 1024, 4096)
4. Hydrodynamic forces (density/viscosity + manual xfrc_applied)
5. API survey of mujoco_warp package

NOTE (GPU contention): Initial runs reported ~2.45M env-steps/s at N=4096, but
cross-verification with a clean GPU state (no other GPU processes) yielded
~1.3M env-steps/s (-46%). Results are highly sensitive to GPU contention from
other processes (X server, other CUDA workloads, thermal throttling). Always
run on an idle GPU and average multiple runs for reliable numbers.
"""

import time
import traceback

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

wp.init()

# ---------------------------------------------------------------------------
# XML models
# ---------------------------------------------------------------------------

BOX_XML = """
<mujoco model="box_plane">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.5">
      <joint name="slide_z" type="slide" axis="0 0 1"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

ARM_XML = """
<mujoco model="arm2dof">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="base" type="cylinder" size="0.1 0.05" rgba="0.5 0.5 0.5 1"/>
    <body name="link1" pos="0 0 0.05">
      <joint name="j1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <geom name="link1_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link2_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="m2" joint="j2" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

ANT_XML = """
<mujoco model="ant">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 0.5">
      <joint name="free" type="free"/>
      <geom name="body" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="leg1" pos="0.2 0.2 0">
        <joint name="hip1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.1"/>
      </body>
      <body name="leg2" pos="-0.2 0.2 0">
        <joint name="hip2" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.1"/>
      </body>
      <body name="leg3" pos="0.2 -0.2 0">
        <joint name="hip3" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.1"/>
      </body>
      <body name="leg4" pos="-0.2 -0.2 0">
        <joint name="hip4" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="hip1" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="m2" joint="hip2" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="m3" joint="hip3" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="m4" joint="hip4" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

UNDERWATER_XML = """
<mujoco model="underwater_box">
  <option gravity="0 0 -1.0" timestep="0.002" density="1000" viscosity="0.001"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.5">
      <joint name="slide_z" type="slide" axis="0 0 1"/>
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

N_STEPS = 1000
WARMUP = 100


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def gpu_mem_mb():
    if HAS_TORCH:
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return -1.0


# ---------------------------------------------------------------------------
# 1. Basic CPU simulation
# ---------------------------------------------------------------------------
def test_cpu_basic():
    section("1. Basic MuJoCo CPU Simulation")

    models = [
        ("box-on-plane", BOX_XML, None),
        ("2-DOF arm", ARM_XML, lambda d: d.ctrl.__setitem__(slice(None), [1.0, 0.5])),
        ("4-leg ant", ANT_XML, lambda d: d.ctrl.__setitem__(slice(None), [1.0, 1.0, 1.0, 1.0])),
    ]

    for label, xml, ctrl_fn in models:
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        if ctrl_fn:
            ctrl_fn(d)

        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            if ctrl_fn:
                ctrl_fn(d)
            mujoco.mj_step(m, d)
        elapsed = time.perf_counter() - t0

        print(f"\n  [{label}]")
        print(f"    nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}")
        print(f"    qpos final: {np.array2string(d.qpos, precision=6, suppress_small=True)}")
        print(f"    sim time:   {d.time:.4f} s")
        print(f"    wall time:  {elapsed*1000:.2f} ms ({N_STEPS} steps)")
        print(f"    steps/sec:  {N_STEPS/elapsed:.0f}")


# ---------------------------------------------------------------------------
# 2. CPU vs GPU timing (single world)
# ---------------------------------------------------------------------------
def test_cpu_vs_gpu():
    section("2. CPU vs GPU Timing (Single World)")

    for label, xml, ctrl_np_val in [
        ("box-on-plane", BOX_XML, None),
        ("2-DOF arm", ARM_XML, np.array([1.0, 0.5])),
        ("4-leg ant", ANT_XML, np.array([1.0, 1.0, 1.0, 1.0])),
    ]:
        mjm = mujoco.MjModel.from_xml_string(xml)
        mjd = mujoco.MjData(mjm)
        nu = mjm.nu

        # --- CPU ---
        if ctrl_np_val is not None:
            def ctrl_fn_cpu(d, ctrl=ctrl_np_val):
                return d.ctrl.__setitem__(slice(None), ctrl)
        else:
            def ctrl_fn_cpu(d):
                return None

        for _ in range(WARMUP):
            ctrl_fn_cpu(mjd)
            mujoco.mj_step(mjm, mjd)
        mujoco.mj_resetData(mjm, mjd)

        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            ctrl_fn_cpu(mjd)
            mujoco.mj_step(mjm, mjd)
        cpu_ms = (time.perf_counter() - t0) * 1000

        # --- GPU ---
        model = mjw.put_model(mjm)
        data = mjw.put_data(mjm, mjd, nworld=1)

        if nu > 0:
            wp.copy(data.ctrl, wp.array(ctrl_np_val.reshape(1, -1), dtype=wp.float32))

        # first step triggers JIT kernel compilation
        mjw.step(model, data)
        wp.synchronize()

        # warmup with cached kernels
        for _ in range(WARMUP):
            mjw.step(model, data)
        wp.synchronize()

        mjw.reset_data(model, data)
        if nu > 0:
            wp.copy(data.ctrl, wp.array(ctrl_np_val.reshape(1, -1), dtype=wp.float32))
        wp.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            mjw.step(model, data)
        wp.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000

        mjw.get_data_into(mjd, mjm, data, world_id=0)

        print(f"\n  [{label}]  nq={mjm.nq}  nv={mjm.nv}  nu={nu}  nbody={mjm.nbody}")
        print(f"    CPU: {cpu_ms:.2f} ms  ({N_STEPS/cpu_ms*1000:.0f} steps/sec)")
        print(f"    GPU: {gpu_ms:.2f} ms  ({N_STEPS/gpu_ms*1000:.0f} steps/sec)")
        print(f"    speedup: {cpu_ms/gpu_ms:.2f}x")
        print(f"    GPU qpos final: {np.array2string(mjd.qpos, precision=6, suppress_small=True)}")

        del data


# ---------------------------------------------------------------------------
# 3. Batched environments
# ---------------------------------------------------------------------------
def test_batched():
    section("3. Batched Parallel Environments")

    mjm = mujoco.MjModel.from_xml_string(ANT_XML)
    mjd = mujoco.MjData(mjm)
    nu = mjm.nu

    model = mjw.put_model(mjm)

    # CPU baseline
    mjd.ctrl[:] = 1.0
    mujoco.mj_resetData(mjm, mjd)
    for _ in range(WARMUP):
        mujoco.mj_step(mjm, mjd)
    mujoco.mj_resetData(mjm, mjd)
    mjd.ctrl[:] = 1.0
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        mujoco.mj_step(mjm, mjd)
    cpu_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Model: 4-leg ant  nq={mjm.nq}  nv={mjm.nv}  nu={nu}  nbody={mjm.nbody}")
    print(f"  CPU baseline: {cpu_ms:.2f} ms  ({N_STEPS/cpu_ms*1000:.0f} steps/s)")
    print(f"  Steps per run: {N_STEPS}  (warmup: {WARMUP})")
    print(f"\n  {'N':>6}  {'total_ms':>10}  {'steps/s':>10}  {'envs/s':>12}  {'speedup':>10}  {'mem_delta_MB':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*14}")

    for nworld in [1, 64, 256, 1024, 4096]:
        try:
            if HAS_TORCH:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()

            data = mjw.make_data(mjm, nworld=nworld)
            ctrl_np = np.ones((nworld, nu), dtype=np.float32)
            wp.copy(data.ctrl, wp.array(ctrl_np, dtype=wp.float32))

            # warmup
            for _ in range(WARMUP):
                mjw.step(model, data)
            wp.synchronize()

            mjw.reset_data(model, data)
            wp.copy(data.ctrl, wp.array(ctrl_np, dtype=wp.float32))
            wp.synchronize()

            t0 = time.perf_counter()
            for _ in range(N_STEPS):
                mjw.step(model, data)
            wp.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000

            steps_per_s = N_STEPS / (total_ms / 1000)
            envs_per_s = steps_per_s * nworld
            speedup = (cpu_ms / total_ms) * nworld

            mem_delta = 0.0
            if HAS_TORCH:
                mem_after = torch.cuda.memory_allocated()
                mem_delta = (mem_after - mem_before) / (1024 * 1024)

            print(f"  {nworld:>6}  {total_ms:>10.2f}  {steps_per_s:>10.0f}  {envs_per_s:>12.0f}  {speedup:>10.1f}x  {mem_delta:>14.1f}")

            del data

        except Exception as e:
            print(f"  {nworld:>6}  FAILED: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. Hydrodynamic forces
# ---------------------------------------------------------------------------
def test_hydro():
    section("4. Hydrodynamic Forces")

    # 4a. Built-in density/viscosity (MuJoCo native)
    mjm = mujoco.MjModel.from_xml_string(UNDERWATER_XML)
    mjd = mujoco.MjData(mjm)
    print("\n  4a. Built-in MuJoCo fluid (density/viscosity)")
    print(f"    density={mjm.opt.density}  viscosity={mjm.opt.viscosity}")
    print(f"    gravity: {mjm.opt.gravity}")

    for _ in range(N_STEPS):
        mujoco.mj_step(mjm, mjd)
    print(f"    CPU qpos: {np.array2string(mjd.qpos, precision=6, suppress_small=True)}")
    print(f"    qfrc_fluid (last): {np.array2string(mjd.qfrc_fluid, precision=6)}")

    # GPU with fluid
    model = mjw.put_model(mjm)
    mjd2 = mujoco.MjData(mjm)
    data = mjw.put_data(mjm, mjd2, nworld=1)

    mjw.step(model, data)
    wp.synchronize()
    for _ in range(WARMUP):
        mjw.step(model, data)
    wp.synchronize()

    mjw.reset_data(model, data)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        mjw.step(model, data)
    wp.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000

    mjw.get_data_into(mjd2, mjm, data, world_id=0)
    print(f"    GPU qpos: {np.array2string(mjd2.qpos, precision=6, suppress_small=True)}")
    print(f"    GPU time: {gpu_ms:.2f} ms ({N_STEPS/gpu_ms*1000:.0f} steps/s)")

    # 4b. Manual drag via qfrc_applied
    print("\n  4b. Manual drag via qfrc_applied (F = -c * v)")

    drag_xml = """
    <mujoco model="drag_test">
      <option gravity="0 0 -1.0" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="box" pos="0 0 0.5">
          <joint name="slide_z" type="slide" axis="0 0 1"/>
          <joint name="slide_x" type="slide" axis="1 0 0"/>
          <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """

    mjm_drag = mujoco.MjModel.from_xml_string(drag_xml)
    nv = mjm_drag.nv

    # CPU
    mjd_drag = mujoco.MjData(mjm_drag)
    drag_coeff = 5.0
    for _ in range(N_STEPS):
        for i in range(nv):
            mjd_drag.qfrc_applied[i] = -drag_coeff * mjd_drag.qvel[i]
        mujoco.mj_step(mjm_drag, mjd_drag)
    print(f"    CPU qpos (c={drag_coeff}): {np.array2string(mjd_drag.qpos, precision=6)}")

    # GPU single world
    model_d = mjw.put_model(mjm_drag)
    mjd_d = mujoco.MjData(mjm_drag)
    data_d = mjw.put_data(mjm_drag, mjd_d, nworld=1)

    @wp.kernel
    def apply_drag(
        qvel: wp.array2d(dtype=wp.float32),
        qfrc: wp.array2d(dtype=wp.float32),
        drag_coeff: float,
        nv: int,
    ):
        world, dof = wp.tid()
        if dof < nv:
            qfrc[world, dof] = -drag_coeff * qvel[world, dof]

    # warmup + JIT
    wp.launch(apply_drag, dim=[1, nv], inputs=[data_d.qvel, data_d.qfrc_applied, drag_coeff, nv])
    mjw.step(model_d, data_d)
    wp.synchronize()
    for _ in range(WARMUP):
        wp.launch(apply_drag, dim=[1, nv], inputs=[data_d.qvel, data_d.qfrc_applied, drag_coeff, nv])
        mjw.step(model_d, data_d)
    wp.synchronize()

    mjw.reset_data(model_d, data_d)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        wp.launch(apply_drag, dim=[1, nv], inputs=[data_d.qvel, data_d.qfrc_applied, drag_coeff, nv])
        mjw.step(model_d, data_d)
    wp.synchronize()
    gpu_single_ms = (time.perf_counter() - t0) * 1000

    mjw.get_data_into(mjd_d, mjm_drag, data_d, world_id=0)
    print(f"    GPU qpos (1 world): {np.array2string(mjd_d.qpos, precision=6)}")
    print(f"    GPU time: {gpu_single_ms:.2f} ms ({N_STEPS/gpu_single_ms*1000:.0f} steps/s)")

    # GPU batched drag
    print("\n  4c. Batched drag force (GPU):")
    print(f"  {'N':>6}  {'total_ms':>10}  {'env-steps/s':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}")

    for nworld in [64, 256, 1024]:
        data_b = mjw.make_data(mjm_drag, nworld=nworld)

        # warmup
        wp.launch(apply_drag, dim=[nworld, nv], inputs=[data_b.qvel, data_b.qfrc_applied, drag_coeff, nv])
        mjw.step(model_d, data_b)
        wp.synchronize()
        for _ in range(WARMUP):
            wp.launch(apply_drag, dim=[nworld, nv], inputs=[data_b.qvel, data_b.qfrc_applied, drag_coeff, nv])
            mjw.step(model_d, data_b)
        wp.synchronize()

        mjw.reset_data(model_d, data_b)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            wp.launch(apply_drag, dim=[nworld, nv], inputs=[data_b.qvel, data_b.qfrc_applied, drag_coeff, nv])
            mjw.step(model_d, data_b)
        wp.synchronize()
        batched_ms = (time.perf_counter() - t0) * 1000

        env_steps_s = nworld * N_STEPS / (batched_ms / 1000)
        print(f"  {nworld:>6}  {batched_ms:>10.2f}  {env_steps_s:>14.0f}")
        del data_b


# ---------------------------------------------------------------------------
# 5. API survey
# ---------------------------------------------------------------------------
def test_api_survey():
    section("5. mujoco_warp API Survey")

    print(f"\n  version: {mjw.__version__}")

    top_level = [x for x in dir(mjw) if not x.startswith('_')]
    print(f"\n  Top-level exports ({len(top_level)}):")
    for name in sorted(top_level):
        obj = getattr(mjw, name)
        kind = type(obj).__name__
        print(f"    {name:40s}  {kind}")

    import mujoco_warp._src.types as types
    annotations = {k: v for k, v in getattr(types.Data, '__annotations__', {}).items()
                   if not k.startswith('_')}
    print(f"\n  Data fields ({len(annotations)}):")
    for name, dtype in sorted(annotations.items()):
        print(f"    {name:40s}  {dtype}")

    import inspect
    funcs = ['step', 'forward', 'put_model', 'put_data', 'make_data',
             'get_data_into', 'get_state', 'set_state', 'reset_data',
             'xfrc_accumulate', 'step1', 'step2']
    print("\n  Key function signatures:")
    for fn_name in funcs:
        fn = getattr(mjw, fn_name, None)
        if fn:
            sig = inspect.signature(fn)
            print(f"    {fn_name}{sig}")
        else:
            print(f"    {fn_name}: NOT FOUND")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("MuJoCo-Warp 3.8.1 GPU Dynamics Benchmark")
    print(f"  mujoco:      {mujoco.__version__}")
    print(f"  mujoco_warp: {mjw.__version__}")
    print(f"  warp:        {wp.__version__}")
    print(f"  device:      {wp.get_device()}")
    print(f"  numpy:       {np.__version__}")
    if HAS_TORCH:
        print(f"  torch:       {torch.__version__}")
    print(f"  steps:       {N_STEPS} (warmup: {WARMUP})")

    test_cpu_basic()
    test_cpu_vs_gpu()
    test_batched()
    test_hydro()
    test_api_survey()

    section("Done")
