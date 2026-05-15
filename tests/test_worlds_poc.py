"""T1.1: Verify newton.ModelBuilder.replicate() worlds API for parallel envs.

Resolves the architecture issue documented in STATUS.md: 8192 free-body
spheres placed in world=-1 caused SolverMuJoCo CPU mj_stackAlloc OOM
because MuJoCo treated them as one giant articulation.

Hypothesis: using ModelBuilder.replicate(template, world_count=N) with a
single-body template assigns each body to its own world, so SolverMuJoCo
sees N independent small worlds and the per-world Hessian / constraint
matrices stay tiny.

Reference: agent newton-worlds-api report; canonical pattern from
.venv/lib/python3.12/site-packages/newton/examples/robot/example_robot_cartpole.py
"""

from __future__ import annotations

import pytest


def _make_single_world_template():
    """Single-body free sphere — one 'world'."""
    import newton

    t = newton.ModelBuilder()
    body = t.add_body()
    t.add_shape_sphere(body, radius=0.1)
    # initial pose: at z=1, identity orientation; zero velocity
    t.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    t.joint_qd = [0.0] * 6
    return t


@pytest.mark.gpu
def test_replicate_64_semi_implicit_steps_correctly() -> None:
    """SolverSemiImplicit + replicate(N=64) — every world body falls under gravity."""
    import newton

    template = _make_single_world_template()
    scene = newton.ModelBuilder()
    scene.replicate(template, world_count=64, spacing=(0.0, 0.0, 0.0))
    model = scene.finalize(device="cuda")

    assert model.world_count == 64, f"expected 64 worlds, got {model.world_count}"
    assert model.body_count == 64, f"expected 64 bodies, got {model.body_count}"

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverSemiImplicit(model)
    dt = 1.0 / 240.0
    solver.step(state_0, state_1, control, None, dt)

    v = state_1.body_qd.numpy()
    assert v.shape == (64, 6), f"got {v.shape}"
    expected_vz = -9.81 * dt
    # all 64 worlds should be falling identically
    for i in range(64):
        assert v[i, 2] == pytest.approx(expected_vz, rel=0.05), (
            f"world {i}: linear-z {v[i, 2]:.4f} != expected {expected_vz:.4f}"
        )


@pytest.mark.gpu
def test_replicate_64_mujoco_no_oom() -> None:
    """SolverMuJoCo + replicate(N=64) — the architectural fix for the OOM seen
    in STATUS.md when 256+ free spheres were placed in world=-1."""
    import newton

    template = _make_single_world_template()
    # MuJoCo solver wants its custom attributes registered on the template
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)

    scene = newton.ModelBuilder()
    scene.replicate(template, world_count=64, spacing=(0.0, 0.0, 0.0))
    model = scene.finalize(device="cuda")

    assert model.world_count == 64

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverMuJoCo(model)
    dt = 1.0 / 240.0
    solver.step(state_0, state_1, control, None, dt)

    v = state_1.body_qd.numpy()
    assert v.shape == (64, 6)
    # all bodies should fall under gravity
    for i in range(64):
        assert v[i, 2] < 0, f"world {i}: expected fall, got {v[i]}"


@pytest.mark.gpu
def test_replicate_256_mujoco_no_oom() -> None:
    """The decisive test: 256 worlds via replicate() must not OOM SolverMuJoCo.

    Without replicate, 256 free spheres in world=-1 previously failed with:
        IndexError: index 19 is out of bounds for axis 0 with size 19
    """
    import newton

    template = _make_single_world_template()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)

    scene = newton.ModelBuilder()
    scene.replicate(template, world_count=256, spacing=(0.0, 0.0, 0.0))
    model = scene.finalize(device="cuda")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverMuJoCo(model)
    solver.step(state_0, state_1, control, None, 1.0 / 240.0)

    # Reaching this line is the test pass — earlier flat-list approach failed
    # with IndexError or mj_stackAlloc before reaching here.
    v = state_1.body_qd.numpy()
    assert v.shape == (256, 6)


@pytest.mark.gpu
def test_replicate_per_world_index_layout() -> None:
    """Confirm joint_q layout: N worlds × 7 floats per free joint = total 7N."""
    import newton

    template = _make_single_world_template()
    scene = newton.ModelBuilder()
    scene.replicate(template, world_count=8, spacing=(0.0, 0.0, 0.0))
    model = scene.finalize(device="cuda")

    # Each world: 7 joint_q (xyz + quat) + 6 joint_qd (linear + angular)
    assert len(model.joint_q.numpy()) == 8 * 7
    assert len(model.joint_qd.numpy()) == 8 * 6
