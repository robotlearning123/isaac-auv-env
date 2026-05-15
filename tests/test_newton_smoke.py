"""Newton smoke test — minimal rigid-body scene + one step on RTX 5090.

Pattern: `add_body()` auto-creates a free joint. joint_q layout = (xyz, quat-xyzw).
Use `eval_fk` to initialize state from joint coords before stepping.
"""
from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_newton_model_metadata() -> None:
    """Confirm model carries expected metadata."""
    import newton

    builder = newton.ModelBuilder()
    builder.add_body()
    builder.add_shape_sphere(0, radius=0.1)
    model = builder.finalize(device="cuda")

    assert model.body_count == 1
    assert model.shape_count >= 1
    assert "cuda" in str(model.device).lower()
    assert model.body_mass.numpy()[0] > 0.0


@pytest.mark.gpu
def test_newton_free_body_falls_semi_implicit() -> None:
    """Free body under gravity gains negative z-velocity after one step."""
    import newton

    builder = newton.ModelBuilder()  # default gravity=-9.81 along Z
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    # add_body() auto-creates a free joint with joint_q = (xyz, quat-xyzw), len=7
    builder.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    builder.joint_qd = [0.0] * 6  # 6 DOFs (3 angular + 3 linear)

    model = builder.finalize(device="cuda")
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverSemiImplicit(model)
    dt = 1.0 / 240.0
    solver.step(state_0, state_1, control, None, dt)

    v = state_1.body_qd.numpy()[0]
    print(f"\nSemiImplicit body_qd: {v}")
    # Newton spatial-vector layout: (linear_xyz, angular_xyz) — verified empirically
    v_lin_z = v[2]
    expected = -9.81 * dt
    assert v_lin_z < 0, f"expected negative z-velocity, got {v_lin_z}"
    assert v_lin_z == pytest.approx(expected, rel=0.05), (
        f"expected ~{expected:.4f}, got {v_lin_z:.4f}"
    )


@pytest.mark.gpu
def test_newton_free_body_falls_mujoco_warp() -> None:
    """Same with SolverMuJoCo (the primary Newton rigid backend)."""
    import newton

    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    builder.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    builder.joint_qd = [0.0] * 6

    model = builder.finalize(device="cuda")
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverMuJoCo(model)
    dt = 1.0 / 240.0
    solver.step(state_0, state_1, control, None, dt)

    v = state_1.body_qd.numpy()[0]
    print(f"\nMuJoCo-Warp body_qd: {v}")
    # Newton spatial layout (linear_xyz, angular_xyz)
    assert v[2] < 0, f"expected fall (negative linear z), got {v}"
    expected = -9.81 * dt
    assert v[2] == pytest.approx(expected, rel=0.05), f"expected ~{expected:.4f}, got {v[2]:.4f}"


@pytest.mark.gpu
def test_newton_8192_envs_finalize() -> None:
    """Build a scene with 8192 parallel rigid bodies — throughput target check.

    Just verifies finalize succeeds at scale; does NOT step (we want to know if
    the builder + GPU memory survive 8k envs even before measuring throughput).
    """
    import newton

    builder = newton.ModelBuilder()
    for _ in range(8192):
        b = builder.add_body()
        builder.add_shape_sphere(b, radius=0.1)
    # set initial joint_q for all 8192 bodies (7 floats each)
    builder.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 8192
    builder.joint_qd = [0.0] * (8192 * 6)

    model = builder.finalize(device="cuda")
    assert model.body_count == 8192
    print(f"\n8192-body model: shape_count={model.shape_count}, "
          f"mass[0]={model.body_mass.numpy()[0]:.4f}")
