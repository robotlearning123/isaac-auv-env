"""Energy conservation tests for Newton rigid-body dynamics.

Physics invariant: a free body without drag/buoyancy maintains constant
kinetic energy.  Under gravity alone, mechanical energy (KE + PE) should
be conserved up to numerical integration error.

Identified as the #1 missing test by independent code review.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.gpu
def test_ke_conservation_no_forces() -> None:
    """Free body, no gravity/drag/buoyancy → KE constant within 0.1% over 1000 steps.

    With zero net force the SemiImplicit integrator should reproduce v=const
    exactly (the velocity update is v += F/m·dt and F=0).
    """
    import newton

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    # joint_q  layout: (x, y, z, qx, qy, qz, qw) — free joint
    builder.joint_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # joint_qd layout: (vx, vy, vz, ωx, ωy, ωz) — verified via eval_fk
    builder.joint_qd = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    model = builder.finalize(device="cuda")
    mass = float(model.body_mass.numpy()[0])

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverSemiImplicit(model)
    dt = 1.0 / 240.0

    # body_qd layout: (linear_xyz, angular_xyz) — see test_newton_smoke.py
    v_init = state_0.body_qd.numpy()[0, :3]
    ke_init = 0.5 * mass * float(np.dot(v_init, v_init))

    n_steps = 1000
    for _ in range(n_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    v_final = state_0.body_qd.numpy()[0, :3]
    ke_final = 0.5 * mass * float(np.dot(v_final, v_final))

    assert ke_init > 0, f"Initial KE should be positive, got {ke_init}"
    rel = abs(ke_final - ke_init) / ke_init
    assert rel < 1e-3, (
        f"KE not conserved (no forces): init={ke_init:.6f}, "
        f"final={ke_final:.6f}, rel_error={rel:.6%}"
    )


@pytest.mark.gpu
def test_mechanical_energy_with_gravity() -> None:
    """Free body under gravity → KE + PE conserved within 1%.

    SemiImplicit Euler drifts by ~0.5·m·g²·dt² per step.  With z₀=5,
    v₀=(1,0,2), mass ~0.004, 100 steps at dt=1/240 the predicted relative
    drift is ~0.16% — well within 1%.
    """
    import newton

    builder = newton.ModelBuilder()  # default gravity = (0, 0, -9.81)
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    builder.joint_q = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]
    builder.joint_qd = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0]  # v=(1,0,2), ω=0

    model = builder.finalize(device="cuda")
    mass = float(model.body_mass.numpy()[0])
    g = 9.81

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverSemiImplicit(model)
    dt = 1.0 / 240.0

    def mechanical_energy(state):
        v = state.body_qd.numpy()[0, :3]
        z = float(state.body_q.numpy()[0, 2])
        ke = 0.5 * mass * float(np.dot(v, v))
        pe = mass * g * z
        return ke + pe

    e_init = mechanical_energy(state_0)

    n_steps = 100
    for _ in range(n_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    e_final = mechanical_energy(state_0)

    assert abs(e_init) > 0, f"Initial energy should be nonzero, got {e_init}"
    rel = abs(e_final - e_init) / abs(e_init)
    assert rel < 0.01, (
        f"Mechanical energy not conserved: init={e_init:.6f}, "
        f"final={e_final:.6f}, rel_error={rel:.4%}"
    )
