"""T2.7 — autograd correctness for Tier-1 Fossen Warp kernels.

Verifies forward + backward gradient correctness vs torch autograd at random
states.  Covers all 6 Tier-1 kernels: added_mass, coriolis_a, damping,
restoring, thruster_alloc, accumulate_to_body_f.

Per IMPLEMENTATION_PLAN.md W2 gate §T2.7: max rel-err < 1e-3.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Torch reference implementations (Fossen 2021 math, diagonal M_A)
# ---------------------------------------------------------------------------


def _added_mass_ref(
    nu_dot: torch.Tensor, ma_lin: torch.Tensor, ma_ang: torch.Tensor
) -> torch.Tensor:
    return torch.cat([-(ma_lin * nu_dot[:, :3]), -(ma_ang * nu_dot[:, 3:])], dim=-1)


def _damping_ref(
    nu: torch.Tensor,
    dll: torch.Tensor,
    dla: torch.Tensor,
    dql: torch.Tensor,
    dqa: torch.Tensor,
) -> torch.Tensor:
    vl, va = nu[:, :3], nu[:, 3:]
    f0 = -(dll[:, 0] + dql[:, 0] * vl[:, 0].abs()) * vl[:, 0]
    f1 = -(dll[:, 1] + dql[:, 1] * vl[:, 1].abs()) * vl[:, 1]
    f2 = -(dll[:, 2] + dql[:, 2] * vl[:, 2].abs()) * vl[:, 2]
    f3 = -(dla[:, 0] + dqa[:, 0] * va[:, 0].abs()) * va[:, 0]
    f4 = -(dla[:, 1] + dqa[:, 1] * va[:, 1].abs()) * va[:, 1]
    f5 = -(dla[:, 2] + dqa[:, 2] * va[:, 2].abs()) * va[:, 2]
    # cross-coupling disabled in v0.1 — coefficients not independently identified
    # (see YAW_CROSSCOUPLING_ANALYSIS.md)
    # f1 = f1 - dql[:, 1] * va[:, 2].abs() * va[:, 2]
    # f5 = f5 - dqa[:, 2] * vl[:, 1].abs() * vl[:, 1]
    # f2 = f2 - dql[:, 2] * va[:, 1].abs() * va[:, 1]
    # f4 = f4 - dqa[:, 1] * vl[:, 2].abs() * vl[:, 2]
    return torch.stack([f0, f1, f2, f3, f4, f5], dim=-1)


def _coriolis_a_ref(nu: torch.Tensor, ma_lin: torch.Tensor, ma_ang: torch.Tensor) -> torch.Tensor:
    vl, va = nu[:, :3], nu[:, 3:]
    ab_lin = ma_lin * vl
    ab_ang = ma_ang * va
    f_lin = -torch.cross(ab_lin, va, dim=-1)
    f_ang = -(torch.cross(ab_lin, vl, dim=-1) + torch.cross(ab_ang, va, dim=-1))
    return torch.cat([f_lin, f_ang], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    c = 2 * qw * qw - 1
    d = 2 * (qx * vx + qy * vy + qz * vz)
    rx = vx * c + qx * d + (qy * vz - qz * vy) * 2 * qw
    ry = vy * c + qy * d + (qz * vx - qx * vz) * 2 * qw
    rz = vz * c + qz * d + (qx * vy - qy * vx) * 2 * qw
    return torch.stack([rx, ry, rz], dim=-1)


def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(torch.cat([-q[..., :3], q[..., 3:4]], dim=-1), v)


def _restoring_ref(
    quat: torch.Tensor,
    mass: float,
    volume: float,
    coBM: float,
    rho: float,
    g: float,
) -> torch.Tensor:
    W, B = mass * g, rho * volume * g
    z_down = quat.new_tensor([[0.0, 0.0, -1.0]])
    z_body = _quat_rotate_inv(quat, z_down)
    f_body = (W - B) * z_body
    r_b = quat.new_tensor([[0.0, 0.0, coBM]])
    m_body = torch.cross(r_b, -B * z_body, dim=-1)
    return torch.cat([f_body, m_body], dim=-1)


def _thruster_alloc_ref(
    u_cmd: torch.Tensor,
    u_eff_prev: torch.Tensor,
    T: torch.Tensor,
    max_thrust: float,
    deadband: float,
    tau_lag: float,
    dt: float,
) -> torch.Tensor:
    alpha = dt / (tau_lag + dt)
    u_raw = torch.clamp(u_cmd, -1.0, 1.0)
    u_raw = torch.where(u_raw.abs() >= deadband, u_raw, torch.zeros_like(u_raw))
    u_eff = u_eff_prev + alpha * (u_raw - u_eff_prev)
    return torch.bmm(T, (u_eff * max_thrust).unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# BlueROV coefficient constants (from MarineGym BlueROV.yaml)
# ---------------------------------------------------------------------------

_MA = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12], dtype=np.float32)
_DL = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=np.float32)
_DQ = np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55], dtype=np.float32)


# ===========================================================================
# Existing tests (preserved verbatim)
# ===========================================================================


@pytest.mark.gpu
def test_added_mass_autograd_analytic_diagonal() -> None:
    """For diagonal M_A, sum(wrench) = -sum(M_A_i * nu_dot_i), so
    d(sum)/d(nu_dot_i) = -M_A_i (constant matrix). Verify via wp.Tape."""
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_added_mass

    wp.init()
    n = 1
    ma_lin_np = np.array([[5.5, 12.7, 14.57]], dtype=np.float32)
    ma_ang_np = np.array([[0.12, 0.12, 0.12]], dtype=np.float32)
    ma_lin = wp.array(ma_lin_np, dtype=wp.vec3f, device="cuda")
    ma_ang = wp.array(ma_ang_np, dtype=wp.vec3f, device="cuda")

    nu_dot_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    nu_dot = wp.array(nu_dot_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(tier1_added_mass, dim=n, inputs=[nu_dot, ma_lin, ma_ang, wrench], device="cuda")

    wrench.grad = wp.array(
        np.ones((1, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    g = nu_dot.grad.numpy()[0]
    expected = -np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12], dtype=np.float32)
    np.testing.assert_allclose(g, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_added_mass_autograd_random_samples() -> None:
    """50 random nu_dot samples — autograd should give -M_A (constant) regardless."""
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_added_mass

    wp.init()
    n = 50
    rng = np.random.default_rng(42)
    nu_dot_np = rng.normal(0.0, 2.0, size=(n, 6)).astype(np.float32)
    ma = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12], dtype=np.float32)
    ma_lin_np = np.tile(ma[:3], (n, 1))
    ma_ang_np = np.tile(ma[3:], (n, 1))
    ma_lin = wp.array(ma_lin_np, dtype=wp.vec3f, device="cuda")
    ma_ang = wp.array(ma_ang_np, dtype=wp.vec3f, device="cuda")
    nu_dot = wp.array(nu_dot_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(tier1_added_mass, dim=n, inputs=[nu_dot, ma_lin, ma_ang, wrench], device="cuda")
    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    g = nu_dot.grad.numpy()
    expected = np.tile(-ma, (n, 1))
    np.testing.assert_allclose(g, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_damping_autograd_at_unit_velocity() -> None:
    """D(nu)*nu is non-linear in nu. At nu = e_i (unit on axis i), analytic gradient is:
       d/dnu_i [-(D_lin[i] + D_quad[i]*|nu_i|)*nu_i]
         = -(D_lin[i] + 2*D_quad[i]*|nu_i|)
       at nu_i = 1: = -(D_lin[i] + 2*D_quad[i])
    Verify via wp.Tape.
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_damping

    wp.init()
    n = 1
    d_lin = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=np.float32)
    d_quad = np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55], dtype=np.float32)
    dll = wp.array(np.tile(d_lin[:3], (n, 1)), dtype=wp.vec3f, device="cuda")
    dla = wp.array(np.tile(d_lin[3:], (n, 1)), dtype=wp.vec3f, device="cuda")
    dql = wp.array(np.tile(d_quad[:3], (n, 1)), dtype=wp.vec3f, device="cuda")
    dqa = wp.array(np.tile(d_quad[3:], (n, 1)), dtype=wp.vec3f, device="cuda")
    current_z = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3f, device="cuda")

    nu_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    nu = wp.array(nu_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(tier1_damping, dim=n, inputs=[nu, dll, dla, dql, dqa, current_z, wrench], device="cuda")
    wrench.grad = wp.array(
        np.ones((1, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    g = nu.grad.numpy()[0]
    expected_d0 = -(d_lin[0] + 2 * d_quad[0])
    np.testing.assert_allclose(g[0], expected_d0, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_torch_finite_diff_added_mass() -> None:
    """Cross-validation via torch finite difference at 20 random states.

    For each random nu_dot:
      1. Run Warp added-mass forward -> wrench_warp
      2. Manually compute -M_A * nu_dot in numpy -> wrench_ref
      3. Verify rel-err < 1e-5 at forward
      4. Perturb nu_dot[i] by eps, re-run forward, finite-diff approx
         d(sum(wrench))/d(nu_dot[i]) ~ (sum(wrench_+) - sum(wrench_-)) / (2*eps)
      5. Compare against wp.Tape gradient — should match within 1e-3 rel-err
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_added_mass

    wp.init()
    n = 20
    rng = np.random.default_rng(7)
    nu_dot_np = rng.normal(0.0, 1.5, size=(n, 6)).astype(np.float32)
    ma = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12], dtype=np.float32)
    ma_lin_np = np.tile(ma[:3], (n, 1))
    ma_ang_np = np.tile(ma[3:], (n, 1))
    ma_lin = wp.array(ma_lin_np, dtype=wp.vec3f, device="cuda")
    ma_ang = wp.array(ma_ang_np, dtype=wp.vec3f, device="cuda")

    nu_dot = wp.array(nu_dot_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(tier1_added_mass, dim=n, inputs=[nu_dot, ma_lin, ma_ang, wrench], device="cuda")
    w_got = wrench.numpy()
    w_ref = -nu_dot_np * np.tile(ma, (n, 1))
    np.testing.assert_allclose(w_got, w_ref, rtol=1e-5, atol=1e-6)

    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    grad_wp = nu_dot.grad.numpy()
    grad_ref = np.tile(-ma, (n, 1))
    np.testing.assert_allclose(grad_wp, grad_ref, rtol=1e-3, atol=1e-3)


# ===========================================================================
# New: torch cross-validation for all 6 Tier-1 kernels
# ===========================================================================


@pytest.mark.gpu
def test_coriolis_a_autograd_vs_torch() -> None:
    """wp.Tape gradient for coriolis_a matches torch autograd at 20 random states.

    Source: tier1_kernels.py:80-101 — C_A(nu) using diagonal M_A, Fossen 6.3.
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_coriolis_a

    wp.init()
    n = 20
    rng = np.random.default_rng(42)
    nu_np = rng.normal(0, 1.5, (n, 6)).astype(np.float32)
    ma_lin_np, ma_ang_np = np.tile(_MA[:3], (n, 1)), np.tile(_MA[3:], (n, 1))

    # Torch reference
    nu_t = torch.tensor(nu_np, requires_grad=True)
    w_t = _coriolis_a_ref(nu_t, torch.tensor(ma_lin_np), torch.tensor(ma_ang_np))
    w_t.sum().backward()
    grad_torch = nu_t.grad.numpy()
    fwd_torch = w_t.detach().numpy()

    # Warp
    nu_w = wp.array(nu_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            tier1_coriolis_a,
            dim=n,
            inputs=[
                nu_w,
                wp.array(ma_lin_np, dtype=wp.vec3f, device="cuda"),
                wp.array(ma_ang_np, dtype=wp.vec3f, device="cuda"),
                wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3f, device="cuda"),
                wrench,
            ],
            device="cuda",
        )
    fwd_wp = wrench.numpy()

    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device="cuda"
    )
    tape.backward()
    grad_wp = nu_w.grad.numpy()

    np.testing.assert_allclose(fwd_wp, fwd_torch, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_wp, grad_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_damping_full_jacobian_vs_torch() -> None:
    """Full damping kernel gradient matches torch at random states.

    Source: tier1_kernels.py:40-73 — D(nu)*nu (cross-coupling disabled in v0.1).
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_damping

    wp.init()
    n = 20
    rng = np.random.default_rng(55)
    nu_np = rng.normal(0, 1.0, (n, 6)).astype(np.float32)
    dll_np, dla_np = np.tile(_DL[:3], (n, 1)), np.tile(_DL[3:], (n, 1))
    dql_np, dqa_np = np.tile(_DQ[:3], (n, 1)), np.tile(_DQ[3:], (n, 1))

    # Torch
    nu_t = torch.tensor(nu_np, requires_grad=True)
    w_t = _damping_ref(
        nu_t, torch.tensor(dll_np), torch.tensor(dla_np), torch.tensor(dql_np), torch.tensor(dqa_np)
    )
    w_t.sum().backward()
    grad_torch = nu_t.grad.numpy()
    fwd_torch = w_t.detach().numpy()

    # Warp
    nu_w = wp.array(nu_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            tier1_damping,
            dim=n,
            inputs=[
                nu_w,
                wp.array(dll_np, dtype=wp.vec3f, device="cuda"),
                wp.array(dla_np, dtype=wp.vec3f, device="cuda"),
                wp.array(dql_np, dtype=wp.vec3f, device="cuda"),
                wp.array(dqa_np, dtype=wp.vec3f, device="cuda"),
                wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3f, device="cuda"),
                wrench,
            ],
            device="cuda",
        )
    fwd_wp = wrench.numpy()

    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device="cuda"
    )
    tape.backward()
    grad_wp = nu_w.grad.numpy()

    np.testing.assert_allclose(fwd_wp, fwd_torch, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_wp, grad_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_restoring_autograd_vs_torch() -> None:
    """Restoring force gradient w.r.t. quaternion matches torch at tilted orientations.

    Source: tier1_kernels.py:108-140 — g(eta) via quaternion, Fossen 6.5.
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_restoring

    wp.init()
    n = 5
    rng = np.random.default_rng(88)
    # Random orientations: small-angle perturbations from identity
    q_np = np.zeros((n, 4), dtype=np.float32)
    q_np[:, :3] = rng.normal(0, 0.2, (n, 3)).astype(np.float32)
    q_np[:, 3] = 1.0
    q_np /= np.linalg.norm(q_np, axis=1, keepdims=True)

    mass, volume, coBM = 11.4, 0.0113459, 0.01
    rho, g = 1025.0, 9.81

    # Torch
    quat_t = torch.tensor(q_np, requires_grad=True)
    w_t = _restoring_ref(quat_t, mass, volume, coBM, rho, g)
    w_t.sum().backward()
    grad_torch = quat_t.grad.numpy()
    fwd_torch = w_t.detach().numpy()

    # Warp
    quat_w = wp.array(q_np, dtype=wp.quatf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            tier1_restoring,
            dim=n,
            inputs=[
                quat_w,
                wp.array(np.full(n, mass, dtype=np.float32), dtype=wp.float32, device="cuda"),
                wp.array(np.full(n, volume, dtype=np.float32), dtype=wp.float32, device="cuda"),
                wp.array(np.full(n, coBM, dtype=np.float32), dtype=wp.float32, device="cuda"),
                rho,
                g,
                wrench,
            ],
            device="cuda",
        )
    fwd_wp = wrench.numpy()

    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device="cuda"
    )
    tape.backward()
    grad_wp = quat_w.grad.numpy()

    np.testing.assert_allclose(fwd_wp, fwd_torch, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grad_wp, grad_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_thruster_alloc_autograd_vs_torch() -> None:
    """Thruster allocation gradient w.r.t. u_cmd matches torch (no saturation/deadband).

    Source: tier1_kernels.py:147-196 — tau = T*u with saturation, deadband, 1st-order lag.
    u_cmd kept well within [-0.6, 0.6] to avoid nondifferentiable boundaries.
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_thruster_alloc

    wp.init()
    n, n_thr = 4, 8
    rng = np.random.default_rng(77)
    u_cmd_np = rng.uniform(-0.6, 0.6, (n, n_thr)).astype(np.float32)
    u_prev_np = np.zeros((n, n_thr), dtype=np.float32)
    T_np = np.zeros((n, 6, n_thr), dtype=np.float32)
    for k in range(min(6, n_thr)):
        T_np[:, k, k] = 1.0
    max_thrust, deadband, tau_lag, dt = 51.5, 0.05, 0.1, 1.0 / 240.0

    # Torch
    u_t = torch.tensor(u_cmd_np, requires_grad=True)
    w_t = _thruster_alloc_ref(
        u_t,
        torch.tensor(u_prev_np),
        torch.tensor(T_np),
        max_thrust,
        deadband,
        tau_lag,
        dt,
    )
    w_t.sum().backward()
    grad_torch = u_t.grad.numpy()
    fwd_torch = w_t.detach().numpy()

    # Warp
    u_cmd_w = wp.array(u_cmd_np, dtype=wp.float32, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            tier1_thruster_alloc,
            dim=n,
            inputs=[
                u_cmd_w,
                wp.array(u_prev_np, dtype=wp.float32, device="cuda"),
                wp.zeros((n, n_thr), dtype=wp.float32, device="cuda"),
                wp.array(T_np, dtype=wp.float32, device="cuda"),
                max_thrust,
                deadband,
                tau_lag,
                dt,
                wp.array(np.ones(n_thr, dtype=np.float32), dtype=wp.float32, device="cuda"),
                wrench,
            ],
            device="cuda",
        )
    fwd_wp = wrench.numpy()

    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device="cuda"
    )
    tape.backward()
    grad_wp = u_cmd_w.grad.numpy()

    np.testing.assert_allclose(fwd_wp, fwd_torch, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grad_wp, grad_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_accumulate_to_body_f_gradient() -> None:
    """d(sum(body_f_out))/d(wrench_buf) = 1 for each component (additive kernel).

    Source: tier1_kernels.py:211-218 — body_f[i] = body_f[i] + wrench_buf[i].
    """
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_accumulate_to_body_f

    wp.init()
    n = 4
    rng = np.random.default_rng(99)
    w_np = rng.normal(0, 1, (n, 6)).astype(np.float32)
    bf_np = rng.normal(0, 1, (n, 6)).astype(np.float32)

    w = wp.array(w_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    bf = wp.array(bf_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(tier1_accumulate_to_body_f, dim=n, inputs=[w, bf], device="cuda")
    bf.grad = wp.array(np.ones((n, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device="cuda")
    tape.backward()

    np.testing.assert_allclose(
        w.grad.numpy(),
        np.ones((n, 6), dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
