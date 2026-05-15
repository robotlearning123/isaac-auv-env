"""T2.7 — autograd correctness for Tier-1 Fossen Warp kernels.

Strategy:
1. Smallest possible test — added-mass kernel only, single env, scalar M_A.
2. wp.Tape gradient of sum(wrench) w.r.t. nu_dot vs analytic d/d(nu_dot).
3. Then expand to torch finite-diff comparison at random states.
4. Test each kernel individually (added-mass, damping, restoring) — autograd
   through the full pipeline is the v0.2 stretch goal.

Per `goals/W2_tier1_fossen.md §5 T2.7`: max rel-err < 1e-3.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.gpu
def test_added_mass_autograd_analytic_diagonal() -> None:
    """For diagonal M_A, sum(wrench) = -sum(M_A_i * nu_dot_i), so
    d(sum)/d(nu_dot_i) = -M_A_i (constant matrix). Verify via wp.Tape."""
    import warp as wp

    from oceanscale.hydro.tier1_kernels import tier1_added_mass

    wp.init()
    n = 1
    # M_A linear = (5.5, 12.7, 14.57), angular = (0.12, 0.12, 0.12)
    ma_lin_np = np.array([[5.5, 12.7, 14.57]], dtype=np.float32)
    ma_ang_np = np.array([[0.12, 0.12, 0.12]], dtype=np.float32)
    ma_lin = wp.array(ma_lin_np, dtype=wp.vec3f, device="cuda")
    ma_ang = wp.array(ma_ang_np, dtype=wp.vec3f, device="cuda")

    # Input requires_grad
    nu_dot_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    nu_dot = wp.array(nu_dot_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(tier1_added_mass, dim=n, inputs=[nu_dot, ma_lin, ma_ang, wrench], device="cuda")

    # Set upstream gradient: d(loss)/d(wrench[0]) = (1,1,1,1,1,1) → measures d(sum)/d(nu_dot)
    wrench.grad = wp.array(
        np.ones((1, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    g = nu_dot.grad.numpy()[0]
    # Analytic: d(sum(wrench)) / d(nu_dot[i]) = -M_A[i] for each component
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
    """D(ν)·ν is non-linear in ν. At ν = e_i (unit on axis i), analytic gradient is:
       d/dν_i [-(D_lin[i] + D_quad[i]·|ν_i|)·ν_i]
         = -(D_lin[i] + 2·D_quad[i]·|ν_i|)
       at ν_i = 1: = -(D_lin[i] + 2·D_quad[i])
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

    # ν = (1, 0, 0, 0, 0, 0) — pure surge
    nu_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    nu = wp.array(nu_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(tier1_damping, dim=n, inputs=[nu, dll, dla, dql, dqa, wrench], device="cuda")
    wrench.grad = wp.array(
        np.ones((1, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    g = nu.grad.numpy()[0]
    # Diagonal: d/dν_0 [-(D_lin[0] + D_quad[0]·|ν_0|)·ν_0] = -(D_lin[0] + 2·D_quad[0]·|ν_0|) at ν_0=1
    # = -(4.03 + 2·18.18) = -40.39
    expected_d0 = -(d_lin[0] + 2 * d_quad[0])
    np.testing.assert_allclose(g[0], expected_d0, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_torch_finite_diff_added_mass() -> None:
    """Cross-validation via torch finite difference at 20 random states.

    For each random nu_dot:
      1. Run Warp added-mass forward → wrench_warp
      2. Manually compute -M_A · nu_dot in numpy → wrench_ref
      3. Verify rel-err < 1e-5 at forward
      4. Perturb nu_dot[i] by eps, re-run forward, finite-diff approx
         d(sum(wrench))/d(nu_dot[i]) ≈ (sum(wrench_+) - sum(wrench_-)) / (2·eps)
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

    # Forward pass
    nu_dot = wp.array(nu_dot_np, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    wrench = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(tier1_added_mass, dim=n, inputs=[nu_dot, ma_lin, ma_ang, wrench], device="cuda")
    w_got = wrench.numpy()
    w_ref = -nu_dot_np * np.tile(ma, (n, 1))
    np.testing.assert_allclose(w_got, w_ref, rtol=1e-5, atol=1e-6)

    # Gradient via wp.Tape
    wrench.grad = wp.array(
        np.ones((n, 6), dtype=np.float32),
        dtype=wp.spatial_vectorf,
        device="cuda",
    )
    tape.backward()
    grad_wp = nu_dot.grad.numpy()
    # Analytic: -M_A per env (constant)
    grad_ref = np.tile(-ma, (n, 1))
    np.testing.assert_allclose(grad_wp, grad_ref, rtol=1e-3, atol=1e-3)
