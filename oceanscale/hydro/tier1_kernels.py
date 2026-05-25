# mypy: ignore-errors
"""Tier-1 Fossen Warp kernels — added-mass, damping, Coriolis (C_A), restoring,
thruster allocation. All operate per-world on (n_envs,) sized arrays.

Math reference: docs/math/fossen.md.
Algorithm port: MarineGym underwaterVehicle.py:229-277 (MIT).
"""

from __future__ import annotations

import warp as wp


# ----------------------------------------------------------------------------
# T2.2 — added-mass M_A · ν̇
# ----------------------------------------------------------------------------
@wp.kernel
def tier1_added_mass(
    nu_dot: wp.array(dtype=wp.spatial_vectorf),  # (n_envs,) per-env body accel (linear, angular)
    M_A: wp.array(dtype=wp.vec3f),  # (n_envs, 2) flattened: [lin, ang] diag values
    M_A_ang: wp.array(dtype=wp.vec3f),  # (n_envs,) angular diag of M_A
    wrench: wp.array(dtype=wp.spatial_vectorf),  # (n_envs,) output, ACCUMULATED
) -> None:
    i = wp.tid()
    nd = nu_dot[i]
    ml = M_A[i]
    ma = M_A_ang[i]
    # spatial_vector layout: (linear_xyz, angular_xyz) per STATUS.md
    nd_lin = wp.spatial_top(nd)
    nd_ang = wp.spatial_bottom(nd)
    f_lin = -wp.vec3f(ml[0] * nd_lin[0], ml[1] * nd_lin[1], ml[2] * nd_lin[2])
    f_ang = -wp.vec3f(ma[0] * nd_ang[0], ma[1] * nd_ang[1], ma[2] * nd_ang[2])
    f = wp.spatial_vector(f_lin, f_ang)
    wp.atomic_add(wrench, i, f)


# ----------------------------------------------------------------------------
# T2.3 — damping D(ν)·ν, includes 4 cross-coupling terms per MarineGym L239-244
# ----------------------------------------------------------------------------
@wp.kernel
def tier1_damping(
    nu: wp.array(dtype=wp.spatial_vectorf),  # (n_envs,) body velocity (vx,vy,vz, p,q,r)
    d_lin_lin: wp.array(dtype=wp.vec3f),  # (n_envs,) linear-damping linear part
    d_lin_ang: wp.array(dtype=wp.vec3f),  # (n_envs,) linear-damping angular part
    d_quad_lin: wp.array(dtype=wp.vec3f),  # (n_envs,) quadratic-damping linear part
    d_quad_ang: wp.array(dtype=wp.vec3f),  # (n_envs,) quadratic-damping angular part
    wrench: wp.array(dtype=wp.spatial_vectorf),  # (n_envs,) output, ACCUMULATED
) -> None:
    i = wp.tid()
    v = nu[i]
    v_lin = wp.spatial_top(v)  # (vx, vy, vz)
    v_ang = wp.spatial_bottom(v)  # (p, q, r)
    dll = d_lin_lin[i]
    dla = d_lin_ang[i]
    dql = d_quad_lin[i]
    dqa = d_quad_ang[i]

    # Diagonal damping
    f0 = -(dll[0] + dql[0] * wp.abs(v_lin[0])) * v_lin[0]
    f1 = -(dll[1] + dql[1] * wp.abs(v_lin[1])) * v_lin[1]
    f2 = -(dll[2] + dql[2] * wp.abs(v_lin[2])) * v_lin[2]
    f3 = -(dla[0] + dqa[0] * wp.abs(v_ang[0])) * v_ang[0]
    f4 = -(dla[1] + dqa[1] * wp.abs(v_ang[1])) * v_ang[1]
    f5 = -(dla[2] + dqa[2] * wp.abs(v_ang[2])) * v_ang[2]
    # cross-coupling disabled in v0.1 — coefficients not independently identified
    # (see YAW_CROSSCOUPLING_ANALYSIS.md)
    # # axis 1 (sway) + axis 5 (yaw): F_sway -= D_quad[1] * |r| * r ;  F_yaw -= D_quad[5] * |v| * v
    # f1 = f1 - dql[1] * wp.abs(v_ang[2]) * v_ang[2]
    # f5 = f5 - dqa[2] * wp.abs(v_lin[1]) * v_lin[1]
    # # axis 2 (heave) + axis 4 (pitch): F_heave -= D_quad[2] * |q| * q ; F_pitch -= D_quad[4] * |w| * w
    # f2 = f2 - dql[2] * wp.abs(v_ang[1]) * v_ang[1]
    # f4 = f4 - dqa[1] * wp.abs(v_lin[2]) * v_lin[2]

    f = wp.spatial_vector(wp.vec3f(f0, f1, f2), wp.vec3f(f3, f4, f5))
    wp.atomic_add(wrench, i, f)


# ----------------------------------------------------------------------------
# T2.4 — Coriolis C_A(ν)·ν (added-mass Coriolis only; rigid C_RB delegated to MJWarp)
# ----------------------------------------------------------------------------
@wp.kernel
def tier1_coriolis_a(
    nu: wp.array(dtype=wp.spatial_vectorf),
    M_A_lin: wp.array(dtype=wp.vec3f),  # diag entries of M_A linear block
    M_A_ang: wp.array(dtype=wp.vec3f),  # diag entries of M_A angular block
    wrench: wp.array(dtype=wp.spatial_vectorf),
) -> None:
    i = wp.tid()
    v = nu[i]
    v_lin = wp.spatial_top(v)
    v_ang = wp.spatial_bottom(v)
    ml = M_A_lin[i]
    ma = M_A_ang[i]
    # ab = M_A · ν  (block-diagonal mul)
    ab_lin = wp.vec3f(ml[0] * v_lin[0], ml[1] * v_lin[1], ml[2] * v_lin[2])
    ab_ang = wp.vec3f(ma[0] * v_ang[0], ma[1] * v_ang[1], ma[2] * v_ang[2])
    # C(ν)·ν via Fossen §6.3 (diagonal M_A simplification):
    # F_lin = -ab_lin × ω
    # M_ang = -(ab_lin × v + ab_ang × ω)
    f_lin = -wp.cross(ab_lin, v_ang)
    f_ang = -(wp.cross(ab_lin, v_lin) + wp.cross(ab_ang, v_ang))
    f = wp.spatial_vector(f_lin, f_ang)
    wp.atomic_add(wrench, i, f)


# ----------------------------------------------------------------------------
# T2.5 — restoring g(η) using quaternion (no Euler) for gimbal-lock safety
# ----------------------------------------------------------------------------
@wp.kernel
def tier1_restoring(
    quat: wp.array(dtype=wp.quatf),  # (n_envs,) body orientation in world (x,y,z,w)
    mass: wp.array(dtype=wp.float32),  # (n_envs,)
    volume: wp.array(dtype=wp.float32),  # (n_envs,)
    coBM: wp.array(
        dtype=wp.float32
    ),  # (n_envs,) COG-COB distance scalar (positive = COB above COG in body z)
    rho_water: wp.float32,  # kg/m^3 (default 1025 salt, 997 fresh)
    g_accel: wp.float32,  # 9.81
    wrench: wp.array(dtype=wp.spatial_vectorf),
) -> None:
    i = wp.tid()
    q = quat[i]
    m = mass[i]
    V = volume[i]
    d = coBM[i]
    W = m * g_accel  # weight (positive scalar)
    B = rho_water * V * g_accel  # buoyancy (positive scalar)

    # World-down unit vector in world frame: (0, 0, -1) for Newton z-up
    # Rotate (0, 0, -1) into body frame via inverse rotation = wp.quat_rotate_inv(q, ...)
    # Restoring force in body frame = (B - W) * R_body_to_world^T * z_world_down  (Fossen §6.5)
    z_world_down = wp.vec3f(0.0, 0.0, -1.0)
    z_body = wp.quat_rotate_inv(q, z_world_down)
    f_body = (W - B) * z_body  # negative buoyancy → sinks; positive → floats up in body z

    # Moment from COB offset: r_b_b × (-B · z_body)
    # COB located at (0, 0, +d) in body frame (above origin in body z)
    r_b = wp.vec3f(0.0, 0.0, d)
    m_body = wp.cross(r_b, -B * z_body)

    f = wp.spatial_vector(f_body, m_body)
    wp.atomic_add(wrench, i, f)


# ----------------------------------------------------------------------------
# T2.6 — thruster allocation τ = T · u  + saturation + deadband + 1st-order time-constant
# ----------------------------------------------------------------------------
@wp.kernel
def tier1_thruster_alloc(
    u_cmd: wp.array2d(dtype=wp.float32),  # (n_envs, n_thrusters) commanded throttle in [-1, +1]
    u_eff_prev: wp.array2d(dtype=wp.float32),  # (n_envs, n_thrusters) previous effective throttle
    u_eff_out: wp.array2d(
        dtype=wp.float32
    ),  # (n_envs, n_thrusters) updated effective throttle (writeback)
    T_matrix: wp.array(
        dtype=wp.float32, ndim=3
    ),  # (n_envs, 6, n_thrusters) per-env allocation matrix
    max_thrust: wp.float32,  # N per thruster (saturation magnitude)
    deadband: wp.float32,  # |u| below this → 0
    tau_lag: wp.float32,  # 1st-order time-constant (seconds)
    dt: wp.float32,  # timestep
    wrench: wp.array(dtype=wp.spatial_vectorf),
) -> None:
    i = wp.tid()
    n_thrusters = u_cmd.shape[1]
    # First-order lag + deadband + saturation per thruster
    alpha = dt / (tau_lag + dt)  # discrete-time approx of dt/τ
    fx = float(0.0)  # noqa: UP018 - Warp needs dynamic variables in kernels.
    fy = float(0.0)  # noqa: UP018 - Warp needs dynamic variables in kernels.
    fz = float(0.0)  # noqa: UP018 - Warp needs dynamic variables in kernels.
    mx = float(0.0)  # noqa: UP018 - Warp needs dynamic variables in kernels.
    my = float(0.0)  # noqa: UP018 - Warp needs dynamic variables in kernels.
    mz = float(0.0)  # noqa: UP018 - Warp needs dynamic variables in kernels.
    for k in range(n_thrusters):
        u_raw = u_cmd[i, k]
        # saturation in [-1, +1]
        if u_raw > 1.0:
            u_raw = 1.0
        if u_raw < -1.0:
            u_raw = -1.0
        # deadband
        if wp.abs(u_raw) < deadband:
            u_raw = 0.0
        # 1st-order lag: u_eff += alpha * (u_raw - u_eff)
        u_prev_k = u_eff_prev[i, k]
        u_eff_k = u_prev_k + alpha * (u_raw - u_prev_k)
        u_eff_out[i, k] = u_eff_k
        # Convert effective throttle to thrust force (N)
        f_k = u_eff_k * max_thrust
        # Apply allocation: each thruster contributes to wrench columns
        fx = fx + T_matrix[i, 0, k] * f_k
        fy = fy + T_matrix[i, 1, k] * f_k
        fz = fz + T_matrix[i, 2, k] * f_k
        mx = mx + T_matrix[i, 3, k] * f_k
        my = my + T_matrix[i, 4, k] * f_k
        mz = mz + T_matrix[i, 5, k] * f_k
    f = wp.spatial_vector(wp.vec3f(fx, fy, fz), wp.vec3f(mx, my, mz))
    wp.atomic_add(wrench, i, f)


# ----------------------------------------------------------------------------
# Helper kernels for force injection wiring (T2.1)
# ----------------------------------------------------------------------------
@wp.kernel
def tier1_zero_wrench(
    wrench: wp.array(dtype=wp.spatial_vectorf),
) -> None:
    i = wp.tid()
    wrench[i] = wp.spatial_vector(wp.vec3f(0.0), wp.vec3f(0.0))


@wp.kernel
def tier1_accumulate_to_body_f(
    wrench_buf: wp.array(dtype=wp.spatial_vectorf),  # per-env (n_envs,)
    body_f: wp.array(dtype=wp.spatial_vectorf),  # Newton state.body_f (n_envs,)
) -> None:
    i = wp.tid()
    # Add hydro wrench to body_f (replaces previous + adds; thrust + gravity etc.
    # already accumulated by Newton via control/state_in)
    body_f[i] = body_f[i] + wrench_buf[i]


@wp.kernel
def tier1_update_nu_dot_ema(
    nu_curr: wp.array(dtype=wp.spatial_vectorf),
    nu_prev: wp.array(dtype=wp.spatial_vectorf),  # in-out: updated to nu_curr
    nu_dot_prev: wp.array(dtype=wp.spatial_vectorf),  # in-out: updated to filtered nu_dot
    alpha: wp.float32,  # EMA filter (0.3 per MarineGym L230)
    dt: wp.float32,
) -> None:
    i = wp.tid()
    v_curr = nu_curr[i]
    v_prev = nu_prev[i]
    nd_prev = nu_dot_prev[i]
    # Componentwise raw acceleration
    diff = v_curr - v_prev
    inv_dt = 1.0 / dt
    nd_raw_lin = wp.spatial_top(diff) * inv_dt
    nd_raw_ang = wp.spatial_bottom(diff) * inv_dt
    # EMA: nd_filtered = (1-α)*nd_prev + α*nd_raw
    nd_prev_lin = wp.spatial_top(nd_prev)
    nd_prev_ang = wp.spatial_bottom(nd_prev)
    nd_lin = (1.0 - alpha) * nd_prev_lin + alpha * nd_raw_lin
    nd_ang = (1.0 - alpha) * nd_prev_ang + alpha * nd_raw_ang
    nu_dot_prev[i] = wp.spatial_vector(nd_lin, nd_ang)
    nu_prev[i] = v_curr
