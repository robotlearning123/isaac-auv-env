"""Spectral CFD Benchmark: FFT + Pseudo-spectral Navier-Stokes + PCG solver.

Targets RTX 5090 (CUDA 12.8). Tests:
  Part 1: 3D FFT (CuPy vs PyTorch) on 64^3..512^3 grids
  Part 2: 2D pseudo-spectral Navier-Stokes (Taylor-Green vortex)
  Part 4: Preconditioned Conjugate Gradient (Warp) for pressure Poisson
"""

import time

import cupy as cp
import numpy as np
import torch
import warp as wp

wp.init()
wp.clear_kernel_cache()

N_WARMUP = 3
N_TRIALS = 10


def bench_fft_3d():
    """Part 1: 3D FFT benchmarks — CuPy vs PyTorch."""
    print("=" * 72)
    print("PART 1: 3D FFT Performance (CuPy vs PyTorch)")
    print("=" * 72)
    print(f"{'Grid':>8} {'CuPy fwd':>10} {'CuPy inv':>10} {'CuPy cells/s':>14} "
          f"{'Torch fwd':>10} {'Torch inv':>10} {'Torch cells/s':>14}")
    print("-" * 72)

    grid_sizes = [64, 128, 256, 512]

    for n in grid_sizes:
        total_cells = n ** 3
        # --- CuPy ---
        a_cp = (cp.random.randn(n, n, n) + 1j * cp.random.randn(n, n, n)).astype(cp.complex128)
        # warmup
        for _ in range(N_WARMUP):
            _ = cp.fft.fftn(a_cp)
            _ = cp.fft.ifftn(a_cp)
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            b = cp.fft.fftn(a_cp)
        cp.cuda.Stream.null.synchronize()
        cp_fwd = (time.perf_counter() - t0) / N_TRIALS

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            _ = cp.fft.ifftn(b)
        cp.cuda.Stream.null.synchronize()
        cp_inv = (time.perf_counter() - t0) / N_TRIALS

        cp_throughput = total_cells / cp_fwd

        # --- PyTorch ---
        a_th = torch.randn(n, n, n, dtype=torch.complex128, device="cuda")
        for _ in range(N_WARMUP):
            _ = torch.fft.fftn(a_th)
            _ = torch.fft.ifftn(a_th)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            b = torch.fft.fftn(a_th)
        torch.cuda.synchronize()
        th_fwd = (time.perf_counter() - t0) / N_TRIALS

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            _ = torch.fft.ifftn(b)
        torch.cuda.synchronize()
        th_inv = (time.perf_counter() - t0) / N_TRIALS

        th_throughput = total_cells / th_fwd

        print(f"{n:>8} {cp_fwd*1e3:>9.3f}ms {cp_inv*1e3:>9.3f}ms {cp_throughput:>13.3e} "
              f"{th_fwd*1e3:>9.3f}ms {th_inv*1e3:>9.3f}ms {th_throughput:>13.3e}")

        del a_cp, b, a_th
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()


def taylor_green_exact(x, y, t, nu, L=2 * np.pi):
    """Exact Taylor-Green vortex solution (2D).

    u = -cos(kx)*sin(ky)*exp(-2*nu*k^2*t)
    v =  sin(kx)*cos(ky)*exp(-2*nu*k^2*t)
    omega = -2*k*cos(kx)*cos(ky)*exp(-2*nu*k^2*t)
    """
    k = 2 * np.pi / L
    decay = np.exp(-2 * nu * k ** 2 * t)
    u = -np.cos(k * x) * np.sin(k * y) * decay
    v = np.sin(k * x) * np.cos(k * y) * decay
    omega = -2 * k * np.cos(k * x) * np.cos(k * y) * decay
    return u, v, omega


def bench_pseudo_spectral_ns():
    """Part 2: 2D pseudo-spectral Navier-Stokes (Taylor-Green vortex)."""
    print()
    print("=" * 72)
    print("PART 2: Pseudo-spectral Navier-Stokes (2D Taylor-Green Vortex)")
    print("=" * 72)

    nu = 0.01  # kinematic viscosity
    L = 2 * np.pi
    dt_frac = 0.5  # CFL fraction

    grid_sizes = [256, 512, 1024, 2048]

    print(f"\n{'Grid':>8} {'dt':>10} {'time/step':>12} {'cells/s':>14} {'L2 error':>12} {'N steps':>8}")
    print("-" * 72)

    for N in grid_sizes:
        dx = L / N
        # Wavenumbers
        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        K2_inv = np.zeros_like(K2)
        K2_inv[K2 > 0] = 1.0 / K2[K2 > 0]

        # Physical grid
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Initial condition (t=0)
        _, _, omega0 = taylor_green_exact(X, Y, 0.0, nu, L)
        omega_hat = np.fft.fft2(omega0)

        # Time step (CFL-based)
        dt = dt_frac * dx / 10.0  # conservative estimate
        T_final = 0.5
        n_steps = int(T_final / dt)
        if n_steps < 1:
            n_steps = 1
            dt = T_final

        # Transfer to GPU (CuPy)
        K2_cp = cp.asarray(K2)
        K2_inv_cp = cp.asarray(K2_inv)
        KX_cp = cp.asarray(KX)
        KY_cp = cp.asarray(KY)
        omega_hat_cp = cp.asarray(omega_hat)
        cp.int32(N)

        # Dealias mask (2/3 rule)
        dealias = cp.ones((N, N))
        kmax = N // 3
        dealias[cp.abs(KX_cp).get() > kmax * (2 * np.pi / L)] = 0
        # Actually use wavenumber indices
        kidx_x = np.fft.fftfreq(N, d=1.0 / N).astype(int)
        kidx_y = np.fft.fftfreq(N, d=1.0 / N).astype(int)
        KIDX_X, KIDX_Y = np.meshgrid(kidx_x, kidx_y, indexing="ij")
        dealias_np = np.ones((N, N))
        dealias_np[np.abs(KIDX_X) > N // 3] = 0
        dealias_np[np.abs(KIDX_Y) > N // 3] = 0
        dealias_cp = cp.asarray(dealias_np)

        def step(
            omega_hat,
            dt,
            nu,
            K2_cp=K2_cp,
            K2_inv_cp=K2_inv_cp,
            KX_cp=KX_cp,
            KY_cp=KY_cp,
            dealias_cp=dealias_cp,
        ):
            # Compute velocity from vorticity
            # psi_hat = -omega_hat / K2
            psi_hat = -omega_hat * K2_inv_cp

            # u = d(psi)/dy = i*KY*psi_hat
            # v = -d(psi)/dx = -i*KX*psi_hat
            u_hat = 1j * KY_cp * psi_hat
            v_hat = -1j * KX_cp * psi_hat

            # Nonlinear term in physical space (dealiased)
            u = cp.fft.ifft2(u_hat).real
            v = cp.fft.ifft2(v_hat).real
            domega_dx = cp.fft.ifft2(1j * KX_cp * omega_hat).real
            domega_dy = cp.fft.ifft2(1j * KY_cp * omega_hat).real

            nonlinear = u * domega_dx + v * domega_dy
            nl_hat = cp.fft.fft2(nonlinear) * dealias_cp

            # Semi-implicit: diffusion exact, advection explicit (RK2 / Heun)
            # Stage 1
            omega_hat_1 = (omega_hat - dt * nl_hat) / (1 + dt * nu * K2_cp)

            # Stage 2 (Heun correction)
            psi_hat_1 = -omega_hat_1 * K2_inv_cp
            u_hat_1 = 1j * KY_cp * psi_hat_1
            v_hat_1 = -1j * KX_cp * psi_hat_1
            u_1 = cp.fft.ifft2(u_hat_1).real
            v_1 = cp.fft.ifft2(v_hat_1).real
            domega_dx_1 = cp.fft.ifft2(1j * KX_cp * omega_hat_1).real
            domega_dy_1 = cp.fft.ifft2(1j * KY_cp * omega_hat_1).real
            nl_1 = u_1 * domega_dx_1 + v_1 * domega_dy_1
            nl_hat_1 = cp.fft.fft2(nl_1) * dealias_cp

            omega_hat_new = omega_hat - dt * 0.5 * (nl_hat + nl_hat_1)
            omega_hat_new = omega_hat_new / (1 + dt * nu * K2_cp)

            return omega_hat_new

        # Warmup
        omega_test = omega_hat_cp.copy()
        for _ in range(3):
            omega_test = step(omega_test, dt, nu)
        cp.cuda.Stream.null.synchronize()

        # Timed run
        omega_hat_cp_run = omega_hat_cp.copy()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            omega_hat_cp_run = step(omega_hat_cp_run, dt, nu)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        time_per_step = elapsed / n_steps
        cells_per_sec = N * N / time_per_step

        # L2 error vs exact solution
        omega_final = cp.fft.ifft2(omega_hat_cp_run).real.get()
        _, _, omega_exact = taylor_green_exact(X, Y, n_steps * dt, nu, L)
        l2_err = np.sqrt(np.mean((omega_final - omega_exact) ** 2)) / np.sqrt(
            np.mean(omega_exact ** 2)
        )

        print(f"{N:>8} {dt:>10.2e} {time_per_step*1e3:>11.3f}ms {cells_per_sec:>13.3e} "
              f"{l2_err:>12.2e} {n_steps:>8}")

        # Free GPU memory
        del omega_hat_cp, omega_hat_cp_run, omega_test
        del K2_cp, K2_inv_cp, KX_cp, KY_cp, dealias_cp
        cp.get_default_memory_pool().free_all_blocks()


def bench_pcg_warp():
    """Part 4: Preconditioned Conjugate Gradient in Warp for pressure Poisson."""
    print()
    print("=" * 72)
    print("PART 4: Preconditioned Conjugate Gradient (Warp) — 3D Pressure Poisson")
    print("=" * 72)

    N = 64  # 64^3 grid
    n_total = N * N * N
    h = 1.0 / (N + 1)
    h2 = h * h

    print(f"Grid: {N}^3 = {n_total} unknowns (interior points), h = {h:.6f}")
    print("Using Dirichlet BC (u=0 on boundary), so grid is interior-only")
    print()

    # For Dirichlet BC interior grid, Laplacian is standard 7-point stencil.
    # Every point has exactly 6 neighbors -> diagonal is always -6/h^2.
    # System is SPD, CG will converge.

    # RHS: -Lap(u) = f, with u = sin(pi*x)*sin(pi*y)*sin(pi*z) on [0,1]^3
    # Use random RHS (realistic pressure Poisson scenario with multiple frequency content)
    rng = np.random.default_rng(42)
    b_flat = rng.standard_normal(n_total)

    @wp.kernel
    def laplacian_3d_stencil(
        p: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
        N: wp.int32,
        h2_inv: wp.float64,
    ):
        idx = wp.tid()
        N2 = N * N
        i = idx / N2
        j = (idx % N2) / N
        k = idx % N

        val = wp.float64(-6.0) * h2_inv * p[idx]

        # +x
        if i + 1 < N:
            val = val + h2_inv * p[(i + 1) * N2 + j * N + k]
        # -x
        if i - 1 >= 0:
            val = val + h2_inv * p[(i - 1) * N2 + j * N + k]
        # +y
        if j + 1 < N:
            val = val + h2_inv * p[i * N2 + (j + 1) * N + k]
        # -y
        if j - 1 >= 0:
            val = val + h2_inv * p[i * N2 + (j - 1) * N + k]
        # +z
        if k + 1 < N:
            val = val + h2_inv * p[i * N2 + j * N + k + 1]
        # -z
        if k - 1 >= 0:
            val = val + h2_inv * p[i * N2 + j * N + k - 1]

        out[idx] = val

    @wp.kernel
    def jacobi_precond(
        r: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        diag_inv: wp.float64,
    ):
        idx = wp.tid()
        z[idx] = r[idx] * diag_inv

    @wp.kernel
    def axpy_kernel(
        a: wp.float64,
        x: wp.array(dtype=wp.float64),
        b: wp.float64,
        y: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
    ):
        idx = wp.tid()
        out[idx] = a * x[idx] + b * y[idx]

    @wp.kernel
    def scale_kernel(
        a: wp.float64,
        x: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
    ):
        idx = wp.tid()
        out[idx] = a * x[idx]

    @wp.kernel
    def copy_kernel(
        src: wp.array(dtype=wp.float64),
        dst: wp.array(dtype=wp.float64),
    ):
        idx = wp.tid()
        dst[idx] = src[idx]

    @wp.kernel
    def add_kernel(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
    ):
        idx = wp.tid()
        out[idx] = x[idx] + y[idx]

    @wp.kernel
    def sub_kernel(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
    ):
        idx = wp.tid()
        out[idx] = x[idx] - y[idx]

    def dot_product(a, b):
        # Use Warp's built-in reduction or manual
        # For simplicity, transfer to CPU for dot product
        # (would use cublas in production)
        a_np = a.numpy()
        b_np = b.numpy()
        return np.dot(a_np, b_np)

    def run_pcg(b_wp, preconditioner="none", max_iter=500, tol=1e-6):
        """Run PCG with specified preconditioner."""
        x = wp.zeros(n_total, dtype=wp.float64)
        r = wp.zeros(n_total, dtype=wp.float64)
        z = wp.zeros(n_total, dtype=wp.float64)
        p_vec = wp.zeros(n_total, dtype=wp.float64)
        Ap = wp.zeros(n_total, dtype=wp.float64)
        wp.zeros(n_total, dtype=wp.float64)

        # r = b - A*x (x=0, so r=b initially)
        wp.launch(copy_kernel, dim=n_total, inputs=[b_wp, r])

        # Apply preconditioner to r -> z
        diag_inv = wp.float64(-1.0 / (6.0 * h2_inv_val))
        if preconditioner == "jacobi":
            wp.launch(jacobi_precond, dim=n_total, inputs=[r, z, diag_inv])
        else:
            wp.launch(copy_kernel, dim=n_total, inputs=[r, z])

        # p = z
        wp.launch(copy_kernel, dim=n_total, inputs=[z, p_vec])

        rz = dot_product(r, z)
        b_norm = np.sqrt(dot_product(b_wp, b_wp))
        if b_norm == 0:
            b_norm = 1.0

        times = []
        converged_at = max_iter

        for k in range(max_iter):
            t0 = time.perf_counter()

            # Ap = A * p
            wp.launch(laplacian_3d_stencil, dim=n_total, inputs=[p_vec, Ap, wp.int32(N), h2_inv_wp])
            wp.synchronize()

            # alpha = rz / pAp
            pAp = dot_product(p_vec, Ap)
            if abs(pAp) < 1e-30:
                converged_at = k
                break
            alpha = rz / pAp

            # x = x + alpha*p
            wp.launch(axpy_kernel, dim=n_total, inputs=[
                wp.float64(alpha), p_vec, wp.float64(1.0), x, x
            ])

            # r = r - alpha*Ap
            wp.launch(axpy_kernel, dim=n_total, inputs=[
                wp.float64(1.0), r, wp.float64(-alpha), Ap, r
            ])

            # Check convergence
            r_norm = np.sqrt(dot_product(r, r))
            if r_norm / b_norm < tol:
                converged_at = k + 1
                wp.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                break

            # Apply preconditioner
            if preconditioner == "jacobi":
                wp.launch(jacobi_precond, dim=n_total, inputs=[r, z, diag_inv])
            else:
                wp.launch(copy_kernel, dim=n_total, inputs=[r, z])

            rz_new = dot_product(r, z)
            beta = rz_new / rz

            # p = z + beta*p
            wp.launch(axpy_kernel, dim=n_total, inputs=[
                wp.float64(1.0), z, wp.float64(beta), p_vec, p_vec
            ])

            rz = rz_new
            wp.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_iter_time = np.mean(times) if times else 0
        total_time = sum(times)
        residual = np.sqrt(dot_product(r, r))

        return converged_at, total_time, avg_iter_time, residual, x

    h2_inv_val = 1.0 / h2
    h2_inv_wp = wp.float64(h2_inv_val)
    b_wp = wp.array(b_flat, dtype=wp.float64)

    print(f"{'Precond':>12} {'Iters':>8} {'Total(s)':>10} {'Avg iter':>12} {'Residual':>14} {'Final |r|/|b|':>14}")
    print("-" * 72)

    for precond in ["none", "jacobi"]:
        # Warmup
        _ = run_pcg(b_wp, precond, max_iter=5, tol=1e-30)

        n_iters, total, avg, resid, x_sol = run_pcg(b_wp, precond, max_iter=500, tol=1e-6)
        b_norm = np.sqrt(np.dot(b_flat, b_flat))
        rel_res = resid / b_norm if b_norm > 0 else resid

        label = {"none": "None", "jacobi": "Jacobi"}[precond]
        print(f"{label:>12} {n_iters:>8} {total:>10.4f} {avg*1e3:>11.4f}ms {resid:>14.3e} {rel_res:>14.3e}")

        del x_sol
        cp.get_default_memory_pool().free_all_blocks()

    # Also benchmark with CuPy sparse direct solve for comparison
    print()
    print("CuPy sparse CG (reference, same matrix):")
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csplg
    from scipy import sparse as sp

    # Build 7-point 3D Laplacian for Dirichlet BC (interior points)
    diag_val = -6.0 * h2_inv_val
    off_val = h2_inv_val
    A_sp = sp.diags(
        [diag_val * np.ones(n_total),
         off_val * np.ones(n_total - N * N),
         off_val * np.ones(n_total - N * N),
         off_val * np.ones(n_total - N),
         off_val * np.ones(n_total - N),
         off_val * np.ones(n_total - 1),
         off_val * np.ones(n_total - 1)],
        [0, N * N, -(N * N), N, -N, 1, -1],
        shape=(n_total, n_total),
        format="csr",
    )
    for i in range(N):
        for j in range(N):
            row = i * N * N + j * N
            if j > 0:
                A_sp[row, row - 1] = 0
            if j < N - 1:
                A_sp[row + N - 1, row + N] = 0

    try:
        A_gpu = csp.csr_matrix(A_sp)
        b_gpu = cp.asarray(b_flat)

        # CuPy CG
        _ = csplg.cg(A_gpu, b_gpu, atol=1e-6, rtol=1e-6)
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        x_cg, _info_cg = csplg.cg(A_gpu, b_gpu, atol=1e-6, rtol=1e-6)
        cp.cuda.Stream.null.synchronize()
        t_cupy_cg = time.perf_counter() - t0

        res_cupy_cg = cp.linalg.norm(A_gpu @ x_cg - b_gpu).get()
        b_norm_gpu = cp.linalg.norm(b_gpu).get()
        print(f"  CG time: {t_cupy_cg:.4f}s, residual: {res_cupy_cg:.3e}, |r|/|b|: {res_cupy_cg/b_norm_gpu:.3e}")

        del A_gpu, b_gpu, x_cg
    except Exception as e:
        print(f"  CuPy CG failed: {e}")

    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    print("GPU: RTX 5090, VRAM: 31 GB")
    print(f"CuPy {cp.__version__}, PyTorch {torch.__version__}, Warp {wp.__version__}")
    print(f"NumPy {np.__version__}")
    print()

    bench_fft_3d()
    bench_pseudo_spectral_ns()
    bench_pcg_warp()

    print()
    print("Benchmark complete.")
