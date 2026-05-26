"""D3Q19 Lattice Boltzmann Method benchmark — PyTorch vs Warp.

Lid-driven cavity and Poiseuille channel flow on GPU.
Reports MLUPS, memory usage, steps/sec, and L2 validation errors vs Ghia et al.

Consolidates the former lbm_torch.py and lbm_warp.py into a single A/B harness.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# D3Q19 lattice constants (shared by both backends)
# ---------------------------------------------------------------------------

CX_NP = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int32)
CY_NP = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int32)
CZ_NP = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int32)

W_NP = np.array([
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=np.float32)

OPP_NP = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], dtype=np.int32)

Q = 19

# Ghia et al. benchmark data
GHIA_RE100_Y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1484,
                          0.1953, 0.2422, 0.2891, 0.3359, 0.3828, 0.4297,
                          0.4766, 0.5234, 0.5703, 0.6172, 0.6641, 0.7109,
                          0.7578, 0.8047, 0.8516, 0.8984, 0.9453, 1.0])
GHIA_RE100_U = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06436, -0.07561,
                          -0.07216, -0.06169, -0.05046, -0.04083, -0.03376, -0.02864,
                          -0.02461, -0.02076, -0.01654, -0.01172, -0.00628, -0.00062,
                          0.00540, 0.01313, 0.02580, 0.05302, 0.14034, 1.0])
GHIA_RE400_Y = GHIA_RE100_Y.copy()
GHIA_RE400_U = np.array([0.0, -0.08186, -0.09266, -0.10358, -0.14591, -0.17152,
                          -0.15496, -0.12253, -0.09140, -0.06506, -0.04391, -0.02772,
                          -0.01619, -0.00849, -0.00321, 0.00106, 0.00528, 0.01075,
                          0.01869, 0.03195, 0.05603, 0.10091, 0.19174, 1.0])
GHIA_RE1000_Y = GHIA_RE100_Y.copy()
GHIA_RE1000_U = np.array([0.0, -0.18109, -0.20196, -0.21619, -0.24929, -0.22457,
                           -0.16316, -0.10929, -0.07422, -0.05010, -0.03416, -0.02336,
                           -0.01594, -0.01036, -0.00538, -0.00031, 0.00595, 0.01456,
                           0.02810, 0.05030, 0.08675, 0.15387, 0.26551, 1.0])

GHIA = {
    100: (GHIA_RE100_Y, GHIA_RE100_U),
    400: (GHIA_RE400_Y, GHIA_RE400_U),
    1000: (GHIA_RE1000_Y, GHIA_RE1000_U),
}


def compute_l2_ghia(centerline_u: np.ndarray, ny: int, re: int) -> float:
    y_positions = np.arange(ny) / (ny - 1)
    ghia_y, ghia_u = GHIA[re]
    our_u_interp = np.interp(ghia_y, y_positions, centerline_u)
    mask = (ghia_y > 0.01) & (ghia_y < 0.99)
    if mask.sum() < 3:
        return float('nan')
    return float(np.sqrt(np.mean((our_u_interp[mask] - ghia_u[mask]) ** 2)))


def mem_estimate_mb(cells: int) -> float:
    return (Q * cells * 4 * 2 + cells * 4 * 4) / (1024 * 1024)


def _safe_mlups(cells: int, times: list[float], warmup: int) -> tuple[float, float, float]:
    """Return (mlups, steps_per_sec, avg_step_ms) guarding against nan/inf."""
    arr = np.array(times[warmup:]) if len(times) > warmup else np.array(times)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    avg = float(np.mean(arr))
    if avg < 1e-9:
        return 0.0, 0.0, 0.0
    return cells * 1e-6 / avg, 1.0 / avg, avg * 1000


def gpu_mem_mb() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()


# ===================================================================
# PyTorch backend
# ===================================================================

def _torch_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TorchLBM:
    def __init__(self):
        import torch
        self.torch = torch
        self.device = torch.device('cuda')
        self.CX = torch.tensor(CX_NP, dtype=torch.int32, device=self.device)
        self.CY = torch.tensor(CY_NP, dtype=torch.int32, device=self.device)
        self.CZ = torch.tensor(CZ_NP, dtype=torch.int32, device=self.device)
        self.W = torch.tensor(W_NP, dtype=torch.float32, device=self.device)
        self.OPP = torch.tensor(OPP_NP, dtype=torch.int32, device=self.device)

    def equilibrium(self, rho, ux, uy, uz):
        cx = self.CX.reshape(Q, 1, 1, 1)
        cy = self.CY.reshape(Q, 1, 1, 1)
        cz = self.CZ.reshape(Q, 1, 1, 1)
        w = self.W.reshape(Q, 1, 1, 1)
        cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
        usq = ux * ux + uy * uy + uz * uz
        return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0))

    def macroscopic(self, f):
        torch = self.torch
        cx = self.CX.reshape(Q, 1, 1, 1)
        cy = self.CY.reshape(Q, 1, 1, 1)
        cz = self.CZ.reshape(Q, 1, 1, 1)
        rho = f.sum(dim=0)
        ux = (f * cx).sum(dim=0) / rho
        uy = (f * cy).sum(dim=0) / rho
        uz = (f * cz).sum(dim=0) / rho

        nx, ny, nz = f.shape[1], f.shape[2], f.shape[3]
        wall = torch.zeros(nx, ny, nz, dtype=torch.bool, device=self.device)
        wall[0, :, :] = True
        wall[nx-1, :, :] = True
        wall[:, 0, :] = True
        wall[:, ny-1, :] = True
        wall[:, :, 0] = True
        wall[:, :, nz-1] = True
        rho[wall] = 1.0
        ux[wall] = 0.0
        uy[wall] = 0.0
        uz[wall] = 0.0
        return rho, ux, uy, uz

    def collide_stream(self, f, rho, ux, uy, uz, omega):
        torch = self.torch
        feq = self.equilibrium(rho, ux, uy, uz)
        f_post = f - omega * (f - feq)
        fnew = torch.zeros_like(f)
        for q in range(Q):
            fnew[q] = torch.roll(torch.roll(torch.roll(
                f_post[q], CX_NP[q].item(), dims=0),
                CY_NP[q].item(), dims=1),
                CZ_NP[q].item(), dims=2)

        nx, ny, nz = f.shape[1], f.shape[2], f.shape[3]
        for q in range(Q):
            oq = OPP_NP[q].item()
            if CX_NP[q] > 0:
                fnew[q, nx-1, :, :] = f_post[oq, nx-1, :, :]
            elif CX_NP[q] < 0:
                fnew[q, 0, :, :] = f_post[oq, 0, :, :]
            if CY_NP[q] > 0:
                fnew[q, :, ny-1, :] = f_post[oq, :, ny-1, :]
            elif CY_NP[q] < 0:
                fnew[q, :, 0, :] = f_post[oq, :, 0, :]
            if CZ_NP[q] > 0:
                fnew[q, :, :, nz-1] = f_post[oq, :, :, nz-1]
            elif CZ_NP[q] < 0:
                fnew[q, :, :, 0] = f_post[oq, :, :, 0]
        return fnew

    def apply_lid_zou_he(self, f, u_lid, ny):
        j = ny - 1
        rho_wall = (
            f[0, :, j, :] + f[1, :, j, :] + f[2, :, j, :] +
            f[5, :, j, :] + f[6, :, j, :] +
            2.0 * (f[3, :, j, :] + f[7, :, j, :] + f[8, :, j, :] +
                   f[15, :, j, :] + f[17, :, j, :])
        )
        f[4, :, j, :] = f[3, :, j, :]
        diff_12 = f[1, :, j, :] - f[2, :, j, :]
        diff_xz = f[11, :, j, :] - f[12, :, j, :] + f[13, :, j, :] - f[14, :, j, :]
        f[9, :, j, :] = f[8, :, j, :] + 0.5 * rho_wall * u_lid - 0.5 * diff_12 - 0.5 * diff_xz
        f[10, :, j, :] = f[7, :, j, :] - 0.5 * rho_wall * u_lid + 0.5 * diff_12 + 0.5 * diff_xz
        f[16, :, j, :] = f[17, :, j, :]
        f[18, :, j, :] = f[15, :, j, :]
        return f

    def run_lid_cavity(self, nx, ny, nz, re, u_lid=0.05, max_steps=5000,
                       warmup_steps=100) -> dict[str, Any] | None:
        torch = self.torch
        nu = u_lid * (nx - 1) / re
        tau = 3.0 * nu + 0.5
        omega = 1.0 / tau

        print(f"  [torch] Lid cavity {nx}x{ny}x{nz}, Re={re}, tau={tau:.4f}")

        rho = torch.ones(nx, ny, nz, dtype=torch.float32, device=self.device)
        ux = torch.zeros(nx, ny, nz, dtype=torch.float32, device=self.device)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        f = self.equilibrium(rho, ux, uy, uz)

        all_times = []
        for _ in range(max_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rho, ux, uy, uz = self.macroscopic(f)
            f = self.apply_lid_zou_he(f, u_lid, ny)
            f = self.collide_stream(f, rho, ux, uy, uz, omega)
            torch.cuda.synchronize()
            all_times.append(time.perf_counter() - t0)

        rho, ux_f, _, _ = self.macroscopic(f)
        torch.cuda.synchronize()
        ux_host = ux_f.cpu().numpy()
        centerline_u = ux_host[nx // 2, :, nz // 2] / u_lid

        l2 = compute_l2_ghia(centerline_u, ny, re)
        cells = nx * ny * nz
        mlups, sps, avg_ms = _safe_mlups(cells, all_times, warmup_steps)

        result = dict(
            backend='torch', nx=nx, ny=ny, nz=nz, re=re, tau=tau,
            mlups=mlups, steps_per_sec=sps,
            avg_step_ms=avg_ms, mem_mb=mem_estimate_mb(cells),
            cells=cells, l2_error=l2, steps=max_steps,
        )
        print(f"    MLUPS: {result['mlups']:.1f}, Step: {result['avg_step_ms']:.2f}ms, L2: {l2:.4f}")
        return result

    def run_channel_flow(self, nx=128, ny=64, nz=64, u_max=0.05,
                         max_steps=5000, warmup_steps=100) -> dict[str, Any]:
        torch = self.torch
        H = ny - 1
        tau = 0.8
        nu = (tau - 0.5) / 3.0
        omega = 1.0 / tau
        re = u_max * H / nu

        print(f"  [torch] Channel flow {nx}x{ny}x{nz}, Re={re:.1f}")

        rho = torch.ones(nx, ny, nz, dtype=torch.float32, device=self.device)
        ux = torch.zeros(nx, ny, nz, dtype=torch.float32, device=self.device)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        f = self.equilibrium(rho, ux, uy, uz)

        all_times = []
        for _ in range(max_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rho, ux, uy, uz = self.macroscopic(f)
            f = self.collide_stream(f, rho, ux, uy, uz, omega)

            rho_in = (1.0 / (1.0 - u_max)) * (
                f[0, 0, :, :] + f[3, 0, :, :] + f[4, 0, :, :] +
                f[5, 0, :, :] + f[6, 0, :, :] +
                f[15, 0, :, :] + f[16, 0, :, :] + f[17, 0, :, :] + f[18, 0, :, :] +
                2.0 * (f[2, 0, :, :] + f[8, 0, :, :] + f[10, 0, :, :] +
                       f[12, 0, :, :] + f[14, 0, :, :])
            )
            f[1, 0, :, :] = f[2, 0, :, :] + (1.0 / 3.0) * rho_in * u_max
            diff_y = f[3, 0, :, :] - f[4, 0, :, :] - f[15, 0, :, :] + f[16, 0, :, :] - f[17, 0, :, :] + f[18, 0, :, :]
            diff_z = f[5, 0, :, :] - f[6, 0, :, :] - f[15, 0, :, :] + f[16, 0, :, :] - f[17, 0, :, :] + f[18, 0, :, :]
            f[7, 0, :, :] = f[10, 0, :, :] + (1.0 / 3.0) * rho_in * u_max + 0.5 * diff_y - 0.5 * diff_z
            f[9, 0, :, :] = f[8, 0, :, :] + (1.0 / 3.0) * rho_in * u_max - 0.5 * diff_y - 0.5 * diff_z
            f[11, 0, :, :] = f[14, 0, :, :] - 0.5 * diff_z
            f[13, 0, :, :] = f[12, 0, :, :] - 0.5 * diff_z
            f[:, nx-1, :, :] = f[:, nx-2, :, :]

            torch.cuda.synchronize()
            all_times.append(time.perf_counter() - t0)

        rho, ux_f, _, _ = self.macroscopic(f)
        torch.cuda.synchronize()
        ux_host = ux_f.cpu().numpy()
        profile = ux_host[nx // 2, :, nz // 2]
        y_arr = np.arange(ny)
        analytical = u_max * 4.0 * (y_arr / H) * (1.0 - y_arr / H)
        interior = np.arange(1, ny - 1)
        l2 = float(np.sqrt(np.mean((profile[interior] - analytical[interior]) ** 2)))

        cells = nx * ny * nz
        mlups, sps, avg_ms = _safe_mlups(cells, all_times, warmup_steps)

        result = dict(
            backend='torch', nx=nx, ny=ny, nz=nz, re=re, tau=tau,
            mlups=mlups, steps_per_sec=sps,
            avg_step_ms=avg_ms, mem_mb=mem_estimate_mb(cells),
            cells=cells, l2_error=l2, u_max_analytical=u_max,
            max_u=float(np.max(np.abs(profile[interior]))),
            steps=max_steps,
        )
        print(f"    MLUPS: {result['mlups']:.1f}, Step: {result['avg_step_ms']:.2f}ms, L2: {l2:.6f}")
        return result


# ===================================================================
# Warp backend
# ===================================================================

def _warp_available() -> bool:
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except ImportError:
        return False


def _define_warp_kernels():
    """Define all Warp kernels. Called once at import time if warp is available."""
    import warp as wp

    @wp.kernel
    def init_equilibrium(
        f: wp.array4d(dtype=wp.float32),
        rho: wp.array3d(dtype=wp.float32),
        ux: wp.array3d(dtype=wp.float32),
        uy: wp.array3d(dtype=wp.float32),
        uz: wp.array3d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
        w_arr: wp.array(dtype=wp.float32),
        cx_arr: wp.array(dtype=wp.int32),
        cy_arr: wp.array(dtype=wp.int32),
        cz_arr: wp.array(dtype=wp.int32),
    ):
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        r = rho[i, j, k]
        vx = ux[i, j, k]; vy = uy[i, j, k]; vz = uz[i, j, k]
        usq = vx * vx + vy * vy + vz * vz
        for q in range(19):
            cu = float(cx_arr[q]) * vx + float(cy_arr[q]) * vy + float(cz_arr[q]) * vz
            f[q, i, j, k] = w_arr[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    @wp.kernel
    def compute_macroscopic(
        f: wp.array4d(dtype=wp.float32),
        rho: wp.array3d(dtype=wp.float32),
        ux: wp.array3d(dtype=wp.float32),
        uy: wp.array3d(dtype=wp.float32),
        uz: wp.array3d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
        cx_arr: wp.array(dtype=wp.int32),
        cy_arr: wp.array(dtype=wp.int32),
        cz_arr: wp.array(dtype=wp.int32),
    ):
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        is_wall = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1)
        if is_wall:
            rho[i, j, k] = 1.0
            ux[i, j, k] = float(0.0); uy[i, j, k] = float(0.0); uz[i, j, k] = float(0.0)
            return
        r = float(0.0); vx = float(0.0); vy = float(0.0); vz = float(0.0)
        for q in range(19):
            fq = f[q, i, j, k]
            r += fq
            vx += fq * float(cx_arr[q]); vy += fq * float(cy_arr[q]); vz += fq * float(cz_arr[q])
        rho[i, j, k] = r
        inv_r = 1.0 / r
        ux[i, j, k] = vx * inv_r; uy[i, j, k] = vy * inv_r; uz[i, j, k] = vz * inv_r

    @wp.kernel
    def collide_bgk(
        f: wp.array4d(dtype=wp.float32),
        fnew: wp.array4d(dtype=wp.float32),
        rho: wp.array3d(dtype=wp.float32),
        ux: wp.array3d(dtype=wp.float32),
        uy: wp.array3d(dtype=wp.float32),
        uz: wp.array3d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
        omega: wp.float32,
        w_arr: wp.array(dtype=wp.float32),
        cx_arr: wp.array(dtype=wp.int32),
        cy_arr: wp.array(dtype=wp.int32),
        cz_arr: wp.array(dtype=wp.int32),
    ):
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        is_wall = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1)
        if is_wall:
            return
        r = rho[i, j, k]; vx = ux[i, j, k]; vy = uy[i, j, k]; vz = uz[i, j, k]
        usq = vx * vx + vy * vy + vz * vz
        for q in range(19):
            cu = float(cx_arr[q]) * vx + float(cy_arr[q]) * vy + float(cz_arr[q]) * vz
            feq = w_arr[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
            fnew[q, i, j, k] = f[q, i, j, k] - omega * (f[q, i, j, k] - feq)

    @wp.kernel
    def stream_pull(
        fnew: wp.array4d(dtype=wp.float32),
        f: wp.array4d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
        cx_arr: wp.array(dtype=wp.int32),
        cy_arr: wp.array(dtype=wp.int32),
        cz_arr: wp.array(dtype=wp.int32),
        opp_arr: wp.array(dtype=wp.int32),
    ):
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        is_wall = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1)
        if is_wall:
            for q in range(19):
                oq = opp_arr[q]
                f[oq, i, j, k] = fnew[q, i, j, k]
        else:
            for q in range(19):
                si = i - cx_arr[q]; sj = j - cy_arr[q]; sk = k - cz_arr[q]
                if si < 0 or si >= nx or sj < 0 or sj >= ny or sk < 0 or sk >= nz:
                    oq = opp_arr[q]
                    f[q, i, j, k] = fnew[oq, i, j, k]
                else:
                    src_wall = (si == 0 or si == nx - 1 or sj == 0 or sj == ny - 1 or sk == 0 or sk == nz - 1)
                    if src_wall:
                        oq = opp_arr[q]
                        f[q, i, j, k] = fnew[oq, i, j, k]
                    else:
                        f[q, i, j, k] = fnew[q, si, sj, sk]

    @wp.kernel
    def copy_walls(
        f: wp.array4d(dtype=wp.float32),
        fnew: wp.array4d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
    ):
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        is_wall = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1)
        if is_wall:
            for q in range(19):
                fnew[q, i, j, k] = f[q, i, j, k]

    @wp.kernel
    def apply_moving_wall_interior(
        f: wp.array4d(dtype=wp.float32),
        fnew: wp.array4d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
        opp_arr: wp.array(dtype=wp.int32),
        w_arr: wp.array(dtype=wp.float32),
        cx_arr: wp.array(dtype=wp.int32),
        cy_arr: wp.array(dtype=wp.int32),
        ux_wall: wp.float32,
    ):
        i, k = wp.tid()
        if i >= nx or k >= nz:
            return
        j = ny - 2
        for q in range(19):
            if cy_arr[q] > 0:
                oq = opp_arr[q]
                w_oq = w_arr[oq]
                cx_oq = float(cx_arr[oq])
                f[q, i, j, k] = fnew[oq, i, j, k] - 6.0 * w_oq * cx_oq * ux_wall

    @wp.kernel
    def copy_f(
        f: wp.array4d(dtype=wp.float32),
        fsrc: wp.array4d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
    ):
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        for q in range(19):
            f[q, i, j, k] = fsrc[q, i, j, k]

    @wp.kernel
    def channel_inlet_zou_he(
        f: wp.array4d(dtype=wp.float32),
        ux_in: wp.float32,
        ny: wp.int32, nz: wp.int32,
    ):
        j, k = wp.tid()
        if j >= ny or k >= nz:
            return
        rho_in = (1.0 / (1.0 - ux_in)) * (
            f[0, 0, j, k] + f[3, 0, j, k] + f[4, 0, j, k] +
            f[5, 0, j, k] + f[6, 0, j, k] +
            f[15, 0, j, k] + f[16, 0, j, k] + f[17, 0, j, k] + f[18, 0, j, k] +
            2.0 * (f[2, 0, j, k] + f[8, 0, j, k] + f[10, 0, j, k] +
                   f[12, 0, j, k] + f[14, 0, j, k])
        )
        f[1, 0, j, k] = f[2, 0, j, k] + (1.0 / 3.0) * rho_in * ux_in
        diff_y = f[3, 0, j, k] - f[4, 0, j, k] - f[15, 0, j, k] + f[16, 0, j, k] - f[17, 0, j, k] + f[18, 0, j, k]
        diff_z = f[5, 0, j, k] - f[6, 0, j, k] - f[15, 0, j, k] + f[16, 0, j, k] - f[17, 0, j, k] + f[18, 0, j, k]
        f[7, 0, j, k] = f[10, 0, j, k] + (1.0 / 3.0) * rho_in * ux_in + 0.5 * diff_y - 0.5 * diff_z
        f[9, 0, j, k] = f[8, 0, j, k] + (1.0 / 3.0) * rho_in * ux_in - 0.5 * diff_y - 0.5 * diff_z
        f[11, 0, j, k] = f[14, 0, j, k] - 0.5 * diff_z
        f[13, 0, j, k] = f[12, 0, j, k] - 0.5 * diff_z

    @wp.kernel
    def channel_outlet_copy(
        f: wp.array4d(dtype=wp.float32),
        nx: wp.int32, ny: wp.int32, nz: wp.int32,
    ):
        j, k = wp.tid()
        if j >= ny or k >= nz:
            return
        for q in range(19):
            f[q, nx - 1, j, k] = f[q, nx - 2, j, k]

    return {
        'init_equilibrium': init_equilibrium,
        'compute_macroscopic': compute_macroscopic,
        'collide_bgk': collide_bgk,
        'stream_pull': stream_pull,
        'copy_walls': copy_walls,
        'apply_moving_wall_interior': apply_moving_wall_interior,
        'copy_f': copy_f,
        'channel_inlet_zou_he': channel_inlet_zou_he,
        'channel_outlet_copy': channel_outlet_copy,
    }


class WarpLBM:
    def __init__(self):
        import warp as wp
        self.wp = wp
        self.kernels = _define_warp_kernels()
        self.lat = {
            'w': wp.array(W_NP, dtype=wp.float32, device='cuda'),
            'cx': wp.array(CX_NP, dtype=wp.int32, device='cuda'),
            'cy': wp.array(CY_NP, dtype=wp.int32, device='cuda'),
            'cz': wp.array(CZ_NP, dtype=wp.int32, device='cuda'),
            'opp': wp.array(OPP_NP, dtype=wp.int32, device='cuda'),
        }

    def run_lid_cavity(self, nx, ny, nz, re, u_lid=0.1, max_steps=5000,
                       warmup_steps=100) -> dict[str, Any] | None:
        wp = self.wp
        k = self.kernels
        lat = self.lat

        nu = u_lid * (nx - 1) / re
        tau = 3.0 * nu + 0.5
        omega_val = 1.0 / tau
        if tau <= 0.51:
            print(f"  [warp] Lid cavity {nx}x{ny}x{nz}, Re={re}, tau={tau:.4f} — SKIPPED (unstable)")
            return None

        print(f"  [warp] Lid cavity {nx}x{ny}x{nz}, Re={re}, tau={tau:.4f}")

        f = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
        fnew = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
        rho = wp.array(np.ones((nx, ny, nz), dtype=np.float32), dtype=wp.float32, device='cuda')
        ux = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
        uy = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
        uz = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')

        wp.launch(k['init_equilibrium'], dim=(nx, ny, nz),
                  inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                          lat['w'], lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')
        wp.synchronize_device('cuda')

        all_times = []
        for _ in range(max_steps):
            t0 = time.perf_counter()
            wp.launch(k['compute_macroscopic'], dim=(nx, ny, nz),
                      inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                              lat['cx'], lat['cy'], lat['cz']],
                      device='cuda')
            wp.launch(k['collide_bgk'], dim=(nx, ny, nz),
                      inputs=[f, fnew, rho, ux, uy, uz, nx, ny, nz,
                              wp.float32(omega_val),
                              lat['w'], lat['cx'], lat['cy'], lat['cz']],
                      device='cuda')
            wp.launch(k['copy_walls'], dim=(nx, ny, nz),
                      inputs=[f, fnew, nx, ny, nz], device='cuda')
            wp.synchronize_device('cuda')
            wp.launch(k['stream_pull'], dim=(nx, ny, nz),
                      inputs=[fnew, f, nx, ny, nz,
                              lat['cx'], lat['cy'], lat['cz'], lat['opp']],
                      device='cuda')
            wp.launch(k['apply_moving_wall_interior'], dim=(nx, nz),
                      inputs=[f, f, nx, ny, nz, lat['opp'], lat['w'],
                              lat['cx'], lat['cy'], wp.float32(u_lid)],
                      device='cuda')
            wp.synchronize_device('cuda')
            all_times.append(time.perf_counter() - t0)

        wp.launch(k['compute_macroscopic'], dim=(nx, ny, nz),
                  inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                          lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')
        wp.synchronize_device('cuda')

        ux_host = ux.numpy()
        centerline_u = ux_host[nx // 2, :, nz // 2] / u_lid

        if np.any(np.isnan(centerline_u)):
            print(f"    WARNING: NaN detected — unstable")
            l2 = float('nan')
        else:
            l2 = compute_l2_ghia(centerline_u, ny, re)

        cells = nx * ny * nz
        mlups, sps, avg_ms = _safe_mlups(cells, all_times, warmup_steps)

        result = dict(
            backend='warp', nx=nx, ny=ny, nz=nz, re=re, tau=tau,
            mlups=mlups, steps_per_sec=sps,
            avg_step_ms=avg_ms, mem_mb=mem_estimate_mb(cells),
            cells=cells, l2_error=l2, steps=max_steps,
        )
        print(f"    MLUPS: {result['mlups']:.1f}, Step: {result['avg_step_ms']:.2f}ms, L2: {l2:.4f}")
        return result

    def run_channel_flow(self, nx=128, ny=64, nz=64, u_max=0.05,
                         max_steps=5000, warmup_steps=100) -> dict[str, Any]:
        wp = self.wp
        k = self.kernels
        lat = self.lat

        H = ny - 1
        tau = 0.8
        nu = (tau - 0.5) / 3.0
        omega_val = 1.0 / tau
        re = u_max * H / nu

        print(f"  [warp] Channel flow {nx}x{ny}x{nz}, Re={re:.1f}")

        f = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
        fnew = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
        rho = wp.array(np.ones((nx, ny, nz), dtype=np.float32), dtype=wp.float32, device='cuda')
        ux = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
        uy = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
        uz = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')

        wp.launch(k['init_equilibrium'], dim=(nx, ny, nz),
                  inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                          lat['w'], lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')
        wp.synchronize_device('cuda')

        all_times = []
        for _ in range(max_steps):
            t0 = time.perf_counter()
            wp.launch(k['compute_macroscopic'], dim=(nx, ny, nz),
                      inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                              lat['cx'], lat['cy'], lat['cz']],
                      device='cuda')
            wp.launch(k['collide_bgk'], dim=(nx, ny, nz),
                      inputs=[f, fnew, rho, ux, uy, uz, nx, ny, nz,
                              wp.float32(omega_val),
                              lat['w'], lat['cx'], lat['cy'], lat['cz']],
                      device='cuda')
            wp.launch(k['copy_walls'], dim=(nx, ny, nz),
                      inputs=[f, fnew, nx, ny, nz], device='cuda')
            wp.synchronize_device('cuda')
            wp.launch(k['stream_pull'], dim=(nx, ny, nz),
                      inputs=[fnew, f, nx, ny, nz,
                              lat['cx'], lat['cy'], lat['cz'], lat['opp']],
                      device='cuda')
            wp.launch(k['channel_inlet_zou_he'], dim=(ny, nz),
                      inputs=[f, wp.float32(u_max), ny, nz], device='cuda')
            wp.launch(k['channel_outlet_copy'], dim=(ny, nz),
                      inputs=[f, nx, ny, nz], device='cuda')
            wp.synchronize_device('cuda')
            all_times.append(time.perf_counter() - t0)

        wp.launch(k['compute_macroscopic'], dim=(nx, ny, nz),
                  inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                          lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')
        wp.synchronize_device('cuda')

        ux_host = ux.numpy()
        profile = ux_host[nx // 2, :, nz // 2]
        y_arr = np.arange(ny)
        analytical = u_max * 4.0 * (y_arr / H) * (1.0 - y_arr / H)
        interior = np.arange(1, ny - 1)

        if np.any(np.isnan(profile)):
            l2 = float('nan')
            max_u = float('nan')
        else:
            l2 = float(np.sqrt(np.mean((profile[interior] - analytical[interior]) ** 2)))
            max_u = float(np.max(np.abs(profile[interior])))

        cells = nx * ny * nz
        mlups, sps, avg_ms = _safe_mlups(cells, all_times, warmup_steps)

        result = dict(
            backend='warp', nx=nx, ny=ny, nz=nz, re=re, tau=tau,
            mlups=mlups, steps_per_sec=sps,
            avg_step_ms=avg_ms, mem_mb=mem_estimate_mb(cells),
            cells=cells, l2_error=l2, u_max_analytical=u_max,
            max_u=max_u, steps=max_steps,
        )
        print(f"    MLUPS: {result['mlups']:.1f}, Step: {result['avg_step_ms']:.2f}ms, L2: {l2:.6f}")
        return result


# ===================================================================
# Unified harness
# ===================================================================

def run_benchmark(backends: list[str], sizes: list[int], steps: int):
    print("=" * 70)
    print("LBM D3Q19 Benchmark — PyTorch vs Warp")
    print("=" * 70)
    print(f"GPU memory before: {gpu_mem_mb()} MB")
    print(f"Backends: {', '.join(backends)}")
    print(f"Sizes: {sizes}, Steps: {steps}")

    solvers: dict[str, TorchLBM | WarpLBM] = {}
    for b in backends:
        if b == 'torch':
            if not _torch_available():
                print("WARNING: PyTorch CUDA not available, skipping torch backend")
                continue
            solvers['torch'] = TorchLBM()
        elif b == 'warp':
            if not _warp_available():
                print("WARNING: Warp CUDA not available, skipping warp backend")
                continue
            solvers['warp'] = WarpLBM()

    if not solvers:
        print("ERROR: No backends available")
        return

    # Warmup
    print("\nWarmup...")
    for name, solver in solvers.items():
        solver.run_lid_cavity(16, 16, 16, re=100, max_steps=50, warmup_steps=5)
    print("Warmup done.")

    # Lid-driven cavity
    print("\n" + "-" * 50)
    print("LID-DRIVEN CAVITY")
    print("-" * 50)

    cavity_results: list[dict[str, Any]] = []
    for n in sizes:
        max_s = min(steps, 5000 if n <= 64 else (3000 if n <= 128 else 2000))
        for re in [100, 400, 1000]:
            for name, solver in solvers.items():
                r = solver.run_lid_cavity(n, n, n, re=re, max_steps=max_s, warmup_steps=100)
                if r is not None:
                    cavity_results.append(r)

    # Channel flow
    print("\n" + "-" * 50)
    print("CHANNEL FLOW (POISEUILLE)")
    print("-" * 50)

    channel_results: list[dict[str, Any]] = []
    for name, solver in solvers.items():
        r = solver.run_channel_flow(nx=128, ny=64, nz=64, max_steps=min(steps, 5000), warmup_steps=100)
        channel_results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Backend':>8} {'Grid':>12} {'Re':>6} {'MLUPS':>10} {'Steps/s':>10} {'Step(ms)':>10} {'Mem(MB)':>8} {'L2':>10}")
    print("-" * 86)
    for r in cavity_results:
        print(f"{r['backend']:>8} {r['nx']:>4}x{r['ny']}x{r['nz']:>4} {r['re']:>6} "
              f"{r['mlups']:>10.1f} {r['steps_per_sec']:>10.1f} {r['avg_step_ms']:>10.2f} "
              f"{r['mem_mb']:>8.1f} {r['l2_error']:>10.4f}")

    if len(solvers) > 1:
        print("\nSpeedup (Warp / PyTorch):")
        for n in sizes:
            for re in [100, 400, 1000]:
                torch_r = [r for r in cavity_results if r['backend'] == 'torch' and r['nx'] == n and r['re'] == re]
                warp_r = [r for r in cavity_results if r['backend'] == 'warp' and r['nx'] == n and r['re'] == re]
                if torch_r and warp_r:
                    speedup = warp_r[0]['mlups'] / torch_r[0]['mlups']
                    print(f"  {n}^3 Re={re}: {speedup:.2f}x")

    print(f"\nChannel flow:")
    for r in channel_results:
        print(f"  [{r['backend']}] MLUPS: {r['mlups']:.1f}, L2: {r['l2_error']:.6f}")

    print(f"\nGPU memory after: {gpu_mem_mb()} MB")


def main():
    parser = argparse.ArgumentParser(description="D3Q19 LBM benchmark: PyTorch vs Warp")
    parser.add_argument("--backend", choices=["torch", "warp", "both"], default="both",
                        help="Backend to benchmark (default: both)")
    parser.add_argument("--size", type=int, nargs="+", default=[64, 128],
                        help="Grid sizes to test (default: 64 128)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Max simulation steps (default: 5000)")
    args = parser.parse_args()

    backends = ["torch", "warp"] if args.backend == "both" else [args.backend]
    run_benchmark(backends, args.size, max(args.steps, 10))


if __name__ == "__main__":
    main()
