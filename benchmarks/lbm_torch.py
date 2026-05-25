"""D3Q19 Lattice Boltzmann Method solver — PyTorch vectorized.

Same physics as lbm_warp.py for direct performance comparison.
Uses PyTorch tensor operations (no custom CUDA kernels).
"""

from __future__ import annotations

import subprocess
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# D3Q19 lattice constants
# ---------------------------------------------------------------------------

CX = torch.tensor([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=torch.int32)
CY = torch.tensor([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=torch.int32)
CZ = torch.tensor([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=torch.int32)

W = torch.tensor([
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=torch.float32)

OPP = torch.tensor([0, 2, 1, 4, 3, 6, 5, 9, 8, 7, 10, 13, 12, 11, 14, 17, 16, 15, 18], dtype=torch.int32)

Q = 19

# Ghia benchmark data (same as warp version)
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


def equilibrium(rho, ux, uy, uz):
    """Compute equilibrium distribution function."""
    device = rho.device
    cx = CX.to(device)
    cy = CY.to(device)
    cz = CZ.to(device)
    w = W.to(device)
    cu = cx.reshape(Q, 1, 1, 1) * ux.unsqueeze(0) + \
         cy.reshape(Q, 1, 1, 1) * uy.unsqueeze(0) + \
         cz.reshape(Q, 1, 1, 1) * uz.unsqueeze(0)

    usq = ux * ux + uy * uy + uz * uz  # (nx, ny, nz)

    w2 = w.reshape(Q, 1, 1, 1)
    return w2 * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0))


def macroscopic(f):
    """Compute density and velocity from distribution functions.
    Wall nodes are skipped to avoid NaN from bounce-back distributions.
    """
    device = f.device
    cx = CX.to(device).reshape(Q, 1, 1, 1)
    cy = CY.to(device).reshape(Q, 1, 1, 1)
    cz = CZ.to(device).reshape(Q, 1, 1, 1)
    rho = f.sum(dim=0)  # (nx, ny, nz)
    ux = (f * cx).sum(dim=0) / rho
    uy = (f * cy).sum(dim=0) / rho
    uz = (f * cz).sum(dim=0) / rho

    # Wall nodes: force valid macroscopic values
    nx, ny, nz = f.shape[1], f.shape[2], f.shape[3]
    wall = torch.zeros(nx, ny, nz, dtype=torch.bool, device=device)
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


def collide_stream(f, rho, ux, uy, uz, omega):
    """BGK collision + streaming with bounce-back."""
    feq = equilibrium(rho, ux, uy, uz)
    f_post = f - omega * (f - feq)

    # Streaming: shift each direction by its velocity
    fnew = torch.zeros_like(f)
    for q in range(Q):
        fnew[q] = torch.roll(torch.roll(torch.roll(f_post[q], CX[q].item(), dims=0),
                                         CY[q].item(), dims=1),
                              CZ[q].item(), dims=2)

    # Bounce-back on walls (overwrite streamed values at boundaries)
    nx, ny, nz = f.shape[1], f.shape[2], f.shape[3]

    # For each wall, the distributions pointing INTO the wall get bounced back
    # x=0 wall: directions with cx=-1 (going left) bounce back
    # x=nx-1 wall: directions with cx=+1 (going right) bounce back
    for q in range(Q):
        oq = OPP[q].item()
        # x boundaries
        if CX[q].item() > 0:
            fnew[q, nx-1, :, :] = f_post[oq, nx-1, :, :]
        elif CX[q].item() < 0:
            fnew[q, 0, :, :] = f_post[oq, 0, :, :]

        # y boundaries
        if CY[q].item() > 0:
            fnew[q, :, ny-1, :] = f_post[oq, :, ny-1, :]
        elif CY[q].item() < 0:
            fnew[q, :, 0, :] = f_post[oq, :, 0, :]

        # z boundaries
        if CZ[q].item() > 0:
            fnew[q, :, :, nz-1] = f_post[oq, :, :, nz-1]
        elif CZ[q].item() < 0:
            fnew[q, :, :, 0] = f_post[oq, :, :, 0]

    return fnew


def apply_lid_velocity_zou_he(f, ux_top, ny):
    """Apply Zou-He velocity BC on top wall (y=ny-1) for lid-driven cavity."""
    j = ny - 1
    # Top wall: moving in x with velocity ux_top, uy=0, uz=0
    # Directions pointing down (into fluid): 4, 9, 10, 16, 18
    rho_wall = (1.0 / (1.0 + 0.0)) * (
        f[0, :, j, :] + f[1, :, j, :] + f[2, :, j, :] +
        f[5, :, j, :] + f[6, :, j, :] +
        2.0 * (f[4, :, j, :] + f[9, :, j, :] + f[10, :, j, :] +
               f[16, :, j, :] + f[18, :, j, :])
    )

    f[4, :, j, :] = f[3, :, j, :] - (1.0 / 3.0) * rho_wall * 0.0
    diff_12 = f[1, :, j, :] - f[2, :, j, :]
    f[9, :, j, :] = f[7, :, j, :] - 0.5 * diff_12 - 0.5 * rho_wall * ux_top
    f[10, :, j, :] = f[8, :, j, :] + 0.5 * diff_12 + 0.5 * rho_wall * ux_top
    f[16, :, j, :] = f[15, :, j, :] - 0.5 * diff_12 - 0.5 * rho_wall * ux_top
    f[18, :, j, :] = f[17, :, j, :] + 0.5 * diff_12 + 0.5 * rho_wall * ux_top

    return f


# ---------------------------------------------------------------------------
# Lid-driven cavity
# ---------------------------------------------------------------------------

def run_lid_cavity(nx, ny, nz, re, u_lid=0.05, max_steps=5000, warmup_steps=100):
    """Run lid-driven cavity with PyTorch LBM."""
    nu = u_lid * (nx - 1) / re
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    print(f"\n  Lid-driven cavity: {nx}x{ny}x{nz}, Re={re}, tau={tau:.4f}, omega={omega:.4f}")

    device = torch.device('cuda')

    # Init
    rho = torch.ones(nx, ny, nz, dtype=torch.float32, device=device)
    ux = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)
    uy = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)
    uz = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)

    f = equilibrium(rho, ux, uy, uz).to(device)

    CX.to(device)
    CY.to(device)
    CZ.to(device)

    all_times = []

    for _step in range(max_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Macroscopic
        rho, ux, uy, uz = macroscopic(f)

        # Apply lid velocity (Zou-He on top wall)
        f = apply_lid_velocity_zou_he(f, u_lid, ny)

        # Collide + stream
        f = collide_stream(f, rho, ux, uy, uz, omega)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        all_times.append(dt)

    # Final validation
    rho, ux_final, _uy_final, _uz_final = macroscopic(f)
    torch.cuda.synchronize()

    ux_host = ux_final.cpu().numpy()
    centerline_u = ux_host[nx // 2, :, nz // 2] / u_lid

    # Compare with Ghia
    y_positions = np.arange(ny) / (ny - 1)
    ghia_y = GHIA_RE100_Y if re == 100 else (GHIA_RE400_Y if re == 400 else GHIA_RE1000_Y)
    ghia_u = GHIA_RE100_U if re == 100 else (GHIA_RE400_U if re == 400 else GHIA_RE1000_U)
    our_u_interp = np.interp(ghia_y, y_positions, centerline_u)
    mask = (ghia_y > 0.01) & (ghia_y < 0.99)
    l2_error = np.sqrt(np.mean((our_u_interp[mask] - ghia_u[mask]) ** 2)) if np.sum(mask) > 2 else float('nan')

    # Performance
    times_arr = np.array(all_times[warmup_steps:])
    avg_time = np.mean(times_arr)
    cells = nx * ny * nz
    mlups = cells * 1e-6 / avg_time
    steps_per_sec = 1.0 / avg_time
    total_mem_bytes = Q * cells * 4 * 2 + cells * 4 * 4  # float32
    mem_mb = total_mem_bytes / (1024 * 1024)

    result = {
        'nx': nx, 'ny': ny, 'nz': nz, 're': re,
        'tau': tau, 'omega': omega,
        'mlups': mlups, 'steps_per_sec': steps_per_sec,
        'avg_step_ms': avg_time * 1000,
        'mem_mb': mem_mb, 'cells': cells,
        'l2_error': l2_error,
        'max_u': float(np.max(np.abs(centerline_u))),
        'steps': max_steps,
    }

    print(f"    MLUPS: {mlups:.1f}, Steps/s: {steps_per_sec:.1f}, "
          f"Avg step: {avg_time*1000:.2f}ms, Mem: {mem_mb:.1f}MB, L2: {l2_error:.4f}")

    return result


# ---------------------------------------------------------------------------
# Channel flow (Poiseuille)
# ---------------------------------------------------------------------------

def run_channel_flow(nx=128, ny=64, nz=64, u_max=0.05, max_steps=5000, warmup_steps=100):
    """Run Poiseuille channel flow."""
    H = ny - 1
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    8.0 * nu * u_max / (H * H)
    omega = 1.0 / tau
    re = u_max * H / nu

    print(f"\n  Channel flow: {nx}x{ny}x{nz}, Re={re:.1f}, tau={tau:.4f}")

    device = torch.device('cuda')
    rho = torch.ones(nx, ny, nz, dtype=torch.float32, device=device)
    ux = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)
    uy = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)
    uz = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)

    f = equilibrium(rho, ux, uy, uz).to(device)

    all_times = []

    for _step in range(max_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        rho, ux, uy, uz = macroscopic(f)

        # Collide + stream
        f = collide_stream(f, rho, ux, uy, uz, omega)

        # Zou-He inlet at x=0
        rho_in = (1.0 / (1.0 - u_max)) * (
            f[0, 0, :, :] + f[3, 0, :, :] + f[4, 0, :, :] +
            f[5, 0, :, :] + f[6, 0, :, :] +
            2.0 * (f[2, 0, :, :] + f[8, 0, :, :] + f[10, 0, :, :] +
                   f[12, 0, :, :] + f[16, 0, :, :])
        )
        f[1, 0, :, :] = f[2, 0, :, :] + (2.0 / 3.0) * rho_in * u_max
        f[9, 0, :, :] = f[8, 0, :, :] - 0.5 * (f[3, 0, :, :] - f[4, 0, :, :]) + (1.0 / 6.0) * rho_in * u_max
        f[10, 0, :, :] = f[7, 0, :, :] + 0.5 * (f[3, 0, :, :] - f[4, 0, :, :]) + (1.0 / 6.0) * rho_in * u_max
        f[11, 0, :, :] = f[12, 0, :, :] - 0.5 * (f[5, 0, :, :] - f[6, 0, :, :]) + (1.0 / 6.0) * rho_in * u_max
        f[16, 0, :, :] = f[17, 0, :, :] + 0.5 * (f[5, 0, :, :] - f[6, 0, :, :]) + (1.0 / 6.0) * rho_in * u_max

        # Open outlet at x=nx-1
        f[:, nx-1, :, :] = f[:, nx-2, :, :]

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        all_times.append(dt)

    # Final validation
    rho, ux_final, _uy_final, _uz_final = macroscopic(f)
    torch.cuda.synchronize()

    ux_host = ux_final.cpu().numpy()
    profile = ux_host[nx // 2, :, nz // 2]

    y_arr = np.arange(ny)
    analytical = u_max * 4.0 * (y_arr / H) * (1.0 - y_arr / H)
    interior = np.arange(1, ny - 1)
    l2_error = np.sqrt(np.mean((profile[interior] - analytical[interior]) ** 2))
    max_u_measured = np.max(np.abs(profile[interior]))

    times_arr = np.array(all_times[warmup_steps:])
    avg_time = np.mean(times_arr)
    cells = nx * ny * nz
    mlups = cells * 1e-6 / avg_time
    steps_per_sec = 1.0 / avg_time
    total_mem_bytes = Q * cells * 4 * 2 + cells * 4 * 4  # float32
    mem_mb = total_mem_bytes / (1024 * 1024)

    result = {
        'nx': nx, 'ny': ny, 'nz': nz, 're': re,
        'tau': tau, 'omega': omega,
        'mlups': mlups, 'steps_per_sec': steps_per_sec,
        'avg_step_ms': avg_time * 1000,
        'mem_mb': mem_mb, 'cells': cells,
        'l2_error': l2_error,
        'max_u': float(max_u_measured),
        'u_max_analytical': u_max,
        'steps': max_steps,
    }

    print(f"    MLUPS: {mlups:.1f}, Steps/s: {steps_per_sec:.1f}, "
          f"Avg step: {avg_time*1000:.2f}ms, Mem: {mem_mb:.1f}MB, "
          f"L2: {l2_error:.6f}, max_u: {max_u_measured:.4f} (analytical: {u_max:.4f})")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LBM D3Q19 PyTorch Vectorized Benchmark — RTX 5090")
    print("=" * 70)

    mem_before = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    print(f"GPU memory before: {mem_before} MB")

    # Warmup
    print("\nWarmup...")
    run_lid_cavity(16, 16, 16, re=100, max_steps=50, warmup_steps=5)
    print("Warmup done.")

    # --- Lid-driven cavity ---
    print("\n" + "-" * 50)
    print("LID-DRIVEN CAVITY")
    print("-" * 50)

    cavity_results = []
    for n in [64, 128]:
        for re in [100, 400, 1000]:
            steps = 5000 if n == 64 else 3000
            r = run_lid_cavity(n, n, n, re=re, max_steps=steps, warmup_steps=100)
            cavity_results.append(r)

    # 256^3 at Re=100
    r = run_lid_cavity(256, 256, 256, re=100, max_steps=2000, warmup_steps=50)
    cavity_results.append(r)

    # --- Channel flow ---
    print("\n" + "-" * 50)
    print("CHANNEL FLOW (POISEUILLE)")
    print("-" * 50)

    channel_result = run_channel_flow(nx=128, ny=64, nz=64, max_steps=5000, warmup_steps=100)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY — PyTorch LBM D3Q19")
    print("=" * 70)

    print("\nLid-driven cavity:")
    print(f"{'Grid':>12} {'Re':>6} {'MLUPS':>10} {'Steps/s':>10} {'Step(ms)':>10} {'Mem(MB)':>10} {'L2 err':>10}")
    for r in cavity_results:
        print(f"{r['nx']:>4}x{r['ny']}x{r['nz']:>4} {r['re']:>6} {r['mlups']:>10.1f} "
              f"{r['steps_per_sec']:>10.1f} {r['avg_step_ms']:>10.2f} {r['mem_mb']:>10.1f} {r['l2_error']:>10.4f}")

    print("\nChannel flow:")
    r = channel_result
    print(f"  Grid: {r['nx']}x{r['ny']}x{r['nz']}, Re={r['re']:.1f}")
    print(f"  MLUPS: {r['mlups']:.1f}, Steps/s: {r['steps_per_sec']:.1f}")
    print(f"  L2 error vs analytical: {r['l2_error']:.6f}")
    print(f"  Max velocity: {r['max_u']:.4f} (analytical: {r['u_max_analytical']:.4f})")

    mem_after = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    print(f"\nGPU memory after: {mem_after} MB")

    for r in cavity_results:
        bytes_per_cell = r['mem_mb'] * 1024 * 1024 / r['cells']
        print(f"  {r['nx']}^3: {bytes_per_cell:.1f} bytes/cell")

    return cavity_results, channel_result


if __name__ == "__main__":
    main()
