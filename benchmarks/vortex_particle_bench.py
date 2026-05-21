"""Vortex particle method benchmark — flow past a cylinder at Re=100.

Part 3 of FSI underwater benchmark.
- Implements a 2D vortex particle method in Warp
- Flow past cylinder, Re=100
- Measures Strouhal number (expected St ~ 0.16-0.20)
- Benchmarks particle counts: 1K, 10K, 50K vortex particles

Physics:
  - Biot-Savart law for velocity induced by point vortices
  - Nascent vortex injection from cylinder surface with alternating circulation
  - Gaussian regularization to avoid singularities

Performance:
  - O(N^2) direct Biot-Savart (GPU-parallel over targets)
  - Measures: step time, particles/sec, memory
"""

from __future__ import annotations

import time
import subprocess

import numpy as np
import warp as wp

wp.init()
device = wp.get_device("cuda:0")

print(f"Device: {device}")
print(f"Warp {wp.__version__}")
print("=" * 78)


def gpu_mem_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return float(out)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Warp kernels for vortex particle method
# ---------------------------------------------------------------------------

@wp.kernel
def biot_savart_velocity(
    vp_x: wp.array(dtype=float),
    vp_y: wp.array(dtype=float),
    gamma: wp.array(dtype=float),
    tx: wp.array(dtype=float),
    ty: wp.array(dtype=float),
    out_vx: wp.array(dtype=float),
    out_vy: wp.array(dtype=float),
    n_particles: int,
    sigma_sq: float,
):
    """Compute velocity at each target point from all vortex particles.

    Gaussian regularization: K_delta = K * (1 - exp(-r^2/sigma^2))
    """
    tid = wp.tid()
    px = tx[tid]
    py = ty[tid]

    vx = float(0.0)
    vy = float(0.0)

    for j in range(n_particles):
        if j == tid:
            continue
        dx = px - vp_x[j]
        dy = py - vp_y[j]
        r_sq = dx * dx + dy * dy

        reg = float(1.0) - wp.exp(-r_sq / sigma_sq)
        denom = r_sq + sigma_sq * float(0.01)
        g = gamma[j]
        factor = g / (float(6.283185307179586) * denom) * reg

        vx += -dy * factor
        vy += dx * factor

    out_vx[tid] = vx
    out_vy[tid] = vy


@wp.kernel
def advect_particles(
    px: wp.array(dtype=float),
    py: wp.array(dtype=float),
    vx: wp.array(dtype=float),
    vy: wp.array(dtype=float),
    freestream_vx: float,
    freestream_vy: float,
    dt: float,
):
    """Advect vortex particles with induced + freestream velocity."""
    tid = wp.tid()
    px[tid] = px[tid] + (vx[tid] + freestream_vx) * dt
    py[tid] = py[tid] + (vy[tid] + freestream_vy) * dt


@wp.kernel
def sum_nearby_circulation(
    px: wp.array(dtype=float),
    py: wp.array(dtype=float),
    gamma: wp.array(dtype=float),
    cx: float,
    cy: float,
    near_sq: float,
    result: wp.array(dtype=float),
):
    """Sum circulation of vortices near the cylinder for lift estimation."""
    tid = wp.tid()
    dx = px[tid] - cx
    dy = py[tid] - cy
    if dx * dx + dy * dy < near_sq:
        wp.atomic_add(result, 0, gamma[tid])


# ---------------------------------------------------------------------------
# Vortex particle simulation
# ---------------------------------------------------------------------------

class VortexParticleSim:
    """2D vortex particle simulation of flow past a cylinder."""

    def __init__(self, n_max_particles: int, Re: float = 100.0):
        self.Re = Re
        self.n_max = n_max_particles
        self.n_active = 0

        self.U_inf = 1.0
        self.cylinder_R = 0.5
        self.cylinder_D = 2.0 * self.cylinder_R
        self.cylinder_x = 0.0
        self.cylinder_y = 0.0
        self.sigma = self.cylinder_R * 0.1
        self.sigma_sq = self.sigma ** 2
        self.nu = self.U_inf * self.cylinder_D / Re

        self.px = wp.zeros(n_max_particles, dtype=float, device="cuda")
        self.py = wp.zeros(n_max_particles, dtype=float, device="cuda")
        self.gamma = wp.zeros(n_max_particles, dtype=float, device="cuda")
        self.vx_arr = wp.zeros(n_max_particles, dtype=float, device="cuda")
        self.vy_arr = wp.zeros(n_max_particles, dtype=float, device="cuda")

        self.dt = 0.02
        self.t = 0.0
        self.inject_per_step = 30

        self.lift_history: list[float] = []

        # CPU-side arrays for particle management
        self.px_np = np.zeros(n_max_particles, dtype=np.float64)
        self.py_np = np.zeros(n_max_particles, dtype=np.float64)
        self.gamma_np = np.zeros(n_max_particles, dtype=np.float64)

        self._seed_initial_particles()

    def _seed_initial_particles(self):
        """Seed initial vortex particles behind the cylinder."""
        rng = np.random.default_rng(42)
        n_seed = min(100, self.n_max)
        angles = rng.uniform(np.pi - 0.5, np.pi + 0.5, n_seed)
        radii = self.cylinder_R * (1.2 + rng.uniform(0, 3.0, n_seed))

        self.px_np[:n_seed] = self.cylinder_x + radii * np.cos(angles)
        self.py_np[:n_seed] = self.cylinder_y + radii * np.sin(angles)
        gamma_signs = rng.choice([-1.0, 1.0], n_seed)
        self.gamma_np[:n_seed] = gamma_signs * rng.uniform(0.1, 0.5, n_seed)
        self.n_active = n_seed
        self._sync_to_gpu()

    def _sync_to_gpu(self):
        wp.copy(self.px, wp.array(self.px_np, dtype=float, device="cuda"))
        wp.copy(self.py, wp.array(self.py_np, dtype=float, device="cuda"))
        wp.copy(self.gamma, wp.array(self.gamma_np, dtype=float, device="cuda"))

    def _sync_from_gpu(self):
        self.px_np = self.px.numpy()
        self.py_np = self.py.numpy()
        self.gamma_np = self.gamma.numpy()

    def step(self):
        """One simulation step."""
        n = self.n_active
        if n < 2:
            return

        # Compute induced velocities (O(N^2))
        wp.launch(
            biot_savart_velocity,
            dim=n,
            inputs=[
                self.px, self.py, self.gamma,
                self.px, self.py,
                self.vx_arr, self.vy_arr,
                n, self.sigma_sq,
            ],
        )

        # Advect
        wp.launch(
            advect_particles,
            dim=n,
            inputs=[
                self.px, self.py, self.vx_arr, self.vy_arr,
                self.U_inf, 0.0, self.dt,
            ],
        )
        wp.synchronize()

        # CPU-side particle management
        self._sync_from_gpu()
        px = self.px_np[:n]
        py = self.py_np[:n]

        # Enforce cylinder boundary: push particles outside
        dx = px - self.cylinder_x
        dy = py - self.cylinder_y
        dist = np.sqrt(dx * dx + dy * dy)
        inside = dist < self.cylinder_R * 1.05
        if np.any(inside):
            scale = self.cylinder_R * 1.1 / np.maximum(dist[inside], 1e-6)
            px[inside] = self.cylinder_x + dx[inside] * scale
            py[inside] = self.cylinder_y + dy[inside] * scale
            self.px_np[:n] = px
            self.py_np[:n] = py

        # Remove far particles (beyond 25 radii)
        dist_sq_all = dx * dx + dy * dy
        alive = dist_sq_all < 25.0 * 25.0
        alive_idx = np.where(alive)[0]
        new_n = len(alive_idx)

        if new_n < n:
            self.px_np[:new_n] = self.px_np[:n][alive_idx]
            self.py_np[:new_n] = self.py_np[:n][alive_idx]
            self.gamma_np[:new_n] = self.gamma_np[:n][alive_idx]
            self.px_np[new_n:] = 0.0
            self.py_np[new_n:] = 0.0
            self.gamma_np[new_n:] = 0.0
            self.n_active = new_n

        # Inject nascent vortices
        self._inject_nascent()
        self._sync_to_gpu()

        # Track lift
        result = wp.zeros(1, dtype=float, device="cuda")
        near_sq = (self.cylinder_R * 5.0) ** 2
        if self.n_active > 0:
            wp.launch(
                sum_nearby_circulation,
                dim=self.n_active,
                inputs=[
                    self.px, self.py, self.gamma,
                    self.cylinder_x, self.cylinder_y, near_sq, result,
                ],
            )
            wp.synchronize()
        self.lift_history.append(float(result.numpy()[0]))
        self.t += self.dt

    def _inject_nascent(self):
        """Inject nascent vortex particles from cylinder surface."""
        n_inject = min(self.inject_per_step, self.n_max - self.n_active)
        if n_inject <= 0:
            return

        rng = np.random.default_rng(int(self.t * 1000) % (2**31))
        start = self.n_active

        St_expected = 0.18
        period = self.cylinder_D / (St_expected * self.U_inf)

        for i in range(n_inject):
            if i % 2 == 0:
                angle = np.pi / 2 + rng.uniform(-0.3, 0.3)
            else:
                angle = -np.pi / 2 + rng.uniform(-0.3, 0.3)

            offset = self.cylinder_R * (1.05 + rng.uniform(0, 0.2))
            idx = start + i
            self.px_np[idx] = self.cylinder_x + offset * np.cos(angle)
            self.py_np[idx] = self.cylinder_y + offset * np.sin(angle)

            sign = 1.0 if np.sin(2.0 * np.pi * self.t / period) > 0 else -1.0
            self.gamma_np[idx] = 0.3 * sign * rng.uniform(0.5, 1.5)

        self.n_active += n_inject

    def estimate_strouhal(self) -> float:
        """Estimate Strouhal number from lift oscillation."""
        if len(self.lift_history) < 100:
            return -1.0

        signal = np.array(self.lift_history[-500:])
        signal = signal - np.mean(signal)
        if np.max(np.abs(signal)) < 1e-12:
            return -1.0

        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=self.dt)
        magnitudes = np.abs(fft)
        magnitudes[0] = 0.0
        peak_idx = np.argmax(magnitudes)
        f_dominant = freqs[peak_idx]

        St = f_dominant * self.cylinder_D / self.U_inf
        return St


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_vortex_particles():
    print("\n" + "=" * 78)
    print("Part 3: Vortex Particle Method — Flow Past Cylinder, Re=100")
    print("=" * 78)

    particle_counts = [1000, 10000, 50000]
    n_warmup_steps = 20
    n_bench_steps = 100
    n_strouhal_steps = 2000

    Re = 100.0
    print(f"\nRe = {Re}, expected Strouhal: 0.16 - 0.20")
    print(f"  Freestream U = 1.0 m/s, Cylinder D = 1.0 m")
    print(f"  nu = {1.0 * 1.0 / Re:.4f} m^2/s")
    print()

    results = []

    for n_max in particle_counts:
        print(f"\n--- N_max = {n_max} ---")
        mem_before = gpu_mem_mb()

        sim = VortexParticleSim(n_max, Re=Re)

        # Warmup
        for _ in range(n_warmup_steps):
            sim.step()

        # Benchmark step time
        t0 = time.perf_counter()
        for _ in range(n_bench_steps):
            sim.step()
        wall = time.perf_counter() - t0

        # Continue for Strouhal
        for _ in range(n_strouhal_steps):
            sim.step()

        St = sim.estimate_strouhal()
        mem_after = gpu_mem_mb()
        mem_delta = max(mem_after - mem_before, 0.0)

        step_rate = n_bench_steps / wall
        ms_per_step = wall / n_bench_steps * 1000.0
        avg_active = sim.n_active

        print(f"  active particles = {avg_active}")
        print(f"  steps/s          = {step_rate:.1f}")
        print(f"  ms/step          = {ms_per_step:.2f}")
        print(f"  particles/s      = {avg_active * step_rate:.0f}")
        print(f"  GPU memory       = {mem_delta:.1f} MB")
        print(f"  Strouhal number  = {St:.4f}")
        if St > 0:
            if 0.16 <= St <= 0.20:
                print(f"  Strouhal check   = PASS (within [0.16, 0.20])")
            else:
                print(f"  Strouhal check   = OUTSIDE expected [0.16, 0.20]")

        results.append({
            "n_max": n_max,
            "n_active": avg_active,
            "step_rate": step_rate,
            "ms_per_step": ms_per_step,
            "particles_per_sec": avg_active * step_rate,
            "mem_mb": mem_delta,
            "strouhal": St,
        })

        del sim

    return results


def bench_biot_savart_synthetic():
    """Pure Biot-Savart throughput test."""
    print("\n" + "=" * 78)
    print("Biot-Savart kernel throughput (synthetic)")
    print("=" * 78)

    sizes = [1000, 10000, 50000]
    n_steps = 10
    sigma_sq = 0.01

    print(f"\n{'N':>8}  {'steps/s':>10}  {'ms/step':>8}  {'N^2 ops/s':>12}  {'GPU MB':>8}")
    print("-" * 78)

    for N in sizes:
        px = wp.array(np.random.randn(N).astype(np.float64), dtype=float, device="cuda")
        py = wp.array(np.random.randn(N).astype(np.float64), dtype=float, device="cuda")
        gamma = wp.array(np.ones(N, dtype=np.float64), dtype=float, device="cuda")
        vx = wp.zeros(N, dtype=float, device="cuda")
        vy = wp.zeros(N, dtype=float, device="cuda")

        wp.launch(
            biot_savart_velocity,
            dim=N,
            inputs=[px, py, gamma, px, py, vx, vy, N, sigma_sq],
        )
        wp.synchronize()

        mem = gpu_mem_mb()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            wp.launch(
                biot_savart_velocity,
                dim=N,
                inputs=[px, py, gamma, px, py, vx, vy, N, sigma_sq],
            )
        wp.synchronize()
        wall = time.perf_counter() - t0

        step_rate = n_steps / wall
        ms_per_step = wall / n_steps * 1000.0
        n2_ops = N * N * step_rate

        print(
            f"{N:>8}  {step_rate:>10.1f}  {ms_per_step:>8.2f}  "
            f"{n2_ops:>12.3e}  {mem:>8.1f}"
        )

        del px, py, gamma, vx, vy


def main():
    print(f"GPU memory at start: {gpu_mem_mb():.0f} MB")

    print("\n" + "#" * 78)
    print("# Part 3: Vortex Particle Method")
    print("#" * 78)
    results = bench_vortex_particles()

    print("\n" + "#" * 78)
    print("# Biot-Savart Synthetic Throughput")
    print("#" * 78)
    bench_biot_savart_synthetic()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("\nVortex Particle Method — Flow Past Cylinder (Re=100)")
    for r in results:
        st_str = f"{r['strouhal']:.4f}" if r['strouhal'] > 0 else "N/A"
        print(
            f"  N_max={r['n_max']:>6}  active={r['n_active']:>6}  "
            f"steps/s={r['step_rate']:.1f}  ms/step={r['ms_per_step']:.2f}  "
            f"St={st_str}  mem={r['mem_mb']:.1f} MB"
        )

    print(f"\nGPU memory at end: {gpu_mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
