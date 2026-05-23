"""Advanced SPH benchmarks — spatial hash grid, full WCSPH dynamics, multi-resolution.

Tests:
1. Spatial hash grid neighbor search vs brute-force (Warp HashGrid)
2. Full 2D dam-break WCSPH simulation (cubic spline kernel, Tait EOS)
3. Multi-resolution: varying smoothing length h

Hardware target: RTX 5090 (sm_120, CUDA 12.8)
"""

from __future__ import annotations

import time

import numpy as np

import warp as wp

wp.init()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def warmup(kernel, dim, inputs, device="cuda", iters=10):
    for _ in range(iters):
        wp.launch(kernel, dim=dim, inputs=inputs, device=device)
    wp.synchronize_device(device)


# ---------------------------------------------------------------------------
# Part 0: Brute-force neighbor search (baseline, from previous bench)
# ---------------------------------------------------------------------------

@wp.kernel
def sph_count_neighbors_bruteforce(
    pos: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    n: wp.int32,
    h: wp.float32,
):
    i = wp.tid()
    pi = pos[i]
    c = wp.int32(0)
    h2 = h * h
    for j in range(n):
        if i != j:
            d = pos[j] - pi
            dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if dist2 < h2:
                c += 1
    counts[i] = c


# ---------------------------------------------------------------------------
# Part 1: Spatial Hash Grid neighbor search (Warp built-in HashGrid)
# ---------------------------------------------------------------------------

@wp.kernel
def sph_count_neighbors_hashgrid(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    h: wp.float32,
):
    i = wp.tid()
    pi = pos[i]
    c = wp.int32(0)
    h2 = h * h
    query = wp.hash_grid_query(grid, pi, h)
    neighbor_idx = wp.int32(0)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx != i:
            d = pos[neighbor_idx] - pi
            dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if dist2 < h2:
                c += 1
    counts[i] = c


@wp.kernel
def sph_count_neighbors_hashgrid_ordered(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    h: wp.float32,
):
    i = wp.tid()
    idx = wp.hash_grid_point_id(grid, i)
    pi = pos[idx]
    c = wp.int32(0)
    h2 = h * h
    query = wp.hash_grid_query(grid, pi, h)
    neighbor_idx = wp.int32(0)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx != idx:
            d = pos[neighbor_idx] - pi
            dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if dist2 < h2:
                c += 1
    counts[idx] = c


def bench_sph_neighbor_hashgrid(particle_counts: list[int], h: float = 0.1) -> list[dict]:
    """Compare brute-force vs HashGrid neighbor search."""
    results = []

    for n in particle_counts:
        positions_np = np.random.uniform(0, 1, (n, 3)).astype(np.float32)
        pos = wp.array(positions_np, dtype=wp.vec3, device="cuda")
        counts_bf = wp.zeros(n, dtype=wp.int32, device="cuda")
        counts_hg = wp.zeros(n, dtype=wp.int32, device="cuda")

        # --- brute-force ---
        if n <= 100_000:
            warmup(sph_count_neighbors_bruteforce, dim=min(n, 1024),
                   inputs=[pos, counts_bf, n, h])
            wp.synchronize_device("cuda")
            t0 = time.perf_counter()
            wp.launch(sph_count_neighbors_bruteforce, dim=n,
                      inputs=[pos, counts_bf, n, h], device="cuda")
            wp.synchronize_device("cuda")
            dt_bf = time.perf_counter() - t0
        else:
            dt_bf = None

        # --- hash grid ---
        grid = wp.HashGrid(dim_x=256, dim_y=256, dim_z=256)
        grid.build(pos, h)

        warmup(sph_count_neighbors_hashgrid, dim=min(n, 1024),
               inputs=[grid.id, pos, counts_hg, h])

        # time grid build + query together
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        grid.build(pos, h)
        wp.launch(sph_count_neighbors_hashgrid, dim=n,
                  inputs=[grid.id, pos, counts_hg, h], device="cuda")
        wp.synchronize_device("cuda")
        dt_hg = time.perf_counter() - t0

        # query-only (grid already built)
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        wp.launch(sph_count_neighbors_hashgrid, dim=n,
                  inputs=[grid.id, pos, counts_hg, h], device="cuda")
        wp.synchronize_device("cuda")
        dt_hg_query = time.perf_counter() - t0

        avg_neighbors = float(counts_hg.numpy().mean())
        r = {
            "N": n,
            "ms_bf": dt_bf * 1000 if dt_bf else None,
            "pps_bf": n / dt_bf if dt_bf else None,
            "ms_hg_total": dt_hg * 1000,
            "pps_hg_total": n / dt_hg,
            "ms_hg_query": dt_hg_query * 1000,
            "pps_hg_query": n / dt_hg_query,
            "avg_neighbors": avg_neighbors,
        }
        if dt_bf:
            r["speedup_total"] = dt_bf / dt_hg
            r["speedup_query"] = dt_bf / dt_hg_query
        results.append(r)

        bf_str = f"{r['ms_bf']:>8.2f} ms | {r['pps_bf']:>12,.0f} p/s" if dt_bf else "   --- skipped ---"
        print(f"  N={n:>7,d} | BF: {bf_str}")
        print(f"           | HG: {r['ms_hg_total']:>8.2f} ms (total) | {r['pps_hg_total']:>12,.0f} p/s | {r['ms_hg_query']:>8.2f} ms (query) | {r['pps_hg_query']:>12,.0f} p/s | avg_nbr={avg_neighbors:.1f}")
        if dt_bf:
            print(f"           | Speedup: {r['speedup_total']:>6.1f}x (total) {r['speedup_query']:>6.1f}x (query)")

    return results


# ---------------------------------------------------------------------------
# Part 2: Full 2D WCSPH Dam Break
# ---------------------------------------------------------------------------

# 2D positions stored as vec3 with z=0
# Cubic spline kernel (2D)

@wp.func
def cubic_spline_W(r: wp.float32, h: wp.float32):
    """Cubic spline kernel value (2D). Normalization: 10/(7*pi*h^2)."""
    q = r / h
    coeff = 10.0 / (7.0 * 3.14159265 * h * h)
    if q > 2.0:
        return 0.0
    elif q > 1.0:
        t = 2.0 - q
        return coeff * 0.25 * t * t * t
    else:
        return coeff * (1.0 - 1.5 * q * q + 0.75 * q * q * q)


@wp.func
def cubic_spline_dWdr(r: wp.float32, h: wp.float32):
    """Derivative of cubic spline kernel dW/dr (2D)."""
    q = r / h
    coeff = 10.0 / (7.0 * 3.14159265 * h * h)
    if q > 2.0:
        return 0.0
    elif q > 1.0:
        t = 2.0 - q
        return coeff * (-0.75 * t * t) / h
    else:
        return coeff * (-3.0 * q + 2.25 * q * q) / h


@wp.kernel
def sph_compute_density(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    mass: wp.float32,
    h: wp.float32,
    rho: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    pi = pos[i]
    density = wp.float32(0.0)

    query = wp.hash_grid_query(grid, pi, 2.0 * h)
    neighbor_idx = wp.int32(0)
    while wp.hash_grid_query_next(query, neighbor_idx):
        diff = pos[neighbor_idx] - pi
        r = wp.length(diff)
        if r < 2.0 * h:
            density += mass * cubic_spline_W(r, h)

    rho[i] = density


@wp.kernel
def sph_compute_forces(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=wp.float32),
    pressure: wp.array(dtype=wp.float32),
    mass: wp.float32,
    h: wp.float32,
    viscosity_alpha: wp.float32,
    dt: wp.float32,
    gravity: wp.float32,
    force: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pi = pos[i]
    vi = vel[i]
    rho_i = rho[i]
    p_i = pressure[i]

    acc = wp.vec3(0.0, gravity, 0.0)  # gravity in -y direction

    query = wp.hash_grid_query(grid, pi, 2.0 * h)
    neighbor_idx = wp.int32(0)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == i:
            continue
        diff = pos[neighbor_idx] - pi
        r = wp.length(diff)
        if r < 2.0 * h and r > 1e-8:
            r_norm = diff / r
            rho_j = rho[neighbor_idx]
            p_j = pressure[neighbor_idx]
            vj = vel[neighbor_idx]

            # Pressure force: -m * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W
            dWdr = cubic_spline_dWdr(r, h)
            pressure_term = -mass * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * dWdr
            acc += pressure_term * r_norm

            # Artificial viscosity (Monaghan)
            v_diff = vi - vj
            vr_dot = v_diff[0] * r_norm[0] + v_diff[1] * r_norm[1] + v_diff[2] * r_norm[2]
            if vr_dot < 0.0:
                c_s = wp.float32(20.0)  # speed of sound (approx)
                mu_ij = h * vr_dot / (r + 0.01 * h * h)
                rho_avg = 0.5 * (rho_i + rho_j)
                pi_visc = (-viscosity_alpha * c_s * mu_ij) / rho_avg
                acc += -mass * pi_visc * dWdr * r_norm

    force[i] = acc


@wp.kernel
def sph_integrate(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
):
    i = wp.tid()
    acc = force[i]
    new_vel = vel[i] + acc * dt
    new_pos = pos[i] + new_vel * dt

    # Boundary reflection (2D box)
    damping = wp.float32(0.5)
    if new_pos[0] < x_min:
        new_pos[0] = x_min
        new_vel[0] = -new_vel[0] * damping
    if new_pos[0] > x_max:
        new_pos[0] = x_max
        new_vel[0] = -new_vel[0] * damping
    if new_pos[1] < y_min:
        new_pos[1] = y_min
        new_vel[1] = -new_vel[1] * damping
    if new_pos[1] > y_max:
        new_pos[1] = y_max
        new_vel[1] = -new_vel[1] * damping

    vel[i] = new_vel
    pos[i] = new_pos


@wp.kernel
def sph_tait_eos(
    rho: wp.array(dtype=wp.float32),
    rho0: wp.float32,
    c_s: wp.float32,
    gamma: wp.float32,
    pressure: wp.array(dtype=wp.float32),
):
    """Tait equation of state for weakly compressible SPH."""
    i = wp.tid()
    pressure[i] = (c_s * c_s * rho0 / gamma) * (wp.pow(rho[i] / rho0, gamma) - 1.0)


@wp.kernel
def compute_kinetic_energy(
    vel: wp.array(dtype=wp.vec3),
    mass: wp.float32,
    energy: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    v = vel[i]
    energy[i] = 0.5 * mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def create_dam_break_2d(n_particles: int) -> tuple[np.ndarray, float, float]:
    """Create 2D dam break initial conditions.

    Returns positions array (Nx3, z=0), smoothing length h, particle spacing.
    Domain: [0, 1] x [0, 0.5], fluid block: [0, 0.3] x [0, 0.3]
    """
    # Calculate particle spacing to match desired count
    fluid_area = 0.3 * 0.3  # fluid block area
    spacing = wp.sqrt(fluid_area / n_particles)

    positions = []
    y = spacing / 2
    while y < 0.3:
        x = spacing / 2
        while x < 0.3:
            positions.append([x, y, 0.0])
            x += spacing
        y += spacing

    pos_np = np.array(positions, dtype=np.float32)
    h = spacing * 1.3  # smoothing length ~1.3x particle spacing (SPH convention)
    return pos_np, float(h), float(spacing)


def bench_dambreak_sph(particle_counts: list[int], n_steps: int = 1000) -> list[dict]:
    """Full 2D dam break WCSPH benchmark."""
    results = []

    for target_n in particle_counts:
        pos_np, h, spacing = create_dam_break_2d(target_n)
        actual_n = len(pos_np)
        print(f"  Target N={target_n:,d}, actual N={actual_n:,d}, h={h:.4f}, spacing={spacing:.4f}")

        # Particle mass: rho0 * spacing^2 (2D)
        rho0 = 1000.0  # water density
        particle_mass = rho0 * spacing * spacing

        # Tait EOS parameters
        c_s = 20.0  # speed of sound (artificial, ~10x max velocity)
        gamma = 7.0
        viscosity_alpha = 0.1
        gravity = -9.81
        dt = 0.1 * h / c_s  # CFL condition

        # Warp arrays
        pos = wp.array(pos_np, dtype=wp.vec3, device="cuda")
        vel = wp.zeros(actual_n, dtype=wp.vec3, device="cuda")
        rho = wp.zeros(actual_n, dtype=wp.float32, device="cuda")
        pressure = wp.zeros(actual_n, dtype=wp.float32, device="cuda")
        force = wp.zeros(actual_n, dtype=wp.vec3, device="cuda")
        ke = wp.zeros(actual_n, dtype=wp.float32, device="cuda")

        grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128)

        # Warmup
        for _ in range(5):
            grid.build(pos, h)
            wp.launch(sph_compute_density, dim=actual_n,
                      inputs=[grid.id, pos, particle_mass, h, rho], device="cuda")
            wp.launch(sph_tait_eos, dim=actual_n,
                      inputs=[rho, rho0, c_s, gamma, pressure], device="cuda")
            wp.launch(sph_compute_forces, dim=actual_n,
                      inputs=[grid.id, pos, vel, rho, pressure, particle_mass,
                              h, viscosity_alpha, dt, gravity, force], device="cuda")
            wp.launch(sph_integrate, dim=actual_n,
                      inputs=[pos, vel, force, dt, 0.0, 1.0, 0.0, 0.5], device="cuda")
        wp.synchronize_device("cuda")

        # Main benchmark loop
        ke_history = []
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()

        for step in range(n_steps):
            # 1. Build spatial hash grid
            grid.build(pos, h)

            # 2. Compute density
            wp.launch(sph_compute_density, dim=actual_n,
                      inputs=[grid.id, pos, particle_mass, h, rho], device="cuda")

            # 3. Tait equation of state
            wp.launch(sph_tait_eos, dim=actual_n,
                      inputs=[rho, rho0, c_s, gamma, pressure], device="cuda")

            # 4. Compute forces (pressure + viscosity + gravity)
            wp.launch(sph_compute_forces, dim=actual_n,
                      inputs=[grid.id, pos, vel, rho, pressure, particle_mass,
                              h, viscosity_alpha, dt, gravity, force], device="cuda")

            # 5. Integrate (symplectic Euler)
            wp.launch(sph_integrate, dim=actual_n,
                      inputs=[pos, vel, force, dt, 0.0, 1.0, 0.0, 0.5], device="cuda")

            # Track kinetic energy periodically
            if step % 100 == 0 or step == n_steps - 1:
                wp.launch(compute_kinetic_energy, dim=actual_n,
                          inputs=[vel, particle_mass, ke], device="cuda")
                wp.synchronize_device("cuda")
                ke_total = float(ke.numpy().sum())
                ke_history.append(ke_total)

        wp.synchronize_device("cuda")
        total_time = time.perf_counter() - t0

        ms_per_step = total_time / n_steps * 1000
        pps = actual_n / (total_time / n_steps)

        # Energy conservation: ratio of final to initial KE
        # (should grow as dam breaks, then oscillate)
        ke_initial = ke_history[0] if ke_history else 0
        ke_final = ke_history[-1] if ke_history else 0

        r = {
            "target_N": target_n,
            "actual_N": actual_n,
            "h": h,
            "spacing": spacing,
            "dt": dt,
            "n_steps": n_steps,
            "total_s": total_time,
            "ms_per_step": ms_per_step,
            "particles_per_sec": pps,
            "ke_initial": ke_initial,
            "ke_final": ke_final,
            "ke_history": ke_history,
        }
        results.append(r)
        print(f"           | {ms_per_step:>8.2f} ms/step | {pps:>12,.0f} p/s | KE: {ke_initial:.4f} -> {ke_final:.4f}")

    return results


# ---------------------------------------------------------------------------
# Part 3: Multi-resolution (varying h)
# ---------------------------------------------------------------------------

def bench_multiresolution_sph(n_particles: int = 50_000,
                               smoothing_lengths: list[float] = None,
                               n_steps: int = 100) -> list[dict]:
    """Test how smoothing length h affects performance and accuracy.

    Uses a fixed particle configuration and tests different search radii.
    Particle spacing is fixed; h varies to simulate different resolution levels.
    """
    if smoothing_lengths is None:
        smoothing_lengths = [0.02, 0.05, 0.1]

    results = []

    # Fixed particle grid: spacing = 0.015, domain ~ [0, L]^2
    spacing = 0.015
    L = wp.sqrt(float(n_particles)) * spacing
    positions = []
    y = spacing / 2
    while y < L:
        x = spacing / 2
        while x < L:
            positions.append([x, y, 0.0])
            x += spacing
        y += spacing
    positions_np_all = np.array(positions, dtype=np.float32)
    if len(positions_np_all) > n_particles:
        positions_np_all = positions_np_all[:n_particles]
    actual_n = len(positions_np_all)

    rho0 = 1000.0
    particle_mass = rho0 * spacing * spacing

    for h in smoothing_lengths:
        c_s = 20.0
        gamma = 7.0
        dt = 0.1 * h / c_s

        pos = wp.array(positions_np_all, dtype=wp.vec3, device="cuda")
        rho = wp.zeros(actual_n, dtype=wp.float32, device="cuda")
        pressure = wp.zeros(actual_n, dtype=wp.float32, device="cuda")

        grid = wp.HashGrid(dim_x=256, dim_y=256, dim_z=256)

        # Warmup
        for _ in range(5):
            grid.build(pos, h)
            wp.launch(sph_compute_density, dim=actual_n,
                      inputs=[grid.id, pos, particle_mass, h, rho], device="cuda")
            wp.launch(sph_tait_eos, dim=actual_n,
                      inputs=[rho, rho0, c_s, gamma, pressure], device="cuda")
        wp.synchronize_device("cuda")

        # Benchmark density computation
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        for _ in range(n_steps):
            grid.build(pos, h)
            wp.launch(sph_compute_density, dim=actual_n,
                      inputs=[grid.id, pos, particle_mass, h, rho], device="cuda")
            wp.launch(sph_tait_eos, dim=actual_n,
                      inputs=[rho, rho0, c_s, gamma, pressure], device="cuda")
        wp.synchronize_device("cuda")
        total_time = time.perf_counter() - t0

        ms_per_step = total_time / n_steps * 1000
        pps = actual_n / (total_time / n_steps)

        # Accuracy: average density (should be ~rho0 for uniform distribution)
        rho_np = rho.numpy()
        avg_rho = float(rho_np.mean())
        std_rho = float(rho_np.std())
        min_rho = float(rho_np.min())
        max_rho = float(rho_np.max())

        r = {
            "h": h,
            "spacing": spacing,
            "N": actual_n,
            "ms_per_step": ms_per_step,
            "particles_per_sec": pps,
            "avg_density": avg_rho,
            "std_density": std_rho,
            "min_density": min_rho,
            "max_density": max_rho,
            "density_error_pct": abs(avg_rho - rho0) / rho0 * 100,
        }
        results.append(r)
        print(f"  h={h:.3f} | N={actual_n:,d} | {ms_per_step:>8.2f} ms/step | {pps:>12,.0f} p/s | rho: avg={avg_rho:.1f} std={std_rho:.1f} [{min_rho:.1f}, {max_rho:.1f}] | err={r['density_error_pct']:.2f}%")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("Advanced SPH Benchmarks — RTX 5090 (sm_120, CUDA 12.8)")
    print("Warp v" + str(wp.__version__))
    print("=" * 90)

    # ---- Part 1: Spatial Hash Grid vs Brute-Force ----
    print("\n" + "=" * 90)
    print("PART 1: SPH Neighbor Search — Brute-Force vs Spatial Hash Grid")
    print("=" * 90)
    print(f"  {'N':>9s} | Method     | {'ms':>10s} | {'particles/s':>14s}")
    p1_results = bench_sph_neighbor_hashgrid([10_000, 50_000, 100_000, 500_000, 1_000_000])

    # ---- Part 2: Full 2D Dam Break WCSPH ----
    print("\n" + "=" * 90)
    print("PART 2: 2D Dam Break WCSPH (1000 steps)")
    print("  Cubic spline kernel, Tait EOS, artificial viscosity, boundary reflection")
    print("=" * 90)
    p2_results = bench_dambreak_sph([20_000, 100_000, 500_000], n_steps=1000)

    # ---- Part 3: Multi-resolution ----
    print("\n" + "=" * 90)
    print("PART 3: Multi-resolution SPH (N=50K, varying h)")
    print("=" * 90)
    p3_results = bench_multiresolution_sph(
        n_particles=50_000,
        smoothing_lengths=[0.02, 0.05, 0.1],
        n_steps=100,
    )

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print("\n[Part 1: Neighbor Search — Brute-Force vs Hash Grid]")
    print(f"  {'N':>9s} | {'BF ms':>8s} | {'BF p/s':>12s} | {'HG ms':>8s} | {'HG p/s':>12s} | {'Speedup':>8s} | {'Avg Nbr':>8s}")
    for r in p1_results:
        bf_ms = f"{r['ms_bf']:>8.2f}" if r['ms_bf'] else "   ---"
        bf_ps = f"{r['pps_bf']:>12,.0f}" if r['pps_bf'] else "        ---"
        sp = f"{r['speedup_query']:>7.1f}x" if 'speedup_query' in r else "     ---"
        print(f"  {r['N']:>9,d} | {bf_ms} | {bf_ps} | {r['ms_hg_query']:>8.2f} | {r['pps_hg_query']:>12,.0f} | {sp} | {r['avg_neighbors']:>8.1f}")

    print("\n[Part 2: 2D Dam Break WCSPH]")
    print(f"  {'N':>9s} | {'h':>6s} | {'dt':>8s} | {'ms/step':>8s} | {'p/s':>12s} | {'KE_i':>10s} | {'KE_f':>10s}")
    for r in p2_results:
        print(f"  {r['actual_N']:>9,d} | {r['h']:>6.4f} | {r['dt']:>8.6f} | {r['ms_per_step']:>8.2f} | {r['particles_per_sec']:>12,.0f} | {r['ke_initial']:>10.4f} | {r['ke_final']:>10.4f}")

    print("\n[Part 3: Multi-resolution (N=50K)]")
    print(f"  {'h':>6s} | {'ms/step':>8s} | {'p/s':>12s} | {'rho_avg':>10s} | {'rho_std':>10s} | {'err%':>6s}")
    for r in p3_results:
        print(f"  {r['h']:>6.3f} | {r['ms_per_step']:>8.2f} | {r['particles_per_sec']:>12,.0f} | {r['avg_density']:>10.1f} | {r['std_density']:>10.1f} | {r['density_error_pct']:>6.2f}")


if __name__ == "__main__":
    main()
