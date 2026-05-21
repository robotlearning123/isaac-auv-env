"""JAX CFD benchmark: 2D advection-diffusion on 512x512 grid.

Tests: jax.lax.scan, jax.vmap, jax.jit
Compares with Warp and PyTorch.
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit

print(f"JAX {jax.__version__}, devices: {jax.devices()}")

N = 512
DX = 1.0 / N
DT = 0.0001
NU = 0.001  # viscosity
VX, VY = 0.5, 0.3  # advection velocities
ITERS = 100
WARMUP = 10
BATCH = 8


def advection_diffusion_step_jax(u, vx, vy, nu, dx, dt):
    """One step of 2D advection-diffusion using centered differences."""
    # Periodic boundary via roll
    uxp = jnp.roll(u, -1, axis=0)
    uxm = jnp.roll(u, 1, axis=0)
    uyp = jnp.roll(u, -1, axis=1)
    uym = jnp.roll(u, 1, axis=1)
    laplacian = (uxp + uxm + uyp + uym - 4 * u) / (dx * dx)
    dudx = (uxp - uxm) / (2 * dx)
    dudy = (uyp - uym) / (2 * dx)
    return u + dt * (nu * laplacian - vx * dudx - vy * dudy)


def bench_jax_scan():
    """jax.lax.scan iterative solver."""
    u = jnp.ones((N, N))
    vx, vy = jnp.float32(VX), jnp.float32(VY)

    step_fn = lambda u, _: (advection_diffusion_step_jax(u, vx, vy, NU, DX, DT), None)
    compiled = jit(lambda u: lax.scan(step_fn, u, None, length=ITERS))

    # Warmup + compile
    _ = compiled(u)
    t0 = time.perf_counter()
    result = compiled(u)
    result[0].block_until_ready()
    compile_and_run = time.perf_counter() - t0
    print(f"JAX lax.scan: {compile_and_run*1000:.1f} ms (compile+{ITERS} iters)")

    # Pure runtime
    t0 = time.perf_counter()
    for _ in range(5):
        result = compiled(u)
        result[0].block_until_ready()
    ms = (time.perf_counter() - t0) / 5 / ITERS * 1000
    print(f"JAX lax.scan: {ms:.3f} ms/iter (warm)")
    return ms


def bench_jax_vmap():
    """jax.vmap batched environments."""
    rng = jax.random.PRNGKey(42)
    batch_u = jax.random.uniform(rng, (BATCH, N, N))
    vx, vy = jnp.float32(VX), jnp.float32(VY)

    single_step = lambda u: advection_diffusion_step_jax(u, vx, vy, NU, DX, DT)
    batched_step = jit(vmap(single_step))

    # Warmup
    _ = batched_step(batch_u)
    batch_u.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        batch_u = batched_step(batch_u)
    batch_u.block_until_ready()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"JAX vmap: {ms:.3f} ms/iter (batch={BATCH}, {N}x{N})")
    return ms


def bench_jax_jit_compile():
    """Measure JIT compile time vs warm performance."""
    u = jnp.ones((N, N))
    vx, vy = jnp.float32(VX), jnp.float32(VY)

    # Cold compile
    fresh_fn = jit(lambda u: advection_diffusion_step_jax(u, vx, vy, NU, DX, DT))
    t0 = time.perf_counter()
    result = fresh_fn(u)
    result.block_until_ready()
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"JAX JIT compile+run (cold): {compile_ms:.1f} ms")

    # Warm
    t0 = time.perf_counter()
    for _ in range(ITERS):
        result = fresh_fn(u)
    result.block_until_ready()
    warm_ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"JAX JIT warm: {warm_ms:.3f} ms/iter")
    print(f"JAX compile overhead: {compile_ms/warm_ms:.0f}x warm iter")
    return compile_ms, warm_ms


def bench_pytorch_ad():
    """PyTorch 2D advection-diffusion baseline."""
    import torch

    u = torch.ones(N, N, device="cuda")
    vx_t, vy_t = VX, VY

    def step(u):
        uxp = torch.roll(u, -1, 0)
        uxm = torch.roll(u, 1, 0)
        uyp = torch.roll(u, -1, 1)
        uym = torch.roll(u, 1, 1)
        lap = (uxp + uxm + uyp + uym - 4 * u) / (DX * DX)
        dudx = (uxp - uxm) / (2 * DX)
        dudy = (uyp - uym) / (2 * DX)
        return u + DT * (NU * lap - vx_t * dudx - vy_t * dudy)

    for _ in range(WARMUP):
        u = step(u)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        u = step(u)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"PyTorch eager: {ms:.3f} ms/iter ({N}x{N})")
    return ms


def bench_warp_ad():
    """Warp 2D advection-diffusion baseline."""
    import warp as wp

    @wp.kernel
    def ad_step(
        u_in: wp.array2d(dtype=wp.float32),
        u_out: wp.array2d(dtype=wp.float32),
        n: int,
        dx: float,
        dt: float,
        nu: float,
        vx: float,
        vy: float,
    ):
        i, j = wp.tid()
        ip = (i + 1) % n
        im = (i - 1 + n) % n
        jp = (j + 1) % n
        jm = (j - 1 + n) % n
        uxp = u_in[ip, j]
        uxm = u_in[im, j]
        uyp = u_in[i, jp]
        uym = u_in[i, jm]
        uc = u_in[i, j]
        lap = (uxp + uxm + uyp + uym - 4.0 * uc) / (dx * dx)
        dudx = (uxp - uxm) / (2.0 * dx)
        dudy = (uyp - uym) / (2.0 * dx)
        u_out[i, j] = uc + dt * (nu * lap - vx * dudx - vy * dudy)

    u_np = np.ones((N, N), dtype=np.float32)
    u_wp = wp.from_numpy(u_np, dtype=wp.float32)
    u_out = wp.zeros_like(u_wp)

    for _ in range(WARMUP):
        wp.launch(ad_step, dim=(N, N), inputs=[u_wp, u_out, N, DX, DT, NU, VX, VY])
        u_wp, u_out = u_out, u_wp
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        wp.launch(ad_step, dim=(N, N), inputs=[u_wp, u_out, N, DX, DT, NU, VX, VY])
        u_wp, u_out = u_out, u_wp
    wp.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"Warp: {ms:.3f} ms/iter ({N}x{N})")
    return ms


def main():
    print(f"=== 2D Advection-Diffusion Benchmark ({N}x{N}, {ITERS} iters) ===")
    print()

    print("--- JAX ---")
    jax_scan_ms = bench_jax_scan()
    jax_vmap_ms = bench_jax_vmap()
    jax_compile_ms, jax_warm_ms = bench_jax_jit_compile()

    print()
    print("--- PyTorch ---")
    torch_ms = bench_pytorch_ad()

    print()
    print("--- Warp ---")
    warp_ms = bench_warp_ad()

    print()
    print("=== Summary ===")
    print(f"  Framework         | ms/iter")
    print(f"  ------------------|--------")
    print(f"  JAX lax.scan      | {jax_scan_ms:.3f}")
    print(f"  JAX vmap(b={BATCH})    | {jax_vmap_ms:.3f}")
    print(f"  JAX single step   | {jax_warm_ms:.3f}")
    print(f"  PyTorch eager     | {torch_ms:.3f}")
    print(f"  Warp              | {warp_ms:.3f}")
    print(f"  JAX JIT compile   | {jax_compile_ms:.1f} (one-time)")


if __name__ == "__main__":
    main()
