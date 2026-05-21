"""Taichi 2D Euler fluid (stable fluids) benchmark on RTX 5090."""
import taichi as ti
import time
import numpy as np

ti.init(arch=ti.cuda)

N = 512
vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
new_vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
pressure = ti.field(dtype=ti.f32, shape=(N, N))
new_p = ti.field(dtype=ti.f32, shape=(N, N))
div = ti.field(dtype=ti.f32, shape=(N, N))


@ti.kernel
def init_vel():
    for i, j in vel:
        vel[i, j] = ti.Vector([ti.sin(i * 0.01) * ti.cos(j * 0.01),
                                ti.cos(i * 0.01) * ti.sin(j * 0.01)])


@ti.kernel
def advect():
    for i, j in vel:
        x = float(i) - vel[i, j][0] * 10.0
        y = float(j) - vel[i, j][1] * 10.0
        x = ti.math.clamp(x, 0.5, float(N) - 1.5)
        y = ti.math.clamp(y, 0.5, float(N) - 1.5)
        ix = int(x)
        iy = int(y)
        fx = x - float(ix)
        fy = y - float(iy)
        new_vel[i, j] = ((1 - fx) * (1 - fy) * vel[ix, iy]
                          + fx * (1 - fy) * vel[ix + 1, iy]
                          + (1 - fx) * fy * vel[ix, iy + 1]
                          + fx * fy * vel[ix + 1, iy + 1])
    for i, j in vel:
        vel[i, j] = new_vel[i, j]


@ti.kernel
def compute_div():
    for i, j in div:
        vl = vel[max(i - 1, 0), j][0]
        vr = vel[min(i + 1, N - 1), j][0]
        vb = vel[i, max(j - 1, 0)][1]
        vt = vel[i, min(j + 1, N - 1)][1]
        div[i, j] = (vr - vl + vt - vb) * 0.5


@ti.kernel
def pressure_jacobi():
    for i, j in new_p:
        pl = pressure[max(i - 1, 0), j]
        pr = pressure[min(i + 1, N - 1), j]
        pb = pressure[i, max(j - 1, 0)]
        pt = pressure[i, min(j + 1, N - 1)]
        new_p[i, j] = (pl + pr + pb + pt - div[i, j]) * 0.25


@ti.kernel
def swap_p():
    for i, j in pressure:
        pressure[i, j] = new_p[i, j]


@ti.kernel
def subtract_grad():
    for i, j in vel:
        pl = pressure[max(i - 1, 0), j]
        pr = pressure[min(i + 1, N - 1), j]
        pb = pressure[i, max(j - 1, 0)]
        pt = pressure[i, min(j + 1, N - 1)]
        vel[i, j][0] -= (pr - pl) * 0.5
        vel[i, j][1] -= (pt - pb) * 0.5


init_vel()

# Warmup
for _ in range(5):
    advect()
    compute_div()
    for _ in range(40):
        pressure_jacobi()
        swap_p()
    subtract_grad()

# Benchmark
times = []
for step in range(100):
    t0 = time.perf_counter()
    advect()
    compute_div()
    for _ in range(40):
        pressure_jacobi()
        swap_p()
    subtract_grad()
    ti.sync()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = np.mean(times[5:])
print(f"Taichi 2D Euler Fluid: grid={N}x{N}, avg step time={avg*1000:.2f} ms, throughput={N*N/avg/1e6:.1f} Mcell/s")
print(f"  min={np.min(times[5:])*1000:.2f} ms, max={np.max(times[5:])*1000:.2f} ms")
