"""Taichi SPH 3D benchmark on RTX 5090."""
import taichi as ti
import time
import numpy as np

ti.init(arch=ti.cuda)

N_PARTICLES = 100000
dim = 3
h = 0.1
mass = 1.0
rest_rho = 1000.0
stiffness = 50000.0
viscosity = 100.0
dt = 0.0001
bound = 3.0

pos = ti.Vector.field(dim, dtype=ti.f32, shape=N_PARTICLES)
vel = ti.Vector.field(dim, dtype=ti.f32, shape=N_PARTICLES)
density = ti.field(dtype=ti.f32, shape=N_PARTICLES)
pressure = ti.field(dtype=ti.f32, shape=N_PARTICLES)


@ti.kernel
def init_particles():
    for i in pos:
        pos[i] = ti.Vector([
            ti.random() * 0.5 + 0.25,
            ti.random() * 0.5 + 0.25,
            ti.random() * 0.5 + 0.25,
        ])
        vel[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def compute_density():
    for i in density:
        rho = 0.0
        for j in range(N_PARTICLES):
            diff = pos[i] - pos[j]
            r2 = diff.dot(diff)
            if r2 < h * h:
                r = ti.sqrt(r2)
                q = 1.0 - r / h
                rho += mass * (315.0 / (64.0 * 3.14159 * h ** 9)) * q ** 3
        density[i] = rho
        pressure[i] = stiffness * (rho - rest_rho)


@ti.kernel
def compute_force():
    for i in vel:
        force = ti.Vector([0.0, 0.0, -9.8 * mass])
        for j in range(N_PARTICLES):
            if i != j:
                diff = pos[i] - pos[j]
                r2 = diff.dot(diff)
                if r2 < h * h:
                    r = ti.sqrt(r2)
                    q = 1.0 - r / h
                    # Pressure force
                    force += -diff.normalized(max(1e-5) * h) * mass * (
                        pressure[i] + pressure[j]
                    ) / (2.0 * ti.max(density[j], 1e-5)) * (
                        -45.0 / (3.14159 * h ** 6)
                    ) * q * q
                    # Viscosity force
                    force += viscosity * mass * (vel[j] - vel[i]) / ti.max(
                        density[j], 1e-5
                    ) * (45.0 / (3.14159 * h ** 6)) * q
        vel[i] += force / mass * dt


@ti.kernel
def advect():
    for i in pos:
        pos[i] += vel[i] * dt
        for d in ti.static(range(dim)):
            if pos[i][d] < 0.0:
                pos[i][d] = 0.0
                vel[i][d] *= -0.5
            if pos[i][d] > bound:
                pos[i][d] = bound
                vel[i][d] *= -0.5


init_particles()

# Warmup (reduced particle count kernel is O(N^2) so keep small)
N_WARMUP = 3
for _ in range(N_WARMUP):
    compute_density()
    compute_force()
    advect()

# Benchmark
times = []
N_STEPS = 10
for step in range(N_STEPS):
    t0 = time.perf_counter()
    compute_density()
    compute_force()
    advect()
    ti.sync()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = np.mean(times[2:])
print(f"Taichi SPH 3D: particles={N_PARTICLES}, avg step time={avg*1000:.2f} ms ({1.0/avg:.1f} FPS)")
print(f"  min={np.min(times[2:])*1000:.2f} ms, max={np.max(times[2:])*1000:.2f} ms")
print(f"  NOTE: O(N^2) brute-force neighbor search; spatial hash would be ~100x faster")
