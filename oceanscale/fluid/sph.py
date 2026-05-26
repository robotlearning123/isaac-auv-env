"""WCSPH solver — Weakly Compressible Smoothed Particle Hydrodynamics on Warp GPU.

Tait equation of state with cubic spline kernel.
All kernels run on GPU via Warp; neighbor search uses wp.HashGrid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

PI = math.pi


# -- Cubic spline kernel (3D), support radius = 2h --
# Monaghan (1992), Ann. Rev. Astron. Astrophys. 30:543


@wp.func
def _cubic_W(r: float, h: float):
    q = r / h
    c = 1.0 / (PI * h * h * h)
    if q <= 1.0:
        return c * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    elif q <= 2.0:
        t = 2.0 - q
        return c * 0.25 * t * t * t
    return 0.0


@wp.func
def _cubic_gradW(r: float, rij: wp.vec3, h: float):
    q = r / h
    if q < 1.0e-8 or q > 2.0:
        return wp.vec3(0.0, 0.0, 0.0)
    c = 1.0 / (PI * h * h * h * h)
    if q <= 1.0:
        ds = -3.0 * q + 2.25 * q * q
    else:
        t = 2.0 - q
        ds = -0.75 * t * t
    return c * ds * (rij / r)


@wp.func
def _cubic_lapW(r: float, h: float):
    q = r / h
    if q < 1.0e-8 or q > 2.0:
        return 0.0
    c = 1.0 / (PI * h * h * h * h * h)
    if q <= 1.0:
        return c * (-9.0 + 9.0 * q)
    return c * (3.0 * (2.0 - q) * (q - 1.0) / q)


@wp.func
def _tait_eos(rho: float, rho0: float, B: float, gamma: float):
    return B * (wp.pow(rho / rho0, gamma) - 1.0)


# -- Warp kernels -----------------------------------------------------------


@wp.kernel
def _compute_density(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    density: wp.array[float],
    mass: float,
    h: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = pos[i]
    rho = 0.0
    neighbors = wp.hash_grid_query(grid, xi, 2.0 * h)
    for j in neighbors:
        rij = xi - pos[j]
        rho += mass * _cubic_W(wp.length(rij), h)
    density[i] = rho


@wp.kernel
def _compute_forces(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    density: wp.array[float],
    force: wp.array[wp.vec3],
    mass: float,
    h: float,
    B: float,
    rho0: float,
    gamma: float,
    mu: float,
    gx: float,
    gy: float,
    gz: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = pos[i]
    vi = vel[i]
    rhoi = density[i]
    Pi = _tait_eos(rhoi, rho0, B, gamma)
    fp = wp.vec3(0.0, 0.0, 0.0)
    fv = wp.vec3(0.0, 0.0, 0.0)
    neighbors = wp.hash_grid_query(grid, xi, 2.0 * h)
    for j in neighbors:
        if j == i:
            continue
        rij = xi - pos[j]
        r = wp.length(rij)
        rhoj = density[j]
        Pj = _tait_eos(rhoj, rho0, B, gamma)
        gW = _cubic_gradW(r, rij, h)
        fp += -mass * (Pi / (rhoi * rhoi) + Pj / (rhoj * rhoj)) * gW
        lW = _cubic_lapW(r, h)
        fv += mu * mass * (vel[j] - vi) / rhoj * lW
    force[i] = fp + fv + wp.vec3(gx, gy, gz)


# -- Config & Solver --------------------------------------------------------


@dataclass
class WCSPHConfig:
    num_particles: int = 10000
    smoothing_length: float = 0.1
    rest_density: float = 1025.0  # seawater kg/m^3
    stiffness: float = 50.0
    viscosity: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    particle_mass: float = 0.0  # auto-computed when <= 0


class WCSPHSolver:
    def __init__(self, config: WCSPHConfig, device: str | None = None):
        self.cfg = config
        self.device = device
        n = config.num_particles
        h = config.smoothing_length
        self.gamma = 7.0
        self.B = config.stiffness * config.rest_density / self.gamma
        self.mass = (
            config.particle_mass
            if config.particle_mass > 0.0
            else config.rest_density * h ** 3 * 0.8
        )
        self.density = wp.zeros(n, dtype=float, device=device)
        self.force = wp.zeros(n, dtype=wp.vec3, device=device)
        cells = max(8, int(1.0 / h))
        self.grid = wp.HashGrid(cells, cells, cells, device=device)

    def step(
        self,
        positions,
        velocities,
        dt: float,
    ):
        """One WCSPH timestep. Returns per-particle accelerations (dv/dt) as numpy (N,3)."""
        n = self.cfg.num_particles
        h = self.cfg.smoothing_length
        gx, gy, gz = self.cfg.gravity
        pos = wp.array(positions, dtype=wp.vec3, device=self.device)
        vel = wp.array(velocities, dtype=wp.vec3, device=self.device)
        self.grid.build(pos, 2.0 * h)
        wp.launch(
            _compute_density, dim=n,
            inputs=[self.grid.id, pos, self.density, self.mass, h],
            device=self.device,
        )
        wp.launch(
            _compute_forces, dim=n,
            inputs=[
                self.grid.id, pos, vel, self.density, self.force,
                self.mass, h, self.B, self.cfg.rest_density,
                self.gamma, self.cfg.viscosity, gx, gy, gz,
            ],
            device=self.device,
        )
        # Force → acceleration
        acc = self.force.numpy() / self.mass
        return acc
