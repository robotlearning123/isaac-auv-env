# mypy: ignore-errors
"""GPU-native ocean surface wave simulation using Warp kernels.

Implements linear Airy wave theory with spectral decomposition (JONSWAP /
Pierson-Moskowitz) for underwater robot training environments. Wave-induced
velocity and pressure fields decay with depth, matching real ocean physics.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.init()

_GRAVITY = 9.81
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _wave_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    out_vel: wp.array(dtype=wp.vec3),
    amplitudes: wp.array(dtype=wp.float32),
    wavenumbers: wp.array(dtype=wp.float32),
    frequencies: wp.array(dtype=wp.float32),
    phases: wp.array(dtype=wp.float32),
    dir_x: wp.array(dtype=wp.float32),
    dir_y: wp.array(dtype=wp.float32),
    n_components: wp.int32,
    depth: wp.float32,
    time: wp.float32,
):
    tid = wp.tid()
    p = positions[tid]
    x = p[0]
    y = p[1]
    z = p[2]

    u = float(0.0)
    v = float(0.0)
    w = float(0.0)

    for c in range(n_components):
        a = amplitudes[c]
        k = wavenumbers[c]
        omega = frequencies[c]
        phi = phases[c]
        dx = dir_x[c]
        dy = dir_y[c]

        theta = k * (dx * x + dy * y) - omega * time + phi

        kd = k * depth
        kz = k * (z + depth)

        cosh_kd = wp.cosh(kd)
        if cosh_kd < 1.0e-8:
            cosh_kd = 1.0e-8

        cosh_kz = wp.cosh(kz)
        sinh_kz = wp.sinh(kz)

        h_factor = cosh_kz / cosh_kd
        v_factor = sinh_kz / cosh_kd

        u_comp = a * omega * h_factor * wp.cos(theta)
        w_comp = a * omega * v_factor * wp.sin(theta)

        u = u + u_comp * dx
        v = v + u_comp * dy
        w = w + w_comp

    out_vel[tid] = wp.vec3(u, v, w)


@wp.kernel
def _wave_pressure_kernel(
    positions: wp.array(dtype=wp.vec3),
    out_pressure: wp.array(dtype=wp.float32),
    amplitudes: wp.array(dtype=wp.float32),
    wavenumbers: wp.array(dtype=wp.float32),
    frequencies: wp.array(dtype=wp.float32),
    phases: wp.array(dtype=wp.float32),
    dir_x: wp.array(dtype=wp.float32),
    dir_y: wp.array(dtype=wp.float32),
    n_components: wp.int32,
    depth: wp.float32,
    time: wp.float32,
    rho: wp.float32,
    g: wp.float32,
):
    tid = wp.tid()
    p = positions[tid]
    x = p[0]
    y = p[1]
    z = p[2]

    dp = float(0.0)

    for c in range(n_components):
        a = amplitudes[c]
        k = wavenumbers[c]
        omega = frequencies[c]
        phi = phases[c]
        dx = dir_x[c]
        dy = dir_y[c]

        theta = k * (dx * x + dy * y) - omega * time + phi

        kd = k * depth
        kz = k * (z + depth)

        cosh_kd = wp.cosh(kd)
        if cosh_kd < 1.0e-8:
            cosh_kd = 1.0e-8

        cosh_kz = wp.cosh(kz)
        dp = dp + rho * g * a * (cosh_kz / cosh_kd) * wp.cos(theta)

    out_pressure[tid] = dp


@wp.kernel
def _wave_surface_kernel(
    x_arr: wp.array(dtype=wp.float32),
    y_arr: wp.array(dtype=wp.float32),
    out_eta: wp.array(dtype=wp.float32),
    amplitudes: wp.array(dtype=wp.float32),
    wavenumbers: wp.array(dtype=wp.float32),
    frequencies: wp.array(dtype=wp.float32),
    phases: wp.array(dtype=wp.float32),
    dir_x: wp.array(dtype=wp.float32),
    dir_y: wp.array(dtype=wp.float32),
    n_components: wp.int32,
    time: wp.float32,
):
    tid = wp.tid()
    x = x_arr[tid]
    y = y_arr[tid]

    eta = float(0.0)

    for c in range(n_components):
        a = amplitudes[c]
        k = wavenumbers[c]
        omega = frequencies[c]
        phi = phases[c]
        dx = dir_x[c]
        dy = dir_y[c]

        theta = k * (dx * x + dy * y) - omega * time + phi
        eta = eta + a * wp.cos(theta)

    out_eta[tid] = eta


# ---------------------------------------------------------------------------
# Spectrum helpers (CPU, run once at init)
# ---------------------------------------------------------------------------


def _solve_dispersion(omega: float, depth: float, tol: float = 1e-6, max_iter: int = 50) -> float:
    """Solve omega^2 = g*k*tanh(k*d) for k using Newton iteration."""
    k = omega * omega / _GRAVITY
    for _ in range(max_iter):
        f = omega * omega - _GRAVITY * k * math.tanh(k * depth)
        fp = -_GRAVITY * (math.tanh(k * depth) + k * depth / (math.cosh(k * depth) ** 2))
        if abs(fp) < 1e-15:
            break
        dk = f / fp
        k = k - dk
        if abs(dk) < tol:
            break
    return max(k, 1e-10)


def _pierson_moskowitz_spectrum(omega: float, hs: float, tp: float) -> float:
    """Pierson-Moskowitz spectral density S(omega)."""
    omega_p = _TWO_PI / tp
    if omega < 1e-10:
        return 0.0
    alpha = 5.0 / 16.0 * hs * hs * omega_p**4
    return alpha / omega**5 * math.exp(-1.25 * (omega_p / omega) ** 4)


def _jonswap_spectrum(omega: float, hs: float, tp: float, gamma: float = 3.3) -> float:
    """JONSWAP spectral density S(omega)."""
    s_pm = _pierson_moskowitz_spectrum(omega, hs, tp)
    omega_p = _TWO_PI / tp
    if omega < 1e-10:
        return 0.0
    sigma = 0.07 if omega <= omega_p else 0.09
    r = math.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    return s_pm * gamma**r


def _generate_components(
    n: int,
    hs: float,
    tp: float,
    direction: float,
    depth: float,
    spectrum: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate wave spectral components."""
    omega_p = _TWO_PI / tp
    omega_min = 0.5 * omega_p
    omega_max = 3.0 * omega_p
    d_omega = (omega_max - omega_min) / n

    spec_fn = _jonswap_spectrum if spectrum == "jonswap" else _pierson_moskowitz_spectrum

    rng = np.random.default_rng(seed)
    omegas = np.zeros(n, dtype=np.float32)
    amps = np.zeros(n, dtype=np.float32)
    ks = np.zeros(n, dtype=np.float32)
    ph = rng.uniform(0, _TWO_PI, n).astype(np.float32)
    dx = np.full(n, math.cos(direction), dtype=np.float32)
    dy = np.full(n, math.sin(direction), dtype=np.float32)

    for i in range(n):
        omega_i = omega_min + (i + 0.5) * d_omega
        s = spec_fn(omega_i, hs, tp)
        omegas[i] = omega_i
        amps[i] = math.sqrt(2.0 * s * d_omega)
        ks[i] = _solve_dispersion(omega_i, depth)

    return amps, ks, omegas, ph, dx, dy


# ---------------------------------------------------------------------------
# OceanWaveField
# ---------------------------------------------------------------------------


class OceanWaveField:
    """GPU-native ocean surface wave model using Warp kernels.

    Implements linear Airy wave theory for underwater environments.
    Provides velocity/pressure fields that decay with depth.
    """

    def __init__(
        self,
        wave_height: float = 1.0,
        wave_period: float = 8.0,
        wave_direction: float = 0.0,
        water_depth: float = 50.0,
        n_components: int = 16,
        spectrum: str = "jonswap",
        density: float = 1025.0,
        device: str = "cuda",
    ):
        self.hs = wave_height
        self.tp = wave_period
        self.direction = wave_direction
        self.depth = water_depth
        self.n_components = n_components
        self.spectrum_type = spectrum
        self.density = density
        self.device = device
        self.time = 0.0

        amps, ks, omegas, phases, dx, dy = _generate_components(
            n_components, wave_height, wave_period, wave_direction,
            water_depth, spectrum,
        )

        self._amplitudes = wp.array(amps, dtype=wp.float32, device=device)
        self._wavenumbers = wp.array(ks, dtype=wp.float32, device=device)
        self._frequencies = wp.array(omegas, dtype=wp.float32, device=device)
        self._phases = wp.array(phases, dtype=wp.float32, device=device)
        self._dir_x = wp.array(dx, dtype=wp.float32, device=device)
        self._dir_y = wp.array(dy, dtype=wp.float32, device=device)

    def get_velocity_at(self, positions: wp.array, time: float | None = None) -> wp.array:
        """Return wave-induced velocity at each 3D position.

        Velocity decays with depth via cosh(k(z+d))/cosh(kd).
        z is negative below the surface (z=0 at surface, z=-depth at seabed).
        """
        t = time if time is not None else self.time
        n = positions.shape[0]
        out = wp.zeros(n, dtype=wp.vec3, device=self.device)
        wp.launch(
            _wave_velocity_kernel,
            dim=n,
            inputs=[
                positions, out,
                self._amplitudes, self._wavenumbers, self._frequencies,
                self._phases, self._dir_x, self._dir_y,
                self.n_components, float(self.depth), float(t),
            ],
            device=self.device,
        )
        return out

    def get_pressure_at(self, positions: wp.array, time: float | None = None) -> wp.array:
        """Return wave-induced dynamic pressure at each 3D position."""
        t = time if time is not None else self.time
        n = positions.shape[0]
        out = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.launch(
            _wave_pressure_kernel,
            dim=n,
            inputs=[
                positions, out,
                self._amplitudes, self._wavenumbers, self._frequencies,
                self._phases, self._dir_x, self._dir_y,
                self.n_components, float(self.depth), float(t),
                float(self.density), float(_GRAVITY),
            ],
            device=self.device,
        )
        return out

    def get_surface_elevation(self, x: wp.array, y: wp.array, time: float | None = None) -> wp.array:
        """Return surface elevation eta(x, y, t)."""
        t = time if time is not None else self.time
        n = x.shape[0]
        out = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.launch(
            _wave_surface_kernel,
            dim=n,
            inputs=[
                x, y, out,
                self._amplitudes, self._wavenumbers, self._frequencies,
                self._phases, self._dir_x, self._dir_y,
                self.n_components, float(t),
            ],
            device=self.device,
        )
        return out

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """Advance wave field time."""
        self.time += dt
