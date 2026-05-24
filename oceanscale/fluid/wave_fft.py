# mypy: ignore-errors
"""FFT-accelerated ocean wave field using Warp tile FFT.

Evaluates surface waves on a regular 2D grid via IFFT, then interpolates
for arbitrary position queries. O(N² log N) per frame instead of O(P × M)
explicit summation (P = query points, M = spectral components).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.init()

_GRAVITY = 9.81
_TWO_PI = 2.0 * math.pi

FFT_GRID = 64
FFT_BLOCK = FFT_GRID // 2
FFT_TILE_M = 1
FFT_TILE_N = FFT_GRID
FFT_TRANSPOSE_DIM = 16


@wp.kernel(module="wave_fft_kernels")
def _fft_rows(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    i, _, _ = wp.tid()
    row = wp.tile_load(x, shape=(FFT_TILE_M, FFT_TILE_N), offset=(i * FFT_TILE_M, 0))
    wp.tile_fft(row)
    wp.tile_store(y, row, offset=(i * FFT_TILE_M, 0))


@wp.kernel(module="wave_fft_kernels")
def _ifft_rows(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    i, _, _ = wp.tid()
    row = wp.tile_load(x, shape=(FFT_TILE_M, FFT_TILE_N), offset=(i * FFT_TILE_M, 0))
    wp.tile_ifft(row)
    wp.tile_store(y, row, offset=(i * FFT_TILE_M, 0))


@wp.kernel(module="wave_fft_transpose")
def _transpose(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    i, j = wp.tid()
    tile = wp.tile_load(
        x,
        shape=(FFT_TRANSPOSE_DIM, FFT_TRANSPOSE_DIM),
        offset=(i * FFT_TRANSPOSE_DIM, j * FFT_TRANSPOSE_DIM),
        storage="shared",
    )
    out = wp.tile_transpose(tile)
    wp.tile_store(y, out, offset=(j * FFT_TRANSPOSE_DIM, i * FFT_TRANSPOSE_DIM))


@wp.kernel
def _normalize_real(
    src: wp.array2d(dtype=wp.vec2f),
    dst: wp.array2d(dtype=wp.float32),
    divisor: wp.float32,
):
    i, j = wp.tid()
    dst[i, j] = src[i, j][0] / divisor


@wp.kernel
def _bilinear_sample_grid(
    grid: wp.array2d(dtype=wp.float32),
    positions: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
    inv_dx: wp.float32,
    origin_x: wp.float32,
    origin_y: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    depth: wp.float32,
    time: wp.float32,
):
    """Sample wave-induced velocity from surface elevation grid via finite differences."""
    tid = wp.tid()
    p = positions[tid]
    fx = (p[0] - origin_x) * inv_dx
    fy = (p[1] - origin_y) * inv_dx

    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    sx = fx - wp.floor(fx)
    sy = fy - wp.floor(fy)

    ix0 = wp.clamp(ix, 0, nx - 1)
    ix1 = wp.clamp(ix + 1, 0, nx - 1)
    iy0 = wp.clamp(iy, 0, ny - 1)
    iy1 = wp.clamp(iy + 1, 0, ny - 1)

    eta00 = grid[ix0, iy0]
    eta10 = grid[ix1, iy0]
    eta01 = grid[ix0, iy1]
    eta11 = grid[ix1, iy1]

    eta = (1.0 - sx) * (1.0 - sy) * eta00 + sx * (1.0 - sy) * eta10 + (1.0 - sx) * sy * eta01 + sx * sy * eta11

    deta_dx = ((1.0 - sy) * (eta10 - eta00) + sy * (eta11 - eta01)) * inv_dx
    deta_dy = ((1.0 - sx) * (eta01 - eta00) + sx * (eta11 - eta10)) * inv_dx

    z = p[2]
    kz_factor = wp.exp(wp.max(z / depth, -5.0))

    u = deta_dx * kz_factor * wp.sqrt(9.81 * depth)
    v = deta_dy * kz_factor * wp.sqrt(9.81 * depth)
    w = eta * kz_factor * 0.1

    out[tid] = wp.vec3(u, v, w)


def _jonswap_2d(grid_size: int, domain_length: float, hs: float, tp: float,
                direction: float, depth: float, gamma: float = 3.3,
                seed: int = 42) -> np.ndarray:
    """Generate 2D JONSWAP spectrum on a wavenumber grid. Returns complex array."""
    dk = _TWO_PI / domain_length
    omega_p = _TWO_PI / tp
    rng = np.random.default_rng(seed)

    spectrum = np.zeros((grid_size, grid_size), dtype=np.complex64)
    half = grid_size // 2

    for i in range(grid_size):
        for j in range(grid_size):
            kx = (i - half) * dk
            ky = (j - half) * dk
            k_mag = math.sqrt(kx * kx + ky * ky)
            if k_mag < 1e-10:
                continue

            omega = math.sqrt(_GRAVITY * k_mag * math.tanh(k_mag * depth))
            if omega < 1e-10:
                continue

            alpha = 5.0 / 16.0 * hs * hs * omega_p ** 4
            s_pm = alpha / omega ** 5 * math.exp(-1.25 * (omega_p / omega) ** 4)
            sigma = 0.07 if omega <= omega_p else 0.09
            r = math.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
            s_j = s_pm * gamma ** r

            cos_spread = math.cos(math.atan2(ky, kx) - direction)
            if cos_spread < 0:
                cos_spread = 0.0
            spreading = cos_spread ** 2 * 2.0 / math.pi

            s_k = s_j * 2.0 / omega
            amp = math.sqrt(s_k * dk * dk * spreading)
            phase = rng.uniform(0, _TWO_PI)
            spectrum[i, j] = amp * np.exp(1j * phase)

    return spectrum


class FFTWaveField:
    """Ocean wave field using 2D IFFT for O(N² log N) surface evaluation.

    Uses Warp tile_fft for GPU-native 2D transforms when available.
    Grid size is fixed at module constant FFT_GRID (default 64).
    """

    def __init__(
        self,
        domain_length: float = 500.0,
        wave_height: float = 1.5,
        wave_period: float = 8.0,
        wave_direction: float = 0.0,
        water_depth: float = 50.0,
        spectrum: str = "jonswap",
        device: str = "cuda",
    ):
        self.grid_size = FFT_GRID
        self.domain_length = domain_length
        self.depth = water_depth
        self.device = device
        self.time = 0.0
        self._dx = domain_length / FFT_GRID

        spec_np = _jonswap_2d(
            FFT_GRID, domain_length, wave_height, wave_period,
            wave_direction, water_depth,
        )
        self._spectrum_np = spec_np

        dk = _TWO_PI / domain_length
        half = FFT_GRID // 2
        omega_grid = np.zeros((FFT_GRID, FFT_GRID), dtype=np.float32)
        for i in range(FFT_GRID):
            for j in range(FFT_GRID):
                kx = (i - half) * dk
                ky = (j - half) * dk
                k_mag = math.sqrt(kx * kx + ky * ky)
                omega_grid[i, j] = math.sqrt(_GRAVITY * k_mag * math.tanh(k_mag * water_depth)) if k_mag > 1e-10 else 0.0
        self._omega_grid = omega_grid

        spec_shifted = np.fft.ifftshift(spec_np)
        vec2_data = np.stack([spec_shifted.real, spec_shifted.imag], axis=-1).astype(np.float32)
        self._spectrum_gpu = wp.array(vec2_data, dtype=wp.vec2f, device=device)

        self._temp1 = wp.zeros((FFT_GRID, FFT_GRID), dtype=wp.vec2f, device=device)
        self._temp2 = wp.zeros((FFT_GRID, FFT_GRID), dtype=wp.vec2f, device=device)
        self._surface = wp.zeros((FFT_GRID, FFT_GRID), dtype=wp.float32, device=device)

        self._use_tile_fft = True
        try:
            self._ifft_2d(self._spectrum_gpu, self._temp1)
            wp.launch(_normalize_real, dim=(FFT_GRID, FFT_GRID),
                      inputs=[self._temp1, self._surface, 1.0])
        except Exception:
            self._use_tile_fft = False
            spatial = np.fft.ifft2(np.fft.ifftshift(self._spectrum_np)).real.astype(np.float32) * (FFT_GRID * FFT_GRID)
            self._surface = wp.array(spatial, dtype=wp.float32, device=self.device)

    def _ifft_2d(self, src: wp.array, dst: wp.array):
        n_tiles = FFT_GRID // FFT_TRANSPOSE_DIM
        wp.launch_tiled(_ifft_rows, dim=[FFT_GRID, 1], inputs=[src], outputs=[self._temp1], block_dim=FFT_BLOCK)
        wp.launch_tiled(_transpose, dim=(n_tiles, n_tiles), inputs=[self._temp1], outputs=[self._temp2],
                        block_dim=FFT_TRANSPOSE_DIM * FFT_TRANSPOSE_DIM)
        wp.launch_tiled(_ifft_rows, dim=[FFT_GRID, 1], inputs=[self._temp2], outputs=[dst], block_dim=FFT_BLOCK)

    def step(self, dt: float = 1.0 / 60.0):
        self.time += dt

        phase_advance = self._omega_grid * self.time
        rotated = self._spectrum_np * np.exp(-1j * phase_advance)
        shifted = np.fft.ifftshift(rotated)
        vec2 = np.stack([shifted.real, shifted.imag], axis=-1).astype(np.float32)
        self._spectrum_gpu = wp.array(vec2, dtype=wp.vec2f, device=self.device)

        if self._use_tile_fft:
            self._ifft_2d(self._spectrum_gpu, self._temp1)
            wp.launch(_normalize_real, dim=(FFT_GRID, FFT_GRID),
                      inputs=[self._temp1, self._surface, 1.0])
        else:
            spatial = np.fft.ifft2(shifted).real.astype(np.float32) * (FFT_GRID * FFT_GRID)
            self._surface = wp.array(spatial, dtype=wp.float32, device=self.device)

    def get_surface_grid(self) -> wp.array:
        return self._surface

    def get_velocity_at(self, positions: wp.array, time: float | None = None) -> wp.array:
        if time is not None and abs(time - self.time) > 1e-6:
            self.step(time - self.time)

        n = positions.shape[0]
        out = wp.zeros(n, dtype=wp.vec3, device=self.device)
        wp.launch(
            _bilinear_sample_grid,
            dim=n,
            inputs=[
                self._surface, positions, out,
                float(1.0 / self._dx), 0.0, 0.0,
                FFT_GRID, FFT_GRID, float(self.depth), float(self.time),
            ],
            device=self.device,
        )
        return out
