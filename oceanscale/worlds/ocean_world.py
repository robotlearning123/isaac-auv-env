# mypy: ignore-errors
"""Virtual ocean environment combining all NVIDIA Warp/Newton features."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from oceanscale.fluid.wave_fft import FFTWaveField
from oceanscale.sensors.ray_dvl import RayDVL
from oceanscale.sensors.ray_sonar import RaySonar

wp.init()


def _fbm_noise(x: np.ndarray, y: np.ndarray, octaves: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros_like(x)
    freq = 1.0
    amp = 1.0
    for _ in range(octaves):
        px = (x * freq * 7.13 + rng.uniform(-100, 100))
        py = (y * freq * 7.13 + rng.uniform(-100, 100))
        result += amp * (np.sin(px) * np.cos(py) + np.sin(px * 0.7 + py * 1.3) * 0.5)
        freq *= 2.0
        amp *= 0.5
    return result


def generate_seabed_mesh(
    size: float, depth: float, resolution: int, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0, size, resolution + 1, dtype=np.float32)
    ys = np.linspace(0, size, resolution + 1, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    noise = _fbm_noise(xg / size, yg / size, octaves=5, seed=seed)
    noise_range = noise.max() - noise.min()
    if noise_range > 1e-8:
        noise = (noise - noise.min()) / noise_range
    relief = depth * 0.15
    zg = -depth + noise * relief
    heightfield = zg.copy()

    n_verts = (resolution + 1) * (resolution + 1)
    verts = np.zeros((n_verts, 3), dtype=np.float32)
    verts[:, 0] = xg.ravel()
    verts[:, 1] = yg.ravel()
    verts[:, 2] = zg.ravel()

    indices = []
    for j in range(resolution):
        for i in range(resolution):
            v0 = j * (resolution + 1) + i
            v1 = v0 + 1
            v2 = v0 + (resolution + 1)
            v3 = v2 + 1
            indices.extend([v0, v2, v1, v1, v2, v3])
    indices = np.array(indices, dtype=np.int32)
    return verts, indices, heightfield


def generate_box_obstacle(
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    corners = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    faces = np.array([
        0,1,2, 0,2,3, 4,6,5, 4,7,6,
        0,4,5, 0,5,1, 2,6,7, 2,7,3,
        0,3,7, 0,7,4, 1,5,6, 1,6,2,
    ], dtype=np.int32)
    return corners, faces


def generate_dock_structure(
    position: tuple[float, float, float],
    size: float = 4.0,
    pillar_radius: float = 0.3,
    depth: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    px, py, pz = position
    all_verts, all_indices = [], []
    offset = 0
    half = size / 2.0
    pillar_positions = [
        (px - half, py - half), (px + half, py - half),
        (px - half, py + half), (px + half, py + half),
    ]
    for ppx, ppy in pillar_positions:
        v, idx = generate_box_obstacle(
            (ppx, ppy, pz - depth / 2), (pillar_radius, pillar_radius, depth / 2),
        )
        all_verts.append(v)
        all_indices.append(idx + offset)
        offset += len(v)
    v, idx = generate_box_obstacle((px, py, pz), (half + 0.5, half + 0.5, 0.2))
    all_verts.append(v)
    all_indices.append(idx + offset)
    return np.concatenate(all_verts), np.concatenate(all_indices)


def generate_rock(
    position: tuple[float, float, float], radius: float = 1.0,
    n_subdivisions: int = 1, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, 1], [-t, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    verts = raw / norms
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    for _ in range(n_subdivisions):
        edge_cache: dict[tuple[int,int], int] = {}
        new_faces = []
        for tri in faces:
            mids = []
            for e in range(3):
                a, b = tri[e], tri[(e + 1) % 3]
                key = (min(a, b), max(a, b))
                if key not in edge_cache:
                    mp = (verts[a] + verts[b]) / 2.0
                    mp /= np.linalg.norm(mp)
                    edge_cache[key] = len(verts)
                    verts = np.vstack([verts, mp])
                mids.append(edge_cache[key])
                m0, m1, m2 = mids if len(mids) == 3 else (0, 0, 0)
            m0, m1, m2 = mids
            new_faces.extend([
                [tri[0], m0, m2], [tri[1], m1, m0],
                [tri[2], m2, m1], [m0, m1, m2],
            ])
        faces = new_faces

    perturbation = 1.0 + rng.uniform(-0.2, 0.2, size=len(verts))
    verts = verts * perturbation[:, None] * radius
    px, py, pz = position
    verts = verts.astype(np.float32) + np.array([px, py, pz], dtype=np.float32)
    idx = np.array(faces, dtype=np.int32).ravel()
    return verts, idx


class OceanWorld:
    """A realistic virtual ocean environment combining all NVIDIA features.

    Contains seabed terrain, FFT wave field, depth-varying ocean currents,
    underwater obstacles (docks, rocks), and sensor-compatible meshes.
    """

    def __init__(
        self,
        domain_size: float = 200.0,
        depth: float = 50.0,
        seabed_resolution: int = 100,
        wave_height: float = 1.5,
        wave_period: float = 8.0,
        current_speed: float = 0.3,
        current_direction: float = 0.0,
        n_obstacles: int = 5,
        seed: int = 42,
        device: str = "cuda:0",
    ):
        self.domain_size = domain_size
        self.depth = depth
        self.device = device
        self.time = 0.0
        self._current_speed = current_speed
        self._current_dir = current_direction
        self._seed = seed

        sb_verts, sb_indices, self._heightfield = generate_seabed_mesh(
            domain_size, depth, seabed_resolution, seed=seed,
        )
        self._seabed_verts = sb_verts
        self._seabed_indices = sb_indices
        self._seabed_res = seabed_resolution
        self._seabed_mesh = wp.Mesh(
            points=wp.array(sb_verts, dtype=wp.vec3, device=device),
            indices=wp.array(sb_indices, dtype=wp.int32, device=device),
        )

        self.wave_field = FFTWaveField(
            wave_height=wave_height,
            wave_period=wave_period,
            domain_length=domain_size,
            water_depth=depth,
            device=device,
        )

        rng = np.random.default_rng(seed + 1)
        obs_verts_list, obs_idx_list = [], []
        offset = 0
        margin = domain_size * 0.1
        for i in range(n_obstacles):
            ox = rng.uniform(margin, domain_size - margin)
            oy = rng.uniform(margin, domain_size - margin)
            seabed_z = self.get_depth_at(ox, oy)
            if rng.random() < 0.3:
                v, idx = generate_dock_structure(
                    (ox, oy, seabed_z + 5.0), size=rng.uniform(3, 6),
                )
            else:
                v, idx = generate_rock(
                    (ox, oy, seabed_z + rng.uniform(0.5, 2.0)),
                    radius=rng.uniform(0.5, 3.0),
                    seed=seed + 100 + i,
                )
            obs_verts_list.append(v)
            obs_idx_list.append(idx + offset)
            offset += len(v)

        if obs_verts_list:
            obs_v = np.concatenate(obs_verts_list)
            obs_i = np.concatenate(obs_idx_list)
        else:
            obs_v = np.zeros((3, 3), dtype=np.float32)
            obs_i = np.array([0, 1, 2], dtype=np.int32)
        self._obstacle_verts = obs_v
        self._obstacle_indices = obs_i
        self._obstacle_mesh = wp.Mesh(
            points=wp.array(obs_v, dtype=wp.vec3, device=device),
            indices=wp.array(obs_i, dtype=wp.int32, device=device),
        )

        full_v = np.concatenate([sb_verts, obs_v])
        full_i = np.concatenate([sb_indices, obs_i + len(sb_verts)])
        self._full_mesh = wp.Mesh(
            points=wp.array(full_v, dtype=wp.vec3, device=device),
            indices=wp.array(full_i, dtype=wp.int32, device=device),
        )

        self._dvl = RayDVL(self._seabed_mesh, device=device)
        self._sonar = RaySonar(self._full_mesh, n_rays=64, fov=90.0, max_range=50.0, device=device)

    @property
    def seabed_mesh(self) -> wp.Mesh:
        return self._seabed_mesh

    @property
    def obstacle_mesh(self) -> wp.Mesh:
        return self._obstacle_mesh

    @property
    def full_mesh(self) -> wp.Mesh:
        return self._full_mesh

    @property
    def dvl(self) -> RayDVL:
        return self._dvl

    @property
    def sonar(self) -> RaySonar:
        return self._sonar

    def step(self, dt: float = 0.01):
        self.time += dt
        self.wave_field.step(dt)

    def get_current_at(self, positions: np.ndarray) -> np.ndarray:
        positions = np.atleast_2d(positions).astype(np.float32)
        n = len(positions)
        result = np.zeros((n, 3), dtype=np.float32)
        z = positions[:, 2]
        depth_factor = np.clip(1.0 + z / self.depth, 0.0, 1.0)
        cx = self._current_speed * math.cos(self._current_dir)
        cy = self._current_speed * math.sin(self._current_dir)
        result[:, 0] = cx * depth_factor
        result[:, 1] = cy * depth_factor

        pts_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        wave_vel = self.wave_field.get_velocity_at(pts_wp, time=self.time).numpy()
        result += wave_vel
        return result

    def get_depth_at(self, x: float, y: float) -> float:
        res = self._seabed_res
        xi = x / self.domain_size * res
        yi = y / self.domain_size * res
        i0 = max(0, min(int(xi), res))
        j0 = max(0, min(int(yi), res))
        return float(self._heightfield[j0, i0])

    def get_surface_elevation(self, x: float, y: float, time: float | None = None) -> float:
        grid = self.wave_field.get_surface_grid().numpy()
        dx = self.domain_size / self.wave_field.grid_size
        gi = int(x / dx) % self.wave_field.grid_size
        gj = int(y / dx) % self.wave_field.grid_size
        return float(grid[gi, gj])

    def measure_dvl(self, position: np.ndarray, orientation: np.ndarray | None = None) -> dict:
        return self._dvl.measure(position, orientation)

    def measure_sonar(self, position: np.ndarray, orientation: np.ndarray | None = None) -> np.ndarray:
        return self._sonar.scan(position, orientation)

    def get_environment_info(self) -> dict:
        return {
            "domain_size": self.domain_size,
            "depth": self.depth,
            "seabed_vertices": len(self._seabed_verts),
            "seabed_triangles": len(self._seabed_indices) // 3,
            "obstacle_vertices": len(self._obstacle_verts),
            "obstacle_triangles": len(self._obstacle_indices) // 3,
            "current_speed": self._current_speed,
            "current_direction": self._current_dir,
            "wave_height": self.wave_field._spectrum_np.max(),
            "time": self.time,
        }
