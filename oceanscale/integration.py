# mypy: ignore-errors
"""End-to-end pipeline: GPU fluid → FSI → Newton physics → sensors → training."""

from __future__ import annotations

from typing import Any, cast

import newton
import numpy as np
import warp as wp

from oceanscale.fluid.mesh_boundary import MeshBoundary, make_box_mesh
from oceanscale.fluid.wave_fft import FFTWaveField
from oceanscale.graph_capture import GraphCapture
from oceanscale.sensors.ray_dvl import RayDVL
from oceanscale.sensors.ray_sonar import RaySonar
from oceanscale.tether import Tether


def _flat_plane_mesh(
    size: float = 100.0, y: float = -50.0, device: str = "cuda:0"
) -> wp.Mesh:
    h = size / 2.0
    verts = np.array(
        [[-h, y, -h], [h, y, -h], [h, y, h], [-h, y, h]], dtype=np.float32
    )
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(idx, dtype=wp.int32, device=device),
    )


def _obstacle_box_mesh(
    center: tuple[float, float, float], half: float, device: str = "cuda:0"
) -> wp.Mesh:
    v, i = make_box_mesh(center=center, half_extents=(half, half, half))
    return wp.Mesh(
        points=wp.array(v, dtype=wp.vec3, device=device),
        indices=wp.array(i.ravel(), dtype=wp.int32, device=device),
    )


@wp.kernel
def _apply_wave_drag(
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    wave_vx: wp.float32,
    wave_vy: wp.float32,
    wave_vz: wp.float32,
    drag_coeff: wp.float32,
):
    i = wp.tid()
    qd = body_qd[i]
    vx = wp.spatial_bottom(qd)[0] - wave_vx
    vy = wp.spatial_bottom(qd)[1] - wave_vy
    vz = wp.spatial_bottom(qd)[2] - wave_vz
    fx = -drag_coeff * vx * wp.abs(vx)
    fy = -drag_coeff * vy * wp.abs(vy)
    fz = -drag_coeff * vz * wp.abs(vz)
    cur = body_f[i]
    body_f[i] = wp.spatial_vectorf(
        wp.spatial_top(cur)[0],
        wp.spatial_top(cur)[1],
        wp.spatial_top(cur)[2],
        wp.spatial_bottom(cur)[0] + fx,
        wp.spatial_bottom(cur)[1] + fy,
        wp.spatial_bottom(cur)[2] + fz,
    )


class IntegratedPipeline:
    """End-to-end pipeline: GPU fluid → FSI → Newton physics → sensors → training."""

    def __init__(
        self,
        seabed_depth: float = -50.0,
        wave_height: float = 1.0,
        wave_period: float = 8.0,
        tether_length: float = 20.0,
        drag_coeff: float = 5.0,
        dt: float = 0.01,
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        self.dt = dt
        self._drag_coeff = drag_coeff
        self._time = 0.0

        self.wave = FFTWaveField(
            wave_height=wave_height,
            wave_period=wave_period,
            spectrum="jonswap",
            device=device,
        )

        self._seabed_mesh = _flat_plane_mesh(size=200.0, y=seabed_depth, device=device)
        self._obstacle_mesh = _obstacle_box_mesh(
            center=(5.0, seabed_depth + 5.0, 5.0), half=1.0, device=device
        )

        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(mass=10.0, com=(0.0, 0.0, 0.0))
        builder.add_shape_box(body=body, hx=0.25, hy=0.15, hz=0.15)
        self._rov_body = body

        self.tether = Tether(
            builder,
            anchor_pos=(0.0, 0.0, 0.0),
            attach_pos=(0.0, -tether_length, 0.0),
            n_segments=8,
        )

        self._rov_box_verts, self._rov_box_indices = make_box_mesh(
            center=(0.0, 0.0, 0.0), half_extents=(0.25, 0.15, 0.15)
        )
        self.mesh_boundary = MeshBoundary(
            self._rov_box_verts, self._rov_box_indices, device=device
        )

        builder.color()
        self.model = builder.finalize(device=device)
        self.solver = newton.solvers.SolverVBD(self.model)
        self.state_in = self.model.state()
        self.state_out = self.model.state()

        self.dvl = RayDVL(self._seabed_mesh, max_range=abs(seabed_depth) * 2.0, device=device)
        self.sonar = RaySonar(self._obstacle_mesh, n_rays=16, fov=90.0, max_range=30.0, device=device)

        self._pos_query = wp.zeros(1, dtype=wp.vec3, device=device)

    def step(self, action: np.ndarray | None = None, dt: float | None = None) -> dict:
        if dt is None:
            dt = self.dt
        self._time += dt

        self.wave.step(dt)

        pos_np = cast(Any, self.state_in.body_q).numpy()[self._rov_body, :3].copy()
        wp.copy(
            self._pos_query,
            wp.array(pos_np.reshape(1, 3).astype(np.float32), dtype=wp.vec3, device=self.device),
        )
        wave_vel_arr = self.wave.get_velocity_at(self._pos_query, time=self._time)
        wp.synchronize()
        wave_vel = wave_vel_arr.numpy()[0].copy()

        self.state_in.clear_forces()
        wp.launch(
            _apply_wave_drag,
            dim=self.model.body_count,
            inputs=[
                self.state_in.body_qd,
                self.state_in.body_f,
                float(wave_vel[0]),
                float(wave_vel[1]),
                float(wave_vel[2]),
                self._drag_coeff,
            ],
            device=self.device,
        )

        self.model.collide(self.state_in)
        contacts = self.model.contacts()
        self.solver.step(self.state_in, self.state_out, self.model.control(), contacts, dt)
        self.state_in, self.state_out = self.state_out, self.state_in

        pos_np = cast(Any, self.state_in.body_q).numpy()[self._rov_body, :3].copy()
        vel_np = cast(Any, self.state_in.body_qd).numpy()[self._rov_body].copy()

        dvl_result = self.dvl.measure(pos_np.astype(np.float32))
        sonar_ranges = self.sonar.scan(pos_np.astype(np.float32))
        tether_tension = self.tether.get_tension_estimate(self.state_in)

        target = np.array([0.0, -10.0, 0.0])
        dist = float(np.linalg.norm(pos_np - target))
        reward = float(np.exp(-dist / 5.0))

        return {
            "wave_velocity": wave_vel,
            "dvl": dvl_result,
            "sonar": sonar_ranges,
            "rov_position": pos_np,
            "rov_velocity": vel_np,
            "tether_tension": tether_tension,
            "reward": reward,
            "time": self._time,
        }

    def reset(self) -> dict:
        self._time = 0.0
        q_np = cast(Any, self.state_in.body_q).numpy()
        qd_np = cast(Any, self.state_in.body_qd).numpy()
        q_np[self._rov_body, :3] = [0.0, 0.0, 0.0]
        q_np[self._rov_body, 3:] = [0.0, 0.0, 0.0, 1.0]
        qd_np[self._rov_body, :] = 0.0
        wp.copy(
            self.state_in.body_q,
            wp.array(q_np, dtype=wp.transformf, device=self.device),
        )
        wp.copy(
            self.state_in.body_qd,
            wp.array(qd_np, dtype=wp.spatial_vectorf, device=self.device),
        )
        return self.step(dt=0.001)

    @property
    def observation_dim(self) -> int:
        return 3 + 6 + 4 + 16 + 1 + 3
