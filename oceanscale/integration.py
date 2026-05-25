# mypy: ignore-errors
"""End-to-end pipeline: GPU fluid → FSI → Newton physics → sensors → training."""

from __future__ import annotations

from typing import Any, cast

import newton
import numpy as np
import warp as wp

from oceanscale.acoustics import AcousticPropagation, WaterColumn
from oceanscale.currents import OceanCurrentField
from oceanscale.fluid.mesh_boundary import MeshBoundary, make_box_mesh
from oceanscale.fluid.wave_fft import FFTWaveField
from oceanscale.propulsion import BuoyancyEngine, PropellerThruster
from oceanscale.sensors.ray_dvl import RayDVL
from oceanscale.sensors.ray_sonar import RaySonar
from oceanscale.tether import Tether


def _flat_plane_mesh(size: float = 100.0, z: float = -50.0, device: str = "cuda:0") -> wp.Mesh:
    h = size / 2.0
    verts = np.array(
        [[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]], dtype=np.float32
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
def _apply_ocean_forces(
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    rov_body: wp.int32,
    wave_vx: wp.float32,
    wave_vy: wp.float32,
    wave_vz: wp.float32,
    current_vx: wp.float32,
    current_vy: wp.float32,
    current_vz: wp.float32,
    drag_coeff: wp.float32,
    thrust_x: wp.float32,
    thrust_y: wp.float32,
    thrust_z: wp.float32,
    buoyancy_z: wp.float32,
):
    i = wp.tid()
    qd = body_qd[i]
    linear = wp.spatial_top(qd)
    vx = linear[0] - wave_vx - current_vx
    vy = linear[1] - wave_vy - current_vy
    vz = linear[2] - wave_vz - current_vz
    fx = -drag_coeff * vx * wp.abs(vx)
    fy = -drag_coeff * vy * wp.abs(vy)
    fz = -drag_coeff * vz * wp.abs(vz)
    if i == rov_body:
        fx = fx + thrust_x
        fy = fy + thrust_y
        fz = fz + thrust_z + buoyancy_z
    cur = body_f[i]
    body_f[i] = wp.spatial_vectorf(
        wp.spatial_top(cur)[0] + fx,
        wp.spatial_top(cur)[1] + fy,
        wp.spatial_top(cur)[2] + fz,
        wp.spatial_bottom(cur)[0],
        wp.spatial_bottom(cur)[1],
        wp.spatial_bottom(cur)[2],
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
        self._seabed_depth = seabed_depth
        self._trim_depth = -10.0
        water_depth = abs(seabed_depth)

        self.water_column = WaterColumn(max_depth=water_depth)
        self.current_field = OceanCurrentField(seabed_depth=water_depth, device=device)
        self.acoustics = AcousticPropagation(self.water_column, max_depth=water_depth, device=device)
        self.propeller = PropellerThruster()
        neutral_volume = 10.0 / self.water_column.density_at(10.0)
        self.buoyancy = BuoyancyEngine(hull_volume=neutral_volume, hull_mass=10.0)

        self.wave = FFTWaveField(
            wave_height=wave_height,
            wave_period=wave_period,
            spectrum="jonswap",
            device=device,
        )

        self._seabed_mesh = _flat_plane_mesh(size=200.0, z=seabed_depth, device=device)
        self._obstacle_center = np.array([5.0, 5.0, seabed_depth + 5.0], dtype=np.float32)
        self._obstacle_mesh = _obstacle_box_mesh(
            center=tuple(self._obstacle_center.tolist()), half=1.0, device=device
        )

        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self._trim_depth), wp.quat_identity()),
            mass=10.0,
            com=(0.0, 0.0, 0.0),
        )
        builder.add_shape_box(body=body, hx=0.25, hy=0.15, hz=0.15)
        self._rov_body = body

        self.tether = Tether(
            builder,
            anchor_pos=(0.0, 0.0, 0.0),
            attach_pos=(0.0, 0.0, -tether_length),
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
        if action is None:
            action = np.zeros(6, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.size < 6:
            action = np.pad(action, (0, 6 - action.size))
        action = np.clip(action[:6], -1.0, 1.0)
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
        current = self.current_field.velocity_at(pos_np.reshape(1, 3), self._time)[0]
        water_density = self.water_column.density_at(float(pos_np[2]))
        sound_speed = self.water_column.sound_speed_at(float(pos_np[2]))
        self.buoyancy.set_target_volume(float(action[2]) * self.buoyancy.max_volume_change)
        self.buoyancy.step(dt)
        buoyancy_force = self.buoyancy.net_buoyancy(water_density)

        qd_np = cast(Any, self.state_in.body_qd).numpy()[self._rov_body].copy()
        linear_vel = qd_np[:3]
        trim_force = -200.0 * (float(pos_np[2]) - self._trim_depth) - 80.0 * float(linear_vel[2])
        thrust = np.array([
            self.propeller.thrust(float(action[0]) * self.propeller.max_rpm, abs(float(linear_vel[0]))),
            self.propeller.thrust(float(action[1]) * self.propeller.max_rpm, abs(float(linear_vel[1]))),
            self.propeller.thrust(float(action[2]) * self.propeller.max_rpm, abs(float(linear_vel[2]))),
        ], dtype=np.float32)

        self.state_in.clear_forces()
        wp.launch(
            _apply_ocean_forces,
            dim=self.model.body_count,
            inputs=[
                self.state_in.body_qd,
                self.state_in.body_f,
                self._rov_body,
                float(wave_vel[0]),
                float(wave_vel[1]),
                float(wave_vel[2]),
                float(current[0]),
                float(current[1]),
                float(current[2]),
                self._drag_coeff,
                float(thrust[0]),
                float(thrust[1]),
                float(thrust[2]),
                float(buoyancy_force + trim_force),
            ],
            device=self.device,
        )

        self.model.collide(self.state_in)
        contacts = self.model.contacts()
        self.solver.step(self.state_in, self.state_out, self.model.control(), contacts, dt)
        self.state_in, self.state_out = self.state_out, self.state_in

        return self._observe(wave_vel, current, water_density, sound_speed)

    def reset(self) -> dict:
        self._time = 0.0
        self.buoyancy.reset()
        for state in (self.state_in, self.state_out):
            q_np = cast(Any, state.body_q).numpy()
            qd_np = cast(Any, state.body_qd).numpy()
            q_np[self._rov_body, :3] = [0.0, 0.0, self._trim_depth]
            q_np[self._rov_body, 3:] = [0.0, 0.0, 0.0, 1.0]
            qd_np[self._rov_body, :] = 0.0
            wp.copy(
                state.body_q,
                wp.array(q_np, dtype=wp.transformf, device=self.device),
            )
            wp.copy(
                state.body_qd,
                wp.array(qd_np, dtype=wp.spatial_vectorf, device=self.device),
            )
            state.clear_forces()

        pos_np = np.array([[0.0, 0.0, self._trim_depth]], dtype=np.float32)
        current = self.current_field.velocity_at(pos_np, self._time)[0]
        water_density = self.water_column.density_at(abs(self._trim_depth))
        sound_speed = self.water_column.sound_speed_at(abs(self._trim_depth))
        return self._observe(np.zeros(3, dtype=np.float32), current, water_density, sound_speed)

    def _observe(
        self,
        wave_vel: np.ndarray,
        current: np.ndarray,
        water_density: float,
        sound_speed: float,
    ) -> dict:
        pos_np = cast(Any, self.state_in.body_q).numpy()[self._rov_body, :3].copy()
        vel_np = cast(Any, self.state_in.body_qd).numpy()[self._rov_body].copy()

        dvl_result = self.dvl.measure(pos_np.astype(np.float32))
        sonar_ranges = self.sonar.scan(pos_np.astype(np.float32))
        tether_tension = self.tether.get_tension_estimate(self.state_in)

        target = np.array([0.0, 0.0, self._trim_depth])
        dist = float(np.linalg.norm(pos_np - target))
        reward = float(np.exp(-dist / 5.0))
        acoustic_loss = self.acoustics.transmission_loss(
            float(max(np.linalg.norm(pos_np - self._obstacle_center), 1.0)),
            source_depth=abs(float(pos_np[2])),
            receiver_depth=abs(float(self._obstacle_center[2])),
        )

        return {
            "wave_velocity": wave_vel,
            "current": current.astype(np.float32),
            "water_density": float(water_density),
            "sound_speed": float(sound_speed),
            "acoustic_loss_db": float(acoustic_loss),
            "dvl": dvl_result,
            "sonar": sonar_ranges,
            "rov_position": pos_np,
            "rov_velocity": vel_np,
            "tether_tension": tether_tension,
            "reward": reward,
            "time": self._time,
        }

    @property
    def observation_dim(self) -> int:
        return 3 + 6 + 4 + 16 + 1 + 3
