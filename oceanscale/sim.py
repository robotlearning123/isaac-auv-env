"""Unified GPU underwater simulation orchestrator.

Composes Newton rigid-body physics, Fossen hydrodynamics, ocean state
(water column, currents, waves), and sensors into a single step() call.
All components share the same physical world and simulation time.

Usage::

    from oceanscale.sim import OceanSim, OceanSimConfig, OceanConfig
    from oceanscale.vehicles import BlueROV2Heavy

    sim = OceanSim(OceanSimConfig(
        vehicle=BlueROV2Heavy(),
        ocean=OceanConfig(current_speed=0.2, wave_height=0.5),
        n_envs=64,
    ))
    obs = sim.reset()
    for _ in range(1000):
        action = np.random.uniform(-1, 1, (64, 6))
        obs = sim.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import warp as wp


@dataclass
class OceanConfig:
    """Ocean environment parameters."""

    rho_water: float = 1025.0
    current_speed: float = 0.0
    current_direction: float = 0.0
    wave_height: float = 0.0
    wave_period: float = 8.0
    surface_temp: float = 20.0
    bottom_temp: float = 4.0
    max_depth: float = 200.0


@dataclass
class SensorMount:
    """Sensor to attach to the vehicle."""

    type: str
    config: Any = None


@dataclass
class OceanSimConfig:
    """Complete simulation configuration."""

    vehicle: Any
    ocean: OceanConfig = field(default_factory=OceanConfig)
    sensors: list[SensorMount] = field(default_factory=list)
    n_envs: int = 1
    dt: float = 1 / 240
    device: str = "cuda:0"
    init_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
    init_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    seabed_mesh: Any = None
    current_drag_coeff: float = 5.0


class OceanSim:
    """Unified GPU underwater simulation.

    Composes Newton physics + Fossen hydrodynamics + ocean state +
    sensors into a single step() call. Replaces the three parallel
    integration paths (ROVEnv, OceanWorld, UnifiedDemo) with one
    shared orchestration loop.
    """

    def __init__(self, config: OceanSimConfig) -> None:
        self.cfg = config
        self.device = config.device
        self.dt = config.dt
        self.n_envs = config.n_envs
        self.time = 0.0

        wp.init()
        self._build_physics()
        self._init_hydro()
        self._init_ocean()
        self._init_sensors()
        self._step_count = np.zeros(config.n_envs, dtype=np.int32)

    def _build_physics(self) -> None:
        import newton as nt

        vehicle = self.cfg.vehicle
        mass = vehicle.mass

        Ix = getattr(vehicle, "Ix", 0.26)
        Iy = getattr(vehicle, "Iy", 0.23)
        Iz = getattr(vehicle, "Iz", 0.37)
        radius = (5.0 * max(Ix, Iy, Iz) / (2.0 * mass)) ** 0.5

        template = nt.ModelBuilder()
        template.gravity = 0.0
        body = template.add_body(mass=mass)
        template.add_shape_sphere(body, radius=radius)

        x, y, z = self.cfg.init_pos
        qx, qy, qz, qw = self.cfg.init_quat
        template.joint_q = [x, y, z, qx, qy, qz, qw]
        template.joint_qd = [0.0] * 6

        scene = nt.ModelBuilder()
        scene.replicate(template, world_count=self.n_envs, spacing=(0.0, 0.0, 0.0))
        self.model = scene.finalize(device=self.device)

        self.state_curr = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        nt.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_curr
        )
        self.solver = nt.solvers.SolverSemiImplicit(self.model)

    def _init_hydro(self) -> None:
        from oceanscale.hydro.tier1 import Tier1
        from oceanscale.hydro.tier1_kernels import tier1_zero_wrench

        vehicle = self.cfg.vehicle
        n_thrusters = vehicle.n_thrusters

        tier1_kw: dict[str, Any] = {
            "n_thrusters": n_thrusters,
        }
        if hasattr(vehicle, "tier1_kwargs"):
            tier1_kw.update(vehicle.tier1_kwargs())

        self.tier1 = Tier1(
            n_envs=self.n_envs,
            device=self.device,
            rho_water=self.cfg.ocean.rho_water,
            **tier1_kw,
        )

        coeffs = vehicle.set_coeffs_kwargs()
        self.tier1.set_coeffs(**coeffs)

        self._n_thrusters = n_thrusters
        self._u_cmd = wp.zeros(
            (self.n_envs, n_thrusters), dtype=wp.float32, device=self.device
        )
        self._quat_buf = wp.zeros(self.n_envs, dtype=wp.quatf, device=self.device)
        self._zero_wrench_kernel = tier1_zero_wrench

        T_matrix = coeffs.get("T_matrix")
        self._t_pinv: np.ndarray | None = None
        if T_matrix is not None:
            self._t_pinv = np.linalg.pinv(
                np.array(T_matrix, dtype=np.float64)
            ).astype(np.float32)

    def _init_ocean(self) -> None:
        ocean = self.cfg.ocean
        self._current_field = None
        self._wave_field = None
        self._water_column = None

        if ocean.current_speed > 0:
            from oceanscale.currents import OceanCurrentField

            self._current_field = OceanCurrentField(
                background_speed=ocean.current_speed,
                background_direction=ocean.current_direction,
                device=self.device,
            )

        if ocean.wave_height > 0:
            from oceanscale.fluid.wave_fft import FFTWaveField

            self._wave_field = FFTWaveField(
                wave_height=ocean.wave_height,
                wave_period=ocean.wave_period,
                device=self.device,
            )

        if ocean.surface_temp != ocean.bottom_temp:
            from oceanscale.ocean_properties import WaterColumn

            self._water_column = WaterColumn(
                surface_temperature=ocean.surface_temp,
                bottom_temperature=ocean.bottom_temp,
                max_depth=ocean.max_depth,
                device=self.device,
            )

    def _init_sensors(self) -> None:
        self._sensors: dict[str, Any] = {}
        mesh = self.cfg.seabed_mesh
        for mount in self.cfg.sensors:
            self._mount_sensor(mount, mesh)

    def _mount_sensor(self, mount: SensorMount, mesh: Any) -> None:
        if mount.type == "dvl" and mesh is not None:
            from oceanscale.sensors.ray_dvl import RayDVL

            self._sensors["dvl"] = RayDVL(seabed_mesh=mesh, device=self.device)
        elif mount.type == "sonar" and mesh is not None:
            from oceanscale.sensors.ray_sonar import RaySonar

            self._sensors["sonar"] = RaySonar(
                environment_mesh=mesh, device=self.device
            )
        elif mount.type == "magnetometer":
            from oceanscale.sensors.magnetometer import Magnetometer, MagnetometerConfig

            cfg = (
                mount.config
                if isinstance(mount.config, MagnetometerConfig)
                else MagnetometerConfig()
            )
            self._sensors["magnetometer"] = Magnetometer(config=cfg)
        elif mount.type == "imaging_sonar" and mesh is not None:
            from oceanscale.sensors.imaging_sonar import ImagingSonar, ImagingSonarConfig

            cfg = (
                mount.config
                if isinstance(mount.config, ImagingSonarConfig)
                else ImagingSonarConfig()
            )
            self._sensors["imaging_sonar"] = ImagingSonar(
                environment_mesh=mesh, config=cfg, device=self.device
            )

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def _extract_quat(self) -> wp.array:
        body_q_t = wp.to_torch(self.state_curr.body_q)
        quat_dst = wp.to_torch(self._quat_buf)
        quat_dst.copy_(body_q_t[:, 3:7])
        return self._quat_buf

    def _get_positions(self) -> np.ndarray:
        return self.state_curr.body_q.numpy()[: self.n_envs, :3].astype(np.float32)

    def _get_orientations(self) -> np.ndarray:
        return self.state_curr.body_q.numpy()[: self.n_envs, 3:7].astype(np.float32)

    def _get_velocities(self) -> np.ndarray:
        qd = self.state_curr.body_qd.numpy()[: self.n_envs]
        return qd.astype(np.float32)

    def _get_body_state_torch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-only body state via wp.to_torch (zero-copy)."""
        body_q = wp.to_torch(self.state_curr.body_q)
        body_qd = wp.to_torch(self.state_curr.body_qd)
        return body_q[:, :3], body_q[:, 3:7], body_qd

    def observe_torch(self) -> dict[str, torch.Tensor]:
        """GPU-native observations — zero-copy torch tensors on device."""
        pos, quat, vel = self._get_body_state_torch()
        return {
            "position": pos,
            "orientation": quat,
            "linear_velocity": vel[:, 3:6],
            "angular_velocity": vel[:, :3],
        }

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _physics_step(self, u_cmd: wp.array) -> None:
        """Run one physics substep (shared by step/step_torch)."""
        self._u_cmd = u_cmd
        body_f = self.state_curr.body_f
        body_qd = self.state_curr.body_qd

        wp.launch(
            self._zero_wrench_kernel,
            dim=self.n_envs,
            inputs=[body_f],
            device=self.device,
        )
        self.tier1.compute_wrench(
            nu=body_qd, quat=self._extract_quat(), u_cmd=self._u_cmd, dt=self.dt
        )
        self.tier1.write_to_body_f(body_f)

        if self._current_field is not None:
            self._apply_current_force()

        self.solver.step(self.state_curr, self.state_next, self.control, None, self.dt)
        self.state_curr, self.state_next = self.state_next, self.state_curr
        self.time += self.dt
        self._step_count += 1

        if self._wave_field is not None:
            self._wave_field.step(self.dt)

    def step_torch(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """GPU-native step: accepts and returns CUDA tensors.

        Args:
            action: (n_envs, 6) or (6,) wrench command in [-1, 1].

        Returns:
            dict with position, orientation, linear_velocity,
            angular_velocity as torch.Tensor on device.
        """
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.n_envs, -1)
        action = action.clamp(-1.0, 1.0)

        if self._t_pinv is not None:
            if not hasattr(self, "_t_pinv_torch"):
                self._t_pinv_torch = torch.from_numpy(self._t_pinv).to(self.device)
            u_cmd = (action @ self._t_pinv_torch.T).clamp(-1.0, 1.0)
        else:
            u_cmd = torch.zeros(
                self.n_envs, self._n_thrusters, device=self.device, dtype=torch.float32
            )
            for i in range(min(6, self._n_thrusters)):
                u_cmd[:, i] = action[:, i]

        u_cmd_wp = wp.from_torch(u_cmd.contiguous(), dtype=wp.float32)
        self._physics_step(u_cmd_wp)
        wp.synchronize()
        return self.observe_torch()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        """Advance simulation by one timestep (numpy I/O).

        Args:
            action: (n_envs, 6) or (6,) wrench command in [-1, 1].

        Returns:
            dict with position, orientation, linear_velocity,
            angular_velocity, time, step_count, and sensors.
        """
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = np.tile(action, (self.n_envs, 1))
        action = np.clip(action, -1.0, 1.0)

        if self._t_pinv is not None:
            u_cmd = np.clip(action @ self._t_pinv.T, -1.0, 1.0)
        else:
            u_cmd = np.zeros((self.n_envs, self._n_thrusters), dtype=np.float32)
            for i in range(min(6, self._n_thrusters)):
                u_cmd[:, i] = action[:, i]

        u_cmd_wp = wp.array(u_cmd, dtype=wp.float32, device=self.device)
        self._physics_step(u_cmd_wp)
        wp.synchronize()
        return self._observe()

    def _apply_current_force(self) -> None:
        positions = self._get_positions()
        current_vel = self._current_field.velocity_at(positions, self.time)
        C_d = self.cfg.current_drag_coeff
        body_f_np = self.state_curr.body_f.numpy()
        for i in range(self.n_envs):
            v = current_vel[i]
            speed = float(np.linalg.norm(v))
            if speed > 1e-4:
                force = C_d * speed * v
                body_f_np[i, 3] += force[0]
                body_f_np[i, 4] += force[1]
                body_f_np[i, 5] += force[2]
        self.state_curr.body_f.assign(body_f_np)

    def _observe(self) -> dict[str, Any]:
        positions = self._get_positions()
        orientations = self._get_orientations()
        vel = self._get_velocities()

        obs: dict[str, Any] = {
            "position": positions,
            "orientation": orientations,
            "linear_velocity": vel[:, 3:6],
            "angular_velocity": vel[:, :3],
            "time": self.time,
            "step_count": self._step_count.copy(),
        }

        if self._sensors:
            sensor_data: dict[str, Any] = {}
            for name, sensor in self._sensors.items():
                if name in ("dvl", "sonar"):
                    reading = sensor.measure(positions[0], orientations[0])
                    sensor_data[name] = reading
                elif name == "magnetometer":
                    sensor_data[name] = sensor.measure(orientations[0])
                elif name == "imaging_sonar":
                    sensor_data[name] = sensor.scan(positions[0], orientations[0])
            obs["sensors"] = sensor_data

        return obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Reset all environments to initial conditions."""
        import newton as nt

        x, y, z = self.cfg.init_pos
        qx, qy, qz, qw = self.cfg.init_quat
        init_q = np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)

        q = self.model.joint_q.numpy()
        qd = self.model.joint_qd.numpy()
        for i in range(self.n_envs):
            q[i * 7 : (i + 1) * 7] = init_q
            qd[i * 6 : (i + 1) * 6] = 0.0
        self.model.joint_q.assign(q)
        self.model.joint_qd.assign(qd)
        nt.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_curr
        )

        self.time = 0.0
        self._step_count[:] = 0
        return self._observe()

    def reset_envs(self, env_ids: np.ndarray) -> dict[str, Any]:
        """Partial reset: reset only specified env indices."""
        import newton as nt

        x, y, z = self.cfg.init_pos
        qx, qy, qz, qw = self.cfg.init_quat
        init_q = np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)

        q = self.model.joint_q.numpy()
        qd = self.model.joint_qd.numpy()
        for i in env_ids:
            q[i * 7 : (i + 1) * 7] = init_q
            qd[i * 6 : (i + 1) * 6] = 0.0
        self.model.joint_q.assign(q)
        self.model.joint_qd.assign(qd)
        nt.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_curr
        )

        self._step_count[env_ids] = 0
        return self._observe()

    def close(self) -> None:
        """Release GPU resources."""
        pass
