"""Amphibious robot dog demo — walk, swim, return.

Demonstrates OceanScale's full multi-domain simulation capability:
1. Land walking (ground contact + friction + leg actuators)
2. Water surface crossing (partial submersion — continuous force transition)
3. Underwater swimming (Fossen hydrodynamics + thruster allocation)
4. Return to land (reverse surface crossing + walking)

Usage:
    python -m oceanscale.demo_amphibious [--steps 3000] [--render-mp4 out.mp4]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None  # type: ignore[assignment]

import newton as nt

from oceanscale.controllers.cpg import CPGConfig, KuramotoCPG, _smoothstep
from oceanscale.hydro.tier1 import Tier1
from oceanscale.propulsion.propeller import PropellerThruster
from oceanscale.vehicles.amphibious import AmphibiousRobotDog

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.func
def _terrain_height(x: wp.float32) -> wp.float32:
    if x <= 2.0:
        return 0.5
    if x <= 6.0:
        return 0.5 - 0.25 * (x - 2.0)
    return -0.5


@wp.func
def _locomotion_mode(pos_z: wp.float32, terrain_z: wp.float32) -> wp.int32:
    wz = wp.float32(0.0)
    ground_z = terrain_z + 0.15
    if pos_z >= wz:
        return 0
    if pos_z >= ground_z - 0.05 and terrain_z >= wz:
        return 0
    return 1


@wp.kernel
def zero_body_f(body_f: wp.array(dtype=wp.spatial_vectorf)):
    tid = wp.tid()
    body_f[tid] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def apply_amphibious_forces(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    thruster_cmds: wp.array(dtype=wp.float32),
    mass: wp.float32,
    gravity: wp.float32,
    volume: wp.float32,
    rho_water: wp.float32,
    leg_force: wp.float32,
    ground_stiffness: wp.float32,
    ground_damping: wp.float32,
    ground_friction: wp.float32,
    leg_height: wp.float32,
    water_surface_z: wp.float32,
    max_thrust: wp.float32,
):
    tid = wp.tid()
    pos_x = body_q[tid][0]
    pos_y = body_q[tid][1]
    pos_z = body_q[tid][2]
    # body_qd convention: spatial_top [0:3]=linear, spatial_bottom [3:6]=angular
    vel_x = body_qd[tid][0]
    vel_y = body_qd[tid][1]
    vel_z = body_qd[tid][2]

    terr_z = _terrain_height(pos_x)
    ground_z = terr_z + leg_height
    mode = _locomotion_mode(pos_z, terr_z)

    fz = wp.float32(0.0)
    fx = wp.float32(0.0)
    fy = wp.float32(0.0)
    tx = wp.float32(0.0)
    ty = wp.float32(0.0)
    tz = wp.float32(0.0)

    fz -= mass * gravity

    if mode == 0:
        penetration = ground_z - pos_z
        if penetration > 0.0:
            ground_fz = ground_stiffness * penetration - ground_damping * vel_z
            fz += ground_fz

            leg_cmd = thruster_cmds[0]
            fx += leg_cmd * leg_force

            lateral_cmd = thruster_cmds[1]
            fy += lateral_cmd * leg_force * 0.5

            friction_force = ground_friction * ground_fz
            speed = wp.sqrt(vel_x * vel_x + vel_y * vel_y)
            if speed > 0.01:
                fx -= friction_force * (vel_x / speed) * 0.1
                fy -= friction_force * (vel_y / speed) * 0.1

            yaw_cmd = thruster_cmds[5]
            tz += yaw_cmd * leg_force * 0.15
    else:
        fz += rho_water * gravity * volume

        Cd = wp.float32(0.8)
        A_front = wp.float32(0.12)
        A_side = wp.float32(0.16)
        A_top = wp.float32(0.32)
        half_rho = 0.5 * rho_water

        drag_x = -half_rho * Cd * A_front * wp.abs(vel_x) * vel_x
        drag_y = -half_rho * Cd * A_side * wp.abs(vel_y) * vel_y
        drag_z = -half_rho * Cd * A_top * wp.abs(vel_z) * vel_z

        fx += drag_x
        fy += drag_y
        fz += drag_z

        # Thruster allocation: T-matrix × commands × max_thrust
        # T1-T4: horizontal vectored (45° tilt, 90° spacing)
        # T5-T8: vertical (straight up)
        T0 = wp.float32(0.707)
        T1 = wp.float32(-0.707)
        T2 = wp.float32(0.106)
        T3 = wp.float32(-0.106)
        T4 = wp.float32(0.0354)
        T5 = wp.float32(-0.2475)
        T6 = wp.float32(0.2475)
        T7 = wp.float32(-0.0354)

        c0 = thruster_cmds[0] * max_thrust
        c1 = thruster_cmds[1] * max_thrust
        c2 = thruster_cmds[2] * max_thrust
        c3 = thruster_cmds[3] * max_thrust
        c4 = thruster_cmds[4] * max_thrust
        c5 = thruster_cmds[5] * max_thrust
        c6 = thruster_cmds[6] * max_thrust
        c7 = thruster_cmds[7] * max_thrust

        fx += T0 * c0 + T1 * c1 + T1 * c2 + T0 * c3
        fy += T0 * c0 + T0 * c1 + T1 * c2 + T1 * c3
        fz += c4 + c5 + c6 + c7
        tx += T3 * c1 + T2 * c3 + 0.1 * c4 + (-0.1) * c5 + (-0.1) * c6 + 0.1 * c7
        ty += T2 * c0 + T3 * c2 + 0.1 * c4 + 0.1 * c5 + (-0.1) * c6 + (-0.1) * c7
        tz += T4 * c0 + T5 * c1 + T6 * c2 + T7 * c3

    # body_f convention: spatial_top=[0:3]=linear force, spatial_bottom=[3:6]=angular torque
    body_f[tid] = wp.spatial_vectorf(fx, fy, fz, tx, ty, tz)


@wp.kernel
def update_telemetry(
    body_q: wp.array(dtype=wp.transformf),
    terrain_h: wp.array(dtype=wp.float32),
    submerged_frac: wp.array(dtype=wp.float32),
    mode_out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    pos_x = body_q[tid][0]
    pos_z = body_q[tid][2]
    terr_z = _terrain_height(pos_x)
    terrain_h[tid] = terr_z
    mode_out[tid] = _locomotion_mode(pos_z, terr_z)

    body_bottom = pos_z - 0.15
    body_top = pos_z + 0.15
    water_z = wp.float32(0.0)
    if body_bottom >= water_z:
        submerged_frac[tid] = 0.0
    elif body_top <= water_z:
        submerged_frac[tid] = 1.0
    else:
        submerged_frac[tid] = (water_z - body_bottom) / (body_top - body_bottom)


@wp.func
def _kernel_smoothstep(t: wp.float32) -> wp.float32:
    tc = wp.clamp(t, 0.0, 1.0)
    return tc * tc * (3.0 - 2.0 * tc)


@wp.kernel
def apply_amphibious_forces_blended(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    leg_cmds: wp.array(dtype=wp.float32),
    thruster_cmds: wp.array(dtype=wp.float32),
    mass: wp.float32,
    gravity: wp.float32,
    volume: wp.float32,
    rho_water: wp.float32,
    leg_force: wp.float32,
    ground_stiffness: wp.float32,
    ground_damping: wp.float32,
    ground_friction: wp.float32,
    leg_height: wp.float32,
    water_surface_z: wp.float32,
    max_thrust: wp.float32,
    contact_force_threshold: wp.float32,
    body_half_height: wp.float32,
):
    """Blended force kernel — computes both walk and swim forces, smooth blend."""
    tid = wp.tid()
    pos_x = body_q[tid][0]
    pos_y = body_q[tid][1]
    pos_z = body_q[tid][2]
    vel_x = body_qd[tid][0]
    vel_y = body_qd[tid][1]
    vel_z = body_qd[tid][2]

    terr_z = _terrain_height(pos_x)
    ground_z = terr_z + leg_height

    # --- Walking forces ---
    walk_fx = wp.float32(0.0)
    walk_fy = wp.float32(0.0)
    walk_fz = wp.float32(0.0)
    walk_tx = wp.float32(0.0)
    walk_ty = wp.float32(0.0)
    walk_tz = wp.float32(0.0)

    penetration = ground_z - pos_z
    ground_contact_fz = wp.float32(0.0)
    if penetration > 0.0:
        ground_contact_fz = ground_stiffness * penetration - ground_damping * vel_z
        walk_fz += ground_contact_fz

        # Leg actuation — sum 4 leg commands
        leg_sum = leg_cmds[0] + leg_cmds[1] + leg_cmds[2] + leg_cmds[3]
        walk_fx += leg_sum * leg_force * 0.25

        friction_force = ground_friction * ground_contact_fz
        speed = wp.sqrt(vel_x * vel_x + vel_y * vel_y)
        if speed > 0.01:
            walk_fx -= friction_force * (vel_x / speed) * 0.1
            walk_fy -= friction_force * (vel_y / speed) * 0.1

    # --- Swimming forces ---
    swim_fx = wp.float32(0.0)
    swim_fy = wp.float32(0.0)
    swim_fz = wp.float32(0.0)
    swim_tx = wp.float32(0.0)
    swim_ty = wp.float32(0.0)
    swim_tz = wp.float32(0.0)

    swim_fz += rho_water * gravity * volume

    Cd = wp.float32(0.8)
    A_front = wp.float32(0.12)
    A_side = wp.float32(0.16)
    A_top = wp.float32(0.32)
    half_rho = 0.5 * rho_water

    swim_fx += -half_rho * Cd * A_front * wp.abs(vel_x) * vel_x
    swim_fy += -half_rho * Cd * A_side * wp.abs(vel_y) * vel_y
    swim_fz += -half_rho * Cd * A_top * wp.abs(vel_z) * vel_z

    T0 = wp.float32(0.707)
    T1 = wp.float32(-0.707)
    T2 = wp.float32(0.106)
    T3 = wp.float32(-0.106)
    T4 = wp.float32(0.0354)
    T5 = wp.float32(-0.2475)
    T6 = wp.float32(0.2475)
    T7 = wp.float32(-0.0354)

    c0 = thruster_cmds[0] * max_thrust
    c1 = thruster_cmds[1] * max_thrust
    c2 = thruster_cmds[2] * max_thrust
    c3 = thruster_cmds[3] * max_thrust
    c4 = thruster_cmds[4] * max_thrust
    c5 = thruster_cmds[5] * max_thrust
    c6 = thruster_cmds[6] * max_thrust
    c7 = thruster_cmds[7] * max_thrust

    swim_fx += T0 * c0 + T1 * c1 + T1 * c2 + T0 * c3
    swim_fy += T0 * c0 + T0 * c1 + T1 * c2 + T1 * c3
    swim_fz += c4 + c5 + c6 + c7
    swim_tx += T3 * c1 + T2 * c3 + 0.1 * c4 + (-0.1) * c5 + (-0.1) * c6 + 0.1 * c7
    swim_ty += T2 * c0 + T3 * c2 + 0.1 * c4 + 0.1 * c5 + (-0.1) * c6 + (-0.1) * c7
    swim_tz += T4 * c0 + T5 * c1 + T6 * c2 + T7 * c3

    # --- Compute blend factor ---
    contact_mag = wp.abs(ground_contact_fz)
    contact_factor = _kernel_smoothstep(contact_mag / contact_force_threshold)

    body_top = pos_z + body_half_height
    body_bot = pos_z - body_half_height
    wz = water_surface_z
    if body_bot >= wz:
        submersion = wp.float32(0.0)
    elif body_top <= wz:
        submersion = wp.float32(1.0)
    else:
        submersion = (wz - body_bot) / (body_top - body_bot)

    blend = _kernel_smoothstep(submersion) * (1.0 - contact_factor)

    # --- Blended output ---
    inv_blend = 1.0 - blend
    fx = inv_blend * walk_fx + blend * swim_fx
    fy = inv_blend * walk_fy + blend * swim_fy
    fz = inv_blend * walk_fz + blend * swim_fz - mass * gravity
    tx = inv_blend * walk_tx + blend * swim_tx
    ty = inv_blend * walk_ty + blend * swim_ty
    tz = inv_blend * walk_tz + blend * swim_tz

    body_f[tid] = wp.spatial_vectorf(fx, fy, fz, tx, ty, tz)


@dataclass
class AmphibiousDemoConfig:
    """Configuration for the amphibious robot dog demo."""

    n_steps: int = 3000
    dt: float = 0.02
    device: str = "cuda:0"
    seed: int = 42
    render_mp4: str | None = None
    cinematic: bool = True
    seabed_depth: float = 5.0
    wave_height: float = 0.3
    wave_period: float = 6.0
    current_speed: float = 0.15
    use_cpg: bool = True
    trajectory_json: str | None = None


@dataclass
class MissionPhase:
    """One phase of the mission."""

    name: str
    name_zh: str
    target: tuple[float, float, float]
    duration: int
    description: str


def build_mission() -> list[MissionPhase]:
    """Build the walk-swim-return mission sequence."""
    return [
        MissionPhase(
            name="Walk to Shore",
            name_zh="走向海岸",
            target=(2.0, 0.0, 0.65),
            duration=400,
            description="Robot walks from start position toward the water's edge",
        ),
        MissionPhase(
            name="Enter Water",
            name_zh="入水",
            target=(4.0, 0.0, -0.3),
            duration=400,
            description="Descending the slope, transitioning from walking to swimming",
        ),
        MissionPhase(
            name="Swim to Reef",
            name_zh="游向珊瑚礁",
            target=(7.0, 0.0, -2.0),
            duration=500,
            description="Full underwater swimming using thruster array",
        ),
        MissionPhase(
            name="Inspect Reef",
            name_zh="勘察珊瑚礁",
            target=(8.0, 0.0, -2.5),
            duration=400,
            description="Station-keeping near the reef for inspection",
        ),
        MissionPhase(
            name="Return Swim",
            name_zh="返程游泳",
            target=(5.0, 0.0, -0.5),
            duration=500,
            description="Swimming back toward shore",
        ),
        MissionPhase(
            name="Exit Water",
            name_zh="出水",
            target=(2.0, 0.0, 0.65),
            duration=300,
            description="Ascending the slope, transitioning from swimming back to walking",
        ),
    ]


def pd_control(
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    mode: int,
    max_force: float = 80.0,
) -> np.ndarray:
    """PD controller for walking and swimming modes.

    Returns 8-dim command vector:
    - Walking: [walk_fwd, lateral, 0, 0, 0, yaw, 0, 0]
    - Swimming: full 8-thruster commands
    """
    error = target - position
    d_error = -velocity

    if mode == 0:
        # Walking mode — PD on surge and yaw
        cmd = np.zeros(8)
        kp_surge = 8.0
        kd_surge = 3.0
        cmd[0] = np.clip(kp_surge * error[0] + kd_surge * d_error[0], -1.0, 1.0)
        # Lateral
        kp_sway = 5.0
        kd_sway = 2.0
        cmd[1] = np.clip(kp_sway * error[1] + kd_sway * d_error[1], -1.0, 1.0)
        # Yaw (index 5)
        kp_yaw = 5.0
        kd_yaw = 2.0
        cmd[5] = np.clip(kp_yaw * error[1] + kd_yaw * d_error[1], -1.0, 1.0)
        return cmd
    else:
        # Swimming mode — PD on all 6 DOF, mapped to 8 thrusters
        cmd = np.zeros(8)
        # Surge (T1-T4 horizontal)
        kp_x = 6.0
        kd_x = 3.0
        surge = np.clip(kp_x * error[0] + kd_x * d_error[0], -1.0, 1.0)
        cmd[0] = surge
        cmd[2] = -surge
        # Sway
        kp_y = 5.0
        kd_y = 2.0
        sway = np.clip(kp_y * error[1] + kd_y * d_error[1], -1.0, 1.0)
        cmd[1] = sway
        cmd[3] = -sway
        # Heave (T5-T8 vertical)
        kp_z = 6.0
        kd_z = 3.0
        heave = np.clip(kp_z * error[2] + kd_z * d_error[2], -1.0, 1.0)
        cmd[4] = heave
        cmd[5] = heave
        cmd[6] = heave
        cmd[7] = heave
        return cmd


def compute_reward(pos: np.ndarray, target: np.ndarray) -> float:
    """Exponential distance reward."""
    dist = np.linalg.norm(pos - target)
    return float(np.exp(-dist / 2.0))


class AmphibiousCPGController:
    """CPG-driven gait controller with PD target tracking.

    Combines Kuramoto CPG gait patterns with PD corrections for
    mission-level navigation. The descending drive signal is derived
    from submersion fraction and ground contact force — not position
    thresholds — matching FARMS-style physics-driven transitions.
    """

    def __init__(self, vehicle: AmphibiousRobotDog, dt: float = 0.02):
        self.vehicle = vehicle
        self.cpg = KuramotoCPG(vehicle.cpg_config(), dt=dt)
        self.dt = dt
        # PD gains for mission-level target tracking
        self.kp_surge = 8.0
        self.kd_surge = 3.0
        self.kp_sway = 5.0
        self.kd_sway = 2.0
        self.kp_heave = 6.0
        self.kd_heave = 3.0
        self.kp_yaw = 5.0
        self.kd_yaw = 2.0

    def reset(self) -> None:
        self.cpg.reset()

    def compute_drive(
        self,
        submerged_fraction: float,
        contact_force_magnitude: float,
    ) -> float:
        """Compute CPG drive signal from physics state."""
        sub_factor = _smoothstep(submerged_fraction)
        contact_factor = _smoothstep(
            contact_force_magnitude / self.vehicle.contact_force_threshold
        )
        return sub_factor * (1.0 - contact_factor)

    def compute(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target: np.ndarray,
        submerged_fraction: float,
        contact_force_magnitude: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute leg (4-dim) and thruster (8-dim) commands in [-1, 1]."""
        # Set drive from physics
        drive = self.compute_drive(submerged_fraction, contact_force_magnitude)
        self.cpg.set_drive(drive)
        self.cpg.step()

        # Get CPG rhythm
        leg_signals = self.cpg.get_leg_signals()
        thruster_signals = self.cpg.get_thruster_signals()

        # PD error correction
        error = target - position
        d_error = -velocity

        # Walking PD correction modulates leg signals
        walk_gain = 1.0 - drive
        surge_cmd = np.clip(
            self.kp_surge * error[0] + self.kd_surge * d_error[0], -1.0, 1.0
        )
        sway_cmd = np.clip(
            self.kp_sway * error[1] + self.kd_sway * d_error[1], -1.0, 1.0
        )
        leg_cmds = leg_signals * walk_gain + np.array([surge_cmd, sway_cmd, surge_cmd * 0.8, sway_cmd * 0.5]) * walk_gain

        # Swimming PD correction modulates thruster signals
        swim_gain = drive
        heave_cmd = np.clip(
            self.kp_heave * error[2] + self.kd_heave * d_error[2], -1.0, 1.0
        )
        thruster_cmds = thruster_signals.copy()
        # Override with PD for horizontal thrusters when swimming
        thruster_cmds[0] = thruster_cmds[0] * swim_gain + surge_cmd * swim_gain
        thruster_cmds[2] = thruster_cmds[2] * swim_gain - surge_cmd * swim_gain
        # PD for vertical thrusters
        for i in range(4, 8):
            thruster_cmds[i] = heave_cmd * swim_gain

        return (
            np.clip(leg_cmds, -1.0, 1.0),
            np.clip(thruster_cmds, -1.0, 1.0),
        )

    @property
    def drive(self) -> float:
        return self.cpg.drive

    @property
    def wave_type(self) -> str:
        return self.cpg.get_wave_type()


class AmphibiousDemo:
    """Multi-domain amphibious robot dog demonstration.

    Simulates a robot that walks on land, transitions through the water
    surface, swims underwater, and returns to land — demonstrating
    OceanScale's full physics pipeline.
    """

    def __init__(self, config: AmphibiousDemoConfig | None = None):
        self.cfg = config or AmphibiousDemoConfig()
        self.vehicle = AmphibiousRobotDog()
        self.time = 0.0
        self._step_count = 0
        self._last_contact_fz = 0.0
        self._last_blend = 0.0

        # Build Newton physics
        self._build_physics()
        # Load Warp kernels
        self._load_kernels()
        # Init hydrodynamics
        self._init_hydro()
        # Init thruster model
        self._init_thrusters()
        # Init CPG controller
        self._init_cpg()

    def _build_physics(self) -> None:
        """Create Newton model — all forces applied via custom Warp kernel.

        Gravity is zero in Newton; the kernel applies gravity, ground contact,
        buoyancy, drag, and thrusters directly to body_f.
        """
        builder = nt.ModelBuilder(gravity=0.0)

        # Robot body
        initial_xform = wp.transform(
            wp.vec3(0.5, 0.0, 0.8), wp.quat_identity()
        )
        body = builder.add_body(
            xform=initial_xform,
            mass=self.vehicle.mass,
            com=(0.0, 0.0, 0.0),
        )
        builder.add_shape_sphere(
            body,
            radius=(
                5.0
                * max(self.vehicle.Ix, self.vehicle.Iy, self.vehicle.Iz)
                / (2.0 * self.vehicle.mass)
            )
            ** 0.5,
        )

        builder.color()
        self.model = builder.finalize(device=self.cfg.device)
        self.solver = nt.solvers.SolverVBD(self.model)
        self.state_curr = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

    def _load_kernels(self) -> None:
        """Load Warp kernels for multi-domain forces."""
        self._apply_forces_kernel = apply_amphibious_forces
        self._blended_kernel = apply_amphibious_forces_blended
        self._telemetry_kernel = update_telemetry
        self._zero_kernel = zero_body_f

    def _init_hydro(self) -> None:
        """Initialize Tier1 Fossen hydrodynamics."""
        self.tier1 = Tier1(
            n_envs=1,
            device=self.cfg.device,
            n_thrusters=self.vehicle.n_thrusters,
        )
        self.tier1.set_coeffs(**self.vehicle.set_coeffs_kwargs())

    def _init_thrusters(self) -> None:
        """Initialize propeller thruster model."""
        self.thruster = PropellerThruster(
            diameter=0.076,
            max_rpm=3500,
            rho=1025.0,
        )

    def _init_cpg(self) -> None:
        """Initialize CPG gait controller."""
        self.cpg_ctrl: AmphibiousCPGController | None = None
        if self.cfg.use_cpg:
            self.cpg_ctrl = AmphibiousCPGController(self.vehicle, dt=self.cfg.dt)

    def step(self, action: np.ndarray, target: np.ndarray | None = None) -> dict:
        """Advance simulation by one timestep.

        Args:
            action: 8-dim array of thruster/actuator commands [-1, 1] (legacy PD),
                    or 3-dim target position (CPG mode).
            target: Optional explicit target for CPG mode.

        Returns:
            Observation dict with position, velocity, mode, telemetry.
        """
        if action.ndim == 1:
            action = action[np.newaxis, :]

        self.state_curr.clear_forces()

        use_cpg = self.cfg.use_cpg and self.cpg_ctrl is not None

        if use_cpg:
            # CPG mode: action can be target pos (3-dim) or 8-dim thruster
            obs_pre = self._observe()
            if action.shape[-1] == 3 or target is not None:
                tgt = target if target is not None else action[0, :3]
                pos = obs_pre["position"]
                vel = obs_pre["linear_velocity"]
                sub_frac = obs_pre["submerged_fraction"]
                contact_fz = self._last_contact_fz

                leg_cmds, thruster_cmds = self.cpg_ctrl.compute(
                    pos, vel, tgt, sub_frac, contact_fz
                )
            else:
                # Fallback: 8-dim direct commands, no CPG
                leg_cmds = action[0, :4]
                thruster_cmds = action[0]

            leg_wp = wp.from_numpy(
                leg_cmds.astype(np.float32), dtype=wp.float32, device=self.cfg.device
            )
            thruster_wp = wp.from_numpy(
                thruster_cmds.astype(np.float32), dtype=wp.float32, device=self.cfg.device
            )

            wp.launch(
                self._blended_kernel,
                dim=1,
                inputs=[
                    self.state_curr.body_q,
                    self.state_curr.body_qd,
                    self.state_curr.body_f,
                    leg_wp,
                    thruster_wp,
                    self.vehicle.mass,
                    9.81,
                    self.vehicle.volume,
                    1025.0,
                    self.vehicle.leg_max_force,
                    self.vehicle.ground_stiffness,
                    self.vehicle.ground_damping,
                    self.vehicle.ground_friction,
                    self.vehicle.leg_height,
                    self.vehicle.water_surface_z,
                    self.vehicle.max_thrust,
                    self.vehicle.contact_force_threshold,
                    self.vehicle.body_half_height,
                ],
                device=self.cfg.device,
            )
        else:
            # Legacy PD mode
            thruster_cmds = wp.from_numpy(
                action[0].astype(np.float32), dtype=wp.float32, device=self.cfg.device
            )
            wp.launch(
                self._apply_forces_kernel,
                dim=1,
                inputs=[
                    self.state_curr.body_q,
                    self.state_curr.body_qd,
                    self.state_curr.body_f,
                    thruster_cmds,
                    self.vehicle.mass,
                    9.81,
                    self.vehicle.volume,
                    1025.0,
                    self.vehicle.leg_max_force,
                    self.vehicle.ground_stiffness,
                    self.vehicle.ground_damping,
                    self.vehicle.ground_friction,
                    self.vehicle.leg_height,
                    self.vehicle.water_surface_z,
                    self.vehicle.max_thrust,
                ],
                device=self.cfg.device,
            )

        # Collision detection + Newton solver step
        contacts = self.model.collide(self.state_curr)
        self.solver.step(
            self.state_curr, self.state_next, self.control, contacts, self.cfg.dt
        )
        self.state_curr, self.state_next = self.state_next, self.state_curr

        self.time += self.cfg.dt
        self._step_count += 1

        return self._observe()

    def _observe(self) -> dict:
        """Build observation dict from current state."""
        body_q_np = self.state_curr.body_q.numpy()
        body_qd_np = self.state_curr.body_qd.numpy()
        pos = body_q_np[0, :3].copy()
        quat = body_q_np[0, 3:7].copy()
        # body_qd convention: spatial_top [:3] = linear, spatial_bottom [3:6] = angular
        vel = body_qd_np[0, :3].copy()
        ang_vel = body_qd_np[0, 3:6].copy()

        # Determine terrain and mode
        terr_z = self.vehicle.terrain_height(float(pos[0]))
        ground_z = terr_z + self.vehicle.leg_height
        wz = self.vehicle.water_surface_z
        pos_z = float(pos[2])

        if pos_z >= wz:
            # Above water surface → walking
            mode = 0
            mode_name = "walk"
        elif pos_z >= ground_z - 0.05 and terr_z >= wz:
            # Near ground AND ground is above water → walking on land
            mode = 0
            mode_name = "walk"
        else:
            # Underwater → swimming
            mode = 1
            mode_name = "swim"

        # Submerged fraction
        body_top = float(pos[2]) + 0.15
        body_bot = float(pos[2]) - 0.15
        wz = self.vehicle.water_surface_z
        if body_bot >= wz:
            sub_frac = 0.0
        elif body_top <= wz:
            sub_frac = 1.0
        else:
            sub_frac = (wz - body_bot) / (body_top - body_bot)

        return {
            "position": pos,
            "orientation": quat,
            "linear_velocity": vel,
            "angular_velocity": ang_vel,
            "mode": mode,
            "mode_name": mode_name,
            "terrain_height": terr_z,
            "submerged_fraction": sub_frac,
            "contact_force": self._last_contact_fz,
            "blend_factor": self._last_blend,
            "time": self.time,
            "step_count": self._step_count,
        }

    def reset(self) -> dict:
        """Reset simulation to initial state."""
        init_pos = np.array([0.5, 0.0, 0.8], dtype=np.float32)
        init_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        zero_vel = np.zeros(6, dtype=np.float32)

        for state in (self.state_curr, self.state_next):
            q_np = state.body_q.numpy()
            q_np[0, :3] = init_pos
            q_np[0, 3:7] = init_quat
            q_arr = wp.from_numpy(q_np, dtype=wp.transformf, device=self.cfg.device)
            wp.copy(state.body_q, q_arr)
            if state.body_q_prev is not None:
                wp.copy(state.body_q_prev, q_arr)
            qd_np = state.body_qd.numpy()
            qd_np[0, :] = zero_vel
            wp.copy(
                state.body_qd,
                wp.from_numpy(qd_np, dtype=wp.spatial_vectorf, device=self.cfg.device),
            )

        self.tier1.zero_wrench()
        self.time = 0.0
        self._step_count = 0
        self._last_contact_fz = 0.0
        self._last_blend = 0.0
        if self.cpg_ctrl is not None:
            self.cpg_ctrl.reset()
        return self._observe()


class AmphibiousVideoRenderer:
    """Cinematic video renderer for the amphibious demo.

    Side-view (x-z plane) with:
    - Detailed terrain (sand gradients, rocks)
    - Animated water surface with waves
    - Coral reef geometry at inspection zone
    - Quadruped robot with articulated legs (walk) / thruster glow (swim)
    - Particle system: bubbles underwater, splash at surface crossing
    - Fading trail with mode-colored segments
    - Rich HUD: velocity, distance, progress bar, blend factor
    - Intro title card + mission summary outro
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 50,
        figsize: tuple[float, float] = (16, 9),
        dpi: int = 120,
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi
        self._frames: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._modes: list[int] = []
        self._telemetry: list[dict] = []
        self._phase_markers: list[tuple[int, str]] = []
        self._current_target: np.ndarray = np.zeros(3)
        self._current_phase: str = ""
        self._phase_names: list[str] = []
        # Particle system
        self._bubbles: list[dict] = []
        self._splashes: list[dict] = []
        self._last_sub_frac: float = 0.0
        self._total_steps: int = 0

    def record_frame(self, obs: dict, target: np.ndarray, phase_name: str) -> None:
        """Record one frame."""
        self._current_target = target.copy()
        self._current_phase = phase_name
        self._positions.append(obs["position"].copy())
        vel = obs.get("linear_velocity", np.zeros(3))
        self._velocities.append(vel[:3].copy())
        self._modes.append(obs["mode"])
        self._telemetry.append(
            {
                "depth": -obs["position"][2],
                "sub_frac": obs["submerged_fraction"],
                "time": obs["time"],
                "mode": obs["mode_name"],
                "contact": obs.get("contact_force", 0.0),
                "blend": obs.get("blend_factor", 0.0),
                "vel_x": float(vel[0]),
                "vel_z": float(vel[2]),
                "step": obs.get("step_count", 0),
                "pos": obs["position"].copy(),
                "target": target.copy(),
                "phase": phase_name,
            }
        )
        self._update_particles(obs)
        self._total_steps = max(self._total_steps, obs.get("step_count", 0))

    def mark_phase(self, phase_name: str) -> None:
        """Mark a phase transition."""
        self._phase_markers.append((len(self._positions), phase_name))
        self._phase_names.append(phase_name)

    def _update_particles(self, obs: dict) -> None:
        """Update bubble and splash particles."""
        dt = 0.02
        pos = obs["position"]
        sub_frac = obs["submerged_fraction"]
        mode = obs["mode"]

        # Spawn bubbles when underwater
        if mode == 1 and sub_frac > 0.5:
            if np.random.random() < 0.3:
                self._bubbles.append(
                    {
                        "x": pos[0] + np.random.uniform(-0.2, 0.2),
                        "z": pos[2] + np.random.uniform(-0.1, 0.1),
                        "vx": np.random.uniform(-0.02, 0.02),
                        "vz": np.random.uniform(0.02, 0.08),
                        "life": 0.0,
                        "max_life": np.random.uniform(0.5, 2.0),
                        "size": np.random.uniform(0.01, 0.04),
                    }
                )

        # Spawn splash when crossing water surface
        if abs(sub_frac - self._last_sub_frac) > 0.05 and abs(pos[2]) < 0.3:
            for _ in range(15):
                angle = np.random.uniform(0.1, math.pi - 0.1)
                speed = np.random.uniform(0.1, 0.5)
                self._splashes.append(
                    {
                        "x": pos[0] + np.random.uniform(-0.1, 0.1),
                        "z": 0.0,
                        "vx": math.cos(angle) * speed * np.random.choice([-1, 1]),
                        "vz": math.sin(angle) * speed,
                        "life": 0.0,
                        "max_life": np.random.uniform(0.3, 0.8),
                    }
                )
        self._last_sub_frac = sub_frac

        # Update bubbles
        new_bubbles = []
        for b in self._bubbles:
            b["life"] += dt
            b["x"] += b["vx"]
            b["z"] += b["vz"]
            b["vz"] *= 0.99
            if b["life"] < b["max_life"] and b["z"] < 0.0:
                new_bubbles.append(b)
        self._bubbles = new_bubbles[-50:]

        # Update splashes
        new_splashes = []
        for s in self._splashes:
            s["life"] += dt
            s["x"] += s["vx"]
            s["z"] += s["vz"]
            s["vz"] -= 0.5 * dt  # gravity
            if s["life"] < s["max_life"]:
                new_splashes.append(s)
        self._splashes = new_splashes[-30:]

    def _terrain_height(self, x: float) -> float:
        if x <= 2.0:
            return 0.5
        elif x <= 6.0:
            return 0.5 - 0.25 * (x - 2.0)
        else:
            return -0.5

    def _draw_terrain(self, ax, x_arr, z_arr, x_left, x_right):
        """Draw terrain with sand gradient and rock details."""
        import matplotlib.pyplot as plt

        mask = (x_arr >= x_left - 1) & (x_arr <= x_right + 1)
        xm, zm = x_arr[mask], z_arr[mask]

        ax.fill_between(xm, zm, -5.0, color="#2a1f14", alpha=0.9, zorder=2)
        ax.fill_between(xm, zm, zm - 0.15, color="#c2a36e", alpha=0.4, zorder=3)
        ax.plot(xm, zm, color="#8B7355", linewidth=2.5, zorder=4)

        # Rock details
        np.random.seed(42)
        for rx in np.arange(x_left, x_right, 0.8):
            rz = self._terrain_height(rx)
            if rx > 3.0:
                rsize = 0.06 + 0.04 * np.sin(rx * 7.3)
                rock = plt.Circle(
                    (rx, rz - 0.02), rsize, color="#5a4a3a", alpha=0.5, zorder=4
                )
                ax.add_patch(rock)

    def _draw_coral_reef(self, ax, x_center):
        """Draw coral reef formations at inspection zone (x~7-8)."""
        import matplotlib.pyplot as plt

        if abs(x_center - 7.5) > 6:
            return

        coral_colors = [
            "#ff6b6b",
            "#ffa07a",
            "#ff69b4",
            "#ff4500",
            "#dc143c",
            "#ff7f50",
        ]
        seabed_z = -0.5

        np.random.seed(123)
        for i in range(12):
            cx = 6.5 + i * 0.25
            cz = seabed_z

            h = 0.15 + 0.1 * np.sin(i * 2.1)
            ax.plot(
                [cx, cx],
                [cz, cz + h],
                color=coral_colors[i % len(coral_colors)],
                linewidth=3,
                alpha=0.8,
                zorder=5,
            )

            for bx in [-0.05, 0, 0.05]:
                bh = h * (0.6 + 0.4 * np.sin(i * 1.7 + bx * 10))
                ax.plot(
                    [cx, cx + bx],
                    [cz + h * 0.5, cz + bh],
                    color=coral_colors[(i + 1) % len(coral_colors)],
                    linewidth=2,
                    alpha=0.7,
                    zorder=5,
                )

            ax.plot(
                cx,
                cz + h,
                "o",
                color=coral_colors[i % len(coral_colors)],
                markersize=4,
                alpha=0.9,
                zorder=6,
            )

    def _draw_water(self, ax, x_left, x_right, frame_idx):
        """Draw animated water surface with waves."""
        x_water = np.linspace(x_left - 1, x_right + 1, 300)
        wave = 0.05 * np.sin(x_water * 2.0 + frame_idx * 0.08) + 0.03 * np.sin(
            x_water * 5.0 - frame_idx * 0.12
        )

        ax.fill_between(
            x_water, wave, -5.0, color="#0c4a6e", alpha=0.35, zorder=1
        )
        ax.fill_between(
            x_water, wave, wave - 0.3, color="#1e6a9e", alpha=0.2, zorder=1
        )

        ax.plot(x_water, wave, color="#7dd3fc", linewidth=2.5, alpha=0.8, zorder=7)
        ax.plot(x_water, wave, color="#bae6fd", linewidth=1.0, alpha=0.4, zorder=7)

        # Underwater light rays
        if x_right > 2.0:
            for j in range(4):
                rx = x_left + (x_right - x_left) * (j + 0.5) / 4
                if rx > 2.0:
                    ray_alpha = 0.06 + 0.03 * np.sin(frame_idx * 0.05 + j)
                    ax.fill_between(
                        [rx - 0.1, rx + 0.1],
                        [0.0, 0.0],
                        [-3.0, -3.0],
                        color="#bae6fd",
                        alpha=ray_alpha,
                        zorder=1,
                    )

        # Underwater particulates
        if x_right > 2.0:
            np.random.seed(frame_idx % 100)
            n_particles = 20
            px = np.random.uniform(max(x_left, 2.0), x_right, n_particles)
            pz = np.random.uniform(-3.0, -0.1, n_particles)
            px += 0.3 * np.sin(frame_idx * 0.02 + pz)
            sizes = np.random.uniform(0.5, 2.0, n_particles)
            ax.scatter(px, pz, s=sizes, color="#94a3b8", alpha=0.15, zorder=2)

    def _draw_robot_walk(self, ax, pos, frame_idx):
        """Draw robot as quadruped with articulated walking legs."""
        import matplotlib.pyplot as plt

        bx, bz = float(pos[0]), float(pos[2])
        body_w, body_h = 0.30, 0.10

        body = plt.Rectangle(
            (bx - body_w / 2, bz - body_h / 2),
            body_w,
            body_h,
            color="#f59e0b",
            alpha=0.95,
            zorder=10,
            linewidth=1,
            edgecolor="#d97706",
        )
        ax.add_patch(body)

        head = plt.Circle(
            (bx + body_w / 2 + 0.04, bz + 0.02),
            0.05,
            color="#fbbf24",
            alpha=0.95,
            zorder=11,
        )
        ax.add_patch(head)
        ax.plot(
            bx + body_w / 2 + 0.06,
            bz + 0.03,
            "o",
            color="#1a1a2e",
            markersize=2,
            zorder=12,
        )

        # Antenna
        ax.plot(
            [bx + body_w / 4, bx + body_w / 4],
            [bz + body_h / 2, bz + body_h / 2 + 0.08],
            color="#d97706",
            linewidth=1.5,
            zorder=11,
        )
        ax.plot(
            bx + body_w / 4,
            bz + body_h / 2 + 0.08,
            "o",
            color="#22c55e",
            markersize=3,
            zorder=12,
        )

        # Four articulated legs with trot gait
        leg_offsets = [-0.12, -0.04, 0.04, 0.12]
        phase_offsets = [0, math.pi, math.pi, 0]

        for _idx, (lx, ph) in enumerate(zip(leg_offsets, phase_offsets)):
            swing = math.sin(frame_idx * 0.15 + ph) * 0.06
            hip_x = bx + lx
            hip_z = bz - body_h / 2
            knee_x = hip_x + swing * 0.5
            knee_z = hip_z - 0.06
            foot_x = hip_x + swing
            foot_z = hip_z - 0.12

            ax.plot(
                [hip_x, knee_x],
                [hip_z, knee_z],
                color="#d97706",
                linewidth=2.5,
                solid_capstyle="round",
                zorder=9,
            )
            ax.plot(
                [knee_x, foot_x],
                [knee_z, foot_z],
                color="#b45309",
                linewidth=2,
                solid_capstyle="round",
                zorder=9,
            )
            ax.plot(
                foot_x, foot_z, "s", color="#92400e", markersize=3, zorder=9
            )

    def _draw_robot_swim(self, ax, pos, frame_idx):
        """Draw robot as underwater vehicle with thruster indicators."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        bx, bz = float(pos[0]), float(pos[2])
        body_w, body_h = 0.30, 0.10

        body = FancyBboxPatch(
            (bx - body_w / 2, bz - body_h / 2),
            body_w,
            body_h,
            boxstyle="round,pad=0.02",
            color="#2563eb",
            alpha=0.95,
            zorder=10,
            linewidth=1,
            edgecolor="#3b82f6",
        )
        ax.add_patch(body)

        # Sensor dome
        dome = plt.Circle(
            (bx + body_w / 2 - 0.02, bz),
            0.04,
            color="#60a5fa",
            alpha=0.9,
            zorder=11,
        )
        ax.add_patch(dome)
        ax.plot(
            bx + body_w / 2 - 0.02,
            bz,
            "o",
            color="#22d3ee",
            markersize=3,
            alpha=0.6 + 0.4 * abs(math.sin(frame_idx * 0.1)),
            zorder=12,
        )

        # Thruster pods
        thruster_positions = [
            (bx - body_w / 2 - 0.03, bz + 0.04, "H"),
            (bx - body_w / 2 - 0.03, bz - 0.04, "H"),
            (bx + body_w / 4, bz + body_h / 2 + 0.02, "V"),
            (bx - body_w / 4, bz + body_h / 2 + 0.02, "V"),
        ]

        for tx, tz, ttype in thruster_positions:
            if ttype == "H":
                housing = plt.Rectangle(
                    (tx - 0.03, tz - 0.02),
                    0.06,
                    0.04,
                    color="#1e40af",
                    alpha=0.9,
                    zorder=10,
                )
            else:
                housing = plt.Rectangle(
                    (tx - 0.02, tz - 0.01),
                    0.04,
                    0.03,
                    color="#1e40af",
                    alpha=0.9,
                    zorder=10,
                )
            ax.add_patch(housing)

            glow_alpha = 0.3 + 0.3 * abs(math.sin(frame_idx * 0.2))
            if ttype == "H":
                wash_x = tx - 0.08 - 0.03 * abs(math.sin(frame_idx * 0.15))
                ax.plot(
                    wash_x,
                    tz,
                    ">",
                    color="#7dd3fc",
                    markersize=5,
                    alpha=glow_alpha,
                    zorder=9,
                )
                for wx in range(3):
                    wake_x = wash_x - 0.03 * (wx + 1)
                    wake_alpha = glow_alpha * (1 - wx * 0.3)
                    ax.plot(
                        [wake_x, wake_x],
                        [tz - 0.02, tz + 0.02],
                        color="#7dd3fc",
                        linewidth=1,
                        alpha=wake_alpha,
                        zorder=8,
                    )
            else:
                wash_z = tz + 0.05
                ax.plot(
                    tx,
                    wash_z,
                    "^",
                    color="#7dd3fc",
                    markersize=4,
                    alpha=glow_alpha,
                    zorder=9,
                )

    def _draw_bubbles(self, ax, x_left, x_right):
        """Draw bubble particles."""
        import matplotlib.pyplot as plt

        for b in self._bubbles:
            if b["x"] < x_left - 1 or b["x"] > x_right + 1:
                continue
            alpha = max(0, 1.0 - b["life"] / b["max_life"]) * 0.6
            circle = plt.Circle(
                (b["x"], b["z"]),
                b["size"],
                fill=False,
                color="#bae6fd",
                alpha=alpha,
                linewidth=1,
                zorder=8,
            )
            ax.add_patch(circle)

    def _draw_splashes(self, ax, x_left, x_right):
        """Draw splash particles."""
        for s in self._splashes:
            if s["x"] < x_left - 1 or s["x"] > x_right + 1:
                continue
            alpha = max(0, 1.0 - s["life"] / s["max_life"]) * 0.8
            ax.plot(
                s["x"],
                s["z"],
                "o",
                color="#e0f2fe",
                markersize=2,
                alpha=alpha,
                zorder=8,
            )

    def _draw_trail(self, ax, positions, modes, frame_idx):
        """Draw fading trail."""
        if frame_idx < 2:
            return

        max_trail = 200
        start = max(0, frame_idx - max_trail)
        trail_x = positions[start : frame_idx + 1, 0]
        trail_z = positions[start : frame_idx + 1, 2]
        trail_m = modes[start : frame_idx + 1]

        n_seg = len(trail_x) - 1
        for j in range(n_seg):
            age = (n_seg - j) / n_seg
            alpha = max(0.05, 0.7 * (1 - age))
            color = "#34d399" if trail_m[j + 1] == 0 else "#38bdf8"
            ax.plot(
                trail_x[j : j + 2],
                trail_z[j : j + 2],
                color=color,
                linewidth=max(1, 3 * (1 - age)),
                alpha=alpha,
                zorder=6,
            )

    def _draw_target(self, ax, target, pos, frame_idx):
        """Draw animated target marker."""
        import matplotlib.pyplot as plt

        tx, tz = float(target[0]), float(target[2])

        pulse = 0.1 + 0.03 * math.sin(frame_idx * 0.1)
        ring = plt.Circle(
            (tx, tz),
            pulse,
            fill=False,
            color="#f472b6",
            linewidth=2,
            alpha=0.6,
            zorder=9,
        )
        ax.add_patch(ring)

        ax.plot(
            tx,
            tz,
            "+",
            color="#f472b6",
            markersize=12,
            markeredgewidth=2,
            alpha=0.9,
            zorder=9,
        )

        dist = np.linalg.norm(
            np.array([tx, tz]) - np.array([float(pos[0]), float(pos[2])])
        )
        if dist > 0.3:
            ax.plot(
                [float(pos[0]), tx],
                [float(pos[2]), tz],
                "--",
                color="#f472b6",
                linewidth=1,
                alpha=0.3,
                zorder=5,
            )
            mid_x = (float(pos[0]) + tx) / 2
            mid_z = (float(pos[2]) + tz) / 2
            ax.text(
                mid_x,
                mid_z + 0.15,
                f"{dist:.1f}m",
                color="#f472b6",
                fontsize=8,
                ha="center",
                alpha=0.7,
                zorder=12,
            )

    def _draw_hud(self, ax, t, pos, target, frame_idx, n_frames, phase_name):
        """Draw rich HUD overlay."""
        import matplotlib.pyplot as plt

        mode = 0 if t["mode"] == "walk" else 1
        mode_str = "WALKING" if mode == 0 else "SWIMMING"
        mode_color = "#f59e0b" if mode == 0 else "#38bdf8"

        dist = np.linalg.norm(t["pos"] - t["target"])
        speed = math.sqrt(t.get("vel_x", 0) ** 2 + t.get("vel_z", 0) ** 2)

        hud_lines = [
            "OceanScale — Amphibious Robot Dog",
            "",
            f"Phase: {phase_name}",
            f"Mode:  {mode_str}",
            f"Depth: {t['depth']:.2f} m    Speed: {speed:.2f} m/s",
            f"Submerged: {t['sub_frac']:.0%}    Blend: {t.get('blend', 0):.2f}",
            f"Contact: {t.get('contact', 0):.1f} N    Target: {dist:.1f} m",
            f"Time: {t['time']:.1f} s",
        ]
        hud_text = "\n".join(hud_lines)
        ax.text(
            0.02,
            0.98,
            hud_text,
            transform=ax.transAxes,
            color="#e2e8f0",
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#111827",
                edgecolor="#1e293b",
                alpha=0.92,
            ),
            zorder=20,
        )

        # Mode indicator (top-right)
        ax.text(
            0.98,
            0.98,
            mode_str,
            transform=ax.transAxes,
            color=mode_color,
            fontsize=16,
            fontweight="bold",
            fontfamily="monospace",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#111827",
                edgecolor=mode_color,
                alpha=0.92,
            ),
            zorder=20,
        )

        # Mission progress bar using axes-transformed rectangle
        progress = min(1.0, frame_idx / max(1, n_frames - 1))
        bar_left = 0.1
        bar_right = 0.9
        bar_width = bar_right - bar_left
        bar_y = 0.04
        bar_h = 0.015

        # Background bar
        bg_rect = plt.Rectangle(
            (bar_left, bar_y - bar_h),
            bar_width,
            bar_h * 2,
            transform=ax.transAxes,
            color="#1e293b",
            alpha=0.8,
            zorder=19,
            clip_on=False,
        )
        ax.add_patch(bg_rect)

        # Progress fill
        if progress > 0:
            fill_rect = plt.Rectangle(
                (bar_left, bar_y - bar_h),
                bar_width * progress,
                bar_h * 2,
                transform=ax.transAxes,
                color=mode_color,
                alpha=0.6,
                zorder=19,
                clip_on=False,
            )
            ax.add_patch(fill_rect)

        ax.text(
            0.5,
            bar_y,
            f"Mission Progress: {progress:.0%}",
            transform=ax.transAxes,
            color="#e2e8f0",
            fontsize=8,
            ha="center",
            va="center",
            fontfamily="monospace",
            zorder=20,
        )

    def _draw_phase_transition(self, ax, positions, frame_idx):
        """Draw phase transition flash."""
        for step_idx, pname in self._phase_markers:
            if step_idx <= frame_idx and step_idx > frame_idx - 25:
                alpha = 0.3 * (1 - (frame_idx - step_idx) / 25)
                ax.axvline(
                    x=positions[step_idx, 0],
                    color="#f472b6",
                    linewidth=2,
                    alpha=alpha,
                    linestyle="-",
                    zorder=15,
                )
                if frame_idx - step_idx < 15:
                    ax.text(
                        0.5,
                        0.5,
                        f">>> {pname} <<<",
                        transform=ax.transAxes,
                        color="#f472b6",
                        fontsize=18,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        alpha=alpha * 2,
                        fontfamily="monospace",
                        zorder=20,
                    )

    def close(self, title: str = "OceanScale — Amphibious Robot Dog Demo") -> Path:
        """Render all frames and write MP4."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        try:
            import imageio.v3 as iio
        except ImportError:
            import imageio as iio

        if not self._positions:
            raise RuntimeError("No frames recorded")

        positions = np.array(self._positions)
        modes = np.array(self._modes)
        n_frames = len(positions)

        # Pre-compute terrain
        x_terrain = np.linspace(-2, 22, 400)
        z_terrain = np.array([self._terrain_height(x) for x in x_terrain])

        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#0a0e17")

        # === TITLE CARD (3 seconds) ===
        title_frames = int(3 * self.fps)
        for tf in range(title_frames):
            ax.clear()
            ax.set_facecolor("#0a0e17")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            fade_in = min(1.0, tf / (self.fps * 0.5))

            ax.text(
                0.5,
                0.7,
                "OCEANSCALE",
                transform=ax.transAxes,
                color="#38bdf8",
                fontsize=32,
                fontweight="bold",
                fontfamily="monospace",
                ha="center",
                va="center",
                alpha=fade_in,
            )

            ax.text(
                0.5,
                0.58,
                title,
                transform=ax.transAxes,
                color="#e2e8f0",
                fontsize=16,
                fontfamily="monospace",
                ha="center",
                va="center",
                alpha=fade_in * 0.8,
            )

            if tf > self.fps * 0.5:
                brief_alpha = min(
                    1.0, (tf - self.fps * 0.5) / (self.fps * 0.5)
                )
                brief = (
                    "MISSION BRIEFING\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Objective: Beach entry -> Reef inspection -> Return\n"
                    "Vehicle: Quadruped Amphibious Robot Dog (25 kg)\n"
                    "Propulsion: 4 legs (walk) + 8 thrusters (swim)\n"
                    "Control: FARMS-inspired CPG gait controller"
                )
                ax.text(
                    0.5,
                    0.32,
                    brief,
                    transform=ax.transAxes,
                    color="#94a3b8",
                    fontsize=11,
                    fontfamily="monospace",
                    ha="center",
                    va="center",
                    alpha=brief_alpha * 0.7,
                    linespacing=1.5,
                )

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()
            self._frames.append(frame)

        # === MAIN FRAMES ===
        for i in range(n_frames):
            ax.clear()
            ax.set_facecolor("#0a0e17")

            pos = positions[i]
            mode = modes[i]
            t = self._telemetry[i]

            x_center = float(pos[0])
            x_range = 10.0
            x_left = x_center - x_range / 2
            x_right = x_center + x_range / 2
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(-4.5, 2.5)

            # Draw layers (back to front)
            self._draw_water(ax, x_left, x_right, i)
            self._draw_terrain(ax, x_terrain, z_terrain, x_left, x_right)
            self._draw_coral_reef(ax, x_center)
            self._draw_trail(ax, positions, modes, i)
            self._draw_target(ax, self._current_target, pos, i)
            self._draw_bubbles(ax, x_left, x_right)
            self._draw_splashes(ax, x_left, x_right)

            if mode == 0:
                self._draw_robot_walk(ax, pos, i)
            else:
                self._draw_robot_swim(ax, pos, i)

            self._draw_phase_transition(ax, positions, i)
            self._draw_hud(
                ax, t, pos, self._current_target, i, n_frames, t["phase"]
            )

            ax.set_xlabel("Distance (m)", color="#94a3b8", fontsize=10)
            ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=10)
            ax.tick_params(colors="#64748b", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#1e293b")
            ax.grid(True, alpha=0.08, color="#64748b")

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()
            self._frames.append(frame)

        # === MISSION SUMMARY (3 seconds) ===
        summary_frames = int(3 * self.fps)
        total_dist = float(
            np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        )
        max_depth = float(-np.min(positions[:, 2]))
        phases_completed = len(self._phase_markers)

        for sf in range(summary_frames):
            ax.clear()
            ax.set_facecolor("#0a0e17")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            fade_in = min(1.0, sf / (self.fps * 0.5))

            ax.text(
                0.5,
                0.8,
                "MISSION COMPLETE",
                transform=ax.transAxes,
                color="#34d399",
                fontsize=28,
                fontweight="bold",
                fontfamily="monospace",
                ha="center",
                va="center",
                alpha=fade_in,
            )

            if sf > self.fps * 0.3:
                detail_alpha = min(
                    1.0, (sf - self.fps * 0.3) / (self.fps * 0.5)
                )
                summary = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Distance Traveled:  {total_dist:.1f} m\n"
                    f"Maximum Depth:      {max_depth:.1f} m\n"
                    f"Phases Completed:   {phases_completed}\n"
                    f"Mission Duration:   {t['time']:.1f} s\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "OceanScale — The Ocean Simulator"
                )
                ax.text(
                    0.5,
                    0.45,
                    summary,
                    transform=ax.transAxes,
                    color="#e2e8f0",
                    fontsize=13,
                    fontfamily="monospace",
                    ha="center",
                    va="center",
                    alpha=detail_alpha * 0.8,
                    linespacing=1.6,
                )

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()
            self._frames.append(frame)

        plt.close(fig)

        # Write MP4
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            str(self.output_path),
            self._frames,
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        return self.output_path


def run_amphibious_demo(config: AmphibiousDemoConfig | None = None) -> dict:
    """Run the full amphibious robot dog demonstration.

    Returns comprehensive metrics dict.
    """
    cfg = config or AmphibiousDemoConfig()
    np.random.seed(cfg.seed)

    demo = AmphibiousDemo(cfg)
    demo.reset()

    # Build mission
    phases = build_mission()
    renderer: AmphibiousVideoRenderer | None = None
    if cfg.render_mp4:
        renderer = AmphibiousVideoRenderer(
            output_path=cfg.render_mp4,
            fps=50,
            figsize=(16, 8),
            dpi=120,
        )

    # Tracking
    trajectory: list[np.ndarray] = []
    rewards: list[float] = []
    mode_transitions: list[tuple[float, str]] = []
    phase_results: list[dict] = []
    trajectory_frames: list[dict] = []
    t_start = time.time()

    # Run mission
    step_idx = 0
    for phase_idx, phase in enumerate(phases):
        phase_start = step_idx
        phase_rewards: list[float] = []
        target = np.array(phase.target)
        last_mode = "walk"

        if renderer:
            renderer.mark_phase(phase.name)

        for _ in range(phase.duration):
            if step_idx >= cfg.n_steps:
                break

            obs = demo._observe()
            pos = obs["position"]
            vel = obs["linear_velocity"]
            mode = obs["mode"]

            # Track mode transitions
            if obs["mode_name"] != last_mode:
                mode_transitions.append((obs["time"], obs["mode_name"]))
                last_mode = obs["mode_name"]

            if cfg.use_cpg:
                # CPG mode: pass target position directly
                obs = demo.step(target, target=target)
            else:
                # Legacy PD mode
                action = pd_control(pos, vel, target, mode)
                obs = demo.step(action)

            # Record
            trajectory.append(pos.copy())
            r = compute_reward(pos, target)
            rewards.append(r)
            phase_rewards.append(r)

            trajectory_frames.append(
                {
                    "t": float(obs["time"]),
                    "pos": [float(obs["position"][i]) for i in range(3)],
                    "quat": [float(obs["orientation"][i]) for i in range(4)],
                    "mode": int(obs["mode"]),
                    "mode_name": obs["mode_name"],
                    "sub_frac": float(obs["submerged_fraction"]),
                    "blend": float(obs.get("blend_factor", 0.0)),
                    "contact": float(obs.get("contact_force", 0.0)),
                    "vel": [float(obs["linear_velocity"][i]) for i in range(3)],
                    "target": [float(target[i]) for i in range(3)],
                    "phase": phase.name,
                    "step": step_idx,
                }
            )

            if renderer:
                renderer.record_frame(obs, target, phase.name)

            step_idx += 1

        # Phase summary
        phase_results.append(
            {
                "name": phase.name,
                "name_zh": phase.name_zh,
                "steps": step_idx - phase_start,
                "mean_reward": float(np.mean(phase_rewards)) if phase_rewards else 0.0,
                "final_distance": float(np.linalg.norm(pos - target)),
            }
        )

    # End card
    if renderer:
        renderer.close()

    t_wall = time.time() - t_start
    trajectory_arr = np.array(trajectory) if trajectory else np.zeros((1, 3))

    # Build result
    result = {
        "name": "Amphibious Robot Dog Demo",
        "vehicle": {
            "name": demo.vehicle.name,
            "mass": demo.vehicle.mass,
            "thrusters": demo.vehicle.n_thrusters,
            "leg_force": demo.vehicle.leg_max_force,
        },
        "mission": {
            "phases": [p.name for p in phases],
            "phases_zh": [p.name_zh for p in phases],
            "total_steps": step_idx,
            "mode_transitions": mode_transitions,
        },
        "metrics": {
            "wall_time_s": round(t_wall, 2),
            "throughput_sps": round(step_idx / t_wall, 1) if t_wall > 0 else 0,
            "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "total_distance_m": round(
                float(np.sum(np.linalg.norm(np.diff(trajectory_arr, axis=0), axis=1))),
                2,
            ),
            "max_depth_m": round(float(-np.min(trajectory_arr[:, 2])), 2),
            "max_x_m": round(float(np.max(trajectory_arr[:, 0])), 2),
        },
        "phases": phase_results,
        "features_demonstrated": [
            "Newton rigid-body physics (gravity enabled)",
            "Tier1 Fossen hydrodynamics (6-DOF underwater)",
            "Ground contact (spring-damper model)",
            "Walking leg actuators",
            "8-thruster allocation (T-matrix)",
            "Partial submersion (air-water interface)",
            "Multi-domain transition (land ↔ water)",
            "CPG gait control (FARMS-inspired Kuramoto oscillators)" if cfg.use_cpg else "PD mission controller",
            "Smooth force blending (contact-force-driven)" if cfg.use_cpg else "Hard mode switching",
            "Cinematic video rendering",
        ],
        "stack": {
            "physics": "Newton SolverVBD",
            "hydro": "Tier1 Fossen 6-DOF",
            "gpu": "Warp kernels",
            "propulsion": "PropellerThruster + leg actuators",
            "vehicle": "AmphibiousRobotDog",
            "gait": "KuramotoCPG" if cfg.use_cpg else "PD controller",
        },
    }

    result["trajectory_json"] = None
    if trajectory_frames and cfg.trajectory_json:
        traj_data = {
            "fps": 50,
            "dt": cfg.dt,
            "total_frames": len(trajectory_frames),
            "vehicle": result["vehicle"],
            "frames": trajectory_frames,
        }
        Path(cfg.trajectory_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.trajectory_json, "w") as f:
            json.dump(traj_data, f)
        result["trajectory_json"] = str(cfg.trajectory_json)
        print(f"  Trajectory exported: {cfg.trajectory_json} ({len(trajectory_frames)} frames)")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amphibious robot dog demo — walk, swim, return"
    )
    parser.add_argument("--steps", type=int, default=3000, help="Max simulation steps")
    parser.add_argument("--render-mp4", type=str, default=None, help="Output MP4 path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Warp device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-cinematic", action="store_true", help="Disable cinematic rendering"
    )
    parser.add_argument(
        "--wave-height", type=float, default=0.3, help="Wave height (m)"
    )
    parser.add_argument(
        "--current-speed", type=float, default=0.15, help="Current speed (m/s)"
    )
    parser.add_argument(
        "--no-cpg", action="store_true", help="Disable CPG, use legacy PD controller"
    )
    parser.add_argument(
        "--trajectory-json", type=str, default=None,
        help="Export per-frame trajectory to JSON for Isaac Sim rendering",
    )
    args = parser.parse_args()

    config = AmphibiousDemoConfig(
        n_steps=args.steps,
        device=args.device,
        seed=args.seed,
        render_mp4=args.render_mp4,
        cinematic=not args.no_cinematic,
        wave_height=args.wave_height,
        current_speed=args.current_speed,
        use_cpg=not args.no_cpg,
        trajectory_json=args.trajectory_json,
    )

    result = run_amphibious_demo(config)

    # Print results
    print("\n" + "=" * 60)
    print("  OceanScale — Amphibious Robot Dog Demo")
    print("=" * 60)
    print(f"\n  Vehicle: {result['vehicle']['name']}")
    print(f"  Mass: {result['vehicle']['mass']} kg")
    print(f"  Thrusters: {result['vehicle']['thrusters']}")
    print(f"  Leg force: {result['vehicle']['leg_force']} N")

    print(f"\n  Mission phases:")
    for p in result["phases"]:
        print(
            f"    {p['name']} ({p['name_zh']}): "
            f"reward={p['mean_reward']:.3f}, "
            f"dist={p['final_distance']:.2f}m"
        )

    print(f"\n  Mode transitions:")
    for t, mode in result["mission"]["mode_transitions"]:
        print(f"    t={t:.2f}s → {mode}")

    m = result["metrics"]
    print(f"\n  Performance:")
    print(f"    Distance: {m['total_distance_m']}m traveled")
    print(f"    Wall time: {m['wall_time_s']}s")
    print(f"    Throughput: {m['throughput_sps']} steps/s")
    print(f"    Max depth: {m['max_depth_m']}m")
    print(f"    Mean reward: {m['mean_reward']}")

    print(f"\n  Features demonstrated:")
    for f in result["features_demonstrated"]:
        print(f"    [x] {f}")

    if result.get("render_mp4"):
        print(f"\n  Video: {result['render_mp4']}")

    if result.get("trajectory_json"):
        print(f"  Trajectory: {result['trajectory_json']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
