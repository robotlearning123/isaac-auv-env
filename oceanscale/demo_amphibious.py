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

    Renders a side-view (x-z plane) with:
    - Terrain profile (beach → slope → seabed)
    - Water surface line
    - Robot trail with mode-colored segments
    - Phase label and telemetry HUD
    - Phase transition markers
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 50,
        figsize: tuple[float, float] = (16, 8),
        dpi: int = 120,
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi
        self._frames: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._modes: list[int] = []
        self._telemetry: list[dict] = []
        self._phase_markers: list[tuple[int, str]] = []
        self._current_target: np.ndarray = np.zeros(3)

    def record_frame(self, obs: dict, target: np.ndarray, phase_name: str) -> None:
        """Record one frame."""
        self._current_target = target.copy()
        self._positions.append(obs["position"].copy())
        self._modes.append(obs["mode"])
        self._telemetry.append(
            {
                "depth": -obs["position"][2],
                "sub_frac": obs["submerged_fraction"],
                "time": obs["time"],
                "mode": obs["mode_name"],
            }
        )

    def mark_phase(self, phase_name: str) -> None:
        """Mark a phase transition."""
        self._phase_markers.append((len(self._positions), phase_name))

    def close(self, title: str = "OceanScale — Amphibious Robot Dog Demo") -> Path:
        """Render all frames and write MP4."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Polygon

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
        x_terrain = np.linspace(-1, 20, 200)
        z_terrain = np.array(
            [
                (
                    0.5
                    if x <= 2.0
                    else (0.5 - 0.25 * (x - 2.0) if x <= 6.0 else -0.5)
                )
                for x in x_terrain
            ]
        )

        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#0a0e17")
        ax.set_facecolor("#0a0e17")

        # Render frames
        for i in range(n_frames):
            ax.clear()
            ax.set_facecolor("#0a0e17")

            pos = positions[i]
            mode = modes[i]
            t = self._telemetry[i]

            # Viewport follows robot
            x_center = pos[0]
            x_range = 10.0
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(-4.5, 2.5)

            # Draw terrain
            mask = (x_terrain >= x_center - x_range / 2 - 1) & (
                x_terrain <= x_center + x_range / 2 + 1
            )
            ax.fill_between(
                x_terrain[mask],
                z_terrain[mask],
                -5.0,
                color="#1a2332",
                alpha=0.8,
            )
            ax.plot(
                x_terrain[mask],
                z_terrain[mask],
                color="#2d4a5e",
                linewidth=2,
            )

            # Water surface
            water_left = x_center - x_range / 2 - 1
            water_right = x_center + x_range / 2 + 1
            ax.axhline(
                y=0.0,
                color="#38bdf8",
                linewidth=2,
                alpha=0.6,
                linestyle="--",
            )
            # Water fill
            ax.fill_between(
                [water_left, water_right],
                [0.0, 0.0],
                [-5.0, -5.0],
                color="#0c4a6e",
                alpha=0.3,
            )

            # Trail
            if i > 1:
                trail_x = positions[: i + 1, 0]
                trail_z = positions[: i + 1, 2]
                trail_modes = modes[: i + 1]
                for j in range(1, len(trail_x)):
                    color = "#34d399" if trail_modes[j] == 0 else "#38bdf8"
                    ax.plot(
                        trail_x[j - 1 : j + 1],
                        trail_z[j - 1 : j + 1],
                        color=color,
                        linewidth=2,
                        alpha=0.5,
                    )

            # Robot body
            robot_color = "#f59e0b" if mode == 0 else "#38bdf8"
            body_w, body_h = 0.4, 0.15
            rect = plt.Rectangle(
                (pos[0] - body_w / 2, pos[1] - body_h / 2),
                body_w,
                body_h,
                angle=0,
                color=robot_color,
                alpha=0.9,
                zorder=10,
            )
            ax.add_patch(rect)

            # Legs (walking mode)
            if mode == 0:
                for lx in [-0.12, 0.12]:
                    ax.plot(
                        [pos[0] + lx, pos[0] + lx],
                        [pos[1] - body_h / 2, pos[1] - body_h / 2 - 0.12],
                        color="#f59e0b",
                        linewidth=3,
                        alpha=0.7,
                    )

            # Propellers (swimming mode)
            if mode == 1:
                for px in [-0.15, 0.15]:
                    ax.plot(
                        pos[0] + px,
                        pos[1],
                        marker="o",
                        markersize=6,
                        color="#38bdf8",
                        alpha=0.8,
                    )

            # Target marker
            ax.plot(
                self._current_target[0],
                self._current_target[2],
                marker="x",
                markersize=10,
                color="#f472b6",
                markeredgewidth=2,
            )

            # Phase transition markers
            for step_idx, pname in self._phase_markers:
                if step_idx <= i and step_idx > i - 50:
                    ax.axvline(
                        x=positions[step_idx, 0],
                        color="#f472b6",
                        linewidth=1,
                        alpha=0.5,
                        linestyle=":",
                    )

            # HUD
            mode_str = "WALKING" if mode == 0 else "SWIMMING"
            mode_color = "#f59e0b" if mode == 0 else "#38bdf8"
            hud_text = (
                f"OceanScale — Amphibious Robot Dog\n"
                f"Phase: {mode_str}  |  "
                f"Depth: {t['depth']:.2f}m  |  "
                f"Submerged: {t['sub_frac']:.0%}  |  "
                f"Time: {t['time']:.1f}s"
            )
            ax.text(
                0.02,
                0.98,
                hud_text,
                transform=ax.transAxes,
                color="#e2e8f0",
                fontsize=10,
                fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#111827",
                    edgecolor="#1e293b",
                    alpha=0.9,
                ),
            )

            # Mode indicator
            ax.text(
                0.98,
                0.98,
                mode_str,
                transform=ax.transAxes,
                color=mode_color,
                fontsize=14,
                fontweight="bold",
                fontfamily="monospace",
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#111827",
                    edgecolor=mode_color,
                    alpha=0.9,
                ),
            )

            # Labels
            ax.set_xlabel("Distance (m)", color="#94a3b8", fontsize=10)
            ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=10)
            ax.tick_params(colors="#64748b", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#1e293b")

            # Grid
            ax.grid(True, alpha=0.1, color="#64748b")

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()
            self._frames.append(frame)

        # Title card (2 seconds)
        title_frames = int(2 * self.fps)
        for _ in range(title_frames):
            ax.clear()
            ax.set_facecolor("#0a0e17")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(
                0.5,
                0.55,
                title,
                transform=ax.transAxes,
                color="#38bdf8",
                fontsize=20,
                fontweight="bold",
                fontfamily="monospace",
                ha="center",
                va="center",
            )
            ax.text(
                0.5,
                0.4,
                "Walk → Swim → Return\nMulti-Domain Ocean Simulation",
                transform=ax.transAxes,
                color="#94a3b8",
                fontsize=14,
                fontfamily="monospace",
                ha="center",
                va="center",
            )
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()
            self._frames.insert(0, frame)

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

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
