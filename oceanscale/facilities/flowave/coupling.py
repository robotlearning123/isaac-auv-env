"""Per-frame physics↔render integration for the Virtual FloWave digital twin.

Implements the five-arrow coupling matrix from architecture §2.5:

    A0 — wave solver s_n(t) → 168 paddle hinge Xform rotations
    A1 — wave solver η(x,y,t) → Water/Surface UsdGeomMesh z values
    A2 — current solver v(x,y,z,t) → Marine-snow UsdGeomPointInstancer positions
    A3 — turbidity → Water/Body MDL binding (static; authored at scene-assembly by Task 8)
    A4 — caustics emerge from PathTracing + A1 mesh + IOR=1.333 (no per-frame code)
    A5 — thruster bubbles (deferred to v1.1)

Architecture reference: docs/virtual_flowave_architecture.md §2.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from .impeller_array import ImpellerArray
from .impeller_usd import ImpellerRingUsd
from .paddle_array import FlapPaddleArray
from .paddle_usd import PaddleRingUsd

if TYPE_CHECKING:
    from .water_usd import WaterSurfaceUsd
    from .wave_probe import WaveProbe

_TANK_RADIUS: float = 12.5  # m — working tank radius (architecture §1.1)
_WATER_DEPTH: float = 2.0   # m — SWL above raisable floor (architecture §1.1)
_SWL: float = _WATER_DEPTH  # still-water level in z-up frame


@dataclass
class CouplingConfig:
    """Configuration for FloWaveCoupling.

    Parameters
    ----------
    spectrum_func : Callable[[float, float], float]
        S(ω, θ) → m²·s/rad. Directional wave energy spectrum.
    n_freqs : int
        Number of Miles-Funke frequency bins J. Default 128.
    omega_range : tuple[float, float]
        (ω_min, ω_max) in rad/s. Default (0.3, 8.0) matching FloWave range.
    impeller_target : tuple[float, float]
        (u, v) m/s bulk current setpoint at mid-depth. Default (0.0, 0.0).
    marine_snow_count : int
        Number of marine-snow particles. Default 50 000.
    marine_snow_radius : float
        Prototype sphere radius (m). Default 0.003 m (3 mm flake).
    spectrum_seed : int
        RNG seed passed to FlapPaddleArray.synthesize_irregular. Default 0.
    rng_seed : int
        RNG seed for particle initialisation. Default 1.
    """

    spectrum_func: Callable[[float, float], float]
    n_freqs: int = 128
    omega_range: tuple[float, float] = (0.3, 8.0)
    impeller_target: tuple[float, float] = (0.0, 0.0)
    marine_snow_count: int = 50_000
    marine_snow_radius: float = 0.003
    spectrum_seed: int = 0
    rng_seed: int = 1


class FloWaveCoupling:
    """Per-frame physics↔render integration loop for the Virtual FloWave.

    Drives arrows A0 (paddle hinge), A1 (water surface mesh), and A2 (marine
    snow particles) on every call to step(). A3 is static and requires no
    per-frame work. A4 emerges from PathTracing. A5 is deferred.

    Usage
    -----
    >>> coupling = FloWaveCoupling(stage, paddle_array, impeller_array,
    ...                            paddle_ring, impeller_ring, water_surface, cfg)
    >>> for t in np.arange(0, 30.0, 1/24):
    ...     coupling.step(t, dt=1/24)

    Parameters
    ----------
    stage : Usd.Stage
        Open USD stage that contains /Tank/MarineSnow (authored here if absent).
    paddle_array : FlapPaddleArray
        Biésel + Snake wave solver.
    impeller_array : ImpellerArray
        Parametric current-field model.
    paddle_ring : PaddleRingUsd
        USD writer for 168 paddle hinge angles (A0).
    impeller_ring : ImpellerRingUsd
        USD writer for 28 impeller blade angles (passive — rotates proportional
        to the filtered current speed; cosmetic only in v1).
    water_surface : WaterSurfaceUsd
        USD writer for the 256×256 water surface mesh (A1).
    cfg : CouplingConfig
        Coupling parameters.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        paddle_array: FlapPaddleArray,
        impeller_array: ImpellerArray,
        paddle_ring: PaddleRingUsd,
        impeller_ring: ImpellerRingUsd,
        water_surface: "WaterSurfaceUsd",
        cfg: CouplingConfig,
        probe: "WaveProbe | None" = None,
    ) -> None:
        self._stage = stage
        self._paddle_array = paddle_array
        self._impeller_array = impeller_array
        self._paddle_ring = paddle_ring
        self._impeller_ring = impeller_ring
        self._water_surface = water_surface
        self._cfg = cfg
        self._probe = probe

        # --- Build wave synth ONCE (shared ε_j, ω_j, k_j across A0 + A1) ---
        synth = paddle_array.synthesize_irregular(
            spectrum_func=cfg.spectrum_func,
            n_freqs=cfg.n_freqs,
            omega_range=cfg.omega_range,
            seed=cfg.spectrum_seed,
        )
        self._paddle_commands = synth.paddle_commands  # t → (N,)
        self._eta_field = synth.eta_field               # (xy, t) → (M,)

        # --- Set impeller current target ---
        u_tgt, v_tgt = cfg.impeller_target
        impeller_array.set_target(u_tgt, v_tgt)

        # --- A2: initialise marine-snow particle positions ---
        self._n_particles: int = cfg.marine_snow_count
        rng = np.random.default_rng(cfg.rng_seed)
        self._particles: np.ndarray = _init_particles(
            rng, self._n_particles, _TANK_RADIUS, _WATER_DEPTH
        )

        # --- Author /Tank/MarineSnow PointInstancer if absent ---
        snow_path = "/Tank/MarineSnow"
        prim = stage.GetPrimAtPath(snow_path)
        if not prim.IsValid():
            inst = UsdGeom.PointInstancer.Define(stage, snow_path)
            proto_path = snow_path + "/Proto"
            proto = UsdGeom.Sphere.Define(stage, proto_path)
            proto.GetRadiusAttr().Set(cfg.marine_snow_radius)
            inst.CreatePrototypesRel().AddTarget(proto_path)
            inst.CreateProtoIndicesAttr().Set([0] * self._n_particles)
        self._snow_instancer: UsdGeom.PointInstancer = UsdGeom.PointInstancer(
            stage.GetPrimAtPath(snow_path)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float) -> None:
        """Advance simulation by dt seconds and update all coupled USD attributes.

        Implements arrows A0, A1, A2. A3 is static (no per-frame work). A4
        emerges automatically. A5 is deferred to v1.1.

        Parameters
        ----------
        t : float
            Current simulation time (s).
        dt : float
            Time step (s).
        """
        # --- A0: paddle hinge angles ---
        self._step_a0(t)

        # --- A1: water surface mesh ---
        self._step_a1(t)

        # --- Wave probe sampling (optional; shares the A1 eta_field) ---
        if self._probe is not None:
            self._probe.record(self._eta_field, t)

        # --- A2: marine snow advection ---
        self._step_a2(t, dt)

        # --- A3: turbidity (static; authored at scene assembly by Task 8) ---
        # No per-frame work.  Validation hook: /Tank/Water/Body material binding
        # must not be modified here.  See architecture §2.5 A3.

        # --- A4: caustics emerge from PathTracing (no code here) ---

        # TODO: A5 — bubble emitter at BlueROV2 thrusters

    def reset(self, t: float = 0.0) -> None:
        """Reset coupling state.

        Re-initialises marine-snow positions, clears the impeller low-pass
        filter, and re-applies the target setpoint.

        Parameters
        ----------
        t : float
            Reset time (written to USD attributes). Default 0.0.
        """
        # Reset impeller filter
        rng = np.random.default_rng(self._cfg.rng_seed)
        self._particles = _init_particles(
            rng, self._n_particles, _TANK_RADIUS, _WATER_DEPTH
        )

        # Reset impeller low-pass state by re-constructing it from scratch
        self._impeller_array._filtered_u = 0.0
        self._impeller_array._filtered_v = 0.0
        u_tgt, v_tgt = self._cfg.impeller_target
        self._impeller_array.set_target(u_tgt, v_tgt)

        # Sync USD to t=0 state
        self._step_a0(t)
        self._step_a1(t)
        self._step_a2(t, dt=0.0)

    @property
    def particle_positions(self) -> np.ndarray:
        """Current (N, 3) particle positions; for testing."""
        return self._particles.copy()

    # ------------------------------------------------------------------
    # Private per-arrow helpers
    # ------------------------------------------------------------------

    def _step_a0(self, t: float) -> None:
        """A0 — wave solver s_n(t) → 168 paddle hinge Xform rotations.

        s_n(t) = surface displacement (m); θ_n = s_n / hinge_depth (small-angle).
        Architecture §2.5 A0.
        """
        s_n = self._paddle_commands(t)  # (168,) surface displacement (m)
        theta_n = s_n / self._paddle_array.hinge_depth  # (168,) radians, small-angle
        self._paddle_ring.set_hinge_angles(theta_n, time=t)

    def _step_a1(self, t: float) -> None:
        """A1 — wave solver η(x,y,t) → Water/Surface mesh z values.

        z = SWL + η(x,y,t) = 2.0 + η.
        Architecture §2.5 A1.

        TODO: Warp kernel for the η evaluation — replace numpy mat-mul with
              wp.launch(eta_kernel, ...) for GPU-parallel field evaluation.
        """
        xy = self._water_surface.xy_grid  # (M, 2) world (x, y) of surface vertices
        eta = self._eta_field(xy, t)       # (M,)
        z = _SWL + eta                     # (M,) — z-up, SWL = 2.0 m
        self._water_surface.set_z_values(z, time=t)

    def _step_a2(self, t: float, dt: float) -> None:
        """A2 — current field v(x,y,z,t) → marine snow PointInstancer positions.

        Forward-Euler advection with cylinder wrap-around topology.
        Architecture §2.5 A2.

        TODO: Warp kernel — replace numpy advection with wp.launch(advect_snow_kernel,
              ...) for GPU-parallel particle update + volume sample via
              wp.volume_sample_v (architecture §2.3 NanoVDB field — v2).
        """
        # Step impeller low-pass filter
        self._impeller_array.step(dt)

        # Sample velocity at current particle positions
        vel = self._impeller_array.sample(self._particles)  # (N, 3) m/s

        # Forward-Euler advect
        self._particles = self._particles + vel * dt

        # Wrap particles back inside the cylinder (toroidal-cylinder topology)
        self._particles = _wrap_cylinder(self._particles, _TANK_RADIUS, _WATER_DEPTH)

        # Write to USD PointInstancer
        positions_vt = Vt.Vec3fArray(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in self._particles]
        )
        tc = Usd.TimeCode(t)
        self._snow_instancer.GetPositionsAttr().Set(positions_vt, tc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _init_particles(
    rng: np.random.Generator,
    n: int,
    tank_radius: float,
    water_depth: float,
) -> np.ndarray:
    """Initialise N particle positions uniformly inside the working cylinder.

    Particles are placed in r ≤ tank_radius, z ∈ [0.2, water_depth - 0.2],
    keeping them away from the floor and free surface.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    n : int
        Number of particles.
    tank_radius : float
        Cylinder radius (m).
    water_depth : float
        Water column height (m).

    Returns
    -------
    np.ndarray
        Shape (N, 3) float64.
    """
    # Uniform sampling inside a disk via rejection sampling on a bounding square
    positions = np.empty((n, 3), dtype=np.float64)
    filled = 0
    while filled < n:
        batch = max(n - filled, 512)
        xy = rng.uniform(-tank_radius, tank_radius, size=(batch * 2, 2))
        r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
        inside = xy[r2 <= tank_radius**2][:batch]
        take = min(len(inside), n - filled)
        positions[filled : filled + take, :2] = inside[:take]
        filled += take

    # z uniformly in [0.2, water_depth - 0.2]
    z_min = 0.2
    z_max = water_depth - 0.2
    positions[:, 2] = rng.uniform(z_min, z_max, size=n)
    return positions


def _wrap_cylinder(
    positions: np.ndarray,
    tank_radius: float,
    water_depth: float,
) -> np.ndarray:
    """Wrap-around topology: particles outside the cylinder re-enter from the opposite side.

    For horizontal escape (r > tank_radius): reflect through origin (x,y → -x,-y)
    and add a small random angular offset to break symmetry.
    For vertical escape (z < floor or z > SWL): reflect about the violated boundary.

    This keeps particle count strictly constant while preventing accumulation
    at the walls.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3).
    tank_radius : float
        Cylinder radius (m).
    water_depth : float
        SWL height (m); also the upper z boundary.

    Returns
    -------
    np.ndarray
        Shape (N, 3), all particles inside the cylinder.
    """
    # TODO: Warp kernel — wrap_cylinder_kernel(pos[:], R, h) on GPU.
    pos = positions.copy()

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.sqrt(x**2 + y**2)

    # Horizontal: re-inject at origin's diametrically opposite point
    outside_h = r > tank_radius
    if np.any(outside_h):
        # Scale back to just inside the wall on the opposite side
        scale = (tank_radius * 0.95) / np.maximum(r[outside_h], 1e-9)
        pos[outside_h, 0] = -x[outside_h] * scale
        pos[outside_h, 1] = -y[outside_h] * scale

    # Vertical: clamp with a small inset so particles stay in working volume
    z_min, z_max = 0.05, water_depth
    pos[:, 2] = np.clip(pos[:, 2], z_min, z_max)

    return pos
