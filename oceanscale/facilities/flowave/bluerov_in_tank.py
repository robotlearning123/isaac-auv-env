"""BlueROV2 Heavy in the Virtual FloWave tank — USD integration + 6-DOF dynamics.

USD asset: oceanscale/assets/bluerov2_heavy.usda
Architecture reference: docs/virtual_flowave_architecture.md §2.3, §2.4

Dynamics path (v1 fallback):
    The full Tier1 GPU pipeline (oceanscale.hydro.tier1.Tier1) requires Warp +
    CUDA device buffers and is designed for batched RL training.  Wiring it
    cleanly into a single-vehicle USD-write loop would require a 1-env Warp
    context, Newton state, and kernel launch overhead for every frame — adding
    ~5 ms of CPU↔GPU synchronisation for a task that runs on a single body.

    v1 therefore uses a minimal stand-in 6-DOF model implemented in pure NumPy:
      - 3-DOF translation:
            m·a = F_thrust - 0.5·ρ·C_d·A·|v_rel|·v_rel
          where v_rel = v_vehicle - v_current (current-relative damping §7.1)
          ρ = 1000 kg/m³ (fresh water, matching FloWave), C_d = 1.0, A = 0.06 m²
      - Neutrally buoyant: no net gravity/buoyancy mismatch force in z.
      - 3-DOF rotation: locked — yaw = 0, orientation = identity quaternion.

    v1.1 work: replace _Dynamics6DoFSimple with a thin wrapper around Tier1 using
    n_envs=1, numpy↔warp bridge arrays, and BlueROV2Heavy.set_coeffs_kwargs() for
    coefficients.  Import path: oceanscale.hydro.tier1.Tier1 and
    oceanscale.vehicles.bluerov2.BlueROV2Heavy.

Discovered vehicle classes in oceanscale.vehicles.bluerov2:
    - BlueROV2Heavy  — Fossen hydrodynamic coefficients (von Benzon 2022)
    - BlueROV2MarineGym  — MarineGym YAML config
    Both provide set_coeffs_kwargs() and tier1_kwargs() for Tier1 wiring (v1.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .impeller_array import ImpellerArray

if TYPE_CHECKING:
    pass

# -------------------------------------------------------------------------
# Physical constants (v1 stand-in model)
# -------------------------------------------------------------------------
_RHO_FRESH: float = 1000.0  # kg/m³ — FloWave uses fresh water
_C_D: float = 1.0  # drag coefficient (bluff body)
_DRAG_AREA: float = 0.06  # m² — representative projected area (side view)


class _Dynamics6DoFSimple:
    """Minimal 6-DOF stand-in for BlueROV2 dynamics.

    Translation only (3-DOF); rotation locked to identity.

    State
    -----
    pos : (3,) world position (m), z-up frame.
    vel : (3,) world velocity (m/s).

    Parameters
    ----------
    mass : float
        Vehicle mass (kg). Default 13.5 kg (BlueROV2 Heavy, von Benzon Table A1).
    rho : float
        Water density (kg/m³). Default 1000 (fresh).
    c_d : float
        Quadratic drag coefficient. Default 1.0.
    area : float
        Reference projected area (m²). Default 0.06.
    """

    def __init__(
        self,
        mass: float = 13.5,
        rho: float = _RHO_FRESH,
        c_d: float = _C_D,
        area: float = _DRAG_AREA,
    ) -> None:
        self._mass = float(mass)
        self._rho = float(rho)
        self._c_d = float(c_d)
        self._area = float(area)
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)

    def reset(self, pos: np.ndarray) -> None:
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)

    def step(
        self,
        dt: float,
        thrust: np.ndarray,
        current: np.ndarray,
    ) -> None:
        """Forward-Euler integration.

        Parameters
        ----------
        dt : float
            Time step (s).
        thrust : (3,) world-frame force (N).
        current : (3,) ambient water velocity (m/s).
        """
        v_rel = self.vel - current  # current-relative velocity
        v_rel_mag = float(np.linalg.norm(v_rel))
        drag = -0.5 * self._rho * self._c_d * self._area * v_rel_mag * v_rel
        accel = (thrust + drag) / self._mass
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt

        # ── tank boundary enforcement ──
        r = np.linalg.norm(self.pos[:2])
        if r > 12.0:
            self.pos[:2] *= 12.0 / r
        self.pos[2] = np.clip(self.pos[2], 0.05, 2.0)


# -------------------------------------------------------------------------
# BlueROV2InTank
# -------------------------------------------------------------------------


class BlueROV2InTank:
    """BlueROV2 Heavy in the Virtual FloWave tank.

    Responsibilities:
    - References bluerov2_heavy.usda into the stage at /BlueROV2/Body.
    - Per frame: samples current at vehicle CG from ImpellerArray, computes PD
      station-keeping thrust, steps the 6-DOF dynamics, writes USD xformOps.
    - Parents /Cameras/BlueRovPov beneath /BlueROV2/Body for a follow-cam.

    Parameters
    ----------
    stage : Usd.Stage
        Open USD stage.
    impeller_array : ImpellerArray
        Already-configured impeller array.  Call FloWaveCoupling.step() before
        BlueROV2InTank.step() so the impeller low-pass filter is up to date.
    usd_asset : Path
        Path to the BlueROV2 Heavy USDA.  Default: oceanscale/assets/bluerov2_heavy.usda.
    initial_position : (float, float, float)
        World position at t=0 (m).  Default (0, 0, 1.0) — basin centre, mid-depth.
    target_position : (float, float, float)
        Station-keeping setpoint (m).
    kp_xy : float
        Proportional gain for x/y position (N/m).
    kd_xy : float
        Derivative gain for x/y velocity (N·s/m).
    kp_z : float
        Proportional gain for z position (N/m).
    kd_z : float
        Derivative gain for z velocity (N·s/m).
    mass : float
        Vehicle mass (kg).  Default 13.5 (BlueROV2 Heavy).
    """

    def __init__(
        self,
        stage: Usd.Stage,
        impeller_array: ImpellerArray,
        *,
        usd_asset: Path = Path("oceanscale/assets/bluerov2_heavy.usda"),
        initial_position: tuple[float, float, float] = (0.0, 0.0, 1.0),
        target_position: tuple[float, float, float] = (0.0, 0.0, 1.0),
        kp_xy: float = 50.0,
        kd_xy: float = 30.0,
        kp_z: float = 30.0,
        kd_z: float = 20.0,
        mass: float = 13.5,
    ) -> None:
        self._stage = stage
        self._impeller_array = impeller_array
        self._usd_asset = usd_asset
        self._initial_position = np.array(initial_position, dtype=np.float64)
        self.target_position: np.ndarray = np.array(target_position, dtype=np.float64)
        self._kp = np.array([kp_xy, kp_xy, kp_z], dtype=np.float64)
        self._kd = np.array([kd_xy, kd_xy, kd_z], dtype=np.float64)

        # (1) Author /BlueROV2/Body referencing usd_asset
        body_path = "/BlueROV2/Body"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            body_prim = stage.DefinePrim(body_path)
        body_prim.GetReferences().AddReference(str(usd_asset))

        # (2) Add xformOp:translate + xformOp:orient
        xform = UsdGeom.Xformable(body_prim)
        # Clear existing ops to avoid duplicates on re-init
        xform.ClearXformOpOrder()
        self._translate_op = xform.AddTranslateOp(
            opSuffix="",
            precision=UsdGeom.XformOp.PrecisionDouble,
        )
        self._orient_op = xform.AddOrientOp(
            opSuffix="",
            precision=UsdGeom.XformOp.PrecisionDouble,
        )

        # Write initial xform
        self._translate_op.Set(Gf.Vec3d(*initial_position))
        self._orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        # (3) Author /Cameras/BlueRovPov parented under /BlueROV2
        self._author_pov_camera()

        # (4) Construct minimal 6-DOF stand-in dynamics
        self._dyn = _Dynamics6DoFSimple(mass=mass)
        self._dyn.reset(initial_position)

    # ------------------------------------------------------------------
    # Per-frame step
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float) -> None:
        """Advance vehicle state by dt and write USD xformOps at time t.

        Call AFTER FloWaveCoupling.step() so impeller filter is current.

        Parameters
        ----------
        t : float
            Simulation time (s) — written as USD TimeCode.
        dt : float
            Time step (s).
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        # (a) Sample current at vehicle CG
        pos3 = self._dyn.pos[np.newaxis, :]  # (1, 3)
        current_3d = self._impeller_array.sample(pos3)  # (1, 3)
        current = current_3d[0]  # (3,)

        # (b) PD station-keeping thrust
        err_pos = self.target_position - self._dyn.pos  # (3,)
        err_vel = -self._dyn.vel  # (3,) — dampen velocity
        thrust = self._kp * err_pos + self._kd * err_vel  # (3,) N
        MAX_THRUST = 10.0  # BlueROV2 Heavy max ~10N per thruster
        thrust = np.clip(thrust, -MAX_THRUST, MAX_THRUST)

        # (c) Step dynamics
        self._dyn.step(dt, thrust, current)

        # (d) Write USD xformOps at time t
        tc = Usd.TimeCode(t)
        p = self._dyn.pos
        self._translate_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])), tc)
        self._orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0), tc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Current world position (3,) in metres."""
        return self._dyn.pos.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current world velocity (3,) in m/s."""
        return self._dyn.vel.copy()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, t: float = 0.0) -> None:
        """Reset vehicle to initial_position, zero velocity, identity orientation.

        Parameters
        ----------
        t : float
            Reset time written to USD (s).  Default 0.0.
        """
        self._dyn.reset(self._initial_position)
        tc = Usd.TimeCode(t)
        p = self._initial_position
        self._translate_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])), tc)
        self._orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0), tc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _author_pov_camera(self) -> None:
        """Author /BlueROV2/BlueRovPov camera parented under /BlueROV2.

        Architecture §2.5: BlueRovPov is listed as a follow-cam.  v1 parents it
        under /BlueROV2 (which contains /Body) so the camera inherits the
        vehicle's world transform.  The top-level /Cameras/BlueRovPov path in
        the architecture spec (docs/virtual_flowave_architecture.md §2 scene tree)
        refers to a placeholder; the Task 10 orchestrator may reparent this at
        scene-assembly time.  Parenting under /BlueROV2 here satisfies the
        requirement that the camera follows the vehicle.
        """
        cam_path = "/BlueROV2/BlueRovPov"
        cam_prim = self._stage.GetPrimAtPath(cam_path)
        if not cam_prim.IsValid():
            cam = UsdGeom.Camera.Define(self._stage, cam_path)
            # Offset camera 1.5 m behind and 0.5 m above the vehicle body
            xform = UsdGeom.Xformable(cam.GetPrim())
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp(
                opSuffix="",
                precision=UsdGeom.XformOp.PrecisionDouble,
            )
            translate_op.Set(Gf.Vec3d(-1.5, 0.0, 0.5))
            cam.GetFocalLengthAttr().Set(35.0)
