"""FloWave impeller ring USD coupling helper.

Provides ImpellerRingUsd: per-frame blade angle writer for the 28-impeller
floor ring (flowave_impeller_ring.usda).

Blade rotation is xformOp:rotateZ (not Y) because the vertical axis is Z
in our right-handed z-up frame. The rotateZ op is named "xformOp:rotateZ_blade"
on each Drive_NN prim so it does not conflict with any world-level rotation.

Architecture reference: docs/virtual_flowave_architecture.md §3.1.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Usd, UsdGeom

_N_IMPELLERS = 28


class ImpellerRingUsd:
    """Blade-angle writer for the 28-impeller FloWave ring in a USD stage.

    Parameters
    ----------
    stage:
        An open Usd.Stage containing the impeller ring prims.
    ring_path:
        USD path to the parent Xform that owns Drive_00 … Drive_27.
        Default: "/Tank/Impeller" (matches flowave_impeller_ring.usda).
    """

    def __init__(
        self,
        stage: Usd.Stage,
        ring_path: str = "/Tank/Impeller",
    ) -> None:
        self._stage = stage
        self._ring_path = ring_path

        # Resolve and cache the 28 rotateZ ops
        self._rotate_ops: list[UsdGeom.XformOp] = []
        for k in range(_N_IMPELLERS):
            drive_path = f"{ring_path}/Drive_{k:02d}"
            prim = stage.GetPrimAtPath(drive_path)
            if not prim.IsValid():
                raise ValueError(
                    f"Prim not found: {drive_path}. "
                    "Run scripts/build_flowave_impeller_ring.py first."
                )
            xformable = UsdGeom.Xformable(prim)
            # Find the rotateZ_blade op by name
            op = None
            for candidate in xformable.GetOrderedXformOps():
                if candidate.GetOpName() == "xformOp:rotateZ_blade":
                    op = candidate
                    break
            if op is None:
                raise ValueError(
                    f"xformOp:rotateZ_blade not found on {drive_path}. "
                    "USD prim may be out of date — re-run build script."
                )
            self._rotate_ops.append(op)

        # Internal state: current angle per impeller (radians), for step_with_rpm
        self._angles_rad: np.ndarray = np.zeros(_N_IMPELLERS, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def impeller_paths(self) -> list[str]:
        """Return the 28 Drive prim paths in index order."""
        return [f"{self._ring_path}/Drive_{k:02d}" for k in range(_N_IMPELLERS)]

    def set_blade_angles(
        self,
        angles_rad: np.ndarray,
        time: float | None = None,
    ) -> None:
        """Write blade rotation angle about Z (radians) to each of 28 impellers.

        The value is converted to degrees before writing to the USD
        xformOp:rotateZ_blade attribute (USD stores degrees).

        Parameters
        ----------
        angles_rad:
            Shape (28,) array of blade angles in radians.
        time:
            USD time code. Pass None to write to the default time.
        """
        angles_rad = np.asarray(angles_rad, dtype=float)
        if angles_rad.shape != (_N_IMPELLERS,):
            raise ValueError(
                f"angles_rad must have shape ({_N_IMPELLERS},), got {angles_rad.shape}"
            )

        angles_deg = np.degrees(angles_rad)
        tc = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()

        for op, deg in zip(self._rotate_ops, angles_deg, strict=True):
            op.Set(float(deg), tc)

        # Keep internal state in sync
        self._angles_rad = angles_rad.copy()

    def step_with_rpm(
        self,
        rpm_array: np.ndarray,
        dt: float,
        time: float,
    ) -> None:
        """Advance each impeller's blade angle by 2π × (rpm/60) × dt and write.

        Parameters
        ----------
        rpm_array:
            Shape (28,) array of impeller RPM values.
        dt:
            Time step in seconds.
        time:
            USD time code to write the resulting angles at.
        """
        rpm_array = np.asarray(rpm_array, dtype=float)
        if rpm_array.shape != (_N_IMPELLERS,):
            raise ValueError(f"rpm_array must have shape ({_N_IMPELLERS},), got {rpm_array.shape}")

        delta = 2.0 * math.pi * (rpm_array / 60.0) * dt
        self._angles_rad = self._angles_rad + delta
        self.set_blade_angles(self._angles_rad, time=time)

    def get_blade_angles(self, time: float | None = None) -> np.ndarray:
        """Read blade rotation angles (radians) back from USD.

        Parameters
        ----------
        time:
            USD time code. Pass None to read from the default time.

        Returns
        -------
        angles_rad : np.ndarray
            Shape (28,) array of blade angles in radians.
        """
        tc = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
        angles_deg = np.array([float(op.Get(tc)) for op in self._rotate_ops], dtype=float)
        return np.radians(angles_deg)
