"""USD coupling helper for the 168-paddle FloWave ring.

This module owns the A0 per-frame update path: wave-solver paddle commands
(s_n in metres) are converted to hinge angles and written via
PaddleRingUsd.set_hinge_angles() on every simulation step.

Architecture ref: docs/virtual_flowave_architecture.md §2.5 A0, §3.1 N2 fix
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Usd, UsdGeom

_N_PADDLES = 168


class PaddleRingUsd:
    """Read/write the 168 animatable hinge-angle attributes on a FloWave ring stage.

    Parameters
    ----------
    stage : Usd.Stage
        An already-open USD stage containing the paddle ring.
    ring_path : str
        Prim path of the PaddleRing Xform, default ``/Tank/PaddleRing``.
    """

    def __init__(self, stage: Usd.Stage, ring_path: str = "/Tank/PaddleRing") -> None:
        self._stage = stage
        self._ring_path = ring_path

        # Eagerly resolve and cache the 168 rotateX ops so per-frame writes
        # avoid repeated prim / attribute lookups.
        self._rx_ops: list[UsdGeom.XformOp] = []
        for n in range(_N_PADDLES):
            prim = stage.GetPrimAtPath(f"{ring_path}/Paddle_{n:03d}")
            if not prim.IsValid():
                raise ValueError(
                    f"Paddle prim not found at {ring_path}/Paddle_{n:03d}. "
                    "Run scripts/build_flowave_paddle_ring.py first."
                )
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            if len(ops) < 3 or ops[2].GetOpName() != "xformOp:rotateX":
                raise RuntimeError(
                    f"Paddle_{n:03d}: expected xformOp:rotateX at index 2, "
                    f"got {[o.GetOpName() for o in ops]}"
                )
            self._rx_ops.append(ops[2])

    @property
    def paddle_paths(self) -> list[str]:
        """Returns 168 paddle prim paths in order n=0..167."""
        return [f"{self._ring_path}/Paddle_{n:03d}" for n in range(_N_PADDLES)]

    def set_hinge_angles(self, angles: np.ndarray, time: float | None = None) -> None:
        """Write per-paddle rotateX angle (radians, shape (168,)) at the given time code.

        If time is None, write to the default (untimed) attribute.
        This is the A0 per-frame update path the coupling code calls every frame.

        Parameters
        ----------
        angles : np.ndarray
            Shape (168,) hinge angles in radians.  Positive = paddle leans
            toward basin centre (conventional forward stroke).
        time : float | None
            USD time code (seconds × frame-rate).  None → default value.

        Raises
        ------
        ValueError
            If ``angles`` does not have exactly 168 elements.
        """
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (_N_PADDLES,):
            raise ValueError(
                f"angles must have shape ({_N_PADDLES},), got {angles.shape}"
            )

        tc = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
        # USD rotateX stores degrees; convert from radians.
        for op, rad in zip(self._rx_ops, angles):
            op.Set(math.degrees(float(rad)), tc)

    def get_hinge_angles(self, time: float | None = None) -> np.ndarray:
        """Read current hinge angles back (radians, shape (168,)); used in tests.

        Parameters
        ----------
        time : float | None
            USD time code to query.  None → default value.

        Returns
        -------
        np.ndarray
            Shape (168,) hinge angles in radians.
        """
        tc = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
        return np.array([math.radians(op.Get(tc)) for op in self._rx_ops], dtype=np.float64)
