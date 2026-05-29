"""
WaterSurfaceUsd — per-frame USD coupling helper for the FloWave water surface mesh.

The 256×256 vertex grid at /Tank/Water/Surface is authored once (by
build_flowave_water.py).  At runtime, the wave-physics solver calls
set_z_values() every frame with the η field computed from the paddle
commands, which overwrites the point positions via a USD time-code.

This module is pure Python / pxr — no Isaac Sim dependency — so it can be
unit-tested without a simulation runtime.

Reference: docs/virtual_flowave_architecture.md §2.5 (A1 coupling arrow), §3.1.
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

# Grid parameters must match the values used in build_flowave_water.py.
_SURFACE_N: int = 256
_SURFACE_HALF: float = 12.0  # metres; covers -12 to +12 on each axis
_SWL: float = 2.0  # still-water level z, metres


class WaterSurfaceUsd:
    """Read/write interface for the wave-animated surface mesh.

    Parameters
    ----------
    stage:
        Open USD stage that contains /Tank/Water/Surface.
    surface_path:
        Path to the UsdGeom.Mesh prim.  Defaults to "/Tank/Water/Surface".
    """

    def __init__(
        self,
        stage: Usd.Stage,
        surface_path: str = "/Tank/Water/Surface",
    ) -> None:
        self._stage = stage
        self._path = surface_path
        prim = stage.GetPrimAtPath(surface_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found in stage: {surface_path}")
        self._mesh = UsdGeom.Mesh(prim)
        # Cache the (N*N, 2) xy grid — it never changes between frames.
        self._xy: np.ndarray = self._build_xy_grid()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        """Total number of mesh vertices (always 65,536 for the 256×256 grid)."""
        return _SURFACE_N * _SURFACE_N

    @property
    def xy_grid(self) -> np.ndarray:
        """(65536, 2) array of (x, y) coordinates for each vertex.

        The wave solver receives these to evaluate η(x, y, t) at every vertex.
        Row order: x-major (vertex [i*256+j] has x=xs[i], y=ys[j]).
        x and y range from -12.0 to +12.0 metres inclusive.
        """
        return self._xy

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def set_z_values(
        self,
        z_array: np.ndarray,
        time: float | None = None,
    ) -> None:
        """Write per-vertex z values to the mesh points attribute.

        Parameters
        ----------
        z_array:
            Shape (65536,) float array of z positions (metres).  Each element
            corresponds to the vertex at the same linear index as xy_grid.
        time:
            USD time code.  None → Usd.TimeCode.Default().
        """
        if z_array.shape != (self.vertex_count,):
            raise ValueError(f"z_array must have shape ({self.vertex_count},); got {z_array.shape}")
        xy = self._xy
        # PERF: replace list comprehension with bulk numpy-to-Vt for 10-100x speedup (see coupling.py)
        pts = [
            Gf.Vec3f(float(xy[k, 0]), float(xy[k, 1]), float(z_array[k]))
            for k in range(self.vertex_count)
        ]
        tc = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
        self._mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts), tc)

    def get_z_values(self, time: float | None = None) -> np.ndarray:
        """Read per-vertex z positions from the mesh.

        Returns
        -------
        np.ndarray
            Shape (65536,) float32 array of z values.
        """
        tc = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
        pts = self._mesh.GetPointsAttr().Get(tc)
        if pts is None:
            raise RuntimeError("No points data at the requested time code")
        return np.array([p[2] for p in pts], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_xy_grid() -> np.ndarray:
        """Build and return the (65536, 2) xy coordinate grid."""
        n = _SURFACE_N
        xs = np.linspace(-_SURFACE_HALF, _SURFACE_HALF, n, dtype=np.float32)
        ys = np.linspace(-_SURFACE_HALF, _SURFACE_HALF, n, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([xg.ravel(), yg.ravel()])
