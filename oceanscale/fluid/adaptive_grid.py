"""Multi-resolution fluid domain using warp.fem Nanogrid / AdaptiveNanogrid."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.fem as fem


class AdaptiveFluidDomain:
    """Multi-resolution fluid domain: fine near ROV, coarse far away.

    Uses warp.fem.adaptive_nanogrid_from_hierarchy with two levels:
    coarse Volume covering the full domain and fine Volume near the focus point.
    """

    def __init__(
        self,
        domain_size: float = 100.0,
        fine_res: float = 1.0,
        coarse_res: float = 4.0,
        fine_radius: float = 10.0,
        focus: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cuda:0",
    ):
        self.domain_size = domain_size
        self.fine_res = fine_res
        self.coarse_res = coarse_res
        self.fine_radius = fine_radius
        self.device = device

        self._build(np.asarray(focus, dtype=np.float32))

    def _build(self, focus: np.ndarray) -> None:
        self._focus = focus.copy()
        half = self.domain_size / 2.0
        center = focus

        self._coarse_vol = wp.Volume.allocate(
            min=[int(center[0] - half), int(center[1] - half), int(center[2] - half)],
            max=[int(center[0] + half), int(center[1] + half), int(center[2] + half)],
            voxel_size=self.coarse_res,
            bg_value=0.0,
            device=self.device,
        )

        r = self.fine_radius
        self._fine_vol = wp.Volume.allocate(
            min=[int(center[0] - r), int(center[1] - r), int(center[2] - r)],
            max=[int(center[0] + r), int(center[1] + r), int(center[2] + r)],
            voxel_size=self.fine_res,
            bg_value=0.0,
            device=self.device,
        )

        self._grid = fem.adaptive_nanogrid_from_hierarchy(
            [self._coarse_vol, self._fine_vol]
        )
        self._space = None

    @property
    def grid(self) -> fem.AdaptiveNanogrid:
        return self._grid

    @property
    def cell_count(self) -> int:
        return self._grid.cell_count()

    @property
    def coarse_cell_count(self) -> int:
        return self._coarse_vol.get_voxel_count()

    @property
    def fine_cell_count(self) -> int:
        return self._fine_vol.get_voxel_count()

    @property
    def focus(self) -> np.ndarray:
        return self._focus.copy()

    def update_focus(self, position: np.ndarray) -> None:
        """Re-center fine resolution region around a new position."""
        self._build(np.asarray(position, dtype=np.float32))
        self._space = None

    def create_function_space(self, degree: int = 1) -> fem.CollocatedFunctionSpace:
        if self._space is None or self._space.geometry is not self._grid:
            self._space = fem.make_polynomial_space(self._grid, degree=degree)
        return self._space
# mypy: ignore-errors
