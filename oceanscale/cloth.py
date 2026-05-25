# mypy: ignore-errors
from __future__ import annotations

from typing import Any, cast

import newton
import numpy as np
import warp as wp
from numpy.typing import NDArray


class UnderwaterCloth:
    """Cloth simulation for underwater flexible structures using Newton add_cloth_grid().

    Uses SolverVBD (vertex block descent) for stable cloth dynamics.
    Suitable for nets, trawl gear, flexible ROV covers, parachute recovery.
    """

    def __init__(
        self,
        builder: newton.ModelBuilder,
        dim_x: int = 10,
        dim_y: int = 10,
        cell_x: float = 0.1,
        cell_y: float = 0.1,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 0.1,
        tri_ke: float = 1000.0,
        tri_ka: float = 1000.0,
        tri_kd: float = 10.0,
        drag_coefficient: float = 0.5,
        fix_top: bool = True,
        experimental_large_grid: bool = False,
    ) -> None:
        if max(dim_x, dim_y) > 8 and not experimental_large_grid:
            raise ValueError(
                "SolverVBD cloth grids above 8x8 require experimental_large_grid=True "
                "and caller-owned substep/iteration stability validation."
            )
        self._dim_x = dim_x
        self._dim_y = dim_y
        self._particle_start = int(builder.particle_count)

        builder.add_cloth_grid(
            pos=wp.vec3(*position),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=mass,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=drag_coefficient,
            fix_top=fix_top,
        )

        self._particle_end = int(builder.particle_count)
        self._rest_positions: NDArray[np.float32] | None = None

    @property
    def particle_count(self) -> int:
        return self._particle_end - self._particle_start

    @property
    def particle_indices(self) -> list[int]:
        return list(range(self._particle_start, self._particle_end))

    def get_positions(self, state: Any) -> NDArray[np.float32]:
        q = cast(NDArray[Any], state.particle_q.numpy())
        return cast(NDArray[np.float32], q[self._particle_start : self._particle_end, :3].copy())

    def get_deformation(self, state: Any) -> float:
        if self._rest_positions is None:
            self._rest_positions = self.get_positions(state)
            return 0.0
        current = self.get_positions(state)
        diff = current - self._rest_positions
        return float(np.sqrt(np.mean(diff**2)))
