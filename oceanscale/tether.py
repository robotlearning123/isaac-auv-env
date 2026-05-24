"""ROV tether/umbilical cable simulation using Newton add_rod().

Uses Newton's cable joint system (chain of capsule bodies) to simulate
the physical dynamics of an ROV umbilical tether. Compatible with
SolverVBD and SolverSemiImplicit backends.
"""

from __future__ import annotations

from typing import Any, cast

import newton
import numpy as np
import warp as wp
from numpy.typing import NDArray


class Tether:
    """Simulates an ROV umbilical cable using Newton's add_rod() API.

    Creates a chain of capsule bodies connected by cable joints,
    representing a tether from a surface anchor point to the ROV.
    """

    def __init__(
        self,
        builder: newton.ModelBuilder,
        anchor_pos: tuple[float, float, float],
        attach_pos: tuple[float, float, float],
        n_segments: int = 10,
        radius: float = 0.01,
        stretch_stiffness: float = 1e4,
        bend_stiffness: float = 100.0,
        stretch_damping: float = 10.0,
        bend_damping: float = 1.0,
        density: float = 1100.0,
    ) -> None:
        self._n_segments = n_segments
        self._radius = radius

        anchor = np.array(anchor_pos, dtype=np.float64)
        attach = np.array(attach_pos, dtype=np.float64)
        direction = attach - anchor
        positions = [
            tuple(anchor + (i / n_segments) * direction)
            for i in range(n_segments + 1)
        ]
        self._length = float(np.linalg.norm(direction))

        cfg = newton.ModelBuilder.ShapeConfig(density=density)

        body_ids, joint_ids = builder.add_rod(
            positions=positions,
            radius=radius,
            cfg=cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
        )
        self._body_ids = body_ids
        self._joint_ids = joint_ids

    @property
    def body_ids(self) -> list[int]:
        return self._body_ids

    @property
    def joint_ids(self) -> list[int]:
        return self._joint_ids

    @property
    def n_segments(self) -> int:
        return self._n_segments

    @property
    def length(self) -> float:
        return self._length

    def get_positions(self, state: Any) -> NDArray[np.float32]:
        """Return (n_segments, 3) array of segment center positions."""
        q = cast(NDArray[Any], state.body_q.numpy())
        return cast(NDArray[np.float32], q[self._body_ids, :3].copy())

    def get_end_to_end_distance(self, state: Any) -> float:
        """Distance between first and last cable segment centers."""
        q = cast(NDArray[Any], state.body_q.numpy())
        p0 = q[self._body_ids[0], :3]
        p1 = q[self._body_ids[-1], :3]
        return float(np.linalg.norm(p1 - p0))

    def get_tension_estimate(self, state: Any) -> float:
        """Rough tension estimate from stretch beyond rest length."""
        d = self.get_end_to_end_distance(state)
        stretch = max(0.0, d - self._length)
        return stretch
