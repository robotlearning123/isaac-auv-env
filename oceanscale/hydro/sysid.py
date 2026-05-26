# Reference: fishsim (https://github.com/srl-ethz/fishsim)
# License: MIT | Paper: Michelis et al., arXiv 2602.23283 (2026)
# Methodology: CMA-ES system identification for MuJoCo fluid parameters
"""System identification interface for hydrodynamic parameter calibration.

Reference module — defines the data format and interface for future
parameter identification against real hardware data. The methodology
follows fishsim's CMA-ES approach validated on tendon-driven fish robots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryRecord:
    """Recorded trajectory for system identification comparison."""

    timestamps: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray
    source: str = "simulation"
    frequency_hz: float | None = None
    phase_offset: float = 0.0


@dataclass
class SysIdConfig:
    """Configuration for CMA-ES hydrodynamic parameter identification.

    The optimization minimizes the position/orientation error between
    simulated and real trajectories by tuning fluid coefficients.
    """

    param_names: list[str] = field(
        default_factory=lambda: [
            "blunt_drag",
            "slender_drag",
            "angular_drag",
            "kutta_lift",
            "magnus_lift",
        ]
    )
    param_bounds: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.01, 5.0),
            (0.1, 20.0),
            (0.1, 10.0),
            (0.0, 10.0),
            (0.0, 2.0),
        ]
    )
    population_size: int = 20
    max_generations: int = 200
    sigma_init: float = 0.5
    metric: str = "position_rmse"


@dataclass
class SysIdResult:
    """Result of a system identification run."""

    params: dict[str, float]
    loss: float
    n_evaluations: int
    trajectory_sim: TrajectoryRecord | None = None
    trajectory_real: TrajectoryRecord | None = None
