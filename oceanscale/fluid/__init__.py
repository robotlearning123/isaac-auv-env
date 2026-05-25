"""OceanScale fluid simulation — multi-fidelity GPU-native ocean physics.

Fluid Fidelity Ladder:
    Level 0: No fluid (Fossen analytical hydrodynamics only)
    Level 1: Grid Eulerian (Chorin projection, spatially varying currents)
    Level 2: SPH (Lagrangian particles, fluid-body interaction)
    Level 3: MPM (Material Point Method via Newton, deformable terrain)
    Level 4: Volume (wp.Volume NanoVDB sparse grid, ocean-scale domains)

All levels are GPU-native on NVIDIA Warp / Newton.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oceanscale.fluid.grid import GridFluidSolver, SPHSolver
    from oceanscale.fluid.mpm import NewtonMPMFluid
    from oceanscale.fluid.volume_solver import VolumeFluidSolver


class FluidLevel(IntEnum):
    NONE = 0
    GRID = 1
    SPH = 2
    MPM = 3
    VOLUME = 4


def create_fluid_solver(level: int | FluidLevel, **kwargs: Any) -> Any:
    """Create a fluid solver at the specified fidelity level.

    Args:
        level: FluidLevel (0-3).
        **kwargs: forwarded to the solver constructor.

    Returns:
        Solver instance, or ``None`` for level 0.

    Raises:
        ValueError: if *level* is not in 0-3.
    """
    level = FluidLevel(level)

    if level == FluidLevel.NONE:
        return None

    if level == FluidLevel.GRID:
        from oceanscale.fluid.grid import GridFluidSolver
        return GridFluidSolver(**kwargs)

    if level == FluidLevel.SPH:
        from oceanscale.fluid.grid import SPHSolver
        return SPHSolver(**kwargs)

    if level == FluidLevel.MPM:
        from oceanscale.fluid.mpm import NewtonMPMFluid
        return NewtonMPMFluid(**kwargs)

    if level == FluidLevel.VOLUME:
        from oceanscale.fluid.volume_solver import VolumeFluidSolver
        return VolumeFluidSolver(**kwargs)

    raise ValueError(f"Unknown fluid level: {level}")


def __getattr__(name: str) -> Any:
    """Lazy imports for heavy GPU-dependent classes."""
    if name == "GridFluidSolver":
        from oceanscale.fluid.grid import GridFluidSolver
        return GridFluidSolver
    if name == "SPHSolver":
        from oceanscale.fluid.grid import SPHSolver
        return SPHSolver
    if name == "NewtonMPMFluid":
        from oceanscale.fluid.mpm import NewtonMPMFluid
        return NewtonMPMFluid
    if name == "OceanWaveField":
        from oceanscale.fluid.wave import OceanWaveField
        return OceanWaveField
    if name == "VolumeFluidSolver":
        from oceanscale.fluid.volume_solver import VolumeFluidSolver
        return VolumeFluidSolver
    raise AttributeError(f"module 'oceanscale.fluid' has no attribute {name!r}")


__all__ = [
    "FluidLevel",
    "GridFluidSolver",
    "NewtonMPMFluid",
    "OceanWaveField",
    "SPHSolver",
    "VolumeFluidSolver",
    "create_fluid_solver",
]
