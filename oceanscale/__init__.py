"""OceanScale — AI-native simulation infrastructure for underwater robotics."""

from typing import Any

__version__ = "0.1.0a0"

from oceanscale.fluid import FluidLevel, create_fluid_solver

__all__ = [
    "FluidLevel",
    "__version__",
    "create_fluid_solver",
]


def __getattr__(name: str) -> Any:
    """Lazy imports for GPU-heavy classes to avoid eager Warp/Newton init."""
    _lazy = {
        "GridFluidSolver": "oceanscale.fluid.grid",
        "SPHSolver": "oceanscale.fluid.grid",
        "NewtonMPMFluid": "oceanscale.fluid.mpm",
        "Tier1": "oceanscale.hydro.tier1",
        "GraphCapture": "oceanscale.graph_capture",
        "DVLSensor": "oceanscale.sensors.dvl",
        "IMUSensor": "oceanscale.sensors.imu",
        "PressureSensor": "oceanscale.sensors.pressure",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'oceanscale' has no attribute {name!r}")
