"""OceanScale — AI-native simulation infrastructure for underwater robotics."""

__version__ = "0.1.0a0"

from oceanscale.fluid import FluidLevel, create_fluid_solver  # noqa: F401

__all__ = [
    "__version__",
    "FluidLevel",
    "create_fluid_solver",
]


def __getattr__(name: str):
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
