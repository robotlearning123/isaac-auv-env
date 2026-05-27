# Adapted from UUV Simulator (https://github.com/uuvsimulator/uuv_simulator)
# and DAVE (https://github.com/Field-Robotics-Lab/dave)
# Licenses: Apache-2.0
# Papers: Manhaes et al., OCEANS 2016 (UUV Sim); Aravind et al., OCEANS 2024 (DAVE)
"""Vehicle fleet definitions from open-source underwater simulation projects."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _asset_path(*parts: str) -> Path:
    return Path(str(files("oceanscale.assets").joinpath(*parts)))


@dataclass(frozen=True)
class RexROV:
    """RexROV heavy-duty ROV — 8 thrusters, 1863 kg, 2.6m length.

    Source: UUV Simulator (Apache-2.0), Manhaes et al. OCEANS 2016.
    """

    name: str = "RexROV"
    mass: float = 1862.87
    volume: float = 1.83826
    length: float = 2.6
    width: float = 1.5
    height: float = 1.6
    n_thrusters: int = 8
    cog: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cob: tuple[float, float, float] = (0.0, 0.0, 0.3)
    density: float = 1028.0

    added_mass: tuple[float, ...] = (700, 1200, 3500, 500, 800, 200)
    linear_damping: tuple[float, ...] = (-70, -70, -700, -200, -300, -100)
    quadratic_damping: tuple[float, ...] = (-700, -900, -1800, -600, -700, -500)
    inertia: tuple[float, ...] = (525.39, 794.2, 691.23)

    @staticmethod
    def mesh_path() -> Path:
        return _asset_path("rexrov", "RexROV_no_props.dae")

    @staticmethod
    def hydro_yaml_path() -> Path:
        return _asset_path("rexrov", "hydro_params.yaml")

    @staticmethod
    def tam_yaml_path() -> Path:
        return _asset_path("rexrov", "TAM.yaml")

    def thruster_allocation_matrix(self) -> np.ndarray:
        with open(self.tam_yaml_path()) as f:
            data = yaml.safe_load(f)
        return np.array(data["tam"], dtype=np.float64)

    def controller_params(self, name: str = "pid_traj") -> dict[str, Any]:
        path = _asset_path("rexrov", "controllers", f"{name}.yaml")
        with open(path) as f:
            return yaml.safe_load(f)


@dataclass(frozen=True)
class SlocumGlider:
    """Slocum electric glider — buoyancy-driven, 69.25 kg.

    Source: DAVE (Apache-2.0), Aravind et al. OCEANS 2024.
    """

    name: str = "Slocum Glider"
    mass: float = 69.25
    volume: float = 0.068
    n_thrusters: int = 1
    added_mass: tuple[float, ...] = (4, 95, 75, 0.4, 27, 32)

    @staticmethod
    def mesh_path() -> Path:
        return _asset_path("gliders", "slocum", "mesh", "Slocum-Glider.dae")

    @staticmethod
    def sdf_path() -> Path:
        return _asset_path("gliders", "slocum", "model.sdf")


@dataclass(frozen=True)
class WaveGlider:
    """Liquid Robotics Wave Glider — surface float + sub unit.

    Source: DAVE (Apache-2.0).
    """

    name: str = "Wave Glider"

    @staticmethod
    def mesh_path() -> Path:
        return _asset_path("gliders", "wave_glider", "mesh", "Wave Glider.dae")

    @staticmethod
    def sdf_path() -> Path:
        return _asset_path("gliders", "wave_glider", "model.sdf")


@dataclass(frozen=True)
class WHOIHybridGlider:
    """WHOI hybrid glider — dual propeller, 50 kg.

    Source: DAVE (Apache-2.0).
    """

    name: str = "WHOI Hybrid Glider"
    mass: float = 50.0
    n_thrusters: int = 2

    @staticmethod
    def mesh_path() -> Path:
        return _asset_path("gliders", "whoi_hybrid", "mesh")

    @staticmethod
    def sdf_path() -> Path:
        return _asset_path("gliders", "whoi_hybrid", "model.sdf")


VEHICLE_REGISTRY: dict[str, type] = {
    "rexrov": RexROV,
    "slocum": SlocumGlider,
    "wave_glider": WaveGlider,
    "whoi_hybrid": WHOIHybridGlider,
}
