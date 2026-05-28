"""YAML-based vehicle configuration loader with TAM and controller support.

Extends VehicleHydroConfig (fleet.py) with thruster allocation matrix loading
and controller parameter discovery. Any vehicle directory following the
RexROV/UUV Simulator pattern (hydro_params.yaml + TAM.yaml) can be loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oceanscale.vehicles.fleet import VehicleHydroConfig


def _asset_path(*parts: str) -> Path:
    return Path(str(files("oceanscale.assets").joinpath(*parts)))


@dataclass(frozen=True)
class VehicleConfig:
    """Full vehicle config: hydro params + TAM + controller discovery.

    Wraps VehicleHydroConfig and adds thruster allocation matrix and
    controller parameter loading from standard directory layouts.
    """

    hydro: VehicleHydroConfig
    _asset_dir: str = ""

    @property
    def name(self) -> str:
        return self.hydro.name

    @property
    def mass(self) -> float:
        return self.hydro.mass

    @property
    def n_thrusters(self) -> int:
        tam = self.thruster_allocation_matrix()
        return tam.shape[1] if tam is not None else 0

    @classmethod
    def from_dir(cls, vehicle_dir: str, name: str | None = None) -> VehicleConfig:
        """Load from a vehicle directory under oceanscale/assets/."""
        hydro_path = _asset_path(vehicle_dir, "hydro_params.yaml")
        hydro = VehicleHydroConfig.from_yaml(hydro_path)
        if name:
            hydro = VehicleHydroConfig(
                name=name,
                mass=hydro.mass,
                volume=hydro.volume,
                added_mass=hydro.added_mass,
                linear_damping=hydro.linear_damping,
                quadratic_damping=hydro.quadratic_damping,
                cog=hydro.cog,
                cob=hydro.cob,
                density=hydro.density,
            )
        return cls(hydro=hydro, _asset_dir=vehicle_dir)

    def thruster_allocation_matrix(self) -> np.ndarray | None:
        if not self._asset_dir:
            return None
        tam_path = _asset_path(self._asset_dir, "TAM.yaml")
        if not tam_path.exists():
            return None
        with open(tam_path) as f:
            data = yaml.safe_load(f)
        return np.array(data["tam"], dtype=np.float64)

    def controller_params(self, controller_name: str) -> dict[str, Any] | None:
        if not self._asset_dir:
            return None
        path = _asset_path(self._asset_dir, "controllers", f"{controller_name}.yaml")
        if not path.exists():
            return None
        with open(path) as f:
            return yaml.safe_load(f)


def load_vehicle(vehicle_dir: str) -> VehicleConfig:
    """Load a vehicle config by directory name from oceanscale/assets/."""
    return VehicleConfig.from_dir(vehicle_dir)


VEHICLE_YAML_DIRS: list[str] = [
    "rexrov",
]
