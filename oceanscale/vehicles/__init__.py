from oceanscale.vehicles.bluerov2 import BlueROV2Heavy, BlueROV2MarineGym, WarpAUV
from oceanscale.vehicles.commercial import (
    COMMERCIAL_VEHICLE_REGISTRY,
    HUGIN1000,
    LAUV,
    REMUS100,
    Girona500,
    SparusII,
)
from oceanscale.vehicles.config import VehicleConfig, load_vehicle
from oceanscale.vehicles.fish import TendonFish
from oceanscale.vehicles.fleet import (
    VEHICLE_REGISTRY,
    RexROV,
    SlocumGlider,
    VehicleHydroConfig,
    WaveGlider,
    WHOIHybridGlider,
)

__all__ = [
    "COMMERCIAL_VEHICLE_REGISTRY",
    "HUGIN1000",
    "LAUV",
    "REMUS100",
    "VEHICLE_REGISTRY",
    "VehicleConfig",
    "VehicleHydroConfig",
    "BlueROV2Heavy",
    "BlueROV2MarineGym",
    "Girona500",
    "RexROV",
    "SlocumGlider",
    "SparusII",
    "TendonFish",
    "WHOIHybridGlider",
    "WarpAUV",
    "WaveGlider",
    "load_vehicle",
]
