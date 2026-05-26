"""Pre-defined underwater environment configurations.

Each preset captures the physical conditions of a common underwater scenario
(water type, depth, currents, terrain) so users can construct environments
with a single call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TerrainType(Enum):
    FLAT_SAND = "flat_sand"
    ROCKY = "rocky"
    CORAL = "coral"
    MUD = "mud"
    ABYSSAL_PLAIN = "abyssal_plain"
    ICE_CEILING = "ice_ceiling"
    CONCRETE = "concrete"


class WaterType(Enum):
    JERLOV_I = "jerlov_I"
    JERLOV_IA = "jerlov_IA"
    JERLOV_IB = "jerlov_IB"
    JERLOV_II = "jerlov_II"
    JERLOV_III = "jerlov_III"
    JERLOV_1C = "jerlov_1C"
    JERLOV_3C = "jerlov_3C"
    JERLOV_5C = "jerlov_5C"
    JERLOV_7C = "jerlov_7C"
    JERLOV_9C = "jerlov_9C"


@dataclass
class CurrentProfile:
    """Depth-varying ocean current."""

    surface_speed: float = 0.3
    bottom_speed: float = 0.05
    direction_deg: float = 0.0
    turbulence_intensity: float = 0.02


@dataclass
class EnvironmentConfig:
    """Complete underwater environment specification."""

    name: str = "default"
    description: str = ""

    # Depth
    min_depth: float = 0.0
    max_depth: float = 50.0
    water_surface_z: float = 0.0

    # Water properties
    water_type: WaterType = WaterType.JERLOV_II
    temperature_c: float = 15.0
    salinity_psu: float = 35.0
    density_kg_m3: float = 1025.0
    sound_speed_m_s: float = 1500.0
    visibility_m: float = 10.0

    # Terrain
    terrain_type: TerrainType = TerrainType.FLAT_SAND
    terrain_size_m: float = 100.0
    terrain_roughness: float = 0.1

    # Currents
    current: CurrentProfile = field(default_factory=CurrentProfile)

    # Obstacles
    n_obstacles: int = 0
    obstacle_types: list[str] = field(default_factory=list)

    # Lighting
    ambient_light: float = 0.3
    sun_penetration_depth: float = 20.0


# ---------------------------------------------------------------------------
# Pre-defined environments
# ---------------------------------------------------------------------------


def HarborEnv() -> EnvironmentConfig:
    """Shallow harbor/port environment — turbid, flat, structures."""
    return EnvironmentConfig(
        name="harbor",
        description="Shallow harbor with dock structures and moored vessels",
        min_depth=0.0,
        max_depth=15.0,
        water_type=WaterType.JERLOV_9C,
        temperature_c=18.0,
        salinity_psu=30.0,
        density_kg_m3=1020.0,
        sound_speed_m_s=1490.0,
        visibility_m=2.0,
        terrain_type=TerrainType.MUD,
        terrain_size_m=200.0,
        terrain_roughness=0.05,
        current=CurrentProfile(surface_speed=0.1, bottom_speed=0.02, turbulence_intensity=0.03),
        n_obstacles=10,
        obstacle_types=["dock_wall", "pylon", "moored_vessel"],
        ambient_light=0.4,
        sun_penetration_depth=5.0,
    )


def CoralReefEnv() -> EnvironmentConfig:
    """Clear tropical coral reef environment."""
    return EnvironmentConfig(
        name="coral_reef",
        description="Tropical coral reef with clear water and complex 3D structures",
        min_depth=0.0,
        max_depth=30.0,
        water_type=WaterType.JERLOV_I,
        temperature_c=26.0,
        salinity_psu=35.0,
        density_kg_m3=1024.0,
        sound_speed_m_s=1530.0,
        visibility_m=30.0,
        terrain_type=TerrainType.CORAL,
        terrain_size_m=150.0,
        terrain_roughness=0.8,
        current=CurrentProfile(surface_speed=0.2, bottom_speed=0.05, turbulence_intensity=0.01),
        n_obstacles=30,
        obstacle_types=["coral_head", "rock", "sand_patch"],
        ambient_light=0.6,
        sun_penetration_depth=25.0,
    )


def IceUnderEnv() -> EnvironmentConfig:
    """Under-ice polar environment — ice ceiling, deep water below."""
    return EnvironmentConfig(
        name="ice_under",
        description="Under-ice environment with ice sheet ceiling and cold deep water",
        min_depth=3.0,
        max_depth=200.0,
        water_surface_z=-3.0,
        water_type=WaterType.JERLOV_IB,
        temperature_c=-1.5,
        salinity_psu=34.0,
        density_kg_m3=1028.0,
        sound_speed_m_s=1440.0,
        visibility_m=20.0,
        terrain_type=TerrainType.ICE_CEILING,
        terrain_size_m=500.0,
        terrain_roughness=0.3,
        current=CurrentProfile(surface_speed=0.05, bottom_speed=0.02, turbulence_intensity=0.005),
        n_obstacles=5,
        obstacle_types=["ice_ridge", "ice_keel"],
        ambient_light=0.15,
        sun_penetration_depth=10.0,
    )


def DeepSeaEnv() -> EnvironmentConfig:
    """Deep-sea hydrothermal vent environment."""
    return EnvironmentConfig(
        name="deep_sea",
        description="Abyssal plain with hydrothermal vents and chimney structures",
        min_depth=1000.0,
        max_depth=4000.0,
        water_type=WaterType.JERLOV_III,
        temperature_c=2.0,
        salinity_psu=35.0,
        density_kg_m3=1045.0,
        sound_speed_m_s=1480.0,
        visibility_m=50.0,
        terrain_type=TerrainType.ABYSSAL_PLAIN,
        terrain_size_m=1000.0,
        terrain_roughness=0.02,
        current=CurrentProfile(surface_speed=0.0, bottom_speed=0.01, turbulence_intensity=0.001),
        n_obstacles=8,
        obstacle_types=["vent_chimney", "mineral_deposit", "tube_worms"],
        ambient_light=0.0,
        sun_penetration_depth=0.0,
    )


def OpenOceanEnv() -> EnvironmentConfig:
    """Open ocean mid-water environment — current dominated, no seabed."""
    return EnvironmentConfig(
        name="open_ocean",
        description="Open ocean with strong currents and no nearby seabed",
        min_depth=0.0,
        max_depth=5000.0,
        water_type=WaterType.JERLOV_IA,
        temperature_c=12.0,
        salinity_psu=35.0,
        density_kg_m3=1025.0,
        sound_speed_m_s=1500.0,
        visibility_m=25.0,
        terrain_type=TerrainType.FLAT_SAND,
        terrain_size_m=10000.0,
        terrain_roughness=0.0,
        current=CurrentProfile(surface_speed=0.5, bottom_speed=0.1, direction_deg=45.0, turbulence_intensity=0.04),
        n_obstacles=0,
        obstacle_types=[],
        ambient_light=0.5,
        sun_penetration_depth=30.0,
    )
