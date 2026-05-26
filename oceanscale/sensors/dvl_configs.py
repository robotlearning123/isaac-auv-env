"""Real-world DVL sensor configurations from hardware datasheets.

# Adapted from DAVE (https://github.com/Field-Robotics-Lab/dave)
# Original: urdf/sensors/nortek_dvl*_description/urdf/*.xacro
# License: Apache-2.0
# Content: DVL hardware specifications from Nortek datasheets
# Modifications: Converted from ROS/xacro to Python dataclasses, added
#   additional models from Teledyne and Sonardyne public datasheets
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DVLConfig:
    """Configuration for a real-world DVL sensor.

    Parameters follow manufacturer datasheets. Use with
    :class:`oceanscale.sensors.ray_dvl.RayDVL` via ``from_config``.
    """

    name: str
    manufacturer: str
    frequency_khz: float
    max_range_m: float
    min_range_m: float
    beam_angle_deg: float
    n_beams: int
    velocity_accuracy_m_s: float
    max_velocity_m_s: float
    update_rate_hz: float
    noise_sigma: float = 0.005
    mass_kg: float = 0.0
    dimensions_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Nortek DVL models (from DAVE project, verified against Nortek datasheets)
# ---------------------------------------------------------------------------

NORTEK_DVL1000_300 = DVLConfig(
    name="Nortek DVL1000-300",
    manufacturer="Nortek",
    frequency_khz=1000.0,
    max_range_m=75.0,
    min_range_m=0.2,
    beam_angle_deg=25.0,
    n_beams=4,
    velocity_accuracy_m_s=0.001,
    max_velocity_m_s=10.0,
    update_rate_hz=8.0,
    noise_sigma=0.005,
    mass_kg=0.15,
    dimensions_mm=(114.0, 114.0, 158.0),
)

NORTEK_DVL1000_4000 = DVLConfig(
    name="Nortek DVL1000-4000",
    manufacturer="Nortek",
    frequency_khz=1000.0,
    max_range_m=75.0,
    min_range_m=0.2,
    beam_angle_deg=25.0,
    n_beams=4,
    velocity_accuracy_m_s=0.001,
    max_velocity_m_s=10.0,
    update_rate_hz=8.0,
    noise_sigma=0.005,
    mass_kg=1.3,
    dimensions_mm=(114.0, 114.0, 167.0),
)

NORTEK_DVL500_300 = DVLConfig(
    name="Nortek DVL500-300",
    manufacturer="Nortek",
    frequency_khz=500.0,
    max_range_m=200.0,
    min_range_m=0.3,
    beam_angle_deg=25.0,
    n_beams=4,
    velocity_accuracy_m_s=0.001,
    max_velocity_m_s=10.0,
    update_rate_hz=8.0,
    noise_sigma=0.005,
    mass_kg=3.5,
    dimensions_mm=(195.0, 195.0, 170.0),
)

NORTEK_DVL500_6000 = DVLConfig(
    name="Nortek DVL500-6000",
    manufacturer="Nortek",
    frequency_khz=500.0,
    max_range_m=200.0,
    min_range_m=0.3,
    beam_angle_deg=25.0,
    n_beams=4,
    velocity_accuracy_m_s=0.001,
    max_velocity_m_s=10.0,
    update_rate_hz=8.0,
    noise_sigma=0.005,
    mass_kg=5.8,
    dimensions_mm=(195.0, 195.0, 232.0),
)

# ---------------------------------------------------------------------------
# Teledyne RDI models (specs from public datasheets)
# ---------------------------------------------------------------------------

TELEDYNE_EXPLORER = DVLConfig(
    name="Teledyne Explorer DVL",
    manufacturer="Teledyne RDI",
    frequency_khz=600.0,
    max_range_m=200.0,
    min_range_m=0.5,
    beam_angle_deg=30.0,
    n_beams=4,
    velocity_accuracy_m_s=0.003,
    max_velocity_m_s=9.0,
    update_rate_hz=5.0,
    noise_sigma=0.003,
    mass_kg=7.7,
    dimensions_mm=(195.0, 195.0, 232.0),
)

TELEDYNE_PATHFINDER = DVLConfig(
    name="Teledyne Pathfinder DVL",
    manufacturer="Teledyne RDI",
    frequency_khz=600.0,
    max_range_m=200.0,
    min_range_m=0.4,
    beam_angle_deg=30.0,
    n_beams=4,
    velocity_accuracy_m_s=0.002,
    max_velocity_m_s=9.0,
    update_rate_hz=7.0,
    noise_sigma=0.002,
    mass_kg=6.0,
    dimensions_mm=(180.0, 180.0, 200.0),
)

TELEDYNE_PIONEER = DVLConfig(
    name="Teledyne Pioneer DVL",
    manufacturer="Teledyne RDI",
    frequency_khz=300.0,
    max_range_m=500.0,
    min_range_m=0.7,
    beam_angle_deg=30.0,
    n_beams=4,
    velocity_accuracy_m_s=0.005,
    max_velocity_m_s=9.0,
    update_rate_hz=4.0,
    noise_sigma=0.005,
    mass_kg=15.0,
    dimensions_mm=(257.0, 257.0, 260.0),
)

TELEDYNE_TASMAN = DVLConfig(
    name="Teledyne Tasman DVL",
    manufacturer="Teledyne RDI",
    frequency_khz=600.0,
    max_range_m=120.0,
    min_range_m=0.3,
    beam_angle_deg=30.0,
    n_beams=4,
    velocity_accuracy_m_s=0.002,
    max_velocity_m_s=9.0,
    update_rate_hz=8.0,
    noise_sigma=0.002,
    mass_kg=1.1,
    dimensions_mm=(108.0, 108.0, 102.0),
)

# ---------------------------------------------------------------------------
# Sonardyne models (specs from public datasheets)
# ---------------------------------------------------------------------------

SONARDYNE_SYRINX = DVLConfig(
    name="Sonardyne Syrinx DVL",
    manufacturer="Sonardyne",
    frequency_khz=600.0,
    max_range_m=200.0,
    min_range_m=0.4,
    beam_angle_deg=30.0,
    n_beams=4,
    velocity_accuracy_m_s=0.003,
    max_velocity_m_s=10.0,
    update_rate_hz=5.0,
    noise_sigma=0.003,
    mass_kg=5.5,
    dimensions_mm=(191.0, 191.0, 216.0),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_DVL_CONFIGS: dict[str, DVLConfig] = {
    "nortek_dvl1000_300": NORTEK_DVL1000_300,
    "nortek_dvl1000_4000": NORTEK_DVL1000_4000,
    "nortek_dvl500_300": NORTEK_DVL500_300,
    "nortek_dvl500_6000": NORTEK_DVL500_6000,
    "teledyne_explorer": TELEDYNE_EXPLORER,
    "teledyne_pathfinder": TELEDYNE_PATHFINDER,
    "teledyne_pioneer": TELEDYNE_PIONEER,
    "teledyne_tasman": TELEDYNE_TASMAN,
    "sonardyne_syrinx": SONARDYNE_SYRINX,
}


def get_dvl_config(name: str) -> DVLConfig:
    """Look up a DVL config by key name."""
    if name not in ALL_DVL_CONFIGS:
        available = ", ".join(sorted(ALL_DVL_CONFIGS))
        raise KeyError(f"Unknown DVL config {name!r}. Available: {available}")
    return ALL_DVL_CONFIGS[name]
