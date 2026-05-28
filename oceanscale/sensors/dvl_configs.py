"""Real-world DVL sensor configurations from hardware datasheets.

# Adapted from DAVE (https://github.com/Field-Robotics-Lab/dave)
# Original: urdf/sensors/nortek_dvl*_description/urdf/*.xacro
# License: Apache-2.0
# Content: DVL hardware specifications from Nortek datasheets
# Modifications: Converted from ROS/xacro to Python dataclasses, added
#   additional models from Teledyne and Sonardyne public datasheets
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


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


def _sdf_dir() -> Path:
    return Path(str(files("oceanscale.assets").joinpath("sensors", "dvl_configs")))


def list_sdf_configs() -> list[str]:
    """List available DVL SDF config files."""
    d = _sdf_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.sdf"))


def load_from_sdf(name: str) -> DVLConfig:
    """Parse a DVL SDF file from assets/sensors/dvl_configs/ and return DVLConfig.

    Args:
        name: SDF filename stem, e.g. "nortek_dvl500_300".
    """
    sdf_path = _sdf_dir() / f"{name}.sdf"
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF config not found: {sdf_path}")

    tree = ET.parse(sdf_path)
    root = tree.getroot()

    ns = {"sdf": "http://sdformat.org/schemas/root.xsd"}
    model = root.find(".//model") or root.find(".//sdf:model", ns)

    def _text(elem, tag, default=""):
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return default

    model_name = model.get("name", name) if model is not None else name

    link = model.find("link") if model is not None else None
    sensor = link.find("sensor") if link is not None else None
    plugin = sensor.find("plugin") if sensor is not None else None

    update_rate = float(_text(plugin, "updateRateHZ", "8.0")) if plugin is not None else 8.0
    noise_sigma = float(_text(plugin, "gaussianNoiseBeamVel", "0.005")) if plugin is not None else 0.005
    min_range = float(_text(plugin, "minRange", "0.2")) if plugin is not None else 0.2
    max_range = float(_text(plugin, "maxRange", "200.0")) if plugin is not None else 200.0
    beam_angle = float(_text(plugin, "beamAngleDeg", "25.0")) if plugin is not None else 25.0

    inertial = link.find("inertial") if link is not None else None
    mass_kg = float(_text(inertial, "mass", "0.0")) if inertial is not None else 0.0

    collision = link.find("collision") if link is not None else None
    geom = collision.find("geometry/cylinder") if collision is not None else None
    if geom is not None:
        radius = float(_text(geom, "radius", "0")) * 2000
        length = float(_text(geom, "length", "0")) * 1000
        dims = (radius, radius, length)
    else:
        dims = (0.0, 0.0, 0.0)

    freq = 1000.0 if "dvl1000" in name else 500.0 if "dvl500" in name else 600.0

    return DVLConfig(
        name=model_name,
        manufacturer="Nortek",
        frequency_khz=freq,
        max_range_m=max_range,
        min_range_m=min_range,
        beam_angle_deg=beam_angle,
        n_beams=4,
        velocity_accuracy_m_s=0.001,
        max_velocity_m_s=10.0,
        update_rate_hz=update_rate,
        noise_sigma=noise_sigma,
        mass_kg=mass_kg,
        dimensions_mm=dims,
    )
