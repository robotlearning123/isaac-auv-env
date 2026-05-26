"""Tests for real-world DVL sensor configurations."""

import numpy as np
import pytest
import warp as wp

from oceanscale.sensors.dvl_configs import (
    ALL_DVL_CONFIGS,
    DVLConfig,
    NORTEK_DVL1000_300,
    NORTEK_DVL500_300,
    SONARDYNE_SYRINX,
    TELEDYNE_EXPLORER,
    TELEDYNE_PATHFINDER,
    TELEDYNE_PIONEER,
    TELEDYNE_TASMAN,
    get_dvl_config,
)
from oceanscale.sensors.ray_dvl import RayDVL


def test_all_configs_count():
    assert len(ALL_DVL_CONFIGS) == 9


@pytest.mark.parametrize("key", list(ALL_DVL_CONFIGS.keys()))
def test_config_valid_ranges(key):
    cfg = ALL_DVL_CONFIGS[key]
    assert cfg.min_range_m > 0
    assert cfg.max_range_m > cfg.min_range_m
    assert cfg.frequency_khz > 0
    assert cfg.n_beams == 4
    assert 0 < cfg.beam_angle_deg < 90
    assert cfg.update_rate_hz > 0
    assert cfg.velocity_accuracy_m_s > 0


def test_get_dvl_config_valid():
    cfg = get_dvl_config("nortek_dvl1000_300")
    assert cfg is NORTEK_DVL1000_300


def test_get_dvl_config_invalid():
    with pytest.raises(KeyError, match="Unknown DVL config"):
        get_dvl_config("nonexistent")


def test_nortek_1000_specs():
    assert NORTEK_DVL1000_300.frequency_khz == 1000.0
    assert NORTEK_DVL1000_300.max_range_m == 75.0
    assert NORTEK_DVL1000_300.beam_angle_deg == 25.0


def test_nortek_500_specs():
    assert NORTEK_DVL500_300.frequency_khz == 500.0
    assert NORTEK_DVL500_300.max_range_m == 200.0


def test_teledyne_beam_angle():
    assert TELEDYNE_EXPLORER.beam_angle_deg == 30.0
    assert TELEDYNE_PATHFINDER.beam_angle_deg == 30.0


def test_pioneer_long_range():
    assert TELEDYNE_PIONEER.max_range_m == 500.0
    assert TELEDYNE_PIONEER.frequency_khz == 300.0


def test_tasman_compact():
    assert TELEDYNE_TASMAN.mass_kg == 1.1
    assert TELEDYNE_TASMAN.max_range_m == 120.0


def test_sonardyne_syrinx():
    assert SONARDYNE_SYRINX.manufacturer == "Sonardyne"
    assert SONARDYNE_SYRINX.max_range_m == 200.0


@pytest.mark.parametrize("key", list(ALL_DVL_CONFIGS.keys()))
def test_from_config_constructs(key):
    cfg = ALL_DVL_CONFIGS[key]
    wp.init()
    verts = np.array([[0, 0, -30], [10, 0, -30], [0, 10, -30]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3f, device="cuda:0"),
        indices=wp.array(indices, dtype=wp.int32, device="cuda:0"),
    )
    dvl = RayDVL.from_config(cfg, mesh, device="cuda:0")
    assert dvl.n_beams == cfg.n_beams
    assert dvl.max_range == cfg.max_range_m
    assert dvl.beam_angle == cfg.beam_angle_deg
