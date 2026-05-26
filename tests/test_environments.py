"""Tests for environment presets."""

from oceanscale.environments import (
    CoralReefEnv,
    DeepSeaEnv,
    EnvironmentConfig,
    HarborEnv,
    IceUnderEnv,
    OpenOceanEnv,
)
from oceanscale.environments.presets import TerrainType, WaterType


def test_default_env():
    env = EnvironmentConfig()
    assert env.name == "default"
    assert env.max_depth > 0


def test_harbor_env():
    env = HarborEnv()
    assert env.name == "harbor"
    assert env.max_depth == 15.0
    assert env.water_type == WaterType.JERLOV_9C
    assert env.visibility_m < 5.0
    assert len(env.obstacle_types) > 0


def test_coral_reef_env():
    env = CoralReefEnv()
    assert env.name == "coral_reef"
    assert env.water_type == WaterType.JERLOV_I
    assert env.visibility_m > 20.0
    assert env.terrain_type == TerrainType.CORAL
    assert env.temperature_c > 20.0


def test_ice_under_env():
    env = IceUnderEnv()
    assert env.name == "ice_under"
    assert env.temperature_c < 0.0
    assert env.terrain_type == TerrainType.ICE_CEILING
    assert env.water_surface_z < 0.0
    assert env.min_depth > 0.0


def test_deep_sea_env():
    env = DeepSeaEnv()
    assert env.name == "deep_sea"
    assert env.max_depth >= 4000.0
    assert env.ambient_light == 0.0
    assert env.sun_penetration_depth == 0.0
    assert env.terrain_type == TerrainType.ABYSSAL_PLAIN


def test_open_ocean_env():
    env = OpenOceanEnv()
    assert env.name == "open_ocean"
    assert env.max_depth >= 5000.0
    assert env.current.surface_speed > 0.3
    assert env.n_obstacles == 0


def test_all_envs_have_valid_water_type():
    for factory in [HarborEnv, CoralReefEnv, IceUnderEnv, DeepSeaEnv, OpenOceanEnv]:
        env = factory()
        assert isinstance(env.water_type, WaterType)
        assert isinstance(env.terrain_type, TerrainType)
        assert env.density_kg_m3 > 900


def test_all_envs_have_current():
    for factory in [HarborEnv, CoralReefEnv, IceUnderEnv, DeepSeaEnv, OpenOceanEnv]:
        env = factory()
        assert env.current is not None
        assert env.current.surface_speed >= 0


def test_env_sound_speed_physical():
    for factory in [HarborEnv, CoralReefEnv, IceUnderEnv, DeepSeaEnv, OpenOceanEnv]:
        env = factory()
        assert 1400.0 < env.sound_speed_m_s < 1600.0
