# mypy: ignore-errors
"""Tests for Isaac Lab 3 native DirectRLEnv task.

When Isaac Lab is not installed, tests verify the module imports
cleanly and the stub class raises ImportError with a clear message.
When Isaac Lab IS installed, tests verify the full DirectRLEnv API.
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_ISAACLAB = importlib.util.find_spec("isaaclab") is not None


class TestImportability:
    def test_module_importable(self):
        from oceanscale.training import isaaclab_task
        assert hasattr(isaaclab_task, "OceanScaleTask")
        assert hasattr(isaaclab_task, "OceanScaleTaskCfg")

    def test_cfg_instantiable(self):
        from oceanscale.training.isaaclab_task import OceanScaleTaskCfg
        cfg = OceanScaleTaskCfg(num_envs=2)
        assert cfg.num_envs == 2
        assert cfg.physics_dt == 1 / 240
        assert cfg.observation_space == 20
        assert cfg.action_space == 6

    def test_obs_act_dims(self):
        from oceanscale.training.isaaclab_task import ACT_DIM, OBS_DIM
        assert OBS_DIM == 20
        assert ACT_DIM == 6


class TestStubWithoutIsaacLab:
    @pytest.mark.skipif(HAS_ISAACLAB, reason="Isaac Lab installed — stub not used")
    def test_stub_raises_import_error(self):
        from oceanscale.training.isaaclab_task import OceanScaleTask
        with pytest.raises(ImportError, match="Isaac Sim 6"):
            OceanScaleTask()

    @pytest.mark.skipif(HAS_ISAACLAB, reason="Isaac Lab installed — stub not used")
    def test_stub_mentions_alternative(self):
        from oceanscale.training.isaaclab_task import OceanScaleTask
        with pytest.raises(ImportError, match="isaaclab_env"):
            OceanScaleTask()


@pytest.mark.skipif(not HAS_ISAACLAB, reason="Isaac Lab not installed")
class TestDirectRLEnv:
    def test_inherits_directrl(self):
        from isaaclab.envs import DirectRLEnv
        from oceanscale.training.isaaclab_task import OceanScaleTask
        assert issubclass(OceanScaleTask, DirectRLEnv)

    def test_has_required_methods(self):
        from oceanscale.training.isaaclab_task import OceanScaleTask
        for method in [
            "_setup_scene", "_pre_physics_step", "_apply_action",
            "_get_observations", "_get_rewards", "_get_dones", "_reset_idx",
        ]:
            assert hasattr(OceanScaleTask, method), f"Missing {method}"

    def test_cfg_is_directrl_ready(self):
        from isaaclab.envs import DirectRLEnvCfg
        from oceanscale.training.isaaclab_task import OceanScaleTaskCfg

        cfg = OceanScaleTaskCfg(num_envs=2, physics_dt=1 / 120, decimation=3)
        assert isinstance(cfg, DirectRLEnvCfg)
        assert hasattr(cfg, "validate")
        assert cfg.sim is not None
        assert cfg.scene is not None
        assert cfg.sim.dt == 1 / 120
        assert cfg.sim.render_interval == 3
        assert cfg.scene.num_envs == 2
        cfg.validate()
