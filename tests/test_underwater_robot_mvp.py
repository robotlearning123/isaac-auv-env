"""Tests for the launchable underwater robot MVP demo."""

from __future__ import annotations

import numpy as np

from oceanscale.mvp import (
    UnderwaterRobotMVPConfig,
    compute_mission_action,
    run_underwater_robot_mvp,
)
from oceanscale.training.isaaclab_env import OCEANSCALE_UNDERWATER_TASK_ID


def test_mission_action_is_bounded() -> None:
    action = compute_mission_action(
        np.array([0.0, 0.0, -10.0], dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.array([100.0, -100.0, -20.0], dtype=np.float32),
    )

    assert action.shape == (6,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert action[0] > 0.0
    assert action[1] < 0.0


def test_underwater_robot_mvp_runs_real_vehicle_stack() -> None:
    result = run_underwater_robot_mvp(UnderwaterRobotMVPConfig(n_steps=8))

    assert result["name"] == "underwater_robot_mvp"
    assert result["vehicle"]["name"] == "BlueROV2 Heavy"
    assert result["vehicle"]["thrusters"] == 8
    assert result["mission"]["type"] == "dock_standoff_approach"
    assert "oceanscale demo underwater-mvp" in result["launch_command"]
    assert result["stack"]["newton_solver"] == "SolverVBD"
    assert result["stack"]["warp_mesh_sensors"] is True
    assert result["stack"]["vehicle_asset_source"] == "usd"
    assert result["stack"]["vehicle_asset_shapes"] >= 9
    assert result["stack"]["isaac_lab_direct_rl_env"] is True
    assert result["stack"]["isaac_lab_task_id"] == OCEANSCALE_UNDERWATER_TASK_ID
    assert result["isaac_lab_probe"]["api"] == "OceanScaleDirectRLEnv"
    assert result["isaac_lab_probe"]["task_id"] == OCEANSCALE_UNDERWATER_TASK_ID
    assert result["isaac_lab_probe"]["gym_registered"] is True
    assert result["isaac_lab_probe"]["controller"] == "mvp_standoff_pd"
    assert result["isaac_lab_probe"]["steps"] == 3
    assert result["isaac_lab_probe"]["observation_dim"] == 33
    assert result["isaac_lab_probe"]["action_dim"] == 6
    assert result["isaac_lab_probe"]["obs_finite"] is True
    assert result["isaac_lab_probe"]["reward_finite"] is True
    assert result["metrics"]["n_steps"] == 8
    assert result["metrics"]["steps_per_sec"] > 0.0
    assert result["metrics"]["sonar_detection_rate"] > 0.0
    assert np.isfinite(result["metrics"]["final_distance_to_target_m"])
    assert np.all(np.isfinite(result["metrics"]["final_position"]))
    assert 1 <= result["trajectory"]["sample_count"] <= 8
    assert result["trajectory"]["samples"][0]["step"] == 1
    assert np.all(np.isfinite(result["trajectory"]["samples"][0]["position"]))
    assert np.allclose(
        result["trajectory"]["samples"][-1]["position"],
        result["metrics"]["final_position"],
    )
