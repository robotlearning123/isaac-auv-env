"""Runnable underwater robot MVP built on the real OceanScale demo stack."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oceanscale.demo import UnifiedDemo
from oceanscale.rendering import VideoExporter
from oceanscale.vehicles import BlueROV2Heavy


@dataclass(frozen=True)
class UnderwaterRobotMVPConfig:
    """Configuration for the BlueROV2 docking-standoff MVP mission."""

    n_steps: int = 240
    device: str = "cuda:0"
    seed: int = 42
    target_standoff_m: float = 2.5
    success_radius_m: float = 0.15
    render_mp4: str | None = None
    trajectory_samples: int = 32
    isaaclab_probe_steps: int = 3
    isaaclab_probe_envs: int = 1


def compute_mission_action(
    position: np.ndarray,
    velocity: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """Compute a bounded 6-DOF command toward a docking standoff target."""

    err = target_position.astype(np.float32) - position.astype(np.float32)
    linear = velocity[:3].astype(np.float32)
    action = np.zeros(6, dtype=np.float32)
    action[0] = np.clip(1.10 * err[0] - 0.20 * linear[0], -1.0, 1.0)
    action[1] = np.clip(1.10 * err[1] - 0.20 * linear[1], -1.0, 1.0)
    action[2] = np.clip(0.25 * err[2] - 0.12 * linear[2], -0.6, 0.6)
    desired_yaw = np.arctan2(float(err[1]), float(err[0])) if np.linalg.norm(err[:2]) > 1e-6 else 0.0
    action[5] = np.clip(0.12 * desired_yaw, -0.4, 0.4)
    return action


def _sample_trajectory(
    positions: list[np.ndarray],
    actions: list[np.ndarray],
    *,
    dt: float,
    target_position: np.ndarray,
    max_samples: int,
) -> list[dict[str, Any]]:
    if not positions or max_samples <= 0:
        return []

    count = min(max_samples, len(positions))
    indices = np.linspace(0, len(positions) - 1, num=count, dtype=np.int64)
    samples: list[dict[str, Any]] = []
    for raw_index in indices:
        index = int(raw_index)
        position = positions[index]
        action = actions[index] if index < len(actions) else np.zeros(6, dtype=np.float32)
        samples.append(
            {
                "step": index + 1,
                "time_s": float((index + 1) * dt),
                "position": position.astype(float).tolist(),
                "action": action.astype(float).tolist(),
                "distance_to_target_m": float(np.linalg.norm(position - target_position)),
            }
        )
    return samples


def _run_isaaclab_probe(
    cfg: UnderwaterRobotMVPConfig,
    target_position: np.ndarray,
) -> dict[str, Any] | None:
    if cfg.isaaclab_probe_steps <= 0:
        return None

    import torch

    from oceanscale.training.isaaclab_env import (
        ACT_DIM,
        HAS_ISAACLAB,
        OBS_DIM,
        OCEANSCALE_UNDERWATER_TASK_ID,
        OceanScaleDirectRLEnv,
        OceanScaleEnvCfg,
        register_oceanscale_isaaclab_tasks,
    )

    task_id = register_oceanscale_isaaclab_tasks()
    env = OceanScaleDirectRLEnv(
        OceanScaleEnvCfg(
            num_envs=cfg.isaaclab_probe_envs,
            episode_length_s=2.0,
            device=cfg.device,
        )
    )
    reward_history: list[float] = []
    terminated_any = False
    truncated_any = False
    try:
        obs_dict, _ = env.reset()
        for _ in range(cfg.isaaclab_probe_steps):
            obs = obs_dict["policy"].detach().cpu().numpy()
            actions = np.stack(
                [
                    compute_mission_action(row[:3], row[3:9], target_position)
                    for row in obs
                ]
            )
            action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=env.device)
            obs_dict, rewards, terminated, truncated, _info = env.step(action_tensor)
            reward_history.extend(rewards.detach().cpu().numpy().astype(float).tolist())
            terminated_any = terminated_any or bool(torch.any(terminated).item())
            truncated_any = truncated_any or bool(torch.any(truncated).item())

        final_obs = obs_dict["policy"]
        return {
            "api": "OceanScaleDirectRLEnv",
            "task_id": task_id,
            "controller": "mvp_standoff_pd",
            "isaac_lab_installed": HAS_ISAACLAB,
            "gym_registered": task_id == OCEANSCALE_UNDERWATER_TASK_ID,
            "num_envs": env.num_envs,
            "steps": cfg.isaaclab_probe_steps,
            "observation_dim": OBS_DIM,
            "action_dim": ACT_DIM,
            "device": str(env.device),
            "obs_finite": bool(torch.all(torch.isfinite(final_obs)).item()),
            "reward_finite": bool(np.all(np.isfinite(reward_history))),
            "mean_reward": float(np.mean(reward_history)) if reward_history else 0.0,
            "terminated_any": terminated_any,
            "truncated_any": truncated_any,
        }
    finally:
        env.close()


def run_underwater_robot_mvp(config: UnderwaterRobotMVPConfig | None = None) -> dict[str, Any]:
    """Run the real underwater robot MVP and return launch-ready metrics."""

    cfg = config or UnderwaterRobotMVPConfig()
    vehicle = BlueROV2Heavy()
    dock_position_tuple = (4.0, 0.0, -10.0)
    dock_position = np.array(dock_position_tuple, dtype=np.float32)
    target_position = dock_position - np.array([cfg.target_standoff_m, 0.0, 0.0], dtype=np.float32)
    demo = UnifiedDemo(
        seabed_depth=-30.0,
        wave_height=1.2,
        wave_period=8.0,
        current_speed=0.25,
        tether_length=25.0,
        dt=0.02,
        device=cfg.device,
        vehicle=vehicle,
        dock_position=dock_position_tuple,
        enable_structures=False,
        asset_source="usd",
    )

    exporter = None
    if cfg.render_mp4 is not None:
        exporter = VideoExporter(
            cfg.render_mp4,
            fps=30,
            view="top",
            cinematic=True,
            title_card="OceanScale underwater robot MVP",
            end_card="BlueROV2 Heavy + Warp/Newton ocean physics",
        )

    obs = demo.reset()
    distances: list[float] = []
    rewards: list[float] = []
    altitudes: list[float] = []
    sonar_hits = 0
    positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    t0 = time.perf_counter()

    try:
        for step in range(cfg.n_steps):
            action = compute_mission_action(
                obs["rov_position"],
                obs["rov_velocity"],
                target_position,
            )
            if exporter is not None:
                exporter.record_frame(
                    {"pos": obs["rov_position"], "quat": obs["rov_orientation"]},
                    target_pos=target_position,
                    action=action,
                    step_time=step * demo.dt,
                )
            actions.append(action.copy())

            obs = demo.step(action)
            distance = float(np.linalg.norm(obs["rov_position"] - target_position))
            ranges = np.asarray(obs["sonar_ranges"], dtype=np.float32)
            distances.append(distance)
            rewards.append(float(obs["reward"]))
            altitudes.append(float(obs["dvl"]["altitude"]))
            positions.append(obs["rov_position"].copy())
            sonar_hits += int(np.any(ranges < demo.sonar.max_range))
            if distance <= cfg.success_radius_m:
                break
    finally:
        if exporter is not None:
            exporter.close()

    wall_time = time.perf_counter() - t0
    steps_executed = len(distances)
    start_distance = float(np.linalg.norm(np.array([0.0, 0.0, -10.0]) - target_position))
    final_distance = distances[-1] if distances else start_distance
    min_distance = min(distances) if distances else start_distance

    isaaclab_probe = _run_isaaclab_probe(cfg, target_position)
    result = {
        "name": "underwater_robot_mvp",
        "launch_command": "oceanscale demo underwater-mvp --steps 240 --device cuda:0",
        "render_mp4": str(Path(cfg.render_mp4)) if cfg.render_mp4 is not None else None,
        "vehicle": {
            "name": "BlueROV2 Heavy",
            "mass_kg": vehicle.mass,
            "volume_m3": vehicle.volume,
            "dimensions_m": [vehicle.length, vehicle.width, vehicle.height],
            "thrusters": vehicle.n_thrusters,
            "source": "von Benzon et al. 2022 Table A1",
        },
        "mission": {
            "type": "dock_standoff_approach",
            "dock_position": dock_position.tolist(),
            "target_position": target_position.tolist(),
            "success_radius_m": cfg.success_radius_m,
            "completed": bool(final_distance <= cfg.success_radius_m),
        },
        "metrics": {
            "n_steps": steps_executed,
            "max_steps": cfg.n_steps,
            "sim_time_s": steps_executed * demo.dt,
            "wall_time_s": wall_time,
            "steps_per_sec": steps_executed / max(wall_time, 1e-9),
            "start_distance_to_target_m": start_distance,
            "final_distance_to_target_m": final_distance,
            "min_distance_to_target_m": min_distance,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_altitude_m": float(np.mean(altitudes)) if altitudes else 0.0,
            "sonar_detection_rate": sonar_hits / max(steps_executed, 1),
            "final_position": obs["rov_position"].tolist(),
        },
        "trajectory": {
            "sample_count": min(max(cfg.trajectory_samples, 0), len(positions)),
            "samples": _sample_trajectory(
                positions,
                actions,
                dt=demo.dt,
                target_position=target_position,
                max_samples=cfg.trajectory_samples,
            ),
        },
        "stack": {
            "warp_mesh_sensors": True,
            "newton_solver": "SolverVBD",
            "wave_model": "FFTWaveField/JONSWAP",
            "ocean_current": True,
            "water_column": True,
            "acoustics": True,
            "propulsion": True,
            "buoyancy": True,
            "tether": False,
            "vehicle_asset_source": demo.asset_source,
            "vehicle_asset_path": demo.asset_path,
            "vehicle_asset_shapes": demo.asset_shape_count,
            "isaac_lab_direct_rl_env": isaaclab_probe is not None,
            "isaac_lab_task_id": (
                isaaclab_probe["task_id"] if isaaclab_probe is not None else None
            ),
            "launch_stability_mode": "robot_only_no_tether_cloth",
        },
    }
    if isaaclab_probe is not None:
        result["isaac_lab_probe"] = isaaclab_probe
    return result
