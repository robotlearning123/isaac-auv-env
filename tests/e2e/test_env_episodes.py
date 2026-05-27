"""End-to-end episode tests with video recording.

Each test runs a full episode in a real environment and records an MP4 video
for visual review. Videos are saved to tmp/test_videos/.

These tests require GPU (CUDA) and are slow — run with:
    pytest tests/e2e/ -v -x
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from oceanscale.rendering.video import VideoExporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_DIR = Path(__file__).resolve().parent.parent.parent / "tmp" / "test_videos"
MAX_STEPS = 300  # short episodes for CI speed


def _extract_state(env: Any, env_idx: int = 0) -> dict[str, np.ndarray]:
    """Extract pos/quat from environment for video recording."""
    state = env.sim.observe_torch()
    pos = state["position"][env_idx].cpu().numpy()
    quat = state["orientation"][env_idx].cpu().numpy()
    return {"pos": pos, "quat": quat}


def _get_target(env: Any) -> np.ndarray | None:
    """Extract target position from env, handling different env interfaces."""
    if hasattr(env, "_target"):
        return env._target.cpu().numpy()
    if hasattr(env, "target_pos"):
        return env.target_pos.cpu().numpy()
    if hasattr(env, "_current_targets"):
        targets = env._current_targets()
        return targets[0].cpu().numpy()
    return None


def _get_dt(env: Any) -> float:
    """Extract dt from env config."""
    if hasattr(env, "cfg") and hasattr(env.cfg, "dt"):
        return env.cfg.dt
    if hasattr(env, "_dt"):
        return env._dt
    return 1.0 / 240.0


def _get_device(env: Any) -> torch.device:
    """Extract device from env."""
    if hasattr(env, "_device"):
        return env._device
    if hasattr(env, "device"):
        return env.device
    return torch.device("cuda:0")


def _run_episode_with_video(
    env: Any,
    video_path: Path,
    max_steps: int = MAX_STEPS,
    cinematic: bool = False,
    title: str | None = None,
) -> dict[str, float]:
    """Run a full episode, record video, return summary stats.

    Uses random actions — the goal is to verify the env runs end-to-end
    and produces a reviewable video, not to solve the task.
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    exporter = VideoExporter(
        str(video_path),
        fps=30,
        view="side",
        cinematic=cinematic,
        title_card=title,
    )

    device = _get_device(env)
    dt = _get_dt(env)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action = env.action_space.sample()
        if isinstance(action, np.ndarray):
            action_t = torch.tensor(action, dtype=torch.float32, device=device)
        else:
            action_t = action

        obs, reward, terminated, truncated, info = env.step(action_t)

        # Record frame from env 0
        state = _extract_state(env, env_idx=0)
        target = _get_target(env)
        action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()
        exporter.record_frame(
            state,
            target_pos=target,
            action=action_np,
            step_time=step * dt,
        )

        total_reward += float(reward) if np.isscalar(reward) else float(reward[0])
        steps += 1

        if np.isscalar(terminated):
            if terminated or truncated:
                break
        else:
            if terminated[0] or truncated[0]:
                break

    exporter.close()
    return {
        "steps": steps,
        "total_reward": total_reward,
        "video_path": str(video_path),
    }


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDockingEpisode:
    """E2E test: full docking approach episode with video."""

    def test_full_episode_records_video(self, video_dir: Path) -> None:
        from oceanscale.envs.docking_env import DockingApproachEnv, DockingApproachEnvCfg

        cfg = DockingApproachEnvCfg(n_envs=1, max_episode_steps=MAX_STEPS)
        env = DockingApproachEnv(cfg)

        stats = _run_episode_with_video(
            env,
            video_dir / "e2e_docking.mp4",
            max_steps=MAX_STEPS,
            cinematic=True,
            title="E2E: Docking Approach — Random Policy",
        )

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert Path(stats["video_path"]).exists()
        assert Path(stats["video_path"]).stat().st_size > 10_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStationKeepingEpisode:
    """E2E test: full station-keeping episode with video."""

    def test_full_episode_records_video(self, video_dir: Path) -> None:
        from oceanscale.envs.station_keeping_env import CurrentStationKeepingEnv

        env = CurrentStationKeepingEnv(n_envs=1, max_episode_steps=MAX_STEPS)

        stats = _run_episode_with_video(
            env,
            video_dir / "e2e_station_keeping.mp4",
            max_steps=MAX_STEPS,
            cinematic=True,
            title="E2E: Station Keeping — Random Policy",
        )

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert Path(stats["video_path"]).exists()
        assert Path(stats["video_path"]).stat().st_size > 10_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestWaypointEpisode:
    """E2E test: full waypoint-following episode with video."""

    def test_full_episode_records_video(self, video_dir: Path) -> None:
        from oceanscale.envs.waypoint_env import WaypointFollowingEnv

        env = WaypointFollowingEnv(n_envs=1, n_waypoints=3, max_episode_steps=MAX_STEPS)

        stats = _run_episode_with_video(
            env,
            video_dir / "e2e_waypoint.mp4",
            max_steps=MAX_STEPS,
            cinematic=True,
            title="E2E: Waypoint Following — Random Policy",
        )

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert Path(stats["video_path"]).exists()
        assert Path(stats["video_path"]).stat().st_size > 10_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestIntegrationPipelineEpisode:
    """E2E test: integration pipeline episode with video."""

    def test_full_pipeline_records_video(self, video_dir: Path) -> None:
        from oceanscale.envs.docking_env import DockingApproachEnv, DockingApproachEnvCfg

        cfg = DockingApproachEnvCfg(n_envs=1, max_episode_steps=MAX_STEPS)
        env = DockingApproachEnv(cfg)

        stats = _run_episode_with_video(
            env,
            video_dir / "e2e_integration_pipeline.mp4",
            max_steps=MAX_STEPS,
            cinematic=True,
            title="E2E: Integration Pipeline — Full Stack",
        )

        assert stats["steps"] > 0
        assert Path(stats["video_path"]).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMultiEnvBatchedEpisode:
    """E2E test: batched multi-env episode (records env 0)."""

    def test_batched_episode_records_video(self, video_dir: Path) -> None:
        from oceanscale.envs.docking_env import DockingApproachEnv, DockingApproachEnvCfg

        cfg = DockingApproachEnvCfg(n_envs=8, max_episode_steps=MAX_STEPS)
        env = DockingApproachEnv(cfg)

        stats = _run_episode_with_video(
            env,
            video_dir / "e2e_batched_8env.mp4",
            max_steps=MAX_STEPS,
            cinematic=False,
            title="E2E: Batched 8-Env — Env[0] Recording",
        )

        assert stats["steps"] > 0
        assert Path(stats["video_path"]).exists()
