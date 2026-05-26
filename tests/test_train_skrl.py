"""Smoke tests for skrl PPO training script (train_skrl.py)."""

from __future__ import annotations

import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTrainSkrlSmoke:
    def test_three_iterations(self) -> None:
        from oceanscale.training.train_skrl import train

        m = train(num_envs=2, max_iterations=3, device="cuda:0")
        assert len(m["iteration"]) == 3
        assert all(isinstance(v, float) for v in m["mean_reward"])
        assert all(isinstance(v, float) for v in m["policy_loss"])
        assert all(isinstance(v, float) for v in m["value_loss"])
        assert all(v > 0 for v in m["fps"])

    def test_metrics_keys(self) -> None:
        from oceanscale.training.train_skrl import train

        m = train(num_envs=2, max_iterations=1, device="cuda:0")
        assert set(m.keys()) == {
            "iteration", "mean_reward", "policy_loss", "value_loss", "fps",
        }

    def test_rewards_finite(self) -> None:
        from oceanscale.training.train_skrl import train

        m = train(num_envs=2, max_iterations=2, device="cuda:0")
        import math
        assert all(math.isfinite(v) for v in m["mean_reward"])
