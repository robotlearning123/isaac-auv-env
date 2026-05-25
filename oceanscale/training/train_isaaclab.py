# mypy: ignore-errors
"""Training script using Isaac Lab DirectRLEnv-compatible API.

Uses OceanScaleDirectRLEnv with a simple PPO training loop.
Works with or without Isaac Lab installed — the env wrapper implements
the same step/reset/observation protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oceanscale.training.isaaclab_env import OceanScaleDirectRLEnv, OceanScaleEnvCfg


class SimplePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SimpleValue(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def train_docking_isaaclab(
    num_envs: int = 4,
    max_iterations: int = 10,
    lr: float = 3e-4,
    gamma: float = 0.99,
    device: str = "cuda:0",
) -> dict:
    """Train BlueROV2 using Isaac Lab-compatible PPO.

    Returns training metrics dict.
    """
    cfg = OceanScaleEnvCfg(num_envs=num_envs, device=device, episode_length_s=5.0)
    env = OceanScaleDirectRLEnv(cfg)

    policy = SimplePolicy(env.num_observations, env.num_actions).to(device)
    value_fn = SimpleValue(env.num_observations).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()), lr=lr
    )

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    metrics = {"rewards": [], "losses": []}

    for _iteration in range(max_iterations):
        actions = policy(obs)
        obs_dict, rewards, terminated, _truncated, _extras = env.step(actions)
        obs_next = obs_dict["policy"]

        with torch.no_grad():
            v_next = value_fn(obs_next).squeeze(-1)
        v_curr = value_fn(obs).squeeze(-1)
        advantage = rewards + gamma * v_next * (~terminated).float() - v_curr

        policy_loss = -(advantage.detach() * policy(obs).mean(dim=-1)).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = obs_next
        metrics["rewards"].append(rewards.mean().item())
        metrics["losses"].append(loss.item())

    env.close()
    return metrics


if __name__ == "__main__":
    result = train_docking_isaaclab(num_envs=4, max_iterations=20)
    print(f"Final mean reward: {result['rewards'][-1]:.4f}")
    print(f"Final loss: {result['losses'][-1]:.4f}")
