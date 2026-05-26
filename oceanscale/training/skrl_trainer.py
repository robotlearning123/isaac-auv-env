"""skrl PPO trainer for GPU-batched ROVEnv.

Uses manual training loop with skrl 2.x GaussianMixin/DeterministicMixin models.
All tensors live on GPU — no CPU↔GPU round-trips per step.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class _Policy(GaussianMixin, Model):  # type: ignore[misc]
    def __init__(self, observation_space: Any, action_space: Any, device: torch.device) -> None:
        cast(Any, Model).__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        cast(Any, GaussianMixin).__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, action_space.shape[0]),
        )
        self.log_std = torch.nn.Parameter(torch.full((action_space.shape[0],), -1.0))

    def compute(
        self, inputs: dict[str, torch.Tensor], role: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.net(inputs["observations"]), {"log_std": self.log_std}


class _Value(DeterministicMixin, Model):  # type: ignore[misc]
    def __init__(self, observation_space: Any, action_space: Any, device: torch.device) -> None:
        cast(Any, Model).__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        cast(Any, DeterministicMixin).__init__(self, clip_actions=False)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )

    def compute(
        self, inputs: dict[str, torch.Tensor], role: str
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.net(inputs["observations"]), {}


_DEFAULT_CFG: dict[str, Any] = dict(
    rollout_steps=512,
    batch_size=4096,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    epochs=4,
)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def train_skrl_ppo(
    env: Any,
    total_timesteps: int = 100_000,
    device: str = "cuda",
    **kwargs: Any,
) -> tuple[_Policy, _Value]:
    """Train PPO with skrl on a GPU-batched ROVEnv.

    Returns (policy, value) models with trained weights on device.
    """
    cfg = {**_DEFAULT_CFG, **kwargs}
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    obs_space = env.observation_space
    act_space = env.action_space
    n_envs = getattr(env, "n_envs", None) or env.num_envs

    policy = cast(_Policy, _Policy(obs_space, act_space, dev).to(dev))
    value = cast(_Value, _Value(obs_space, act_space, dev).to(dev))
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=cfg["learning_rate"],
    )

    def _get_value(obs_np: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
        with torch.no_grad():
            v, _ = value.compute({"observations": t}, "")
        return v.squeeze(-1)

    def _act(obs_np: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
        with torch.no_grad():
            mean, _ = policy.compute({"observations": t}, role="")
            std = policy.log_std.exp()
            dist = cast(Any, torch.distributions.Normal(mean, std))
            a = cast(torch.Tensor, dist.sample())
            logp = cast(torch.Tensor, dist.log_prob(a).sum(-1, keepdim=True))
        return a.cpu().numpy(), logp

    gamma = cfg["gamma"]
    lam = cfg["gae_lambda"]
    rollout = cfg["rollout_steps"]
    bsize = cfg["batch_size"]
    clip_eps = cfg["clip_eps"]
    epochs = cfg["epochs"]

    obs, _ = env.reset()
    total = 0

    while total < total_timesteps:
        mb_obs, mb_act, mb_logp, mb_rew, mb_done, mb_val = [], [], [], [], [], []

        for _ in range(rollout):
            actions, logp = _act(obs)
            vals = _get_value(obs)
            mb_obs.append(_to_numpy(obs).copy())
            mb_act.append(_to_numpy(actions).copy())
            mb_logp.append(logp)
            mb_val.append(vals.cpu().numpy())
            obs, reward, terminated, truncated, _ = env.step(actions)
            mb_rew.append(_to_numpy(reward).astype(np.float32))
            mb_done.append((_to_numpy(terminated) | _to_numpy(truncated)).astype(np.float32))
            total += n_envs

        # GAE
        last_val = _get_value(obs).cpu().numpy()
        mb_val.append(last_val)
        adv = np.zeros((rollout, n_envs), dtype=np.float32)
        ret = np.zeros((rollout, n_envs), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(rollout)):
            delta = mb_rew[t] + gamma * mb_val[t + 1] * (1 - mb_done[t]) - mb_val[t]
            gae = delta + gamma * lam * (1 - mb_done[t]) * gae
            adv[t] = gae
            ret[t] = gae + mb_val[t]

        b_obs = torch.as_tensor(np.concatenate(mb_obs), dtype=torch.float32, device=dev)
        b_act = torch.as_tensor(np.concatenate(mb_act), dtype=torch.float32, device=dev)
        b_logp = torch.as_tensor(
            np.concatenate([l.detach().cpu().numpy() for l in mb_logp]),
            dtype=torch.float32,
            device=dev,
        )
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=dev)
        b_ret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=dev)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(epochs):
            idx = torch.randperm(b_obs.shape[0], device=dev)
            for start in range(0, b_obs.shape[0], bsize):
                i = idx[start : start + bsize]
                mb_o, mb_a = b_obs[i], b_act[i]
                mb_old_logp, mb_adv_b, mb_ret_b = b_logp[i], b_adv[i], b_ret[i]

                mean, _ = policy.compute({"observations": mb_o}, role="")
                dist = cast(Any, torch.distributions.Normal(mean, policy.log_std.exp()))
                new_logp = cast(torch.Tensor, dist.log_prob(mb_a).sum(-1, keepdim=True))

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv_b.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv_b.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()

                v_pred, _ = value.compute({"observations": mb_o}, role="")
                value_loss = ((v_pred.squeeze(-1) - mb_ret_b) ** 2).mean()

                entropy = cast(torch.Tensor, dist.entropy().sum(-1).mean())
                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value.parameters()),
                    cfg["max_grad_norm"],
                )
                optimizer.step()

    return policy, value
