"""skrl PPO training script for OceanScaleDirectRLEnv.

All-GPU training loop using skrl GaussianMixin/DeterministicMixin models.
No CPU round-trips — rollout, GAE, and PPO update stay on device.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, cast

import torch
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from oceanscale.training.isaaclab_env import OceanScaleDirectRLEnv, OceanScaleEnvCfg

_NN = torch.nn

class Policy(GaussianMixin, Model):
    def __init__(self, obs_sp, act_sp, device):
        cast(Any, Model).__init__(self, observation_space=obs_sp, action_space=act_sp, device=device)
        cast(Any, GaussianMixin).__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.net = _NN.Sequential(
            _NN.Linear(obs_sp.shape[0], 128), _NN.Tanh(),
            _NN.Linear(128, 64), _NN.Tanh(), _NN.Linear(64, act_sp.shape[0]))
        self.log_std = _NN.Parameter(torch.full((act_sp.shape[0],), -1.0))
    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std}

class Value(DeterministicMixin, Model):
    def __init__(self, obs_sp, act_sp, device):
        cast(Any, Model).__init__(self, observation_space=obs_sp, action_space=act_sp, device=device)
        cast(Any, DeterministicMixin).__init__(self, clip_actions=False)
        self.net = _NN.Sequential(
            _NN.Linear(obs_sp.shape[0], 128), _NN.Tanh(),
            _NN.Linear(128, 64), _NN.Tanh(), _NN.Linear(64, 1))
    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}

def train(num_envs: int = 64, max_iterations: int = 100, device: str = "cuda:0") -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    env = OceanScaleDirectRLEnv(OceanScaleEnvCfg(num_envs=num_envs, device=device))
    pi = Policy(env.observation_space, env.action_space, dev).to(dev)
    vf = Value(env.observation_space, env.action_space, dev).to(dev)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), lr=3e-4)
    ROLLOUT, GAMMA, LAM, CLIP = 32, 0.99, 0.95, 0.2
    bsz = max(1, num_envs * ROLLOUT // 4)
    metrics = {k: [] for k in ("iteration", "mean_reward", "policy_loss", "value_loss", "fps")}
    obs, params = env.reset()[0]["policy"], list(pi.parameters()) + list(vf.parameters())

    for it in range(max_iterations):
        t0 = time.time()
        ro_o, ro_a, ro_l, ro_r, ro_d, ro_v = [], [], [], [], [], []
        for _ in range(ROLLOUT):
            with torch.no_grad():
                m, _ = pi.compute({"observations": obs}, "")
                d = torch.distributions.Normal(m, pi.log_std.exp())
                act = d.sample()
                lp = d.log_prob(act).sum(-1, keepdim=True)
                v, _ = vf.compute({"observations": obs}, "")
            ro_o.append(obs); ro_a.append(act); ro_l.append(lp); ro_v.append(v.squeeze(-1))
            od, rew, term, trunc, _ = env.step(act)
            obs = od["policy"]; ro_r.append(rew); ro_d.append((term | trunc).float())

        with torch.no_grad():
            lv, _ = vf.compute({"observations": obs}, "")
        ro_v.append(lv.squeeze(-1))
        adv, gae = torch.zeros(ROLLOUT, num_envs, device=dev), torch.zeros(num_envs, device=dev)
        for t in reversed(range(ROLLOUT)):
            delta = ro_r[t] + GAMMA * ro_v[t+1] * (1 - ro_d[t]) - ro_v[t]
            gae = delta + GAMMA * LAM * (1 - ro_d[t]) * gae; adv[t] = gae
        b_o, b_a, b_lp = torch.cat(ro_o), torch.cat(ro_a), torch.cat(ro_l)
        b_ret = (adv + torch.stack(ro_v[:-1])).reshape(-1)
        b_adv = adv.reshape(-1); b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        pls = vls = 0.0
        for _ in range(4):
            idx = torch.randperm(b_o.shape[0], device=dev)
            for s in range(0, b_o.shape[0], bsz):
                i = idx[s:s+bsz]
                m, _ = pi.compute({"observations": b_o[i]}, "")
                d = torch.distributions.Normal(m, pi.log_std.exp())
                ratio = (d.log_prob(b_a[i]).sum(-1, keepdim=True) - b_lp[i]).exp()
                pl = -torch.min(ratio * b_adv[i].unsqueeze(-1),
                                torch.clamp(ratio, 1-CLIP, 1+CLIP) * b_adv[i].unsqueeze(-1)).mean()
                vl = ((vf.compute({"observations": b_o[i]}, "")[0].squeeze(-1) - b_ret[i]) ** 2).mean()
                (pl + 0.5*vl - 0.01*d.entropy().sum(-1).mean()).backward()
                _NN.utils.clip_grad_norm_(params, 0.5); opt.step(); opt.zero_grad()
                pls += pl.item(); vls += vl.item()

        dt, mr = max(time.time() - t0, 1e-6), torch.stack(ro_r).sum(0).mean().item()
        fps = num_envs * ROLLOUT / dt
        for k, v in zip(metrics, [float(it), mr, pls, vls, fps]): metrics[k].append(v)
        print(f"[{it:3d}] reward={mr:.3f} ploss={pls:.4f} vloss={vls:.4f} fps={fps:.0f}")

    env.close()
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    train(num_envs=a.num_envs, max_iterations=a.iterations, device=a.device)
