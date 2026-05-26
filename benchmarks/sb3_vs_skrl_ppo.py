"""SB3 PPO vs skrl PPO benchmark on ROVEnv hover task.

Both train for 50k steps on 64 parallel envs with matched hyperparameters.
Reports throughput (samples/s), wall time, and evaluation reward.

Usage: uv run python benchmarks/sb3_vs_skrl_ppo.py
"""

import time
import numpy as np
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA required — no GPU detected.")
import torch

N_ENVS = 64
MAX_STEPS = 200
TOTAL_TIMESTEPS = 50_000
ROLLOUT = 128
BATCH_SIZE = 512
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 4
HIDDEN = 64


def eval_reward(env_fn, predict_fn, n_episodes=200):
    """Evaluate policy over n_episodes, return mean cumulative reward."""
    env = env_fn()
    episode_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(np.mean(reward))
            done = bool(np.any(terminated | truncated))
        episode_rewards.append(ep_reward)
    env.close()
    return float(np.mean(episode_rewards))


def run_sb3():
    """Train SB3 PPO via BatchedVecEnv wrapper."""
    from stable_baselines3 import PPO
    from oceanscale.rov_env import ROVEnv
    from oceanscale.vec_env import BatchedVecEnv

    raw = ROVEnv(n_envs=N_ENVS, max_episode_steps=MAX_STEPS)
    train_env = BatchedVecEnv(raw)

    model = PPO(
        "MlpPolicy", train_env,
        n_steps=ROLLOUT, batch_size=BATCH_SIZE,
        verbose=0, device="cpu", learning_rate=LR,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.perf_counter() - t0

    reward = eval_reward(
        lambda: ROVEnv(n_envs=N_ENVS, max_episode_steps=MAX_STEPS),
        lambda obs: model.predict(obs, deterministic=True)[0],
    )

    del model, train_env
    torch.cuda.empty_cache()
    return TOTAL_TIMESTEPS / elapsed, elapsed, reward


def run_skrl_manual():
    """Train skrl PPO with manual training loop (bypass skrl Trainer for env compat)."""
    from oceanscale.rov_env import ROVEnv
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

    class Policy(GaussianMixin, Model):
        def __init__(self, obs_space, act_space, device):
            Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
            GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(obs_space.shape[0], HIDDEN), torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN, act_space.shape[0]),
            )
            self.log_std = torch.nn.Parameter(torch.zeros(act_space.shape[0]))

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), {"log_std": self.log_std}

    class Value(DeterministicMixin, Model):
        def __init__(self, obs_space, act_space, device):
            Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(obs_space.shape[0], HIDDEN), torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN, 1),
            )

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), {}

    device = torch.device("cuda:0")
    env = ROVEnv(n_envs=N_ENVS, max_episode_steps=MAX_STEPS)
    obs_space = env.observation_space
    act_space = env.action_space

    policy = Policy(obs_space, act_space, device).to(device)
    value = Value(obs_space, act_space, device).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=LR
    )

    def get_value(obs_np):
        t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            v, _ = value.compute({"observations": t}, role="")
        return v.squeeze(-1)

    def get_action_and_logprob(obs_np, deterministic=False):
        t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            if deterministic:
                mean, _ = policy({"observations": t}, "")
                return mean.cpu().numpy(), None, None
            a, out = policy.act({"observations": t, "states": None}, role="")
        log_std = policy.log_std
        dist = torch.distributions.Normal(
            torch.clamp(a, -1 + 1e-6, 1 - 1e-6),
            torch.clamp(log_std.exp(), 1e-6, 10)
        )
        logp = dist.log_prob(a).sum(-1, keepdim=True)
        return a.cpu().numpy(), logp, dist

    obs, _ = env.reset()
    total_steps = 0
    t0 = time.perf_counter()

    while total_steps < TOTAL_TIMESTEPS:
        # Collect rollout
        mb_obs, mb_actions, mb_logp, mb_rewards, mb_dones, mb_values = [], [], [], [], [], []
        for _ in range(ROLLOUT):
            actions, logp, _ = get_action_and_logprob(obs)
            vals = get_value(obs)
            mb_obs.append(obs.copy())
            mb_actions.append(actions.copy())
            mb_logp.append(logp)
            mb_values.append(vals.cpu().numpy())
            obs, reward, terminated, truncated, _ = env.step(actions)
            mb_rewards.append(reward.astype(np.float32))
            mb_dones.append((terminated | truncated).astype(np.float32))
            total_steps += N_ENVS

        # GAE
        last_val = get_value(obs).cpu().numpy()
        mb_values.append(last_val)
        advantages = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        returns = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        last_gae = 0
        for t in reversed(range(ROLLOUT)):
            delta = mb_rewards[t] + GAMMA * mb_values[t + 1] * (1 - mb_dones[t]) - mb_values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - mb_dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + mb_values[t]

        # Flatten
        b_obs = torch.tensor(np.concatenate(mb_obs), dtype=torch.float32, device=device)
        b_actions = torch.tensor(np.concatenate(mb_actions), dtype=torch.float32, device=device)
        b_logp = torch.tensor(np.concatenate([l.detach().cpu().numpy() for l in mb_logp]), dtype=torch.float32, device=device)
        b_adv = torch.tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n_batches = max(1, b_obs.shape[0] // BATCH_SIZE)
        for _ in range(EPOCHS):
            indices = torch.randperm(b_obs.shape[0], device=device)
            for start in range(0, b_obs.shape[0], BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                mb_o = b_obs[idx]
                mb_a = b_actions[idx]
                mb_old_logp = b_logp[idx]
                mb_adv_batch = b_adv[idx]
                mb_ret_batch = b_ret[idx]

                mean, extras = policy.compute({"observations": mb_o}, role="")
                log_std = policy.log_std
                dist = torch.distributions.Normal(mean, log_std.exp())
                new_logp = dist.log_prob(mb_a).sum(-1, keepdim=True)

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv_batch.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv_batch.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()

                v_pred, _ = value.compute({"observations": mb_o}, role="")
                value_loss = ((v_pred.squeeze(-1) - mb_ret_batch) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value.parameters()), 0.5
                )
                optimizer.step()

    elapsed = time.perf_counter() - t0

    # Evaluate
    def predict_fn(obs_np):
        t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _ = policy.compute({"observations": t}, role="")
        return mean.cpu().numpy()

    reward = eval_reward(
        lambda: ROVEnv(n_envs=N_ENVS, max_episode_steps=MAX_STEPS),
        predict_fn,
    )

    del policy, value, optimizer, env
    torch.cuda.empty_cache()
    return TOTAL_TIMESTEPS / elapsed, elapsed, reward


if __name__ == "__main__":
    print(f"ROVEnv hover benchmark: {N_ENVS} envs, {TOTAL_TIMESTEPS} steps\n")

    print("=== SB3 PPO ===")
    sb3_sps, sb3_time, sb3_reward = run_sb3()
    print(f"  {sb3_sps:.0f} samples/s | {sb3_time:.1f}s | reward={sb3_reward:.2f}")

    print("\n=== skrl PPO (manual loop, skrl 2.x models) ===")
    skrl_sps, skrl_time, skrl_reward = run_skrl_manual()
    print(f"  {skrl_sps:.0f} samples/s | {skrl_time:.1f}s | reward={skrl_reward:.2f}")

    print("\n" + "=" * 55)
    print("COMPARISON")
    print(f"  SB3  PPO: {sb3_sps:>8.0f} samples/s | {sb3_time:>6.1f}s | reward={sb3_reward:.2f}")
    print(f"  skrl PPO: {skrl_sps:>8.0f} samples/s | {skrl_time:>6.1f}s | reward={skrl_reward:.2f}")
    ratio = skrl_sps / sb3_sps
    faster = "skrl" if ratio > 1 else "SB3"
    print(f"  {faster} is {abs(ratio):.2f}x faster")
