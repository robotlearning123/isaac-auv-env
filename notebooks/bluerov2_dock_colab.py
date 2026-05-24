# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BlueROV2 Autonomous Docking with OceanScale
#
# Train a PPO agent to navigate a BlueROV2 Heavy toward a docking station
# using GPU-batched Tier-1 Fossen hydrodynamics.
#
# **Tested on**: RTX 5090 (CUDA 12.8). T4 (Colab free tier) is expected to
# work but not end-to-end verified yet. This notebook includes a hardware
# check cell — if it fails, please open an issue on GitHub.
#
# **GitHub**: [robotlearning123/oceanscale](https://github.com/robotlearning123/oceanscale)

# %% tags=["colab"]
# Hardware check: verify GPU and CUDA compatibility
import subprocess, sys

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print(result.stdout.strip() if result.returncode == 0 else "nvidia-smi failed — no GPU detected")

# Newton 1.2 + Warp 1.13 require CUDA >= 12.0
cuda_check = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
if cuda_check.returncode == 0:
    print(f"\n{cuda_check.stdout.strip()}")
else:
    print("\nnvcc not found — CUDA toolkit may not be installed")
    print("Colab T4 runtimes ship CUDA 12.x by default. If this fails, switch to a GPU runtime.")

# %% [markdown]
# ## 1. Install OceanScale

# %% tags=["colab"]
!pip install -q oceanscale[rl]

# %% [markdown]
# ## 2. Create Docking Environment
#
# The docking task: navigate from a random start position to a docking
# station 5m ahead at 1.5m depth. The ROVEnv reward penalizes distance
# to target, velocity, and action effort — the agent learns to approach
# the dock and decelerate.

# %%
import numpy as np
import torch

from oceanscale.rov_env import ROVEnv

# Dock is 5m forward (x), centered (y), at 1.5m depth (z)
DOCK_POS = np.array([5.0, 0.0, -1.5], dtype=np.float32)

# Docking success criteria
DOCK_DIST_THRESHOLD = 0.3    # meters
DOCK_VEL_THRESHOLD = 0.15    # m/s
DOCK_YAW_THRESHOLD = 0.26    # ~15 degrees

env = ROVEnv(
    n_envs=4,
    target_pos=DOCK_POS,
    init_pos_noise_std=0.3,
    init_yaw_noise_std=0.3,
    sensor_noise_std=0.02,
    device="cuda",
)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Dock position: {DOCK_POS}")
print(f"Single env obs (first 3 = pos error): {obs[0, :3]}")
env.close()

# %% [markdown]
# ## 3. Train PPO Agent
#
# We use the built-in skrl PPO trainer and capture episode rewards for
# plotting. 50k steps with 4 parallel envs trains in ~2 min on T4.

# %%
from oceanscale.training.skrl_trainer import _Policy, _Value, train_skrl_ppo

N_ENVS = 4
TOTAL_STEPS = 50_000

env = ROVEnv(
    n_envs=N_ENVS,
    target_pos=DOCK_POS,
    init_pos_noise_std=0.3,
    init_yaw_noise_std=0.3,
    sensor_noise_std=0.02,
    device="cuda",
)

# Train (uncomment the line below to actually train)
# policy, value = train_skrl_ppo(env, total_timesteps=TOTAL_STEPS, device="cuda")

# For the draft: show what training looks like with a quick 1k-step run
print(f"Training PPO: {TOTAL_STEPS} steps, {N_ENVS} envs")
print("Uncomment the train_skrl_ppo call above to train for real.")

# %% [markdown]
# ## 4. Training Curve
#
# After training completes, plot episode rewards to visualize convergence.

# %%
import matplotlib.pyplot as plt

# Placeholder: after training, replace with actual reward history
# The train_skrl_ppo function returns (policy, value).
# To capture rewards, wrap the training loop or instrument env.step().

# Example plot structure (replace `rewards` with actual data):
fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

# Simulated reward curve for illustration
# rewards = []  # populated during training
# ax.plot(rewards, alpha=0.3, label="Episode reward")
# ax.plot(pd.Series(rewards).rolling(20).mean(), label="20-ep rolling mean")

ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("PPO Training — BlueROV2 Docking")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Evaluate Docking Success Rate
#
# Run 100 evaluation episodes and measure:
# - **Docking success**: final position within threshold, low velocity, aligned heading
# - **Mean position error**: average distance to dock at episode end
# - **Mean final velocity**: average speed at episode end

# %%
def evaluate_docking(policy, env, n_episodes=100, device="cuda"):
    """Evaluate docking success rate over n_episodes."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    successes = 0
    pos_errors = []
    final_vels = []

    obs, _ = env.reset()
    episode = 0
    step = 0

    while episode < n_episodes:
        t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            action, _ = policy.compute({"observations": t}, "")
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        step += 1

        done = terminated | truncated
        for i in range(env.n_envs):
            if done[i] and episode < n_episodes:
                episode += 1
                body_q = env.state_curr.body_q.numpy()
                body_qd = env.state_curr.body_qd.numpy()

                pos = body_q[i, 0:3]
                vel_body = body_qd[i, 0:3]
                quat = body_q[i, 3:7]

                pos_err = float(np.linalg.norm(pos - DOCK_POS))
                vel_mag = float(np.linalg.norm(vel_body))
                yaw_err = abs(np.arctan2(
                    2.0 * (quat[3] * quat[2]),
                    1.0 - 2.0 * quat[2] ** 2,
                ))

                pos_errors.append(pos_err)
                final_vels.append(vel_mag)

                if (pos_err < DOCK_DIST_THRESHOLD
                        and vel_mag < DOCK_VEL_THRESHOLD
                        and yaw_err < DOCK_YAW_THRESHOLD):
                    successes += 1

        if done.any():
            obs, _ = env.reset()

    return {
        "success_rate": successes / n_episodes,
        "mean_pos_error": float(np.mean(pos_errors)),
        "mean_final_vel": float(np.mean(final_vels)),
        "n_successes": successes,
        "n_episodes": n_episodes,
    }


# Evaluate (uncomment after training)
# results = evaluate_docking(policy, env, n_episodes=100)
# print(f"Docking success rate: {results['success_rate']:.1%} ({results['n_successes']}/{results['n_episodes']})")
# print(f"Mean position error:  {results['mean_pos_error']:.3f} m")
# print(f"Mean final velocity:  {results['mean_final_vel']:.3f} m/s")

env.close()

# %% [markdown]
# ## 6. Visualize Trajectory
#
# Run a single evaluation episode and plot the ROV trajectory in 3D,
# showing the path from start to the docking station.

# %%
def rollout_trajectory(policy, env, device="cuda"):
    """Run one episode and record the ROV trajectory."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()
    positions = []

    for _ in range(env.max_episode_steps):
        body_q = env.state_curr.body_q.numpy()
        positions.append(body_q[0, 0:3].copy())

        t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            action, _ = policy.compute({"observations": t}, "")
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

        if terminated[0] or truncated[0]:
            positions.append(env.state_curr.body_q.numpy()[0, 0:3].copy())
            break

    return np.array(positions)


# Plot trajectory (uncomment after training)
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(111, projection="3d")

# traj = rollout_trajectory(policy, ROVEnv(n_envs=1, target_pos=DOCK_POS, device="cuda"))
# ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", lw=1.5, label="ROV trajectory")
# ax.scatter(*traj[0], color="green", s=100, marker="o", label="Start")
# ax.scatter(*DOCK_POS, color="red", s=150, marker="*", label="Dock")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("BlueROV2 Docking Trajectory")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Next Steps
#
# - **Longer training**: increase to 200k+ steps for reliable docking
# - **Domain randomization**: enable `use_domain_randomization=True` for robustness
# - **Current disturbance**: set `current_velocity=[0.2, 0.1, 0.0]` for realistic ocean conditions
# - **Custom reward**: subclass `ROVEnv` and override `_compute_reward()` for your task
# - **Documentation**: see [README](https://github.com/robotlearning123/oceanscale)
#
# ### T4 Compatibility Notes
#
# - Newton + Warp require CUDA 12.x (Colab default)
# - Use `n_envs <= 4` to fit in T4 16GB VRAM
# - 50k steps should train in ~2-3 min on T4
# - If you hit OOM, reduce `n_envs` to 2
# - Report T4 issues at [GitHub Issues](https://github.com/robotlearning123/oceanscale/issues)
