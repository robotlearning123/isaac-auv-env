"""PPO training script for BlueROV2 station-keeping.

Usage:
    uv run python train.py                          # default: 4 envs, 100k steps
    uv run python train.py --n_envs 16 --total 1M   # larger batch
    uv run python train.py --device cpu              # CPU fallback
    uv run python train.py --render-mp4 eval.mp4     # render eval episode after training

Requires: uv sync --extra rl
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    from oceanscale.rov_env import ROVEnv
    from oceanscale.vec_env import BatchedVecEnv

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = ROVEnv(n_envs=args.n_envs, device=args.device, sensor_noise_std=0.02)
    vec_env = BatchedVecEnv(env)

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device="auto",
        seed=args.seed,
        tensorboard_log=args.tensorboard_log if args.tensorboard_log else None,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total // 10, 1000),
        save_path=str(checkpoint_dir),
        name_prefix="bluerov2_ppo",
    )

    callbacks = [checkpoint_cb]

    tensorboard_log = args.tensorboard_log

    print(f"Training PPO: {args.total} steps, {args.n_envs} envs, device={args.device}"
          + (f", tensorboard={tensorboard_log}" if tensorboard_log else ""))
    model.learn(
        total_timesteps=args.total,
        callback=CallbackList(callbacks),
        progress_bar=True,
        tb_log_name="bluerov2_ppo" if tensorboard_log else None,
    )

    final_path = checkpoint_dir / "bluerov2_station_keep_final"
    model.save(str(final_path))
    print(f"Model saved to {final_path}.zip")
    vec_env.close()

    if args.render_mp4:
        _render_eval(args)


def _render_eval(args: argparse.Namespace) -> None:
    """Run 1 eval episode and write MP4 using the trained model."""
    from stable_baselines3 import PPO

    from oceanscale.rendering import VideoExporter
    from oceanscale.rov_env import ROVEnv

    env = ROVEnv(n_envs=1, device=args.device, sensor_noise_std=0.0)
    model_path = str(Path(args.checkpoint_dir) / "bluerov2_station_keep_final")
    model = PPO.load(model_path)

    exporter = VideoExporter(args.render_mp4, fps=30, view="side")

    obs, info = env.reset()
    target_pos = info.get("target_position", env.target_pos)

    for _ in range(env.max_episode_steps):
        body_q = env.state_curr.body_q.numpy()
        pos = body_q[0, 0:3]
        quat = body_q[0, 3:7]

        exporter.record_frame(
            {"pos": pos, "quat": quat},
            target_pos=target_pos,
        )

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated[0] or truncated[0]:
            break

    exporter.close()
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BlueROV2 PPO agent")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--total", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--tensorboard_log", type=str, default=None,
                        help="TensorBoard log directory (e.g. runs/). Disabled by default.")
    parser.add_argument("--render-mp4", type=str, default=None, dest="render_mp4",
                        help="After training, render 1 eval episode to this MP4 path.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
