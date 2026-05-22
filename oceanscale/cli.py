"""OceanScale CLI — demo and train subcommands.

Usage:
    oceanscale --version
    oceanscale demo bluerov2-hover [--render-mp4 PATH] [--device DEVICE] [--seed SEED]
    oceanscale train bluerov2-hover [--total N] [--n_envs N] [--device DEVICE] [--seed SEED]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _data_dir() -> Path:
    return Path(__file__).parent / "data"


def _version(args: argparse.Namespace) -> None:
    from oceanscale import __version__

    print(f"oceanscale {__version__}")


def _demo_bluerov2_hover(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from oceanscale.rendering import VideoExporter
    from oceanscale.rov_env import ROVEnv
    from oceanscale.vec_env import BatchedVecEnv

    data = _data_dir()
    model_path = data / "bluerov2_station_keep_final"
    vec_norm_path = data / "vec_normalize.pkl"

    if not model_path.with_suffix(".zip").exists():
        print(f"Pretrained model not found at {model_path}.zip", file=sys.stderr)
        print("Run 'oceanscale train bluerov2-hover' first.", file=sys.stderr)
        sys.exit(1)

    env = ROVEnv(n_envs=1, device=args.device, sensor_noise_std=0.0)
    vec_env = BatchedVecEnv(env)

    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    model = PPO.load(str(model_path))

    render_mp4 = args.render_mp4
    cinematic = getattr(args, "cinematic", False)
    exporter = None
    if render_mp4:
        exporter = VideoExporter(
            render_mp4, fps=30, view="side", cinematic=cinematic,
            title_card="OceanScale v0.1 — BlueROV2 hover with PPO" if cinematic else None,
            end_card="10.5x faster than PyBullet at n=64\npip install oceanscale" if cinematic else None,
        )

    obs = vec_env.reset()
    target_pos = env.target_pos.copy()
    dt = 1.0 / 30.0  # approx display timestep

    total_reward = 0.0
    for step in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)

        if exporter is not None:
            body_q = env.state_curr.body_q.numpy()
            pos = body_q[0, 0:3]
            quat = body_q[0, 3:7]
            state_dict = {"pos": pos, "quat": quat}
            step_time = step * dt
            action_arr = np.asarray(action).flatten()
            exporter.record_frame(
                state_dict, target_pos=target_pos,
                action=action_arr if cinematic else None,
                step_time=step_time if cinematic else None,
            )

        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])

        if done[0]:
            break

    print(f"Demo complete: {step + 1} steps, total_reward={total_reward:.2f}")

    body_q = env.state_curr.body_q.numpy()
    final_pos = body_q[0, 0:3]
    depth_err = abs(target_pos[2] - final_pos[2])
    pos_err = float(np.linalg.norm(target_pos - final_pos))
    print(f"Final position: {final_pos}")
    print(f"Target position: {target_pos}")
    print(f"Position error: {pos_err:.4f}m, Depth error: {depth_err:.4f}m")

    if exporter is not None:
        exporter.close()
        print(f"MP4 saved to {render_mp4}")

    vec_env.close()


def _train_bluerov2_hover(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from oceanscale.rov_env import ROVEnv
    from oceanscale.vec_env import BatchedVecEnv

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = ROVEnv(n_envs=args.n_envs, device=args.device, sensor_noise_std=0.02)
    vec_env = BatchedVecEnv(env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)

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
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total // 10, 1000),
        save_path=str(checkpoint_dir),
        name_prefix="bluerov2_ppo",
    )

    print(f"Training PPO: {args.total} steps, {args.n_envs} envs, device={args.device}")
    model.learn(
        total_timesteps=args.total,
        callback=CallbackList([checkpoint_cb]),
        progress_bar=True,
    )

    final_path = checkpoint_dir / "bluerov2_station_keep_final"
    model.save(str(final_path))
    print(f"Model saved to {final_path}.zip")

    vec_norm_path = checkpoint_dir / "vec_normalize.pkl"
    vec_env.save(str(vec_norm_path))
    print(f"VecNormalize stats saved to {vec_norm_path}")

    vec_env.close()

    if args.render_mp4:
        args_demo = argparse.Namespace(
            device=args.device,
            render_mp4=args.render_mp4,
        )
        _demo_bluerov2_hover(args_demo)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oceanscale",
        description="OceanScale — GPU-native underwater robotics simulator",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(dest="command")

    # demo subcommand
    demo_parser = subparsers.add_parser("demo", help="Run a pretrained policy demo")
    demo_parser.add_argument(
        "task",
        choices=["bluerov2-hover"],
        help="Demo task to run",
    )
    demo_parser.add_argument("--render-mp4", type=str, default=None, help="Output MP4 path")
    demo_parser.add_argument("--cinematic", action="store_true", help="Use cinematic 4-panel rendering")
    demo_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train an RL policy")
    train_parser.add_argument(
        "task",
        choices=["bluerov2-hover"],
        help="Training task",
    )
    train_parser.add_argument("--total", type=int, default=100_000, help="Total timesteps")
    train_parser.add_argument("--n_envs", type=int, default=4, help="Parallel envs")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", dest="checkpoint_dir"
    )
    train_parser.add_argument("--render-mp4", type=str, default=None, help="Render after train")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        _version(args)
        return

    if args.command == "demo":
        if args.task == "bluerov2-hover":
            _demo_bluerov2_hover(args)
    elif args.command == "train":
        if args.task == "bluerov2-hover":
            _train_bluerov2_hover(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
