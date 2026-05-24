"""OceanScale CLI — demo and train subcommands.

Usage:
    oceanscale --version
    oceanscale demo bluerov2-hover [--render-mp4 PATH] [--device DEVICE] [--seed SEED]
    oceanscale demo bluerov2-dock [--timesteps N] [--n_envs N] [--eval-episodes N] [--device DEVICE]
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
    import torch

    from oceanscale.rendering import VideoExporter
    from oceanscale.rov_env import ROVEnv
    from oceanscale.training.skrl_trainer import _Policy

    data = _data_dir()
    model_path = data / "bluerov2_skrl_policy.pt"

    if not model_path.exists():
        print(f"Pretrained model not found at {model_path}", file=sys.stderr)
        print("Run 'oceanscale train bluerov2-hover' first.", file=sys.stderr)
        sys.exit(1)

    env = ROVEnv(n_envs=1, device=args.device, sensor_noise_std=0.0)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = _Policy(env.observation_space, env.action_space, dev).to(dev)
    policy.load_state_dict(torch.load(str(model_path), map_location=dev, weights_only=True))
    policy.eval()

    render_mp4 = args.render_mp4
    cinematic = getattr(args, "cinematic", False)
    exporter = None
    if render_mp4:
        exporter = VideoExporter(
            render_mp4, fps=30, view="side", cinematic=cinematic,
            title_card="OceanScale v0.1 — BlueROV2 hover with PPO (skrl)" if cinematic else None,
            end_card="10.5x faster than PyBullet at n=64\npip install oceanscale" if cinematic else None,
        )

    obs, _ = env.reset()
    target_pos = env.target_pos.copy()
    dt = 1.0 / 30.0

    total_reward = 0.0
    for step in range(env.max_episode_steps):
        t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            action, _ = policy.compute({"observations": t}, "")
        action_np = action.cpu().numpy()

        if exporter is not None:
            body_q = env.state_curr.body_q.numpy()
            pos = body_q[0, 0:3]
            quat = body_q[0, 3:7]
            state_dict = {"pos": pos, "quat": quat}
            step_time = step * dt
            exporter.record_frame(
                state_dict, target_pos=target_pos,
                action=action_np.flatten() if cinematic else None,
                step_time=step_time if cinematic else None,
            )

        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += float(np.mean(reward))

        if terminated[0] or truncated[0]:
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

    env.close()


def _train_bluerov2_hover(args: argparse.Namespace) -> None:
    import torch

    from oceanscale.rov_env import ROVEnv

    if getattr(args, "legacy", False):
        from stable_baselines3 import PPO
        print("WARNING: --legacy uses SB3 CPU trainer (deprecated)")
        env = ROVEnv(n_envs=args.n_envs, device=args.device, sensor_noise_std=0.02)
        model = PPO("MlpPolicy", env, n_steps=128, batch_size=256, verbose=1, device=args.device)
        model.learn(total_timesteps=args.total)
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        model.save(str(Path(args.checkpoint_dir) / "bluerov2_sb3_model"))
        print(f"Legacy model saved to {args.checkpoint_dir}/bluerov2_sb3_model.zip")
        return

    from oceanscale.training.skrl_trainer import train_skrl_ppo

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = ROVEnv(n_envs=args.n_envs, device=args.device, sensor_noise_std=0.02)

    print(f"Training skrl PPO: {args.total} steps, {args.n_envs} envs, device={args.device}")
    policy, value = train_skrl_ppo(env, total_timesteps=args.total, device=args.device)

    policy_path = checkpoint_dir / "bluerov2_skrl_policy.pt"
    torch.save(policy.state_dict(), str(policy_path))
    print(f"Policy saved to {policy_path}")

    value_path = checkpoint_dir / "bluerov2_skrl_value.pt"
    torch.save(value.state_dict(), str(value_path))
    print(f"Value network saved to {value_path}")

    env.close()

    if args.render_mp4:
        args_demo = argparse.Namespace(
            device=args.device,
            render_mp4=args.render_mp4,
        )
        _demo_bluerov2_hover(args_demo)


def _demo_bluerov2_dock(args: argparse.Namespace) -> None:
    import time

    import torch

    from oceanscale.envs.docking_env import DockingApproachEnv
    from oceanscale.training.skrl_trainer import train_skrl_ppo

    n_envs = args.n_envs
    timesteps = args.timesteps
    eval_episodes = args.eval_episodes
    device = args.device

    print(f"Creating DockingApproachEnv(n_envs={n_envs})...")
    env = DockingApproachEnv(n_envs=n_envs, device=device)

    print(f"Training skrl PPO: {timesteps} steps, {n_envs} envs, device={device}")
    t0 = time.perf_counter()
    policy, _value = train_skrl_ppo(env, total_timesteps=timesteps, device=device)
    train_time = time.perf_counter() - t0
    print(f"Training complete in {train_time:.1f}s")

    # Evaluate
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_eval = eval_episodes
    obs, _ = env.reset()
    successes = 0
    episode_count = 0
    dones = np.zeros(n_envs, dtype=bool)

    while episode_count < n_eval:
        t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            mean, _ = policy.compute({"observations": t}, role="")
        obs, reward, terminated, truncated, info = env.step(mean.cpu().numpy())

        newly_done = (terminated | truncated) & ~dones
        if "success" in info:
            successes += int(np.sum(info["success"][newly_done]))
        episode_count += int(np.sum(newly_done))
        dones = terminated | truncated

        if np.all(dones):
            obs, _ = env.reset()
            dones = np.zeros(n_envs, dtype=bool)

    success_rate = successes / max(episode_count, 1)
    print(f"Evaluation: {episode_count} episodes, {successes} successes ({success_rate:.1%})")

    # Save checkpoint
    out_dir = Path("oceanscale_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "bluerov2_dock.pt"
    torch.save(policy.state_dict(), str(ckpt_path))

    print(f"\n--- Summary ---")
    print(f"Success rate:    {success_rate:.1%} ({successes}/{episode_count})")
    print(f"Training time:   {train_time:.1f}s")
    print(f"Checkpoint:      {ckpt_path}")

    env.close()


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
        choices=["bluerov2-hover", "bluerov2-dock"],
        help="Demo task to run",
    )
    demo_parser.add_argument("--render-mp4", type=str, default=None, help="Output MP4 path")
    demo_parser.add_argument("--cinematic", action="store_true", help="Use cinematic 4-panel rendering")
    demo_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    demo_parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps (bluerov2-dock)")
    demo_parser.add_argument("--n_envs", type=int, default=256, help="Parallel envs (bluerov2-dock)")
    demo_parser.add_argument("--eval-episodes", type=int, default=1000, help="Eval episodes (bluerov2-dock)")

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
    train_parser.add_argument("--legacy", action="store_true",
                              help="Use legacy SB3 trainer (deprecated, will be removed in v0.2)")
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
        elif args.task == "bluerov2-dock":
            _demo_bluerov2_dock(args)
    elif args.command == "train":
        if args.task == "bluerov2-hover":
            _train_bluerov2_hover(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
