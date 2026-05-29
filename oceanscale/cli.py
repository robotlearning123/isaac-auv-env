"""OceanScale CLI — demo and train subcommands.

Usage:
    oceanscale --version
    oceanscale demo bluerov2-hover [--render-mp4 PATH] [--device DEVICE] [--seed SEED]
    oceanscale demo bluerov2-dock [--timesteps N] [--n_envs N] [--eval-episodes N] [--device DEVICE]
    oceanscale demo underwater-mvp [--steps N] [--render-mp4 PATH] [--device DEVICE]
    oceanscale train bluerov2-hover [--total N] [--n_envs N] [--device DEVICE] [--seed SEED]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np


def _data_dir() -> Path:
    return Path(__file__).parent / "data"


def _version(args: argparse.Namespace) -> None:
    from oceanscale import __version__

    print(f"oceanscale {__version__}")


def _resolve_hover_checkpoint(data: Path) -> Path | None:
    skrl_path = data / "bluerov2_skrl_policy.pt"
    if skrl_path.exists():
        return skrl_path

    return None


def _demo_bluerov2_hover(args: argparse.Namespace) -> None:
    import torch

    from oceanscale.rendering import VideoExporter
    from oceanscale.rov_env import ROVEnv

    data = _data_dir()
    checkpoint = _resolve_hover_checkpoint(data)

    if checkpoint is None:
        print(f"Pretrained model not found in {data}", file=sys.stderr)
        print("Expected bluerov2_skrl_policy.pt.", file=sys.stderr)
        print("Run 'oceanscale train bluerov2-hover' first.", file=sys.stderr)
        sys.exit(1)

    env = ROVEnv(n_envs=1, device=args.device, sensor_noise_std=0.0)

    model_path = checkpoint
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from oceanscale.training.skrl_trainer import _Policy

    policy: Any = cast(Any, _Policy)(env.observation_space, env.action_space, dev).to(dev)
    policy.load_state_dict(torch.load(str(model_path), map_location=dev, weights_only=True))
    policy.eval()
    print(f"Using skrl checkpoint: {model_path}")

    render_mp4 = args.render_mp4
    cinematic = getattr(args, "cinematic", False)
    exporter = None
    if render_mp4:
        exporter = VideoExporter(
            render_mp4,
            fps=30,
            view="side",
            cinematic=cinematic,
            title_card="OceanScale v0.1 — BlueROV2 hover policy smoke" if cinematic else None,
            end_card="Source install: uv sync --extra dev\nuv run oceanscale demo bluerov2-hover"
            if cinematic
            else None,
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
            assert env.state_curr.body_q is not None
            body_q = cast(Any, env.state_curr.body_q).numpy()
            pos = body_q[0, 0:3]
            quat = body_q[0, 3:7]
            state_dict = {"pos": pos, "quat": quat}
            step_time = step * dt
            exporter.record_frame(
                state_dict,
                target_pos=target_pos,
                action=action_np.flatten() if cinematic else None,
                step_time=step_time if cinematic else None,
            )

        obs, reward, terminated, truncated, _info = env.step(action_np)
        total_reward += float(np.mean(reward))

        if terminated[0] or truncated[0]:
            break

    print(f"Demo complete: {step + 1} steps, total_reward={total_reward:.2f}")

    assert env.state_curr.body_q is not None
    body_q = cast(Any, env.state_curr.body_q).numpy()
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

    from oceanscale.envs.docking_env import DockingApproachEnv, DockingApproachEnvCfg
    from oceanscale.training.skrl_trainer import train_skrl_ppo

    n_envs = args.n_envs
    timesteps = args.timesteps
    eval_episodes = args.eval_episodes
    device = args.device

    print(f"Creating DockingApproachEnv(n_envs={n_envs})...")
    cfg = DockingApproachEnvCfg(n_envs=n_envs, device=device)
    env = DockingApproachEnv(cfg)

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
            mean, _ = cast(Any, policy.compute)({"observations": t}, role="")
        obs, _reward, terminated, truncated, info = env.step(mean.cpu().numpy())

        terminated_arr = np.asarray(terminated.cpu() if hasattr(terminated, "cpu") else terminated, dtype=bool)
        truncated_arr = np.asarray(truncated.cpu() if hasattr(truncated, "cpu") else truncated, dtype=bool)
        newly_done = (terminated_arr | truncated_arr) & ~dones
        if "success" in info:
            s = info["success"]
            success_arr = np.asarray(s.cpu() if hasattr(s, "cpu") else s, dtype=bool)
            successes += int(np.sum(success_arr[newly_done]))
        episode_count += int(np.sum(newly_done))
        dones = terminated_arr | truncated_arr

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

    print("\n--- Summary ---")
    print(f"Success rate:    {success_rate:.1%} ({successes}/{episode_count})")
    print(f"Training time:   {train_time:.1f}s")
    print(f"Checkpoint:      {ckpt_path}")

    env.close()


def _demo_underwater_mvp(args: argparse.Namespace) -> None:
    from oceanscale.mvp import UnderwaterRobotMVPConfig, run_underwater_robot_mvp

    result = run_underwater_robot_mvp(
        UnderwaterRobotMVPConfig(
            n_steps=args.steps,
            device=args.device,
            seed=args.seed,
            render_mp4=args.render_mp4,
        )
    )

    print("OceanScale underwater robot MVP")
    print(f"Vehicle: {result['vehicle']['name']} ({result['vehicle']['thrusters']} thrusters)")
    print(f"Mission: {result['mission']['type']}")
    print(f"Completed: {result['mission']['completed']}")
    print(f"Final distance: {result['metrics']['final_distance_to_target_m']:.3f} m")
    print(f"Best distance: {result['metrics']['min_distance_to_target_m']:.3f} m")
    print(f"Sonar detection rate: {result['metrics']['sonar_detection_rate']:.1%}")
    print(f"Throughput: {result['metrics']['steps_per_sec']:.0f} steps/s")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Metrics JSON saved to {args.output_json}")


def _flowave_wavegen(args: argparse.Namespace) -> None:
    """Handler for: oceanscale flowave wavegen [...]

    Composes the FloWave tank stage, drives paddles+surface from
    FlapPaddleArray.synthesize_regular or synthesize_irregular (JONSWAP),
    attaches WaveProbe gauges, and exports:
      <out>/flowave_wavegen.usda   — animated USD stage
      <out>/wave_probes.npz        — eta time series at gauge positions
    """
    import importlib.util
    import math

    import numpy as np
    from pxr import Usd

    from oceanscale.facilities.flowave.paddle_array import FlapPaddleArray
    from oceanscale.facilities.flowave.paddle_usd import PaddleRingUsd
    from oceanscale.facilities.flowave.water_usd import WaterSurfaceUsd
    from oceanscale.facilities.flowave.wave_probe import WaveProbe

    _hero_path = Path(__file__).parents[1] / "scripts" / "virtual_flowave_hero.py"
    _spec = importlib.util.spec_from_file_location("virtual_flowave_hero", _hero_path)
    _hero_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_hero_mod)  # type: ignore[union-attr]
    compose_stage = _hero_mod.compose_stage

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth: float = args.depth
    direction: float = args.direction  # radians
    n_frames: int = args.frames
    dt: float = args.dt
    particles_enabled: bool = not args.no_particles
    fps: float = 1.0 / dt

    # ----------------------------------------------------------------
    # Build paddle array
    # ----------------------------------------------------------------
    paddle_array = FlapPaddleArray(R=12.5, N=168, h=depth, hinge_depth=1.9)

    # ----------------------------------------------------------------
    # Choose wave synthesis mode
    # ----------------------------------------------------------------
    if args.wave == "regular":
        H: float = args.height
        T: float = args.period
        synth = paddle_array.synthesize_regular(H=H, T=T, depth=depth, direction=direction)
        meta: dict = {
            "wave": "regular",
            "H": H,
            "T": T,
            "depth": depth,
            "direction": direction,
            "n_frames": n_frames,
            "dt": dt,
        }
    else:
        # irregular — JONSWAP Cos2s
        Hs: float = args.hs
        Tp: float = args.tp
        seed: int = getattr(args, "seed", 42)

        def _jonswap(omega: float, theta: float) -> float:
            if omega <= 0.0:
                return 0.0
            g = 9.81
            omega_p = 2.0 * math.pi / Tp
            alpha_pm = 5.0 / 16.0 * Hs ** 2 * omega_p ** 4 / g ** 2
            sigma = 0.07 if omega <= omega_p else 0.09
            r = math.exp(-((omega - omega_p) ** 2) / (2.0 * sigma ** 2 * omega_p ** 2))
            S_j = (
                alpha_pm
                * g ** 2
                / omega ** 5
                * math.exp(-1.25 * (omega_p / omega) ** 4)
                * 3.3 ** r
            )
            dangle = (theta - direction) / 2.0
            c_s = 2.0 ** 19 * math.factorial(10) ** 2 / (math.pi * math.factorial(20))
            D = max(c_s * math.cos(dangle) ** 20, 0.0)
            return S_j * D

        synth = paddle_array.synthesize_irregular(
            spectrum_func=_jonswap,
            n_freqs=128,
            omega_range=(0.3, 8.0),
            seed=seed,
        )
        meta = {
            "wave": "irregular",
            "Hs": Hs,
            "Tp": Tp,
            "depth": depth,
            "direction": direction,
            "n_frames": n_frames,
            "dt": dt,
        }

    # ----------------------------------------------------------------
    # Wave probes: centre + two downstream positions
    # ----------------------------------------------------------------
    gauge_xy = np.array(
        [
            [0.0, 0.0],    # tank centre
            [3.0, 0.0],    # 3 m downstream along +x
            [6.0, 0.0],    # 6 m downstream along +x
        ],
        dtype=np.float32,
    )
    probe = WaveProbe(gauge_xy)

    # ----------------------------------------------------------------
    # Compose USD stage
    # ----------------------------------------------------------------
    print("Composing USD stage ...")
    stage = compose_stage()
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(n_frames - 1))
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)

    paddle_ring = PaddleRingUsd(stage, ring_path="/Tank/PaddleRing")
    water_surface = WaterSurfaceUsd(stage, surface_path="/Tank/Water/Surface")

    # ----------------------------------------------------------------
    # Marine snow (optional, minimal — author empty instancer)
    # ----------------------------------------------------------------
    if particles_enabled:
        from pxr import UsdGeom, Gf, Vt
        from oceanscale.facilities.flowave.coupling import _init_particles, _wrap_cylinder
        import numpy as _np
        snow_path = "/Tank/MarineSnow"
        prim = stage.GetPrimAtPath(snow_path)
        if not prim.IsValid():
            inst = UsdGeom.PointInstancer.Define(stage, snow_path)
            proto_path = snow_path + "/Proto"
            proto = UsdGeom.Sphere.Define(stage, proto_path)
            proto.GetRadiusAttr().Set(0.003)
            inst.CreatePrototypesRel().AddTarget(proto_path)
        snow_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(snow_path))
        rng = _np.random.default_rng(1)
        particles = _init_particles(rng, 50_000, 12.5, depth)

    # ----------------------------------------------------------------
    # Simulation loop
    # ----------------------------------------------------------------
    _SWL: float = depth
    for frame in range(n_frames):
        sim_t = frame * dt
        usd_t = float(frame)

        # A0 — paddle hinge angles
        s_n = synth.paddle_commands(sim_t)
        theta_n = s_n / paddle_array.hinge_depth
        paddle_ring.set_hinge_angles(theta_n, time=usd_t)

        # A1 — water surface
        xy = water_surface.xy_grid
        eta = synth.eta_field(xy, sim_t)
        z = _SWL + eta
        water_surface.set_z_values(z, time=usd_t)

        # Wave probe record
        probe.record(synth.eta_field, sim_t)

        # A2 marine snow (forward-Euler, no impeller — zero velocity)
        if particles_enabled:
            from pxr import Usd as _Usd
            from oceanscale.facilities.flowave.coupling import _wrap_cylinder as _wc
            particles = _wc(particles, 12.5, depth)
            positions_vt = Vt.Vec3fArray(
                [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in particles]
            )
            tc = _Usd.TimeCode(usd_t)
            snow_instancer.GetPositionsAttr().Set(positions_vt, tc)

        if frame % 10 == 0:
            print(f"frame {frame}/{n_frames}  sim_t={sim_t:.3f}s")

    # ----------------------------------------------------------------
    # Export outputs
    # ----------------------------------------------------------------
    usd_path = out_dir / "flowave_wavegen.usda"
    stage.GetRootLayer().Export(str(usd_path))
    print(f"Exported USD: {usd_path}")

    npz_path = out_dir / "wave_probes.npz"
    probe.save_npz(str(npz_path), meta=meta)
    print(f"Exported wave probes: {npz_path}")


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
        choices=["bluerov2-hover", "bluerov2-dock", "underwater-mvp"],
        help="Demo task to run",
    )
    demo_parser.add_argument("--render-mp4", type=str, default=None, help="Output MP4 path")
    demo_parser.add_argument(
        "--cinematic", action="store_true", help="Use cinematic 4-panel rendering"
    )
    demo_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    demo_parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Training timesteps (bluerov2-dock)"
    )
    demo_parser.add_argument(
        "--n_envs", type=int, default=256, help="Parallel envs (bluerov2-dock)"
    )
    demo_parser.add_argument(
        "--eval-episodes", type=int, default=1000, help="Eval episodes (bluerov2-dock)"
    )
    demo_parser.add_argument(
        "--steps", type=int, default=240, help="Simulation steps (underwater-mvp)"
    )
    demo_parser.add_argument(
        "--output-json", type=str, default=None, help="Write demo metrics JSON"
    )

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

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")

    # eval robustness-profile
    profile_parser = eval_sub.add_parser(
        "robustness-profile", help="Generate evaluation profile definition"
    )
    profile_parser.add_argument(
        "--episodes-per-case", type=int, default=20, help="Episodes per evaluation case"
    )
    profile_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    profile_parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    profile_parser.add_argument(
        "--output-md", type=str, default=None, help="Output Markdown path"
    )

    # eval robustness-run
    run_parser = eval_sub.add_parser("robustness-run", help="Run evaluation cases")
    run_parser.add_argument(
        "--episodes-per-case", type=int, default=20, help="Episodes per evaluation case"
    )
    run_parser.add_argument(
        "--task", type=str, default="station-keeping", help="Task name"
    )
    run_parser.add_argument("--case-limit", type=int, default=None, help="Max cases to run")
    run_parser.add_argument(
        "--action-mode",
        type=str,
        default="zero",
        help="Action mode (zero, proportional, damped)",
    )
    run_parser.add_argument(
        "--steps-per-episode", type=int, default=10, help="Steps per episode"
    )
    run_parser.add_argument("--device", type=str, default="cpu", help="Device")
    run_parser.add_argument("--output-jsonl", type=str, default=None, help="Output JSONL path")
    run_parser.add_argument(
        "--summary-json", type=str, default=None, help="Summary JSON path"
    )

    # eval robustness-report
    report_parser = eval_sub.add_parser(
        "robustness-report", help="Generate report from JSONL records"
    )
    report_parser.add_argument(
        "--input-jsonl", type=str, required=True, help="Input JSONL path"
    )
    report_parser.add_argument(
        "--output-json", type=str, default=None, help="Output report JSON path"
    )
    report_parser.add_argument(
        "--output-md", type=str, default=None, help="Output report Markdown path"
    )

    # eval robustness-suite
    suite_parser = eval_sub.add_parser(
        "robustness-suite", help="Run evaluation suite across tasks and action modes"
    )
    suite_parser.add_argument(
        "--tasks", nargs="+", required=True, help="Task names"
    )
    suite_parser.add_argument(
        "--action-modes", nargs="+", required=True, help="Action modes"
    )
    suite_parser.add_argument(
        "--episodes-per-case", type=int, default=20, help="Episodes per evaluation case"
    )
    suite_parser.add_argument("--case-limit", type=int, default=None, help="Max cases per run")
    suite_parser.add_argument(
        "--steps-per-episode", type=int, default=10, help="Steps per episode"
    )
    suite_parser.add_argument("--device", type=str, default="cpu", help="Device")
    suite_parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory"
    )

    # flowave subcommand
    flowave_parser = subparsers.add_parser("flowave", help="FloWave tank simulation commands")
    flowave_sub = flowave_parser.add_subparsers(dest="flowave_command")

    # flowave wavegen
    wavegen_parser = flowave_sub.add_parser(
        "wavegen", help="Generate wave field and export animated USD + wave probe data"
    )
    wavegen_parser.add_argument(
        "--wave",
        choices=["regular", "irregular"],
        default="regular",
        help="Wave type (default: regular)",
    )
    wavegen_parser.add_argument(
        "--height", type=float, default=0.1, dest="height",
        help="Regular wave height H peak-to-trough (m, default 0.1)",
    )
    wavegen_parser.add_argument(
        "--period", type=float, default=2.0, dest="period",
        help="Regular wave period T (s, default 2.0)",
    )
    wavegen_parser.add_argument(
        "--hs", type=float, default=0.4, dest="hs",
        help="Significant wave height Hs (m, irregular, default 0.4)",
    )
    wavegen_parser.add_argument(
        "--tp", type=float, default=2.0, dest="tp",
        help="Peak period Tp (s, irregular, default 2.0)",
    )
    wavegen_parser.add_argument(
        "--depth", type=float, default=2.0, help="Water depth (m, default 2.0)"
    )
    wavegen_parser.add_argument(
        "--direction", type=float, default=0.0,
        help="Wave propagation direction (rad CCW from +x, default 0)",
    )
    wavegen_parser.add_argument(
        "--frames", type=int, default=60, help="Number of simulation frames (default 60)"
    )
    wavegen_parser.add_argument(
        "--dt", type=float, default=0.0333, help="Time step per frame (s, default 0.0333)"
    )
    wavegen_parser.add_argument(
        "--out", type=str, default="./flowave_out",
        help="Output directory (default: ./flowave_out)",
    )
    wavegen_parser.add_argument(
        "--no-particles", action="store_true", help="Disable marine-snow particles"
    )

    return parser


def _eval_robustness_profile(args: argparse.Namespace) -> None:
    from oceanscale.evaluation import (
        profile_to_dict,
        profile_to_markdown,
        underwater_robustness_profile,
    )

    profile = underwater_robustness_profile(
        episodes_per_case=args.episodes_per_case,
        seed=args.seed,
    )
    data = profile_to_dict(profile)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Profile JSON saved to {args.output_json}")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        md = profile_to_markdown(profile)
        out.write_text(md, encoding="utf-8")
        print(f"Profile Markdown saved to {args.output_md}")


def _eval_robustness_run(
    args: argparse.Namespace,
    *,
    evaluate_case: object | None = None,
) -> None:
    from oceanscale.evaluation import (
        CaseEvaluator,
        collect_run_metadata,
        records_to_jsonl,
        run_evaluation_profile,
        summarize_evaluation_run,
        underwater_robustness_profile,
    )

    profile = underwater_robustness_profile(
        episodes_per_case=args.episodes_per_case,
        seed=42,
    )

    run_metadata = collect_run_metadata(
        task_name=args.task,
        action_mode=args.action_mode,
        device=args.device,
        steps_per_episode=args.steps_per_episode,
        profile_name=profile.name,
        profile_seed=profile.seed,
        episodes_per_case=args.episodes_per_case,
        case_limit=args.case_limit,
    )

    if evaluate_case is None:
        from oceanscale.evaluation import resolve_case_evaluator

        evaluate_case = resolve_case_evaluator(
            args.task,
            action_mode=args.action_mode,
            steps_per_episode=args.steps_per_episode,
            device=args.device,
        )

    records = run_evaluation_profile(
        profile,
        task_name=args.task,
        evaluate_case=cast(CaseEvaluator, evaluate_case),
        case_limit=args.case_limit,
        action_mode=args.action_mode,
        run_metadata=run_metadata,
    )

    if args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(records_to_jsonl(records), encoding="utf-8")
        print(f"Records JSONL saved to {args.output_jsonl}")

    if args.summary_json:
        summary = summarize_evaluation_run(
            profile,
            records,
            task_name="station-keeping",
            action_mode=args.action_mode,
            run_metadata=run_metadata,
        )
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary JSON saved to {args.summary_json}")


def _eval_robustness_report(args: argparse.Namespace) -> None:
    from oceanscale.evaluation import (
        evaluation_records_from_jsonl,
        evaluation_report_to_markdown,
        summarize_evaluation_records,
    )

    jsonl_text = Path(args.input_jsonl).read_text(encoding="utf-8")
    loaded = evaluation_records_from_jsonl(jsonl_text)
    report = summarize_evaluation_records(loaded)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report JSON saved to {args.output_json}")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        md = evaluation_report_to_markdown(report)
        out.write_text(md, encoding="utf-8")
        print(f"Report Markdown saved to {args.output_md}")


def _eval_robustness_suite(
    args: argparse.Namespace,
    *,
    evaluate_case_factory: object | None = None,
) -> None:
    from oceanscale.evaluation import (
        GENESIS_WORLD_REFERENCE,
        CaseEvaluator,
        collect_run_metadata,
        evaluation_report_to_markdown,
        records_to_jsonl,
        run_evaluation_profile,
        summarize_evaluation_run,
        underwater_robustness_profile,
    )

    profile = underwater_robustness_profile(
        episodes_per_case=args.episodes_per_case,
        seed=42,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("eval_suite_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for task_name in args.tasks:
        for action_mode in args.action_modes:
            run_metadata = collect_run_metadata(
                task_name=task_name,
                action_mode=action_mode,
                device=args.device,
                steps_per_episode=args.steps_per_episode,
                profile_name=profile.name,
                profile_seed=profile.seed,
                episodes_per_case=args.episodes_per_case,
                case_limit=args.case_limit,
            )

            if evaluate_case_factory is not None:
                evaluate_case = cast(
                    CaseEvaluator,
                    cast(object, evaluate_case_factory)(task_name, action_mode),
                )
            else:
                from oceanscale.evaluation import resolve_case_evaluator

                evaluate_case = resolve_case_evaluator(
                    task_name,
                    action_mode=action_mode,
                    steps_per_episode=args.steps_per_episode,
                    device=args.device,
                )

            records = run_evaluation_profile(
                profile,
                task_name=task_name,
                evaluate_case=evaluate_case,
                case_limit=args.case_limit,
                action_mode=action_mode,
                run_metadata=run_metadata,
            )

            slug = f"{task_name}_{action_mode}"
            records_path = output_dir / f"{slug}_records.jsonl"
            summary_path = output_dir / f"{slug}_summary.json"
            report_json_path = output_dir / f"{slug}_report.json"
            report_md_path = output_dir / f"{slug}_report.md"

            records_path.write_text(records_to_jsonl(records), encoding="utf-8")

            summary = summarize_evaluation_run(
                profile,
                records,
                task_name=task_name,
                action_mode=action_mode,
                run_metadata=run_metadata,
            )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            report = summarize_evaluation_run(
                profile,
                records,
                task_name=task_name,
                action_mode=action_mode,
                run_metadata=run_metadata,
            )
            report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report_md_path.write_text(
                evaluation_report_to_markdown(report), encoding="utf-8"
            )

            artifacts.append(
                {
                    "task_name": task_name,
                    "action_mode": action_mode,
                    "case_count": len(records),
                    "status_counts": dict(
                        {r.status: sum(1 for x in records if x.status == r.status) for r in records}
                    )
                    if records
                    else {},
                    "records_jsonl": records_path.name,
                    "summary_json": summary_path.name,
                    "report_json": report_json_path.name,
                    "report_md": report_md_path.name,
                }
            )

    manifest = {
        "suite_type": "underwater_robustness_suite",
        "profile": profile.name,
        "tasks": list(args.tasks),
        "action_modes": list(args.action_modes),
        "case_limit": args.case_limit,
        "steps_per_episode": args.steps_per_episode,
        "source_reference": GENESIS_WORLD_REFERENCE,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Suite manifest saved to {output_dir / 'manifest.json'}")


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
        elif args.task == "underwater-mvp":
            _demo_underwater_mvp(args)
    elif args.command == "train":
        if args.task == "bluerov2-hover":
            _train_bluerov2_hover(args)
    elif args.command == "eval":
        if args.eval_command == "robustness-profile":
            _eval_robustness_profile(args)
        elif args.eval_command == "robustness-run":
            _eval_robustness_run(args)
        elif args.eval_command == "robustness-report":
            _eval_robustness_report(args)
        elif args.eval_command == "robustness-suite":
            _eval_robustness_suite(args)
        else:
            parser.parse_args(["eval", "--help"])
    elif args.command == "flowave":
        if args.flowave_command == "wavegen":
            _flowave_wavegen(args)
        else:
            parser.parse_args(["flowave", "--help"])
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
