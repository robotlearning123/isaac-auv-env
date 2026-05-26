#!/usr/bin/env python3
"""Run the next bounded Isaac Lab 3 official tutorial/demo smoke batch."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/storage/isaacsim-6.0-official")
DEFAULT_LAB = DEFAULT_ROOT / "sources" / "IsaacLab-release-3.0.0-beta2"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    args: tuple[str, ...]
    timeout_s: int
    markers: tuple[str, ...]
    expected_timeout: bool = True
    marker_grace_s: float = 8.0
    failure_markers: tuple[str, ...] = ("Traceback (most recent call last):", "RuntimeError:", "AttributeError:")


CASES: tuple[SmokeCase, ...] = (
    SmokeCase(
        name="tutorial_00_log_time",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/00_sim/log_time.py", "--headless"),
        timeout_s=75,
        markers=("[INFO]: Setup complete...", "[INFO] Logging experiment to directory:"),
    ),
    SmokeCase(
        name="tutorial_00_spawn_prims",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/00_sim/spawn_prims.py", "--headless"),
        timeout_s=180,
        markers=("[INFO]: Setup complete...",),
    ),
    SmokeCase(
        name="tutorial_00_set_rendering_mode",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/00_sim/set_rendering_mode.py", "--headless"),
        timeout_s=180,
        markers=("[INFO]: Setup complete...",),
    ),
    SmokeCase(
        name="tutorial_01_add_new_robot",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/01_assets/add_new_robot.py",
            "--num_envs",
            "1",
            "--headless",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting Jetbot and Dofbot state..."),
    ),
    SmokeCase(
        name="tutorial_01_run_rigid_object",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/01_assets/run_rigid_object.py", "--headless"),
        timeout_s=90,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting object state..."),
    ),
    SmokeCase(
        name="tutorial_01_run_articulation",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/01_assets/run_articulation.py", "--headless"),
        timeout_s=90,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robot state..."),
    ),
    SmokeCase(
        name="tutorial_01_run_deformable_object_physx",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/01_assets/run_deformable_object.py",
            "--backend",
            "physx",
            "--viz",
            "none",
        ),
        timeout_s=120,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting object state..."),
    ),
    SmokeCase(
        name="tutorial_01_run_deformable_object_newton",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/01_assets/run_deformable_object.py",
            "--backend",
            "newton",
            "--viz",
            "none",
        ),
        timeout_s=120,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting object state..."),
    ),
    SmokeCase(
        name="tutorial_01_run_surface_gripper_cpu",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/01_assets/run_surface_gripper.py",
            "--device",
            "cpu",
            "--headless",
        ),
        timeout_s=120,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robot state..."),
    ),
    SmokeCase(
        name="tutorial_02_create_scene",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/02_scene/create_scene.py", "--headless"),
        timeout_s=90,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robot state..."),
    ),
    SmokeCase(
        name="tutorial_03_create_cartpole_base_env",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/03_envs/create_cartpole_base_env.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=90,
        markers=("[INFO]: Resetting environment...", "[Env 0]: Pole joint:"),
    ),
    SmokeCase(
        name="tutorial_03_run_cartpole_rl_env",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/03_envs/run_cartpole_rl_env.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=90,
        markers=("[INFO]: Resetting environment...", "[Env 0]: Pole joint:"),
    ),
    SmokeCase(
        name="tutorial_03_create_cube_base_env",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/03_envs/create_cube_base_env.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=120,
        markers=("[INFO]: Resetting environment...", "[Step:"),
    ),
    SmokeCase(
        name="tutorial_03_create_quadruped_base_env",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/03_envs/create_quadruped_base_env.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=180,
        markers=("[INFO]: Resetting environment...",),
    ),
    SmokeCase(
        name="tutorial_04_add_sensors_on_robot",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/04_sensors/add_sensors_on_robot.py",
            "--num_envs",
            "1",
            "--headless",
            "--enable_cameras",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "Received shape of rgb   image:"),
    ),
    SmokeCase(
        name="tutorial_04_run_ray_caster_camera",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/04_sensors/run_ray_caster_camera.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "Received shape of depth image:"),
    ),
    SmokeCase(
        name="tutorial_04_run_ray_caster",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/04_sensors/run_ray_caster.py", "--headless"),
        timeout_s=120,
        markers=("[INFO]: Setup complete...",),
    ),
    SmokeCase(
        name="tutorial_04_run_frame_transformer",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/04_sensors/run_frame_transformer.py", "--headless"),
        timeout_s=120,
        markers=("[INFO]: Setup complete...",),
    ),
    SmokeCase(
        name="tutorial_04_run_usd_camera",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/04_sensors/run_usd_camera.py",
            "--headless",
            "--enable_cameras",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "Received shape of rgb image"),
    ),
    SmokeCase(
        name="tutorial_05_run_diff_ik",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/05_controllers/run_diff_ik.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...",),
    ),
    SmokeCase(
        name="tutorial_05_run_osc",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/tutorials/05_controllers/run_osc.py",
            "--num_envs",
            "4",
            "--headless",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...",),
    ),
    SmokeCase(
        name="demo_quadcopter",
        args=("./isaaclab.sh", "-p", "scripts/demos/quadcopter.py", "--viz", "none"),
        timeout_s=120,
        markers=("[INFO]: Setup complete...", ">>>>>>>> Reset!"),
    ),
    SmokeCase(
        name="demo_sensor_contact",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/demos/sensors/contact_sensor.py",
            "--num_envs",
            "1",
            "--headless",
            "--viz",
            "none",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robot state...", "Received contact force of:"),
    ),
    SmokeCase(
        name="demo_sensor_raycaster",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/demos/sensors/raycaster_sensor.py",
            "--num_envs",
            "1",
            "--headless",
            "--viz",
            "none",
        ),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robot state...", "Ray cast hit results:"),
    ),
    SmokeCase(
        name="demo_arms",
        args=("./isaaclab.sh", "-p", "scripts/demos/arms.py", "--headless", "--viz", "none"),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robots state..."),
    ),
    SmokeCase(
        name="demo_quadrupeds",
        args=("./isaaclab.sh", "-p", "scripts/demos/quadrupeds.py", "--headless", "--viz", "none"),
        timeout_s=180,
        markers=("[INFO]: Setup complete...", "[INFO]: Resetting robots state..."),
    ),
)


def stop_process_group(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, sig)


def run_case(case: SmokeCase, lab_root: Path, logs_dir: Path, env: dict[str, str]) -> dict[str, object]:
    log_path = logs_dir / f"{case.name}.log"
    start = time.monotonic()
    proc = subprocess.Popen(
        case.args,
        cwd=lab_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    timed_out = False
    marker_stopped = False
    output_chunks: list[str] = []
    deadline = start + case.timeout_s
    settle_deadline: float | None = None

    while True:
        now = time.monotonic()
        if proc.poll() is not None:
            output_chunks.append(proc.stdout.read() or "")
            break
        if settle_deadline is not None and now >= settle_deadline:
            marker_stopped = True
            stop_process_group(proc, signal.SIGINT)
            break
        if now >= deadline:
            timed_out = True
            stop_process_group(proc, signal.SIGINT)
            break

        wait_s = min(0.5, max(0.0, deadline - now))
        if settle_deadline is not None:
            wait_s = min(wait_s, max(0.0, settle_deadline - now))
        readable, _, _ = select.select([proc.stdout], [], [], wait_s)
        if not readable:
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        output_chunks.append(line)
        current_output = "".join(output_chunks)
        if case.expected_timeout and settle_deadline is None and all(marker in current_output for marker in case.markers):
            settle_deadline = time.monotonic() + case.marker_grace_s

    if timed_out or marker_stopped:
        stop_process_group(proc, signal.SIGINT)
        try:
            output_chunks.append(proc.communicate(timeout=25)[0] or "")
        except subprocess.TimeoutExpired:
            stop_process_group(proc, signal.SIGKILL)
            output_chunks.append(proc.communicate(timeout=10)[0] or "")
    else:
        proc.wait(timeout=10)

    output = "".join(output_chunks)
    elapsed = time.monotonic() - start
    exit_code = proc.returncode

    log_path.write_text("$ " + " ".join(case.args) + "\n" + output + f"EXIT:{exit_code}\n", encoding="utf-8")

    markers_found = {marker: marker in output for marker in case.markers}
    failures_found = {marker: marker in output for marker in case.failure_markers}
    stop_ok = (not timed_out and not marker_stopped and exit_code == 0) or (
        (timed_out or marker_stopped) and case.expected_timeout
    )
    passed = stop_ok and all(markers_found.values()) and not any(failures_found.values())
    tail = "\n".join(output.splitlines()[-30:])
    print(
        json.dumps(
            {
                "name": case.name,
                "passed": passed,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "marker_stopped": marker_stopped,
                "markers": markers_found,
                "failures": failures_found,
                "log_path": str(log_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed and tail:
        print(f"--- tail: {case.name} ---\n{tail}", flush=True)
    return {
        "name": case.name,
        "passed": passed,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "marker_stopped": marker_stopped,
        "expected_timeout": case.expected_timeout,
        "elapsed_s": round(elapsed, 3),
        "log_path": str(log_path),
        "markers": markers_found,
        "failures": failures_found,
        "command": " ".join(case.args),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_ROOT / "logs" / "official-next-smokes-2026-05-25")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.logs_dir / "summary.json"
    selected = [case for case in CASES if args.case is None or case.name in set(args.case)]
    if not selected:
        raise SystemExit("No smoke cases selected.")

    env = os.environ.copy()
    venv = args.root / "venv"
    env.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "VIRTUAL_ENV": str(venv),
            "PATH": f"{venv / 'bin'}:{env.get('PATH', '')}",
        }
    )

    results = []
    for case in selected:
        print(f"\n=== {case.name} ===", flush=True)
        results.append(run_case(case, args.lab_root, args.logs_dir, env))

    summary = {
        "root": str(args.root),
        "lab_root": str(args.lab_root),
        "logs_dir": str(args.logs_dir),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
