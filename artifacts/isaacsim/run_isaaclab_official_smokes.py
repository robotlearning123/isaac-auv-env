#!/usr/bin/env python3
"""Run a bounded Isaac Lab 3 official smoke batch against the local Isaac Sim 6 venv."""

from __future__ import annotations

import argparse
import json
import os
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
    expected_timeout: bool = False


CASES: tuple[SmokeCase, ...] = (
    SmokeCase(
        name="list_envs_cartpole",
        args=("./isaaclab.sh", "-p", "scripts/environments/list_envs.py", "--keyword", "Cartpole"),
        timeout_s=180,
        markers=("Available Environments in Isaac Lab", "Isaac-Cartpole-Direct-v0"),
    ),
    SmokeCase(
        name="list_envs_cartpole_presets",
        args=(
            "./isaaclab.sh",
            "-p",
            "scripts/environments/list_envs.py",
            "--keyword",
            "Cartpole",
            "--show_presets",
        ),
        timeout_s=180,
        markers=("physics: newton_mjwarp", "renderer: isaacsim_rtx_renderer"),
    ),
    SmokeCase(
        name="tutorial_00_create_empty",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/00_sim/create_empty.py", "--viz", "none"),
        timeout_s=90,
        markers=("[INFO]: Setup complete...",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="tutorial_00_launch_app",
        args=("./isaaclab.sh", "-p", "scripts/tutorials/00_sim/launch_app.py", "--size", "0.5", "--viz", "none"),
        timeout_s=90,
        markers=("[INFO]: Setup complete...",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="pytest_core_light",
        args=(
            "./isaaclab.sh",
            "-p",
            "-m",
            "pytest",
            "source/isaaclab/test/deps/test_torch.py",
            "source/isaaclab/test/utils/test_version.py",
            "-q",
        ),
        timeout_s=240,
        markers=("36 passed",),
    ),
    SmokeCase(
        name="pytest_sim_context_headless",
        args=(
            "./isaaclab.sh",
            "-p",
            "-m",
            "pytest",
            "source/isaaclab/test/sim/test_build_simulation_context_headless.py",
            "-q",
            "-s",
        ),
        timeout_s=360,
        markers=("15 passed",),
    ),
)


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
    timed_out = False
    output = ""
    try:
        output = proc.communicate(timeout=case.timeout_s)[0] or ""
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGINT)
        try:
            output = proc.communicate(timeout=20)[0] or ""
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output = proc.communicate(timeout=10)[0] or ""
    elapsed = time.monotonic() - start
    exit_code = proc.returncode

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(case.args) + "\n")
        log_file.write(output)
        log_file.write(f"EXIT:{exit_code}\n")
    sys.stdout.write(output)
    sys.stdout.flush()
    markers_found = {marker: (marker in output) for marker in case.markers}
    timeout_ok = (not timed_out and exit_code == 0) or (timed_out and case.expected_timeout)
    passed = timeout_ok and all(markers_found.values())
    return {
        "name": case.name,
        "passed": passed,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "expected_timeout": case.expected_timeout,
        "elapsed_s": round(elapsed, 3),
        "log_path": str(log_path),
        "markers": markers_found,
        "command": " ".join(case.args),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_ROOT / "logs" / "official-smokes-2026-05-25")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--case", action="append", default=None, help="Run only this case name; may be repeated.")
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
