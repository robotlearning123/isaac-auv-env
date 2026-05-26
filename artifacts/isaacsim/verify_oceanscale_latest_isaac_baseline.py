#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAAC_ROOT = Path(os.environ.get("ISAAC_ROOT", "/mnt/storage/isaacsim-6.0-official"))
ISAAC_ROOT = DEFAULT_ISAAC_ROOT
ISAAC_VENV_PYTHON = ISAAC_ROOT / "venv/bin/python"
DEFAULT_LOG_ROOT = ISAAC_ROOT / "logs"

SOURCE_BASELINE = {
    "isaacsim_develop": {
        "path": ISAAC_ROOT / "sources/IsaacSim-develop",
        "head": "f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc",
    },
    "isaaclab_develop": {
        "path": ISAAC_ROOT / "sources/IsaacLab-develop",
        "head": "f39f5b231b84daab09c1a1ecb1b7efa4bd56340f",
    },
    "isaaclab_release_3_0_0_beta2": {
        "path": ISAAC_ROOT / "sources/IsaacLab-release-3.0.0-beta2",
        "head": "9fe080c1a1c73e8f8a7a8f971f98517876a99861",
    },
    "isaaclab_main": {
        "path": ISAAC_ROOT / "sources/IsaacLab-main-2026-05-25",
        "head": "54a65ea830c6002e17dc18c77831fa60e43937bc",
    },
    "newton_v1_2_0": {
        "path": ISAAC_ROOT / "sources/newton-v1.2.0",
        "head": "a886e3fb411137d8a6ff370a1f3da427eccefbed",
    },
}

OCEANSCALE_VERSION_EXPECTED = {
    "oceanscale": "0.1.0a0",
    "warp-lang": "1.13.0",
    "newton": "1.2.0",
    "torch": "2.11.0+cu128",
    "gymnasium": "1.2.3",
    "isaacsim": None,
    "isaaclab": None,
    "isaaclab_newton": None,
}

ISAAC_VERSION_EXPECTED = {
    "isaacsim": "6.0.0.0",
    "isaacsim-app": "6.0.0.0",
    "isaacsim-kernel": "6.0.0.0",
    "isaaclab": "6.0.0",
    "isaaclab_tasks": "1.10.0",
    "isaaclab_assets": "0.3.4",
    "isaaclab_newton": "0.12.0",
    "isaaclab_experimental": "0.0.5",
    "warp-lang": "1.13.0",
    "newton": "1.0.0",
    "torch": "2.10.0+cu128",
}

OCEANSCALE_TEST_COMMAND = [
    "uv",
    "run",
    "pytest",
    "tests/test_environment.py",
    "tests/test_newton_smoke.py",
    "tests/test_isaaclab_task.py",
    "tests/test_isaaclab_training.py",
    "tests/test_ocean_sim.py",
    "tests/test_rov_env.py",
    "tests/test_version_compat.py",
    "tests/hydro/test_tier1_smoke.py",
    "-q",
]

ISAAC_TEST_COMMAND = [
    str(ISAAC_VENV_PYTHON),
    "-m",
    "pytest",
    "tests/test_isaaclab_task.py",
    "tests/test_isaaclab_training.py",
    "-q",
]


def configure_paths(isaac_root: Path) -> None:
    global ISAAC_ROOT, ISAAC_VENV_PYTHON, DEFAULT_LOG_ROOT
    global SOURCE_BASELINE, ISAAC_TEST_COMMAND

    ISAAC_ROOT = isaac_root.expanduser().resolve()
    ISAAC_VENV_PYTHON = ISAAC_ROOT / "venv/bin/python"
    DEFAULT_LOG_ROOT = ISAAC_ROOT / "logs"
    SOURCE_BASELINE = {
        "isaacsim_develop": {
            "path": ISAAC_ROOT / "sources/IsaacSim-develop",
            "head": "f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc",
        },
        "isaaclab_develop": {
            "path": ISAAC_ROOT / "sources/IsaacLab-develop",
            "head": "f39f5b231b84daab09c1a1ecb1b7efa4bd56340f",
        },
        "isaaclab_release_3_0_0_beta2": {
            "path": ISAAC_ROOT / "sources/IsaacLab-release-3.0.0-beta2",
            "head": "9fe080c1a1c73e8f8a7a8f971f98517876a99861",
        },
        "isaaclab_main": {
            "path": ISAAC_ROOT / "sources/IsaacLab-main-2026-05-25",
            "head": "54a65ea830c6002e17dc18c77831fa60e43937bc",
        },
        "newton_v1_2_0": {
            "path": ISAAC_ROOT / "sources/newton-v1.2.0",
            "head": "a886e3fb411137d8a6ff370a1f3da427eccefbed",
        },
    }
    ISAAC_TEST_COMMAND = [
        str(ISAAC_VENV_PYTHON),
        "-m",
        "pytest",
        "tests/test_isaaclab_task.py",
        "tests/test_isaaclab_training.py",
        "-q",
    ]


@dataclass
class CommandRecord:
    name: str
    command: list[str]
    cwd: str
    returncode: int
    elapsed_seconds: float
    stdout_log: str
    stderr_log: str
    stdout_tail: str
    stderr_tail: str


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run_command(
    name: str,
    command: list[str],
    log_dir: Path,
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    check: bool = True,
) -> CommandRecord:
    start = time.monotonic()
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=process_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = round(time.monotonic() - start, 3)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)
    record = CommandRecord(
        name=name,
        command=command,
        cwd=str(cwd),
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    return record


def load_versions(
    name: str,
    packages: list[str],
    python_command: list[str],
    log_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, str | None], CommandRecord]:
    code = (
        "import importlib.metadata as md, json\n"
        f"packages = {packages!r}\n"
        "versions = {}\n"
        "for package in packages:\n"
        "    try:\n"
        "        versions[package] = md.version(package)\n"
        "    except md.PackageNotFoundError:\n"
        "        versions[package] = None\n"
        "print(json.dumps(versions, sort_keys=True))\n"
    )
    record = run_command(
        name,
        [*python_command, "-c", code],
        log_dir,
        env=env,
        timeout=120,
    )
    lines = [line for line in record.stdout_tail.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{name} did not print version JSON")
    return json.loads(lines[-1]), record


def assert_versions(
    label: str,
    observed: dict[str, str | None],
    expected: dict[str, str | None],
) -> list[str]:
    errors = []
    for package, expected_version in expected.items():
        observed_version = observed.get(package)
        if observed_version != expected_version:
            errors.append(
                f"{label}: {package} expected {expected_version!r}, observed {observed_version!r}"
            )
    return errors


def probe_isaaclab_contract(
    log_dir: Path, env: dict[str, str]
) -> tuple[dict[str, Any], CommandRecord, list[str]]:
    code = (
        "import json\n"
        "import gymnasium as gym\n"
        "from oceanscale.training.isaaclab_env import OceanScaleDirectRLEnv\n"
        "result = {\n"
        "    'isaaclab_direct_rl_env_importable': False,\n"
        "    'isaaclab_direct_rl_abstract_methods': [],\n"
        "    'isaaclab_direct_rl_hooks': [\n"
        "        '_pre_physics_step', '_apply_action', '_get_observations',\n"
        "        '_get_rewards', '_get_dones', '_reset_idx'\n"
        "    ],\n"
        "    'gym_adapter': {\n"
        "        'class': 'OceanScaleDirectRLEnv',\n"
        "        'is_gym_env': issubclass(OceanScaleDirectRLEnv, gym.Env),\n"
        "        'is_isaaclab_direct_rl_env': False,\n"
        "        'mode': 'gym_vector_adapter',\n"
        "    },\n"
        "    'native_task': {\n"
        "        'class': 'OceanScaleTask',\n"
        "        'importable': False,\n"
        "        'is_isaaclab_direct_rl_env': False,\n"
        "        'hooks_present': [],\n"
        "        'hooks_missing': [],\n"
        "        'cfg_is_direct_rl_env_cfg': False,\n"
        "        'cfg_has_validate': False,\n"
        "        'cfg_has_sim': False,\n"
        "        'cfg_has_scene': False,\n"
        "        'instantiation_tested': False,\n"
        "        'mode': 'native_scaffold_unverified',\n"
        "    },\n"
        "}\n"
        "try:\n"
        "    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg\n"
        "except Exception as exc:\n"
        "    result['isaaclab_direct_rl_env_error'] = repr(exc)\n"
        "else:\n"
        "    abstract_methods = sorted(getattr(DirectRLEnv, '__abstractmethods__', ()))\n"
        "    result['isaaclab_direct_rl_env_importable'] = True\n"
        "    result['isaaclab_direct_rl_abstract_methods'] = abstract_methods\n"
        "    result['gym_adapter']['is_isaaclab_direct_rl_env'] = issubclass(OceanScaleDirectRLEnv, DirectRLEnv)\n"
        "    try:\n"
        "        from oceanscale.training.isaaclab_task import OceanScaleTask, OceanScaleTaskCfg\n"
        "    except Exception as exc:\n"
        "        result['native_task']['error'] = repr(exc)\n"
        "    else:\n"
        "        cfg = OceanScaleTaskCfg(num_envs=1)\n"
        "        hooks_present = [\n"
        "            name for name in result['isaaclab_direct_rl_hooks'] if name in OceanScaleTask.__dict__\n"
        "        ]\n"
        "        hooks_missing = [\n"
        "            name for name in result['isaaclab_direct_rl_hooks'] if name not in OceanScaleTask.__dict__\n"
        "        ]\n"
        "        cfg_is_direct = isinstance(cfg, DirectRLEnvCfg)\n"
        "        cfg_has_validate = hasattr(cfg, 'validate')\n"
        "        cfg_has_sim = getattr(cfg, 'sim', None) is not None\n"
        "        cfg_has_scene = getattr(cfg, 'scene', None) is not None\n"
        "        result['native_task'].update({\n"
        "            'importable': True,\n"
        "            'is_isaaclab_direct_rl_env': issubclass(OceanScaleTask, DirectRLEnv),\n"
        "            'hooks_present': hooks_present,\n"
        "            'hooks_missing': hooks_missing,\n"
        "            'cfg_is_direct_rl_env_cfg': cfg_is_direct,\n"
        "            'cfg_has_validate': cfg_has_validate,\n"
        "            'cfg_has_sim': cfg_has_sim,\n"
        "            'cfg_has_scene': cfg_has_scene,\n"
        "        })\n"
        "        if result['native_task']['is_isaaclab_direct_rl_env'] and not hooks_missing:\n"
        "            if cfg_is_direct and cfg_has_validate and cfg_has_sim and cfg_has_scene:\n"
        "                result['native_task']['mode'] = 'isaaclab_native_direct_rl_cfg_ready'\n"
        "            else:\n"
        "                result['native_task']['mode'] = 'isaaclab_native_direct_rl_env_scaffold'\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    record = run_command(
        "isaaclab_contract_probe",
        [str(ISAAC_VENV_PYTHON), "-c", code],
        log_dir,
        env={**env, "PYTHONPATH": str(REPO_ROOT)},
        timeout=120,
    )
    lines = [line for line in record.stdout_tail.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("isaaclab_contract_probe did not print JSON")
    result = json.loads(lines[-1])
    warnings = []
    if result.get("isaaclab_direct_rl_env_importable") and not result.get("gym_adapter", {}).get(
        "is_isaaclab_direct_rl_env"
    ):
        warnings.append(
            "OceanScaleDirectRLEnv is a Gymnasium vector adapter, not an Isaac Lab DirectRLEnv subclass."
        )
    native_task = result.get("native_task", {})
    if native_task.get("mode") == "isaaclab_native_direct_rl_env_scaffold":
        warnings.append(
            "OceanScaleTask is a DirectRLEnv subclass scaffold, but its cfg is not runtime-ready DirectRLEnvCfg."
        )
    if native_task.get("mode") == "isaaclab_native_direct_rl_cfg_ready":
        warnings.append(
            "OceanScaleTask has a DirectRLEnvCfg-ready config, but native environment instantiation is not tested yet."
        )
    return result, record, warnings


def probe_native_task_runtime(
    log_dir: Path, env: dict[str, str]
) -> tuple[dict[str, Any], CommandRecord, list[str]]:
    code = r'''
from __future__ import annotations

import json
import sys
import traceback

from isaaclab.app import AppLauncher

launcher = AppLauncher({
    "headless": True,
    "device": "cuda:0",
    "visualizer": ["none"],
    "enable_cameras": False,
})
simulation_app = launcher.app
result = {
    "app_started": True,
    "env_created": False,
    "reset_done": False,
    "steps_done": 0,
    "num_envs": 16,
    "num_steps": 64,
    "action_scale": 0.35,
    "obs_shape": None,
    "reward_shape": None,
    "terminated_shape": None,
    "truncated_shape": None,
    "obs_abs_max": None,
    "reward_min": None,
    "reward_max": None,
    "terminated_count": 0,
    "truncated_count": 0,
    "passed": False,
}

try:
    import torch
    from oceanscale.training.isaaclab_task import OceanScaleTask, OceanScaleTaskCfg

    torch.manual_seed(20260525)
    cfg = OceanScaleTaskCfg(num_envs=result["num_envs"], physics_dt=1.0 / 120.0, decimation=2)
    cfg.seed = 20260525
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = result["num_envs"]
    cfg.init_pos_noise = 0.05

    env = OceanScaleTask(cfg)
    result["env_created"] = True
    obs, _ = env.reset()
    result["reset_done"] = True
    obs_policy = obs["policy"]
    if list(obs_policy.shape) != [result["num_envs"], 20]:
        raise RuntimeError(f"unexpected obs shape: {list(obs_policy.shape)}")
    if not torch.isfinite(obs_policy).all():
        raise RuntimeError("non-finite observation after reset")

    obs_abs_max = float(obs_policy.abs().max().item())
    reward_min = None
    reward_max = None
    terminated_count = 0
    truncated_count = 0

    for _ in range(result["num_steps"]):
        actions = (
            (torch.rand((env.num_envs, 6), device=env.device) * 2.0 - 1.0)
            * result["action_scale"]
        )
        obs, reward, terminated, truncated, _ = env.step(actions)
        obs_policy = obs["policy"]
        if not torch.isfinite(obs_policy).all():
            raise RuntimeError(f"non-finite observation at step {result['steps_done'] + 1}")
        if not torch.isfinite(reward).all():
            raise RuntimeError(f"non-finite reward at step {result['steps_done'] + 1}")
        reward_min = (
            float(reward.min().item())
            if reward_min is None
            else min(reward_min, float(reward.min().item()))
        )
        reward_max = (
            float(reward.max().item())
            if reward_max is None
            else max(reward_max, float(reward.max().item()))
        )
        obs_abs_max = max(obs_abs_max, float(obs_policy.abs().max().item()))
        terminated_count += int(terminated.sum().item())
        truncated_count += int(truncated.sum().item())
        result["steps_done"] += 1
    result.update({
        "obs_shape": list(obs_policy.shape),
        "reward_shape": list(reward.shape),
        "terminated_shape": list(terminated.shape),
        "truncated_shape": list(truncated.shape),
        "obs_abs_max": obs_abs_max,
        "reward_min": reward_min,
        "reward_max": reward_max,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "passed": True,
    })
    print("OCEANSCALE_NATIVE_TASK_STRESS_PASS", flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    env.close()
except BaseException as exc:
    result["error"] = repr(exc)
    result["traceback"] = traceback.format_exc()
    print("OCEANSCALE_NATIVE_TASK_STRESS_FAIL", flush=True)
    print(result["traceback"], flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    sys.exit(1)
finally:
    sys.stdout.flush()
    simulation_app.close()
'''
    record = run_command(
        "oceanscale_native_task_runtime_smoke",
        [str(ISAAC_VENV_PYTHON), "-c", code],
        log_dir,
        env={**env, "PYTHONPATH": str(REPO_ROOT)},
        timeout=240,
        check=False,
    )
    stdout = Path(record.stdout_log).read_text(encoding="utf-8")
    lines = [line for line in stdout.splitlines() if line.strip()]
    json_lines = [line for line in lines if line.startswith("{") and line.endswith("}")]
    result = json.loads(json_lines[-1]) if json_lines else {"passed": False}
    result["marker_found"] = "OCEANSCALE_NATIVE_TASK_STRESS_PASS" in stdout
    result["returncode"] = record.returncode

    errors = []
    if record.returncode != 0 or not result.get("passed") or not result["marker_found"]:
        errors.append("oceanscale_native_task_runtime_smoke: native random-action stress failed")
    return result, record, errors


def check_sources(log_dir: Path) -> tuple[dict[str, Any], list[CommandRecord], list[str]]:
    results: dict[str, Any] = {}
    records: list[CommandRecord] = []
    errors: list[str] = []
    for name, baseline in SOURCE_BASELINE.items():
        path = Path(baseline["path"])
        expected_head = str(baseline["head"])
        if not path.exists():
            errors.append(f"{name}: missing source tree at {path}")
            results[name] = {"path": str(path), "status": "missing"}
            continue
        head_record = run_command(
            f"source_{name}_head",
            ["git", "rev-parse", "HEAD"],
            log_dir,
            cwd=path,
        )
        status_record = run_command(
            f"source_{name}_status",
            ["git", "status", "--short", "--branch"],
            log_dir,
            cwd=path,
        )
        records.extend([head_record, status_record])
        observed_head = head_record.stdout_tail.strip()
        status = status_record.stdout_tail.strip()
        dirty_lines = [
            line
            for line in status.splitlines()
            if line and not line.startswith("## ")
        ]
        if observed_head != expected_head:
            errors.append(
                f"{name}: expected head {expected_head}, observed {observed_head}"
            )
        if dirty_lines:
            errors.append(f"{name}: source tree is dirty: {dirty_lines!r}")
        results[name] = {
            "path": str(path),
            "expected_head": expected_head,
            "observed_head": observed_head,
            "status": status,
            "clean": not dirty_lines,
        }
    return results, records, errors


def build_report(output_dir: Path, skip_tests: bool) -> tuple[dict[str, Any], bool]:
    command_records: list[CommandRecord] = []
    errors: list[str] = []
    warnings: list[str] = []

    sources, source_records, source_errors = check_sources(output_dir)
    command_records.extend(source_records)
    errors.extend(source_errors)

    oceanscale_versions, record = load_versions(
        "oceanscale_dev_versions",
        list(OCEANSCALE_VERSION_EXPECTED),
        ["uv", "run", "python"],
        output_dir,
    )
    command_records.append(record)
    errors.extend(
        assert_versions(
            "oceanscale_dev_versions",
            oceanscale_versions,
            OCEANSCALE_VERSION_EXPECTED,
        )
    )

    isaac_env = {
        "PYTHONNOUSERSITE": "1",
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    isaac_versions, record = load_versions(
        "isaac_validation_versions",
        list(ISAAC_VERSION_EXPECTED),
        [str(ISAAC_VENV_PYTHON)],
        output_dir,
        env=isaac_env,
    )
    command_records.append(record)
    errors.extend(
        assert_versions(
            "isaac_validation_versions",
            isaac_versions,
            ISAAC_VERSION_EXPECTED,
        )
    )

    isaaclab_contract, record, contract_warnings = probe_isaaclab_contract(
        output_dir, isaac_env
    )
    command_records.append(record)
    warnings.extend(contract_warnings)

    test_results: dict[str, str] = {"status": "skipped"}
    native_task_runtime: dict[str, Any] = {"status": "skipped"}
    if not skip_tests:
        native_task_runtime, record, runtime_errors = probe_native_task_runtime(
            output_dir, isaac_env
        )
        command_records.append(record)
        errors.extend(runtime_errors)
        isaaclab_contract["native_task"]["instantiation_tested"] = bool(
            native_task_runtime.get("passed")
        )
        if native_task_runtime.get("passed"):
            warnings = [
                warning
                for warning in warnings
                if "native environment instantiation is not tested yet" not in warning
            ]

        for name, command, env in [
            ("oceanscale_core_tests", OCEANSCALE_TEST_COMMAND, None),
            (
                "isaaclab_adapter_tests",
                ISAAC_TEST_COMMAND,
                {
                    **isaac_env,
                    "PYTHONPATH": str(REPO_ROOT),
                },
            ),
        ]:
            try:
                command_records.append(
                    run_command(
                        name,
                        command,
                        output_dir,
                        env=env,
                        timeout=600,
                    )
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        test_results = {
            record.name: "pass" if record.returncode == 0 else "fail"
            for record in command_records
            if record.name
            in {
                "oceanscale_core_tests",
                "isaaclab_adapter_tests",
                "oceanscale_native_task_runtime_smoke",
            }
        }
        test_results["oceanscale_native_task_runtime_smoke"] = (
            "pass" if native_task_runtime.get("passed") else "fail"
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(REPO_ROOT),
        "isaac_root": str(ISAAC_ROOT),
        "output_dir": str(output_dir),
        "scope": "OceanScale latest Isaac ecosystem baseline, not full official demo sweep",
        "sources": sources,
        "versions": {
            "oceanscale_dev": oceanscale_versions,
            "isaac_validation": isaac_versions,
        },
        "isaaclab_contract": isaaclab_contract,
        "native_task_runtime": native_task_runtime,
        "tests": test_results,
        "commands": [asdict(record) for record in command_records],
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    return report, not errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the OceanScale latest-Isaac baseline without running the full "
            "official Isaac demo sweep."
        )
    )
    parser.add_argument(
        "--isaac-root",
        type=Path,
        default=DEFAULT_ISAAC_ROOT,
        help=(
            "Isaac root containing venv/, logs/, and sources/. Defaults to $ISAAC_ROOT "
            "or /mnt/storage/isaacsim-6.0-official."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for logs and summary.json.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Check source heads and package versions only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_paths(args.isaac_root)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        output_dir = DEFAULT_LOG_ROOT / f"oceanscale-latest-isaac-baseline-{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report, passed = build_report(output_dir, args.skip_tests)
    summary_path = output_dir / "summary.json"
    write_text(summary_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "summary": str(summary_path)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
