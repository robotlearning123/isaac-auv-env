"""Evaluation profile definitions for OceanScale public-readiness gates."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

type Scalar = bool | int | float | str
type RunMetadata = dict[str, object]


@dataclass(frozen=True)
class EvaluationAxis:
    category: str
    name: str
    parameter: str
    unit: str
    baseline: Scalar
    values: tuple[Scalar, ...]
    rationale: str


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    description: str
    task_family: str
    episodes_per_case: int
    seed: int
    axes: tuple[EvaluationAxis, ...]
    metrics: tuple[str, ...]
    claim_guardrails: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    axis: str
    parameter: str
    value: Scalar | None
    unit: str
    episodes: int
    seed: int
    overrides: dict[str, Scalar]


@dataclass(frozen=True)
class EvaluationRecord:
    case_id: str
    task_name: str
    action_mode: str
    status: str
    episodes: int
    seed: int
    overrides: dict[str, Scalar]
    applied_overrides: dict[str, Scalar]
    unsupported_overrides: dict[str, Scalar]
    metrics: dict[str, float | int]
    wall_time_s: float
    run_metadata: RunMetadata
    environment_config: dict[str, object] = field(default_factory=dict)
    error: str | None = None


GENESIS_WORLD_REFERENCE = {
    "name": "Genesis World 1.0 evaluation-first pattern",
    "url": "https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward",
    "adoption": "method_reference_only",
    "dependency": "none",
}

CaseEvaluator = Callable[[EvaluationCase], dict[str, object]]
ACTION_MODES = ("zero", "proportional", "damped")


def underwater_robustness_profile(
    episodes_per_case: int = 20,
    seed: int = 42,
) -> EvaluationProfile:
    """Return the default evaluation-first profile for underwater robotics."""
    if episodes_per_case < 1:
        raise ValueError("episodes_per_case must be >= 1")

    axes = (
        EvaluationAxis(
            category="ocean",
            name="current strength",
            parameter="current_speed_mps",
            unit="m/s",
            baseline=0.2,
            values=(0.0, 0.5, 1.0, 2.0),
            rationale="Field trials are dominated by current uncertainty; this is the first robustness axis.",
        ),
        EvaluationAxis(
            category="ocean",
            name="current direction",
            parameter="current_direction_deg",
            unit="deg",
            baseline=0,
            values=(0, 45, 90, 180),
            rationale="AUV/ROV controllers should not only work with head-on or aligned currents.",
        ),
        EvaluationAxis(
            category="ocean",
            name="water density",
            parameter="water_density_kg_m3",
            unit="kg/m^3",
            baseline=1025.0,
            values=(1020.0, 1025.0, 1030.0),
            rationale="Salinity and temperature shift buoyancy and hydrodynamic response.",
        ),
        EvaluationAxis(
            category="vehicle",
            name="mass scale",
            parameter="mass_scale",
            unit="ratio",
            baseline=1.0,
            values=(0.9, 1.0, 1.1),
            rationale="Payload and ballast changes are common between field deployments.",
        ),
        EvaluationAxis(
            category="vehicle",
            name="drag scale",
            parameter="drag_scale",
            unit="ratio",
            baseline=1.0,
            values=(0.85, 1.0, 1.15),
            rationale="Drag uncertainty is one of the main sim-to-real gap sources for underwater vehicles.",
        ),
        EvaluationAxis(
            category="vehicle",
            name="thruster efficiency",
            parameter="thruster_efficiency_scale",
            unit="ratio",
            baseline=1.0,
            values=(0.8, 0.9, 1.0),
            rationale="Thruster wear, battery state, and fouling change actuation authority.",
        ),
        EvaluationAxis(
            category="sensor",
            name="visibility",
            parameter="visibility_m",
            unit="m",
            baseline=20.0,
            values=(30.0, 10.0, 3.0),
            rationale="Turbidity is the underwater analogue of visual-domain perturbation.",
        ),
        EvaluationAxis(
            category="sensor",
            name="DVL dropout",
            parameter="dvl_dropout_probability",
            unit="probability",
            baseline=0.0,
            values=(0.0, 0.05, 0.2),
            rationale="Bottom lock and water-track dropouts strongly affect localization robustness.",
        ),
        EvaluationAxis(
            category="sensor",
            name="IMU gyro bias",
            parameter="imu_gyro_bias_deg_s",
            unit="deg/s",
            baseline=0.0,
            values=(0.0, 0.1, 0.5),
            rationale="Bias sensitivity is a practical pre-deployment sensor-health check.",
        ),
        EvaluationAxis(
            category="task",
            name="target offset",
            parameter="target_offset_m",
            unit="m",
            baseline=0.0,
            values=(0.0, 0.25, 0.5, 1.0),
            rationale="Docking and station-keeping should be evaluated against placement uncertainty.",
        ),
        EvaluationAxis(
            category="task",
            name="obstacle count",
            parameter="obstacle_count",
            unit="count",
            baseline=0,
            values=(0, 3, 8),
            rationale="Scene layout perturbations catch brittle policies that overfit one clean scenario.",
        ),
    )

    return EvaluationProfile(
        name="underwater_robustness_v0",
        description=(
            "A controlled evaluation profile for testing underwater robot policies across ocean, "
            "vehicle, sensor, and task perturbations before field testing."
        ),
        task_family="station_keeping_docking_waypoint",
        episodes_per_case=episodes_per_case,
        seed=seed,
        axes=axes,
        metrics=(
            "success_rate",
            "episode_return",
            "mean_position_error_m",
            "max_position_error_m",
            "energy_proxy",
            "constraint_violations",
            "wall_time_s",
        ),
        claim_guardrails=(
            "Protocol generation does not prove sim-to-real transfer.",
            "No sim-to-real claim is valid until a real underwater robot experiment reports a quantified transfer gap.",
            "Genesis World is a method reference only; OceanScale does not add Genesis, Quadrants, or Nyx as runtime dependencies.",
            "Report measured results only after running rollouts with fixed seeds and recorded hardware.",
        ),
    )


def generate_evaluation_cases(profile: EvaluationProfile) -> list[EvaluationCase]:
    cases = [
        EvaluationCase(
            case_id=f"{profile.name}:baseline",
            category="baseline",
            axis="baseline",
            parameter="",
            value=None,
            unit="",
            episodes=profile.episodes_per_case,
            seed=profile.seed,
            overrides={},
        )
    ]

    for axis in profile.axes:
        for value in axis.values:
            cases.append(
                EvaluationCase(
                    case_id=f"{profile.name}:{axis.category}:{_slug(axis.name)}:{_value_slug(value)}",
                    category=axis.category,
                    axis=axis.name,
                    parameter=axis.parameter,
                    value=value,
                    unit=axis.unit,
                    episodes=profile.episodes_per_case,
                    seed=profile.seed + len(cases),
                    overrides={axis.parameter: value},
                )
            )

    return cases


def run_evaluation_profile(
    profile: EvaluationProfile,
    task_name: str,
    evaluate_case: CaseEvaluator,
    case_limit: int | None = None,
    action_mode: str = "zero",
    run_metadata: RunMetadata | None = None,
) -> list[EvaluationRecord]:
    cases = generate_evaluation_cases(profile)
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be >= 1")
        cases = cases[:case_limit]

    metadata = dict(run_metadata or {"task_name": task_name, "action_mode": action_mode})
    records: list[EvaluationRecord] = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            result = evaluate_case(case)
            metrics = _numeric_metrics(result.get("metrics", {}))
            applied_overrides = _scalar_map(result.get("applied_overrides", {}))
            unsupported_overrides = _scalar_map(result.get("unsupported_overrides", {}))
            environment_config = _case_environment_config(
                result.get("environment_config", {}),
                requested_overrides=case.overrides,
                applied_overrides=applied_overrides,
                unsupported_overrides=unsupported_overrides,
            )
            wall_time_s = float(metrics.get("wall_time_s", time.perf_counter() - t0))
            records.append(
                EvaluationRecord(
                    case_id=case.case_id,
                    task_name=task_name,
                    action_mode=action_mode,
                    status=str(result.get("status", "completed")),
                    episodes=case.episodes,
                    seed=case.seed,
                    overrides=case.overrides,
                    applied_overrides=applied_overrides,
                    unsupported_overrides=unsupported_overrides,
                    metrics=metrics,
                    wall_time_s=wall_time_s,
                    run_metadata=metadata,
                    environment_config=environment_config,
                )
            )
        except Exception as exc:
            records.append(
                EvaluationRecord(
                    case_id=case.case_id,
                    task_name=task_name,
                    action_mode=action_mode,
                    status="failed",
                    episodes=case.episodes,
                    seed=case.seed,
                    overrides=case.overrides,
                    applied_overrides={},
                    unsupported_overrides=case.overrides,
                    metrics={},
                    wall_time_s=time.perf_counter() - t0,
                    run_metadata=metadata,
                    environment_config=_case_environment_config(
                        {},
                        requested_overrides=case.overrides,
                        applied_overrides={},
                        unsupported_overrides=case.overrides,
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return records


def summarize_evaluation_run(
    profile: EvaluationProfile,
    records: list[EvaluationRecord],
    task_name: str,
    action_mode: str = "zero",
    run_metadata: RunMetadata | None = None,
) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    metric_values: dict[str, list[float]] = {}
    unsupported_parameters: set[str] = set()
    environment_configs = _unique_dicts(
        _case_environment_config(
            record.environment_config,
            requested_overrides=record.overrides,
            applied_overrides=record.applied_overrides,
            unsupported_overrides=record.unsupported_overrides,
        )
        for record in records
    )

    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
        unsupported_parameters.update(record.unsupported_overrides)
        for key, value in record.metrics.items():
            metric_values.setdefault(key, []).append(float(value))

    mean_metrics = {
        key: sum(values) / len(values) for key, values in metric_values.items() if values
    }
    metric_distributions = _metric_distributions(metric_values)

    return {
        "profile": profile.name,
        "task_name": task_name,
        "action_mode": action_mode,
        "case_count": len(records),
        "status_counts": status_counts,
        "mean_metrics": mean_metrics,
        "metric_distributions": metric_distributions,
        "unsupported_parameters": sorted(unsupported_parameters),
        "environment_configs": environment_configs,
        "run_metadata": dict(run_metadata or (records[0].run_metadata if records else {})),
        "source_reference": GENESIS_WORLD_REFERENCE,
        "claim_guardrails": list(profile.claim_guardrails),
    }


def records_to_jsonl(records: list[EvaluationRecord]) -> str:
    return "".join(json.dumps(_record_to_dict(record), sort_keys=True) + "\n" for record in records)


def evaluation_records_from_jsonl(jsonl_text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSONL at line {line_number}: expected object")
        records.append(value)
    return records


def summarize_evaluation_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("evaluation report requires at least one record")

    status_counts: dict[str, int] = {}
    metric_values: dict[str, list[float]] = {}
    unsupported_parameters: set[str] = set()
    failed_cases: list[dict[str, str | None]] = []

    first_record = records[0] if records else {}
    environment_configs = _unique_dicts(
        _case_environment_config(
            record.get("environment_config", {}),
            requested_overrides=_scalar_map(record.get("overrides", {})),
            applied_overrides=_scalar_map(record.get("applied_overrides", {})),
            unsupported_overrides=_scalar_map(record.get("unsupported_overrides", {})),
        )
        for record in records
    )
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        unsupported = record.get("unsupported_overrides", {})
        if isinstance(unsupported, dict):
            unsupported_parameters.update(str(key) for key in unsupported)

        metrics = record.get("metrics", {})
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(key, str) and isinstance(value, bool | int | float):
                    metric_values.setdefault(key, []).append(float(value))

        if status != "completed":
            failed_cases.append(
                {
                    "case_id": str(record.get("case_id", "")),
                    "status": status,
                    "error": _optional_string(record.get("error")),
                }
            )

    mean_metrics = {
        key: sum(values) / len(values) for key, values in metric_values.items() if values
    }
    metric_distributions = _metric_distributions(metric_values)

    return {
        "profile": _profile_from_case_id(_optional_string(first_record.get("case_id"))),
        "task_name": _optional_string(first_record.get("task_name")),
        "action_mode": _optional_string(first_record.get("action_mode")),
        "case_count": len(records),
        "status_counts": status_counts,
        "mean_metrics": mean_metrics,
        "metric_distributions": metric_distributions,
        "unsupported_parameters": sorted(unsupported_parameters),
        "failed_cases": failed_cases,
        "environment_configs": environment_configs,
        "run_metadata": _dict_or_empty(first_record.get("run_metadata")),
        "source_reference": GENESIS_WORLD_REFERENCE,
        "claim_guardrails": [
            "This report is not a sim-to-real result.",
            "Failed and unsupported cases must remain visible in public summaries.",
        ],
    }


def evaluation_report_to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# OceanScale Evaluation Report",
        "",
        "This report summarizes measured rollout records. It is not a sim-to-real result.",
        "",
        "## Run",
        "",
        f"- Profile: {_markdown_value(report.get('profile'))}",
        f"- Task: {_markdown_value(report.get('task_name'))}",
        f"- Action mode: {_markdown_value(report.get('action_mode'))}",
        f"- Cases: {_markdown_value(report.get('case_count'))}",
        "",
        "## Status Counts",
        "",
    ]
    status_counts = _dict_or_empty(report.get("status_counts"))
    if status_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(status_counts.items()))
    else:
        lines.append("- No records")

    lines.extend(["", "## Mean Metrics", ""])
    mean_metrics = _dict_or_empty(report.get("mean_metrics"))
    if mean_metrics:
        lines.extend(f"- {key}: {value}" for key, value in sorted(mean_metrics.items()))
    else:
        lines.append("- No numeric metrics")

    lines.extend(["", "## Metric Distributions", ""])
    distributions = _dict_or_empty(report.get("metric_distributions"))
    if distributions:
        for key, value in sorted(distributions.items()):
            if isinstance(value, dict):
                fields = ", ".join(
                    f"{item_key}={item_value}" for item_key, item_value in value.items()
                )
                lines.append(f"- {key}: {fields}")
    else:
        lines.append("- No numeric metrics")

    lines.extend(["", "## Unsupported Parameters", ""])
    unsupported = report.get("unsupported_parameters")
    if isinstance(unsupported, list) and unsupported:
        lines.extend(f"- {item}" for item in unsupported)
    else:
        lines.append("- None")

    lines.extend(["", "## Failed Cases", ""])
    failed_cases = report.get("failed_cases")
    if isinstance(failed_cases, list) and failed_cases:
        for item in failed_cases:
            if isinstance(item, dict):
                case_id = _markdown_value(item.get("case_id"))
                status = _markdown_value(item.get("status"))
                error = _markdown_value(item.get("error"))
                lines.append(f"- {case_id}: {status}; {error}")
    else:
        lines.append("- None")

    lines.extend(["", "## Environment Configs", ""])
    environment_configs = report.get("environment_configs")
    if isinstance(environment_configs, list) and environment_configs:
        for index, config in enumerate(environment_configs, start=1):
            if isinstance(config, dict):
                lines.append(f"### Config {index}")
                lines.extend(f"- {key}: {value}" for key, value in _flatten_mapping(config))
    else:
        lines.append("- Not recorded")

    lines.extend(["", "## Run Metadata", ""])
    metadata = _dict_or_empty(report.get("run_metadata"))
    if metadata:
        lines.extend(f"- {key}: {value}" for key, value in _flatten_mapping(metadata))
    else:
        lines.append("- Not recorded")

    lines.extend(["", "## Source Reference", ""])
    source_reference = _dict_or_empty(report.get("source_reference"))
    if source_reference:
        lines.extend(f"- {key}: {value}" for key, value in _flatten_mapping(source_reference))
    else:
        lines.append("- Not recorded")

    lines.extend(["", "## Claim Guardrails", ""])
    guardrails = report.get("claim_guardrails")
    if isinstance(guardrails, list) and guardrails:
        lines.extend(f"- {item}" for item in guardrails)
    else:
        lines.append("- Not recorded")

    return "\n".join(lines) + "\n"


def collect_run_metadata(
    task_name: str,
    action_mode: str,
    device: str,
    steps_per_episode: int,
    *,
    profile_name: str | None = None,
    profile_seed: int | None = None,
    episodes_per_case: int | None = None,
    case_limit: int | None = None,
) -> RunMetadata:
    if steps_per_episode < 1:
        raise ValueError("steps_per_episode must be >= 1")

    git_revision = _git_stdout(("rev-parse", "--short", "HEAD"))
    dirty_state = _git_status_short()
    gpu_inventory = _gpu_inventory()
    metadata: RunMetadata = {
        "task_name": task_name,
        "action_mode": action_mode,
        "device": device,
        "steps_per_episode": steps_per_episode,
        "profile_name": profile_name,
        "profile_seed": profile_seed,
        "episodes_per_case": episodes_per_case,
        "case_limit": case_limit,
        "git_revision": git_revision,
        "git_dirty": bool(dirty_state),
        "git_dirty_known": dirty_state is not None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "package_versions": _package_versions(),
    }
    metadata.update(gpu_inventory)
    return metadata


def resolve_case_evaluator(
    task_name: str,
    action_mode: str,
    steps_per_episode: int,
    device: str,
) -> CaseEvaluator:
    if action_mode not in ACTION_MODES:
        modes = ", ".join(ACTION_MODES)
        raise ValueError(f"unknown action mode {action_mode!r}; expected one of: {modes}")

    if task_name == "station-keeping":
        return lambda case: evaluate_station_keeping_case(
            case,
            action_mode=action_mode,
            steps_per_episode=steps_per_episode,
            device=device,
        )
    if task_name == "docking":
        return lambda case: evaluate_docking_case(
            case,
            action_mode=action_mode,
            steps_per_episode=steps_per_episode,
            device=device,
        )
    if task_name == "waypoint":
        return lambda case: evaluate_waypoint_case(
            case,
            action_mode=action_mode,
            steps_per_episode=steps_per_episode,
            device=device,
        )
    raise ValueError(f"unknown task {task_name!r}")


def resolve_zero_action_evaluator(
    task_name: str,
    steps_per_episode: int,
    device: str,
) -> CaseEvaluator:
    return resolve_case_evaluator(
        task_name,
        action_mode="zero",
        steps_per_episode=steps_per_episode,
        device=device,
    )


def evaluate_station_keeping_case(
    case: EvaluationCase,
    action_mode: str = "zero",
    steps_per_episode: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    if steps_per_episode < 1:
        raise ValueError("steps_per_episode must be >= 1")

    import numpy as np
    import torch

    from oceanscale.envs.station_keeping_env import CurrentStationKeepingEnv

    supported = {"current_speed_mps", "current_direction_deg", "target_offset_m"}
    unsupported_overrides = {
        key: value for key, value in case.overrides.items() if key not in supported
    }
    applied_overrides: dict[str, Scalar] = {}
    environment_config = {
        "env_class": "CurrentStationKeepingEnv",
        "task_name": "station-keeping",
        "action_mode": action_mode,
        "device": device,
        "n_envs": 1,
        "decimation": 1,
        "max_episode_steps": steps_per_episode,
        "sensor_noise_std": 0.0,
        "init_pos_noise_std": 0.0,
        "init_yaw_noise_std": 0.0,
        "current_speed": 0.0,
        "supported_overrides": sorted(supported),
    }

    env = CurrentStationKeepingEnv(
        n_envs=1,
        device=device,
        decimation=1,
        max_episode_steps=steps_per_episode,
        sensor_noise_std=0.0,
        init_pos_noise_std=0.0,
        init_yaw_noise_std=0.0,
        current_speed=0.0,
    )
    environment_config["observation_dim"] = _space_dimension(env.observation_space, default=29)
    environment_config["action_dim"] = _space_dimension(env.action_space, default=6)
    returns: list[float] = []
    position_errors: list[float] = []
    successes = 0
    energy_proxy = 0.0
    constraint_violations = 0
    t0 = time.perf_counter()

    try:
        for episode in range(case.episodes):
            obs, _ = env.reset(seed=case.seed + episode)
            _apply_station_keeping_overrides(env, case, applied_overrides, torch)
            obs = env._get_obs()

            total_reward = 0.0
            success = False
            for _ in range(steps_per_episode):
                action = _action_from_observation(
                    obs,
                    action_mode,
                    np,
                    task_name="station-keeping",
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                obs_arr = np.asarray(obs, dtype=np.float32)
                position_errors.append(float(np.linalg.norm(obs_arr[:3])))
                energy_proxy += float(np.dot(action, action))
                if not np.all(np.isfinite(obs_arr)):
                    constraint_violations += 1
                success = success or bool(info.get("success", False))
                if bool(terminated) or bool(truncated):
                    break

            returns.append(total_reward)
            successes += int(success)
    finally:
        env.close()

    wall_time_s = time.perf_counter() - t0
    mean_error = float(np.mean(position_errors)) if position_errors else 0.0
    max_error = float(np.max(position_errors)) if position_errors else 0.0

    return {
        "status": "completed",
        "metrics": {
            "success_rate": successes / max(case.episodes, 1),
            "episode_return": float(np.mean(returns)) if returns else 0.0,
            "mean_position_error_m": mean_error,
            "max_position_error_m": max_error,
            "energy_proxy": energy_proxy,
            "constraint_violations": constraint_violations,
            "wall_time_s": wall_time_s,
        },
        "applied_overrides": applied_overrides,
        "unsupported_overrides": unsupported_overrides,
        "environment_config": environment_config,
    }


def evaluate_station_keeping_zero_action_case(
    case: EvaluationCase,
    steps_per_episode: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    return evaluate_station_keeping_case(
        case,
        action_mode="zero",
        steps_per_episode=steps_per_episode,
        device=device,
    )


def evaluate_docking_case(
    case: EvaluationCase,
    action_mode: str = "zero",
    steps_per_episode: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    if steps_per_episode < 1:
        raise ValueError("steps_per_episode must be >= 1")

    import numpy as np
    import torch

    from oceanscale.envs.docking_env import DockingApproachEnv, DockingApproachEnvCfg

    supported = {"current_speed_mps", "current_direction_deg", "target_offset_m"}
    unsupported_overrides = {
        key: value for key, value in case.overrides.items() if key not in supported
    }
    applied_overrides: dict[str, Scalar] = {}
    environment_config = {
        "env_class": "DockingApproachEnv",
        "config_class": "DockingApproachEnvCfg",
        "task_name": "docking",
        "action_mode": action_mode,
        "device": device,
        "n_envs": 1,
        "decimation": 1,
        "max_episode_steps": steps_per_episode,
        "sensor_noise_std": 0.0,
        "dock_offset_range": 0.0,
        "current_speed_max": 0.0,
        "init_vel_max": 0.0,
        "supported_overrides": sorted(supported),
    }
    env = DockingApproachEnv(
        DockingApproachEnvCfg(
            n_envs=1,
            device=device,
            decimation=1,
            max_episode_steps=steps_per_episode,
            sensor_noise_std=0.0,
            dock_offset_range=0.0,
            current_speed_max=0.0,
            init_vel_max=0.0,
        )
    )
    environment_config["observation_dim"] = _space_dimension(env.observation_space, default=31)
    environment_config["action_dim"] = _space_dimension(env.action_space, default=6)
    returns: list[float] = []
    position_errors: list[float] = []
    successes = 0
    energy_proxy = 0.0
    constraint_violations = 0
    t0 = time.perf_counter()

    try:
        for episode in range(case.episodes):
            obs, _ = env.reset(seed=case.seed + episode)
            _apply_docking_overrides(env, case, applied_overrides, torch)
            obs = env._get_observations()

            total_reward = 0.0
            success = False
            for _ in range(steps_per_episode):
                action = _action_from_observation(
                    obs,
                    action_mode,
                    np,
                    task_name="docking",
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                position_errors.append(_first_float(info.get("range_to_dock"), default=0.0))
                obs_arr = _as_numpy(obs)
                energy_proxy += float(np.dot(action, action))
                if not np.all(np.isfinite(obs_arr)):
                    constraint_violations += 1
                success = success or _success_from_info(info)
                if bool(terminated) or bool(truncated):
                    break

            returns.append(total_reward)
            successes += int(success)
    finally:
        env.close()

    wall_time_s = time.perf_counter() - t0
    return _rollout_result(
        case=case,
        returns=returns,
        position_errors=position_errors,
        successes=successes,
        energy_proxy=energy_proxy,
        constraint_violations=constraint_violations,
        wall_time_s=wall_time_s,
        applied_overrides=applied_overrides,
        unsupported_overrides=unsupported_overrides,
        environment_config=environment_config,
    )


def evaluate_docking_zero_action_case(
    case: EvaluationCase,
    steps_per_episode: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    return evaluate_docking_case(
        case,
        action_mode="zero",
        steps_per_episode=steps_per_episode,
        device=device,
    )


def evaluate_waypoint_case(
    case: EvaluationCase,
    action_mode: str = "zero",
    steps_per_episode: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    if steps_per_episode < 1:
        raise ValueError("steps_per_episode must be >= 1")

    import numpy as np
    import torch

    from oceanscale.envs.waypoint_env import WaypointFollowingEnv

    supported = {"target_offset_m"}
    unsupported_overrides = {
        key: value for key, value in case.overrides.items() if key not in supported
    }
    applied_overrides: dict[str, Scalar] = {}
    time_limit_s = max(steps_per_episode, 1) / 240.0
    environment_config = {
        "env_class": "WaypointFollowingEnv",
        "task_name": "waypoint",
        "action_mode": action_mode,
        "device": device,
        "n_envs": 1,
        "n_waypoints": 1,
        "time_limit_s": time_limit_s,
        "decimation": 1,
        "sensor_noise_std": 0.0,
        "supported_overrides": sorted(supported),
    }
    env = WaypointFollowingEnv(
        n_envs=1,
        n_waypoints=1,
        time_limit=time_limit_s,
        decimation=1,
        sensor_noise_std=0.0,
        device=device,
    )
    environment_config["observation_dim"] = _space_dimension(env.observation_space, default=29)
    environment_config["action_dim"] = _space_dimension(env.action_space, default=6)
    returns: list[float] = []
    position_errors: list[float] = []
    successes = 0
    energy_proxy = 0.0
    constraint_violations = 0
    t0 = time.perf_counter()

    try:
        for episode in range(case.episodes):
            obs, _ = env.reset(seed=case.seed + episode)
            _apply_waypoint_overrides(env, case, applied_overrides, torch)
            obs = env._get_obs()

            total_reward = 0.0
            success = False
            for _ in range(steps_per_episode):
                action = _action_from_observation(
                    obs,
                    action_mode,
                    np,
                    task_name="waypoint",
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                position_errors.append(_first_float(info.get("distance_to_wp"), default=0.0))
                obs_arr = _as_numpy(obs)
                energy_proxy += float(np.dot(action, action))
                if not np.all(np.isfinite(obs_arr)):
                    constraint_violations += 1
                success = success or _success_from_info(info)
                if bool(terminated) or bool(truncated):
                    break

            returns.append(total_reward)
            successes += int(success)
    finally:
        env.close()

    wall_time_s = time.perf_counter() - t0
    return _rollout_result(
        case=case,
        returns=returns,
        position_errors=position_errors,
        successes=successes,
        energy_proxy=energy_proxy,
        constraint_violations=constraint_violations,
        wall_time_s=wall_time_s,
        applied_overrides=applied_overrides,
        unsupported_overrides=unsupported_overrides,
        environment_config=environment_config,
    )


def evaluate_waypoint_zero_action_case(
    case: EvaluationCase,
    steps_per_episode: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    return evaluate_waypoint_case(
        case,
        action_mode="zero",
        steps_per_episode=steps_per_episode,
        device=device,
    )


def profile_to_dict(profile: EvaluationProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "description": profile.description,
        "task_family": profile.task_family,
        "episodes_per_case": profile.episodes_per_case,
        "seed": profile.seed,
        "source_reference": GENESIS_WORLD_REFERENCE,
        "axes": [
            {
                "category": axis.category,
                "name": axis.name,
                "parameter": axis.parameter,
                "unit": axis.unit,
                "baseline": axis.baseline,
                "values": list(axis.values),
                "rationale": axis.rationale,
            }
            for axis in profile.axes
        ],
        "metrics": list(profile.metrics),
        "claim_guardrails": list(profile.claim_guardrails),
        "cases": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "axis": case.axis,
                "parameter": case.parameter,
                "value": case.value,
                "unit": case.unit,
                "episodes": case.episodes,
                "seed": case.seed,
                "overrides": case.overrides,
            }
            for case in generate_evaluation_cases(profile)
        ],
    }


def profile_to_markdown(profile: EvaluationProfile) -> str:
    cases = generate_evaluation_cases(profile)
    lines = [
        f"# OceanScale Evaluation Profile: {profile.name}",
        "",
        profile.description,
        "",
        "## Source Pattern",
        "",
        (
            "Genesis World 1.0 is used here as an evaluation-first method reference. "
            "OceanScale keeps Newton + Warp as the runtime stack and adds no Genesis dependency."
        ),
        "",
        "## Claim Guardrails",
        "",
    ]
    lines.extend(f"- {guardrail}" for guardrail in profile.claim_guardrails)
    lines.extend(
        [
            "",
            "## Metrics",
            "",
        ]
    )
    lines.extend(f"- {metric}" for metric in profile.metrics)
    lines.extend(
        [
            "",
            "## Perturbation Axes",
            "",
            "| Category | Axis | Parameter | Baseline | Values |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for axis in profile.axes:
        values = ", ".join(str(value) for value in axis.values)
        baseline = f"{axis.baseline} {axis.unit}".strip()
        lines.append(
            f"| {axis.category} | {axis.name} | {axis.parameter} | {baseline} | {values} |"
        )
    lines.extend(
        [
            "",
            "## Case Count",
            "",
            "- Baseline cases: 1",
            f"- Perturbation cases: {len(cases) - 1}",
            f"- Episodes per case: {profile.episodes_per_case}",
            f"- Total planned episodes: {len(cases) * profile.episodes_per_case}",
        ]
    )
    return "\n".join(lines) + "\n"


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def _value_slug(value: Scalar) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p").replace("-", "m")
    return str(value).lower().replace(" ", "_").replace("-", "m")


def _record_to_dict(record: EvaluationRecord) -> dict[str, object]:
    return {
        "case_id": record.case_id,
        "task_name": record.task_name,
        "action_mode": record.action_mode,
        "status": record.status,
        "episodes": record.episodes,
        "seed": record.seed,
        "overrides": _scalar_map(record.overrides),
        "applied_overrides": _scalar_map(record.applied_overrides),
        "unsupported_overrides": _scalar_map(record.unsupported_overrides),
        "metrics": _numeric_metrics(record.metrics),
        "wall_time_s": record.wall_time_s,
        "run_metadata": _json_object_or_empty(record.run_metadata),
        "environment_config": _json_object_or_empty(record.environment_config),
        "error": record.error,
    }


def _profile_from_case_id(case_id: str | None) -> str | None:
    if not case_id:
        return None
    return case_id.split(":", maxsplit=1)[0]


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _dict_or_empty(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _json_object_or_empty(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _case_environment_config(
    environment_config: object,
    *,
    requested_overrides: dict[str, Scalar],
    applied_overrides: dict[str, Scalar],
    unsupported_overrides: dict[str, Scalar],
) -> dict[str, object]:
    config = _json_object_or_empty(environment_config)
    if requested_overrides:
        config["requested_overrides"] = _json_safe_value(requested_overrides)
    if applied_overrides:
        config["applied_overrides"] = _json_safe_value(applied_overrides)
    if unsupported_overrides:
        config["unsupported_overrides"] = _json_safe_value(unsupported_overrides)
    return config


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            native_value = item()
        except (TypeError, ValueError):
            native_value = None
        if native_value is not None and native_value is not value:
            return _json_safe_value(native_value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _unique_dicts(values: Iterable[object]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in values:
        config = _json_object_or_empty(value)
        if not config:
            continue
        encoded = json.dumps(config, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        unique.append(config)
    return unique


def _flatten_mapping(
    value: dict[str, object],
    prefix: str = "",
) -> list[tuple[str, object]]:
    flattened: list[tuple[str, object]] = []
    for key, item in sorted(value.items()):
        item_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.extend(_flatten_mapping(dict(item), item_key))
        elif isinstance(item, list | tuple):
            flattened.append((item_key, ", ".join(str(v) for v in item)))
        else:
            flattened.append((item_key, item))
    return flattened


def _markdown_value(value: object) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _action_from_observation(
    obs: object,
    action_mode: str,
    np_module: Any,
    *,
    task_name: str | None = None,
) -> Any:
    if action_mode == "zero":
        return np_module.zeros(6, dtype=np_module.float32)
    if action_mode not in ACTION_MODES:
        modes = ", ".join(ACTION_MODES)
        raise ValueError(f"unknown action mode {action_mode!r}; expected one of: {modes}")

    obs_arr = np_module.asarray(_as_numpy(obs), dtype=np_module.float32)
    if obs_arr.ndim > 1:
        obs_arr = obs_arr[0]
    action = np_module.zeros(6, dtype=np_module.float32)
    position_error = np_module.zeros(3, dtype=np_module.float32)
    position_error[: min(3, obs_arr.size)] = obs_arr[:3]
    if action_mode == "proportional":
        action[:3] = np_module.clip(0.35 * position_error, -1.0, 1.0)
        return action

    velocity = np_module.zeros(3, dtype=np_module.float32)
    start, stop = _linear_velocity_slice(task_name)
    if obs_arr.size >= stop:
        velocity[:] = obs_arr[start:stop]
    action[:3] = np_module.clip(0.45 * position_error - 0.15 * velocity, -1.0, 1.0)
    return action


def _linear_velocity_slice(task_name: str | None) -> tuple[int, int]:
    if task_name in {"station-keeping", "waypoint"}:
        return 7, 10
    if task_name == "docking":
        return 10, 13
    raise ValueError("task_name is required for damped action mode")


def _git_stdout(args: tuple[str, ...]) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"

    if result.returncode != 0:
        return "unknown"
    value = result.stdout.strip()
    return value or "unknown"


def _git_status_short() -> str | None:
    try:
        result = subprocess.run(
            ("git", "status", "--short"),
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy",
        "torch": "torch",
        "warp": "warp-lang",
        "newton": "newton",
        "gymnasium": "gymnasium",
    }
    versions: dict[str, str] = {}
    for label, package_name in packages.items():
        try:
            versions[label] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[label] = "unknown"
    return versions


def _gpu_inventory() -> dict[str, object]:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _unknown_gpu_inventory()

    if result.returncode != 0:
        return _unknown_gpu_inventory()

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    names: list[str] = []
    drivers: list[str] = []
    memory_totals: list[str] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 1:
            names.append(parts[0])
        if len(parts) >= 2:
            drivers.append(parts[1])
        if len(parts) >= 3:
            memory_totals.append(parts[2])

    driver_versions = sorted(set(drivers))
    return {
        "gpu_inventory_known": True,
        "gpu_count": len(rows),
        "gpu_names": "; ".join(names),
        "nvidia_driver_version": "; ".join(driver_versions) if driver_versions else "unknown",
        "gpu_memory_total_mib": "; ".join(memory_totals),
    }


def _unknown_gpu_inventory() -> dict[str, object]:
    return {
        "gpu_inventory_known": False,
        "gpu_count": 0,
        "gpu_names": "",
        "nvidia_driver_version": "unknown",
        "gpu_memory_total_mib": "",
    }


def _metric_distributions(
    metric_values: dict[str, list[float]],
) -> dict[str, dict[str, float | int]]:
    distributions: dict[str, dict[str, float | int]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        sorted_values = sorted(values)
        distributions[key] = {
            "count": len(sorted_values),
            "min": sorted_values[0],
            "p50": _upper_percentile(sorted_values, 50),
            "p90": _upper_percentile(sorted_values, 90),
            "p95": _upper_percentile(sorted_values, 95),
            "max": sorted_values[-1],
        }
    return distributions


def _upper_percentile(sorted_values: list[float], percentile: int) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100")
    index = math.ceil((percentile / 100.0) * (len(sorted_values) - 1))
    return sorted_values[index]


def _numeric_metrics(value: object) -> dict[str, float | int]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float | int] = {}
    for key, metric in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(metric, bool | int | float):
            metrics[key] = int(metric) if isinstance(metric, bool) else metric
            continue
        if isinstance(metric, str):
            continue
        try:
            metrics[key] = float(metric)
        except (TypeError, ValueError):
            continue
    return metrics


def _scalar_map(value: object) -> dict[str, Scalar]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, bool | int | float | str)
    }


def _apply_station_keeping_overrides(
    env: Any,
    case: EvaluationCase,
    applied_overrides: dict[str, Scalar],
    torch_module: Any,
) -> None:
    if "target_offset_m" in case.overrides:
        offset = float(case.overrides["target_offset_m"])
        env.target_pos = torch_module.tensor(
            (offset, 0.0, -1.5),
            dtype=torch_module.float32,
            device=env.device,
        )
        applied_overrides["target_offset_m"] = case.overrides["target_offset_m"]

    current_keys = {"current_speed_mps", "current_direction_deg"}
    if not current_keys.intersection(case.overrides):
        return

    import numpy as np

    speed = float(case.overrides.get("current_speed_mps", 0.2))
    direction = np.deg2rad(float(case.overrides.get("current_direction_deg", 0.0)))
    env._current_params[:, 0] = speed
    env._current_params[:, 1] = direction
    env._current_params[:, 2] = 30.0
    env._current_params[:, 3] = 0.0
    env._current_params[:, 4] = 0.0
    env._compute_current()
    if "current_speed_mps" in case.overrides:
        applied_overrides["current_speed_mps"] = case.overrides["current_speed_mps"]
    if "current_direction_deg" in case.overrides:
        applied_overrides["current_direction_deg"] = case.overrides["current_direction_deg"]


def _apply_docking_overrides(
    env: Any,
    case: EvaluationCase,
    applied_overrides: dict[str, Scalar],
    torch_module: Any,
) -> None:
    if "target_offset_m" in case.overrides:
        offset = float(case.overrides["target_offset_m"])
        target = torch_module.tensor(
            (offset, 0.0, -1.5),
            dtype=torch_module.float32,
            device=env._device,
        )
        env._target = target
        env._dock_positions[:, :] = target
        applied_overrides["target_offset_m"] = case.overrides["target_offset_m"]

    current_keys = {"current_speed_mps", "current_direction_deg"}
    if not current_keys.intersection(case.overrides):
        return

    import numpy as np

    speed = float(case.overrides.get("current_speed_mps", 0.2))
    direction = np.deg2rad(float(case.overrides.get("current_direction_deg", 0.0)))
    env._current_force[:, 0] = env.cfg.current_coupling * speed * np.cos(direction)
    env._current_force[:, 1] = env.cfg.current_coupling * speed * np.sin(direction)
    env._current_force[:, 2] = 0.0
    if "current_speed_mps" in case.overrides:
        applied_overrides["current_speed_mps"] = case.overrides["current_speed_mps"]
    if "current_direction_deg" in case.overrides:
        applied_overrides["current_direction_deg"] = case.overrides["current_direction_deg"]


def _apply_waypoint_overrides(
    env: Any,
    case: EvaluationCase,
    applied_overrides: dict[str, Scalar],
    torch_module: Any,
) -> None:
    if "target_offset_m" not in case.overrides:
        return
    offset = float(case.overrides["target_offset_m"])
    env._waypoints[:, :, 0] += torch_module.tensor(
        offset, dtype=torch_module.float32, device=env.device
    )
    applied_overrides["target_offset_m"] = case.overrides["target_offset_m"]


def _rollout_result(
    case: EvaluationCase,
    returns: list[float],
    position_errors: list[float],
    successes: int,
    energy_proxy: float,
    constraint_violations: int,
    wall_time_s: float,
    applied_overrides: dict[str, Scalar],
    unsupported_overrides: dict[str, Scalar],
    environment_config: dict[str, object] | None = None,
) -> dict[str, object]:
    import numpy as np

    mean_error = float(np.mean(position_errors)) if position_errors else 0.0
    max_error = float(np.max(position_errors)) if position_errors else 0.0
    return {
        "status": "completed",
        "metrics": {
            "success_rate": successes / max(case.episodes, 1),
            "episode_return": float(np.mean(returns)) if returns else 0.0,
            "mean_position_error_m": mean_error,
            "max_position_error_m": max_error,
            "energy_proxy": energy_proxy,
            "constraint_violations": constraint_violations,
            "wall_time_s": wall_time_s,
        },
        "applied_overrides": applied_overrides,
        "unsupported_overrides": unsupported_overrides,
        "environment_config": environment_config or {},
    }


def _as_numpy(value: object) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return value


def _space_dimension(space: object, default: int) -> int:
    shape = getattr(space, "shape", None)
    if not isinstance(shape, tuple | list) or not shape:
        return default
    try:
        return math.prod(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return default


def _first_float(value: object, default: float) -> float:
    import numpy as np

    if value is None:
        return default
    array = np.asarray(_as_numpy(value), dtype=np.float32).reshape(-1)
    if array.size == 0:
        return default
    return float(array[0])


def _success_from_info(info: dict[str, Any]) -> bool:
    import numpy as np

    if "success" not in info:
        return False
    value = np.asarray(_as_numpy(info["success"])).reshape(-1)
    return bool(value.size and np.any(value))
