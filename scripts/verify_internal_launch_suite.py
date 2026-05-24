from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("artifacts/LAUNCH_READINESS_MANIFEST_INTERNAL_2026-05-24.json")
BENCHMARK = Path("benchmarks/competitive/results.json")
COMPLETION_AUDIT = Path("artifacts/LAUNCH_COMPLETION_AUDIT_INTERNAL_2026-05-24.md")
EXTERNAL_STATE = Path("artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json")
PACKAGE_STATE = Path("artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json")

REQUIRED_SOURCE_KEYS = {
    "benchmark",
    "package",
    "package_version_normalized",
    "website_package",
    "workflow_directory",
    "data_directory",
    "internal_report",
    "final_gate",
    "evidence_index",
    "decision_matrix",
    "boss_packet",
    "runbook",
    "hardening_backlog",
    "handoff",
    "completion_audit",
    "test_matrix",
    "package_data_evidence",
    "product_capability_evidence",
    "package_data_policy_tests",
    "version_identity_tests",
    "launch_suite_verifier",
    "launch_suite_verifier_tests",
    "external_launch_state",
    "external_launch_state_script",
    "external_launch_state_tests",
    "package_artifact_state",
    "package_artifact_state_script",
    "package_artifact_state_tests",
}

REQUIRED_AREAS = {
    "product",
    "data",
    "benchmark",
    "tests",
    "pipeline",
    "version_identity",
    "public_surface",
}

REQUIRED_APPROVAL_PHRASES = {
    "delete oceanscale/data/vec_normalize.pkl",
    "deploy or update public website",
    "publish to PyPI or TestPyPI",
}

REQUIRED_AUDIT_REQUIREMENTS = {
    "Review all tests",
    "Verify tech stack",
    "Prepare benchmark evidence",
    "Prepare data/package evidence",
    "Prepare product evidence",
    "Prepare pipeline evidence",
    "Prepare results/report suite",
    "Respect not public",
    "Be launch-ready",
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _load_json(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads((root / path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _check(name: str, ok: bool, detail: str) -> Check:
    return Check(name=name, ok=ok, detail=detail)


def _manifest(root: Path) -> dict[str, Any]:
    return _load_json(root, MANIFEST)


def _benchmark(root: Path) -> dict[str, Any]:
    return _load_json(root, BENCHMARK)


def _source_paths_exist(root: Path, source_of_truth: dict[str, Any]) -> Check:
    missing = []
    for key in sorted(REQUIRED_SOURCE_KEYS):
        value = source_of_truth.get(key)
        if not isinstance(value, str) or not (root / value).exists():
            missing.append(f"{key}={value!r}")

    return _check(
        "source_of_truth_paths",
        not missing,
        "all manifest source_of_truth paths exist" if not missing else ", ".join(missing),
    )


def _benchmark_check(root: Path, manifest: dict[str, Any]) -> Check:
    benchmark = _benchmark(root)
    areas = manifest.get("areas", {})
    manifest_benchmark = areas.get("benchmark", {}) if isinstance(areas, dict) else {}
    throughputs = benchmark.get("batched_throughput")
    sensors = benchmark.get("sensors")
    physics = benchmark.get("physics")
    fidelity = benchmark.get("fidelity")

    throughput_by_env: dict[int, float] = {}
    if isinstance(throughputs, list):
        for row in throughputs:
            if isinstance(row, dict):
                n_envs = row.get("n_envs")
                env_steps = row.get("env_steps_per_second")
                if isinstance(n_envs, int) and isinstance(env_steps, int | float):
                    throughput_by_env[n_envs] = float(env_steps)

    terms = physics.get("terms", {}) if isinstance(physics, dict) else {}
    ok = (
        benchmark.get("benchmark") == "oceanscale_standard"
        and benchmark.get("version") == manifest_benchmark.get("version")
        and all(throughput_by_env.get(n, 0) > 0 for n in (64, 256, 1024, 4096))
        and isinstance(sensors, dict)
        and sensors.get("imu") is True
        and sensors.get("dvl") is True
        and sensors.get("depth_pressure") is True
        and sensors.get("sonar_imaging") is False
        and sensors.get("camera_model") is False
        and sensors.get("acoustic_comms") is False
        and terms.get("cross_coupling_damping") is False
        and isinstance(fidelity, dict)
        and fidelity.get("pass") is True
    )
    return _check(
        "benchmark_truth",
        ok,
        "benchmark JSON matches manifest version, throughput, sensor truth, and fidelity pass",
    )


def _completion_audit_check(root: Path) -> Check:
    text = (root / COMPLETION_AUDIT).read_text()
    missing = sorted(item for item in REQUIRED_AUDIT_REQUIREMENTS if item not in text)
    return _check(
        "completion_audit_scope",
        not missing,
        "completion audit covers all objective areas" if not missing else ", ".join(missing),
    )


def _external_state_check(root: Path) -> Check:
    state = _load_json(root, EXTERNAL_STATE)
    status = state.get("status", {})
    checks = state.get("checks", {})
    pypi = checks.get("pypi_availability", {}) if isinstance(checks, dict) else {}
    runner = checks.get("workflow_runner_policy", {}) if isinstance(checks, dict) else {}
    ok = (
        state.get("scope") == "internal_alpha_readiness"
        and state.get("public_launch") is False
        and isinstance(status, dict)
        and status.get("release_deploy") == "deferred_manual_only"
        and status.get("public_pypi_publish") == "deferred_not_public"
        and status.get("pypi_availability") == "deferred_not_public"
        and status.get("workflow_runner_policy") == "green"
        and status.get("release_policy") == "green"
        and status.get("public_pypi_policy") == "green"
        and status.get("overall_public_launch_pipeline") == "deferred_not_public"
        and isinstance(pypi, dict)
        and pypi.get("available") is False
        and isinstance(runner, dict)
        and runner.get("status") == "green"
    )
    return _check(
        "external_launch_state",
        ok,
        "external pipeline artifact records ARC runners and deferred public release/PyPI",
    )


def _package_artifact_state_check(root: Path) -> Check:
    state = _load_json(root, PACKAGE_STATE)
    status = state.get("status", {})
    wheel = state.get("wheel", {})
    sdist = state.get("sdist", {})
    install = state.get("install_smoke", {})
    source = state.get("source_data", {})
    ok = (
        state.get("scope") == "internal_alpha_readiness"
        and state.get("public_launch") is False
        and isinstance(status, dict)
        and status.get("package_artifact") == "green"
        and status.get("source_data_hygiene") == "yellow"
        and isinstance(wheel, dict)
        and wheel.get("has_supported_zip") is True
        and wheel.get("has_supported_npz") is True
        and wheel.get("has_unsupported_pkl") is False
        and wheel.get("entry_point_present") is True
        and isinstance(sdist, dict)
        and sdist.get("has_supported_zip") is True
        and sdist.get("has_supported_npz") is True
        and sdist.get("has_unsupported_pkl") is False
        and isinstance(install, dict)
        and install.get("passed") is True
        and isinstance(source, dict)
        and source.get("source_pkl_exists") is True
    )
    return _check(
        "package_artifact_state",
        ok,
        "package artifact is green while source data hygiene remains owner-decision yellow",
    )


def verify(root: Path = ROOT) -> list[Check]:
    manifest = _manifest(root)
    source_of_truth = manifest.get("source_of_truth", {})
    areas = manifest.get("areas", {})
    approvals_required = manifest.get("approvals_required", [])
    go_no_go = manifest.get("go_no_go", {})

    checks = [
        _check("manifest_exists", (root / MANIFEST).is_file(), str(MANIFEST)),
        _check("manifest_scope", manifest.get("scope") == "internal_alpha_readiness", "internal scope"),
        _check("not_public", manifest.get("public_launch") is False, "public_launch is false"),
        _check(
            "required_source_keys",
            isinstance(source_of_truth, dict)
            and REQUIRED_SOURCE_KEYS.issubset(source_of_truth.keys()),
            "all required source_of_truth keys are present",
        ),
        _source_paths_exist(root, source_of_truth if isinstance(source_of_truth, dict) else {}),
        _check(
            "required_areas",
            isinstance(areas, dict) and REQUIRED_AREAS.issubset(areas.keys()),
            "manifest covers product/data/benchmark/tests/pipeline/version/public_surface",
        ),
        _benchmark_check(root, manifest),
        _check(
            "test_gate_recorded",
            isinstance(areas, dict)
            and "passed" in str(areas.get("tests", {}).get("recorded_results", ""))
            and "xfailed" in str(areas.get("tests", {}).get("recorded_results", "")),
            "full pytest result recorded in manifest",
        ),
        _check(
            "pipeline_policy_recorded",
            isinstance(areas, dict)
            and areas.get("pipeline", {}).get("status") == "deferred_not_public"
            and "disabled" in str(areas.get("pipeline", {}).get("publish_pypi", {})),
            "not-release and no-public-PyPI policy remains explicit",
        ),
        _check(
            "public_launch_no_go",
            isinstance(go_no_go, dict) and go_no_go.get("public_launch") == "no_go",
            "public launch is no_go",
        ),
        _check(
            "public_surface_deferred",
            isinstance(areas, dict)
            and areas.get("public_surface", {}).get("status") == "deferred",
            "public surface deferred per not-public instruction",
        ),
        _check(
            "approvals_required",
            isinstance(approvals_required, list)
            and REQUIRED_APPROVAL_PHRASES.issubset(set(map(str, approvals_required))),
            "approval boundaries are explicit",
        ),
        _completion_audit_check(root),
        _external_state_check(root),
        _package_artifact_state_check(root),
    ]
    return checks


def main() -> int:
    checks = verify()
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
