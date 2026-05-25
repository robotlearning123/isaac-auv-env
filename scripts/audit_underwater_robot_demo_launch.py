#!/usr/bin/env python3
"""Requirement audit for the underwater robot demo launch candidate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_isaacsim_underwater_demo import (  # noqa: E402
    EXPECTED_TASK_ID,
    MAX_FINAL_DISTANCE_M,
    MIN_EXTERNAL_ROBOT_TRIANGLES,
    MIN_PATH_TRACING_SPP,
    SUMMARY_PATH,
    verify_demo_artifacts,
)
from verify_underwater_robot_demo_release import verify_release_gate  # noqa: E402

Status = Literal["passed", "partial", "missing", "failed"]

WEBSITE_COMPONENT = REPO_ROOT / "website" / "src" / "components" / "UnderwaterRobotDemo.astro"
WEBSITE_VIDEO = REPO_ROOT / "website" / "public" / "videos" / "oceanscale-underwater-robot-demo.mp4"
WEBSITE_POSTER = (
    REPO_ROOT / "website" / "public" / "images" / "oceanscale-underwater-robot-demo.png"
)
WEBSITE_MANIFEST = REPO_ROOT / "website" / "public" / "demo" / "underwater_robot_demo_manifest.json"
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "isaacsim_underwater_demo.py"
ARTIFACT_VERIFIER = REPO_ROOT / "scripts" / "verify_isaacsim_underwater_demo.py"
RELEASE_GATE = REPO_ROOT / "scripts" / "verify_underwater_robot_demo_release.py"
EXTERNAL_ROBOT_ASSET = (
    REPO_ROOT
    / "artifacts"
    / "isaacsim"
    / "external_assets"
    / "bluerov2_gz_heavy"
    / "bluerov2_gz_heavy.usdc"
)


@dataclass(frozen=True)
class RequirementResult:
    requirement: str
    status: Status
    evidence: Any
    next_step: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _result(
    requirement: str,
    status: Status,
    evidence: Any,
    *,
    next_step: str | None = None,
) -> RequirementResult:
    return RequirementResult(
        requirement=requirement,
        status=status,
        evidence=evidence,
        next_step=next_step,
    )


def _status_from_bool(ok: bool) -> Status:
    return "passed" if ok else "failed"


def _summary() -> dict[str, Any]:
    return _read_json(SUMMARY_PATH)


def _external_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("external_robot_asset_evidence")
    if isinstance(value, dict):
        return value
    fallback = summary.get("external_robot_mesh", {})
    if isinstance(fallback, dict) and isinstance(fallback.get("asset_evidence"), dict):
        return fallback["asset_evidence"]
    return {}


def _audit_real_robot(summary: dict[str, Any]) -> RequirementResult:
    external = _external_evidence(summary)
    triangles = external.get("triangle_instance_count")
    ok = (
        EXTERNAL_ROBOT_ASSET.exists()
        and external.get("vehicle") == "BlueROV2 Heavy"
        and external.get("source_license") == "MIT"
        and isinstance(triangles, int)
        and triangles >= MIN_EXTERNAL_ROBOT_TRIANGLES
        and external.get("thruster_count") == 8
        and "external_high_fidelity_mesh" in set(summary.get("robot_visual_features", []))
    )
    return _result(
        "real_robot_asset",
        _status_from_bool(ok),
        {
            "asset": str(EXTERNAL_ROBOT_ASSET.relative_to(REPO_ROOT)),
            "vehicle": external.get("vehicle"),
            "source_repo": external.get("source_repo"),
            "source_commit": external.get("source_commit"),
            "source_license": external.get("source_license"),
            "triangles": triangles,
            "thrusters": external.get("thruster_count"),
        },
        next_step=None
        if ok
        else "Restore the external BlueROV2 Heavy USD asset and provenance metadata.",
    )


def _audit_ocean_visual_floor(summary: dict[str, Any], artifact_ok: bool) -> RequirementResult:
    visual_effects = set(summary.get("visual_effects", []))
    required_effects = {
        "open_ocean_backdrop",
        "bathymetric_seafloor",
        "procedural_pbr_material_textures",
        "depth_graded_water_volume",
        "multi_scale_caustic_lattice",
        "top_light_shafts",
        "marine_snow",
        "rov_headlight_beams",
        "robot_relative_thruster_wake_plumes",
        "robot_weathering_detail",
    }
    spp = summary.get("render_quality", {}).get("pathtracing_total_spp")
    ok = (
        artifact_ok
        and summary.get("renderer") == "Isaac Sim PathTracing"
        and isinstance(spp, int)
        and spp >= MIN_PATH_TRACING_SPP
        and summary.get("requested_resolution") == [1920, 1080]
        and required_effects <= visual_effects
    )
    return _result(
        "ocean_visual_quality_floor",
        "passed" if ok else "failed",
        {
            "renderer": summary.get("renderer"),
            "spp": spp,
            "resolution": summary.get("requested_resolution"),
            "missing_effects": sorted(required_effects - visual_effects),
        },
        next_step=None if ok else "Regenerate the demo with all required ocean visual layers.",
    )


def _audit_unity_unreal_quality_claim(summary: dict[str, Any]) -> RequirementResult:
    return _result(
        "ue_unity_quality_claim",
        "partial",
        {
            "machine_floor": {
                "renderer": summary.get("renderer"),
                "pathtracing_total_spp": summary.get("render_quality", {}).get(
                    "pathtracing_total_spp"
                ),
                "resolution": summary.get("requested_resolution"),
                "visual_effect_count": len(summary.get("visual_effects", [])),
            },
            "unverified": [
                "human art-direction review against UE/Unity-quality target",
                "fresh high-sample rerender or longer cinematic cut",
                "independent visual QA beyond current automated layout and artifact checks",
            ],
        },
        next_step="Run a focused visual quality pass: compare capture/video against the target bar, then rerender with the agreed fixes.",
    )


def _audit_mission(summary: dict[str, Any]) -> RequirementResult:
    mission = summary.get("mission_evidence", {})
    runtime = summary.get("isaac_lab_runtime_smoke", {})
    final_distance = mission.get("final_distance_to_target_m")
    ok = (
        mission.get("completed") is True
        and isinstance(final_distance, int | float)
        and final_distance <= MAX_FINAL_DISTANCE_M
        and mission.get("sonar_detection_rate") == 1.0
        and runtime.get("ok") is True
        and runtime.get("task") == EXPECTED_TASK_ID
        and runtime.get("step_obs_finite") is True
    )
    return _result(
        "robot_mission_demo",
        _status_from_bool(ok),
        {
            "completed": mission.get("completed"),
            "final_distance_to_target_m": final_distance,
            "max_allowed_distance_m": MAX_FINAL_DISTANCE_M,
            "sonar_detection_rate": mission.get("sonar_detection_rate"),
            "isaac_lab_task": runtime.get("task"),
            "isaac_lab_runtime_ok": runtime.get("ok"),
        },
        next_step=None if ok else "Fix the mission evidence or Isaac Lab runtime smoke.",
    )


def _audit_demo_deliverables(artifact_ok: bool) -> RequirementResult:
    required_paths = [
        SUMMARY_PATH,
        GENERATOR_SCRIPT,
        ARTIFACT_VERIFIER,
        RELEASE_GATE,
        WEBSITE_COMPONENT,
        WEBSITE_VIDEO,
        WEBSITE_POSTER,
        WEBSITE_MANIFEST,
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.exists()]
    return _result(
        "demo_deliverables",
        "passed" if artifact_ok and not missing else "failed",
        {
            "artifact_verifier_ok": artifact_ok,
            "missing": missing,
            "website_component": str(WEBSITE_COMPONENT.relative_to(REPO_ROOT)),
            "public_video": str(WEBSITE_VIDEO.relative_to(REPO_ROOT)),
            "public_poster": str(WEBSITE_POSTER.relative_to(REPO_ROOT)),
            "public_manifest": str(WEBSITE_MANIFEST.relative_to(REPO_ROOT)),
        },
        next_step=None if artifact_ok and not missing else "Restore missing demo deliverables.",
    )


def _audit_reproducibility(run_release_gate: bool, run_browser: bool) -> RequirementResult:
    if not run_release_gate:
        return _result(
            "local_reproducibility_gate",
            "partial",
            {
                "release_gate_script": str(RELEASE_GATE.relative_to(REPO_ROOT)),
                "current_run": "not_run",
            },
            next_step="Run this audit with --run-release-gate before calling the MVP launch candidate current.",
        )

    release_result = verify_release_gate(run_browser=run_browser)
    failed = [check for check in release_result["checks"] if not check["ok"]]
    return _result(
        "local_reproducibility_gate",
        "passed" if release_result["ok"] else "failed",
        {
            "release_gate_ok": release_result["ok"],
            "browser_smoke": "run" if run_browser else "skipped",
            "failed_checks": failed,
        },
        next_step=None if release_result["ok"] else "Fix the failing release gate checks.",
    )


def _audit_hosted_launch() -> RequirementResult:
    return _result(
        "hosted_public_demo",
        "missing",
        {
            "local_preview_verified": True,
            "hosted_url_verified_this_audit": False,
        },
        next_step="After owner approval, deploy/preview the website and verify the hosted #demo surface.",
    )


def audit_launch_candidate(
    *,
    run_release_gate: bool = False,
    run_browser: bool = False,
    run_ffprobe: bool = True,
) -> dict[str, Any]:
    artifact_result = verify_demo_artifacts(run_ffprobe=run_ffprobe)
    summary = _summary()
    requirements = [
        _audit_real_robot(summary),
        _audit_ocean_visual_floor(summary, artifact_result["ok"]),
        _audit_unity_unreal_quality_claim(summary),
        _audit_mission(summary),
        _audit_demo_deliverables(artifact_result["ok"]),
        _audit_reproducibility(run_release_gate, run_browser),
        _audit_hosted_launch(),
    ]
    blocking_statuses = {"failed", "missing"}
    launch_candidate = all(
        item.status not in blocking_statuses
        for item in requirements
        if item.requirement != "hosted_public_demo"
    )
    goal_complete = all(item.status == "passed" for item in requirements)
    return {
        "objective": "like real ue/unity quality ocean scene with real robot demo",
        "launch_candidate": launch_candidate,
        "goal_complete": goal_complete,
        "artifact_verifier_ok": artifact_result["ok"],
        "requirements": [asdict(item) for item in requirements],
        "next_focus": [
            item.next_step for item in requirements if item.status != "passed" and item.next_step
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the underwater robot demo satisfies the launch objective.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable audit JSON.")
    parser.add_argument(
        "--run-release-gate",
        action="store_true",
        help="Run the release gate as part of the reproducibility requirement.",
    )
    parser.add_argument(
        "--run-browser",
        action="store_true",
        help="Include Playwright preview smoke when --run-release-gate is set.",
    )
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Skip MP4 stream checks in the artifact verifier.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every launch objective requirement is passed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_launch_candidate(
        run_release_gate=args.run_release_gate,
        run_browser=args.run_browser,
        run_ffprobe=not args.no_ffprobe,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "COMPLETE" if result["goal_complete"] else "PARTIAL"
        print(f"underwater robot demo launch audit: {status}")
        print(f"- launch_candidate: {result['launch_candidate']}")
        print(f"- goal_complete: {result['goal_complete']}")
        for requirement in result["requirements"]:
            print(
                f"- {requirement['status']}: {requirement['requirement']}"
                + (
                    f" -> {requirement['next_step']}"
                    if requirement.get("next_step") and requirement["status"] != "passed"
                    else ""
                )
            )

    if args.require_complete and not result["goal_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
