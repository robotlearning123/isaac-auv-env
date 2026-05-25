#!/usr/bin/env python3
"""Fast verifier for the canonical Isaac Sim underwater robot demo artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "isaacsim"
SUMMARY_PATH = ARTIFACT_DIR / "underwater_robot_demo_pathtracing64_summary.json"
STAGE_PATH = ARTIFACT_DIR / "underwater_robot_demo_pathtracing64.usda"
CAPTURE_PATH = ARTIFACT_DIR / "underwater_robot_demo_pathtracing64.png"
VIDEO_PATH = ARTIFACT_DIR / "underwater_robot_demo_pathtracing64.mp4"
FRAMES_DIR = ARTIFACT_DIR / "underwater_robot_demo_pathtracing64_frames"

EXPECTED_TASK_ID = "OceanScale-UnderwaterRobot-Direct-v0"
EXPECTED_RESOLUTION = [1920, 1080]
EXPECTED_FRAME_COUNT = 24
EXPECTED_FPS = 12.0
MIN_PATH_TRACING_SPP = 64
MAX_FINAL_DISTANCE_M = 0.15
MIN_EXTERNAL_ROBOT_TRIANGLES = 100_000

REQUIRED_VISUAL_EFFECTS = {
    "procedural_pbr_material_textures",
    "depth_graded_water_volume",
    "multi_scale_caustic_lattice",
    "robot_weathering_detail",
    "robot_relative_thruster_wake_plumes",
}
REQUIRED_ROBOT_FEATURES = {
    "external_high_fidelity_mesh",
    "subtle_robot_scuffs",
    "localized_biofilm_patches",
    "silt_runoff_streaks",
    "robot_relative_bubble_trails",
}
REQUIRED_STAGE_TOKENS = {
    "UsdUVTexture",
    "UsdPrimvarReader_float2",
    "primvars:st",
    "DepthGradedWaterVolume",
    "CausticLattice",
    "RobotWeatheringDetail",
    "RobotThrusterWakeDetail",
    "ExternalRobotVisual",
    "OceanScaleDirectRLEnv",
    EXPECTED_TASK_ID,
}
FORBIDDEN_STAGE_TOKENS = {
    "CameraWaterVeil",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _png_size(path: Path) -> list[int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return [int(width), int(height)]


def _fps_from_rate(rate: str | None) -> float | None:
    if not rate:
        return None
    if "/" not in rate:
        return float(rate)
    numerator, denominator = rate.split("/", maxsplit=1)
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


def _ffprobe_video(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is not available")

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": int(stream["nb_frames"]),
        "fps": _fps_from_rate(stream.get("r_frame_rate")),
        "duration": float(stream["duration"]),
        "bit_rate": int(stream["bit_rate"]),
    }


def _record(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    evidence: Any,
    *,
    detail: str | None = None,
) -> None:
    check: dict[str, Any] = {
        "name": name,
        "ok": bool(ok),
        "evidence": evidence,
    }
    if detail is not None:
        check["detail"] = detail
    checks.append(check)


def verify_demo_artifacts(*, run_ffprobe: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    artifacts = {
        "summary": str(SUMMARY_PATH),
        "stage": str(STAGE_PATH),
        "capture": str(CAPTURE_PATH),
        "video": str(VIDEO_PATH),
        "frames_dir": str(FRAMES_DIR),
    }

    for name, path in (
        ("summary_exists", SUMMARY_PATH),
        ("stage_exists", STAGE_PATH),
        ("capture_exists", CAPTURE_PATH),
        ("video_exists", VIDEO_PATH),
        ("frames_dir_exists", FRAMES_DIR),
    ):
        _record(checks, name, path.exists(), str(path))

    if not all(check["ok"] for check in checks):
        return {"ok": False, "artifacts": artifacts, "checks": checks}

    summary = _read_json(SUMMARY_PATH)
    stage_source = STAGE_PATH.read_text(encoding="utf-8")

    _record(checks, "renderer_is_path_tracing", summary.get("renderer") == "Isaac Sim PathTracing", summary.get("renderer"))
    spp = summary.get("render_quality", {}).get("pathtracing_total_spp")
    _record(checks, "path_tracing_spp", isinstance(spp, int) and spp >= MIN_PATH_TRACING_SPP, spp)
    _record(checks, "capture_resolution", summary.get("capture", {}).get("resolution") == EXPECTED_RESOLUTION, summary.get("capture", {}).get("resolution"))
    _record(checks, "requested_resolution", summary.get("requested_resolution") == EXPECTED_RESOLUTION, summary.get("requested_resolution"))

    capture_size = _png_size(CAPTURE_PATH)
    _record(checks, "capture_png_size", capture_size == EXPECTED_RESOLUTION, capture_size)

    frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
    _record(checks, "sequence_frame_count", len(frame_paths) == EXPECTED_FRAME_COUNT, len(frame_paths))
    _record(checks, "summary_frame_count", summary.get("sequence", {}).get("frame_count") == EXPECTED_FRAME_COUNT, summary.get("sequence", {}).get("frame_count"))
    _record(checks, "sequence_resolution", summary.get("sequence", {}).get("resolution") == EXPECTED_RESOLUTION, summary.get("sequence", {}).get("resolution"))
    if frame_paths:
        frame_sizes = {_png_size(path)[0] for path in frame_paths}, {_png_size(path)[1] for path in frame_paths}
        _record(
            checks,
            "sequence_png_sizes",
            frame_sizes == ({EXPECTED_RESOLUTION[0]}, {EXPECTED_RESOLUTION[1]}),
            {"widths": sorted(frame_sizes[0]), "heights": sorted(frame_sizes[1])},
        )

    visual_effects = set(summary.get("visual_effects", []))
    robot_features = set(summary.get("robot_visual_features", []))
    _record(checks, "required_visual_effects", REQUIRED_VISUAL_EFFECTS <= visual_effects, sorted(REQUIRED_VISUAL_EFFECTS - visual_effects))
    _record(checks, "required_robot_features", REQUIRED_ROBOT_FEATURES <= robot_features, sorted(REQUIRED_ROBOT_FEATURES - robot_features))

    mission = summary.get("mission_evidence", {})
    _record(checks, "mission_completed", mission.get("completed") is True, mission.get("completed"))
    _record(
        checks,
        "mission_final_distance",
        isinstance(mission.get("final_distance_to_target_m"), int | float)
        and mission["final_distance_to_target_m"] <= MAX_FINAL_DISTANCE_M,
        mission.get("final_distance_to_target_m"),
    )
    _record(checks, "mission_sonar_detection", mission.get("sonar_detection_rate") == 1.0, mission.get("sonar_detection_rate"))

    runtime = summary.get("isaac_lab_runtime_smoke", {})
    _record(checks, "isaac_lab_runtime_ok", runtime.get("ok") is True, runtime.get("ok"))
    _record(checks, "isaac_lab_task", runtime.get("task") == EXPECTED_TASK_ID, runtime.get("task"))
    _record(checks, "isaac_lab_device", runtime.get("device") == "cuda:0", runtime.get("device"))
    _record(checks, "isaac_lab_step_obs_finite", runtime.get("step_obs_finite") is True, runtime.get("step_obs_finite"))

    external_evidence = summary.get("external_robot_asset_evidence", {})
    if not external_evidence:
        external_evidence = summary.get("external_robot_mesh", {}).get("asset_evidence", {})
    triangle_count = external_evidence.get("triangle_instance_count")
    _record(
        checks,
        "external_robot_triangles",
        isinstance(triangle_count, int) and triangle_count >= MIN_EXTERNAL_ROBOT_TRIANGLES,
        triangle_count,
    )
    _record(checks, "external_robot_license", external_evidence.get("source_license") == "MIT", external_evidence.get("source_license"))

    material_assets = summary.get("material_assets", {})
    for name, raw_path in material_assets.items():
        path = Path(raw_path)
        exists = path.exists()
        _record(checks, f"material_asset_exists:{name}", exists, str(path))
        if exists:
            _record(checks, f"material_asset_size:{name}", _png_size(path) == [512, 512], _png_size(path))
            _record(checks, f"material_asset_usd_binding:{name}", f"@{path.as_posix()}@" in stage_source, str(path))

    missing_stage_tokens = sorted(token for token in REQUIRED_STAGE_TOKENS if token not in stage_source)
    forbidden_stage_tokens = sorted(token for token in FORBIDDEN_STAGE_TOKENS if token in stage_source)
    _record(checks, "required_stage_tokens", not missing_stage_tokens, missing_stage_tokens)
    _record(checks, "forbidden_stage_tokens", not forbidden_stage_tokens, forbidden_stage_tokens)

    if run_ffprobe:
        try:
            video = _ffprobe_video(VIDEO_PATH)
        except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as exc:
            _record(checks, "video_ffprobe", False, str(VIDEO_PATH), detail=str(exc))
        else:
            artifacts["video_probe"] = video
            _record(checks, "video_resolution", [video["width"], video["height"]] == EXPECTED_RESOLUTION, [video["width"], video["height"]])
            _record(checks, "video_frame_count", video["nb_frames"] == EXPECTED_FRAME_COUNT, video["nb_frames"])
            _record(checks, "video_fps", video["fps"] == EXPECTED_FPS, video["fps"])
            _record(checks, "video_duration", 1.9 <= video["duration"] <= 2.1, video["duration"])

    return {
        "ok": all(check["ok"] for check in checks),
        "artifacts": artifacts,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the canonical Isaac Sim underwater robot demo artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable verification payload.",
    )
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Skip MP4 stream metadata checks.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional path to write the verification payload as a demo manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_demo_artifacts(run_ffprobe=not args.no_ffprobe)

    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"underwater robot demo verification: {status}")
        for check in result["checks"]:
            marker = "ok" if check["ok"] else "fail"
            print(f"- {marker}: {check['name']} -> {check['evidence']}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
