#!/usr/bin/env python3
"""Build the optional Isaac Sim underwater robot demo scene.

Run this with an Isaac Sim Python environment, not the default OceanScale venv:

    PYTHONNOUSERSITE=1 \
    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libgcc_s.so.1" \
    /home/robot/miniconda3/envs/isaac5/bin/python scripts/isaacsim_underwater_demo.py \
      --output-usd /tmp/oceanscale_isaac_underwater_demo.usda
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_RENDERER = "RaytracedLighting"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "isaacsim" / "underwater_robot_demo.usda"
DEFAULT_CAPTURE = REPO_ROOT / "artifacts" / "isaacsim" / "underwater_robot_demo.png"
DEFAULT_SEQUENCE_DIR = REPO_ROOT / "artifacts" / "isaacsim" / "underwater_robot_demo_frames"
DEFAULT_VIDEO = REPO_ROOT / "artifacts" / "isaacsim" / "underwater_robot_demo.mp4"
DEFAULT_ROBOT_ASSET = REPO_ROOT / "oceanscale" / "assets" / "bluerov2_heavy.usda"
USD_ASSET_SUFFIXES = {".usd", ".usda", ".usdc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an Isaac Sim RTX-ready OceanScale BlueROV2 underwater scene.",
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="USD stage to write.",
    )
    parser.add_argument(
        "--robot-usd",
        type=Path,
        default=DEFAULT_ROBOT_ASSET,
        help="OpenUSD BlueROV2 Heavy asset to reference.",
    )
    parser.add_argument(
        "--external-robot-mesh",
        type=Path,
        default=None,
        help="Optional licensed OpenUSD robot asset used as the high-fidelity robot visual.",
    )
    parser.add_argument(
        "--external-robot-scale",
        type=float,
        default=1.0,
        help="Scale applied to --external-robot-mesh visual overlay.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of animated frames to author.",
    )
    parser.add_argument(
        "--run-frames",
        type=int,
        default=3,
        help="Isaac Sim update frames to execute before saving.",
    )
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default=DEFAULT_RENDERER,
        help="Isaac RTX renderer mode.",
    )
    parser.add_argument(
        "--quality-preset",
        choices=("balanced", "cinematic"),
        default="balanced",
        help="RTX quality settings preset.",
    )
    parser.add_argument(
        "--pathtracing-spp",
        type=int,
        default=64,
        help="Path tracing total samples per pixel when --renderer PathTracing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Isaac Sim viewport width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Isaac Sim viewport height.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Isaac Sim with the viewport UI instead of headless generation.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable scene summary.",
    )
    parser.add_argument(
        "--mission-json",
        type=Path,
        default=None,
        help="Optional OceanScale underwater-mvp metrics JSON used to drive the Isaac timeline.",
    )
    parser.add_argument(
        "--isaaclab-runtime-smoke-json",
        type=Path,
        default=None,
        help="Optional real Isaac Lab DirectRLEnv smoke JSON to embed in the scene evidence.",
    )
    parser.add_argument(
        "--capture-png",
        type=Path,
        default=None,
        help=f"Optional Isaac Sim viewport capture path. Default target: {DEFAULT_CAPTURE}.",
    )
    parser.add_argument(
        "--capture-warmup-frames",
        type=int,
        default=40,
        help="Isaac Sim update frames before viewport capture.",
    )
    parser.add_argument(
        "--capture-sequence-dir",
        type=Path,
        default=None,
        help=f"Optional directory for timeline viewport PNG frames. Default target: {DEFAULT_SEQUENCE_DIR}.",
    )
    parser.add_argument(
        "--sequence-frame-count",
        type=int,
        default=0,
        help="Number of viewport frames to capture across the authored timeline.",
    )
    parser.add_argument(
        "--sequence-fps",
        type=int,
        default=30,
        help="FPS metadata used when sampling the timeline and encoding MP4.",
    )
    parser.add_argument(
        "--capture-mp4",
        type=Path,
        default=None,
        help=f"Optional MP4 path assembled from captured sequence frames. Default target: {DEFAULT_VIDEO}.",
    )
    return parser.parse_args()


def _load_mission_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_isaaclab_runtime_smoke(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ok": raw.get("ok"),
        "api": raw.get("api"),
        "task": raw.get("task"),
        "isaaclab_version": raw.get("isaaclab_version"),
        "isaaclab_tasks_version": raw.get("isaaclab_tasks_version"),
        "num_envs": raw.get("num_envs"),
        "device": raw.get("device"),
        "oceanscale_task_registered": raw.get("oceanscale_task_registered"),
        "preloaded_packages": raw.get("preloaded_packages", []),
        "observation_space": raw.get("observation_space"),
        "action_space": raw.get("action_space"),
        "action_shape": raw.get("action_shape"),
        "reset_obs_finite": raw.get("reset_obs_finite"),
        "step_obs_finite": raw.get("step_obs_finite"),
        "reward_finite": raw.get("reward_finite"),
    }


def _usd_asset_evidence(usd_path: Path, prim_path: str | None = None) -> dict[str, Any]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD asset: {usd_path}")
    prim = stage.GetPrimAtPath(prim_path) if prim_path is not None else stage.GetDefaultPrim()
    if not prim.IsValid():
        raise RuntimeError(f"USD asset evidence prim is invalid: {usd_path} {prim_path}")

    evidence: dict[str, Any] = {
        "asset": str(usd_path),
        "prim": str(prim.GetPath()),
    }
    for key, attr_name in (
        ("asset_role", "oceanscale:assetRole"),
        ("vehicle", "oceanscale:vehicle"),
        ("source_repo", "oceanscale:sourceRepo"),
        ("source_commit", "oceanscale:sourceCommit"),
        ("source_license", "oceanscale:sourceLicense"),
        ("source_model", "oceanscale:sourceModel"),
        ("source_asset_type", "oceanscale:sourceAssetType"),
        ("geometry_reference", "oceanscale:geometryReference"),
        ("dynamics_reference", "oceanscale:dynamicsReference"),
        ("thruster_count", "oceanscale:thrusterCount"),
        ("visual_mesh_count", "oceanscale:visualMeshCount"),
        ("unique_collada_mesh_count", "oceanscale:uniqueColladaMeshCount"),
        ("triangle_instance_count", "oceanscale:triangleInstanceCount"),
        ("cad_derived", "oceanscale:cadDerived"),
        ("upgrade_candidate", "oceanscale:upgradeCandidate"),
    ):
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            evidence[key] = attr.Get()
    return evidence


def _robot_asset_evidence(robot_usd: Path) -> dict[str, Any]:
    return _usd_asset_evidence(robot_usd, "/World/BlueROV2Heavy")


def _external_usd_asset_evidence(usd_path: Path) -> dict[str, Any]:
    reference_prim_path = _external_robot_reference_prim_path(usd_path)
    return _usd_asset_evidence(usd_path, reference_prim_path)


def _trajectory_positions_from_summary(
    mission_summary: dict[str, Any] | None,
) -> list[tuple[float, float, float]]:
    if mission_summary is None:
        return []

    samples = mission_summary.get("trajectory", {}).get("samples", [])
    positions: list[tuple[float, float, float]] = []
    for sample in samples:
        raw_position = sample.get("position")
        if not isinstance(raw_position, list) or len(raw_position) < 3:
            continue
        positions.append((float(raw_position[0]), float(raw_position[1]), float(raw_position[2])))
    return positions


def _mission_evidence_summary(mission_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if mission_summary is None:
        return None

    metrics = mission_summary.get("metrics", {})
    mission = mission_summary.get("mission", {})
    trajectory = mission_summary.get("trajectory", {})
    evidence = {
        "completed": bool(mission.get("completed", False)),
        "final_distance_to_target_m": metrics.get("final_distance_to_target_m"),
        "min_distance_to_target_m": metrics.get("min_distance_to_target_m"),
        "sonar_detection_rate": metrics.get("sonar_detection_rate"),
        "steps_per_sec": metrics.get("steps_per_sec"),
        "trajectory_sample_count": trajectory.get("sample_count"),
    }
    isaaclab_probe = mission_summary.get("isaac_lab_probe")
    if isinstance(isaaclab_probe, dict):
        evidence["isaac_lab_probe"] = {
            "api": isaaclab_probe.get("api"),
            "task_id": isaaclab_probe.get("task_id"),
            "controller": isaaclab_probe.get("controller"),
            "gym_registered": isaaclab_probe.get("gym_registered"),
            "steps": isaaclab_probe.get("steps"),
            "observation_dim": isaaclab_probe.get("observation_dim"),
            "action_dim": isaaclab_probe.get("action_dim"),
            "obs_finite": isaaclab_probe.get("obs_finite"),
            "reward_finite": isaaclab_probe.get("reward_finite"),
        }
    return evidence


def _resolve_external_robot_usd(
    external_robot_mesh: Path | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if external_robot_mesh is None:
        return None, None

    mesh_path = external_robot_mesh.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"External robot mesh does not exist: {mesh_path}")

    suffix = mesh_path.suffix.lower()
    if suffix in USD_ASSET_SUFFIXES:
        return mesh_path, {
            "source": str(mesh_path),
            "usd": str(mesh_path),
            "converted": False,
        }

    raise ValueError(
        "--external-robot-mesh must point to a pre-converted OpenUSD asset "
        f"with one of {sorted(USD_ASSET_SUFFIXES)}; got {mesh_path.suffix}"
    )


def _start_simulation_app(gui: bool, *, width: int, height: int, renderer: str) -> Any:
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": not gui,
            "width": width,
            "height": height,
            "renderer": renderer,
        }
    )


def _run_async_with_app_updates(
    app: Any,
    awaitable: Any,
    *,
    max_frames: int,
    description: str,
) -> Any:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    task = loop.create_task(awaitable)
    for _ in range(max_frames):
        app.update()
        loop.run_until_complete(asyncio.sleep(0))
        if task.done():
            return task.result()

    task.cancel()
    loop.run_until_complete(asyncio.sleep(0))
    raise RuntimeError(f"{description} did not finish after {max_frames} frames")


def _apply_render_quality(
    *,
    renderer: str,
    quality_preset: str,
    pathtracing_spp: int,
) -> dict[str, Any]:
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set_string("/rtx/rendermode", renderer)
    settings.set_bool("/rtx/newDenoiser/enabled", True)
    settings.set_bool("/rtx/materialDb/syncLoads", True)
    settings.set_bool("/rtx/hydra/materialSyncLoads", True)
    settings.set_bool("/omni.kit.plugin/syncUsdLoads", True)

    applied: dict[str, Any] = {
        "renderer": renderer,
        "quality_preset": quality_preset,
        "denoiser": True,
        "material_sync_loads": True,
    }

    if quality_preset == "cinematic":
        settings.set_bool("/app/captureFrame/setAlphaTo1", True)
        settings.set_bool("/app/hydraEngine/waitIdle", True)
        settings.set_bool("/app/renderer/waitIdle", True)
        settings.set_bool("/app/docks/disabled", True)
        settings.set_bool("/app/viewport/forceHideFps", True)
        settings.set_bool("/app/viewport/defaults/hud/visible", False)
        settings.set_bool("/app/viewport/grid/enabled", False)
        settings.set_bool("/app/viewport/showLayerMenu", False)
        settings.set_bool("/app/window/hideUi", True)
        settings.set_int("/persistent/app/viewport/displayOptions", 0)
        settings.set_bool("/persistent/app/viewport/hud/visible", False)
        settings.set_bool("/rtx/shadows/enabled", True)
        settings.set_bool("/rtx/reflections/enabled", True)
        settings.set_bool("/rtx/ambientOcclusion/enabled", True)
        settings.set_int("/rtx/post/tonemap/op", 6)
        settings.set_bool("/rtx/pathtracing/lightcache/cached/enabled", True)
        settings.set_bool("/rtx/raytracing/lightcache/spatialCache/enabled", True)
        applied.update(
            {
                "capture_alpha_opaque": True,
                "capture_overlay_hidden": True,
                "wait_idle": True,
                "shadows": True,
                "reflections": True,
                "ambient_occlusion": True,
                "tonemap_op": 6,
            }
        )

    if renderer == "PathTracing":
        settings.set_float("/rtx/pathtracing/totalSpp", float(pathtracing_spp))
        applied["pathtracing_total_spp"] = pathtracing_spp

    return applied


def _make_material(
    stage: Any,
    path: str,
    *,
    color: tuple[float, float, float],
    roughness: float = 0.72,
    metallic: float = 0.0,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] | None = None,
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _make_textured_material(
    stage: Any,
    path: str,
    *,
    texture_path: Path,
    fallback_color: tuple[float, float, float],
    roughness: float = 0.72,
    metallic: float = 0.0,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] | None = None,
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))

    uv_reader = UsdShade.Shader.Define(stage, f"{path}/UvReader")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, f"{path}/AlbedoTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture_path.as_posix())
    )
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        uv_reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(*fallback_color, 1.0)
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_png_rgb(
    path: Path,
    *,
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
) -> None:
    if len(pixels) != width * height:
        raise ValueError(f"Expected {width * height} pixels for {path}, got {len(pixels)}")

    raw_rows = []
    for y in range(height):
        row = pixels[y * width : (y + 1) * width]
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row))
    payload = b"".join(raw_rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(payload, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _texture_noise(x: int, y: int, seed: int) -> float:
    value = (x * 374761393 + y * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((value ^ (value >> 16)) & 0xFF) / 255.0


def _clamp_byte(value: float) -> int:
    return max(0, min(255, round(value)))


def _ensure_demo_texture_assets(texture_dir: Path) -> dict[str, Path]:
    from math import sin

    width = 512
    height = 512

    seafloor_pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            u = x / width
            v = y / height
            ripple = 0.5 + 0.5 * sin(52.0 * v + 8.0 * sin(7.0 * u))
            grain = _texture_noise(x, y, 17)
            shell = 1.0 if _texture_noise(x // 3, y // 3, 31) > 0.965 else 0.0
            seafloor_pixels.append(
                (
                    _clamp_byte(22 + 24 * ripple + 19 * grain + 46 * shell),
                    _clamp_byte(34 + 28 * ripple + 22 * grain + 38 * shell),
                    _clamp_byte(28 + 20 * ripple + 16 * grain + 28 * shell),
                )
            )

    rock_pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            grain = _texture_noise(x, y, 43)
            vein = 0.5 + 0.5 * sin(0.08 * x + 0.035 * y + 4.0 * grain)
            algae = 1.0 if _texture_noise(x // 8, y // 8, 59) > 0.87 else 0.0
            rock_pixels.append(
                (
                    _clamp_byte(48 + 38 * vein + 16 * grain - 10 * algae),
                    _clamp_byte(57 + 42 * vein + 18 * grain + 24 * algae),
                    _clamp_byte(50 + 34 * vein + 12 * grain + 5 * algae),
                )
            )

    assets = {
        "seafloor_albedo": texture_dir / "seafloor_albedo.png",
        "rock_albedo": texture_dir / "rock_albedo.png",
    }
    _write_png_rgb(assets["seafloor_albedo"], width=width, height=height, pixels=seafloor_pixels)
    _write_png_rgb(assets["rock_albedo"], width=width, height=height, pixels=rock_pixels)
    return assets


def _bind(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _mission_translate_op(
    xformable: Any,
    *,
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None = None,
) -> Any:
    from pxr import Gf

    translate = xformable.AddTranslateOp(opSuffix="mission")
    if mission_positions and len(mission_positions) >= 2:
        positions = [Gf.Vec3d(*position) for position in mission_positions]
        translate.Set(positions[-1])
        for index, position in enumerate(positions):
            frame = 1 + round(index * max(0, frames - 1) / max(1, len(positions) - 1))
            translate.Set(position, frame)
        return translate

    start = Gf.Vec3d(-1.9, -0.18, -10.0)
    finish = Gf.Vec3d(1.36, 0.0, -9.99)
    translate.Set(finish)
    for frame in range(1, max(2, frames) + 1):
        alpha = (frame - 1) / max(1, frames - 1)
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        translate.Set(start * (1.0 - eased) + finish * eased, frame)
    return translate


def _mission_rotate_op(
    xformable: Any,
    *,
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None = None,
) -> Any:
    from math import atan2, degrees, sin

    from pxr import Gf

    rotate = xformable.AddRotateXYZOp(opSuffix="mission")
    if mission_positions and len(mission_positions) >= 2:
        final_yaw = 0.0
        for index, position in enumerate(mission_positions):
            prev_position = mission_positions[max(0, index - 1)]
            dx = position[0] - prev_position[0]
            dy = position[1] - prev_position[1]
            yaw = degrees(atan2(dy, dx)) if abs(dx) + abs(dy) > 1e-6 else final_yaw
            final_yaw = yaw
            frame = 1 + round(index * max(0, frames - 1) / max(1, len(mission_positions) - 1))
            alpha = index / max(1, len(mission_positions) - 1)
            pitch = -1.8 * sin(alpha * 3.14159)
            roll = 1.2 * sin(alpha * 6.28318)
            rotate.Set(Gf.Vec3f(pitch, roll, yaw + 3.0), frame)
        rotate.Set(Gf.Vec3f(0.0, 0.0, final_yaw + 3.0))
        return rotate

    rotate.Set(Gf.Vec3f(0.0, 0.0, 3.0))
    return rotate


def _cube(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: Any,
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate != (0.0, 0.0, 0.0):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    _bind(cube.GetPrim(), material)
    return cube


def _cylinder(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    radius: float,
    height: float,
    material: Any,
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    from pxr import Gf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate != (0.0, 0.0, 0.0):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    _bind(cylinder.GetPrim(), material)
    return cylinder


def _cone(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    radius: float,
    height: float,
    material: Any,
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    from pxr import Gf, UsdGeom

    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    xform = UsdGeom.Xformable(cone.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate != (0.0, 0.0, 0.0):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    _bind(cone.GetPrim(), material)
    return cone


def _curve(
    stage: Any,
    path: str,
    *,
    points: tuple[tuple[float, float, float], ...],
    width: float,
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    curve.CreateWidthsAttr([width])
    _bind(curve.GetPrim(), material)
    return curve


def _sphere(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    radius: float,
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translate))
    _bind(sphere.GetPrim(), material)
    return sphere


def _ellipsoid(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(1.0)
    sphere.CreateDoubleSidedAttr(True)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    _bind(sphere.GetPrim(), material)
    return sphere


def _irregular_rock(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    yaw: float,
    seed: int,
    material: Any,
) -> Any:
    from math import cos, radians, sin

    from pxr import Gf, Sdf, UsdGeom

    lat_steps = 5
    lon_steps = 10
    yaw_rad = radians(yaw)
    points = []
    uvs = []
    for lat_index in range(lat_steps + 1):
        theta = -1.5708 + 3.14159 * lat_index / lat_steps
        for lon_index in range(lon_steps):
            phi = 6.28318 * lon_index / lon_steps
            noise = (
                1.0
                + 0.13 * sin(seed * 1.73 + lon_index * 2.17)
                + 0.08 * cos(seed * 0.91 + lat_index * 1.41 + lon_index * 0.63)
            )
            x = scale[0] * noise * cos(theta) * cos(phi)
            y = scale[1] * noise * cos(theta) * sin(phi)
            z = scale[2] * noise * sin(theta)
            if lat_index == 0:
                z *= 0.35
            rx = x * cos(yaw_rad) - y * sin(yaw_rad)
            ry = x * sin(yaw_rad) + y * cos(yaw_rad)
            points.append(Gf.Vec3f(translate[0] + rx, translate[1] + ry, translate[2] + z))
            uvs.append(Gf.Vec2f(2.0 * lon_index / lon_steps, lat_index / lat_steps))

    counts: list[int] = []
    indices: list[int] = []
    for lat_index in range(lat_steps):
        for lon_index in range(lon_steps):
            a = lat_index * lon_steps + lon_index
            b = lat_index * lon_steps + (lon_index + 1) % lon_steps
            c = (lat_index + 1) * lon_steps + (lon_index + 1) % lon_steps
            d = (lat_index + 1) * lon_steps + lon_index
            counts.append(4)
            indices.extend((a, b, c, d))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreatePointsAttr(points)
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(uvs)
    _bind(mesh.GetPrim(), material)
    return mesh


def _vertical_water_plane(
    stage: Any,
    path: str,
    *,
    y: float,
    z: float,
    width: float,
    height: float,
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom

    half_width = width * 0.5
    half_height = height * 0.5
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-half_width, y, z - half_height),
            Gf.Vec3f(half_width, y, z - half_height),
            Gf.Vec3f(half_width, y, z + half_height),
            Gf.Vec3f(-half_width, y, z + half_height),
        ]
    )
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), material)
    return mesh


def _light_shaft_sheet(
    stage: Any,
    path: str,
    *,
    top: tuple[float, float, float],
    bottom: tuple[float, float, float],
    half_width: float,
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(top[0] - half_width, top[1], top[2]),
            Gf.Vec3f(top[0] + half_width, top[1], top[2]),
            Gf.Vec3f(bottom[0] + half_width * 0.35, bottom[1], bottom[2]),
            Gf.Vec3f(bottom[0] - half_width * 0.35, bottom[1], bottom[2]),
        ]
    )
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), material)
    return mesh


def _add_seafloor_detail(stage: Any, *, materials: dict[str, Any]) -> None:
    for index, (x, y, radius, yaw) in enumerate(
        (
            (-3.8, 2.6, 0.18, 14.0),
            (-2.8, 1.1, 0.12, -22.0),
            (-1.7, -0.7, 0.08, 31.0),
            (-1.1, 2.8, 0.16, 8.0),
            (-0.4, 1.4, 0.10, -17.0),
            (0.6, -1.3, 0.13, 26.0),
            (1.2, 2.2, 0.17, -35.0),
            (1.7, 0.95, 0.09, 12.0),
            (2.7, -0.55, 0.11, -28.0),
            (3.4, 1.8, 0.15, 20.0),
        )
    ):
        _irregular_rock(
            stage,
            f"/World/SeafloorDetail/Rock_{index}",
            translate=(x, y, -11.82 + radius * 0.28),
            scale=(radius * 1.75, radius * 0.95, radius * 0.52),
            yaw=yaw,
            seed=index + 3,
            material=materials["rock"],
        )

    for index, (x, y, width) in enumerate(
        (
            (-2.6, -1.15, 0.92),
            (-1.4, 0.92, 0.72),
            (-0.2, -0.62, 0.86),
            (1.05, 1.28, 0.66),
            (2.15, -1.08, 0.76),
            (3.05, 2.18, 0.84),
        )
    ):
        _curve(
            stage,
            f"/World/SeafloorDetail/Ripple_{index}",
            points=(
                (x - width * 0.50, y - 0.03, -11.90),
                (x - width * 0.18, y + 0.04, -11.895),
                (x + width * 0.16, y - 0.02, -11.90),
                (x + width * 0.50, y + 0.035, -11.895),
            ),
            width=0.020,
            material=materials["silt_ridge"],
        )

    for index in range(18):
        x = -3.5 + (index % 9) * 0.78
        y = -1.65 + (index // 9) * 3.25 + 0.18 * ((index * 5) % 3 - 1)
        radius = 0.025 + 0.007 * (index % 4)
        _irregular_rock(
            stage,
            f"/World/SeafloorDetail/Pebble_{index:02d}",
            translate=(x, y, -11.86),
            scale=(radius * 1.6, radius * 1.0, radius * 0.42),
            yaw=13.0 * index,
            seed=31 + index,
            material=materials["shell_hash" if index % 5 == 0 else "rock"],
        )

    for index, (x, y, sx, sy) in enumerate(
        (
            (-2.1, 0.15, 0.46, 0.11),
            (-0.55, -0.92, 0.38, 0.08),
            (0.86, 0.62, 0.52, 0.12),
            (2.30, 0.28, 0.44, 0.10),
        )
    ):
        _ellipsoid(
            stage,
            f"/World/SeafloorDetail/DarkSedimentPatch_{index}",
            translate=(x, y, -11.875),
            scale=(sx, sy, 0.012),
            material=materials["dark_sediment"],
        )

    for index, (x, y, z, radius) in enumerate(
        (
            (-2.1, -0.7, -9.0, 0.007),
            (-1.2, 0.2, -8.4, 0.006),
            (0.2, -0.5, -8.9, 0.007),
            (1.0, 0.4, -9.4, 0.008),
            (1.9, -0.2, -8.7, 0.006),
            (2.8, 0.6, -9.2, 0.007),
        )
    ):
        _sphere(
            stage,
            f"/World/WaterColumn/Particulate_{index}",
            translate=(x, y, z),
            radius=radius,
            material=materials["particulate"],
        )


def _add_water_column_effects(stage: Any, *, materials: dict[str, Any]) -> None:
    from math import cos, sin

    _vertical_water_plane(
        stage,
        "/World/WaterColumn/OpenOceanBackdrop",
        y=18.0,
        z=-9.7,
        width=120.0,
        height=46.0,
        material=materials["deep_water"],
    )
    _vertical_water_plane(
        stage,
        "/World/WaterColumn/MidBackscatter",
        y=11.0,
        z=-9.7,
        width=96.0,
        height=36.0,
        material=materials["water_haze"],
    )
    _vertical_water_plane(
        stage,
        "/World/WaterColumn/NearBackscatter",
        y=6.8,
        z=-9.6,
        width=72.0,
        height=26.0,
        material=materials["water_haze_near"],
    )

    for index, x in enumerate((-4.2, -2.7, 2.5, 4.1)):
        _light_shaft_sheet(
            stage,
            f"/World/WaterColumn/TopLightShaft_{index}",
            top=(x, 5.6 + 0.28 * sin(index), -5.80),
            bottom=(x + 0.22 * sin(index + 0.8), 3.05 + 0.20 * cos(index), -10.15),
            half_width=0.055 + 0.010 * (index % 2),
            material=materials["light_shaft"],
        )

    for index in range(72):
        x = -4.65 + (index % 18) * 0.55 + 0.05 * sin(index * 1.7)
        y = -1.95 + (index // 18) * 1.18 + 0.10 * sin(index * 0.73)
        z = -7.80 - 0.10 * (index % 7) - 0.28 * (index // 18)
        radius = 0.006 + 0.002 * ((index * 7) % 4)
        _sphere(
            stage,
            f"/World/WaterColumn/MarineSnow_{index:02d}",
            translate=(x, y, z),
            radius=radius,
            material=materials["marine_snow"],
        )

    for index in range(10):
        x = -2.9 + index * 0.55
        y = -1.25 + 0.18 * cos(index * 0.9)
        _curve(
            stage,
            f"/World/Caustics/SeafloorRibbon_{index:02d}",
            points=(
                (x, y, -11.90),
                (x + 0.18, y + 0.05 * sin(index), -11.895),
                (x + 0.42, y - 0.04 * cos(index), -11.90),
            ),
            width=0.014 + 0.002 * (index % 3),
            material=materials["caustic_soft"],
        )


def _add_depth_graded_water_volume(stage: Any, *, materials: dict[str, Any]) -> None:
    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/WaterColumn/DepthGradedWaterVolume")
    root.GetPrim().CreateAttribute("oceanscale:visualRole", Sdf.ValueTypeNames.String).Set(
        "camera-to-background depth-graded underwater haze"
    )
    for index, (name, y, z, width, height, material_key) in enumerate(
        (
            ("WorkingDistanceHaze", 2.65, -9.70, 118.0, 42.0, "depth_haze_mid"),
            ("BackgroundFalloff", 7.90, -9.55, 126.0, 48.0, "depth_haze_far"),
        )
    ):
        plane = _vertical_water_plane(
            stage,
            f"/World/WaterColumn/DepthGradedWaterVolume/{name}",
            y=y,
            z=z,
            width=width,
            height=height,
            material=materials[material_key],
        )
        plane.GetPrim().CreateAttribute("oceanscale:depthLayerIndex", Sdf.ValueTypeNames.Int).Set(index)


def _add_caustic_lattice(stage: Any, *, materials: dict[str, Any]) -> None:
    from math import cos, sin

    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/Caustics/CausticLattice")
    root.GetPrim().CreateAttribute("oceanscale:visualRole", Sdf.ValueTypeNames.String).Set(
        "multi-scale seafloor caustic lattice"
    )
    for index in range(18):
        row = index // 6
        col = index % 6
        x = -4.25 + col * 1.48 + 0.08 * sin(index * 1.31)
        y = -2.10 + row * 1.26 + 0.10 * cos(index * 0.73)
        span = 0.58 + 0.08 * ((index + row) % 3)
        _curve(
            stage,
            f"/World/Caustics/CausticLattice/FineRibbon_{index:02d}",
            points=(
                (x - span * 0.50, y + 0.020 * sin(index), -11.905),
                (x - span * 0.18, y + 0.060 * cos(index * 0.7), -11.895),
                (x + span * 0.16, y - 0.050 * sin(index * 0.9), -11.900),
                (x + span * 0.50, y + 0.030 * cos(index * 1.1), -11.905),
            ),
            width=0.0045 + 0.001 * (index % 3),
            material=materials["caustic_lattice"],
        )

    for index, (x, y, yaw) in enumerate(
        (
            (-0.05, -0.62, -6.0),
            (0.62, -0.46, 8.0),
            (1.26, -0.28, -10.0),
            (1.88, -0.10, 5.0),
        )
    ):
        _cube(
            stage,
            f"/World/Caustics/CausticLattice/PipelineCaustic_{index}",
            translate=(x, y, -11.905),
            scale=(0.42, 0.006, 0.004),
            rotate=(0.0, 0.0, yaw),
            material=materials["caustic_lattice"],
        )


def _wave_surface(stage: Any, path: str, *, frames: int, material: Any) -> Any:
    from math import sin

    from pxr import Gf, UsdGeom

    size = 12.0
    cells = 32
    points = []
    for y in range(cells + 1):
        for x in range(cells + 1):
            px = -size * 0.5 + size * x / cells
            py = -size * 0.5 + size * y / cells
            points.append(Gf.Vec3f(px, py, -0.02))

    counts: list[int] = []
    indices: list[int] = []
    for y in range(cells):
        for x in range(cells):
            i = y * (cells + 1) + x
            counts.append(4)
            indices.extend([i, i + 1, i + cells + 2, i + cells + 1])

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    points_attr = mesh.CreatePointsAttr(points)

    for frame in (1, max(1, frames // 2), max(1, frames)):
        t = frame / 30.0
        animated = []
        for point in points:
            z = 0.12 * sin(point[0] * 1.15 + t * 1.7) + 0.06 * sin(point[1] * 1.9 - t * 1.1)
            animated.append(Gf.Vec3f(point[0], point[1], z))
        points_attr.Set(animated, frame)

    _bind(mesh.GetPrim(), material)
    return mesh


def _seafloor_mesh(stage: Any, path: str, *, material: Any) -> Any:
    from math import cos, sin

    from pxr import Gf, Sdf, UsdGeom

    width = 12.0
    depth = 11.5
    x_cells = 32
    y_cells = 28
    y_start = -4.0
    points = []
    uvs = []
    for y_index in range(y_cells + 1):
        for x_index in range(x_cells + 1):
            x = -width * 0.5 + width * x_index / x_cells
            y = y_start + depth * y_index / y_cells
            z = (
                -11.94
                + 0.055 * sin(x * 0.95 + y * 0.34)
                + 0.035 * cos(x * 1.8 - y * 0.52)
                + 0.012 * y_index / y_cells
            )
            points.append(Gf.Vec3f(x, y, z))
            uvs.append(Gf.Vec2f(5.2 * x_index / x_cells, 6.4 * y_index / y_cells))

    counts: list[int] = []
    indices: list[int] = []
    for y_index in range(y_cells):
        for x_index in range(x_cells):
            base = y_index * (x_cells + 1) + x_index
            counts.append(4)
            indices.extend([base, base + 1, base + x_cells + 2, base + x_cells + 1])

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreatePointsAttr(points)
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(uvs)
    _bind(mesh.GetPrim(), material)
    return mesh


def _mission_path(
    stage: Any,
    path: str,
    *,
    material: Any,
    mission_positions: list[tuple[float, float, float]] | None = None,
) -> Any:
    from pxr import Gf, UsdGeom

    if mission_positions and len(mission_positions) >= 2:
        points = [Gf.Vec3f(*position) for position in mission_positions]
    else:
        points = [
            Gf.Vec3f(-1.9, -0.18, -10.0),
            Gf.Vec3f(-0.8, -0.08, -10.0),
            Gf.Vec3f(0.35, -0.02, -9.98),
            Gf.Vec3f(1.36, 0.0, -9.99),
        ]
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([0.00035])
    _bind(curve.GetPrim(), material)
    return curve


def _add_robot_lighting_rig(
    stage: Any,
    *,
    materials: dict[str, Any],
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None,
) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    root = UsdGeom.Xform.Define(stage, "/World/RobotLightingRig")
    xform = UsdGeom.Xformable(root.GetPrim())
    _mission_translate_op(xform, frames=frames, mission_positions=mission_positions)
    _mission_rotate_op(xform, frames=frames, mission_positions=mission_positions)

    for index, y in enumerate((-0.16, 0.16)):
        _sphere(
            stage,
            f"/World/RobotLightingRig/Headlamp_{index}",
            translate=(0.54, y, -0.09),
            radius=0.006,
            material=materials["front_led"],
        )
        light = UsdLux.SphereLight.Define(stage, f"/World/RobotLightingRig/HeadlampGlow_{index}")
        light.CreateIntensityAttr(60.0)
        light.CreateRadiusAttr(0.08)
        light.CreateColorAttr(Gf.Vec3f(0.66, 0.96, 0.90))
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.68, y, -0.11))

    for index in range(8):
        x = 0.66 + 0.050 * index
        y = -0.18 + 0.02 * (index % 5)
        z = -0.18 + 0.015 * ((index * 3) % 4)
        _sphere(
            stage,
            f"/World/RobotLightingRig/BeamParticulate_{index:02d}",
            translate=(x, y, z),
            radius=0.0015 + 0.0005 * (index % 3),
            material=materials["lit_particulate"],
        )


def _add_robot_weathering_detail(
    stage: Any,
    *,
    materials: dict[str, Any],
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None,
) -> None:
    from math import sin

    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/RobotWeatheringDetail")
    root.GetPrim().CreateAttribute("oceanscale:visualRole", Sdf.ValueTypeNames.String).Set(
        "robot weathering and inspection-camera material detail layer"
    )
    root.GetPrim().CreateAttribute("oceanscale:weatheredRobotOverlay", Sdf.ValueTypeNames.Bool).Set(
        True
    )
    xform = UsdGeom.Xformable(root.GetPrim())
    _mission_translate_op(xform, frames=frames, mission_positions=mission_positions)
    _mission_rotate_op(xform, frames=frames, mission_positions=mission_positions)

    for index, (x, y, z, sx, sy, yaw) in enumerate(
        (
            (-0.16, -0.055, 0.345, 0.060, 0.003, -5.0),
            (0.08, 0.050, 0.350, 0.070, 0.003, 4.0),
            (0.27, -0.030, 0.342, 0.050, 0.003, 8.0),
        )
    ):
        _cube(
            stage,
            f"/World/RobotWeatheringDetail/SaltScuff_{index}",
            translate=(x, y, z),
            scale=(sx, sy, 0.0012),
            rotate=(0.0, 0.0, yaw),
            material=materials["robot_salt_scuff"],
        )

    for index, (x, y, z, sx, sy) in enumerate(
        (
            (0.36, -0.175, 0.030, 0.018, 0.004),
            (0.28, 0.175, 0.026, 0.015, 0.004),
            (-0.28, -0.165, 0.025, 0.014, 0.004),
            (-0.35, 0.155, 0.022, 0.016, 0.004),
        )
    ):
        _ellipsoid(
            stage,
            f"/World/RobotWeatheringDetail/BiofilmPatch_{index}",
            translate=(x, y, z),
            scale=(sx, sy, 0.006),
            material=materials["robot_biofilm"],
        )

    for index, (x, y, z, dx, dz) in enumerate(
        (
            (0.30, -0.210, 0.125, 0.045, -0.025),
            (0.08, -0.220, 0.105, 0.040, -0.030),
            (-0.16, 0.215, 0.115, 0.040, -0.025),
            (-0.32, 0.205, 0.070, 0.035, -0.022),
        )
    ):
        _curve(
            stage,
            f"/World/RobotWeatheringDetail/SiltRunoff_{index}",
            points=(
                (x, y, z),
                (x + dx * 0.45, y + 0.004 * sin(index + 1), z + dz * 0.55),
                (x + dx, y - 0.006 * sin(index + 2), z + dz),
            ),
            width=0.0025 + 0.0005 * (index % 2),
            material=materials["robot_silt_streak"],
        )


def _add_robot_thruster_wake_detail(
    stage: Any,
    *,
    materials: dict[str, Any],
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None,
) -> None:
    from math import sin

    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/RobotThrusterWakeDetail")
    root.GetPrim().CreateAttribute("oceanscale:visualRole", Sdf.ValueTypeNames.String).Set(
        "robot-relative thruster wake and bubble detail layer"
    )
    root.GetPrim().CreateAttribute("oceanscale:robotRelativeWake", Sdf.ValueTypeNames.Bool).Set(True)
    xform = UsdGeom.Xformable(root.GetPrim())
    _mission_translate_op(xform, frames=frames, mission_positions=mission_positions)
    _mission_rotate_op(xform, frames=frames, mission_positions=mission_positions)

    for index, (y, z, lateral_drift) in enumerate(
        (
            (-0.205, -0.135, -0.012),
            (0.205, -0.135, 0.012),
            (-0.170, 0.095, -0.018),
            (0.170, 0.095, 0.018),
        )
    ):
        _curve(
            stage,
            f"/World/RobotThrusterWakeDetail/WakePlume_{index}",
            points=(
                (-0.36, y, z),
                (-0.46, y + lateral_drift * 0.35, z + 0.006 * sin(index + 0.4)),
                (-0.61, y + lateral_drift, z + 0.012 * sin(index + 1.2)),
            ),
            width=0.0045,
            material=materials["wake_mist"],
        )
        _curve(
            stage,
            f"/World/RobotThrusterWakeDetail/WakeShear_{index}",
            points=(
                (-0.35, y, z),
                (-0.54, y + lateral_drift, z + 0.012 * sin(index + 1)),
                (-0.78, y + lateral_drift * 1.7, z + 0.020 * sin(index + 2)),
            ),
            width=0.003 + 0.0005 * (index % 2),
            material=materials["wake_mist"],
        )
        for bubble in range(4):
            _sphere(
                stage,
                f"/World/RobotThrusterWakeDetail/Bubble_{index}_{bubble}",
                translate=(
                    -0.54 - bubble * 0.070 + 0.010 * sin(index + bubble),
                    y + lateral_drift * bubble + 0.010 * sin(1.7 * bubble + index),
                    z + 0.018 * bubble + 0.006 * sin(index * 0.9 + bubble),
                ),
                radius=0.004 + 0.0012 * bubble,
                material=materials["bubble"],
            )


def _add_robot_reference(
    stage: Any,
    robot_usd: Path,
    *,
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None = None,
    visible: bool = True,
) -> Any:
    from pxr import Sdf, UsdGeom

    robot = UsdGeom.Xform.Define(stage, "/World/BlueROV2Heavy")
    robot.GetPrim().GetReferences().AddReference(str(robot_usd), "/World/BlueROV2Heavy")
    robot.GetPrim().CreateAttribute("oceanscale:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        str(robot_usd)
    )
    robot.GetPrim().CreateAttribute("oceanscale:thrusterCount", Sdf.ValueTypeNames.Int).Set(8)
    robot.GetPrim().CreateAttribute("oceanscale:vehicle", Sdf.ValueTypeNames.String).Set(
        "BlueROV2 Heavy"
    )

    xform = UsdGeom.Xformable(robot.GetPrim())
    _mission_translate_op(xform, frames=frames, mission_positions=mission_positions)
    _mission_rotate_op(xform, frames=frames, mission_positions=mission_positions)
    if not visible:
        UsdGeom.Imageable(robot.GetPrim()).MakeInvisible()
    return robot


def _add_external_robot_visual(
    stage: Any,
    external_robot_usd: Path,
    *,
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None,
    scale: float,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/ExternalRobotVisual")
    reference_prim_path = _external_robot_reference_prim_path(external_robot_usd)
    if reference_prim_path is None:
        root.GetPrim().GetReferences().AddReference(str(external_robot_usd))
    else:
        root.GetPrim().GetReferences().AddReference(str(external_robot_usd), reference_prim_path)
    root.GetPrim().CreateAttribute("oceanscale:externalRobotVisual", Sdf.ValueTypeNames.Bool).Set(True)
    root.GetPrim().CreateAttribute("oceanscale:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        str(external_robot_usd)
    )
    if reference_prim_path is not None:
        root.GetPrim().CreateAttribute("oceanscale:sourceAssetPrim", Sdf.ValueTypeNames.String).Set(
            reference_prim_path
        )
    root.GetPrim().CreateAttribute("oceanscale:visualRole", Sdf.ValueTypeNames.String).Set(
        "licensed high-fidelity robot mesh"
    )
    xform = UsdGeom.Xformable(root.GetPrim())
    _mission_translate_op(xform, frames=frames, mission_positions=mission_positions)
    _mission_rotate_op(xform, frames=frames, mission_positions=mission_positions)
    if scale != 1.0:
        xform.AddScaleOp(opSuffix="externalMesh").Set(Gf.Vec3f(scale, scale, scale))


def _external_robot_reference_prim_path(external_robot_usd: Path) -> str | None:
    from pxr import Usd

    asset_stage = Usd.Stage.Open(str(external_robot_usd))
    if asset_stage is None:
        return None

    default_prim = asset_stage.GetDefaultPrim()
    if default_prim.IsValid():
        return str(default_prim.GetPath())

    for candidate in ("/World/BlueROV2Heavy", "/BlueROV2Heavy"):
        if asset_stage.GetPrimAtPath(candidate).IsValid():
            return candidate
    return None


def _add_robot_visual_shell(
    stage: Any,
    *,
    materials: dict[str, Any],
    frames: int,
    mission_positions: list[tuple[float, float, float]] | None = None,
) -> None:
    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/BlueROV2HeavyVisual")
    root.GetPrim().CreateAttribute("oceanscale:visualShell", Sdf.ValueTypeNames.Bool).Set(True)
    root.GetPrim().CreateAttribute("oceanscale:thrusterCount", Sdf.ValueTypeNames.Int).Set(8)
    root_xform = UsdGeom.Xformable(root.GetPrim())
    _mission_translate_op(root_xform, frames=frames, mission_positions=mission_positions)
    _mission_rotate_op(root_xform, frames=frames, mission_positions=mission_positions)

    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/Hull",
        translate=(0.0, 0.0, 0.0),
        scale=(0.48, 0.30, 0.16),
        material=materials["robot_hull"],
    )
    _cylinder(
        stage,
        "/World/BlueROV2HeavyVisual/ElectronicsTube",
        translate=(0.02, 0.0, 0.02),
        radius=0.165,
        height=0.72,
        rotate=(0.0, 90.0, 0.0),
        material=materials["robot_hull"],
    )
    for name, x in (("ForwardEndCap", 0.39), ("AftEndCap", -0.35)):
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/{name}",
            translate=(x, 0.0, 0.02),
            radius=0.172,
            height=0.025,
            rotate=(0.0, 90.0, 0.0),
            material=materials["rubber"],
        )
        for index, (y, z) in enumerate(
            ((0.0, 0.17), (0.12, 0.12), (0.17, 0.0), (0.12, -0.12), (0.0, -0.17), (-0.12, -0.12), (-0.17, 0.0), (-0.12, 0.12))
        ):
            _sphere(
                stage,
                f"/World/BlueROV2HeavyVisual/{name}Bolt_{index}",
                translate=(x + (0.018 if x > 0.0 else -0.018), y, 0.02 + z),
                radius=0.010,
                material=materials["stainless"],
            )
    for index, y in enumerate((-0.31, 0.31)):
        _cube(
            stage,
            f"/World/BlueROV2HeavyVisual/AcrylicSidePlate_{index}",
            translate=(0.02, y, -0.01),
            scale=(0.44, 0.012, 0.21),
            material=materials["acrylic_plate"],
        )
        for bolt_index, (x, z) in enumerate(((-0.22, 0.12), (0.22, 0.12), (-0.22, -0.12), (0.22, -0.12))):
            _sphere(
                stage,
                f"/World/BlueROV2HeavyVisual/AcrylicSidePlate_{index}_Bolt_{bolt_index}",
                translate=(x, y + (0.018 if y > 0.0 else -0.018), z),
                radius=0.012,
                material=materials["stainless"],
            )
    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/TopFloat",
        translate=(0.0, 0.0, 0.28),
        scale=(0.58, 0.24, 0.10),
        material=materials["robot_float"],
    )
    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/BottomSkid",
        translate=(0.0, 0.0, -0.24),
        scale=(0.56, 0.26, 0.035),
        material=materials["robot_frame"],
    )
    _sphere(
        stage,
        "/World/BlueROV2HeavyVisual/FrontSensor",
        translate=(0.50, 0.0, 0.02),
        radius=0.075,
        material=materials["sensor_glass"],
    )
    _sphere(
        stage,
        "/World/BlueROV2HeavyVisual/CameraDome",
        translate=(0.57, 0.0, -0.02),
        radius=0.052,
        material=materials["camera_glass"],
    )
    _cylinder(
        stage,
        "/World/BlueROV2HeavyVisual/ForwardSonar",
        translate=(0.58, 0.0, 0.095),
        radius=0.040,
        height=0.045,
        rotate=(0.0, 90.0, 0.0),
        material=materials["payload_sensor"],
    )
    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/DvlPod",
        translate=(0.02, 0.0, -0.31),
        scale=(0.18, 0.13, 0.045),
        material=materials["payload_sensor"],
    )
    for index, (x, y) in enumerate(((0.09, 0.055), (0.09, -0.055), (-0.09, 0.055), (-0.09, -0.055))):
        _sphere(
            stage,
            f"/World/BlueROV2HeavyVisual/DvlTransducer_{index}",
            translate=(x, y, -0.36),
            radius=0.020,
            material=materials["camera_glass"],
        )
    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/OceanScaleNameplate",
        translate=(0.22, -0.326, 0.09),
        scale=(0.18, 0.006, 0.035),
        material=materials["nameplate"],
    )
    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/ManipulatorBase",
        translate=(0.52, -0.18, -0.16),
        scale=(0.055, 0.050, 0.040),
        material=materials["robot_frame"],
    )
    _cube(
        stage,
        "/World/BlueROV2HeavyVisual/ManipulatorArm",
        translate=(0.72, -0.18, -0.16),
        scale=(0.22, 0.020, 0.020),
        material=materials["robot_frame"],
    )
    for index, y in enumerate((-0.215, -0.145)):
        _cube(
            stage,
            f"/World/BlueROV2HeavyVisual/ManipulatorClaw_{index}",
            translate=(0.94, y, -0.14),
            scale=(0.08, 0.010, 0.016),
            material=materials["robot_frame"],
        )
    _curve(
        stage,
        "/World/BlueROV2HeavyVisual/TetherLead",
        points=((-0.48, 0.0, 0.15), (-0.72, 0.02, 0.24), (-1.05, 0.05, 0.42)),
        width=0.022,
        material=materials["tether_cable"],
    )
    _cylinder(
        stage,
        "/World/BlueROV2HeavyVisual/TetherStrainRelief",
        translate=(-0.48, 0.0, 0.15),
        radius=0.030,
        height=0.11,
        rotate=(0.0, 90.0, 0.0),
        material=materials["rubber"],
    )
    for ring_index, x in enumerate((-0.54, -0.60, -0.66)):
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/TetherBendLimiter_{ring_index}",
            translate=(x, 0.004 * ring_index, 0.18 + 0.020 * ring_index),
            radius=0.026,
            height=0.020,
            rotate=(0.0, 72.0, 0.0),
            material=materials["rubber"],
        )
    for index, y in enumerate((-0.13, 0.13)):
        _sphere(
            stage,
            f"/World/BlueROV2HeavyVisual/ForwardLed_{index}",
            translate=(0.52, y, -0.07),
            radius=0.045,
            material=materials["front_led"],
        )
        _cube(
            stage,
            f"/World/BlueROV2HeavyVisual/ForwardBeam_{index}",
            translate=(1.02, y, -0.07),
            scale=(0.58, 0.018, 0.018),
            material=materials["front_beam"],
        )

    for index, y in enumerate((-0.34, 0.34)):
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/SideRail_{index}",
            translate=(0.0, y, -0.03),
            radius=0.025,
            height=1.08,
            rotate=(0.0, 90.0, 0.0),
            material=materials["robot_frame"],
        )
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/SkidRail_{index}",
            translate=(0.0, y * 0.82, -0.34),
            radius=0.022,
            height=0.96,
            rotate=(0.0, 90.0, 0.0),
            material=materials["robot_frame"],
        )

    for index, (x, y) in enumerate(
        ((0.42, 0.28), (0.42, -0.28), (-0.42, 0.28), (-0.42, -0.28), (0.0, 0.28), (0.0, -0.28))
    ):
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/FrameStandoff_{index}",
            translate=(x, y, -0.08),
            radius=0.014,
            height=0.50,
            material=materials["stainless"],
        )

    horizontal_thrusters = [
        (0.24, 0.34, 0.08),
        (0.24, -0.34, 0.08),
        (-0.24, 0.34, 0.08),
        (-0.24, -0.34, 0.08),
    ]
    vertical_thrusters = [
        (0.34, 0.24, -0.12),
        (0.34, -0.24, -0.12),
        (-0.34, 0.24, -0.12),
        (-0.34, -0.24, -0.12),
    ]
    for index, (x, y, z) in enumerate(horizontal_thrusters, start=1):
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/Thruster_{index}",
            translate=(x, y, z),
            radius=0.065,
            height=0.13,
            rotate=(90.0, 0.0, 0.0),
            material=materials["thruster"],
        )
        _cube(
            stage,
            f"/World/BlueROV2HeavyVisual/Thruster_{index}_Blade",
            translate=(x, y, z),
            scale=(0.018, 0.006, 0.085),
            material=materials["propeller"],
        )
        for blade_index, blade_z in enumerate((-0.050, 0.050), start=2):
            _cube(
                stage,
                f"/World/BlueROV2HeavyVisual/Thruster_{index}_RotorBlade_{blade_index}",
                translate=(x, y, z + blade_z),
                scale=(0.070, 0.005, 0.012),
                material=materials["propeller"],
            )
        for bubble in range(3):
            _sphere(
                stage,
                f"/World/BlueROV2HeavyVisual/Thruster_{index}_Bubble_{bubble}",
                translate=(x - 0.15 - bubble * 0.08, y + 0.015 * (bubble - 1), z + 0.02 * bubble),
                radius=0.016 + 0.005 * bubble,
                material=materials["bubble"],
            )
        _curve(
            stage,
            f"/World/BlueROV2HeavyVisual/Thruster_{index}_Wake",
            points=(
                (x - 0.07, y, z),
                (x - 0.30, y + 0.02, z + 0.015),
                (x - 0.50, y - 0.015, z + 0.030),
            ),
            width=0.020,
            material=materials["wake"],
        )
    for index, (x, y, z) in enumerate(vertical_thrusters, start=5):
        _cylinder(
            stage,
            f"/World/BlueROV2HeavyVisual/Thruster_{index}",
            translate=(x, y, z),
            radius=0.060,
            height=0.13,
            material=materials["thruster"],
        )
        _cube(
            stage,
            f"/World/BlueROV2HeavyVisual/Thruster_{index}_Blade",
            translate=(x, y, z),
            scale=(0.078, 0.006, 0.014),
            material=materials["propeller"],
        )
        for blade_index, blade_y in enumerate((-0.045, 0.045), start=2):
            _cube(
                stage,
                f"/World/BlueROV2HeavyVisual/Thruster_{index}_RotorBlade_{blade_index}",
                translate=(x, y + blade_y, z),
                scale=(0.014, 0.070, 0.006),
                material=materials["propeller"],
            )
        for bubble in range(2):
            _sphere(
                stage,
                f"/World/BlueROV2HeavyVisual/Thruster_{index}_Bubble_{bubble}",
                translate=(x + 0.025 * (bubble - 0.5), y, z + 0.15 + bubble * 0.10),
                radius=0.014 + 0.006 * bubble,
                material=materials["bubble"],
            )


def _add_subsea_inspection_target(stage: Any, *, materials: dict[str, Any]) -> None:
    from math import sin

    _cylinder(
        stage,
        "/World/SubseaInspection/Pipeline",
        translate=(2.25, 0.05, -11.62),
        radius=0.075,
        height=2.15,
        rotate=(0.0, 90.0, 0.0),
        material=materials["pipeline"],
    )
    _cylinder(
        stage,
        "/World/SubseaInspection/PipelineSleeve",
        translate=(1.36, 0.05, -11.58),
        radius=0.092,
        height=0.34,
        rotate=(0.0, 90.0, 0.0),
        material=materials["pipeline_sleeve"],
    )
    _cylinder(
        stage,
        "/World/SubseaInspection/BeaconPost",
        translate=(1.36, 0.0, -11.32),
        radius=0.026,
        height=0.34,
        material=materials["pipeline_sleeve"],
    )
    _cylinder(
        stage,
        "/World/SubseaInspection/AcousticBeacon",
        translate=(1.36, 0.0, -11.12),
        radius=0.045,
        height=0.12,
        material=materials["beacon_lens"],
    )
    for index, y in enumerate((-0.095, 0.095)):
        _cylinder(
            stage,
            f"/World/SubseaInspection/PipelineClampBolt_{index}",
            translate=(1.36, y, -11.46),
            radius=0.012,
            height=0.07,
            rotate=(90.0, 0.0, 0.0),
            material=materials["pipeline_sleeve"],
        )
    for index, (x, y, z, sx, sy) in enumerate(
        (
            (1.02, -0.04, -11.50, 0.16, 0.030),
            (1.48, 0.12, -11.53, 0.20, 0.036),
            (2.02, -0.08, -11.55, 0.18, 0.032),
            (2.66, 0.10, -11.57, 0.22, 0.040),
        )
    ):
        _ellipsoid(
            stage,
            f"/World/SubseaInspection/BiofoulingPatch_{index}",
            translate=(x, y, z),
            scale=(sx, sy, 0.018),
            material=materials["biofouling"],
        )
    for index, (x, y) in enumerate(((1.12, -0.11), (1.56, 0.16), (2.34, -0.13))):
        _curve(
            stage,
            f"/World/SubseaInspection/MarineGrowth_{index}",
            points=(
                (x, y, -11.52),
                (x + 0.03, y + 0.02 * sin(index + 1), -11.42),
                (x + 0.06, y - 0.01, -11.35),
            ),
            width=0.010,
            material=materials["marine_growth"],
        )
    _curve(
        stage,
        "/World/SubseaInspection/SeabedCable",
        points=(
            (-0.35, -0.68, -11.83),
            (0.22, -0.52, -11.82),
            (0.78, -0.30, -11.80),
            (1.36, -0.04, -11.77),
            (2.18, 0.18, -11.76),
        ),
        width=0.018,
        material=materials["tether_cable"],
    )
    for index, x in enumerate((0.82, 1.90, 2.72)):
        _ellipsoid(
            stage,
            f"/World/SubseaInspection/SiltDrift_{index}",
            translate=(x, 0.18 - index * 0.15, -11.72),
            scale=(0.34, 0.10, 0.025),
            material=materials["silt_ridge"],
        )


def _add_lights(stage: Any) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(4.5)
    dome.CreateColorAttr(Gf.Vec3f(0.46, 0.82, 0.86))

    key = UsdLux.RectLight.Define(stage, "/World/Lighting/SoftTopKey")
    key.CreateIntensityAttr(6200.0)
    key.CreateWidthAttr(5.5)
    key.CreateHeightAttr(4.0)
    key.CreateColorAttr(Gf.Vec3f(0.62, 0.92, 0.86))
    key_xform = UsdLux.RectLight(key).GetPrim()

    light_xform = UsdGeom.Xformable(key_xform)
    light_xform.AddTranslateOp().Set(Gf.Vec3d(-1.2, -2.7, -6.0))
    light_xform.AddRotateXYZOp().Set(Gf.Vec3f(58.0, 0.0, -20.0))

    fill = UsdLux.SphereLight.Define(stage, "/World/Lighting/RobotFill")
    fill.CreateIntensityAttr(1900.0)
    fill.CreateRadiusAttr(0.9)
    fill.CreateColorAttr(Gf.Vec3f(0.18, 0.72, 0.78))
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, -1.25, -9.0))

    target = UsdLux.SphereLight.Define(stage, "/World/Lighting/AcousticBeaconGlow")
    target.CreateIntensityAttr(8.0)
    target.CreateRadiusAttr(0.04)
    target.CreateColorAttr(Gf.Vec3f(0.52, 0.72, 0.66))
    UsdGeom.Xformable(target.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(1.36, 0.0, -11.08))

    camera_fill = UsdLux.SphereLight.Define(stage, "/World/Lighting/CameraSoftbox")
    camera_fill.CreateIntensityAttr(1600.0)
    camera_fill.CreateRadiusAttr(1.8)
    camera_fill.CreateColorAttr(Gf.Vec3f(0.42, 0.86, 0.90))
    UsdGeom.Xformable(camera_fill.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(-3.6, -3.2, -8.25)
    )


def _add_camera(stage: Any) -> None:
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(30.0)
    camera.CreateFocusDistanceAttr(3.25)
    camera.CreateFStopAttr(96.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 500.0))
    camera.CreateHorizontalApertureAttr(29.0)
    eye = Gf.Vec3d(-1.88, -2.18, -8.86)
    target = Gf.Vec3d(0.42, -0.04, -10.20)
    view = Gf.Matrix4d(1.0).SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTransformOp().Set(view.GetInverse())
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def _mission_float(mission_summary: dict[str, Any], section: str, name: str, default: float) -> float:
    value = mission_summary.get(section, {}).get(name, default)
    return float(value) if isinstance(value, (int, float)) else default


def _add_telemetry_panel(
    stage: Any,
    *,
    materials: dict[str, Any],
    mission_summary: dict[str, Any] | None,
) -> None:
    if mission_summary is None:
        return

    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/MissionTelemetry")
    final_distance = _mission_float(mission_summary, "metrics", "final_distance_to_target_m", 0.0)
    min_distance = _mission_float(mission_summary, "metrics", "min_distance_to_target_m", final_distance)
    start_distance = _mission_float(mission_summary, "metrics", "start_distance_to_target_m", 1.0)
    sonar_rate = _mission_float(mission_summary, "metrics", "sonar_detection_rate", 0.0)
    steps_per_sec = _mission_float(mission_summary, "metrics", "steps_per_sec", 0.0)
    completed = bool(mission_summary.get("mission", {}).get("completed", False))

    root.GetPrim().CreateAttribute("oceanscale:mvpCompleted", Sdf.ValueTypeNames.Bool).Set(completed)
    root.GetPrim().CreateAttribute("oceanscale:mvpFinalDistanceM", Sdf.ValueTypeNames.Float).Set(
        final_distance
    )
    root.GetPrim().CreateAttribute("oceanscale:mvpMinDistanceM", Sdf.ValueTypeNames.Float).Set(
        min_distance
    )
    root.GetPrim().CreateAttribute("oceanscale:mvpSonarDetectionRate", Sdf.ValueTypeNames.Float).Set(
        sonar_rate
    )
    root.GetPrim().CreateAttribute("oceanscale:mvpStepsPerSec", Sdf.ValueTypeNames.Float).Set(
        steps_per_sec
    )

    _cube(
        stage,
        "/World/MissionTelemetry/Panel",
        translate=(-2.25, 0.62, -9.15),
        scale=(0.38, 0.018, 0.22),
        material=materials["telemetry_panel"],
    )
    progress = max(0.0, min(1.0, 1.0 - final_distance / max(start_distance, 1e-6)))
    bars = (
        ("DistanceProgress", progress, -8.90),
        ("BestDistanceProgress", max(0.0, min(1.0, 1.0 - min_distance / max(start_distance, 1e-6))), -9.06),
        ("SonarDetection", max(0.0, min(1.0, sonar_rate)), -9.22),
    )
    for name, fraction, z in bars:
        length = 0.30 * max(0.04, fraction)
        _cube(
            stage,
            f"/World/MissionTelemetry/{name}",
            translate=(-2.40 + length * 0.5, 0.58, z - 0.10),
            scale=(length, 0.024, 0.024),
            material=materials["telemetry_good"],
        )
    _sphere(
        stage,
        "/World/MissionTelemetry/CompletionBeacon",
        translate=(-2.55, 0.58, -8.86),
        radius=0.032,
        material=materials["telemetry_good" if completed else "target"],
    )


def _capture_viewport(app: Any, capture_png: Path, *, warmup_frames: int) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utils

    capture_png.parent.mkdir(parents=True, exist_ok=True)
    viewport_window = viewport_utils.get_active_viewport_window()
    viewport_window.viewport_api.camera_path = "/World/Camera"

    for _ in range(max(0, warmup_frames)):
        app.update()

    _capture_next_viewport_frame(app, capture_png)

    return {
        "path": str(capture_png),
        "camera": "/World/Camera",
        "resolution": list(viewport_window.viewport_api.resolution),
        "warmup_frames": warmup_frames,
    }


def _capture_next_viewport_frame(app: Any, path: Path) -> None:
    import omni.kit.renderer_capture
    import omni.kit.viewport.utility as viewport_utils

    path.parent.mkdir(parents=True, exist_ok=True)
    viewport = viewport_utils.get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active Isaac Sim viewport is available for capture")

    capture = viewport_utils.capture_viewport_to_file(viewport, file_path=str(path))
    result = _wait_for_viewport_capture(app, capture, max_frames=240)
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    for _ in range(3):
        app.update()

    if not result or not path.exists():
        raise RuntimeError(f"Isaac Sim viewport capture was not written: {path}")


def _wait_for_viewport_capture(app: Any, capture: Any, *, max_frames: int) -> Any:
    return _run_async_with_app_updates(
        app,
        capture.wait_for_result(completion_frames=3),
        max_frames=max_frames,
        description="Isaac Sim viewport capture",
    )


def _capture_sequence(
    app: Any,
    sequence_dir: Path,
    *,
    authored_frames: int,
    frame_count: int,
    fps: int,
    warmup_frames: int,
) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utils
    import omni.timeline

    if frame_count < 1:
        raise ValueError("--sequence-frame-count must be positive when sequence capture is enabled")

    sequence_dir.mkdir(parents=True, exist_ok=True)
    viewport_window = viewport_utils.get_active_viewport_window()
    viewport_window.viewport_api.camera_path = "/World/Camera"
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_time_codes_per_second(fps)

    for _ in range(max(0, warmup_frames)):
        app.update()

    paths: list[Path] = []
    for index in range(frame_count):
        alpha = index / max(1, frame_count - 1)
        stage_frame = 1 + round(alpha * max(0, authored_frames - 1))
        timeline.set_current_time(stage_frame / fps)
        app.update()
        frame_path = sequence_dir / f"frame_{index:04d}.png"
        _capture_next_viewport_frame(app, frame_path)
        paths.append(frame_path)

    return {
        "directory": str(sequence_dir),
        "frame_count": frame_count,
        "fps": fps,
        "first_frame": str(paths[0]),
        "last_frame": str(paths[-1]),
        "resolution": list(viewport_window.viewport_api.resolution),
    }


def _encode_mp4(sequence_dir: Path, output_mp4: Path, *, fps: int) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --capture-mp4 but was not found on PATH")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(sequence_dir / "frame_%04d.png"),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if not output_mp4.exists():
        raise RuntimeError(f"MP4 was not written: {output_mp4}")
    return {
        "path": str(output_mp4),
        "fps": fps,
        "encoder": "ffmpeg/libx264",
        "ffmpeg_stderr": result.stderr.splitlines()[-6:],
    }


def _build_stage(
    output_usd: Path,
    robot_usd: Path,
    frames: int,
    *,
    mission_summary: dict[str, Any] | None = None,
    external_robot_usd: Path | None = None,
    external_robot_scale: float = 1.0,
    isaaclab_runtime_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(frames)
    stage.SetTimeCodesPerSecond(30)
    mission_positions = _trajectory_positions_from_summary(mission_summary)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "oceanscale:demo": "isaacsim_underwater_robot",
        "oceanscale:isaacLabEnv": "OceanScaleDirectRLEnv",
        "oceanscale:source": "OceanScale Newton+Warp underwater MVP",
    }
    robot_asset_evidence = _robot_asset_evidence(robot_usd)
    external_robot_asset_evidence = (
        _external_usd_asset_evidence(external_robot_usd)
        if external_robot_usd is not None
        else None
    )
    texture_assets = _ensure_demo_texture_assets(output_usd.parent / "procedural_textures")

    materials = {
        "seafloor": _make_textured_material(
            stage,
            "/World/Materials/Seafloor",
            texture_path=texture_assets["seafloor_albedo"],
            fallback_color=(0.105, 0.145, 0.125),
            emissive=(0.006, 0.015, 0.014),
        ),
        "silt_ridge": _make_material(
            stage,
            "/World/Materials/SiltRidge",
            color=(0.16, 0.21, 0.18),
            roughness=0.92,
            emissive=(0.006, 0.014, 0.012),
        ),
        "dark_sediment": _make_material(
            stage,
            "/World/Materials/DarkSediment",
            color=(0.055, 0.085, 0.072),
            roughness=0.96,
            emissive=(0.001, 0.004, 0.003),
        ),
        "shell_hash": _make_material(
            stage,
            "/World/Materials/ShellHash",
            color=(0.42, 0.47, 0.39),
            roughness=0.72,
            emissive=(0.008, 0.012, 0.008),
        ),
        "pipeline": _make_material(
            stage,
            "/World/Materials/AgedPipeline",
            color=(0.18, 0.24, 0.21),
            roughness=0.86,
            metallic=0.22,
            emissive=(0.004, 0.010, 0.008),
        ),
        "pipeline_sleeve": _make_material(
            stage,
            "/World/Materials/BiofouledSleeve",
            color=(0.24, 0.30, 0.25),
            roughness=0.92,
            metallic=0.10,
            emissive=(0.006, 0.014, 0.010),
        ),
        "beacon_lens": _make_material(
            stage,
            "/World/Materials/AcousticBeaconLens",
            color=(0.42, 0.74, 0.66),
            roughness=0.18,
            opacity=0.72,
            emissive=(0.030, 0.085, 0.065),
        ),
        "biofouling": _make_material(
            stage,
            "/World/Materials/Biofouling",
            color=(0.18, 0.25, 0.15),
            roughness=0.96,
            emissive=(0.004, 0.010, 0.004),
        ),
        "marine_growth": _make_material(
            stage,
            "/World/Materials/MarineGrowth",
            color=(0.12, 0.25, 0.18),
            roughness=0.88,
            opacity=0.82,
            emissive=(0.002, 0.014, 0.008),
        ),
        "water": _make_material(
            stage,
            "/World/Materials/WaterVolume",
            color=(0.03, 0.22, 0.24),
            roughness=0.18,
            opacity=0.002,
            emissive=(0.0, 0.002, 0.002),
        ),
        "deep_water": _make_material(
            stage,
            "/World/Materials/DeepWaterBackdrop",
            color=(0.012, 0.095, 0.105),
            roughness=0.30,
            opacity=0.36,
            emissive=(0.0, 0.010, 0.012),
        ),
        "water_haze": _make_material(
            stage,
            "/World/Materials/WaterHaze",
            color=(0.04, 0.33, 0.34),
            roughness=0.24,
            opacity=0.030,
            emissive=(0.001, 0.014, 0.014),
        ),
        "water_haze_near": _make_material(
            stage,
            "/World/Materials/WaterHazeNear",
            color=(0.07, 0.44, 0.43),
            roughness=0.20,
            opacity=0.006,
            emissive=(0.0, 0.004, 0.004),
        ),
        "depth_haze_mid": _make_material(
            stage,
            "/World/Materials/DepthHazeMid",
            color=(0.035, 0.26, 0.27),
            roughness=0.26,
            opacity=0.010,
            emissive=(0.0, 0.007, 0.007),
        ),
        "depth_haze_far": _make_material(
            stage,
            "/World/Materials/DepthHazeFar",
            color=(0.018, 0.15, 0.17),
            roughness=0.30,
            opacity=0.028,
            emissive=(0.0, 0.010, 0.011),
        ),
        "light_shaft": _make_material(
            stage,
            "/World/Materials/LightShaft",
            color=(0.035, 0.18, 0.18),
            roughness=0.28,
            opacity=0.003,
            emissive=(0.0, 0.0, 0.0),
        ),
        "surface": _make_material(
            stage,
            "/World/Materials/WaveSurface",
            color=(0.08, 0.46, 0.46),
            roughness=0.12,
            opacity=0.58,
            emissive=(0.012, 0.07, 0.065),
        ),
        "caustic": _make_material(
            stage,
            "/World/Materials/CausticTrace",
            color=(0.18, 0.62, 0.58),
            emissive=(0.035, 0.18, 0.16),
            opacity=0.28,
        ),
        "caustic_soft": _make_material(
            stage,
            "/World/Materials/CausticSoft",
            color=(0.16, 0.58, 0.54),
            roughness=0.24,
            opacity=0.18,
            emissive=(0.020, 0.12, 0.105),
        ),
        "caustic_lattice": _make_material(
            stage,
            "/World/Materials/CausticLattice",
            color=(0.18, 0.58, 0.54),
            roughness=0.30,
            opacity=0.070,
            emissive=(0.004, 0.042, 0.038),
        ),
        "mission_trace": _make_material(
            stage,
            "/World/Materials/MissionTrace",
            color=(0.20, 0.50, 0.48),
            roughness=0.35,
            opacity=0.018,
            emissive=(0.001, 0.006, 0.006),
        ),
        "rock": _make_textured_material(
            stage,
            "/World/Materials/Rock",
            texture_path=texture_assets["rock_albedo"],
            fallback_color=(0.20, 0.25, 0.22),
            roughness=0.9,
            emissive=(0.01, 0.018, 0.016),
        ),
        "particulate": _make_material(
            stage,
            "/World/Materials/Particulate",
            color=(0.68, 0.86, 0.78),
            roughness=0.45,
            opacity=0.36,
            emissive=(0.025, 0.045, 0.035),
        ),
        "marine_snow": _make_material(
            stage,
            "/World/Materials/MarineSnow",
            color=(0.78, 0.92, 0.86),
            roughness=0.55,
            opacity=0.44,
            emissive=(0.030, 0.055, 0.045),
        ),
        "target": _make_material(
            stage,
            "/World/Materials/Target",
            color=(0.68, 0.56, 0.24),
            opacity=0.38,
            emissive=(0.010, 0.008, 0.002),
        ),
        "robot_hull": _make_material(
            stage,
            "/World/Materials/RobotHull",
            color=(0.70, 0.76, 0.74),
            roughness=0.56,
            metallic=0.05,
        ),
        "robot_float": _make_material(
            stage,
            "/World/Materials/RobotFloat",
            color=(0.88, 0.92, 0.86),
            roughness=0.64,
        ),
        "robot_frame": _make_material(
            stage,
            "/World/Materials/RobotFrame",
            color=(0.015, 0.025, 0.028),
            roughness=0.5,
            metallic=0.2,
        ),
        "rubber": _make_material(
            stage,
            "/World/Materials/Rubber",
            color=(0.006, 0.008, 0.008),
            roughness=0.78,
            metallic=0.02,
        ),
        "stainless": _make_material(
            stage,
            "/World/Materials/StainlessHardware",
            color=(0.52, 0.60, 0.58),
            roughness=0.32,
            metallic=0.65,
        ),
        "payload_sensor": _make_material(
            stage,
            "/World/Materials/PayloadSensor",
            color=(0.04, 0.12, 0.13),
            roughness=0.22,
            metallic=0.08,
            emissive=(0.0, 0.020, 0.024),
        ),
        "nameplate": _make_material(
            stage,
            "/World/Materials/Nameplate",
            color=(0.08, 0.42, 0.44),
            roughness=0.24,
            metallic=0.20,
            emissive=(0.008, 0.080, 0.080),
        ),
        "thruster": _make_material(
            stage,
            "/World/Materials/ThrusterBlack",
            color=(0.025, 0.030, 0.032),
            roughness=0.42,
            metallic=0.15,
        ),
        "sensor_glass": _make_material(
            stage,
            "/World/Materials/SensorGlass",
            color=(0.05, 0.46, 0.48),
            roughness=0.16,
            metallic=0.05,
            emissive=(0.0, 0.06, 0.06),
        ),
        "camera_glass": _make_material(
            stage,
            "/World/Materials/CameraGlass",
            color=(0.02, 0.20, 0.22),
            roughness=0.08,
            metallic=0.02,
            emissive=(0.0, 0.025, 0.025),
        ),
        "acrylic_plate": _make_material(
            stage,
            "/World/Materials/AcrylicPlate",
            color=(0.22, 0.72, 0.76),
            roughness=0.18,
            opacity=0.34,
            emissive=(0.02, 0.10, 0.10),
        ),
        "tether_cable": _make_material(
            stage,
            "/World/Materials/TetherCable",
            color=(0.01, 0.012, 0.012),
            roughness=0.62,
        ),
        "front_led": _make_material(
            stage,
            "/World/Materials/ForwardLed",
            color=(0.56, 0.84, 0.78),
            roughness=0.12,
            emissive=(0.08, 0.18, 0.14),
        ),
        "front_beam": _make_material(
            stage,
            "/World/Materials/ForwardBeam",
            color=(0.16, 0.54, 0.50),
            roughness=0.2,
            opacity=0.030,
            emissive=(0.006, 0.030, 0.026),
        ),
        "lit_particulate": _make_material(
            stage,
            "/World/Materials/LitParticulate",
            color=(0.78, 0.98, 0.90),
            roughness=0.42,
            opacity=0.20,
            emissive=(0.004, 0.015, 0.010),
        ),
        "bubble": _make_material(
            stage,
            "/World/Materials/Bubble",
            color=(0.68, 0.92, 0.95),
            roughness=0.08,
            opacity=0.38,
            emissive=(0.020, 0.070, 0.075),
        ),
        "wake": _make_material(
            stage,
            "/World/Materials/ThrusterWake",
            color=(0.32, 0.82, 0.78),
            roughness=0.22,
            opacity=0.24,
            emissive=(0.055, 0.24, 0.22),
        ),
        "wake_mist": _make_material(
            stage,
            "/World/Materials/SubtleThrusterWakeMist",
            color=(0.18, 0.58, 0.55),
            roughness=0.32,
            opacity=0.032,
            emissive=(0.002, 0.012, 0.011),
        ),
        "propeller": _make_material(
            stage,
            "/World/Materials/Propeller",
            color=(0.06, 0.07, 0.07),
            roughness=0.28,
            metallic=0.25,
        ),
        "robot_salt_scuff": _make_material(
            stage,
            "/World/Materials/RobotSaltScuff",
            color=(0.72, 0.78, 0.70),
            roughness=0.94,
            opacity=0.42,
            emissive=(0.004, 0.006, 0.004),
        ),
        "robot_biofilm": _make_material(
            stage,
            "/World/Materials/RobotBiofilm",
            color=(0.08, 0.17, 0.11),
            roughness=0.98,
            opacity=0.46,
            emissive=(0.001, 0.004, 0.002),
        ),
        "robot_silt_streak": _make_material(
            stage,
            "/World/Materials/RobotSiltStreak",
            color=(0.11, 0.16, 0.13),
            roughness=0.96,
            opacity=0.58,
            emissive=(0.001, 0.003, 0.002),
        ),
        "telemetry_panel": _make_material(
            stage,
            "/World/Materials/TelemetryPanel",
            color=(0.02, 0.10, 0.11),
            roughness=0.35,
            opacity=0.34,
            emissive=(0.004, 0.025, 0.025),
        ),
        "telemetry_good": _make_material(
            stage,
            "/World/Materials/TelemetryGood",
            color=(0.16, 0.86, 0.58),
            roughness=0.24,
            emissive=(0.08, 0.50, 0.28),
        ),
    }

    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr(9.81)

    _seafloor_mesh(
        stage,
        "/World/Seafloor",
        material=materials["seafloor"],
    )
    _vertical_water_plane(
        stage,
        "/World/WaterVolume",
        y=4.85,
        z=-9.35,
        width=72.0,
        height=18.0,
        material=materials["water"],
    )
    for index, x in enumerate((-3.0, -1.0, 1.0, 3.0)):
        _cube(
            stage,
            f"/World/Caustics/Trace_{index}",
            translate=(x, -0.8 + 0.24 * index, -11.92),
            scale=(1.15, 0.018, 0.012),
            material=materials["caustic"],
        )
    _add_seafloor_detail(stage, materials=materials)
    _add_water_column_effects(stage, materials=materials)
    _add_depth_graded_water_volume(stage, materials=materials)
    _add_caustic_lattice(stage, materials=materials)

    _wave_surface(stage, "/World/WaveSurface", frames=frames, material=materials["surface"])
    _mission_path(
        stage,
        "/World/MissionPath",
        material=materials["mission_trace"],
        mission_positions=mission_positions,
    )
    _add_subsea_inspection_target(stage, materials=materials)
    _add_robot_reference(
        stage,
        robot_usd,
        frames=frames,
        mission_positions=mission_positions,
        visible=external_robot_usd is None,
    )
    if external_robot_usd is not None:
        _add_external_robot_visual(
            stage,
            external_robot_usd,
            frames=frames,
            mission_positions=mission_positions,
            scale=external_robot_scale,
        )
    else:
        _add_robot_visual_shell(
            stage,
            materials=materials,
            frames=frames,
            mission_positions=mission_positions,
        )
    _add_robot_weathering_detail(
        stage,
        materials=materials,
        frames=frames,
        mission_positions=mission_positions,
    )
    _add_robot_thruster_wake_detail(
        stage,
        materials=materials,
        frames=frames,
        mission_positions=mission_positions,
    )
    _add_robot_lighting_rig(
        stage,
        materials=materials,
        frames=frames,
        mission_positions=mission_positions,
    )
    _add_telemetry_panel(stage, materials=materials, mission_summary=mission_summary)
    _add_lights(stage)
    _add_camera(stage)

    task = UsdGeom.Xform.Define(stage, "/World/IsaacLabTask")
    task.GetPrim().CreateAttribute("oceanscale:directRlEnv", Sdf.ValueTypeNames.String).Set(
        "oceanscale.training.isaaclab_env.OceanScaleDirectRLEnv"
    )
    isaaclab_task_id = None
    if mission_summary is not None:
        stack_summary = mission_summary.get("stack", {})
        if isinstance(stack_summary, dict):
            isaaclab_task_id = stack_summary.get("isaac_lab_task_id")
        if isaaclab_task_id is None:
            probe_summary = mission_summary.get("isaac_lab_probe", {})
            if isinstance(probe_summary, dict):
                isaaclab_task_id = probe_summary.get("task_id")
    if isaaclab_task_id is not None:
        task.GetPrim().CreateAttribute("oceanscale:isaacLabTaskId", Sdf.ValueTypeNames.String).Set(
            str(isaaclab_task_id)
        )
    task.GetPrim().CreateAttribute("oceanscale:task", Sdf.ValueTypeNames.String).Set(
        "dock_standoff_approach"
    )
    if mission_summary is not None:
        evidence = _mission_evidence_summary(mission_summary)
        if evidence is not None:
            task.GetPrim().CreateAttribute("oceanscale:mvpEvidence", Sdf.ValueTypeNames.String).Set(
                json.dumps(evidence, sort_keys=True)
            )
    if isaaclab_runtime_smoke is not None:
        task.GetPrim().CreateAttribute("oceanscale:isaacLabRuntimeSmoke", Sdf.ValueTypeNames.String).Set(
            json.dumps(isaaclab_runtime_smoke, sort_keys=True)
        )
    task.GetPrim().CreateAttribute("oceanscale:robotAssetEvidence", Sdf.ValueTypeNames.String).Set(
        json.dumps(robot_asset_evidence, sort_keys=True)
    )
    if external_robot_asset_evidence is not None:
        task.GetPrim().CreateAttribute(
            "oceanscale:externalRobotAssetEvidence", Sdf.ValueTypeNames.String
        ).Set(json.dumps(external_robot_asset_evidence, sort_keys=True))

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_usd))
    summary = {
        "stage": str(output_usd),
        "robot_asset": str(robot_usd),
        "external_robot_asset": str(external_robot_usd) if external_robot_usd is not None else None,
        "frames": frames,
        "renderer": "Isaac Sim RaytracedLighting",
        "robot": "BlueROV2 Heavy",
        "thrusters": 8,
        "robot_asset_evidence": robot_asset_evidence,
        "isaac_lab_env": "OceanScaleDirectRLEnv",
        "mission_trajectory_samples": len(mission_positions),
        "material_assets": {name: str(path) for name, path in texture_assets.items()},
        "visual_effects": [
            "open_ocean_backdrop",
            "bathymetric_seafloor",
            "procedural_pbr_material_textures",
            "irregular_rock_field",
            "layered_sediment_patches",
            "layered_water_backscatter",
            "depth_graded_water_volume",
            "top_light_shafts",
            "marine_snow",
            "soft_seafloor_caustics",
            "multi_scale_caustic_lattice",
            "subtle_mission_trace",
            "subsea_pipeline_inspection_target",
            "biofouled_pipeline_detail",
            "rov_headlight_beams",
            "low_angle_inspection_camera",
            "thruster_bubbles",
            "thruster_wake",
            "robot_relative_thruster_wake_plumes",
            "robot_weathering_detail",
        ],
        "robot_visual_features": [
            "sampled_vehicle_attitude",
            "electronics_tube_endcaps",
            "acrylic_plate_bolts",
            "frame_standoffs",
            "dvl_pod",
            "forward_sonar",
            "thruster_rotor_blades",
            "tether_strain_relief",
            "subtle_robot_scuffs",
            "localized_biofilm_patches",
            "silt_runoff_streaks",
            "robot_relative_bubble_trails",
        ],
    }
    if external_robot_usd is not None:
        summary["robot_visual_features"].append("external_high_fidelity_mesh")
    if external_robot_asset_evidence is not None:
        summary["external_robot_asset_evidence"] = external_robot_asset_evidence
    evidence = _mission_evidence_summary(mission_summary)
    if evidence is not None:
        summary["mission_evidence"] = evidence
        if "isaac_lab_probe" in evidence:
            summary["isaac_lab_probe"] = evidence["isaac_lab_probe"]
            summary["isaac_lab_task_id"] = evidence["isaac_lab_probe"].get("task_id")
    if isaaclab_runtime_smoke is not None:
        summary["isaac_lab_runtime_smoke"] = isaaclab_runtime_smoke
    return summary


def main() -> None:
    args = parse_args()
    robot_usd = args.robot_usd.resolve()
    output_usd = args.output_usd.resolve()
    if not robot_usd.exists():
        raise FileNotFoundError(f"Robot USD does not exist: {robot_usd}")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be positive")
    if args.pathtracing_spp < 1:
        raise ValueError("--pathtracing-spp must be positive")
    if args.external_robot_scale <= 0.0:
        raise ValueError("--external-robot-scale must be positive")
    if args.capture_warmup_frames < 0:
        raise ValueError("--capture-warmup-frames must be non-negative")
    if args.sequence_frame_count < 0:
        raise ValueError("--sequence-frame-count must be non-negative")
    if args.sequence_fps < 1:
        raise ValueError("--sequence-fps must be at least 1")
    if args.capture_mp4 is not None and args.sequence_frame_count == 0:
        raise ValueError("--capture-mp4 requires --sequence-frame-count")
    if args.mission_json is not None and not args.mission_json.exists():
        raise FileNotFoundError(f"Mission JSON does not exist: {args.mission_json}")
    if (
        args.isaaclab_runtime_smoke_json is not None
        and not args.isaaclab_runtime_smoke_json.exists()
    ):
        raise FileNotFoundError(
            f"Isaac Lab runtime smoke JSON does not exist: {args.isaaclab_runtime_smoke_json}"
        )
    if args.external_robot_mesh is not None and not args.external_robot_mesh.exists():
        raise FileNotFoundError(f"External robot mesh does not exist: {args.external_robot_mesh}")

    external_robot_usd, external_robot_summary = _resolve_external_robot_usd(
        args.external_robot_mesh,
    )

    app = _start_simulation_app(
        args.gui,
        width=args.width,
        height=args.height,
        renderer=args.renderer,
    )
    try:
        render_quality = _apply_render_quality(
            renderer=args.renderer,
            quality_preset=args.quality_preset,
            pathtracing_spp=args.pathtracing_spp,
        )
        mission_summary = _load_mission_summary(args.mission_json)
        isaaclab_runtime_smoke = _load_isaaclab_runtime_smoke(args.isaaclab_runtime_smoke_json)
        summary = _build_stage(
            output_usd,
            robot_usd,
            args.frames,
            mission_summary=mission_summary,
            external_robot_usd=external_robot_usd,
            external_robot_scale=args.external_robot_scale,
            isaaclab_runtime_smoke=isaaclab_runtime_smoke,
        )
        summary["render_quality"] = render_quality
        summary["renderer"] = f"Isaac Sim {args.renderer}"
        summary["requested_resolution"] = [args.width, args.height]
        if external_robot_summary is not None:
            if "external_robot_asset_evidence" in summary:
                external_robot_summary["asset_evidence"] = summary[
                    "external_robot_asset_evidence"
                ]
                external_robot_summary["reference_prim"] = summary[
                    "external_robot_asset_evidence"
                ].get("prim")
            summary["external_robot_mesh"] = external_robot_summary
            summary["external_robot_scale"] = args.external_robot_scale
        if args.mission_json is not None:
            summary["mission_json"] = str(args.mission_json.resolve())
        if args.isaaclab_runtime_smoke_json is not None:
            summary["isaaclab_runtime_smoke_json"] = str(
                args.isaaclab_runtime_smoke_json.resolve()
            )
        for _ in range(max(0, args.run_frames)):
            app.update()
        if args.capture_png is not None:
            summary["capture"] = _capture_viewport(
                app,
                args.capture_png.resolve(),
                warmup_frames=args.capture_warmup_frames,
            )
        if args.sequence_frame_count:
            sequence_dir = (
                args.capture_sequence_dir.resolve()
                if args.capture_sequence_dir is not None
                else DEFAULT_SEQUENCE_DIR.resolve()
            )
            summary["sequence"] = _capture_sequence(
                app,
                sequence_dir,
                authored_frames=args.frames,
                frame_count=args.sequence_frame_count,
                fps=args.sequence_fps,
                warmup_frames=args.capture_warmup_frames,
            )
            if args.capture_mp4 is not None:
                summary["video"] = _encode_mp4(
                    sequence_dir,
                    args.capture_mp4.resolve(),
                    fps=args.sequence_fps,
                )
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
