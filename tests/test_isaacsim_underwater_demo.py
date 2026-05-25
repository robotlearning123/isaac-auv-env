"""Static checks for the optional Isaac Sim demo launcher."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

from oceanscale.training.isaaclab_env import OCEANSCALE_UNDERWATER_TASK_ID

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "isaacsim_underwater_demo.py"
REPO_ROOT = SCRIPT_PATH.parents[1]
ISAAC_ARTIFACTS = REPO_ROOT / "artifacts" / "isaacsim"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("isaacsim_underwater_demo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isaacsim_demo_keeps_isaac_imports_runtime_only() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    heavy_modules = {"isaacsim", "omni", "pxr", "isaaclab"}
    top_level_imports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_imports.append(node.module.split(".", maxsplit=1)[0])

    assert heavy_modules.isdisjoint(top_level_imports)


def test_isaacsim_demo_references_packaged_bluerov_asset() -> None:
    module = _load_script_module()

    assert module.DEFAULT_ROBOT_ASSET.name == "bluerov2_heavy.usda"
    assert module.DEFAULT_ROBOT_ASSET.exists()
    assert module.DEFAULT_WIDTH == 1920
    assert module.DEFAULT_HEIGHT == 1080
    assert module.DEFAULT_RENDERER == "RaytracedLighting"
    assert module.DEFAULT_OUTPUT.parts[-3:] == ("artifacts", "isaacsim", "underwater_robot_demo.usda")
    assert module.DEFAULT_CAPTURE.parts[-3:] == (
        "artifacts",
        "isaacsim",
        "underwater_robot_demo.png",
    )
    assert module.DEFAULT_SEQUENCE_DIR.parts[-3:] == (
        "artifacts",
        "isaacsim",
        "underwater_robot_demo_frames",
    )
    assert module.DEFAULT_VIDEO.parts[-3:] == (
        "artifacts",
        "isaacsim",
        "underwater_robot_demo.mp4",
    )


def test_isaacsim_demo_tracks_cinematic_water_effects() -> None:
    module = _load_script_module()
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert hasattr(module, "_add_water_column_effects")
    for marker in (
        "layered_water_backscatter",
        "top_light_shafts",
        "marine_snow",
        "soft_seafloor_caustics",
        "procedural_pbr_material_textures",
        "irregular_rock_field",
        "layered_sediment_patches",
        "subsea_pipeline_inspection_target",
        "depth_graded_water_volume",
        "biofouled_pipeline_detail",
        "multi_scale_caustic_lattice",
        "rov_headlight_beams",
        "low_angle_inspection_camera",
        "thruster_bubbles",
        "thruster_wake",
        "robot_relative_thruster_wake_plumes",
        "robot_weathering_detail",
    ):
        assert marker in source

    for marker in (
        "_make_textured_material",
        "_ensure_demo_texture_assets",
        "_add_depth_graded_water_volume",
        "_add_caustic_lattice",
        "UsdUVTexture",
        "UsdPrimvarReader_float2",
        "TexCoord2fArray",
    ):
        assert marker in source


def test_isaacsim_demo_tracks_robot_visual_fidelity_markers() -> None:
    module = _load_script_module()
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert hasattr(module, "_mission_rotate_op")
    for marker in (
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
        "_add_robot_weathering_detail",
        "_add_robot_thruster_wake_detail",
    ):
        assert marker in source


def test_isaacsim_demo_supports_external_robot_mesh_input() -> None:
    module = _load_script_module()
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert module.USD_ASSET_SUFFIXES == {".usd", ".usda", ".usdc"}
    for marker in (
        "--external-robot-mesh",
        "--isaaclab-runtime-smoke-json",
        "pre-converted OpenUSD asset",
        "external_high_fidelity_mesh",
        "_resolve_external_robot_usd",
        "_load_isaaclab_runtime_smoke",
        "_external_robot_reference_prim_path",
        "sourceAssetPrim",
    ):
        assert marker in source


def test_isaacsim_demo_summarizes_real_isaaclab_runtime_smoke() -> None:
    module = _load_script_module()
    smoke_path = ISAAC_ARTIFACTS / "oceanscale_isaaclab_task_smoke.json"

    smoke = module._load_isaaclab_runtime_smoke(smoke_path)

    assert smoke["ok"] is True
    assert smoke["api"] == "Isaac Lab DirectRLEnv"
    assert smoke["task"] == OCEANSCALE_UNDERWATER_TASK_ID
    assert smoke["isaaclab_version"] == "0.47.7"
    assert smoke["isaaclab_tasks_version"] == "0.11.7"
    assert smoke["num_envs"] == 1
    assert smoke["device"] == "cuda:0"
    assert smoke["oceanscale_task_registered"] is True
    assert smoke["action_shape"] == [1, 6]
    assert smoke["reset_obs_finite"] is True
    assert smoke["step_obs_finite"] is True
    assert smoke["reward_finite"] is True
    assert {package["name"] for package in smoke["preloaded_packages"]} == {"warp", "newton"}


def test_isaacsim_demo_reads_packaged_robot_asset_evidence() -> None:
    module = _load_script_module()

    evidence = module._robot_asset_evidence(module.DEFAULT_ROBOT_ASSET)

    assert evidence["asset_role"] == "mvp_visual_proxy"
    assert evidence["vehicle"] == "BlueROV2 Heavy"
    assert evidence["thruster_count"] == 8
    assert evidence["cad_derived"] is False
    assert evidence["geometry_reference"] == (
        "scratch OpenUSD from public dimensions and von Benzon Table A1"
    )
    assert evidence["dynamics_reference"] == "von Benzon et al. 2022 Table A1"
    assert "clydemcqueen/bluerov2_gz" in evidence["upgrade_candidate"]


def test_isaacsim_demo_reads_external_robot_asset_evidence() -> None:
    module = _load_script_module()
    external_asset = ISAAC_ARTIFACTS / "external_assets" / "bluerov2_gz_heavy" / (
        "bluerov2_gz_heavy.usdc"
    )

    evidence = module._external_usd_asset_evidence(external_asset)

    assert evidence["asset_role"] == "external_high_fidelity_mesh"
    assert evidence["vehicle"] == "BlueROV2 Heavy"
    assert evidence["source_repo"] == "https://github.com/clydemcqueen/bluerov2_gz"
    assert evidence["source_commit"] == "661264b719ffd2dcdd0d0990de80547d6029cc16"
    assert evidence["source_license"] == "MIT"
    assert evidence["source_asset_type"] == "SDF + Collada visual mesh"
    assert evidence["thruster_count"] == 8
    assert evidence["visual_mesh_count"] == 9
    assert evidence["unique_collada_mesh_count"] == 3
    assert evidence["triangle_instance_count"] > 100_000


def test_isaacsim_demo_extracts_underwater_mvp_trajectory() -> None:
    module = _load_script_module()
    summary = {
        "metrics": {
            "final_distance_to_target_m": 0.12,
            "min_distance_to_target_m": 0.12,
            "sonar_detection_rate": 1.0,
            "steps_per_sec": 120.0,
        },
        "mission": {"completed": True},
        "trajectory": {
            "sample_count": 2,
            "samples": [
                {"position": [0.0, 0.0, -10.0]},
                {"position": [1.4, 0.0, -9.98]},
            ],
        },
        "isaac_lab_probe": {
            "api": "OceanScaleDirectRLEnv",
            "task_id": OCEANSCALE_UNDERWATER_TASK_ID,
            "controller": "mvp_standoff_pd",
            "gym_registered": True,
            "steps": 3,
            "observation_dim": 33,
            "action_dim": 6,
            "obs_finite": True,
            "reward_finite": True,
        },
    }

    assert module._trajectory_positions_from_summary(summary) == [
        (0.0, 0.0, -10.0),
        (1.4, 0.0, -9.98),
    ]
    assert module._mission_evidence_summary(summary) == {
        "completed": True,
        "final_distance_to_target_m": 0.12,
        "min_distance_to_target_m": 0.12,
        "sonar_detection_rate": 1.0,
        "steps_per_sec": 120.0,
        "trajectory_sample_count": 2,
        "isaac_lab_probe": {
            "api": "OceanScaleDirectRLEnv",
            "task_id": OCEANSCALE_UNDERWATER_TASK_ID,
            "controller": "mvp_standoff_pd",
            "gym_registered": True,
            "steps": 3,
            "observation_dim": 33,
            "action_dim": 6,
            "obs_finite": True,
            "reward_finite": True,
        },
    }


def test_canonical_isaacsim_artifact_carries_mvp_evidence() -> None:
    summary_path = ISAAC_ARTIFACTS / "underwater_robot_demo_pathtracing64_summary.json"
    stage_path = ISAAC_ARTIFACTS / "underwater_robot_demo_pathtracing64.usda"
    capture_path = ISAAC_ARTIFACTS / "underwater_robot_demo_pathtracing64.png"
    video_path = ISAAC_ARTIFACTS / "underwater_robot_demo_pathtracing64.mp4"
    frames_dir = ISAAC_ARTIFACTS / "underwater_robot_demo_pathtracing64_frames"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mission_json = Path(summary["mission_json"])
    evidence = summary["mission_evidence"]
    isaaclab_probe = summary["isaac_lab_probe"]
    isaaclab_runtime_smoke = summary["isaac_lab_runtime_smoke"]
    robot_asset_evidence = summary["robot_asset_evidence"]
    external_robot_mesh = summary["external_robot_mesh"]
    external_asset_evidence = external_robot_mesh["asset_evidence"]

    assert stage_path.exists()
    assert capture_path.exists()
    assert video_path.exists()
    assert mission_json.name == "underwater_robot_demo_pathtracing64_mission.json"
    assert mission_json.exists()
    assert summary["renderer"] == "Isaac Sim PathTracing"
    assert summary["render_quality"]["pathtracing_total_spp"] == 64
    assert summary["requested_resolution"] == [1920, 1080]
    assert summary["capture"]["resolution"] == [1920, 1080]
    assert summary["sequence"]["frame_count"] == 24
    assert len(list(frames_dir.glob("frame_*.png"))) == 24
    assert summary["external_robot_asset"].endswith("bluerov2_gz_heavy.usdc")
    assert "external_high_fidelity_mesh" in summary["robot_visual_features"]
    assert "depth_graded_water_volume" in summary["visual_effects"]
    assert "multi_scale_caustic_lattice" in summary["visual_effects"]
    assert "robot_relative_thruster_wake_plumes" in summary["visual_effects"]
    assert "robot_relative_bubble_trails" in summary["robot_visual_features"]
    assert Path(summary["isaaclab_runtime_smoke_json"]).name == (
        "oceanscale_isaaclab_task_smoke.json"
    )

    assert evidence["completed"] is True
    assert evidence["final_distance_to_target_m"] <= 0.15
    assert evidence["sonar_detection_rate"] == 1.0
    assert isaaclab_probe == evidence["isaac_lab_probe"]
    assert isaaclab_probe["api"] == "OceanScaleDirectRLEnv"
    assert isaaclab_probe["task_id"] == OCEANSCALE_UNDERWATER_TASK_ID
    assert isaaclab_probe["gym_registered"] is True
    assert isaaclab_probe["controller"] == "mvp_standoff_pd"
    assert isaaclab_probe["observation_dim"] == 33
    assert isaaclab_probe["action_dim"] == 6
    assert isaaclab_probe["obs_finite"] is True
    assert isaaclab_probe["reward_finite"] is True
    assert isaaclab_runtime_smoke["ok"] is True
    assert isaaclab_runtime_smoke["api"] == "Isaac Lab DirectRLEnv"
    assert isaaclab_runtime_smoke["task"] == OCEANSCALE_UNDERWATER_TASK_ID
    assert isaaclab_runtime_smoke["isaaclab_version"] == "0.47.7"
    assert isaaclab_runtime_smoke["isaaclab_tasks_version"] == "0.11.7"
    assert isaaclab_runtime_smoke["oceanscale_task_registered"] is True
    assert isaaclab_runtime_smoke["action_shape"] == [1, 6]
    assert isaaclab_runtime_smoke["reset_obs_finite"] is True
    assert isaaclab_runtime_smoke["step_obs_finite"] is True
    assert isaaclab_runtime_smoke["reward_finite"] is True
    assert {package["name"] for package in isaaclab_runtime_smoke["preloaded_packages"]} == {
        "warp",
        "newton",
    }
    assert robot_asset_evidence["asset_role"] == "mvp_visual_proxy"
    assert robot_asset_evidence["vehicle"] == "BlueROV2 Heavy"
    assert robot_asset_evidence["thruster_count"] == 8
    assert robot_asset_evidence["cad_derived"] is False
    assert "clydemcqueen/bluerov2_gz" in robot_asset_evidence["upgrade_candidate"]
    assert external_asset_evidence["asset_role"] == "external_high_fidelity_mesh"
    assert external_asset_evidence["vehicle"] == "BlueROV2 Heavy"
    assert external_asset_evidence["source_license"] == "MIT"
    assert external_asset_evidence["visual_mesh_count"] == 9
    assert external_asset_evidence["unique_collada_mesh_count"] == 3
    assert external_asset_evidence["triangle_instance_count"] > 100_000

    stage_source = stage_path.read_text(encoding="utf-8")
    assert "oceanscale:mvpEvidence" in stage_source
    assert "oceanscale:isaacLabRuntimeSmoke" in stage_source
    assert "oceanscale:isaacLabTaskId" in stage_source
    assert "oceanscale:robotAssetEvidence" in stage_source
    assert "ExternalRobotVisual" in stage_source
    assert "DepthGradedWaterVolume" in stage_source
    assert "CausticLattice" in stage_source
    assert "RobotThrusterWakeDetail" in stage_source
    assert "bluerov2_gz_heavy.usdc" in stage_source
    assert "mvp_visual_proxy" in stage_source
    assert OCEANSCALE_UNDERWATER_TASK_ID in stage_source
    assert "OceanScaleDirectRLEnv" in stage_source
