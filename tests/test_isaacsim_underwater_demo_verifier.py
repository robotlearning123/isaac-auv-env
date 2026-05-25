"""Tests for the fast Isaac Sim underwater demo artifact verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "verify_isaacsim_underwater_demo.py"
)


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location(
        "verify_isaacsim_underwater_demo",
        VERIFY_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_accepts_canonical_demo_artifacts_without_isaac() -> None:
    module = _load_verifier_module()

    result = module.verify_demo_artifacts(run_ffprobe=False)
    check_names = {check["name"] for check in result["checks"]}

    assert result["ok"] is True
    assert "renderer_is_path_tracing" in check_names
    assert "path_tracing_spp" in check_names
    assert "required_visual_effects" in check_names
    assert "required_robot_features" in check_names
    assert "isaac_lab_runtime_ok" in check_names
    assert "external_robot_triangles" in check_names
    assert "forbidden_stage_tokens" in check_names
    assert all(check["ok"] for check in result["checks"])


def test_verifier_parses_video_rate() -> None:
    module = _load_verifier_module()

    assert module._fps_from_rate("12/1") == 12.0
    assert module._fps_from_rate("30000/1001") == 30000 / 1001
    assert module._fps_from_rate("0/0") is None
