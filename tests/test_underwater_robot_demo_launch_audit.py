"""Tests for the underwater robot launch-candidate audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_underwater_robot_demo_launch.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_underwater_robot_demo_launch",
        AUDIT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launch_audit_preserves_full_goal_scope() -> None:
    module = _load_audit_module()

    result = module.audit_launch_candidate(run_ffprobe=False)
    requirements = {item["requirement"]: item for item in result["requirements"]}

    assert result["objective"] == "like real ue/unity quality ocean scene with real robot demo"
    assert result["artifact_verifier_ok"] is True
    assert result["goal_complete"] is False
    assert requirements["real_robot_asset"]["status"] == "passed"
    assert requirements["ocean_visual_quality_floor"]["status"] == "passed"
    assert requirements["robot_mission_demo"]["status"] == "passed"
    assert requirements["ue_unity_quality_claim"]["status"] == "partial"
    assert requirements["local_reproducibility_gate"]["status"] == "partial"
    assert requirements["hosted_public_demo"]["status"] == "missing"


def test_launch_audit_can_include_release_gate_without_browser() -> None:
    module = _load_audit_module()

    result = module.audit_launch_candidate(
        run_release_gate=True,
        run_browser=False,
        run_ffprobe=False,
    )
    requirements = {item["requirement"]: item for item in result["requirements"]}

    assert requirements["local_reproducibility_gate"]["status"] == "passed"
    assert requirements["local_reproducibility_gate"]["evidence"]["browser_smoke"] == "skipped"
    assert result["goal_complete"] is False
