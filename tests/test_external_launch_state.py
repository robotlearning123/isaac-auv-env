from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_external_launch_state.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("refresh_external_launch_state", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_state_defers_public_release_and_pypi_when_disabled() -> None:
    script = _load_script()

    status = script.evaluate_state(
        release_deploy={
            "runs": [
                {"status": "completed", "conclusion": "success"},
                {"status": "completed", "conclusion": "success"},
            ]
        },
        pypi_availability={"available": False},
        workflow_runner_policy={"status": "green"},
        release_policy={"status": "green"},
        public_pypi_policy={"status": "green"},
    )

    assert status["release_deploy"] == "deferred_manual_only"
    assert status["public_pypi_publish"] == "deferred_not_public"
    assert status["pypi_availability"] == "deferred_not_public"
    assert status["workflow_runner_policy"] == "green"
    assert status["overall_public_launch_pipeline"] == "deferred_not_public"


def test_scan_workflow_runners_finds_forbidden_labels(tmp_path: Path) -> None:
    script = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "compat.yml").write_text(
        "jobs:\n"
        "  test:\n"
        "    runs-on: [self-hosted, gpu, rtx5090]\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
    )

    result = script.scan_workflow_runners(tmp_path)

    assert result["status"] == "red"
    assert len(result["violations"]) == 2
    assert {violation["path"] for violation in result["violations"]} == {
        ".github/workflows/compat.yml"
    }


def test_scan_workflow_runners_requires_oceanscale_arc(tmp_path: Path) -> None:
    script = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "compat.yml").write_text(
        "jobs:\n"
        "  test:\n"
        "    runs-on: oceanscale-arc\n"
    )

    result = script.scan_workflow_runners(tmp_path)

    assert result["status"] == "green"
    assert len(result["violations"]) == 0


def test_public_pypi_policy_detects_no_publish_workflow(tmp_path: Path) -> None:
    script = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "publish-pypi.yml").write_text(
        "name: Package Build Check (No Publish)\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "permissions:\n"
        "  contents: read\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: oceanscale-arc\n"
    )

    result = script.scan_public_pypi_policy(tmp_path)

    assert result["status"] == "green"
    assert result["public_publish_enabled"] is False


def test_release_policy_detects_tag_trigger(tmp_path: Path) -> None:
    script = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "release.yml").write_text(
        "on:\n"
        "  push:\n"
        "    tags:\n"
        "      - 'v*.*.*'\n"
    )

    result = script.scan_release_policy(tmp_path)

    assert result["status"] == "red"
    assert result["automatic_tag_deploy"] is True
