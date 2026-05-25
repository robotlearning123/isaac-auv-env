"""Tests for the underwater robot demo release gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_GATE_PATH = REPO_ROOT / "scripts" / "verify_underwater_robot_demo_release.py"


def _load_release_gate_module():
    spec = importlib.util.spec_from_file_location(
        "verify_underwater_robot_demo_release",
        RELEASE_GATE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_gate_static_checks_bind_verified_assets() -> None:
    module = _load_release_gate_module()

    result = module.verify_release_gate(
        run_pytest=False,
        run_build=False,
        run_browser=False,
        run_ffprobe=False,
    )
    checks = {check["name"]: check for check in result["checks"]}

    assert result["ok"] is True
    assert checks["isaac_artifact_verifier"]["ok"] is True
    assert checks["website_static_demo_surface"]["ok"] is True
    assert (
        checks["website_static_demo_surface"]["evidence"]["requirements"][
            "public_video_matches_canonical"
        ]
        is True
    )
    assert (
        checks["website_static_demo_surface"]["evidence"]["requirements"][
            "public_poster_matches_canonical"
        ]
        is True
    )


def test_release_gate_orchestrates_pytest_build_and_browser_smoke() -> None:
    module = _load_release_gate_module()
    commands: list[tuple[str, list[str], Path, int]] = []
    browser_calls: list[tuple[Path, int]] = []

    def fake_command_runner(
        name: str,
        args: list[str],
        cwd: Path,
        timeout: int,
    ) -> Any:
        commands.append((name, list(args), cwd, timeout))
        return module.ReleaseCheck(name, True, "fake pass", {"args": list(args), "cwd": str(cwd)})

    def fake_browser_runner(screenshots_dir: Path, timeout: int) -> Any:
        browser_calls.append((screenshots_dir, timeout))
        return module.ReleaseCheck(
            "website_browser_layout",
            True,
            "fake pass",
            {"screenshots_dir": str(screenshots_dir)},
        )

    result = module.verify_release_gate(
        run_ffprobe=False,
        timeout=17,
        command_runner=fake_command_runner,
        browser_runner=fake_browser_runner,
    )

    assert result["ok"] is True
    assert (
        "targeted_pytest",
        [sys.executable, "-m", "pytest", *module.PYTEST_TARGETS, "-q"],
        REPO_ROOT,
        17,
    ) in commands
    assert ("website_build", ["pnpm", "build"], module.WEBSITE_DIR, 17) in commands
    assert browser_calls == [(module.DEFAULT_SCREENSHOTS_DIR, 17)]
