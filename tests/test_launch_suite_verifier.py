from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_internal_launch_suite.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_internal_launch_suite", VERIFIER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_internal_launch_suite_verifier_passes() -> None:
    verifier = _load_verifier()
    checks = verifier.verify(ROOT)

    failed = [check for check in checks if not check.ok]

    assert not failed, [(check.name, check.detail) for check in failed]


def test_internal_launch_suite_verifier_covers_goal_scope() -> None:
    verifier = _load_verifier()
    check_names = {check.name for check in verifier.verify(ROOT)}

    assert {
        "benchmark_truth",
        "completion_audit_scope",
        "pipeline_policy_recorded",
        "public_launch_no_go",
        "source_of_truth_paths",
    }.issubset(check_names)
