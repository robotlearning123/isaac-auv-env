from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json")
WORKFLOW_FIELDS = "databaseId,status,conclusion,displayTitle,workflowName,event,createdAt,url"
FORBIDDEN_RUNNER_FRAGMENTS = ("self-hosted", "ubuntu-latest")
EXPECTED_ARC_RUNNER = "oceanscale-arc"


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def run_command(command: list[str], cwd: Path = ROOT) -> CommandResult:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def collect_workflow_runs(workflow: str, limit: int = 5, root: Path = ROOT) -> dict[str, Any]:
    result = run_command(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--limit",
            str(limit),
            "--json",
            WORKFLOW_FIELDS,
        ],
        cwd=root,
    )
    runs: list[dict[str, Any]] = []
    parse_error = ""
    if result.exit_code == 0:
        try:
            parsed = json.loads(result.stdout)
            if isinstance(parsed, list):
                runs = [row for row in parsed if isinstance(row, dict)]
        except json.JSONDecodeError as exc:
            parse_error = str(exc)

    return {
        "workflow": workflow,
        "command": result.command,
        "exit_code": result.exit_code,
        "runs": runs,
        "stderr": result.stderr,
        "parse_error": parse_error,
    }


def query_pypi(package: str = "oceanscale", root: Path = ROOT) -> dict[str, Any]:
    result = run_command(["uvx", "--from", "pip", "pip", "index", "versions", package], cwd=root)
    return {
        "package": package,
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "available": result.exit_code == 0,
    }


def scan_workflow_runners(root: Path = ROOT) -> dict[str, Any]:
    workflow_dir = root / ".github" / "workflows"
    matches: list[dict[str, Any]] = []
    for path in sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "runs-on:" in line or any(fragment in line for fragment in FORBIDDEN_RUNNER_FRAGMENTS):
                relative = path.relative_to(root).as_posix()
                forbidden = [
                    fragment for fragment in FORBIDDEN_RUNNER_FRAGMENTS if fragment in line
                ]
                wrong_arc = "runs-on:" in line and EXPECTED_ARC_RUNNER not in line
                matches.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "text": line.strip(),
                        "forbidden_fragments": forbidden,
                        "expected_arc": EXPECTED_ARC_RUNNER,
                        "wrong_arc": wrong_arc,
                    }
                )

    violations = [match for match in matches if match["forbidden_fragments"] or match["wrong_arc"]]
    return {
        "status": "red" if violations else "green",
        "matches": matches,
        "violations": violations,
    }


def scan_release_policy(root: Path = ROOT) -> dict[str, Any]:
    text = (root / ".github" / "workflows" / "release.yml").read_text()
    automatic_tag_deploy = "tags:" in text or "v*.*.*" in text
    return {
        "status": "red" if automatic_tag_deploy else "green",
        "automatic_tag_deploy": automatic_tag_deploy,
        "manual_dispatch": "workflow_dispatch:" in text,
    }


def scan_public_pypi_policy(root: Path = ROOT) -> dict[str, Any]:
    text = (root / ".github" / "workflows" / "publish-pypi.yml").read_text()
    public_publish_action = "pypa/gh-action-pypi-publish" in text
    public_pypi_environment = "environment: pypi" in text
    oidc_publish_permission = "id-token: write" in text
    tag_trigger = "tags:" in text or "v*.*.*" in text
    public_publish_enabled = (
        public_publish_action or public_pypi_environment or oidc_publish_permission or tag_trigger
    )
    return {
        "status": "red" if public_publish_enabled else "green",
        "public_publish_action": public_publish_action,
        "public_pypi_environment": public_pypi_environment,
        "oidc_publish_permission": oidc_publish_permission,
        "tag_trigger": tag_trigger,
        "public_publish_enabled": public_publish_enabled,
    }


def _all_completed_with(runs: list[dict[str, Any]], conclusion: str) -> bool:
    return bool(runs) and all(
        run.get("status") == "completed" and run.get("conclusion") == conclusion for run in runs
    )


def evaluate_state(
    release_deploy: dict[str, Any],
    pypi_availability: dict[str, Any],
    workflow_runner_policy: dict[str, Any],
    release_policy: dict[str, Any],
    public_pypi_policy: dict[str, Any],
) -> dict[str, Any]:
    release_runs = release_deploy.get("runs", [])
    release_all_success = isinstance(release_runs, list) and _all_completed_with(
        release_runs, "success"
    )
    pypi_available = pypi_availability.get("available") is True
    runner_policy_green = workflow_runner_policy.get("status") == "green"
    release_policy_green = release_policy.get("status") == "green"
    public_pypi_policy_green = public_pypi_policy.get("status") == "green"
    public_launch_deferred = release_policy_green and public_pypi_policy_green
    overall_green = (
        release_all_success
        and pypi_available
        and runner_policy_green
        and not public_launch_deferred
    )

    return {
        "release_deploy": "deferred_manual_only"
        if release_policy_green
        else ("green" if release_all_success else "yellow_or_red"),
        "public_pypi_publish": "deferred_not_public"
        if public_pypi_policy_green
        else "red",
        "pypi_availability": "deferred_not_public"
        if public_pypi_policy_green
        else ("green" if pypi_available else "red"),
        "workflow_runner_policy": workflow_runner_policy.get("status", "unknown"),
        "release_policy": release_policy.get("status", "unknown"),
        "public_pypi_policy": public_pypi_policy.get("status", "unknown"),
        "overall_public_launch_pipeline": "green"
        if overall_green
        else ("deferred_not_public" if public_launch_deferred and runner_policy_green else "red"),
    }


def build_state(root: Path = ROOT) -> dict[str, Any]:
    release_deploy = collect_workflow_runs("Release Deploy", root=root)
    historical_publish_pypi = collect_workflow_runs("Publish to PyPI", root=root)
    pypi_availability = query_pypi(root=root)
    workflow_runner_policy = scan_workflow_runners(root=root)
    release_policy = scan_release_policy(root=root)
    public_pypi_policy = scan_public_pypi_policy(root=root)
    status = evaluate_state(
        release_deploy=release_deploy,
        pypi_availability=pypi_availability,
        workflow_runner_policy=workflow_runner_policy,
        release_policy=release_policy,
        public_pypi_policy=public_pypi_policy,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "internal_alpha_readiness",
        "public_launch": False,
        "repository": str(root),
        "checks": {
            "historical_publish_pypi": historical_publish_pypi,
            "release_deploy": release_deploy,
            "pypi_availability": pypi_availability,
            "workflow_runner_policy": workflow_runner_policy,
            "release_policy": release_policy,
            "public_pypi_policy": public_pypi_policy,
        },
        "status": status,
        "approvals_required_to_fix": [],
    }


def write_state(state: dict[str, Any], output: Path, root: Path = ROOT) -> Path:
    target = output if output.is_absolute() else root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh read-only external launch state.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    state = build_state(ROOT)
    target = write_state(state, args.output, ROOT)
    print(f"Wrote {target.relative_to(ROOT)}")
    print(json.dumps(state["status"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
