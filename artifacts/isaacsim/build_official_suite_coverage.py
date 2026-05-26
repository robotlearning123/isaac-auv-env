#!/usr/bin/env python3
"""Build coverage from official suite inventory plus recorded smoke summaries."""

from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/mnt/storage/isaacsim-6.0-official")
DEFAULT_LOGS = DEFAULT_ROOT / "logs"
DEFAULT_INVENTORY = Path("/home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_inventory.json")
DEFAULT_JSON = Path("/home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_coverage.json")
DEFAULT_MARKDOWN = Path("/home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_coverage.md")


@dataclass(frozen=True)
class RunRecord:
    name: str
    suite: str | None
    relative_path: str | None
    passed: bool
    source_variant: str
    summary_path: str
    log_path: str | None
    command: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_isaaclab_script(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        if part == "-p" and index + 1 < len(parts):
            script = parts[index + 1]
            if script.startswith("/"):
                return None
            return script
    return None


def parse_command(command: str, isaacsim_root: Path, isaaclab_root: Path) -> tuple[str | None, str | None]:
    isaacsim_prefix = isaacsim_root.as_posix().rstrip("/") + "/"
    isaaclab_prefix = isaaclab_root.as_posix().rstrip("/") + "/"
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for token in parts:
        if token.startswith(isaacsim_prefix):
            return "isaacsim", token.removeprefix(isaacsim_prefix)
        if token.startswith(isaaclab_prefix):
            return "isaaclab", token.removeprefix(isaaclab_prefix)
    lab_script = parse_isaaclab_script(command)
    if lab_script is not None:
        return "isaaclab", lab_script
    return None, None


def source_variant(summary: dict[str, Any]) -> str:
    lab_root = str(summary.get("lab_root", ""))
    if "smoke-patches" in lab_root or "oceanscale-smoke-patches" in lab_root:
        return "patched"
    return "clean"


def collect_records(logs_root: Path, isaacsim_root: Path, isaaclab_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for summary_path in sorted(logs_root.rglob("summary.json")):
        try:
            summary = load_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        variant = source_variant(summary)
        for result in summary.get("results", []):
            command = str(result.get("command", ""))
            suite, relative_path = parse_command(command, isaacsim_root, isaaclab_root)
            records.append(
                RunRecord(
                    name=str(result.get("name", "")),
                    suite=suite,
                    relative_path=relative_path,
                    passed=bool(result.get("passed", False)),
                    source_variant=variant,
                    summary_path=str(summary_path),
                    log_path=result.get("log_path"),
                    command=command,
                )
            )
    return records


def best_status(records: list[RunRecord], variant: str) -> str:
    relevant = [record for record in records if record.source_variant == variant]
    if any(record.passed for record in relevant):
        return "passed"
    if relevant:
        return "failed"
    return "uncovered"


def build_coverage(inventory: dict[str, Any], records: list[RunRecord]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    runs_by_path: dict[str, list[RunRecord]] = defaultdict(list)
    unmatched: list[RunRecord] = []
    for record in records:
        if record.relative_path:
            runs_by_path[record.relative_path].append(record)
        else:
            unmatched.append(record)

    for entry in inventory["entrypoints"]:
        relpath = entry["relative_path"]
        path_records = runs_by_path.get(relpath, [])
        rows.append(
            {
                **entry,
                "clean_status": best_status(path_records, "clean"),
                "patched_status": best_status(path_records, "patched"),
                "runs": [record.__dict__ for record in path_records],
            }
        )

    by_suite_status = Counter((row["suite"], row["clean_status"]) for row in rows)
    by_class_status = Counter((row["runnable_class"], row["clean_status"]) for row in rows)
    by_category_status = Counter((row["category"], row["clean_status"]) for row in rows)
    by_signal_status: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for signal in row["signals"]:
            by_signal_status[(signal, row["clean_status"])] += 1

    return {
        "inventory_path": str(DEFAULT_INVENTORY),
        "logs_root": str(DEFAULT_LOGS),
        "total_entrypoints": len(rows),
        "total_summary_records": len(records),
        "matched_summary_records": len(records) - len(unmatched),
        "unmatched_summary_records": [record.__dict__ for record in unmatched],
        "counts": {
            "by_suite_status": {f"{suite}:{status}": count for (suite, status), count in sorted(by_suite_status.items())},
            "by_runnable_class_status": {
                f"{klass}:{status}": count for (klass, status), count in sorted(by_class_status.items())
            },
            "by_category_status": {
                f"{category}:{status}": count for (category, status), count in sorted(by_category_status.items())
            },
            "by_signal_status": {
                f"{signal}:{status}": count for (signal, status), count in sorted(by_signal_status.items())
            },
        },
        "entrypoints": rows,
    }


def status_count(rows: list[dict[str, Any]], suite: str | None = None) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if suite is None or row["suite"] == suite:
            counter[row["clean_status"]] += 1
    return counter


def write_markdown(coverage: dict[str, Any], path: Path) -> None:
    rows = coverage["entrypoints"]
    overall = status_count(rows)
    sim = status_count(rows, "isaacsim")
    lab = status_count(rows, "isaaclab")
    clean_failures = [row for row in rows if row["clean_status"] == "failed"]
    patched_successes = [
        row for row in rows if row["clean_status"] == "failed" and row["patched_status"] == "passed"
    ]
    next_candidates = [
        row
        for row in rows
        if row["clean_status"] == "uncovered"
        and row["suite"] == "isaacsim"
        and row["runnable_class"] in {"headless_candidate", "bounded_expected_timeout", "gui_or_patch_headless"}
    ]

    lines = [
        "# Isaac Sim 6 / Isaac Lab 3 Official Suite Coverage",
        "",
        f"Inventory entrypoints: {coverage['total_entrypoints']}",
        f"Smoke summary records: {coverage['total_summary_records']}",
        f"Matched summary records: {coverage['matched_summary_records']}",
        "",
        "## Clean Source Coverage",
        "",
        "| Scope | Passed | Failed | Uncovered |",
        "| --- | ---: | ---: | ---: |",
        f"| all | {overall['passed']} | {overall['failed']} | {overall['uncovered']} |",
        f"| isaacsim | {sim['passed']} | {sim['failed']} | {sim['uncovered']} |",
        f"| isaaclab | {lab['passed']} | {lab['failed']} | {lab['uncovered']} |",
        "",
        "## Clean Failures",
        "",
        "| Suite | Path | Runs |",
        "| --- | --- | --- |",
    ]
    for row in clean_failures[:60]:
        run_bits = []
        for record in row["runs"]:
            if record["source_variant"] != "clean":
                continue
            status = "pass" if record["passed"] else "fail"
            run_bits.append(f"{status}: {record['name']}")
        lines.append(f"| {row['suite']} | {row['relative_path']} | {'; '.join(run_bits)} |")

    lines.extend(["", "## Patched Successes For Clean Failures", "", "| Path | Patch Evidence |", "| --- | --- |"])
    for row in patched_successes[:60]:
        run_bits = [
            f"{'pass' if record['passed'] else 'fail'}: {record['name']}"
            for record in row["runs"]
            if record["source_variant"] == "patched"
        ]
        lines.append(f"| {row['relative_path']} | {'; '.join(run_bits)} |")

    lines.extend(["", "## Next Isaac Sim Candidates", "", "| Class | Path | Signals |", "| --- | --- | --- |"])
    for row in next_candidates[:80]:
        lines.append(
            f"| {row['runnable_class']} | {row['relative_path']} | {', '.join(row['signals']) or '-'} |"
        )

    unmatched = coverage["unmatched_summary_records"]
    lines.extend(["", "## Unmatched Summary Records", "", "| Name | Command |", "| --- | --- |"])
    for record in unmatched[:40]:
        lines.append(f"| {record['name']} | {record['command']} |")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    inventory = load_json(args.inventory)
    isaacsim_root = Path(inventory["isaacsim_root"])
    isaaclab_root = Path(inventory["isaaclab_root"])
    records = collect_records(args.logs_root, isaacsim_root, isaaclab_root)
    coverage = build_coverage(inventory, records)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    write_markdown(coverage, args.markdown)

    rows = coverage["entrypoints"]
    print(json.dumps({"overall": dict(status_count(rows)), "output": str(args.markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
