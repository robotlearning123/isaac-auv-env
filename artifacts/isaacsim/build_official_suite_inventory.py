#!/usr/bin/env python3
"""Build an inventory of official Isaac Sim 6 and Isaac Lab 3 entrypoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/storage/isaacsim-6.0-official")
DEFAULT_ISAACSIM = DEFAULT_ROOT / "sources" / "IsaacSim-develop"
DEFAULT_ISAACLAB = DEFAULT_ROOT / "sources" / "IsaacLab-release-3.0.0-beta2"


@dataclass(frozen=True)
class Entrypoint:
    suite: str
    kind: str
    category: str
    relative_path: str
    runnable_class: str
    signals: tuple[str, ...]
    suggested_strategy: str


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def has_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def classify_path(relative_path: str, text: str) -> tuple[str, tuple[str, ...]]:
    path_lower = relative_path.lower()
    signals: list[str] = []

    if "ros2" in path_lower or has_any(text, ("ros2", "rclpy", "ROS_DISTRO")):
        signals.append("ros2")
    if "replicator" in path_lower or "sdg" in path_lower or has_any(text, ("omni.replicator", "rep.orchestrator")):
        signals.append("replicator_sdg")
    if "benchmark" in path_lower or "benchmarks" in path_lower:
        signals.append("benchmark")
    if "/testing/" in f"/{path_lower}/" or "/test/" in f"/{path_lower}/" or Path(relative_path).name.startswith("test_"):
        signals.append("test")
    if "camera" in path_lower or "sensor" in path_lower or has_any(text, ("enable_cameras", "Camera", "Lidar", "Radar")):
        signals.append("sensor_or_camera")
    if has_any(
        text,
        (
            'SimulationApp({"headless": False',
            'SimulationApp({"headless":False',
            "--viz kit",
            "--visualizer kit",
        ),
    ):
        signals.append("gui_default")
    if has_any(text, ("while simulation_app.is_running()", "while True:", "while app.is_running()")):
        signals.append("long_running_loop")
    if has_any(
        text,
        (
            "ISAAC_NUCLEUS_DIR",
            "NUCLEUS_ASSET_ROOT_DIR",
            "ISAACLAB_NUCLEUS_DIR",
            "get_assets_root_path",
            "nucleus",
            "omniverse://",
        ),
    ):
        signals.append("remote_or_nucleus_asset")
    if has_any(text, ("git lfs", "lfs", ".usd", ".urdf", ".mjcf", ".xml")):
        signals.append("asset_conversion_or_file_asset")
    if has_any(text, ("train.py", "max_iterations", "checkpoint", "tensorboard", "rl_games", "rsl_rl", "skrl", "sb3")):
        signals.append("training_or_rl")
    if "mimic" in path_lower or "robomimic" in path_lower:
        signals.append("mimic_or_imitation")
    if "teleop" in path_lower or has_any(text, ("keyboard", "spacemouse", "websocket", "teleoperation")):
        signals.append("teleoperation")
    if "newton" in path_lower or "newton" in text.lower():
        signals.append("newton")

    if path_lower.startswith("source/standalone_examples/api/"):
        category = path_lower.split("/")[3]
    elif path_lower.startswith("source/standalone_examples/tutorials/"):
        category = "standalone_tutorial"
    elif path_lower.startswith("source/standalone_examples/benchmarks/"):
        category = "standalone_benchmark"
    elif path_lower.startswith("source/standalone_examples/testing/"):
        parts = path_lower.split("/")
        category = "standalone_testing/" + (parts[3] if len(parts) > 3 else "unknown")
    elif path_lower.startswith("source/standalone_examples/replicator/"):
        category = "standalone_replicator"
    elif path_lower.startswith("scripts/tutorials/"):
        parts = path_lower.split("/")
        category = "lab_tutorial/" + (parts[2] if len(parts) > 2 else "unknown")
    elif path_lower.startswith("scripts/demos/"):
        parts = path_lower.split("/")
        category = "lab_demo/" + (parts[2] if len(parts) > 2 else "root")
    elif path_lower.startswith("scripts/environments/"):
        category = "lab_environment"
    elif path_lower.startswith("scripts/benchmarks/"):
        category = "lab_benchmark"
    elif path_lower.startswith("scripts/reinforcement_learning/"):
        category = "lab_reinforcement_learning"
    elif path_lower.startswith("scripts/imitation_learning/"):
        category = "lab_imitation_learning"
    elif path_lower.startswith("scripts/tools/"):
        category = "lab_tool"
    elif "/test/" in f"/{path_lower}/" or Path(path_lower).name.startswith("test_"):
        category = "lab_source_test"
    else:
        category = "other"

    return category, tuple(sorted(set(signals)))


def runnable_class(signals: tuple[str, ...]) -> tuple[str, str]:
    signal_set = set(signals)
    if "ros2" in signal_set:
        return "requires_ros2", "classify ROS distro/workspace first; do not run in base smoke batch"
    if "teleoperation" in signal_set:
        return "interactive_or_device_dependent", "manual/device validation with explicit hardware or websocket setup"
    if "mimic_or_imitation" in signal_set:
        return "optional_extra_blocked_or_long", "depends on optional mimic/robomimic stack; run after optional install is fixed"
    if "training_or_rl" in signal_set or "benchmark" in signal_set:
        return "long_runtime", "run as bounded benchmark/training subset with max_iterations/timeout"
    if "gui_default" in signal_set:
        return "gui_or_patch_headless", "prefer official --test/--headless path or run with DISPLAY/streaming"
    if "remote_or_nucleus_asset" in signal_set:
        return "asset_or_network_dependent", "pre-cache assets and record network/cache state before marking stable"
    if "long_running_loop" in signal_set:
        return "bounded_expected_timeout", "run with PYTHONUNBUFFERED and marker-based expected-timeout"
    return "headless_candidate", "run in bounded headless smoke batch with marker/exit checks"


def collect_python_files(root: Path, suite: str, base: Path, kind: str) -> list[Entrypoint]:
    entries: list[Entrypoint] = []
    for path in sorted(base.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        text = read_text(path)
        category, signals = classify_path(rel, text)
        run_class, strategy = runnable_class(signals)
        entries.append(
            Entrypoint(
                suite=suite,
                kind=kind,
                category=category,
                relative_path=rel,
                runnable_class=run_class,
                signals=signals,
                suggested_strategy=strategy,
            )
        )
    return entries


def build_inventory(isaacsim_root: Path, isaaclab_root: Path) -> dict[str, object]:
    entries: list[Entrypoint] = []
    entries.extend(
        collect_python_files(
            isaacsim_root,
            "isaacsim",
            isaacsim_root / "source" / "standalone_examples",
            "standalone_example",
        )
    )
    entries.extend(collect_python_files(isaaclab_root, "isaaclab", isaaclab_root / "scripts", "script"))
    entries.extend(collect_python_files(isaaclab_root, "isaaclab", isaaclab_root / "source", "source_test"))

    by_suite = Counter(entry.suite for entry in entries)
    by_class = Counter(entry.runnable_class for entry in entries)
    by_category = Counter(entry.category for entry in entries)
    by_signal: Counter[str] = Counter()
    for entry in entries:
        by_signal.update(entry.signals)

    class_examples: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        if len(class_examples[entry.runnable_class]) < 12:
            class_examples[entry.runnable_class].append(entry.relative_path)

    return {
        "isaacsim_root": str(isaacsim_root),
        "isaaclab_root": str(isaaclab_root),
        "total_entrypoints": len(entries),
        "counts": {
            "by_suite": dict(sorted(by_suite.items())),
            "by_runnable_class": dict(sorted(by_class.items())),
            "by_category": dict(sorted(by_category.items())),
            "by_signal": dict(sorted(by_signal.items())),
        },
        "examples_by_runnable_class": dict(sorted(class_examples.items())),
        "entrypoints": [asdict(entry) for entry in entries],
    }


def write_markdown(inventory: dict[str, object], output_path: Path) -> None:
    counts = inventory["counts"]
    assert isinstance(counts, dict)
    by_class = counts["by_runnable_class"]
    by_signal = counts["by_signal"]
    by_category = counts["by_category"]

    lines = [
        "# Isaac Sim 6 / Isaac Lab 3 Official Suite Inventory",
        "",
        f"Isaac Sim source: {inventory['isaacsim_root']}",
        f"Isaac Lab source: {inventory['isaaclab_root']}",
        f"Total Python entrypoints inventoried: {inventory['total_entrypoints']}",
        "",
        "## Runnable Classes",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for key, value in by_class.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Signals", "", "| Signal | Count |", "| --- | ---: |"])
    for key, value in by_signal.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Top Categories", "", "| Category | Count |", "| --- | ---: |"])
    for key, value in sorted(by_category.items(), key=lambda item: (-item[1], item[0]))[:40]:
        lines.append(f"| {key} | {value} |")

    examples = inventory["examples_by_runnable_class"]
    assert isinstance(examples, dict)
    lines.extend(["", "## Examples By Runnable Class", ""])
    for key, paths in examples.items():
        lines.append(f"### {key}")
        for path in paths:
            lines.append(f"- {path}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacsim-root", type=Path, default=DEFAULT_ISAACSIM)
    parser.add_argument("--isaaclab-root", type=Path, default=DEFAULT_ISAACLAB)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("/home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_inventory.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("/home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_inventory.md"),
    )
    args = parser.parse_args()

    inventory = build_inventory(args.isaacsim_root, args.isaaclab_root)
    args.json.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    write_markdown(inventory, args.markdown)
    print(json.dumps({k: inventory[k] for k in ("total_entrypoints", "counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
