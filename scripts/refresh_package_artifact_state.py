from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json")
DEFAULT_BUILD_DIR = Path("/tmp/oceanscale-package-artifact-state-20260524")
SUPPORTED_DATA_FILES = {
    "oceanscale/data/bluerov2_station_keep_final.zip",
    "oceanscale/data/vec_normalize.npz",
}
UNSUPPORTED_DATA_FILES = {"oceanscale/data/vec_normalize.pkl"}


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


def source_data_state(root: Path = ROOT) -> dict[str, Any]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = pyproject["tool"]["setuptools"]["package-data"]["oceanscale"]
    data_files = []
    for path in sorted((root / "oceanscale" / "data").glob("*")):
        if path.is_file():
            data_files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "supported": path.name in {
                        "bluerov2_station_keep_final.zip",
                        "vec_normalize.npz",
                    },
                    "packaged_by_pattern": (
                        path.suffix == ".zip" and "data/*.zip" in patterns
                    )
                    or (path.suffix == ".npz" and "data/*.npz" in patterns),
                }
            )

    return {
        "package_data_patterns": patterns,
        "data_files": data_files,
        "supported_files_exist": all((root / path).is_file() for path in SUPPORTED_DATA_FILES),
        "source_pkl_exists": (root / "oceanscale/data/vec_normalize.pkl").is_file(),
    }


def build_artifacts(build_dir: Path = DEFAULT_BUILD_DIR, root: Path = ROOT) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(["uv", "build", "--out-dir", str(build_dir)], cwd=root)
    wheels = sorted(path for path in build_dir.glob("*.whl") if path.is_file())
    sdists = sorted(path for path in build_dir.glob("*.tar.gz") if path.is_file())
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "build_dir": str(build_dir),
        "wheels": [str(path) for path in wheels],
        "sdists": [str(path) for path in sdists],
    }


def twine_check(paths: list[Path], root: Path = ROOT) -> dict[str, Any]:
    result = run_command(
        ["uvx", "--from", "twine", "twine", "check", *[str(path) for path in paths]],
        cwd=root,
    )
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.exit_code == 0,
    }


def _metadata_version(metadata: str) -> str:
    for line in metadata.splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    return ""


def inspect_wheel(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        sizes = {info.filename: info.file_size for info in archive.infolist()}
        metadata_name = next((name for name in names if name.endswith(".dist-info/METADATA")), "")
        entry_points_name = next(
            (name for name in names if name.endswith(".dist-info/entry_points.txt")), ""
        )
        metadata = archive.read(metadata_name).decode() if metadata_name else ""
        entry_points = archive.read(entry_points_name).decode() if entry_points_name else ""

    data_entries = sorted(name for name in names if name.startswith("oceanscale/data/"))
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "metadata_version": _metadata_version(metadata),
        "entry_point_present": "oceanscale = oceanscale.cli:main" in entry_points,
        "data_entries": [{"path": name, "size": sizes[name]} for name in data_entries],
        "has_supported_zip": "oceanscale/data/bluerov2_station_keep_final.zip" in names,
        "has_supported_npz": "oceanscale/data/vec_normalize.npz" in names,
        "has_unsupported_pkl": any(name.endswith(".pkl") for name in data_entries),
    }


def inspect_sdist(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()

    data_members = sorted(
        (member for member in members if "/oceanscale/data/" in member.name and member.isfile()),
        key=lambda member: member.name,
    )
    names = [member.name for member in members]
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "data_entries": [{"path": member.name, "size": member.size} for member in data_members],
        "has_supported_zip": any(name.endswith("/oceanscale/data/bluerov2_station_keep_final.zip") for name in names),
        "has_supported_npz": any(name.endswith("/oceanscale/data/vec_normalize.npz") for name in names),
        "has_unsupported_pkl": any(name.endswith(".pkl") for name in names),
    }


def install_smoke(wheel: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="oceanscale-wheel-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = venv_dir / "bin" / "python"
        install = run_command([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], cwd=Path("/tmp"))
        probe = run_command(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata as md, importlib.resources as res, json, oceanscale; "
                    "data=res.files('oceanscale.data'); "
                    "entries=sorted(child.name for child in data.iterdir()); "
                    "print(json.dumps({"
                    "'version': oceanscale.__version__, "
                    "'metadata_version': md.version('oceanscale'), "
                    "'entry_points': [str(ep) for ep in md.entry_points(group='console_scripts') "
                    "if ep.name == 'oceanscale'], "
                    "'data_entries': entries, "
                    "'has_zip': 'bluerov2_station_keep_final.zip' in entries, "
                    "'has_npz': 'vec_normalize.npz' in entries, "
                    "'has_pkl': 'vec_normalize.pkl' in entries"
                    "}))"
                ),
            ],
            cwd=Path("/tmp"),
        )
        parsed: dict[str, Any] = {}
        if probe.exit_code == 0:
            parsed = json.loads(probe.stdout)

        return {
            "install": {
                "command": install.command,
                "exit_code": install.exit_code,
                "stdout": install.stdout,
                "stderr": install.stderr,
            },
            "probe": {
                "command": probe.command,
                "exit_code": probe.exit_code,
                "stdout": probe.stdout,
                "stderr": probe.stderr,
                "parsed": parsed,
            },
            "passed": install.exit_code == 0
            and probe.exit_code == 0
            and parsed.get("version") == parsed.get("metadata_version")
            and parsed.get("has_zip") is True
            and parsed.get("has_npz") is True
            and parsed.get("has_pkl") is False
            and bool(parsed.get("entry_points")),
        }


def evaluate_state(state: dict[str, Any]) -> dict[str, str]:
    wheel = state.get("wheel", {})
    sdist = state.get("sdist", {})
    install = state.get("install_smoke", {})
    twine = state.get("twine_check", {})
    source = state.get("source_data", {})
    package_green = (
        wheel.get("has_supported_zip") is True
        and wheel.get("has_supported_npz") is True
        and wheel.get("has_unsupported_pkl") is False
        and wheel.get("entry_point_present") is True
        and sdist.get("has_supported_zip") is True
        and sdist.get("has_supported_npz") is True
        and sdist.get("has_unsupported_pkl") is False
        and install.get("passed") is True
        and twine.get("passed") is True
    )
    return {
        "package_artifact": "green" if package_green else "red",
        "source_data_hygiene": "yellow" if source.get("source_pkl_exists") else "green",
    }


def build_state(root: Path = ROOT, build_dir: Path = DEFAULT_BUILD_DIR) -> dict[str, Any]:
    source = source_data_state(root)
    build = build_artifacts(build_dir, root)
    wheels = [Path(path) for path in build["wheels"]]
    sdists = [Path(path) for path in build["sdists"]]
    wheel = inspect_wheel(wheels[-1]) if wheels else {}
    sdist = inspect_sdist(sdists[-1]) if sdists else {}
    twine = twine_check([*wheels, *sdists], root) if wheels and sdists else {}
    install = install_smoke(wheels[-1]) if wheels else {}
    state = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "internal_alpha_readiness",
        "public_launch": False,
        "repository": str(root),
        "source_data": source,
        "build": build,
        "twine_check": twine,
        "wheel": wheel,
        "sdist": sdist,
        "install_smoke": install,
        "approvals_required_to_fix": ["delete oceanscale/data/vec_normalize.pkl"],
    }
    state["status"] = evaluate_state(state)
    return state


def write_state(state: dict[str, Any], output: Path, root: Path = ROOT) -> Path:
    target = output if output.is_absolute() else root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh package artifact state.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    args = parser.parse_args(argv)

    state = build_state(ROOT, args.build_dir)
    target = write_state(state, args.output, ROOT)
    print(f"Wrote {target.relative_to(ROOT)}")
    print(json.dumps(state["status"], indent=2, sort_keys=True))
    return 0 if state["status"]["package_artifact"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
