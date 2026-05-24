from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

from packaging.version import Version

import oceanscale

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return str(pyproject["project"]["version"])


def test_pyproject_version_matches_installed_metadata() -> None:
    assert Version(_pyproject_version()) == Version(importlib.metadata.version("oceanscale"))


def test_pyproject_version_matches_package_dunder_version() -> None:
    assert Version(_pyproject_version()) == Version(oceanscale.__version__)
