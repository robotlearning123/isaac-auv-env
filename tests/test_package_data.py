from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "oceanscale" / "data"


def _package_data_patterns() -> list[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    return list(package_data["oceanscale"])


def test_supported_data_artifacts_exist() -> None:
    assert (DATA_DIR / "bluerov2_station_keep_final.zip").is_file()
    assert (DATA_DIR / "vec_normalize.npz").is_file()


def test_pyproject_packages_supported_data_artifacts() -> None:
    patterns = _package_data_patterns()

    assert "data/*.zip" in patterns
    assert "data/*.npz" in patterns


def test_pyproject_does_not_package_pickle_artifacts() -> None:
    patterns = _package_data_patterns()

    assert "data/*.pkl" not in patterns
    assert "data/*" not in patterns


def test_source_pickle_artifacts_are_not_packaged() -> None:
    patterns = _package_data_patterns()
    pickle_artifacts = sorted(DATA_DIR.glob("*.pkl"))

    for artifact in pickle_artifacts:
        relative = artifact.relative_to(ROOT / "oceanscale").as_posix()
        assert relative not in patterns
