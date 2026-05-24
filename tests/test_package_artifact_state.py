from __future__ import annotations

import importlib.util
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_package_artifact_state.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("refresh_package_artifact_state", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inspect_wheel_detects_supported_data_and_entry_point(tmp_path: Path) -> None:
    script = _load_script()
    wheel = tmp_path / "oceanscale-0.1.0a0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("oceanscale/data/bluerov2_station_keep_final.zip", b"zip")
        archive.writestr("oceanscale/data/vec_normalize.npz", b"npz")
        archive.writestr("oceanscale-0.1.0a0.dist-info/METADATA", "Version: 0.1.0a0\n")
        archive.writestr(
            "oceanscale-0.1.0a0.dist-info/entry_points.txt",
            "[console_scripts]\noceanscale = oceanscale.cli:main\n",
        )

    result = script.inspect_wheel(wheel)

    assert result["metadata_version"] == "0.1.0a0"
    assert result["entry_point_present"] is True
    assert result["has_supported_zip"] is True
    assert result["has_supported_npz"] is True
    assert result["has_unsupported_pkl"] is False


def test_inspect_sdist_detects_unsupported_pickle(tmp_path: Path) -> None:
    script = _load_script()
    sdist = tmp_path / "oceanscale-0.1.0a0.tar.gz"
    data_dir = tmp_path / "oceanscale-0.1.0a0" / "oceanscale" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "bluerov2_station_keep_final.zip").write_bytes(b"zip")
    (data_dir / "vec_normalize.npz").write_bytes(b"npz")
    (data_dir / "vec_normalize.pkl").write_bytes(b"pkl")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(tmp_path / "oceanscale-0.1.0a0", arcname="oceanscale-0.1.0a0")

    result = script.inspect_sdist(sdist)

    assert result["has_supported_zip"] is True
    assert result["has_supported_npz"] is True
    assert result["has_unsupported_pkl"] is True


def test_evaluate_state_marks_clean_package_yellow_for_source_pkl() -> None:
    script = _load_script()
    status = script.evaluate_state(
        {
            "wheel": {
                "has_supported_zip": True,
                "has_supported_npz": True,
                "has_unsupported_pkl": False,
                "entry_point_present": True,
            },
            "sdist": {
                "has_supported_zip": True,
                "has_supported_npz": True,
                "has_unsupported_pkl": False,
            },
            "install_smoke": {"passed": True},
            "twine_check": {"passed": True},
            "source_data": {"source_pkl_exists": True},
        }
    )

    assert status["package_artifact"] == "green"
    assert status["source_data_hygiene"] == "yellow"
