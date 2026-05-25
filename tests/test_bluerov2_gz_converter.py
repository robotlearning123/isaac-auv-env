"""Checks for the optional bluerov2_gz Heavy mesh converter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_bluerov2_gz_to_usd.py"
MODEL_DIR = Path("/tmp/oceanscale-asset-candidates/bluerov2_gz/models/bluerov2_heavy")


def _load_converter_module():
    spec = importlib.util.spec_from_file_location("convert_bluerov2_gz_to_usd", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_converter_keeps_usd_imports_runtime_only() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "from pxr import" not in source.split("def parse_collada_mesh", maxsplit=1)[0]
    assert "DEFAULT_MODEL_DIR" in source
    assert "external_high_fidelity_mesh" in source


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="bluerov2_gz source checkout is not available")
def test_converter_authors_real_bluerov2_heavy_openusd(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")

    module = _load_converter_module()
    output_usd = tmp_path / "bluerov2_gz_heavy.usdc"

    summary = module.convert_model_to_usd(MODEL_DIR, output_usd)

    assert output_usd.exists()
    assert summary["root_prim"] == module.ROOT_PRIM
    assert summary["visual_mesh_count"] == 9
    assert summary["unique_collada_mesh_count"] == 3
    assert summary["triangle_instance_count"] > 100_000
    assert summary["source_license"] == "MIT"

    stage = Usd.Stage.Open(str(output_usd))
    root = stage.GetPrimAtPath(module.ROOT_PRIM)
    assert root.IsValid()
    assert stage.GetDefaultPrim() == root
    assert root.GetAttribute("oceanscale:assetRole").Get() == "external_high_fidelity_mesh"
    assert root.GetAttribute("oceanscale:vehicle").Get() == "BlueROV2 Heavy"
    assert root.GetAttribute("oceanscale:sourceLicense").Get() == "MIT"
    assert root.GetAttribute("oceanscale:thrusterCount").Get() == 8
    assert root.GetAttribute("oceanscale:visualMeshCount").Get() == 9
    assert root.GetAttribute("oceanscale:uniqueColladaMeshCount").Get() == 3
    assert stage.GetPrimAtPath(f"{module.ROOT_PRIM}/base_link/base_link_visual/mesh").IsValid()
    for index in range(1, 9):
        assert stage.GetPrimAtPath(
            f"{module.ROOT_PRIM}/thruster{index}/thruster_prop_visual/mesh"
        ).IsValid()
