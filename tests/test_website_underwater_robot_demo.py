"""Website wiring checks for the verified underwater robot demo assets."""

from __future__ import annotations

import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENT = REPO_ROOT / "website" / "src" / "components" / "UnderwaterRobotDemo.astro"
PUBLIC = REPO_ROOT / "website" / "public"
VIDEO = PUBLIC / "videos" / "oceanscale-underwater-robot-demo.mp4"
POSTER = PUBLIC / "images" / "oceanscale-underwater-robot-demo.png"
MANIFEST = PUBLIC / "demo" / "underwater_robot_demo_manifest.json"


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_website_demo_uses_verified_isaac_artifacts() -> None:
    source = COMPONENT.read_text(encoding="utf-8")

    assert "/videos/oceanscale-underwater-robot-demo.mp4" in source
    assert "/images/oceanscale-underwater-robot-demo.png" in source
    assert "/demo/underwater_robot_demo_manifest.json" in source
    assert "Isaac Sim PathTracing" in source
    assert "Verification: PASS" in source
    assert "<canvas" not in source
    assert "WebGL" not in source


def test_website_demo_public_assets_match_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["ok"] is True
    assert manifest["artifacts"]["video_probe"]["width"] == 1920
    assert manifest["artifacts"]["video_probe"]["height"] == 1080
    assert manifest["artifacts"]["video_probe"]["nb_frames"] == 24
    assert manifest["artifacts"]["video_probe"]["fps"] == 12.0
    assert _png_size(POSTER) == (1920, 1080)

    with VIDEO.open("rb") as handle:
        header = handle.read(16)
    assert b"ftyp" in header
