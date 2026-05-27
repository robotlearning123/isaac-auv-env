"""Shared fixtures for OceanScale tests."""

from __future__ import annotations

from pathlib import Path

import pytest

VIDEO_DIR = Path(__file__).resolve().parent.parent / "tmp" / "test_videos"


@pytest.fixture()
def video_dir() -> Path:
    """Return the test video output directory, creating it if needed."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return VIDEO_DIR
