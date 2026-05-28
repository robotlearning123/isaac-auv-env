"""Score a video using Kimi K2.6 multimodal via NVIDIA NIM API.

Extracts key frames from an MP4, sends them to Kimi K2.6 for visual quality
critique, and returns a structured score.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

import urllib.request

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MODEL = "moonshotai/kimi-k2.6"


def extract_frames(video_path: str, n_frames: int = 6) -> list[bytes]:
    """Extract evenly-spaced frames from video as JPEG bytes."""
    frames_raw = iio.imread(video_path, plugin="pyav")
    total = len(frames_raw)
    if total == 0:
        raise ValueError("No frames in video")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    result = []
    for idx in indices:
        img = frames_raw[idx]
        if Image is not None:
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            result.append(buf.getvalue())
        else:
            result.append(img.tobytes())
    return result


def score_video(video_path: str) -> dict:
    """Send video frames to Kimi K2.6 for visual quality scoring."""
    print(f"Extracting frames from {video_path}...")
    frames = extract_frames(video_path, n_frames=8)
    print(f"Extracted {len(frames)} frames")

    # Build multimodal message
    content_blocks = [
        {
            "type": "text",
            "text": (
                "You are a professional visual quality critic for scientific simulation videos. "
                "Analyze these frames from an ocean robotics simulation demo video.\n\n"
                "Score each criterion from 1-10, then give an overall score:\n\n"
                "1. ROBOT VISUALIZATION: Is the robot recognizable? Does it show legs when walking "
                "and thrusters when swimming? Is it more than a simple rectangle?\n"
                "2. ENVIRONMENT: Does the underwater scene have coral/reef details, sand textures, "
                "water surface effects, and underwater light/particulates?\n"
                "3. PARTICLE EFFECTS: Are there bubbles during underwater phase and splash effects "
                "at water surface crossing?\n"
                "4. HUD/TELEMETRY: Does the heads-up display show velocity, distance to target, "
                "mission progress, and blend factor?\n"
                "5. NARRATIVE: Is there an intro sequence, mission objective, phase transitions, "
                "and a mission summary at the end?\n"
                "6. TRAIL: Does the trail fade over distance instead of showing full history?\n"
                "7. TARGET MARKER: Is the target waypoint clearly shown with distance indicator?\n"
                "8. OVERALL VISUAL QUALITY: General cinematographic quality, color palette, "
                "layout, readability.\n\n"
                "Respond in this EXACT JSON format (no markdown, no code fence):\n"
                '{\n'
                '  "robot_visualization": <score>,\n'
                '  "environment": <score>,\n'
                '  "particle_effects": <score>,\n'
                '  "hud_telemetry": <score>,\n'
                '  "narrative": <score>,\n'
                '  "trail": <score>,\n'
                '  "target_marker": <score>,\n'
                '  "overall": <score>,\n'
                '  "comments": "<brief critique and improvement suggestions>"\n'
                '}'
            ),
        }
    ]

    for i, frame_bytes in enumerate(frames):
        b64 = base64.b64encode(frame_bytes).decode("utf-8")
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content_blocks}],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    print(f"Sending to Kimi K2.6 ({MODEL})...")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body[:500]}", file=sys.stderr)
        return {"error": f"HTTP {e.code}", "body": body[:500]}

    text = resp_data["choices"][0]["message"]["content"]
    print(f"\nRaw response:\n{text}\n")

    # Parse JSON from response
    try:
        # Try direct parse
        scores = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        import re
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            try:
                scores = json.loads(match.group())
            except json.JSONDecodeError:
                scores = {"raw": text}
        else:
            scores = {"raw": text}

    return scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video.mp4>")
        sys.exit(1)
    if not API_KEY:
        print("Set NVIDIA_API_KEY env var (NVIDIA NIM API key).")
        sys.exit(1)

    result = score_video(sys.argv[1])
    print("\n=== SCORES ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
