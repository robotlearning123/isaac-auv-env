#!/usr/bin/env python3
"""Release gate for the verified underwater robot demo and website surface."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_isaacsim_underwater_demo import (  # noqa: E402
    CAPTURE_PATH,
    EXPECTED_FPS,
    EXPECTED_FRAME_COUNT,
    EXPECTED_RESOLUTION,
    VIDEO_PATH,
    _ffprobe_video,
    _png_size,
    verify_demo_artifacts,
)

WEBSITE_DIR = REPO_ROOT / "website"
COMPONENT = WEBSITE_DIR / "src" / "components" / "UnderwaterRobotDemo.astro"
EN_PAGE = WEBSITE_DIR / "src" / "pages" / "index.astro"
ZH_PAGE = WEBSITE_DIR / "src" / "pages" / "zh" / "index.astro"
PUBLIC_DIR = WEBSITE_DIR / "public"
PUBLIC_VIDEO = PUBLIC_DIR / "videos" / "oceanscale-underwater-robot-demo.mp4"
PUBLIC_POSTER = PUBLIC_DIR / "images" / "oceanscale-underwater-robot-demo.png"
PUBLIC_MANIFEST = PUBLIC_DIR / "demo" / "underwater_robot_demo_manifest.json"
DEFAULT_SCREENSHOTS_DIR = REPO_ROOT / "artifacts" / "isaacsim" / "release_gate_layout"

VIDEO_ROUTE = "/videos/oceanscale-underwater-robot-demo.mp4"
POSTER_ROUTE = "/images/oceanscale-underwater-robot-demo.png"
MANIFEST_ROUTE = "/demo/underwater_robot_demo_manifest.json"
PYTEST_TARGETS = [
    "tests/test_isaacsim_underwater_demo_verifier.py",
    "tests/test_website_underwater_robot_demo.py",
]


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    ok: bool
    detail: str
    evidence: Any


CommandRunner = Callable[[str, Sequence[str], Path, int], ReleaseCheck]
BrowserRunner = Callable[[Path, int], ReleaseCheck]


def _check(name: str, ok: bool, detail: str, evidence: Any) -> ReleaseCheck:
    return ReleaseCheck(name=name, ok=bool(ok), detail=detail, evidence=evidence)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _tail(value: str, limit: int = 1600) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _run_command(name: str, args: Sequence[str], cwd: Path, timeout: int) -> ReleaseCheck:
    started_at = time.monotonic()
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _check(
            name,
            False,
            str(exc),
            {"args": list(args), "cwd": str(cwd), "timeout_seconds": timeout},
        )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return _check(
        name,
        completed.returncode == 0,
        f"exit {completed.returncode}",
        {
            "args": list(args),
            "cwd": str(cwd),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "output_tail": _tail(output),
        },
    )


def _expected_probe(probe: dict[str, Any]) -> bool:
    return (
        probe.get("width") == EXPECTED_RESOLUTION[0]
        and probe.get("height") == EXPECTED_RESOLUTION[1]
        and probe.get("nb_frames") == EXPECTED_FRAME_COUNT
        and probe.get("fps") == EXPECTED_FPS
    )


def _check_website_static(artifact_result: dict[str, Any]) -> ReleaseCheck:
    required_paths = [COMPONENT, EN_PAGE, ZH_PAGE, PUBLIC_VIDEO, PUBLIC_POSTER, PUBLIC_MANIFEST]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.exists()]
    if missing:
        return _check(
            "website_static_demo_surface", False, "missing required website files", missing
        )

    source = COMPONENT.read_text(encoding="utf-8")
    en_page = EN_PAGE.read_text(encoding="utf-8")
    zh_page = ZH_PAGE.read_text(encoding="utf-8")
    manifest = _read_json(PUBLIC_MANIFEST)
    public_video_probe = _ffprobe_video(PUBLIC_VIDEO)
    current_probe = artifact_result.get("artifacts", {}).get("video_probe")

    source_requirements = {
        "video_route": VIDEO_ROUTE in source,
        "poster_route": POSTER_ROUTE in source,
        "manifest_route": MANIFEST_ROUTE in source,
        "verified_status": "Verification: PASS" in source,
        "no_canvas": "<canvas" not in source,
        "no_webgl": "WebGL" not in source,
        "en_page_imports_component": "UnderwaterRobotDemo" in en_page,
        "zh_page_imports_component": "UnderwaterRobotDemo" in zh_page,
    }
    asset_requirements = {
        "public_manifest_ok": manifest.get("ok") is True,
        "public_manifest_probe": _expected_probe(
            manifest.get("artifacts", {}).get("video_probe", {})
        ),
        "public_video_probe": _expected_probe(public_video_probe),
        "public_poster_size": _png_size(PUBLIC_POSTER) == EXPECTED_RESOLUTION,
        "public_video_matches_canonical": _sha256(PUBLIC_VIDEO) == _sha256(VIDEO_PATH),
        "public_poster_matches_canonical": _sha256(PUBLIC_POSTER) == _sha256(CAPTURE_PATH),
    }
    if isinstance(current_probe, dict):
        asset_requirements["public_manifest_matches_current_probe"] = (
            manifest.get("artifacts", {}).get("video_probe") == current_probe
        )

    requirements = source_requirements | asset_requirements
    failures = sorted(name for name, ok in requirements.items() if not ok)
    return _check(
        "website_static_demo_surface",
        not failures,
        "website demo uses the verified Isaac assets" if not failures else ", ".join(failures),
        {
            "requirements": requirements,
            "public_video_probe": public_video_probe,
            "public_manifest": str(PUBLIC_MANIFEST.relative_to(REPO_ROOT)),
            "component": str(COMPONENT.relative_to(REPO_ROOT)),
        },
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[str], timeout: int) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            return False, f"preview exited early with {process.returncode}: {_tail(output)}"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True, f"http {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    return False, last_error or "preview did not become ready"


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _check_browser_layout(base_url: str, screenshots_dir: Path) -> ReleaseCheck:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return _check("website_browser_layout", False, str(exc), "playwright import failed")

    screenshots_dir.mkdir(parents=True, exist_ok=True)
    console_messages: list[str] = []
    layouts: dict[str, Any] = {}
    screenshot_paths: dict[str, str] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for name, viewport in {
                "desktop": {"width": 1440, "height": 1200},
                "mobile": {"width": 390, "height": 844},
            }.items():
                page = browser.new_page(viewport=viewport)
                page.on(
                    "console",
                    lambda message: console_messages.append(f"{message.type}: {message.text}"),
                )
                page.goto(f"{base_url}/#demo", wait_until="networkidle")
                page.wait_for_selector("#demo video", state="visible", timeout=10_000)
                layout = page.evaluate(
                    """() => {
                      const section = document.querySelector('#demo');
                      const title = section?.querySelector('.robot-demo__title');
                      const video = section?.querySelector('video');
                      const nav = document.querySelector('header, nav');
                      const titleRect = title?.getBoundingClientRect();
                      const videoRect = video?.getBoundingClientRect();
                      const navRect = nav?.getBoundingClientRect();
                      const html = document.documentElement;
                      return {
                        titleTop: titleRect?.top ?? -1,
                        navBottom: navRect?.bottom ?? 0,
                        titleClearOfNav: Boolean(titleRect && titleRect.top >= ((navRect?.bottom ?? 0) + 8)),
                        videoWidth: Math.round(videoRect?.width ?? 0),
                        videoHeight: Math.round(videoRect?.height ?? 0),
                        videoPoster: video?.getAttribute('poster') ?? '',
                        videoSrc: video?.currentSrc || video?.getAttribute('src') || '',
                        sourceCount: video?.querySelectorAll('source').length ?? 0,
                        scrollWidth: html.scrollWidth,
                        clientWidth: html.clientWidth,
                        noHorizontalOverflow: html.scrollWidth <= html.clientWidth + 1,
                      };
                    }"""
                )
                layouts[name] = layout
                screenshot_path = screenshots_dir / f"underwater_robot_demo_{name}.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                screenshot_paths[name] = str(screenshot_path.relative_to(REPO_ROOT))
                page.close()
        finally:
            browser.close()

    layout_ok = (
        layouts.get("desktop", {}).get("titleClearOfNav") is True
        and layouts.get("mobile", {}).get("titleClearOfNav") is True
        and layouts.get("desktop", {}).get("videoWidth", 0) >= 1000
        and layouts.get("desktop", {}).get("videoHeight", 0) >= 560
        and layouts.get("mobile", {}).get("videoWidth", 0) >= 320
        and layouts.get("mobile", {}).get("videoHeight", 0) >= 180
        and layouts.get("desktop", {}).get("noHorizontalOverflow") is True
        and layouts.get("mobile", {}).get("noHorizontalOverflow") is True
        and layouts.get("desktop", {}).get("videoPoster") == POSTER_ROUTE
        and layouts.get("mobile", {}).get("videoPoster") == POSTER_ROUTE
        and all(layout.get("sourceCount") == 1 for layout in layouts.values())
    )
    warnings = [
        message for message in console_messages if message.startswith(("warning:", "error:"))
    ]
    return _check(
        "website_browser_layout",
        layout_ok and not warnings,
        "demo is visible and clear in desktop/mobile preview"
        if layout_ok and not warnings
        else "browser layout smoke failed",
        {
            "layouts": layouts,
            "screenshots": screenshot_paths,
            "console_warnings": warnings,
        },
    )


def _run_browser_smoke(screenshots_dir: Path, timeout: int) -> ReleaseCheck:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    try:
        process = subprocess.Popen(
            ["pnpm", "preview", "--host", "127.0.0.1", "--port", str(port)],
            cwd=WEBSITE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return _check(
            "website_browser_layout",
            False,
            str(exc),
            {"args": ["pnpm", "preview"], "cwd": str(WEBSITE_DIR)},
        )

    try:
        ready, detail = _wait_for_http(base_url, process, timeout)
        if not ready:
            return _check("website_browser_layout", False, detail, {"url": base_url})
        return _check_browser_layout(base_url, screenshots_dir)
    finally:
        with contextlib.suppress(Exception):
            _stop_process(process)


def verify_release_gate(
    *,
    run_pytest: bool = True,
    run_build: bool = True,
    run_browser: bool = True,
    run_ffprobe: bool = True,
    timeout: int = 300,
    screenshots_dir: Path = DEFAULT_SCREENSHOTS_DIR,
    command_runner: CommandRunner = _run_command,
    browser_runner: BrowserRunner = _run_browser_smoke,
) -> dict[str, Any]:
    checks: list[ReleaseCheck] = []

    artifact_result = verify_demo_artifacts(run_ffprobe=run_ffprobe)
    checks.append(
        _check(
            "isaac_artifact_verifier",
            artifact_result["ok"],
            "canonical Isaac artifact verifier passed"
            if artifact_result["ok"]
            else "canonical Isaac artifact verifier failed",
            {
                "failed_checks": [
                    check["name"] for check in artifact_result["checks"] if not check["ok"]
                ],
                "artifacts": artifact_result["artifacts"],
            },
        )
    )
    checks.append(_check_website_static(artifact_result))

    if run_pytest:
        checks.append(
            command_runner(
                "targeted_pytest",
                [sys.executable, "-m", "pytest", *PYTEST_TARGETS, "-q"],
                REPO_ROOT,
                timeout,
            )
        )
    if run_build:
        checks.append(command_runner("website_build", ["pnpm", "build"], WEBSITE_DIR, timeout))
    if run_browser:
        checks.append(browser_runner(screenshots_dir, timeout))

    return {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the canonical underwater robot demo is release-ready on the website.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full release gate payload.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip targeted pytest checks.")
    parser.add_argument(
        "--skip-build", action="store_true", help="Skip the website production build."
    )
    parser.add_argument(
        "--skip-browser", action="store_true", help="Skip Playwright preview layout smoke."
    )
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Skip ffprobe checks for the canonical Isaac artifact verifier.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each command and preview startup.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=DEFAULT_SCREENSHOTS_DIR,
        help="Directory for browser smoke screenshots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_release_gate(
        run_pytest=not args.skip_pytest,
        run_build=not args.skip_build,
        run_browser=not args.skip_browser,
        run_ffprobe=not args.no_ffprobe,
        timeout=args.timeout,
        screenshots_dir=args.screenshots_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"underwater robot demo release gate: {status}")
        for check in result["checks"]:
            marker = "ok" if check["ok"] else "fail"
            print(f"- {marker}: {check['name']} -> {check['detail']}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
