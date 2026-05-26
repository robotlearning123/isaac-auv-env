"""Generate vector figures for the OceanScale working paper."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

OUT_DIR = Path("research/figures")

INK = "#17202a"
MUTED = "#52606d"
LINE = "#d6dee6"
PANEL = "#f7fafc"
GREEN = "#0f766e"
BLUE = "#1d4ed8"
ORANGE = "#9a3412"
RED = "#b91c1c"
AMBER = "#b45309"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text_lines(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if len(trial) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def t(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    family: str = "Arial, Helvetica, sans-serif",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="{family}">{esc(value)}</text>'
    )


def wrapped_text(
    x: float,
    y: float,
    value: str,
    *,
    width: int,
    size: int = 24,
    line_height: int = 30,
    fill: str = INK,
    weight: int = 400,
) -> str:
    lines = text_lines(value, width)
    return "\n".join(
        t(x, y + i * line_height, line, size=size, fill=fill, weight=weight)
        for i, line in enumerate(lines)
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "white",
    stroke: str = LINE,
    rx: int = 10,
    sw: float = 2,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = LINE, sw: float = 2) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" />'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, *, stroke: str = MUTED) -> str:
    return (
        line(x1, y1, x2, y2, stroke=stroke, sw=2.5)
        + f'<path d="M {x2:.1f} {y2:.1f} l -12 -7 l 0 14 z" fill="{stroke}" />'
    )


def svg(width: int, height: int, body: Iterable[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<rect width="100%" height="100%" fill="white" />',
            *body,
            "</svg>",
        ]
    )


def write_svg(name: str, width: int, height: int, body: Iterable[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(svg(width, height, body) + "\n", encoding="utf-8")


def figure_1_architecture() -> None:
    body: list[str] = [
        t(70, 70, "Figure 1 | OceanScale system architecture", size=34, weight=700),
        t(70, 108, "Evidence-bounded Tier-1 simulator stack for batched underwater robotics workloads", size=20, fill=MUTED),
    ]
    columns = [
        ("Inputs", 70, ["Vehicle assets", "Hydro coefficients", "Ocean fields", "Task configs"]),
        ("OceanScale Core", 470, ["Newton multi-worlds", "Warp hydro kernels", "Fossen 6-DOF", "Vectorized ROVEnv"]),
        ("Outputs", 870, ["Env-steps/s", "RL rollouts", "Fidelity metrics", "CLI demos"]),
        ("Boundaries", 1270, ["No sim-to-real claim", "No public release claim", "No contact-rich claim", "No world-model claim"]),
    ]
    for title, x, items in columns:
        body.append(rect(x, 170, 300, 540, fill=PANEL, stroke=LINE, rx=8))
        body.append(t(x + 22, 215, title, size=26, weight=700, fill=GREEN if title != "Boundaries" else ORANGE))
        for i, item in enumerate(items):
            y = 260 + i * 94
            body.append(rect(x + 24, y, 252, 56, fill="white", stroke=LINE, rx=6))
            body.append(t(x + 42, y + 36, item, size=20))
    for x in [390, 790, 1190]:
        body.append(arrow(x, 440, x + 54, 440))
    body.append(rect(470, 750, 700, 92, fill="#eef7f5", stroke=GREEN, rx=8, sw=2.5))
    body.append(t(495, 786, "Current strongest claim", size=22, weight=700, fill=GREEN))
    body.append(t(495, 820, "High-throughput, headless, vectorized dynamics for underwater robotics RL workloads", size=20))
    write_svg("fig1_system_architecture.svg", 1600, 900, body)


def figure_2_throughput() -> None:
    data = [(1, 1326), (64, 85824), (256, 303206), (1024, 1272106), (4096, 4588922)]
    max_v = max(v for _, v in data)
    body: list[str] = [
        t(70, 70, "Figure 2 | Standardized OceanScale throughput", size=34, weight=700),
        t(70, 108, "RTX 5090, Torch CUDA 12.8, Warp 1.13.0, Newton 1.2.0; exact labels, log-scaled bars", size=20, fill=MUTED),
    ]
    x0, y0, w, row_h = 260, 190, 1070, 96
    body.append(t(70, 180, "Parallel envs", size=20, weight=700))
    body.append(t(1320, 180, "Env-steps/s", size=20, weight=700, anchor="end"))
    import math

    for i, (envs, val) in enumerate(data):
        y = y0 + i * row_h
        body.append(t(85, y + 38, f"n = {envs}", size=22, weight=700))
        log_w = (math.log10(val) / math.log10(max_v)) * w
        body.append(rect(x0, y, w, 46, fill="#eef2f7", stroke="none", rx=5, sw=0))
        body.append(rect(x0, y, log_w, 46, fill=BLUE if envs < 4096 else GREEN, stroke="none", rx=5, sw=0))
        body.append(t(1320, y + 33, f"{val:,}", size=22, weight=700, anchor="end"))
    body.append(rect(70, 730, 1330, 88, fill=PANEL, stroke=LINE, rx=8))
    body.append(t(95, 765, "Interpretation", size=22, weight=700, fill=GREEN))
    body.append(t(95, 798, "The main performance claim is batched throughput, not single-environment latency.", size=20))
    write_svg("fig2_throughput_scaling.svg", 1500, 880, body)


def figure_3_pybullet() -> None:
    data = [
        ("PyBullet n=1", 1661, "1.00x"),
        ("OceanScale n=1", 337, "0.20x"),
        ("OceanScale n=16", 5076, "3.06x"),
        ("OceanScale n=64", 17427, "10.49x"),
    ]
    max_v = max(v for _, v, _ in data)
    body: list[str] = [
        t(70, 70, "Figure 3 | Matched PyBullet baseline comparison", size=34, weight=700),
        t(70, 108, "BlueROV2 hover task, 30K steps/run, two runs; matched comparison, not bit-for-bit identical", size=20, fill=MUTED),
    ]
    x0, y0, w, row_h = 320, 195, 900, 105
    for i, (label, val, speedup) in enumerate(data):
        y = y0 + i * row_h
        color = GREEN if "OceanScale n=64" in label else BLUE if "OceanScale" in label else ORANGE
        body.append(t(80, y + 39, label, size=22, weight=700))
        body.append(rect(x0, y, w, 50, fill="#eef2f7", stroke="none", rx=5, sw=0))
        body.append(rect(x0, y, (val / max_v) * w, 50, fill=color, stroke="none", rx=5, sw=0))
        body.append(t(1240, y + 35, f"{val:,} env-steps/s", size=21, weight=700))
        body.append(t(1450, y + 35, speedup, size=21, weight=700, fill=color, anchor="end"))
    body.append(rect(70, 690, 1390, 95, fill=PANEL, stroke=LINE, rx=8))
    body.append(t(95, 728, "Boundary", size=22, weight=700, fill=ORANGE))
    body.append(t(95, 760, "PyBullet wins at n=1; OceanScale's advantage emerges from vectorized rollouts.", size=20))
    write_svg("fig3_pybullet_comparison.svg", 1530, 860, body)


def figure_4_fidelity() -> None:
    metrics = [
        ("Position error", "0.013%", GREEN),
        ("Attitude RMS", "0.0025 deg", GREEN),
        ("Position RMS", "0.000223 m", BLUE),
        ("Position max", "0.000414 m", BLUE),
    ]
    body: list[str] = [
        t(70, 70, "Figure 4 | Fidelity evidence and caveats", size=34, weight=700),
        t(70, 108, "von Benzon BlueROV2 Heavy scenario: 5.0 N surge, 1000 steps, dt = 0.01 s", size=20, fill=MUTED),
    ]
    for i, (label, value, color) in enumerate(metrics):
        x = 80 + (i % 2) * 675
        y = 190 + (i // 2) * 190
        body.append(rect(x, y, 595, 140, fill=PANEL, stroke=LINE, rx=8))
        body.append(t(x + 32, y + 52, label, size=22, fill=MUTED, weight=700))
        body.append(t(x + 32, y + 108, value, size=44, fill=color, weight=800))
    body.append(rect(80, 620, 1270, 128, fill="#fff7ed", stroke=ORANGE, rx=8, sw=2))
    body.append(t(110, 660, "Caveat for paper text", size=22, weight=700, fill=ORANGE))
    body.append(wrapped_text(110, 700, "The regression suite also includes a frozen self-generated snapshot and parameter-fidelity guardrails with known differences; do not claim full tank or deployment validation.", width=118, size=20, fill=INK, line_height=30))
    write_svg("fig4_fidelity_evidence.svg", 1420, 820, body)


def figure_5_readiness() -> None:
    items = [
        ("Full tests", "PASS", GREEN, "1027 passed"),
        ("Internal verifier", "PASS", GREEN, "15 checks"),
        ("Benchmark JSON", "PASS", GREEN, "valid artifact"),
        ("Ruff", "FAIL", RED, "73 errors"),
        ("Mypy", "FAIL", RED, "299 errors"),
        ("Public release", "DEFERRED", AMBER, "not public"),
        ("PyPI", "DEFERRED", AMBER, "no publish"),
        ("Live claims", "STALE", AMBER, "alignment needed"),
    ]
    body: list[str] = [
        t(70, 70, "Figure 5 | Readiness matrix for paper claims", size=34, weight=700),
        t(70, 108, "Separate local runtime evidence from release, type-safety, and public-surface readiness", size=20, fill=MUTED),
    ]
    card_w, card_h = 310, 142
    for i, (label, status, color, note) in enumerate(items):
        x = 70 + (i % 4) * 350
        y = 190 + (i // 4) * 205
        body.append(rect(x, y, card_w, card_h, fill=PANEL, stroke=LINE, rx=8))
        body.append(t(x + 24, y + 44, label, size=22, weight=700))
        body.append(rect(x + 24, y + 66, 144, 36, fill=color, stroke="none", rx=18, sw=0))
        body.append(t(x + 96, y + 91, status, size=16, weight=800, fill="white", anchor="middle"))
        body.append(t(x + 24, y + 126, note, size=19, fill=MUTED))
    body.append(rect(70, 640, 1360, 92, fill="#eef7f5", stroke=GREEN, rx=8))
    body.append(t(95, 678, "Use in manuscript", size=22, weight=700, fill=GREEN))
    body.append(t(95, 711, "Claim alpha research readiness; avoid release-grade or public-launch language until failed/deferred gates are resolved.", size=20))
    write_svg("fig5_readiness_matrix.svg", 1500, 800, body)


def main() -> None:
    figure_1_architecture()
    figure_2_throughput()
    figure_3_pybullet()
    figure_4_fidelity()
    figure_5_readiness()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
