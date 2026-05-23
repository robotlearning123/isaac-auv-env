"""VideoExporter — headless MP4 rendering for ROVEnv episodes.

Uses matplotlib Agg backend (no GUI) + imageio-ffmpeg for MP4 output.
Renders top-down (x-y) or side (x-z) view showing vehicle position,
orientation arrow, target hover point, and trajectory trail.

Usage:
    exporter = VideoExporter('/tmp/ep.mp4', fps=30, view='side')
    for state in episode_states:
        exporter.record_frame(state)
    exporter.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v3 as iio
except ImportError:
    raise ImportError(
        "imageio + imageio-ffmpeg required. Install with: uv pip install imageio imageio-ffmpeg"
    )


def _quat_yaw(q: np.ndarray) -> float:
    """Extract yaw angle from quaternion (xyzw)."""
    return float(2.0 * np.arctan2(q[2], q[3]))


class VideoExporter:
    """Record frames from ROV state and export to MP4.

    Parameters
    ----------
    output_path : str
        Destination MP4 file path.
    fps : int
        Frames per second for output video.
    view : str
        'top' for x-y plane (bird's eye), 'side' for x-z plane (elevation).
    trail_length : int
        Number of past positions to draw as trajectory trail.
    figsize : tuple[int, int]
        Matplotlib figure size in inches.
    dpi : int
        Dots per inch for rendered frames.
    cinematic : bool
        If True, use 4-panel layout with HUD (side view + top view +
        depth error chart + thruster bars).
    title_card : str | None
        If set, prepend a title card frame with this text.
    end_card : str | None
        If set, append an end card frame with this text.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        view: str = "side",
        trail_length: int = 200,
        figsize: tuple[int, int] = (8, 6),
        dpi: int = 100,
        cinematic: bool = False,
        title_card: str | None = None,
        end_card: str | None = None,
    ) -> None:
        if view not in ("top", "side"):
            raise ValueError(f"view must be 'top' or 'side', got {view!r}")

        self._output_path = Path(output_path)
        self._fps = fps
        self._view = view
        self._trail_length = trail_length
        self._figsize = figsize
        self._dpi = dpi
        self._cinematic = cinematic
        self._title_card = title_card
        self._end_card = end_card

        self._frames: list[np.ndarray] = []
        self._trail: list[np.ndarray] = []
        self._depth_errors: list[float] = []
        self._thruster_actions: list[np.ndarray] = []
        self._step_times: list[float] = []

        # Axis mapping: which position indices to plot
        # top view: x=0, y=1; side view: x=0, z=2
        self._ix = 0
        self._iy = 1 if view == "top" else 2

        self._target: np.ndarray | None = None

        if cinematic:
            self._figsize = (16, 9)
            self._dpi = 100
            self._fig, self._axes = plt.subplots(
                2, 2, figsize=self._figsize, dpi=self._dpi,
                gridspec_kw={"hspace": 0.35, "wspace": 0.3},
            )
        else:
            self._fig, self._ax = plt.subplots(figsize=figsize, dpi=dpi)

    def record_frame(
        self,
        state_dict: dict[str, Any],
        target_pos: np.ndarray | None = None,
        action: np.ndarray | None = None,
        step_time: float | None = None,
    ) -> None:
        """Capture a single frame from environment state.

        Parameters
        ----------
        state_dict : dict
            Must contain 'pos' (3,) and 'quat' (4,). 'vel' (3,) accepted
            but unused by the current renderer.
            Optionally 'target_pos' (3,) — sets target if not yet set.
        target_pos : np.ndarray, optional
            Override target position for this frame.
        action : np.ndarray, optional
            Thruster action vector (8,) for cinematic thruster bar chart.
        step_time : float, optional
            Elapsed simulation time for cinematic HUD.
        """
        pos = np.asarray(state_dict["pos"], dtype=np.float64)
        quat = np.asarray(state_dict["quat"], dtype=np.float64)

        if target_pos is not None:
            self._target = np.asarray(target_pos, dtype=np.float64)
        elif "target_pos" in state_dict and self._target is None:
            self._target = np.asarray(state_dict["target_pos"], dtype=np.float64)

        self._trail.append(pos.copy())
        if len(self._trail) > self._trail_length:
            self._trail = self._trail[-self._trail_length:]

        if self._target is not None:
            self._depth_errors.append(abs(pos[2] - self._target[2]))

        if action is not None:
            self._thruster_actions.append(np.asarray(action).copy())

        if step_time is not None:
            self._step_times.append(step_time)

        if self._cinematic:
            self._render_cinematic(pos, quat, action)
        else:
            self._render_frame(pos, quat)

    def _draw_trail(self, ax: plt.Axes, trail: np.ndarray, ix: int, iy: int) -> None:
        n = len(trail)
        for i in range(max(0, n - 100), n - 1):
            alpha = 0.1 + 0.9 * (i - max(0, n - 100)) / min(n, 100)
            ax.plot(
                trail[i : i + 2, ix], trail[i : i + 2, iy],
                color="steelblue", alpha=alpha, linewidth=1.5,
            )

    def _draw_vehicle(self, ax: plt.Axes, pos: np.ndarray, quat: np.ndarray,
                      ix: int, iy: int, yaw: float) -> None:
        arrow_len = 0.3
        if iy == 2:  # side view
            dx, dy = arrow_len * np.cos(yaw), 0.0
        else:  # top view
            dx, dy = arrow_len * np.cos(yaw), arrow_len * np.sin(yaw)
        ax.annotate(
            "", xy=(pos[ix] + dx, pos[iy] + dy), xytext=(pos[ix], pos[iy]),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=2),
        )
        ax.plot(pos[ix], pos[iy], marker="o", color="navy", markersize=6)

    def _auto_scale(self, ax: plt.Axes, trail: np.ndarray, pos: np.ndarray,
                    ix: int, iy: int) -> None:
        margin = 1.5
        all_x = np.append(trail[:, ix], pos[ix])
        all_y = np.append(trail[:, iy], pos[iy])
        if self._target is not None:
            all_x = np.append(all_x, self._target[ix])
            all_y = np.append(all_y, self._target[iy])
        cx, cy = np.mean(all_x), np.mean(all_y)
        span = max(np.ptp(all_x) + 2 * margin, np.ptp(all_y) + 2 * margin, 2.0)
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)

    def _render_cinematic(self, pos: np.ndarray, quat: np.ndarray,
                          action: np.ndarray | None) -> None:
        axes = self._axes
        for ax in axes.flat:
            ax.cla()

        yaw = _quat_yaw(quat)
        trail = np.array(self._trail) if self._trail else np.array([pos])

        # Top-left: side view (x-z)
        ax_side = axes[0, 0]
        labels = {0: "X (m)", 2: "Z (m)"}
        ax_side.set_xlabel(labels[0], fontsize=8)
        ax_side.set_ylabel(labels[2], fontsize=8)
        ax_side.set_title("Side View", fontsize=9, fontweight="bold")
        ax_side.set_aspect("equal")
        ax_side.grid(True, alpha=0.3)
        self._draw_trail(ax_side, trail, 0, 2)
        if self._target is not None:
            ax_side.plot(self._target[0], self._target[2], marker="x", color="red",
                         markersize=10, markeredgewidth=2)
        self._draw_vehicle(ax_side, pos, quat, 0, 2, yaw)
        self._auto_scale(ax_side, trail, pos, 0, 2)
        ax_side.tick_params(labelsize=7)

        # Top-right: top view (x-y)
        ax_top = axes[0, 1]
        ax_top.set_xlabel("X (m)", fontsize=8)
        ax_top.set_ylabel("Y (m)", fontsize=8)
        ax_top.set_title("Top View", fontsize=9, fontweight="bold")
        ax_top.set_aspect("equal")
        ax_top.grid(True, alpha=0.3)
        self._draw_trail(ax_top, trail, 0, 1)
        if self._target is not None:
            ax_top.plot(self._target[0], self._target[1], marker="x", color="red",
                         markersize=10, markeredgewidth=2)
        self._draw_vehicle(ax_top, pos, quat, 0, 1, yaw)
        self._auto_scale(ax_top, trail, pos, 0, 1)
        ax_top.tick_params(labelsize=7)

        # Bottom-left: depth error time series (last 10s)
        ax_depth = axes[1, 0]
        ax_depth.set_xlabel("Time (s)", fontsize=8)
        ax_depth.set_ylabel("Depth Error (m)", fontsize=8)
        ax_depth.set_title("Depth Error", fontsize=9, fontweight="bold")
        ax_depth.grid(True, alpha=0.3)
        if self._depth_errors:
            window = 10.0  # last 10 seconds
            if self._step_times:
                t_arr = np.array(self._step_times)
                e_arr = np.array(self._depth_errors)
                mask = t_arr >= t_arr[-1] - window
                ax_depth.plot(t_arr[mask], e_arr[mask], color="crimson", linewidth=1.5)
                ax_depth.fill_between(t_arr[mask], 0, e_arr[mask], alpha=0.15, color="crimson")
            else:
                n_de = len(self._depth_errors)
                max_pts = int(window * self._fps)
                recent = self._depth_errors[-max_pts:]
                ax_depth.plot(recent, color="crimson", linewidth=1.5)
        ax_depth.tick_params(labelsize=7)
        ax_depth.set_ylim(bottom=0)

        # Bottom-right: thruster activity bars
        ax_thr = axes[1, 1]
        ax_thr.set_title("Thruster Activity", fontsize=9, fontweight="bold")
        ax_thr.set_ylim(-1, 1)
        ax_thr.set_ylabel("Thrust [-1, +1]", fontsize=8)
        ax_thr.grid(True, alpha=0.3, axis="y")
        if action is not None and len(action) >= 8:
            colors = ["#2196F3" if v >= 0 else "#FF5722" for v in action[:8]]
            ax_thr.bar(range(1, 9), action[:8], color=colors, edgecolor="white",
                       linewidth=0.5, width=0.7)
            ax_thr.set_xticks(range(1, 9))
            ax_thr.set_xticklabels([f"T{i}" for i in range(1, 9)], fontsize=6)
        elif self._thruster_actions:
            last = self._thruster_actions[-1]
            if len(last) >= 8:
                colors = ["#2196F3" if v >= 0 else "#FF5722" for v in last[:8]]
                ax_thr.bar(range(1, 9), last[:8], color=colors, edgecolor="white",
                           linewidth=0.5, width=0.7)
                ax_thr.set_xticks(range(1, 9))
                ax_thr.set_xticklabels([f"T{i}" for i in range(1, 9)], fontsize=6)
        ax_thr.tick_params(labelsize=7)

        # HUD overlay
        depth_err = abs(pos[2] - self._target[2]) if self._target is not None else 0
        pos_err = float(np.linalg.norm(pos - self._target)) if self._target is not None else 0
        t_now = self._step_times[-1] if self._step_times else len(self._frames) / self._fps
        hud = f"Depth: {pos[2]:.2f}m  Target: {self._target[2]:.2f}m  |  " \
              f"Err: {depth_err:.3f}m  Pos: {pos_err:.3f}m  |  t={t_now:.1f}s"
        self._fig.suptitle(hud, fontsize=9, fontfamily="monospace", color="#333333", y=0.98)

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        self._frames.append(buf[:, :, :3].copy())

    def _render_frame(self, pos: np.ndarray, quat: np.ndarray) -> None:
        ax = self._ax
        ax.cla()

        ix, iy = self._ix, self._iy

        labels = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}
        ax.set_xlabel(labels[ix])
        ax.set_ylabel(labels[iy])
        ax.set_title(f"BlueROV2 — {self._view} view" + (f"  (z={pos[2]:.2f}m)" if self._view == "top" else ""))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        trail = np.array(self._trail) if self._trail else np.array([pos])
        self._draw_trail(ax, trail, ix, iy)

        if self._target is not None:
            ax.plot(
                self._target[ix], self._target[iy],
                marker="x", color="red", markersize=12, markeredgewidth=2,
                label="target",
            )

        ax.plot(pos[ix], pos[iy], marker="o", color="navy", markersize=8, label="ROV")

        yaw = _quat_yaw(quat)
        self._draw_vehicle(ax, pos, quat, ix, iy, yaw)
        self._auto_scale(ax, trail, pos, ix, iy)

        ax.legend(loc="upper right", fontsize=8)

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        self._frames.append(buf[:, :, :3].copy())

    def _make_card_frame(self, text: str) -> np.ndarray:
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor("#0a0e1a")
        fig.patch.set_facecolor("#0a0e1a")
        ax.axis("off")
        for line in text.split("\n"):
            pass
        ax.text(0.5, 0.55, text, transform=ax.transAxes,
                fontsize=16, fontfamily="monospace", color="white",
                ha="center", va="center", linespacing=1.8)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()
        plt.close(fig)
        return frame

    def close(self) -> Path:
        """Flush buffered frames to MP4 and return output path.

        Returns
        -------
        Path
            Path to the written MP4 file.
        """
        if not self._frames:
            raise RuntimeError("No frames recorded — call record_frame() first")

        all_frames = []

        # Title card (2s)
        if self._title_card:
            title_frame = self._make_card_frame(self._title_card)
            for _ in range(self._fps * 2):
                all_frames.append(title_frame)

        all_frames.extend(self._frames)

        # End card (2s)
        if self._end_card:
            end_frame = self._make_card_frame(self._end_card)
            for _ in range(self._fps * 2):
                all_frames.append(end_frame)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        iio.imwrite(
            str(self._output_path),
            all_frames,
            fps=self._fps,
            codec="libx264",
            pixelformat="yuv420p",
        )

        plt.close(self._fig)

        n_frames = len(all_frames)
        duration = n_frames / self._fps
        self._frames.clear()

        print(
            f"VideoExporter: wrote {n_frames} frames "
            f"({duration:.1f}s @ {self._fps}fps) → {self._output_path}"
        )
        return self._output_path

    @property
    def frame_count(self) -> int:
        """Number of frames buffered."""
        return len(self._frames)
