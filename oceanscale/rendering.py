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
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        view: str = "side",
        trail_length: int = 200,
        figsize: tuple[int, int] = (8, 6),
        dpi: int = 100,  # 800x600 → resized to 800x608 by ffmpeg (macro_block_size=16)
    ) -> None:
        if view not in ("top", "side"):
            raise ValueError(f"view must be 'top' or 'side', got {view!r}")

        self._output_path = Path(output_path)
        self._fps = fps
        self._view = view
        self._trail_length = trail_length
        self._figsize = figsize
        self._dpi = dpi

        self._frames: list[np.ndarray] = []
        self._trail: list[np.ndarray] = []

        # Axis mapping: which position indices to plot
        # top view: x=0, y=1; side view: x=0, z=2
        self._ix = 0
        self._iy = 1 if view == "top" else 2

        # Pre-create figure and axes (reuse for every frame)
        self._fig, self._ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Target position (set once via first frame or manually)
        self._target: np.ndarray | None = None

    def record_frame(
        self,
        state_dict: dict[str, Any],
        target_pos: np.ndarray | None = None,
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
        """
        pos = np.asarray(state_dict["pos"], dtype=np.float64)
        quat = np.asarray(state_dict["quat"], dtype=np.float64)

        if target_pos is not None:
            self._target = np.asarray(target_pos, dtype=np.float64)
        elif "target_pos" in state_dict and self._target is None:
            self._target = np.asarray(state_dict["target_pos"], dtype=np.float64)

        # Append to trail
        self._trail.append(pos.copy())
        if len(self._trail) > self._trail_length:
            self._trail = self._trail[-self._trail_length :]

        self._render_frame(pos, quat)

    def _render_frame(self, pos: np.ndarray, quat: np.ndarray) -> None:
        ax = self._ax
        ax.cla()

        ix, iy = self._ix, self._iy

        # Axis labels
        labels = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}
        ax.set_xlabel(labels[ix])
        ax.set_ylabel(labels[iy])
        ax.set_title(f"BlueROV2 — {self._view} view" + (f"  (z={pos[2]:.2f}m)" if self._view == "top" else ""))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Trajectory trail (fade alpha)
        if len(self._trail) > 1:
            trail = np.array(self._trail)
            n = len(trail)
            # Draw segments with increasing alpha
            for i in range(max(0, n - 100), n - 1):
                alpha = 0.1 + 0.9 * (i - max(0, n - 100)) / min(n, 100)
                ax.plot(
                    trail[i : i + 2, ix],
                    trail[i : i + 2, iy],
                    color="steelblue",
                    alpha=alpha,
                    linewidth=1.5,
                )

        # Target point
        if self._target is not None:
            ax.plot(
                self._target[ix], self._target[iy],
                marker="x", color="red", markersize=12, markeredgewidth=2,
                label="target",
            )

        # Vehicle position
        ax.plot(
            pos[ix], pos[iy],
            marker="o", color="navy", markersize=8,
            label="ROV",
        )

        # Orientation arrow (yaw projection onto this plane)
        yaw = _quat_yaw(quat)
        arrow_len = 0.3
        if self._view == "top":
            dx = arrow_len * np.cos(yaw)
            dy = arrow_len * np.sin(yaw)
        else:
            # Side view: show forward component from yaw
            dx = arrow_len * np.cos(yaw)
            dy = 0.0  # no pitch info in simple side view
        ax.annotate(
            "",
            xy=(pos[ix] + dx, pos[iy] + dy),
            xytext=(pos[ix], pos[iy]),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=2),
        )

        # Auto-scale with padding
        margin = 1.5
        if len(self._trail) > 0:
            trail = np.array(self._trail)
            all_x = np.concatenate([trail[:, ix], [pos[ix]]])
            all_y = np.concatenate([trail[:, iy], [pos[iy]]])
            if self._target is not None:
                all_x = np.append(all_x, self._target[ix])
                all_y = np.append(all_y, self._target[iy])
            x_center = np.mean(all_x)
            y_center = np.mean(all_y)
            span = max(
                np.ptp(all_x) + 2 * margin,
                np.ptp(all_y) + 2 * margin,
                2.0,
            )
            ax.set_xlim(x_center - span / 2, x_center + span / 2)
            ax.set_ylim(y_center - span / 2, y_center + span / 2)

        ax.legend(loc="upper right", fontsize=8)

        self._fig.canvas.draw()

        # Extract RGBA pixel buffer and convert to RGB
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        self._frames.append(buf[:, :, :3].copy())

    def close(self) -> Path:
        """Flush buffered frames to MP4 and return output path.

        Returns
        -------
        Path
            Path to the written MP4 file.
        """
        if not self._frames:
            raise RuntimeError("No frames recorded — call record_frame() first")

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        iio.imwrite(
            str(self._output_path),
            self._frames,
            fps=self._fps,
            codec="libx264",
            pixelformat="yuv420p",
        )

        plt.close(self._fig)

        n_frames = len(self._frames)
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
