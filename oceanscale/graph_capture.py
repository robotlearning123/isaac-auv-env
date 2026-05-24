"""CUDA Graph capture for zero-overhead simulation replay.

Wraps wp.ScopedCapture to capture a fixed GPU execution path.
On replay, kernel launches bypass Python dispatch entirely.

Limitation: the captured graph is a fixed execution path.
Conditional branches, CPU-side logic, and dynamic allocation
cannot vary between replays. Only in-place array writes are
visible across replays (pointer identity is baked in).
"""

from __future__ import annotations

from typing import Any

import warp as wp


class GraphCapture:
    """Capture a repeatable GPU workload as a CUDA graph."""

    def __init__(
        self,
        step_fn: Any,
        *args: Any,
        warmup_steps: int = 3,
        device: str = "cuda:0",
    ) -> None:
        self._graph: Any = None
        self._device = device

        for _ in range(warmup_steps):
            step_fn(*args)
        wp.synchronize()

        with wp.ScopedCapture(device=self._device) as capture:
            step_fn(*args)

        self._graph = capture.graph

    def replay(self) -> None:
        """Replay the captured CUDA graph with zero Python dispatch overhead."""
        if self._graph is None:
            raise RuntimeError("No graph captured")
        wp.capture_launch(self._graph)

    @property
    def is_captured(self) -> bool:
        return self._graph is not None
