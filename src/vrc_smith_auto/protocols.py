from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from vrc_smith_auto.window import WindowRect


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Single captured BGR frame with timestamp and sequential index."""

    image_bgr: np.ndarray
    timestamp_s: float
    frame_idx: int
    screen_rect: WindowRect | None = None

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])


@runtime_checkable
class FrameSource(Protocol):
    """Abstract frame provider (live DXGI/GDI capture or offline video/image)."""

    def read(self) -> FramePacket | None:
        """Return the next FramePacket, or None on idle/no-new-frame or EOF."""
        ...

    def close(self) -> None:
        """Release underlying capture handles or video resources."""
        ...


@runtime_checkable
class MouseBackend(Protocol):
    """Abstract in-place mouse click driver (Win32 SendInput or test spy)."""

    @property
    def is_pressed(self) -> bool:
        """True while the left mouse button is currently held down."""
        ...

    def press_left(
        self,
        duration_ms: float,
        timestamp_s: float | None = None,
    ) -> bool:
        """Initiate a non-blocking left press scheduled to release after duration_ms."""
        ...

    def tick(self, now_s: float) -> None:
        """Advance scheduled release state; release button if now_s >= release_at."""
        ...

    def release_all(self) -> None:
        """Immediately release any held mouse button (idempotent, safe in finally)."""
        ...


@runtime_checkable
class SmithyInput(MouseBackend, Protocol):
    def move_cursor(self, rect: WindowRect | None, x: int, y: int) -> bool: ...

    def press_tab(self, rect: WindowRect | None) -> bool: ...
