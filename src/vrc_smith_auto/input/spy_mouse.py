from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from vrc_smith_auto.input.win32_sendinput import (
    _validate_duration_ms,
    _validate_finite_float,
)
from vrc_smith_auto.window import WindowRect


@dataclass(frozen=True, slots=True)
class RecordedClick:
    """Recorded click event emitted by SpyMouseBackend for dry-run and offline tests."""

    timestamp_s: float
    duration_ms: float
    release_timestamp_s: float | None = None
    released: bool = False
    frame_idx: int | None = None


class SpyMouseBackend:
    """In-memory MouseBackend test spy and --dry-run driver."""

    def __init__(self, clock_fn: Callable[[], float] = time.perf_counter) -> None:
        self._clock_fn = clock_fn
        self._pressed = False
        self._release_at_s: float | None = None
        self._last_now_s: float | None = None
        self._current_frame_idx: int | None = None
        self._clicks: list[RecordedClick] = []

    @property
    def is_pressed(self) -> bool:
        return self._pressed

    @property
    def clicks(self) -> tuple[RecordedClick, ...]:
        return tuple(self._clicks)

    @property
    def click_count(self) -> int:
        return len(self._clicks)

    def set_frame_idx(self, frame_idx: int | None) -> None:
        self._current_frame_idx = frame_idx

    def move_cursor(self, rect: WindowRect | None, x: int, y: int) -> bool:
        return True

    def press_tab(self, rect: WindowRect | None) -> bool:
        return True

    def press_left(self, duration_ms: float, timestamp_s: float | None = None) -> bool:
        dur_ms = _validate_duration_ms(duration_ms)
        ts = _validate_finite_float(timestamp_s, "timestamp_s") if timestamp_s is not None else None
        if self._pressed:
            return False

        now = ts if ts is not None else float(self._clock_fn())
        self._pressed = True
        self._release_at_s = now + dur_ms / 1000.0
        self._last_now_s = now if self._last_now_s is None else max(self._last_now_s, now)
        self._clicks.append(
            RecordedClick(
                timestamp_s=now,
                duration_ms=dur_ms,
                frame_idx=self._current_frame_idx,
            )
        )
        return True

    def tick(self, now_s: float) -> None:
        now = _validate_finite_float(now_s, "now_s")
        self._last_now_s = now if self._last_now_s is None else max(self._last_now_s, now)
        if self._pressed and self._release_at_s is not None and now >= self._release_at_s:
            if self._clicks:
                self._clicks[-1] = replace(
                    self._clicks[-1],
                    release_timestamp_s=now,
                    released=True,
                )
            self._pressed = False
            self._release_at_s = None

    def release_all(self) -> None:
        if not self._pressed:
            return
        release_ts = (
            self._last_now_s if self._last_now_s is not None else float(self._clock_fn())
        )
        if self._clicks:
            self._clicks[-1] = replace(
                self._clicks[-1],
                release_timestamp_s=release_ts,
                released=True,
            )
        self._pressed = False
        self._release_at_s = None

    def clear(self) -> None:
        self._clicks.clear()
        self._pressed = False
        self._release_at_s = None
        self._last_now_s = None

    def __enter__(self) -> SpyMouseBackend:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release_all()
