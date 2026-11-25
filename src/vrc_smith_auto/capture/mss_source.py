from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np

from vrc_smith_auto.capture.base import CaptureError, WindowRegionTracker, normalize_bgr
from vrc_smith_auto.config import WindowConfig
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.window import (
    WindowRect,
    enable_dpi_awareness,
    find_vrchat_window,
    get_client_rect,
)


def _is_screenshot_error(exc: BaseException) -> bool:
    return any(cls.__name__ == "ScreenShotError" for cls in type(exc).__mro__)


def _select_mss_monitor(sct: Any, monitor_idx: int) -> WindowRect:
    monitors = list(getattr(sct, "monitors", ()))
    # monitors[0] is the combined virtual desktop bounding box; monitors[1:] are physical displays.
    if len(monitors) <= 1:
        raise CaptureError("No physical monitors reported by mss")

    phys = monitors[1:]
    if any(isinstance(m, dict) and "is_primary" in m for m in phys):
        primaries = [m for m in phys if isinstance(m, dict) and bool(m.get("is_primary"))]
        others = [m for m in phys if not (isinstance(m, dict) and bool(m.get("is_primary")))]
        ordered = primaries + others
    else:
        primary: Any = None
        try:
            primary = getattr(sct, "primary_monitor", None)
        except Exception:
            primary = None

        match_idx: int | None = None
        if isinstance(primary, dict):
            for i, m in enumerate(phys):
                if m is primary:
                    match_idx = i
                    break
            if match_idx is None:
                for i, m in enumerate(phys):
                    if m == primary:
                        match_idx = i
                        break

        if match_idx is not None:
            ordered = [phys[match_idx]] + [m for i, m in enumerate(phys) if i != match_idx]
        else:
            ordered = list(phys)

    if monitor_idx < 0 or monitor_idx >= len(ordered):
        raise CaptureError(
            f"monitor_idx {monitor_idx} out of range (mss reported {len(ordered)} physical monitors)"
        )
    mon = ordered[monitor_idx]

    try:
        rect = WindowRect.from_xywh(
            left=int(mon["left"]),
            top=int(mon["top"]),
            width=int(mon["width"]),
            height=int(mon["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureError(f"Invalid mss monitor metadata {mon!r}: {exc}") from exc

    if not rect.is_valid:
        raise CaptureError(f"mss monitor reported non-positive dimensions: {rect}")
    return rect


class MssCapture:
    """Cross-process GDI screen capture source backed by mss."""

    def __init__(
        self,
        *,
        monitor_idx: int = 0,
        crop_to_window: bool = True,
        window_config: WindowConfig | None = None,
        region: WindowRect | tuple[int, int, int, int] | None = None,
        mss_factory: Callable[[], Any] | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        find_window_fn: Callable[..., int | None] = find_vrchat_window,
        get_client_rect_fn: Callable[[int], WindowRect | None] = get_client_rect,
    ) -> None:
        if isinstance(monitor_idx, bool) or not isinstance(monitor_idx, int) or monitor_idx < 0:
            raise ValueError(f"monitor_idx must be an integer >= 0, got {monitor_idx!r}")

        if mss_factory is None:
            if sys.platform == "win32":
                enable_dpi_awareness()
            try:
                import mss  # noqa: PLC0415
            except Exception as exc:
                raise CaptureError(f"Failed to import mss: {exc}") from exc
            factory = getattr(mss, "MSS", None) or mss.mss
            try:
                sct = factory()
            except Exception as exc:
                raise CaptureError(f"Failed to initialize mss.MSS(): {exc}") from exc
        else:
            try:
                sct = mss_factory()
            except Exception as exc:
                raise CaptureError(f"Failed to initialize mss factory: {exc}") from exc

        self._sct = sct
        self._clock_fn = clock_fn
        self._closed = False
        self._frame_idx = 0

        try:
            self._monitor_rect = _select_mss_monitor(sct, monitor_idx)
            self._region_tracker = WindowRegionTracker(
                window_config=window_config,
                crop_to_window=crop_to_window,
                static_region=region,
                clock_fn=clock_fn,
                find_window_fn=find_window_fn,
                get_client_rect_fn=get_client_rect_fn,
            )
        except Exception:
            self.close()
            raise

    @property
    def monitor_rect(self) -> WindowRect:
        return self._monitor_rect

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def read(self) -> FramePacket | None:
        if self._closed or self._sct is None:
            return None

        target = self._region_tracker.resolve_screen_rect(self._monitor_rect)
        try:
            shot = self._sct.grab(target.as_mss_monitor())
        except Exception as exc:
            self._region_tracker.invalidate()
            if _is_screenshot_error(exc) or isinstance(exc, (OSError, RuntimeError, ValueError)):
                raise CaptureError(f"mss grab failed: {exc}") from exc
            raise

        image_bgr = normalize_bgr(np.asarray(shot))
        idx = self._frame_idx
        self._frame_idx += 1
        return FramePacket(
            image_bgr=image_bgr,
            timestamp_s=float(self._clock_fn()),
            frame_idx=idx,
            screen_rect=target,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sct = self._sct
        self._sct = None
        if sct is not None and hasattr(sct, "close"):
            sct.close()

    def __enter__(self) -> MssCapture:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
