from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
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

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0
MONITORINFOF_PRIMARY = 0x00000001
SM_CXSCREEN = 0
SM_CYSCREEN = 1
HGDI_ERROR = ctypes.c_void_p(-1).value


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wintypes.BYTE),
        ("rgbGreen", wintypes.BYTE),
        ("rgbRed", wintypes.BYTE),
        ("rgbReserved", wintypes.BYTE),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", _RGBQUAD * 1),
    ]


_MONITORENUMPROC = (
    ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(_RECT),
        wintypes.LPARAM,
    )
    if sys.platform == "win32"
    else None
)


class _GdiApi:
    """Lazy-initialized Win32 user32/gdi32 ctypes bindings with Win64-safe signatures."""

    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._configure()

    def _configure(self) -> None:
        u = self.user32
        u.GetDC.argtypes = [wintypes.HWND]
        u.GetDC.restype = wintypes.HDC
        u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        u.ReleaseDC.restype = ctypes.c_int
        u.EnumDisplayMonitors.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(_RECT),
            ctypes.c_void_p,
            wintypes.LPARAM,
        ]
        u.EnumDisplayMonitors.restype = wintypes.BOOL
        u.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFOEXW)]
        u.GetMonitorInfoW.restype = wintypes.BOOL
        u.GetSystemMetrics.argtypes = [ctypes.c_int]
        u.GetSystemMetrics.restype = ctypes.c_int

        g = self.gdi32
        g.CreateCompatibleDC.argtypes = [wintypes.HDC]
        g.CreateCompatibleDC.restype = wintypes.HDC
        g.DeleteDC.argtypes = [wintypes.HDC]
        g.DeleteDC.restype = wintypes.BOOL
        g.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        g.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        g.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        g.SelectObject.restype = wintypes.HGDIOBJ
        g.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        g.DeleteObject.restype = wintypes.BOOL
        g.BitBlt.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.DWORD,
        ]
        g.BitBlt.restype = wintypes.BOOL
        g.GetDIBits.argtypes = [
            wintypes.HDC,
            wintypes.HBITMAP,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.c_void_p,
            ctypes.POINTER(_BITMAPINFO),
            wintypes.UINT,
        ]
        g.GetDIBits.restype = ctypes.c_int

    def enum_monitors(self) -> list[tuple[WindowRect, bool]]:
        monitors: list[tuple[WindowRect, bool]] = []
        cb_type = _MONITORENUMPROC or ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(_RECT),
            wintypes.LPARAM,
        )

        def _on_monitor(
            hmon: Any,
            _hdc: Any,
            _lprc: Any,
            _lparam: int,
        ) -> bool:
            mi = _MONITORINFOEXW()
            mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
            if self.user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                rect = WindowRect(
                    left=int(mi.rcMonitor.left),
                    top=int(mi.rcMonitor.top),
                    right=int(mi.rcMonitor.right),
                    bottom=int(mi.rcMonitor.bottom),
                )
                is_primary = bool(mi.dwFlags & MONITORINFOF_PRIMARY)
                if rect.is_valid:
                    monitors.append((rect, is_primary))
            return True

        cb = cb_type(_on_monitor)
        self.user32.EnumDisplayMonitors(None, None, cb, 0)
        return monitors


_GDI_API: _GdiApi | None = None


def _get_gdi_api() -> _GdiApi:
    global _GDI_API
    if _GDI_API is None:
        _GDI_API = _GdiApi()
    return _GDI_API


def _resolve_gdi_monitor(api: Any, monitor_idx: int) -> WindowRect:
    monitors: list[tuple[WindowRect, bool]] = []
    if hasattr(api, "enum_monitors"):
        monitors = list(api.enum_monitors())

    if monitors:
        ordered = [rect for rect, is_primary in monitors if is_primary] + [
            rect for rect, is_primary in monitors if not is_primary
        ]
        if monitor_idx >= len(ordered):
            raise CaptureError(
                f"monitor_idx {monitor_idx} out of range (found {len(ordered)} GDI monitors)"
            )
        return ordered[monitor_idx]

    if monitor_idx > 0:
        raise CaptureError(f"monitor_idx {monitor_idx} out of range (no monitors enumerated)")

    width = int(api.user32.GetSystemMetrics(SM_CXSCREEN))
    height = int(api.user32.GetSystemMetrics(SM_CYSCREEN))
    if width <= 0 or height <= 0:
        raise CaptureError(f"Invalid primary screen metrics: {width}x{height}")
    return WindowRect.from_xywh(0, 0, width, height)


class GdiBitBltCapture:
    """Zero-external-dependency Win32 GDI BitBlt + GetDIBits capture fallback."""

    def __init__(
        self,
        *,
        monitor_idx: int = 0,
        crop_to_window: bool = True,
        window_config: WindowConfig | None = None,
        region: WindowRect | tuple[int, int, int, int] | None = None,
        api: Any | None = None,
        get_last_error_fn: Callable[[], int] | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        find_window_fn: Callable[..., int | None] = find_vrchat_window,
        get_client_rect_fn: Callable[[int], WindowRect | None] = get_client_rect,
    ) -> None:
        if isinstance(monitor_idx, bool) or not isinstance(monitor_idx, int) or monitor_idx < 0:
            raise ValueError(f"monitor_idx must be an integer >= 0, got {monitor_idx!r}")

        if api is None:
            if sys.platform != "win32":
                raise CaptureError("GdiBitBltCapture is only supported on Windows")
            enable_dpi_awareness()
            api = _get_gdi_api()

        self._api = api
        self._get_last_error_fn = get_last_error_fn or ctypes.get_last_error
        self._clock_fn = clock_fn
        self._closed = False
        self._frame_idx = 0

        self._src_dc: Any = None
        self._mem_dc: Any = None
        self._hbitmap: Any = None
        self._bmp_size: tuple[int, int] | None = None
        self._bgra_buf: np.ndarray | None = None
        self._bmi: _BITMAPINFO | None = None

        try:
            self._monitor_rect = _resolve_gdi_monitor(api, monitor_idx)
            self._region_tracker = WindowRegionTracker(
                window_config=window_config,
                crop_to_window=crop_to_window,
                static_region=region,
                clock_fn=clock_fn,
                find_window_fn=find_window_fn,
                get_client_rect_fn=get_client_rect_fn,
            )
            src_dc = api.user32.GetDC(None)
            if not src_dc:
                err = self._get_last_error_fn()
                raise CaptureError(f"user32.GetDC(None) failed (winerror={err})")
            self._src_dc = src_dc

            mem_dc = api.gdi32.CreateCompatibleDC(src_dc)
            if not mem_dc:
                err = self._get_last_error_fn()
                raise CaptureError(f"gdi32.CreateCompatibleDC failed (winerror={err})")
            self._mem_dc = mem_dc
        except Exception:
            self.close()
            raise

    @property
    def monitor_rect(self) -> WindowRect:
        return self._monitor_rect

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def _ensure_bitmap(self, width: int, height: int) -> None:
        if self._hbitmap is not None and self._bmp_size == (width, height):
            return

        if self._hbitmap is not None:
            old_hbmp = self._hbitmap
            self._hbitmap = None
            self._bmp_size = None
            self._api.gdi32.DeleteObject(old_hbmp)

        hbmp = self._api.gdi32.CreateCompatibleBitmap(self._src_dc, width, height)
        if not hbmp:
            err = self._get_last_error_fn()
            raise CaptureError(f"gdi32.CreateCompatibleBitmap({width}x{height}) failed (winerror={err})")

        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = int(width)
        bmi.bmiHeader.biHeight = -int(height)  # Top-down DIB orientation
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        bmi.bmiHeader.biSizeImage = int(width * height * 4)

        self._hbitmap = hbmp
        self._bmp_size = (width, height)
        self._bgra_buf = np.empty((height, width, 4), dtype=np.uint8)
        self._bmi = bmi

    def read(self) -> FramePacket | None:
        if self._closed or self._src_dc is None or self._mem_dc is None:
            return None

        target = self._region_tracker.resolve_screen_rect(self._monitor_rect)
        width, height = target.width, target.height
        if width <= 0 or height <= 0:
            raise CaptureError(f"Invalid GDI target rectangle: {target}")

        self._ensure_bitmap(width, height)
        assert self._hbitmap is not None
        assert self._bgra_buf is not None
        assert self._bmi is not None

        old_obj = self._api.gdi32.SelectObject(self._mem_dc, self._hbitmap)
        if not old_obj or old_obj == HGDI_ERROR:
            err = self._get_last_error_fn()
            raise CaptureError(f"gdi32.SelectObject failed (winerror={err})")

        try:
            if sys.platform == "win32":
                ctypes.set_last_error(0)
            ok = self._api.gdi32.BitBlt(
                self._mem_dc,
                0,
                0,
                width,
                height,
                self._src_dc,
                target.left,
                target.top,
                SRCCOPY | CAPTUREBLT,
            )
            if not ok:
                err = self._get_last_error_fn()
                self._region_tracker.invalidate()
                raise CaptureError(f"gdi32.BitBlt failed (winerror={err})")
        finally:
            self._api.gdi32.SelectObject(self._mem_dc, old_obj)

        if sys.platform == "win32":
            ctypes.set_last_error(0)
        scanlines = int(
            self._api.gdi32.GetDIBits(
                self._src_dc,
                self._hbitmap,
                0,
                height,
                self._bgra_buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self._bmi),
                DIB_RGB_COLORS,
            )
        )
        if scanlines <= 0 or scanlines != height:
            err = self._get_last_error_fn()
            raise CaptureError(
                f"gdi32.GetDIBits failed (scanlines={scanlines}, expected={height}, winerror={err})"
            )

        image_bgr = normalize_bgr(self._bgra_buf)
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

        hbmp = self._hbitmap
        mem_dc = self._mem_dc
        src_dc = self._src_dc
        self._hbitmap = None
        self._bmp_size = None
        self._bgra_buf = None
        self._bmi = None
        self._mem_dc = None
        self._src_dc = None

        try:
            if hbmp is not None:
                self._api.gdi32.DeleteObject(hbmp)
        finally:
            try:
                if mem_dc is not None:
                    self._api.gdi32.DeleteDC(mem_dc)
            finally:
                if src_dc is not None:
                    self._api.user32.ReleaseDC(None, src_dc)

    def __enter__(self) -> GdiBitBltCapture:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
