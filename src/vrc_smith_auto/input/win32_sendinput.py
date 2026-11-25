from __future__ import annotations

import ctypes
import logging
import math
import sys
import threading
import time
from ctypes import wintypes
from typing import Any, Callable

from vrc_smith_auto.window import (
    WindowRect,
    get_client_rect,
    is_current_process_elevated,
    is_foreground_window,
    is_target_elevated,
)

logger = logging.getLogger(__name__)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

TAB_SCAN_CODE = 0x0F
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

ERROR_ACCESS_DENIED = 5
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_u", _INPUT_UNION),
    ]


def build_mouse_left_input(flags: int) -> INPUT:
    """Build an in-place Win32 mouse INPUT struct (no MOUSEEVENTF_MOVE or MOUSEEVENTF_ABSOLUTE)."""
    return INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=int(flags),
            time=0,
            dwExtraInfo=0,
        ),
    )


def _validate_duration_ms(duration_ms: float) -> float:
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        raise ValueError(f"duration_ms must be a finite number > 0, got {duration_ms!r}")
    val = float(duration_ms)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"duration_ms must be > 0 and finite, got {duration_ms!r}")
    return val


def _validate_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return val


_USER32_SENDINPUT: Any = None


def _get_win32_send_input() -> Callable[[INPUT], int]:
    global _USER32_SENDINPUT
    if sys.platform != "win32":
        raise RuntimeError("Win32 SendInput is only supported on Windows")
    if _USER32_SENDINPUT is None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        _USER32_SENDINPUT = user32.SendInput

    send_input = _USER32_SENDINPUT

    def _call(inp: INPUT) -> int:
        ctypes.set_last_error(0)
        return int(send_input(1, ctypes.byref(inp), ctypes.sizeof(INPUT)))

    return _call


class Win32SendInputMouse:
    """Non-blocking, in-place Win32 SendInput left mouse click driver."""

    def __init__(
        self,
        *,
        require_foreground: bool = True,
        target_hwnd: int | None = None,
        hwnd_provider: Callable[[], int | None] | None = None,
        check_uipi: bool = True,
        send_input_fn: Callable[[INPUT], int] | None = None,
        get_last_error_fn: Callable[[], int] | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        is_foreground_fn: Callable[[int], bool] = is_foreground_window,
        is_target_elevated_fn: Callable[[int], bool | None] = is_target_elevated,
        is_current_elevated_fn: Callable[[], bool | None] = is_current_process_elevated,
    ) -> None:
        self._require_foreground = bool(require_foreground)
        self._target_hwnd = target_hwnd
        self._hwnd_provider = hwnd_provider
        self._check_uipi = bool(check_uipi)
        self._send_input_fn = send_input_fn
        self._get_last_error_fn = get_last_error_fn or ctypes.get_last_error
        self._clock_fn = clock_fn
        self._is_foreground_fn = is_foreground_fn
        self._is_target_elevated_fn = is_target_elevated_fn
        self._is_current_elevated_fn = is_current_elevated_fn

        self._lock = threading.RLock()
        self._pressed = False
        self._release_at_s: float | None = None
        self._last_error: int | None = None
        self._last_blocked_reason: str | None = None
        self._uipi_blocked = False

    @property
    def is_pressed(self) -> bool:
        with self._lock:
            return self._pressed

    @property
    def last_error(self) -> int | None:
        with self._lock:
            return self._last_error

    @property
    def last_blocked_reason(self) -> str | None:
        with self._lock:
            return self._last_blocked_reason

    @property
    def uipi_blocked(self) -> bool:
        with self._lock:
            return self._uipi_blocked

    def set_target_hwnd(self, hwnd: int | None) -> None:
        with self._lock:
            self._target_hwnd = hwnd

    def _resolve_hwnd(self) -> int | None:
        if self._target_hwnd is not None:
            return self._target_hwnd
        if self._hwnd_provider is not None:
            return self._hwnd_provider()
        return None

    def _send(self, inp: INPUT) -> int:
        if self._send_input_fn is None:
            self._send_input_fn = _get_win32_send_input()
        return int(self._send_input_fn(inp))

    def _allows_input(self, hwnd: int | None) -> bool:
        if self._require_foreground:
            if hwnd is None or hwnd <= 0:
                self._last_blocked_reason = "no_target_window"
                return False
            if not self._is_foreground_fn(hwnd):
                self._last_blocked_reason = "not_foreground"
                return False

        if self._check_uipi and hwnd is not None and hwnd > 0:
            target_elevated = self._is_target_elevated_fn(hwnd)
            current_elevated = self._is_current_elevated_fn()
            if target_elevated is True and current_elevated is False:
                self._uipi_blocked = True
                self._last_blocked_reason = "uipi_elevation_mismatch"
                logger.warning(
                    "UIPI elevation mismatch: target window (hwnd=%s) is running elevated "
                    "while this tool is not. Run this tool as Administrator or run VRChat un-elevated.",
                    hwnd,
                )
                return False
        return True

    def client_rect(self) -> WindowRect | None:
        hwnd = self._resolve_hwnd()
        return get_client_rect(hwnd) if hwnd else None

    def _allows_ui_input(self, rect: WindowRect | None) -> bool:
        self._last_blocked_reason = None
        hwnd = self._resolve_hwnd()
        current = get_client_rect(hwnd) if hwnd else None
        if current is None:
            self._last_blocked_reason = "no_target_window"
            return False
        if rect != current:
            self._last_blocked_reason = "window_moved_or_resized"
            return False
        return self._allows_input(hwnd)

    def move_cursor(self, rect: WindowRect | None, x: int, y: int) -> bool:
        if rect is None or not (0 <= x < rect.width and 0 <= y < rect.height):
            return False
        with self._lock:
            if not self._allows_ui_input(rect):
                return False
            if not ctypes.windll.user32.SetCursorPos(rect.left + x, rect.top + y):
                self._last_blocked_reason = "set_cursor_pos_failed"
                return False
            return True

    def press_tab(self, rect: WindowRect | None) -> bool:
        with self._lock:
            if not self._allows_ui_input(rect):
                return False
            down = INPUT(
                type=INPUT_KEYBOARD, ki=KEYBDINPUT(wScan=TAB_SCAN_CODE, dwFlags=KEYEVENTF_SCANCODE)
            )
            up = INPUT(
                type=INPUT_KEYBOARD,
                ki=KEYBDINPUT(wScan=TAB_SCAN_CODE, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP),
            )
            if self._send(down) != 1:
                self._last_blocked_reason = f"tab_key_down_failed_{self._get_last_error_fn()}"
                return False
            try:
                time.sleep(0.10)
            finally:
                released = self._send(up) == 1
            if not released:
                raise RuntimeError("Failed to release Tab after opening or closing Smithy")
            return True

    def move_relative(self, rect: WindowRect, dx: int, dy: int) -> bool:
        """Send ordinary mouse motion for applications that render their own cursor."""
        with self._lock:
            if not self._allows_ui_input(rect):
                return False
            event = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=dx, dy=dy, dwFlags=MOUSEEVENTF_MOVE))
            if self._send(event) == 1:
                return True
            self._last_blocked_reason = f"mouse_move_failed_{self._get_last_error_fn()}"
            return False

    def press_left(self, duration_ms: float, timestamp_s: float | None = None) -> bool:
        dur_ms = _validate_duration_ms(duration_ms)
        ts = _validate_finite_float(timestamp_s, "timestamp_s") if timestamp_s is not None else None

        with self._lock:
            self._last_blocked_reason = None

            if self._pressed:
                self._last_blocked_reason = "already_pressed"
                return False

            if not self._allows_input(self._resolve_hwnd()):
                return False

            inp = build_mouse_left_input(MOUSEEVENTF_LEFTDOWN)
            sent = self._send(inp)
            if sent != 1:
                err = int(self._get_last_error_fn())
                self._last_error = err
                if err == ERROR_ACCESS_DENIED:
                    self._uipi_blocked = True
                    self._last_blocked_reason = "uipi_access_denied"
                    logger.warning(
                        "SendInput(LEFTDOWN) failed with ERROR_ACCESS_DENIED (5) — likely blocked by UIPI. "
                        "Try running as Administrator."
                    )
                else:
                    self._last_blocked_reason = f"send_input_failed_{err}"
                    logger.warning("SendInput(LEFTDOWN) returned %s (winerror=%s)", sent, err)
                return False

            now = ts if ts is not None else float(self._clock_fn())
            self._uipi_blocked = False
            self._pressed = True
            self._release_at_s = now + dur_ms / 1000.0
            return True

    def _release_left(self) -> None:
        try:
            inp = build_mouse_left_input(MOUSEEVENTF_LEFTUP)
            sent = self._send(inp)
            if sent != 1:
                err = int(self._get_last_error_fn())
                self._last_error = err
                if err == ERROR_ACCESS_DENIED:
                    self._uipi_blocked = True
                logger.warning("SendInput(LEFTUP) returned %s (winerror=%s)", sent, err)
        finally:
            self._pressed = False
            self._release_at_s = None

    def tick(self, now_s: float) -> None:
        now = _validate_finite_float(now_s, "now_s")
        with self._lock:
            if self._pressed and self._release_at_s is not None and now >= self._release_at_s:
                self._release_left()

    def release_all(self) -> None:
        with self._lock:
            if self._pressed:
                self._release_left()

    def __enter__(self) -> Win32SendInputMouse:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release_all()
