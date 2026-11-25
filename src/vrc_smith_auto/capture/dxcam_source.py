from __future__ import annotations

from collections.abc import Mapping
import sys
import time
from typing import Any, Callable

from vrc_smith_auto.capture.base import CaptureError, WindowRegionTracker, normalize_bgr
from vrc_smith_auto.config import WindowConfig
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.window import (
    WindowRect,
    enable_dpi_awareness,
    find_vrchat_window,
    get_client_rect,
)


def _is_com_error(exc: BaseException) -> bool:
    return any(cls.__name__ == "COMError" for cls in type(exc).__mro__)


def _extract_output_desktop_rect(cam: Any) -> WindowRect:
    output = getattr(cam, "_output", None)
    desc = getattr(output, "desc", None)
    coords = getattr(desc, "DesktopCoordinates", None)
    if coords is not None:
        try:
            rect = WindowRect(
                left=int(coords.left),
                top=int(coords.top),
                right=int(coords.right),
                bottom=int(coords.bottom),
            )
            if rect.is_valid:
                return rect
        except (AttributeError, TypeError, ValueError):
            pass

    width = int(getattr(cam, "width", 0))
    height = int(getattr(cam, "height", 0))
    if width <= 0 or height <= 0:
        raise CaptureError(f"dxcam camera reported invalid dimensions: {width}x{height}")
    return WindowRect.from_xywh(0, 0, width, height)


def _is_output_primary(out_obj: Any, output_metadata: Any) -> bool:
    if isinstance(output_metadata, Mapping):
        devicename = getattr(out_obj, "devicename", None)
        if devicename is not None:
            meta = output_metadata.get(devicename)
            if isinstance(meta, (list, tuple)) and len(meta) > 1:
                return bool(meta[1])
    return bool(getattr(out_obj, "is_primary", False))


def _resolve_dxcam_factory(dxcam_mod: Any) -> Any:
    factory = getattr(dxcam_mod, "__factory", None) or getattr(dxcam_mod, "_factory", None)
    if factory is None:
        dx_factory_cls = getattr(dxcam_mod, "DXFactory", None)
        if callable(dx_factory_cls):
            try:
                factory = dx_factory_cls()
            except Exception:
                factory = None
    return factory


def _select_dxcam_candidates(
    dxcam_mod: Any,
    monitor_idx: int,
    device_idx: int | None,
) -> list[tuple[int, int | None]]:
    factory = _resolve_dxcam_factory(dxcam_mod)
    outputs = getattr(factory, "outputs", None)
    output_metadata = getattr(factory, "output_metadata", None)

    if isinstance(outputs, (list, tuple)):
        if device_idx is not None:
            if device_idx < 0 or device_idx >= len(outputs):
                raise CaptureError(
                    f"dxcam device_idx {device_idx} out of range (found {len(outputs)} devices)"
                )
            dev_outputs = outputs[device_idx]
            if monitor_idx >= len(dev_outputs):
                raise CaptureError(
                    f"monitor_idx {monitor_idx} out of range for device {device_idx} "
                    f"(found {len(dev_outputs)} outputs)"
                )
            primaries = [
                (device_idx, o_idx)
                for o_idx, out_obj in enumerate(dev_outputs)
                if _is_output_primary(out_obj, output_metadata)
            ]
            others = [
                (device_idx, o_idx)
                for o_idx, out_obj in enumerate(dev_outputs)
                if not _is_output_primary(out_obj, output_metadata)
            ]
            ordered = primaries + others
            if monitor_idx == 0:
                return ordered
            return [ordered[monitor_idx]]

        flattened: list[tuple[int, int, Any]] = [
            (d_idx, o_idx, out_obj)
            for d_idx, dev_outs in enumerate(outputs)
            for o_idx, out_obj in enumerate(dev_outs)
        ]
        if not flattened:
            raise CaptureError("No DXGI outputs available for dxcam")

        primaries = [
            (d_idx, o_idx)
            for d_idx, o_idx, out_obj in flattened
            if _is_output_primary(out_obj, output_metadata)
        ]
        others = [
            (d_idx, o_idx)
            for d_idx, o_idx, out_obj in flattened
            if not _is_output_primary(out_obj, output_metadata)
        ]
        ordered = primaries + others

        if monitor_idx == 0:
            return ordered

        if monitor_idx >= len(ordered):
            raise CaptureError(
                f"monitor_idx {monitor_idx} out of range (found {len(ordered)} DXGI outputs)"
            )
        return [ordered[monitor_idx]]

    # Fallback when dxcam_module is a minimal test double without __factory.outputs.
    d_idx = 0 if device_idx is None else device_idx
    o_idx = None if monitor_idx == 0 else monitor_idx
    return [(d_idx, o_idx)]


class DxcamCapture:
    """Low-latency DXGI Desktop Duplication capture source backed by dxcam."""

    def __init__(
        self,
        *,
        monitor_idx: int = 0,
        crop_to_window: bool = True,
        window_config: WindowConfig | None = None,
        region: WindowRect | tuple[int, int, int, int] | None = None,
        new_frame_only: bool = True,
        backend: str = "dxgi",
        device_idx: int | None = None,
        dxcam_module: Any | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        find_window_fn: Callable[..., int | None] = find_vrchat_window,
        get_client_rect_fn: Callable[[int], WindowRect | None] = get_client_rect,
    ) -> None:
        if isinstance(monitor_idx, bool) or not isinstance(monitor_idx, int) or monitor_idx < 0:
            raise ValueError(f"monitor_idx must be an integer >= 0, got {monitor_idx!r}")

        if dxcam_module is None:
            if sys.platform != "win32":
                raise CaptureError("DxcamCapture is only supported on Windows")
            # Ensure Per-Monitor-V2 DPI awareness is set before dxcam's import-time SetProcessDpiAwareness(2).
            enable_dpi_awareness()
            try:
                import dxcam as _dxcam  # noqa: PLC0415
            except Exception as exc:
                raise CaptureError(f"Failed to import dxcam: {exc}") from exc
            dxcam_mod = _dxcam
        else:
            dxcam_mod = dxcam_module

        self._new_frame_only = bool(new_frame_only)
        self._clock_fn = clock_fn
        self._closed = False
        self._frame_idx = 0
        self._cam: Any = None

        candidates = _select_dxcam_candidates(dxcam_mod, monitor_idx, device_idx)
        errors: list[str] = []
        cam: Any = None
        for cand_dev, cand_out in candidates:
            created: Any = None
            try:
                created = dxcam_mod.create(
                    device_idx=cand_dev,
                    output_idx=cand_out,
                    output_color="BGR",
                    backend=backend,
                    processor_backend="cv2",
                )
                if created is None:
                    raise CaptureError(
                        f"dxcam.create(device_idx={cand_dev}, output_idx={cand_out}) returned None"
                    )
                cam = created
                break
            except Exception as exc:
                if created is not None and hasattr(created, "release"):
                    try:
                        created.release()
                    except Exception:
                        pass
                if (
                    _is_com_error(exc)
                    or isinstance(exc, (CaptureError, IndexError, RuntimeError, ValueError, OSError))
                ):
                    errors.append(f"(dev={cand_dev}, out={cand_out}): {exc}")
                    continue
                raise

        if cam is None:
            raise CaptureError("dxcam initialization failed: " + "; ".join(errors))

        self._cam = cam
        try:
            self._monitor_rect = _extract_output_desktop_rect(cam)
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
        if self._closed or self._cam is None:
            return None

        target = self._region_tracker.resolve_screen_rect(self._monitor_rect)
        local_region = target.to_monitor_relative(self._monitor_rect)
        if local_region is None:
            return None

        try:
            frame = self._cam.grab(
                region=local_region,
                copy=True,
                new_frame_only=self._new_frame_only,
            )
        except Exception as exc:
            self._region_tracker.invalidate()
            if (
                _is_com_error(exc)
                or isinstance(exc, (OSError, RuntimeError, ValueError))
            ):
                raise CaptureError(f"dxcam grab failed: {exc}") from exc
            raise

        if frame is None:
            return None

        image_bgr = normalize_bgr(frame)
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
        cam = self._cam
        self._cam = None
        if cam is not None:
            try:
                if getattr(cam, "is_capturing", False) and hasattr(cam, "stop"):
                    cam.stop()
            finally:
                if hasattr(cam, "release"):
                    cam.release()

    def __enter__(self) -> DxcamCapture:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
