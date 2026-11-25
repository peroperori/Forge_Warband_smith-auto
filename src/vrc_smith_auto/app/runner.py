from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from vrc_smith_auto.app.hotkeys import HotkeyController
from vrc_smith_auto.app.preview import PreviewWindow
from vrc_smith_auto.app.smithy import SmithyAutomation
from vrc_smith_auto.capture.offline_source import ImageFrameSource, VideoFileFrameSource
from vrc_smith_auto.config import AppConfig
from vrc_smith_auto.control.state_machine import TriggerDecision, TriggerStateMachine
from vrc_smith_auto.input.spy_mouse import SpyMouseBackend
from vrc_smith_auto.input.win32_sendinput import Win32SendInputMouse
from vrc_smith_auto.protocols import FramePacket, FrameSource, MouseBackend, SmithyInput
from vrc_smith_auto.vision.detector import GaugeDetector
from vrc_smith_auto.vision.models import DetectionResult

logger = logging.getLogger(__name__)


def _sleep_until(deadline: float) -> None:
    """Sleep until a perf_counter deadline, busy-waiting the last ~1.5 ms."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > 0.0015:
            time.sleep(remaining - 0.001)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary returned by Runner.run() / run_pipeline()."""

    frames_processed: int
    detections: int
    clicks: int
    click_frames: tuple[int, ...]
    stopped_by_hotkey: bool
    stopped_by_preview: bool
    elapsed_s: float


class Runner:
    """Real-time / offline execution loop coupling capture, vision, control, mouse, and hotkeys."""

    def __init__(
        self,
        source: FrameSource,
        mouse: MouseBackend,
        config: AppConfig | None = None,
        detector: GaugeDetector | None = None,
        state_machine: TriggerStateMachine | None = None,
        hotkeys: HotkeyController | None = None,
        *,
        preview: PreviewWindow | None = None,
        show_preview: bool = False,
        stop_on_eof: bool | None = None,
        max_frames: int | None = None,
        idle_sleep_s: float = 0.001,
        clock_fn: Callable[[], float] = time.perf_counter,
        on_frame: Callable[[FramePacket, DetectionResult | None, TriggerDecision | None], None] | None = None,
    ) -> None:
        self._source = source
        self._mouse = mouse
        self._config = config or AppConfig()
        self._detector = detector or GaugeDetector(self._config)
        self._state_machine = state_machine or TriggerStateMachine(self._config)
        self._hotkeys = hotkeys
        self._preview = preview
        self._show_preview = bool(show_preview) or (preview is not None)
        if stop_on_eof is None:
            self._stop_on_eof = isinstance(source, (VideoFileFrameSource, ImageFrameSource))
        else:
            self._stop_on_eof = bool(stop_on_eof)
        self._max_frames = max_frames
        self._idle_sleep_s = max(0.0, float(idle_sleep_s))
        self._clock_fn = clock_fn
        self._on_frame = on_frame

    def _release_after_hold(self, now_s: float, hold_ms: float) -> None:
        """Release the button in this iteration instead of a later frame.

        The target game judges on pointer-up.  ``mouse.tick()`` is normally
        polled once per captured frame, so a 30 ms hold would defer the release
        by 2–3 frames and silently consume the narrow-Gold timing budget.  Wait
        out ``hold_ms`` now and release inside the same iteration so the judged
        moment stays aligned with the trigger decision.
        """
        hold_s = max(0.0, float(hold_ms) / 1000.0)
        if hold_s > 0.0:
            _sleep_until(time.perf_counter() + hold_s)
        self._mouse.tick(now_s + hold_s)

    def run(self) -> RunResult:
        frames_processed = 0
        detections = 0
        clicks = 0
        click_frames: list[int] = []
        stopped_by_hotkey = False
        stopped_by_preview = False
        paused = False
        smithy = None
        client_rect = None
        if self._config.auto_fire_up:
            ui_input = self._mouse
            if isinstance(ui_input, Win32SendInputMouse):
                client_rect = ui_input.client_rect
                if self._config.dry_run:
                    ui_input = SpyMouseBackend()
            if not isinstance(ui_input, SmithyInput):
                raise ValueError("auto_fire_up requires cursor and Tab support from the input backend")
            smithy = SmithyAutomation(
                ui_input, self._config.auto_sell,
                use_rendered_cursor=self._config.window.title.casefold() == "vrchat",
            )
        gauge_was_allowed = smithy is None
        gauge_missing_frames = 0
        smithy_probe_after = self._config.trigger.rearm_lost_frames

        last_pkt_ts: float | None = None
        last_pkt_wall: float | None = None
        last_tick_ts: float | None = None
        fps_ema: float | None = None
        prev_frame_wall: float | None = None

        own_preview = self._show_preview and self._preview is None
        preview_win = PreviewWindow() if own_preview else self._preview

        start_wall = float(self._clock_fn())
        next_idle_report = start_wall + 5.0
        reported_gauge = False
        try:
            while True:
                if self._max_frames is not None and frames_processed >= self._max_frames:
                    break

                if self._hotkeys is not None:
                    events = self._hotkeys.poll()
                    if events.emergency_stop:
                        self._mouse.release_all()
                        stopped_by_hotkey = True
                        break
                    if events.toggle_armed:
                        paused = not paused
                        logger.info("Automation %s", "paused" if paused else "resumed")
                        if paused:
                            self._mouse.release_all()
                            self._state_machine.reset()

                pkt = self._source.read()
                if pkt is None:
                    if self._stop_on_eof:
                        break
                    now_wall = float(self._clock_fn())
                    if now_wall >= next_idle_report:
                        logger.warning("Waiting for capture frames (%d processed)", frames_processed)
                        next_idle_report = now_wall + 10.0
                    if last_pkt_ts is not None and last_pkt_wall is not None:
                        tick_ts = last_pkt_ts + max(0.0, now_wall - last_pkt_wall)
                        if last_tick_ts is not None:
                            tick_ts = max(last_tick_ts, tick_ts)
                        last_tick_ts = tick_ts
                        self._mouse.tick(tick_ts)
                    elif self._mouse.is_pressed:
                        self._mouse.tick(now_wall)
                    if self._idle_sleep_s > 0.0:
                        time.sleep(self._idle_sleep_s)
                    continue

                now_wall = float(self._clock_fn())
                next_idle_report = now_wall + 5.0
                if not self._stop_on_eof and frames_processed == 0:
                    logger.info("Receiving frames (%dx%d). Looking for the timing gauge and forge controls.", pkt.width, pkt.height)
                if prev_frame_wall is not None:
                    dt_wall = now_wall - prev_frame_wall
                    if dt_wall > 1e-6:
                        inst_fps = 1.0 / dt_wall
                        fps_ema = inst_fps if fps_ema is None else (0.85 * fps_ema + 0.15 * inst_fps)
                prev_frame_wall = now_wall

                raw_ts = pkt.timestamp_s
                if isinstance(raw_ts, bool) or not isinstance(raw_ts, (int, float)):
                    raise ValueError(
                        f"FramePacket.timestamp_s must be a finite number, got {raw_ts!r}"
                    )
                pkt_ts = float(raw_ts)
                if not math.isfinite(pkt_ts):
                    raise ValueError(
                        f"FramePacket.timestamp_s must be finite, got {raw_ts!r}"
                    )
                now_s = pkt_ts if last_tick_ts is None else max(last_tick_ts, pkt_ts)
                last_pkt_ts = now_s
                last_pkt_wall = now_wall
                last_tick_ts = now_s

                frames_processed += 1
                self._mouse.tick(now_s)

                det: DetectionResult | None = None
                decision: TriggerDecision | None = None

                gauge_allowed = not paused
                if gauge_allowed and smithy is not None:
                    # Give the original timing loop first use of active forge frames.
                    # A few missed frames can occur while the gauge is still running.
                    if gauge_was_allowed:
                        det = self._detector.detect_packet(pkt)
                        gauge_missing_frames = 0 if det.gauge_present else gauge_missing_frames + 1
                    if not gauge_was_allowed or gauge_missing_frames >= smithy_probe_after:
                        gauge_allowed = smithy.allows_gauge(replace(pkt, timestamp_s=now_s), client_rect)
                if not gauge_allowed:
                    det = None
                    gauge_missing_frames = 0
                if gauge_was_allowed and not gauge_allowed and not paused:
                    self._state_machine.reset()
                    self._detector.reset()
                gauge_was_allowed = gauge_allowed

                if gauge_allowed:
                    if det is None:
                        det = self._detector.detect_packet(pkt)
                    if det.gauge_present:
                        detections += 1
                    allow_trigger = not self._mouse.is_pressed
                    decision = self._state_machine.update(
                        det,
                        now_s,
                        allow_trigger=allow_trigger,
                    )

                    if decision.should_click:
                        set_frame_idx = getattr(self._mouse, "set_frame_idx", None)
                        if callable(set_frame_idx):
                            set_frame_idx(pkt.frame_idx)

                        if self._config.dry_run and isinstance(self._mouse, Win32SendInputMouse):
                            logger.debug(
                                "dry_run active: suppressed Win32SendInputMouse click at frame %s",
                                pkt.frame_idx,
                            )
                            clicks += 1
                            click_frames.append(pkt.frame_idx)
                        else:
                            pressed = self._mouse.press_left(decision.hold_ms, timestamp_s=now_s)
                            if pressed:
                                clicks += 1
                                click_frames.append(pkt.frame_idx)
                                release_ms = decision.hold_ms
                                if decision.release_delay_s > 0.0:
                                    release_ms = max(release_ms, decision.release_delay_s * 1000.0)
                                self._release_after_hold(now_s, release_ms)
                                if not self._stop_on_eof and clicks == 1:
                                    logger.info("Sent first timed Hit click")

                    if det.gauge_present and not self._stop_on_eof and not reported_gauge:
                        logger.info("Timing gauge detected")
                        reported_gauge = True

                if self._on_frame is not None:
                    self._on_frame(pkt, det, decision)

                if preview_win is not None:
                    keep_open = preview_win.show(
                        pkt.image_bgr,
                        det,
                        state=decision,
                        paused=not gauge_allowed,
                        fps=fps_ema,
                    )
                    if not keep_open:
                        stopped_by_preview = True
                        break
        finally:
            try:
                self._mouse.release_all()
            finally:
                if own_preview and preview_win is not None:
                    preview_win.close()

        elapsed_s = max(0.0, float(self._clock_fn()) - start_wall)
        return RunResult(
            frames_processed=frames_processed,
            detections=detections,
            clicks=clicks,
            click_frames=tuple(click_frames),
            stopped_by_hotkey=stopped_by_hotkey,
            stopped_by_preview=stopped_by_preview,
            elapsed_s=elapsed_s,
        )


def run_pipeline(
    source: FrameSource,
    mouse: MouseBackend,
    config: AppConfig | None = None,
    detector: GaugeDetector | None = None,
    state_machine: TriggerStateMachine | None = None,
    hotkeys: HotkeyController | None = None,
    **kwargs: Any,
) -> RunResult:
    """Convenience functional wrapper around Runner(...).run()."""
    return Runner(
        source=source,
        mouse=mouse,
        config=config,
        detector=detector,
        state_machine=state_machine,
        hotkeys=hotkeys,
        **kwargs,
    ).run()
