from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Callable

from vrc_smith_auto.protocols import FramePacket, SmithyInput
from vrc_smith_auto.vision.smithy import SmithyVision
from vrc_smith_auto.window import WindowRect

logger = logging.getLogger(__name__)


class SmithyAutomation:
    """Advance repair and selling using fresh frames, pausing the gauge meanwhile."""

    def __init__(self, mouse: SmithyInput, auto_sell: bool, *, use_rendered_cursor: bool = False) -> None:
        self._mouse = mouse
        self._auto_sell = auto_sell
        self._move_relative = getattr(mouse, "move_relative", None) if use_rendered_cursor else None
        self._vision = SmithyVision(auto_sell, perspective_storage=self._move_relative is not None)
        self._state = "gauge"
        self._next_probe = 0.0
        self._deadline = 0.0
        self._previous_count = 13
        self._sale_retry_at = 0.0
        self._positioned_hit = False
        self._hit_point: tuple[int, int] | None = None
        self._last_status: str | None = None
        self._aim_status: str | None = None
        self._tab_retry_at: float | None = None
        self._fire_up_check_at = 0.0
        self._fire_up_enabled_frames = 0

    def _report(self, status: str) -> None:
        if status != self._last_status:
            logger.info("%s", status)
            self._last_status = status

    def _gauge_ready(self) -> bool:
        return self._state == "gauge" and (self._move_relative is None or self._positioned_hit)

    def _aim(self, rect: WindowRect | None, point: tuple[int, int]) -> bool:
        self._aim_status = None
        if self._move_relative is None or rect is None:
            return self._mouse.move_cursor(rect, *point)
        cursor = self._vision.cursor()
        if cursor is None:
            if self._state != "sell":
                self._aim_status = "waiting for the VRChat pointer. Enable cursor mode with Tab"
                return False
            # In camera mode the center reticle can disappear between controls.
            # Use the center to steer, then require a visible pointer to click.
            cursor = (rect.width // 2, rect.height // 2)
            if abs(point[0] - cursor[0]) <= 13 and abs(point[1] - cursor[1]) <= 8:
                self._aim_status = "waiting for the VRChat pointer over the storage slot"
                return False
        dx, dy = point[0] - cursor[0], point[1] - cursor[1]
        if abs(dx) <= 13 and abs(dy) <= 8:
            return True
        step_x = max(-24, min(24, round(dx * 0.35)))
        step_y = max(-24, min(24, round(dy * 0.35)))
        if self._move_relative(rect, step_x, step_y):
            self._aim_status = "moving the VRChat pointer"
        return False

    def _click(self, rect: WindowRect | None, point: tuple[int, int], *, double: bool = False) -> bool:
        if not self._aim(rect, point):
            return False
        for index in range(2 if double else 1):
            if index:
                time.sleep(0.045)
            now = time.perf_counter()
            if not self._mouse.press_left(35, timestamp_s=now):
                return False
            try:
                time.sleep(0.035)
            finally:
                self._mouse.tick(now + 0.035)
        return True

    def _enter(self, state: str, now: float, delay: float = 0.12) -> None:
        self._state = state
        self._next_probe = now + delay
        self._deadline = now + 8.0
        self._report(f"Forge state: {state.replace('_', ' ')}")

    def _client_frame(self, packet: FramePacket, client: WindowRect | None) -> FramePacket | None:
        # Offline sources supply image coordinates. Live frames can include the
        # whole monitor, but actions require the whole VRChat client to be visible.
        if packet.screen_rect is None:
            return packet
        captured = packet.screen_rect
        if client is None or not client.is_valid:
            return None
        x, y = client.left - captured.left, client.top - captured.top
        if x < 0 or y < 0 or x + client.width > packet.width or y + client.height > packet.height:
            return None
        return replace(packet, image_bgr=packet.image_bgr[y:y + client.height, x:x + client.width],
                       screen_rect=client)

    def allows_gauge(
        self, packet: FramePacket, client_rect: Callable[[], WindowRect | None] | None = None
    ) -> bool:
        now = packet.timestamp_s
        if now < self._next_probe:
            return self._gauge_ready()
        probe_started = time.perf_counter()
        self._next_probe = now + 0.12
        if self._state not in ("gauge", "fire_up") and now >= self._deadline:
            raise RuntimeError(f"Smithy automation timed out during {self._state.replace('_', ' ')}")

        client = client_rect() if client_rect is not None else packet.screen_rect
        captured = packet.screen_rect
        packet = self._client_frame(packet, client)
        if packet is None:
            if client is None:
                self._report("Waiting for a visible VRChat window")
            else:
                self._report(
                    f"Waiting for the full VRChat client in the capture "
                    f"(client {client.as_ltrb()}, capture {captured.as_ltrb()})"
                )
            if self._state == "fire_up" and now >= self._deadline:
                raise RuntimeError("Fire Up timed out because the full VRChat client is outside the capture")
            return self._move_relative is None and self._state == "gauge"
        rect = packet.screen_rect
        self._vision.prepare(packet.image_bgr)

        def find(name: str) -> tuple[int, int] | None:
            point = self._vision.button(name)
            if point is None and self._move_relative is not None:
                point = self._vision.hovered_button(name)
            return point

        if self._state == "gauge":
            if self._move_relative is not None and self._auto_sell and self._last_status is None:
                if find("smithy") is not None:
                    self._enter("open_smithy", now, 0.0)
                    return False
                occupied = self._vision.storage()
                if occupied:
                    self._enter("sell", now, 0.0)
                    return False
            # The hovered Hit can match both references under its pointer glow.
            # Prefer the enabled state unless Fire Up is also visibly available.
            hit = find("hit") if self._move_relative is not None else None
            hit_disabled = find("hit_unavailable")
            fire_up_ready = (
                hit_disabled is not None and (
                    self._move_relative is None or hit is None
                    or self._vision.hovered_button("fire_up", hit=hit_disabled) is not None
                )
            )
            if not fire_up_ready:
                if self._move_relative is not None and self._positioned_hit:
                    cursor = self._vision.cursor()
                    self._positioned_hit = (
                        cursor is not None and self._hit_point is not None
                        and abs(cursor[0] - self._hit_point[0]) <= 13
                        and abs(cursor[1] - self._hit_point[1]) <= 8
                    )
                if not self._positioned_hit:
                    hit = hit or find("hit")
                    if hit is None:
                        self._report("Waiting for a recognizable Hit or Fire Up control")
                    elif self._aim(rect, hit):
                        self._positioned_hit = True
                        self._hit_point = hit
                        self._report("Cursor positioned over Hit")
                    else:
                        reason = self._aim_status or getattr(self._mouse, "last_blocked_reason", None)
                        self._report(f"Waiting to position the cursor over Hit ({reason or 'input blocked'})")
                return self._gauge_ready()
            self._positioned_hit = False
            self._enter("fire_up", now)
        if self._state == "fire_up":
            point = find("fire_up")
            if self._move_relative is not None:
                anchor = self._vision.hovered_button("hit_unavailable") or find("hit_unavailable")
                if anchor is not None:
                    if point is None or not (
                        35 * packet.height / 1080 <= anchor[0] - point[0] <= 250 * packet.height / 1080
                        and abs(anchor[1] - point[1]) <= 35 * packet.height / 1080
                    ):
                        point = self._vision.hovered_button("fire_up", hit=anchor)
                else:
                    point = None
            if point is not None and self._click(rect, point):
                logger.info("Sent Fire Up click")
                clicked_at = now + time.perf_counter() - probe_started
                self._fire_up_check_at = clicked_at + 1.5
                self._fire_up_enabled_frames = 0
                self._enter("after_fire_up", clicked_at, 0.5 if self._auto_sell else 0.12)
                return False
            if point is not None:
                reason = getattr(self._mouse, "last_blocked_reason", None)
                status = self._aim_status or f"input was blocked ({reason or 'cursor movement or click failed'})"
            else:
                hit = find("hit")
                if hit is not None and self._aim(rect, hit):
                    self._positioned_hit = True
                    self._hit_point = hit
                    self._enter("gauge", now)
                    return True
                if hit is not None:
                    reason = self._aim_status or getattr(self._mouse, "last_blocked_reason", None)
                    status = f"the cursor could not move to Hit ({reason or 'cursor movement failed'})"
                elif find("fire_up_unavailable") is not None:
                    self._report("Waiting for Fire Up to become available")
                    self._deadline = now + 8.0
                    return False
                else:
                    status = "the enabled Fire Up button was not recognized"
            self._report(f"Waiting for Fire Up because {status}")
            if now >= self._deadline:
                raise RuntimeError(f"Fire Up timed out because {status}")
            return False
        elif self._state == "after_fire_up":
            if not self._auto_sell or find("fire_up_unavailable") is not None:
                self._enter("return_to_hit", now, 0.0)
            else:
                enabled = find("fire_up") is not None and find("hit_unavailable") is not None
                self._fire_up_enabled_frames = self._fire_up_enabled_frames + 1 if enabled else 0
                if not enabled and find("hit") is not None:
                    self._enter("return_to_hit", now, 0.0)
                elif now >= self._fire_up_check_at and self._fire_up_enabled_frames >= 2:
                    logger.info("Fire Up and disabled Hit are still visible after the click. Checking Smithy for full storage.")
                    self._enter("open_menu", now, 0.0)
                else:
                    self._report("Waiting for the forge controls to settle after Fire Up")
                    return False
        if self._state == "open_menu":
            if self._mouse.press_tab(rect):
                self._tab_retry_at = now + 1.4
                self._enter("open_smithy", now, 0.25)
        elif self._state == "open_smithy":
            point = self._vision.hovered_button("smithy") if self._move_relative is not None else None
            if point is None:
                point = self._vision.button("smithy")
            if point is not None:
                self._tab_retry_at = None
                if self._click(rect, point):
                    logger.info("Sent Smithy click")
                    self._previous_count = 13
                    self._sale_retry_at = 0.0
                    self._enter("sell", now, 0.25)
                else:
                    reason = self._aim_status or getattr(self._mouse, "last_blocked_reason", None)
                    self._report(f"Waiting to click Smithy ({reason or 'mouse click failed'})")
            elif self._move_relative is not None and self._tab_retry_at is not None and now >= self._tab_retry_at:
                if self._mouse.press_tab(rect):
                    self._tab_retry_at = None
                    self._report("Waiting for the Smithy popup after retrying Tab")
            else:
                self._report("Waiting to recognize the Smithy button")
        elif self._state == "sell":
            occupied = self._vision.storage()
            if occupied is None:
                self._report("Waiting to recognize the storage grid")
            elif occupied == []:
                if self._mouse.press_tab(rect):
                    self._enter("after_sell", now, 0.25)
            elif len(occupied) < self._previous_count or now >= self._sale_retry_at:
                if self._click(rect, occupied[0], double=True):
                    logger.info("Double-clicked an occupied slot (%d before click)", len(occupied))
                    progressed = len(occupied) < self._previous_count
                    self._previous_count = len(occupied)
                    self._sale_retry_at = now + 0.9
                    self._next_probe = now + 0.25
                    if progressed:
                        self._deadline = now + 8.0
                else:
                    reason = self._aim_status or getattr(self._mouse, "last_blocked_reason", None)
                    self._report(f"Storage has {len(occupied)} occupied slots. {reason or 'Waiting to click'}")
        elif self._state in ("return_to_hit", "after_sell"):
            point = find("hit")
            if point is not None and self._aim(rect, point):
                self._positioned_hit = True
                self._hit_point = point
                self._enter("gauge", now)
                return True
            if self._state == "after_sell" and find("hit_unavailable") is not None:
                self._enter("fire_up", now)
        return False
