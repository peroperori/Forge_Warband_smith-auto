from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vrc_smith_auto.app.runner import Runner
from vrc_smith_auto.app.smithy import SmithyAutomation
from vrc_smith_auto.config import AppConfig
from vrc_smith_auto.input import SpyMouseBackend
from vrc_smith_auto.input import win32_sendinput
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.vision.smithy import SmithyVision, _load
from vrc_smith_auto.window import WindowRect


def scene(size, *placements):
    width, height = size
    frame = np.full((height, width, 3), 38, np.uint8)
    for reference, x, y in placements:
        image = _load(reference) if isinstance(reference, str) else reference
        scaled = cv2.resize(
            image, (round(image.shape[1] * width / 1920), round(image.shape[0] * height / 1080)),
            interpolation=cv2.INTER_AREA if width < 1920 else cv2.INTER_LINEAR,
        )
        frame[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    return frame


class RecordingInput(SpyMouseBackend):
    def __init__(self):
        super().__init__()
        self.events = []
        self.cursor = None

    def move_cursor(self, rect, x, y):
        self.cursor = x, y
        return True

    def press_left(self, duration_ms, timestamp_s=None):
        self.events.append(self.cursor)
        return super().press_left(duration_ms, timestamp_s)

    def press_tab(self, rect):
        self.events.append("Tab")
        return True


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (1024, 768), (800, 450), (2560, 1440)])
def test_scaled_templates_distinguish_button_states_and_storage(size):
    vision = SmithyVision(auto_sell=True)
    for name, other in [("hit", "hit_unavailable"), ("hit_unavailable", "hit"),
                        ("fire_up", "fire_up_unavailable"), ("fire_up_unavailable", "fire_up")]:
        vision.prepare(scene(size, (name, 231, 117)))
        assert vision.button(name) is not None
        assert vision.button(other) is None

    vision.prepare(scene(size, ("empty_storage", 231, 117)))
    assert vision.storage() == []
    vision.prepare(scene(size, ("full_storage", 231, 117)))
    assert len(vision.storage()) == 12
    vision.prepare(scene(size))
    assert vision.storage() is None


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)])
@pytest.mark.parametrize("distance_scale", [0.53, 0.62, 0.83, 1.19, 1.49])
def test_button_matching_adjusts_to_apparent_size(size, distance_scale):
    width, height = size
    scale = height / 1080 * distance_scale
    vision = SmithyVision(auto_sell=True)
    for name, other in [
        ("hit", "hit_unavailable"),
        ("hit_unavailable", "hit"),
        ("fire_up", "fire_up_unavailable"),
        ("fire_up_unavailable", "fire_up"),
        ("smithy", None),
    ]:
        image = cv2.resize(_load(name), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        x, y = 519, 173
        frame = np.full((height, width, 3), 38, np.uint8)
        frame[y:y + h, x:x + w] = image
        vision.prepare(frame)

        point = vision.button(name)
        assert point is not None, name
        assert abs(point[0] - (x + w // 2)) <= 3
        assert abs(point[1] - (y + h // 2)) <= 3
        if other is not None:
            assert vision.button(other) is None


def test_smithy_hover_accepts_pointer_inside_button_beyond_label(monkeypatch):
    frame = scene((1920, 1080), ("smithy", 750, 480))
    vision = SmithyVision(auto_sell=True)
    pointer = (960, 525)
    monkeypatch.setattr(vision, "cursor", lambda: pointer)
    vision.prepare(frame)

    assert vision.hovered_button("smithy") == pointer

    pointer = (1060, 525)
    assert vision.hovered_button("smithy") is None


def test_runner_sells_mixed_storage_then_resumes_gold(monkeypatch):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(sleep=lambda _: None, perf_counter=lambda: 0))
    size = (1280, 720)
    empty = _load("empty_storage")
    mixed = empty.copy()
    mixed[46:76, 29:57] = (185, 29, 101)
    mixed[102:132, 84:112] = (19, 161, 61)
    one_left = mixed.copy()
    one_left[46:76, 29:57] = empty[46:76, 29:57]
    forge = scene(size, ("hit_unavailable", 231, 117), ("fire_up", 231, 225))
    smithy = scene(size, ("smithy", 231, 117))
    two_items = scene(size, (mixed, 231, 117))
    hit = scene(size, ("hit", 231, 117))
    replay = [
        (0.0, forge), (0.499, forge), (0.6, forge), (1.5, forge), (1.8, smithy),
        (2.1, two_items), (2.4, two_items), (2.7, scene(size, (one_left, 231, 117))),
        (3.0, scene(size, (empty, 231, 117))), (3.3, hit),
    ]
    frames = iter(FramePacket(image, ts, i) for i, (ts, image) in enumerate(replay))
    mouse = RecordingInput()
    seen = []
    runner = Runner(
        source=SimpleNamespace(read=lambda: next(frames, None)),
        mouse=mouse,
        config=AppConfig.from_dict({"trigger": {"auto_fire_up": True, "auto_sell": True}}),
        stop_on_eof=True,
        detector=SimpleNamespace(
            reset=lambda: None,
            detect_packet=lambda packet: seen.append(packet.frame_idx) or SimpleNamespace(gauge_present=True),
        ),
        state_machine=SimpleNamespace(
            reset=lambda: None,
            update=lambda *args, **kwargs: SimpleNamespace(should_click=True, hold_ms=1, release_delay_s=0),
        ),
    )
    result = runner.run()
    assert seen == [9]
    assert result.click_frames == (9,)
    assert mouse.events == [
        (268, 243), "Tab", (325, 140),
        (260, 158), (260, 158), (296, 195), (296, 195), "Tab", (295, 140),
    ]


def test_perspective_storage_can_locate_grid_from_full_reference():
    vision = SmithyVision(auto_sell=True, perspective_storage=True)
    vision._storage_features.pop(0)
    frame = scene((1920, 1080), ("full_storage", 620, 310))
    vision.prepare(frame)

    slots = vision.storage()
    assert slots is not None
    expected = [(620 + x, 310 + y) for y in (57, 112, 166) for x in (35, 91, 146, 201)]
    np.testing.assert_allclose(slots, expected, atol=2)


def test_perspective_storage_tracks_partial_grid_through_motion_and_empty_state(monkeypatch):
    empty = _load("empty_storage")
    partial = empty.copy()
    centers = [(43, 61), (98, 61), (154, 117), (209, 172)]
    for x, y in centers:
        cv2.rectangle(partial, (x - 21, y - 21), (x + 21, y + 21), (161, 153, 148), 3)
        cv2.rectangle(partial, (x - 9, y - 8), (x + 9, y + 8), (85, 130, 190), -1)
    cv2.fillConvexPoly(partial, np.int32([(84, 115), (110, 126), (95, 139)]), (205, 178, 35))
    height, width = empty.shape[:2]
    corners = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
    vision = SmithyVision(auto_sell=True, perspective_storage=True)
    find_panel = vision._find_storage_transform
    monkeypatch.setattr(vision, "_find_storage_transform",
                        lambda bounds: None if offset == 37 else find_panel(bounds))

    for offset, panel, remaining in [(0, partial, centers), (37, partial, centers), (68, empty, [])]:
        destination = np.float32([(330 + offset, 180), (582 + offset, 184),
                                  (588 + offset, 423), (335 + offset, 428)])
        transform = cv2.getPerspectiveTransform(corners, destination)
        vision.prepare(cv2.warpPerspective(panel, transform, (1280, 1080)))
        occupied = vision.storage()
        assert occupied is not None
        assert len(occupied) == len(remaining)
        if remaining:
            expected = cv2.perspectiveTransform(np.float32([remaining]), transform)[0]
            np.testing.assert_allclose(occupied, expected, atol=3)


def test_runner_gives_active_gauge_priority_over_smithy(monkeypatch):
    probes = []
    detections = []
    smithy = SimpleNamespace(
        allows_gauge=lambda packet, _: probes.append(packet.frame_idx) or packet.frame_idx == 0
    )
    monkeypatch.setattr("vrc_smith_auto.app.runner.SmithyAutomation", lambda *args, **kwargs: smithy)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frames = iter(FramePacket(frame, i * 0.2, i) for i in range(8))

    def detect(packet):
        detections.append(packet.frame_idx)
        return SimpleNamespace(gauge_present=packet.frame_idx in {0, 2, 3})

    def update(detection, *_args, **_kwargs):
        return SimpleNamespace(should_click=detection.gauge_present, hold_ms=1, release_delay_s=0)

    result = Runner(
        source=SimpleNamespace(read=lambda: next(frames, None)),
        mouse=RecordingInput(),
        config=AppConfig.from_dict({"trigger": {"auto_fire_up": True, "rearm_lost_frames": 3}}),
        stop_on_eof=True,
        detector=SimpleNamespace(reset=lambda: None, detect_packet=detect),
        state_machine=SimpleNamespace(reset=lambda: None, update=update),
    ).run()

    assert probes == [0, 6, 7]
    assert detections == list(range(7))
    assert result.click_frames == (0, 2, 3)


@pytest.mark.parametrize("auto_sell", [False, True])
def test_fire_up_resumes_when_selling_is_disabled_or_fire_up_is_unavailable(monkeypatch, auto_sell):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(sleep=lambda _: None, perf_counter=lambda: 0))
    mouse = RecordingInput()
    automation = SmithyAutomation(mouse, auto_sell)
    size = (1280, 720)
    forge = scene(size, ("hit_unavailable", 231, 117), ("fire_up", 231, 225))
    ready = scene(size, ("hit", 231, 117), ("fire_up_unavailable", 231, 225))
    assert not automation.allows_gauge(FramePacket(forge, 0.0, 0))
    assert automation.allows_gauge(FramePacket(ready, 0.6, 1))
    assert mouse.events == [(268, 243)]
    assert mouse.cursor == (295, 140)


def test_delayed_disabled_fire_up_does_not_open_smithy(monkeypatch):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(
        sleep=lambda _: None, perf_counter=lambda: 0,
    ))
    mouse = RecordingInput()
    automation = SmithyAutomation(mouse, auto_sell=True)
    size = (1280, 720)
    forge = scene(size, ("hit_unavailable", 400, 117), ("fire_up", 231, 117))
    ready = scene(size, ("hit", 400, 117), ("fire_up_unavailable", 231, 117))
    for index, timestamp in enumerate((0.0, 0.55, 0.85)):
        assert not automation.allows_gauge(FramePacket(forge, timestamp, index))
    assert automation.allows_gauge(FramePacket(ready, 1.1, 3))
    assert len(mouse.events) == 1


def test_fire_up_waits_for_enabled_button_and_accepts_height_scaled_ui(monkeypatch):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(sleep=lambda _: None, perf_counter=lambda: 0))
    mouse = RecordingInput()
    automation = SmithyAutomation(mouse, auto_sell=True)
    size = (1024, 768)
    disabled = scene(size, ("hit_unavailable", 231, 117), ("fire_up_unavailable", 231, 225))
    enabled = scene(size, ("hit_unavailable", 231, 117))
    image = _load("fire_up")
    scaled = cv2.resize(image, None, fx=768 / 1080, fy=768 / 1080, interpolation=cv2.INTER_AREA)
    h, w = scaled.shape[:2]
    enabled[225:225 + h, 231:231 + w] = scaled

    assert not automation.allows_gauge(FramePacket(disabled, 0.0, 0))
    assert not automation.allows_gauge(FramePacket(disabled, 9.0, 1))
    assert mouse.events == []
    assert not automation.allows_gauge(FramePacket(enabled, 9.2, 2))
    assert mouse.events == [(231 + w // 2, 225 + h // 2)]


def test_fire_up_timeout_reports_when_input_is_blocked():
    mouse = RecordingInput()
    mouse.move_cursor = lambda *args: False
    mouse.last_blocked_reason = "not_foreground"
    automation = SmithyAutomation(mouse, auto_sell=True)
    forge = scene((1280, 720), ("hit_unavailable", 231, 117), ("fire_up", 231, 225))
    assert not automation.allows_gauge(FramePacket(forge, 0.0, 0))
    with pytest.raises(RuntimeError, match="input was blocked.*not_foreground"):
        automation.allows_gauge(FramePacket(forge, 8.1, 1))


def test_relative_aim_waits_for_a_new_frame_before_clicking(monkeypatch):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(
        sleep=lambda _: None, perf_counter=lambda: 0,
    ))
    mouse = RecordingInput()
    moves = []
    mouse.move_relative = lambda rect, dx, dy: moves.append((dx, dy)) or True
    automation = SmithyAutomation(mouse, auto_sell=False, use_rendered_cursor=True)
    cursor = (100, 135)
    monkeypatch.setattr(automation._vision, "cursor", lambda: cursor)
    monkeypatch.setattr(automation._vision, "hovered_button", lambda *args, **kwargs: None)
    size = (1280, 720)
    frame = scene(size, ("hit_unavailable", 360, 117), ("fire_up", 231, 117))
    rect = WindowRect.from_xywh(0, 0, *size)

    assert not automation.allows_gauge(FramePacket(frame, 0.0, 0, rect))
    assert moves == [(24, 0)]
    assert mouse.events == []

    cursor = (268, 135)
    assert not automation.allows_gauge(FramePacket(frame, 0.2, 1, rect))
    assert len(mouse.events) == 1


def test_ui_input_uses_client_origin_and_requires_current_foreground_window(monkeypatch):
    rect = WindowRect.from_xywh(-1260, 90, 1280, 720, hwnd=42)
    foreground = False
    sent = []
    moves = []
    monkeypatch.setattr(win32_sendinput, "get_client_rect", lambda _: rect)
    monkeypatch.setattr(win32_sendinput, "time", SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setattr(
        win32_sendinput.ctypes, "windll",
        SimpleNamespace(user32=SimpleNamespace(SetCursorPos=lambda x, y: moves.append((x, y)) or 1)),
        raising=False,
    )
    mouse = win32_sendinput.Win32SendInputMouse(
        target_hwnd=42,
        check_uipi=False,
        is_foreground_fn=lambda _: foreground,
        send_input_fn=lambda event: sent.append((event.type, event.ki.wScan, event.ki.dwFlags)) or 1,
        get_last_error_fn=lambda: 0,
    )
    assert not mouse.move_cursor(rect, 300, 120)
    assert not mouse.press_tab(rect)
    foreground = True
    assert not mouse.move_cursor(WindowRect.from_xywh(-1261, 90, 1280, 720, hwnd=42), 300, 120)
    assert mouse.move_cursor(rect, 300, 120)
    assert mouse.press_tab(rect)
    assert moves == [(-960, 210)]
    assert sent == [(1, 0x0F, 0x08), (1, 0x0F, 0x0A)]
