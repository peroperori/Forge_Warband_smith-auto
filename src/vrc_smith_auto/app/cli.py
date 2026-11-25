from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from vrc_smith_auto.app.hotkeys import HotkeyController
from vrc_smith_auto.app.preview import PreviewWindow
from vrc_smith_auto.app.runner import Runner
from vrc_smith_auto.capture import (
    CaptureError,
    ImageFrameSource,
    VideoFileFrameSource,
    create_live_capture,
)
from vrc_smith_auto.config import VALID_CAPTURE_BACKENDS, AppConfig
from vrc_smith_auto.control import TriggerStateMachine
from vrc_smith_auto.input import SpyMouseBackend, Win32SendInputMouse
from vrc_smith_auto.protocols import FramePacket, MouseBackend
from vrc_smith_auto.vision.detector import GaugeDetector
from vrc_smith_auto.window import enable_dpi_awareness, find_vrchat_window

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vrc-smith-auto",
        description="VRChat Desktop blacksmith timing gauge auto-clicker and verification CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Run live capture and auto-click loop against VRChat desktop window.",
    )
    p_run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML or JSON AppConfig file.",
    )
    p_run.add_argument(
        "--backend",
        choices=VALID_CAPTURE_BACKENDS,
        default=None,
        help="Screen capture backend override (auto, dxcam, mss, gdi).",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Run detection and state machine without sending OS input.",
    )
    p_run.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Show live OpenCV debug overlay window (press 'q' to quit).",
    )
    p_run.add_argument(
        "--no-focus-guard",
        action="store_true",
        default=None,
        help="Allow clicking even when VRChat is not the foreground window.",
    )

    p_img = sub.add_parser(
        "verify-image",
        help="Run offline detection and trigger check on a single image file.",
    )
    p_img.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to PNG/JPG image file.",
    )
    p_img.add_argument(
        "--expect-clicks",
        type=int,
        default=0,
        help="Expected click count (non-zero exit if actual clicks differ). Default: 0.",
    )
    p_img.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON result to stdout.",
    )

    p_vid = sub.add_parser(
        "verify-video",
        help="Run deterministic offline replay on a video file.",
    )
    p_vid.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to MP4/AVI video file.",
    )
    p_vid.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use SpyMouseBackend dry-run mode (always dry-run for offline verification).",
    )
    p_vid.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON result to stdout.",
    )

    p_bench = sub.add_parser(
        "benchmark",
        help="Measure per-frame processing latency and verify SLA threshold.",
    )
    src_group = p_bench.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to video file to preload and benchmark.",
    )
    src_group.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Benchmark live screen capture + vision + control loop.",
    )
    p_bench.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of timed frames after warmup. Default: 200.",
    )
    p_bench.add_argument(
        "--max-p95-ms",
        type=float,
        default=16.7,
        help="Maximum allowed p95 per-frame latency in milliseconds. Default: 16.7.",
    )
    p_bench.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON metrics to stdout.",
    )

    return parser


def _print_err(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = AppConfig.from_file(args.config) if args.config is not None else AppConfig()
    except (OSError, ValueError) as exc:
        _print_err(f"Failed to load config: {exc}")
        return 2

    overrides: dict[str, Any] = {}
    if args.backend is not None:
        overrides["capture_backend"] = args.backend
    if args.dry_run is True:
        overrides["dry_run"] = True
    if args.no_focus_guard is True:
        overrides["require_foreground"] = False

    if overrides:
        try:
            cfg = cfg.with_overrides(overrides)
        except ValueError as exc:
            _print_err(f"Invalid configuration override: {exc}")
            return 2

    if sys.platform == "win32":
        enable_dpi_awareness()

    try:
        logger.info("Initializing screen capture (backend=%s, monitor=%d)", cfg.capture.backend, cfg.capture.monitor_idx)
        source = create_live_capture(cfg, probe=True)
    except (CaptureError, OSError, RuntimeError, ValueError) as exc:
        _print_err(f"Failed to initialize live capture: {exc}")
        return 1

    mouse: MouseBackend
    if cfg.dry_run and not cfg.auto_fire_up:
        mouse = SpyMouseBackend()
    else:
        class_arg = cfg.window.class_name or None
        proc_arg = "VRChat.exe" if cfg.window.title == "VRChat" else None
        mouse = Win32SendInputMouse(
            require_foreground=cfg.require_foreground,
            hwnd_provider=lambda: find_vrchat_window(
                cfg.window.title,
                class_arg,
                process_name=proc_arg,
            ),
            check_uipi=True,
        )

    hotkeys = HotkeyController()
    preview = PreviewWindow() if args.preview else None

    try:
        logger.info(
            "Started with %s capture. %s F8 pauses or resumes. Esc stops.",
            type(source).__name__,
            "Dry run." if cfg.dry_run else "Keep VRChat in the foreground." if cfg.require_foreground else "",
        )
        runner = Runner(
            source=source,
            mouse=mouse,
            config=cfg,
            hotkeys=hotkeys,
            preview=preview,
            show_preview=bool(args.preview),
            stop_on_eof=False,
        )
        result = runner.run()
    except KeyboardInterrupt:
        return 0
    except (CaptureError, OSError, RuntimeError, ValueError) as exc:
        _print_err(f"Runtime failure: {exc}")
        return 1
    finally:
        if preview is not None:
            preview.close()
        source.close()

    print(
        f"Stopped: frames={result.frames_processed} clicks={result.clicks} "
        f"elapsed_s={result.elapsed_s:.2f} hotkey_stop={result.stopped_by_hotkey}"
    )
    return 0


def _cmd_verify_image(args: argparse.Namespace) -> int:
    if args.expect_clicks < 0:
        _print_err("--expect-clicks must be >= 0")
        return 2

    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False)
    try:
        source = ImageFrameSource(args.image, repeat=1)
    except (FileNotFoundError, CaptureError, OSError, ValueError) as exc:
        _print_err(str(exc))
        return 1

    mouse = SpyMouseBackend()
    try:
        result = Runner(source=source, mouse=mouse, config=cfg, stop_on_eof=True).run()
    finally:
        source.close()

    passed = result.clicks == args.expect_clicks
    payload = {
        "image": str(args.image),
        "frames_processed": result.frames_processed,
        "detections": result.detections,
        "clicks": result.clicks,
        "click_frames": list(result.click_frames),
        "expected_clicks": int(args.expect_clicks),
        "passed": passed,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"verify-image: image={args.image} detections={result.detections} "
            f"clicks={result.clicks} expected={args.expect_clicks} passed={passed}"
        )

    return 0 if passed else 1


def _cmd_verify_video(args: argparse.Namespace) -> int:
    del args.dry_run  # Offline video verification always uses SpyMouseBackend
    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False)
    try:
        source = VideoFileFrameSource(args.video)
    except (FileNotFoundError, CaptureError, OSError, ValueError) as exc:
        _print_err(str(exc))
        return 1

    mouse = SpyMouseBackend()
    try:
        result = Runner(source=source, mouse=mouse, config=cfg, stop_on_eof=True).run()
        fps = float(source.fps)
    finally:
        source.close()

    payload = {
        "video": str(args.video),
        "fps": fps,
        "frames_processed": result.frames_processed,
        "detections": result.detections,
        "clicks": result.clicks,
        "click_frames": list(result.click_frames),
        "elapsed_s": round(result.elapsed_s, 6),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"verify-video: video={args.video} frames={result.frames_processed} "
            f"detections={result.detections} clicks={result.clicks} "
            f"click_frames={list(result.click_frames)} elapsed_s={result.elapsed_s:.3f}"
        )
    return 0


def _percentile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(math.ceil(q * n)) - 1))
    return sorted_vals[idx]


def _cmd_benchmark(args: argparse.Namespace) -> int:
    iterations = int(args.iterations)
    max_p95_ms = float(args.max_p95_ms)
    if iterations <= 0:
        _print_err("--iterations must be > 0")
        return 2
    if not math.isfinite(max_p95_ms) or max_p95_ms <= 0.0:
        _print_err("--max-p95-ms must be a finite number > 0")
        return 2

    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False)
    detector = GaugeDetector(cfg)
    sm = TriggerStateMachine(cfg)
    mouse = SpyMouseBackend()

    warmup = 10
    timings_ms: list[float] = []

    if args.live:
        if sys.platform == "win32":
            enable_dpi_awareness()
        try:
            source = create_live_capture(cfg, probe=True)
        except (CaptureError, OSError, RuntimeError, ValueError) as exc:
            _print_err(f"Live capture unavailable for benchmark: {exc}")
            return 1

        try:
            # Warmup up to `warmup` frames with a 3s guard so static desktops don't hang
            warm_done = 0
            warm_deadline = time.perf_counter() + 3.0
            while warm_done < warmup and time.perf_counter() < warm_deadline:
                pkt = source.read()
                if pkt is None:
                    time.sleep(0.001)
                    continue
                det = detector.detect_packet(pkt)
                sm.update(det, pkt.timestamp_s)
                warm_done += 1

            # Timed iterations (each capturing + detecting + stepping state machine)
            deadline = time.perf_counter() + max(10.0, iterations * 0.2)
            while len(timings_ms) < iterations:
                if time.perf_counter() >= deadline:
                    _print_err(
                        f"Timed out waiting for live frames ({len(timings_ms)}/{iterations} captured)."
                    )
                    return 1
                t0 = time.perf_counter()
                pkt = source.read()
                if pkt is None:
                    time.sleep(0.001)
                    continue
                det = detector.detect_packet(pkt)
                sm.update(det, pkt.timestamp_s)
                mouse.tick(pkt.timestamp_s)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
        finally:
            source.close()
    else:
        preload_limit = min(max(iterations + warmup, 60), 300)
        try:
            source = VideoFileFrameSource(
                args.video,
                preload=True,
                loop=True,
                max_frames=preload_limit,
            )
        except (FileNotFoundError, CaptureError, OSError, ValueError) as exc:
            _print_err(str(exc))
            return 1

        packets: list[FramePacket] = []
        try:
            for _ in range(warmup + iterations):
                pkt = source.read()
                if pkt is None:
                    break
                packets.append(pkt)
        finally:
            source.close()

        if not packets:
            _print_err(f"Video contains no readable frames: {args.video}")
            return 1

        warm_slice = packets[: min(warmup, len(packets))]
        for pkt in warm_slice:
            det = detector.detect_packet(pkt)
            sm.update(det, pkt.timestamp_s)

        detector.reset()
        sm.reset()
        mouse.clear()

        timed_slice = packets[len(warm_slice) : len(warm_slice) + iterations]
        if not timed_slice:
            timed_slice = packets

        for pkt in timed_slice:
            t0 = time.perf_counter()
            det = detector.detect_packet(pkt)
            dec = sm.update(det, pkt.timestamp_s)
            if dec.should_click:
                mouse.set_frame_idx(pkt.frame_idx)
                mouse.press_left(dec.hold_ms, timestamp_s=pkt.timestamp_s)
            mouse.tick(pkt.timestamp_s)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        warm_done = len(warm_slice)

    timings_sorted = sorted(timings_ms)
    n = len(timings_sorted)
    mean_ms = sum(timings_sorted) / n
    min_ms = timings_sorted[0]
    max_ms = timings_sorted[-1]
    p50_ms = _percentile(timings_sorted, 0.50)
    p95_ms = _percentile(timings_sorted, 0.95)
    passed = p95_ms <= max_p95_ms

    payload = {
        "mode": "live" if args.live else "video",
        "iterations": n,
        "warmup": warm_done,
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(p50_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "max_p95_ms": max_p95_ms,
        "passed": passed,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"benchmark ({payload['mode']}): n={n} mean={mean_ms:.3f}ms "
            f"p50={p50_ms:.3f}ms p95={p95_ms:.3f}ms max={max_ms:.3f}ms "
            f"(limit={max_p95_ms:.3f}ms) passed={passed}"
        )
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        return _cmd_run(args)
    if args.command == "verify-image":
        return _cmd_verify_image(args)
    if args.command == "verify-video":
        return _cmd_verify_video(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)

    _print_err(f"Unknown command: {args.command}")
    return 2
