from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CaptureBackendName = Literal["auto", "dxcam", "mss", "gdi"]
VALID_CAPTURE_BACKENDS: tuple[str, ...] = ("auto", "dxcam", "mss", "gdi")


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {type(value).__name__}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return fval


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_str(value: Any, name: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str, got {type(value).__name__}")
    if non_empty and not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class HSVRange:
    """Inclusive OpenCV HSV thresholds (H: 0..179, S: 0..255, V: 0..255)."""

    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def __post_init__(self) -> None:
        h_min = _require_int(self.h_min, "h_min")
        h_max = _require_int(self.h_max, "h_max")
        s_min = _require_int(self.s_min, "s_min")
        s_max = _require_int(self.s_max, "s_max")
        v_min = _require_int(self.v_min, "v_min")
        v_max = _require_int(self.v_max, "v_max")

        if not (0 <= h_min <= h_max <= 179):
            raise ValueError(f"Invalid H range [{h_min}, {h_max}] (expected 0 <= h_min <= h_max <= 179)")
        if not (0 <= s_min <= s_max <= 255):
            raise ValueError(f"Invalid S range [{s_min}, {s_max}] (expected 0 <= s_min <= s_max <= 255)")
        if not (0 <= v_min <= v_max <= 255):
            raise ValueError(f"Invalid V range [{v_min}, {v_max}] (expected 0 <= v_min <= v_max <= 255)")

        object.__setattr__(self, "h_min", h_min)
        object.__setattr__(self, "h_max", h_max)
        object.__setattr__(self, "s_min", s_min)
        object.__setattr__(self, "s_max", s_max)
        object.__setattr__(self, "v_min", v_min)
        object.__setattr__(self, "v_max", v_max)

    def to_bounds(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return ((h_min, s_min, v_min), (h_max, s_max, v_max)) for cv2.inRange."""
        return (
            (self.h_min, self.s_min, self.v_min),
            (self.h_max, self.s_max, self.v_max),
        )


@dataclass(frozen=True, slots=True)
class HSVConfig:
    """Empirical HSV bounds for Gold (完美), Silver (不错), and Red bar."""

    gold: HSVRange = field(default_factory=lambda: HSVRange(15, 30, 100, 255, 140, 255))
    silver: HSVRange = field(default_factory=lambda: HSVRange(0, 179, 0, 45, 135, 240))
    red_low: HSVRange = field(default_factory=lambda: HSVRange(0, 12, 100, 255, 135, 255))
    red_high: HSVRange = field(default_factory=lambda: HSVRange(168, 179, 100, 255, 135, 255))

    def __post_init__(self) -> None:
        for field_name in ("gold", "silver", "red_low", "red_high"):
            val = getattr(self, field_name)
            if not isinstance(val, HSVRange):
                raise ValueError(f"hsv.{field_name} must be an HSVRange instance")


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Resolution-scaled geometric validation and ROI tracking parameters."""

    reference_height: int = 1080
    min_gold_width_px: float = 16.0
    max_gold_width_px: float = 120.0
    min_gold_height_px: float = 8.0
    max_gold_height_px: float = 150.0
    min_gold_fill_ratio: float = 0.45
    min_silver_neighbor_ratio: float = 0.25
    morph_bridge_kernel_h: int = 15
    red_min_width_ratio: float = 0.85
    red_max_thickness_ratio: float = 0.60
    red_max_axis_offset_ratio: float = 0.60
    acquire_downscale: float = 0.5
    roi_padding_px: int = 180
    max_lost_frames: int = 3

    def __post_init__(self) -> None:
        ref_h = _require_int(self.reference_height, "geometry.reference_height")
        if ref_h <= 0:
            raise ValueError("geometry.reference_height must be > 0")

        min_w = _require_float(self.min_gold_width_px, "geometry.min_gold_width_px")
        max_w = _require_float(self.max_gold_width_px, "geometry.max_gold_width_px")
        if min_w <= 0.0 or max_w < min_w:
            raise ValueError("geometry gold width bounds require 0 < min_gold_width_px <= max_gold_width_px")

        min_h = _require_float(self.min_gold_height_px, "geometry.min_gold_height_px")
        max_h = _require_float(self.max_gold_height_px, "geometry.max_gold_height_px")
        if min_h <= 0.0 or max_h < min_h:
            raise ValueError("geometry gold height bounds require 0 < min_gold_height_px <= max_gold_height_px")

        fill_ratio = _require_float(self.min_gold_fill_ratio, "geometry.min_gold_fill_ratio")
        if not (0.0 < fill_ratio <= 1.0):
            raise ValueError("geometry.min_gold_fill_ratio must be in (0.0, 1.0]")

        silver_ratio = _require_float(self.min_silver_neighbor_ratio, "geometry.min_silver_neighbor_ratio")
        if not (0.0 <= silver_ratio <= 1.0):
            raise ValueError("geometry.min_silver_neighbor_ratio must be in [0.0, 1.0]")

        bridge_k = _require_int(self.morph_bridge_kernel_h, "geometry.morph_bridge_kernel_h")
        if bridge_k < 1:
            raise ValueError("geometry.morph_bridge_kernel_h must be >= 1")

        red_min_w = _require_float(self.red_min_width_ratio, "geometry.red_min_width_ratio")
        if red_min_w <= 0.0:
            raise ValueError("geometry.red_min_width_ratio must be > 0.0")

        red_max_t = _require_float(self.red_max_thickness_ratio, "geometry.red_max_thickness_ratio")
        if red_max_t <= 0.0:
            raise ValueError("geometry.red_max_thickness_ratio must be > 0.0")

        red_max_off = _require_float(self.red_max_axis_offset_ratio, "geometry.red_max_axis_offset_ratio")
        if red_max_off <= 0.0:
            raise ValueError("geometry.red_max_axis_offset_ratio must be > 0.0")

        downscale = _require_float(self.acquire_downscale, "geometry.acquire_downscale")
        if not (0.0 < downscale <= 1.0):
            raise ValueError("geometry.acquire_downscale must be in (0.0, 1.0]")

        roi_pad = _require_int(self.roi_padding_px, "geometry.roi_padding_px")
        if roi_pad < 0:
            raise ValueError("geometry.roi_padding_px must be >= 0")

        max_lost = _require_int(self.max_lost_frames, "geometry.max_lost_frames")
        if max_lost < 1:
            raise ValueError("geometry.max_lost_frames must be >= 1")

        object.__setattr__(self, "reference_height", ref_h)
        object.__setattr__(self, "min_gold_width_px", min_w)
        object.__setattr__(self, "max_gold_width_px", max_w)
        object.__setattr__(self, "min_gold_height_px", min_h)
        object.__setattr__(self, "max_gold_height_px", max_h)
        object.__setattr__(self, "min_gold_fill_ratio", fill_ratio)
        object.__setattr__(self, "min_silver_neighbor_ratio", silver_ratio)
        object.__setattr__(self, "morph_bridge_kernel_h", bridge_k)
        object.__setattr__(self, "red_min_width_ratio", red_min_w)
        object.__setattr__(self, "red_max_thickness_ratio", red_max_t)
        object.__setattr__(self, "red_max_axis_offset_ratio", red_max_off)
        object.__setattr__(self, "acquire_downscale", downscale)
        object.__setattr__(self, "roi_padding_px", roi_pad)
        object.__setattr__(self, "max_lost_frames", max_lost)


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    """Trigger state machine hysteresis, debounce, and mouse hold parameters."""

    # The target game judges on mouse *release* (Unity UI pointer-up), so the
    # button is tapped and released within the same Runner iteration.  A long
    # hold only adds 2–3 frames of latency to the judged moment.  Raise this
    # only for a game that judges on pointer-down.
    hold_ms: float = 1.0
    cooldown_ms: float = 200.0
    entry_margin: float = 0.95
    exit_margin: float = 1.25
    # Half a 60 FPS frame of capture -> detect -> SendInput latency.  This
    # advances the trigger by ~1 sample at high speed without predicting a
    # near-miss reversal (a bar turning around just outside Gold) into a click.
    # Raise towards 16.7–33.3 ms if late strikes still land past the far edge.
    lead_ms: float = 8.3
    # Maximum predicted offset in normalised Gold half-lengths.  The old 0.5
    # cap could only compensate ~18 ms at 500 px/s on an 18 px Gold zone, i.e.
    # less than the pipeline latency it was meant to cancel.
    max_lead_offset: float = 2.0
    rearm_lost_frames: int = 3
    dry_run: bool = False
    auto_fire_up: bool = False
    auto_sell: bool = False

    def __post_init__(self) -> None:
        hold = _require_float(self.hold_ms, "trigger.hold_ms")
        if hold <= 0.0:
            raise ValueError("trigger.hold_ms must be > 0.0")

        cooldown = _require_float(self.cooldown_ms, "trigger.cooldown_ms")
        if cooldown < 0.0:
            raise ValueError("trigger.cooldown_ms must be >= 0.0")

        entry = _require_float(self.entry_margin, "trigger.entry_margin")
        if entry <= 0.0:
            raise ValueError("trigger.entry_margin must be > 0.0")

        exit_m = _require_float(self.exit_margin, "trigger.exit_margin")
        if exit_m < entry:
            raise ValueError("trigger.exit_margin must be >= trigger.entry_margin")

        lead = _require_float(self.lead_ms, "trigger.lead_ms")
        if lead < 0.0:
            raise ValueError("trigger.lead_ms must be >= 0.0")

        max_lead = _require_float(self.max_lead_offset, "trigger.max_lead_offset")
        if max_lead < 0.0:
            raise ValueError("trigger.max_lead_offset must be >= 0.0")

        rearm = _require_int(self.rearm_lost_frames, "trigger.rearm_lost_frames")
        if rearm < 1:
            raise ValueError("trigger.rearm_lost_frames must be >= 1")

        dry_run = _require_bool(self.dry_run, "trigger.dry_run")
        _require_bool(self.auto_fire_up, "trigger.auto_fire_up")
        _require_bool(self.auto_sell, "trigger.auto_sell")

        object.__setattr__(self, "hold_ms", hold)
        object.__setattr__(self, "cooldown_ms", cooldown)
        object.__setattr__(self, "entry_margin", entry)
        object.__setattr__(self, "exit_margin", exit_m)
        object.__setattr__(self, "lead_ms", lead)
        object.__setattr__(self, "max_lead_offset", max_lead)
        object.__setattr__(self, "rearm_lost_frames", rearm)
        object.__setattr__(self, "dry_run", dry_run)


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Screen capture backend configuration."""

    backend: CaptureBackendName = "auto"
    target_fps: int = 60
    monitor_idx: int = 0
    crop_to_window: bool = True

    def __post_init__(self) -> None:
        backend = _require_str(self.backend, "capture.backend", non_empty=True)
        if backend not in VALID_CAPTURE_BACKENDS:
            raise ValueError(
                f"capture.backend must be one of {VALID_CAPTURE_BACKENDS}, got {backend!r}"
            )
        target_fps = _require_int(self.target_fps, "capture.target_fps")
        if target_fps <= 0:
            raise ValueError("capture.target_fps must be > 0")

        monitor_idx = _require_int(self.monitor_idx, "capture.monitor_idx")
        if monitor_idx < 0:
            raise ValueError("capture.monitor_idx must be >= 0")

        crop = _require_bool(self.crop_to_window, "capture.crop_to_window")

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "target_fps", target_fps)
        object.__setattr__(self, "monitor_idx", monitor_idx)
        object.__setattr__(self, "crop_to_window", crop)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Target window matching and focus-guard configuration."""

    title: str = "VRChat"
    class_name: str = "UnityWndClass"
    require_foreground: bool = True
    rect_refresh_s: float = 1.0

    def __post_init__(self) -> None:
        title = _require_str(self.title, "window.title", non_empty=True)
        class_name = _require_str(self.class_name, "window.class_name")
        require_fg = _require_bool(self.require_foreground, "window.require_foreground")
        refresh_s = _require_float(self.rect_refresh_s, "window.rect_refresh_s")
        if refresh_s <= 0.0:
            raise ValueError("window.rect_refresh_s must be > 0.0")

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "require_foreground", require_fg)
        object.__setattr__(self, "rect_refresh_s", refresh_s)


_FLAT_ALIASES: dict[str, tuple[str, ...]] = {
    "capture_backend": ("capture", "backend"),
    "target_fps": ("capture", "target_fps"),
    "monitor_idx": ("capture", "monitor_idx"),
    "crop_to_window": ("capture", "crop_to_window"),
    "window_title": ("window", "title"),
    "window_class_name": ("window", "class_name"),
    "require_foreground": ("window", "require_foreground"),
    "rect_refresh_s": ("window", "rect_refresh_s"),
    "hold_ms": ("trigger", "hold_ms"),
    "cooldown_ms": ("trigger", "cooldown_ms"),
    "entry_margin": ("trigger", "entry_margin"),
    "exit_margin": ("trigger", "exit_margin"),
    "lead_ms": ("trigger", "lead_ms"),
    "max_lead_offset": ("trigger", "max_lead_offset"),
    "rearm_lost_frames": ("trigger", "rearm_lost_frames"),
    "dry_run": ("trigger", "dry_run"),
    "auto_fire_up": ("trigger", "auto_fire_up"),
    "auto_sell": ("trigger", "auto_sell"),
}

_SECTION_KEYS: dict[str, set[str]] = {
    "capture": {"backend", "target_fps", "monitor_idx", "crop_to_window"},
    "window": {"title", "class_name", "require_foreground", "rect_refresh_s"},
    "trigger": {
        "hold_ms",
        "cooldown_ms",
        "entry_margin",
        "exit_margin",
        "lead_ms",
        "max_lead_offset",
        "rearm_lost_frames",
        "dry_run",
        "auto_fire_up",
        "auto_sell",
    },
    "geometry": {
        "reference_height",
        "min_gold_width_px",
        "max_gold_width_px",
        "min_gold_height_px",
        "max_gold_height_px",
        "min_gold_fill_ratio",
        "min_silver_neighbor_ratio",
        "morph_bridge_kernel_h",
        "red_min_width_ratio",
        "red_max_thickness_ratio",
        "red_max_axis_offset_ratio",
        "acquire_downscale",
        "roi_padding_px",
        "max_lost_frames",
    },
}

_HSV_SUB_KEYS: set[str] = {"gold", "silver", "red_low", "red_high"}
_HSV_RANGE_KEYS: set[str] = {"h_min", "h_max", "s_min", "s_max", "v_min", "v_max"}


def _set_nested_path(base: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if len(path) == 2:
        section, field_name = path
        if section == "hsv":
            if field_name not in _HSV_SUB_KEYS:
                raise ValueError(f"Unknown config key: {section}.{field_name}")
            if not isinstance(value, Mapping):
                raise ValueError(f"hsv.{field_name} must be a mapping")
            _merge_hsv_range(base["hsv"][field_name], value, prefix=f"hsv.{field_name}")
            return
        if section not in _SECTION_KEYS or field_name not in _SECTION_KEYS[section]:
            raise ValueError(f"Unknown config key: {section}.{field_name}")
        base[section][field_name] = value
        return

    if len(path) == 3 and path[0] == "hsv":
        _, sub_key, range_key = path
        if sub_key not in _HSV_SUB_KEYS or range_key not in _HSV_RANGE_KEYS:
            raise ValueError(f"Unknown config key: {'.'.join(path)}")
        base["hsv"][sub_key][range_key] = value
        return

    raise ValueError(f"Unknown config key: {'.'.join(path)}")


def _merge_hsv_range(target: dict[str, Any], updates: Mapping[str, Any], *, prefix: str) -> None:
    for k, v in updates.items():
        if not isinstance(k, str) or k not in _HSV_RANGE_KEYS:
            raise ValueError(f"Unknown config key: {prefix}.{k}")
        target[k] = v


def _merge_config_dict(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if not isinstance(key, str):
            raise ValueError(f"Config keys must be strings, got {type(key).__name__}")

        if "." in key:
            parts = tuple(key.split("."))
            if any(not p for p in parts):
                raise ValueError(f"Invalid dotted config key: {key!r}")
            _set_nested_path(base, parts, value)
            continue

        if key in _FLAT_ALIASES:
            _set_nested_path(base, _FLAT_ALIASES[key], value)
            continue

        if key == "hsv":
            if not isinstance(value, Mapping):
                raise ValueError("hsv must be a mapping")
            for sub_key, sub_val in value.items():
                if not isinstance(sub_key, str) or sub_key not in _HSV_SUB_KEYS:
                    raise ValueError(f"Unknown config key: hsv.{sub_key}")
                if not isinstance(sub_val, Mapping):
                    raise ValueError(f"hsv.{sub_key} must be a mapping")
                _merge_hsv_range(base["hsv"][sub_key], sub_val, prefix=f"hsv.{sub_key}")
            continue

        if key in _SECTION_KEYS:
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping")
            for sub_key, sub_val in value.items():
                if not isinstance(sub_key, str) or sub_key not in _SECTION_KEYS[key]:
                    raise ValueError(f"Unknown config key: {key}.{sub_key}")
                base[key][sub_key] = sub_val
            continue

        raise ValueError(f"Unknown config key: {key}")

    return base


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration for vrc_smith_auto."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    hsv: HSVConfig = field(default_factory=HSVConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.capture, CaptureConfig):
            raise ValueError("capture must be a CaptureConfig instance")
        if not isinstance(self.window, WindowConfig):
            raise ValueError("window must be a WindowConfig instance")
        if not isinstance(self.trigger, TriggerConfig):
            raise ValueError("trigger must be a TriggerConfig instance")
        if not isinstance(self.hsv, HSVConfig):
            raise ValueError("hsv must be an HSVConfig instance")
        if not isinstance(self.geometry, GeometryConfig):
            raise ValueError("geometry must be a GeometryConfig instance")

    # Flat convenience accessors matching PROJECT.md Interface Contracts
    @property
    def capture_backend(self) -> CaptureBackendName:
        return self.capture.backend

    @property
    def window_title(self) -> str:
        return self.window.title

    @property
    def require_foreground(self) -> bool:
        return self.window.require_foreground

    @property
    def hold_ms(self) -> float:
        return self.trigger.hold_ms

    @property
    def cooldown_ms(self) -> float:
        return self.trigger.cooldown_ms

    @property
    def entry_margin(self) -> float:
        return self.trigger.entry_margin

    @property
    def exit_margin(self) -> float:
        return self.trigger.exit_margin

    @property
    def lead_ms(self) -> float:
        return self.trigger.lead_ms

    @property
    def dry_run(self) -> bool:
        return self.trigger.dry_run

    @property
    def auto_fire_up(self) -> bool:
        return self.trigger.auto_fire_up

    @property
    def auto_sell(self) -> bool:
        return self.trigger.auto_sell

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> AppConfig:
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
        merged = _merge_config_dict(cls().to_dict(), data)
        return cls(
            capture=CaptureConfig(**merged["capture"]),
            window=WindowConfig(**merged["window"]),
            trigger=TriggerConfig(**merged["trigger"]),
            hsv=HSVConfig(
                gold=HSVRange(**merged["hsv"]["gold"]),
                silver=HSVRange(**merged["hsv"]["silver"]),
                red_low=HSVRange(**merged["hsv"]["red_low"]),
                red_high=HSVRange(**merged["hsv"]["red_high"]),
            ),
            geometry=GeometryConfig(**merged["geometry"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> AppConfig:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config in {p}: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError(f"JSON config in {p} must be a mapping")
        return cls.from_dict(loaded)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        try:
            loaded = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config in {p}: {exc}") from exc
        if loaded is None:
            return cls()
        if not isinstance(loaded, Mapping):
            raise ValueError(f"YAML config in {p} must be a mapping")
        return cls.from_dict(loaded)

    @classmethod
    def from_file(cls, path: str | Path) -> AppConfig:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(p)
        if suffix == ".json":
            return cls.from_json(p)
        raise ValueError(f"Unsupported config file extension {suffix!r} (expected .yaml, .yml, or .json)")

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AppConfig:
        merged = self.to_dict()
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise ValueError(f"overrides must be a mapping, got {type(overrides).__name__}")
            _merge_config_dict(merged, overrides)
        if kwargs:
            _merge_config_dict(merged, kwargs)
        return AppConfig(
            capture=CaptureConfig(**merged["capture"]),
            window=WindowConfig(**merged["window"]),
            trigger=TriggerConfig(**merged["trigger"]),
            hsv=HSVConfig(
                gold=HSVRange(**merged["hsv"]["gold"]),
                silver=HSVRange(**merged["hsv"]["silver"]),
                red_low=HSVRange(**merged["hsv"]["red_low"]),
                red_high=HSVRange(**merged["hsv"]["red_high"]),
            ),
            geometry=GeometryConfig(**merged["geometry"]),
        )
