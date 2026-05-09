"""Persistent data models and session/settings serialisation."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_SESSION_FILTER   = "Luminos-Sitzung (*.luminos)"
_SESSION_VERSION  = 1
_APP_SETTINGS_PATH    = Path.home() / ".config" / "luminos" / "settings.json"
_RECENT_SESSIONS_PATH = Path.home() / ".config" / "luminos" / "recent_sessions.json"
_RECENT_MAX = 10


# ── Per-image editing state ───────────────────────────────────────────────────


@dataclasses.dataclass
class _EditSettings:
    """All per-image editing slider values and curve control points."""
    exposure: int = 0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    black_point: int = 0
    white_point: int = 100
    wb_temp: int = 6500
    wb_tint: int = 0
    vibrance: int = 0
    saturation: int = 100
    sharpening: int = 0
    noise_reduction: int = 0
    vignette: int = 0
    grain: int = 0
    angle: int = 0
    split_shadow_hue: int = 0
    split_shadow_sat: int = 0
    split_highlight_hue: int = 0
    split_highlight_sat: int = 0
    split_balance: int = 0
    curve_points: dict = dataclasses.field(
        default_factory=lambda: {
            ch: [(0.0, 0.0), (1.0, 1.0)] for ch in ("master", "r", "g", "b")
        }
    )


@dataclasses.dataclass
class _ImageEntry:
    """All per-image data held in the main thread."""
    path: str
    # Full-res float32 — loaded on activation, released on deactivation.
    raw_image: np.ndarray | None = None
    # Downscaled float32 for interactive processing (kept for all loaded images).
    raw_preview: np.ndarray | None = None
    # Cached inversion result (kept for all loaded images).
    inverted_preview: np.ndarray | None = None
    orange_mask: tuple[float, float, float] | None = None
    # Raw EXIF bytes read from the source file at import time; embedded on JPEG export.
    exif_bytes: bytes | None = None
    # Film process: "c41" (colour with orange-mask compensation) or "bw" (black-and-white).
    film_type: str = "c41"
    # Accumulated crop in original-image normalised coords (x0, y0, x1, y1) ∈ [0, 1].
    crop_region_norm: tuple[float, float, float, float] | None = None
    # Cumulative 90° CCW rotation steps (0–3) applied to raw_image at export time.
    rotation_steps: int = 0
    # Crop/rotation undo stack.
    crop_history: list = dataclasses.field(default_factory=list)
    # Per-image editing state (sliders + curve), saved on image switch.
    settings: _EditSettings = dataclasses.field(default_factory=_EditSettings)


# ── Application preferences ───────────────────────────────────────────────────


@dataclasses.dataclass
class _AppSettings:
    """Persistent user preferences, stored in ~/.config/luminos/settings.json."""
    # Import
    default_film_type: str = "c41"
    preview_long_edge: int = 1500
    # Export
    default_export_format: str = "tiff"
    default_jpeg_quality: int = 95
    default_output_dir: str = ""
    export_suffix: str = "_positive"
    # UI
    preview_debounce_ms: int = 50
    histogram_log_scale: bool = False
    recent_sessions_max: int = 10


def _load_app_settings() -> _AppSettings:
    try:
        with open(_APP_SETTINGS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        s = _AppSettings()
        for field in dataclasses.fields(s):
            if field.name in d:
                setattr(s, field.name, d[field.name])
        return s
    except (OSError, json.JSONDecodeError, TypeError):
        return _AppSettings()


def _save_app_settings(s: _AppSettings) -> None:
    try:
        _APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        d = dataclasses.asdict(s)
        with open(_APP_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


# ── Edit-settings serialisation ───────────────────────────────────────────────


def _settings_to_dict(s: _EditSettings) -> dict:
    return {
        "exposure": s.exposure,
        "contrast": s.contrast,
        "highlights": s.highlights,
        "shadows": s.shadows,
        "black_point": s.black_point,
        "white_point": s.white_point,
        "wb_temp": s.wb_temp,
        "wb_tint": s.wb_tint,
        "vibrance": s.vibrance,
        "saturation": s.saturation,
        "sharpening": s.sharpening,
        "noise_reduction": s.noise_reduction,
        "vignette": s.vignette,
        "grain": s.grain,
        "angle": s.angle,
        "split_shadow_hue": s.split_shadow_hue,
        "split_shadow_sat": s.split_shadow_sat,
        "split_highlight_hue": s.split_highlight_hue,
        "split_highlight_sat": s.split_highlight_sat,
        "split_balance": s.split_balance,
        "curve_points": {
            ch: [list(p) for p in pts]
            for ch, pts in s.curve_points.items()
        },
    }


def _settings_from_dict(d: dict) -> _EditSettings:
    defaults = _EditSettings()
    curve_raw = d.get("curve_points", {})
    curve_points = {
        ch: [tuple(p) for p in pts]
        for ch, pts in curve_raw.items()
    } if curve_raw else defaults.curve_points
    return _EditSettings(
        exposure=d.get("exposure", defaults.exposure),
        contrast=d.get("contrast", defaults.contrast),
        highlights=d.get("highlights", defaults.highlights),
        shadows=d.get("shadows", defaults.shadows),
        black_point=d.get("black_point", defaults.black_point),
        white_point=d.get("white_point", defaults.white_point),
        wb_temp=d.get("wb_temp", defaults.wb_temp),
        wb_tint=d.get("wb_tint", defaults.wb_tint),
        vibrance=d.get("vibrance", defaults.vibrance),
        saturation=d.get("saturation", defaults.saturation),
        sharpening=d.get("sharpening", defaults.sharpening),
        noise_reduction=d.get("noise_reduction", defaults.noise_reduction),
        vignette=d.get("vignette", defaults.vignette),
        grain=d.get("grain", defaults.grain),
        angle=d.get("angle", defaults.angle),
        split_shadow_hue=d.get("split_shadow_hue", defaults.split_shadow_hue),
        split_shadow_sat=d.get("split_shadow_sat", defaults.split_shadow_sat),
        split_highlight_hue=d.get("split_highlight_hue", defaults.split_highlight_hue),
        split_highlight_sat=d.get("split_highlight_sat", defaults.split_highlight_sat),
        split_balance=d.get("split_balance", defaults.split_balance),
        curve_points=curve_points,
    )


# ── Recent-sessions persistence ───────────────────────────────────────────────


def _load_recent_sessions() -> list[str]:
    try:
        with open(_RECENT_SESSIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [p for p in data if os.path.exists(p)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def _add_recent_session(path: str) -> None:
    recents = _load_recent_sessions()
    path = str(Path(path).resolve())
    recents = [p for p in recents if p != path]
    recents.insert(0, path)
    recents = recents[:_RECENT_MAX]
    try:
        _RECENT_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_RECENT_SESSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(recents, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


# ── Session serialisation / IO ────────────────────────────────────────────────


class SessionManager:
    """
    Handles session file serialisation, deserialisation, and recent-session tracking.

    Keeps all JSON/file-system logic out of MainWindow so the UI only deals with
    plain Python objects.
    """

    def serialize(
        self,
        active_path: str | None,
        entries: dict,
    ) -> dict:
        """Build a JSON-serialisable dict from the current set of image entries."""
        images = []
        for entry_path, entry in entries.items():
            images.append({
                "path": entry_path,
                "film_type": entry.film_type,
                "crop_region_norm": list(entry.crop_region_norm) if entry.crop_region_norm else None,
                "rotation_steps": entry.rotation_steps,
                "settings": _settings_to_dict(entry.settings),
            })
        return {
            "version": _SESSION_VERSION,
            "active_path": active_path,
            "images": images,
        }

    def save_file(
        self,
        path: str,
        active_path: str | None,
        entries: dict,
    ) -> str | None:
        """
        Write the session to *path*.

        Returns an error message string on failure, or ``None`` on success.
        After a successful save the path is added to the recent-sessions list.
        """
        data = self.serialize(active_path, entries)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            return str(exc)
        _add_recent_session(path)
        return None

    def load_file(self, path: str) -> tuple[dict | None, str | None]:
        """
        Read and parse a session file.

        Returns ``(data_dict, None)`` on success or ``(None, error_message)`` on failure.
        After a successful load the path is added to the recent-sessions list.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            return None, str(exc)
        _add_recent_session(path)
        return data, None

    def apply_entry_data(self, entry: "_ImageEntry", saved: dict) -> None:
        """Apply deserialised per-image data onto an existing ``_ImageEntry``."""
        entry.film_type = saved.get("film_type", "c41")
        entry.crop_region_norm = (
            tuple(saved["crop_region_norm"]) if saved.get("crop_region_norm") else None
        )
        entry.rotation_steps = saved.get("rotation_steps", 0)
        entry.settings = _settings_from_dict(saved.get("settings", {}))

    # ── Recent sessions ───────────────────────────────────────────────────────

    def load_recent(self) -> list[str]:
        return _load_recent_sessions()

    def add_recent(self, path: str) -> None:
        _add_recent_session(path)

    def clear_recent(self) -> None:
        try:
            _RECENT_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_RECENT_SESSIONS_PATH, "w", encoding="utf-8") as f:
                json.dump([], f)
        except OSError:
            pass
