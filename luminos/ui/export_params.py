"""Build export processing parameters from persisted edit settings."""

from __future__ import annotations

import numpy as np

from luminos.core.color import temp_tint_to_rgb_multipliers
from luminos.core.curves import make_lut
from luminos.ui.session import _EditSettings, _ImageEntry
from luminos.ui.workers import ProcessingParams


def split_toning_params_from_settings(
    settings: _EditSettings,
) -> tuple[float, float, float, float, float] | None:
    """Return split-toning params for the pipeline, or None when inactive."""
    shadow_sat = settings.split_shadow_sat / 100.0
    highlight_sat = settings.split_highlight_sat / 100.0
    if shadow_sat < 1e-4 and highlight_sat < 1e-4:
        return None
    return (
        float(settings.split_shadow_hue),
        shadow_sat,
        float(settings.split_highlight_hue),
        highlight_sat,
        settings.split_balance / 100.0,
    )


def curve_luts_from_settings(settings: _EditSettings) -> dict[str, np.ndarray]:
    """Build master/R/G/B curve LUTs from saved curve control points."""
    return {
        ch: make_lut(settings.curve_points.get(ch, [(0.0, 0.0), (1.0, 1.0)]))
        for ch in ("master", "r", "g", "b")
    }


def processing_params_from_settings(
    settings: _EditSettings,
    entry: _ImageEntry,
) -> ProcessingParams:
    """Convert persisted per-image edit settings to worker processing params."""
    return ProcessingParams(
        exposure_stops=settings.exposure / 10.0,
        white_balance=temp_tint_to_rgb_multipliers(settings.wb_temp, settings.wb_tint),
        mask=entry.orange_mask,
        black_point=settings.black_point / 100.0,
        white_point=settings.white_point / 100.0,
        saturation=settings.saturation / 100.0,
        curve_luts=curve_luts_from_settings(settings),
        sharpening=settings.sharpening / 100.0,
        angle=settings.angle / 10.0,
        contrast=settings.contrast / 100.0,
        highlights=settings.highlights / 100.0,
        shadows=settings.shadows / 100.0,
        vibrance=settings.vibrance / 100.0,
        noise_reduction=settings.noise_reduction / 100.0,
        vignette=settings.vignette / 100.0,
        grain=settings.grain / 100.0,
        film_type=entry.film_type,
        split_toning=split_toning_params_from_settings(settings),
        exif_bytes=entry.exif_bytes,
    )
