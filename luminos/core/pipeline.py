"""
Core conversion pipeline.

Orchestrates: load → invert → white balance → exposure → levels →
              saturation → tone curve → sharpening → export.
All steps operate on float32 (H, W, 3) arrays in range 0–1.
"""

from __future__ import annotations

import os

import numpy as np

from luminos.io.raw_loader import load_raw
from luminos.io.tiff_loader import load_tiff
from luminos.io.export import save_tiff
from luminos.core.inversion import invert, invert_bw
from luminos.core.color import (
    apply_white_balance,
    apply_exposure,
    apply_levels,
    apply_saturation,
    apply_sharpening,
    apply_contrast,
    apply_highlights,
    apply_shadows,
    apply_vibrance,
    apply_noise_reduction,
    apply_vignette,
    apply_grain,
    apply_split_toning,
)


def load_image(path: str) -> np.ndarray:
    """Load RAW or TIFF into a float32 (H, W, 3) array."""
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".orf", ".raf", ".rw2"}
    if ext in raw_exts:
        return load_raw(path)
    return load_tiff(path)


def process(
    image: np.ndarray,
    exposure_stops: float = 0.0,
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: tuple[float, float, float] | None = None,
    black_point: float = 0.0,
    white_point: float = 1.0,
    saturation: float = 1.0,
    curve_luts: dict | None = None,
    sharpening: float = 0.0,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    vibrance: float = 0.0,
    noise_reduction: float = 0.0,
    vignette: float = 0.0,
    grain: float = 0.0,
    film_type: str = "c41",
    split_toning: tuple[float, float, float, float, float] | None = None,
) -> np.ndarray:
    """
    Run the full conversion pipeline on a loaded float32 image.

    Steps:
      1. Invert (C-41 orange mask compensation)
      2. White balance
      3. Exposure (EV stops)
      4. Contrast (S-curve)
      5. Highlights / Shadows (luminance-masked adjustments)
      6. Input levels (black / white point)
      7. Vibrance (targeted saturation)
      8. Saturation
      9. Tone curves (master + per-channel)
     10. Sharpening (unsharp mask)

    Args:
        image:          float32 (H, W, 3) array in range 0–1.
        exposure_stops: EV adjustment.
        white_balance:  Per-channel (R, G, B) multipliers.
        mask:           Pre-computed orange mask values; None → auto-detect.
                        Ignored when film_type == "bw".
        film_type:      "c41" (default) for colour negatives with orange-mask
                        compensation; "bw" for black-and-white negatives.
        black_point:    Input black point (0.0 = no change).
        white_point:    Input white point (1.0 = no change).
        saturation:     Saturation factor (1.0 = no change, 0 = greyscale).
        curve_luts:     Dict with keys 'master','r','g','b', each a 256-entry
                        float32 LUT; None = skip curve step.
        sharpening:      Unsharp mask amount (0 = off, 1 = strong).
        contrast:        S-curve contrast in [-1, 1]. 0 = no change.
        highlights:      Highlight adjustment in [-1, 1]. 0 = no change.
        shadows:         Shadow adjustment in [-1, 1]. 0 = no change.
        vibrance:        Targeted saturation in [-1, 1]. 0 = no change.
        noise_reduction: Gaussian blur denoising (0 = off, 1 = strong).
        vignette:        Radial vignette in [-1, 1]. >0 darkens edges.
        grain:           Film grain strength (0 = off, 1 = strong).
    """
    if film_type == "bw":
        result = invert_bw(image)
    else:
        result = invert(image, mask=mask)
    result = apply_white_balance(result, white_balance)
    result = apply_exposure(result, exposure_stops)

    if abs(contrast) > 1e-4:
        result = apply_contrast(result, contrast)
    if abs(highlights) > 1e-4:
        result = apply_highlights(result, highlights)
    if abs(shadows) > 1e-4:
        result = apply_shadows(result, shadows)

    if black_point > 0.0 or white_point < 1.0:
        result = apply_levels(result, black_point, white_point)

    if abs(vibrance) > 1e-4:
        result = apply_vibrance(result, vibrance)
    if abs(saturation - 1.0) > 1e-4:
        result = apply_saturation(result, saturation)

    if split_toning is not None:
        result = apply_split_toning(result, *split_toning)

    if curve_luts is not None:
        from luminos.core.curves import apply_curves, all_identity
        if not all_identity(curve_luts):
            result = apply_curves(result, curve_luts)

    if noise_reduction > 1e-4:
        result = apply_noise_reduction(result, noise_reduction)

    if sharpening > 1e-4:
        result = apply_sharpening(result, sharpening)

    if abs(vignette) > 1e-4:
        result = apply_vignette(result, vignette)

    if grain > 1e-4:
        result = apply_grain(result, grain)

    return result


def convert_file(
    input_path: str,
    output_path: str,
    exposure_stops: float = 0.0,
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Load, process, and save a single image. Convenience wrapper for CLI use."""
    image = load_image(input_path)
    result = process(image, exposure_stops=exposure_stops, white_balance=white_balance)
    save_tiff(result, output_path)
    print(f"Saved: {output_path}")
