"""C-41 negative film inversion with orange mask compensation."""

import logging
import math

import numpy as np

log = logging.getLogger(__name__)


def detect_orange_mask(image: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate the C-41 orange mask from the 97.5–99.5 percentile pixels.

    Why not the top 0.5%:
      The absolute brightest pixels are the film rebate (base carrier, no
      emulsion), which are too bright and would skew the mask estimate upward.
      The 97.5–99.5 range reliably lands on the unexposed emulsion edge,
      which holds the pure mask density.

    Args:
        image: float32 (H, W, 3) array in range 0–1.

    Returns:
        (r, g, b) mask values as float.
    """
    brightness = image.mean(axis=2).ravel()
    lo_thresh = np.percentile(brightness, 97.5)
    hi_thresh = np.percentile(brightness, 99.5)

    mask_pixels = image.reshape(-1, 3)[
        (brightness >= lo_thresh) & (brightness <= hi_thresh)
    ]

    if mask_pixels.size == 0:
        log.warning(
            "Orange mask detection: no pixels found in 97.5–99.5 percentile range "
            "(flat or very low-contrast scan?); falling back to whole-image median"
        )
        return (
            float(np.median(image[:, :, 0])),
            float(np.median(image[:, :, 1])),
            float(np.median(image[:, :, 2])),
        )

    r = float(np.median(mask_pixels[:, 0]))
    g = float(np.median(mask_pixels[:, 1]))
    b = float(np.median(mask_pixels[:, 2]))

    log.debug("Orange mask detected: R=%.4f G=%.4f B=%.4f", r, g, b)
    return r, g, b


def invert_c41(
    image: np.ndarray,
    mask: tuple[float, float, float] | None = None,
    clip_percentile: float = 0.1,
) -> np.ndarray:
    """
    Invert a C-41 color negative scan (float32, range 0–1).

    Steps:
      1. Detect orange mask from bright film-base pixels (or use provided values).
      2. Divide each channel by its mask value — removes the orange cast.
      3. Clip to [0, 1].
      4. Invert: 1 − normalised.
      5. Auto white balance: stretch each channel independently to full range.
         Corrects any residual colour imbalance after mask removal.
      6. Clip clip_percentile at both ends and re-stretch to 0–1.
         Sets black and white points robustly, discarding extreme outliers.

    Args:
        image:           float32 (H, W, 3) array in range 0–1.
        mask:            (r, g, b) mask values; None → auto-detect.
        clip_percentile: percent of pixels clipped at each histogram end
                         (default 0.1 %).

    Returns:
        float32 (H, W, 3) array in range 0–1.
    """
    if mask is None:
        mask = detect_orange_mask(image)

    mask_r, mask_g, mask_b = mask

    # Divide per channel to normalise the orange cast.
    work = image.astype(np.float32, copy=True)
    work[:, :, 0] /= mask_r if mask_r > 0.0 else 1.0
    work[:, :, 1] /= mask_g if mask_g > 0.0 else 1.0
    work[:, :, 2] /= mask_b if mask_b > 0.0 else 1.0
    np.clip(work, 0.0, 1.0, out=work)

    # Invert: dark shadow areas on the negative become bright highlights.
    np.subtract(1.0, work, out=work)

    # Auto white balance — stretch each channel independently to 0–1.
    # Keeps relative tone curve intact; only removes per-channel offset/scale.
    awb_lo = work.min(axis=(0, 1))
    awb_hi = work.max(axis=(0, 1))
    awb_span = awb_hi - awb_lo
    awb_span[awb_span == 0.0] = 1.0
    work = (work - awb_lo) / awb_span

    # Black and white point: clip and re-stretch.
    p_lo = np.percentile(work, clip_percentile, axis=(0, 1))
    p_hi = np.percentile(work, 100.0 - clip_percentile, axis=(0, 1))
    np.clip(work, p_lo, p_hi, out=work)
    span = p_hi - p_lo
    span[span == 0.0] = 1.0
    work = (work - p_lo) / span

    np.clip(work, 0.0, 1.0, out=work)
    return work


def invert_bw(
    image: np.ndarray,
    clip_percentile: float = 0.1,
) -> np.ndarray:
    """
    Invert a black-and-white negative scan (float32, range 0–1).

    No orange-mask compensation — B&W negatives carry no colour cast.
    Converts to luminance, inverts, auto-stretches to full range, and
    returns float32 RGB with equal R=G=B channels so the rest of the
    editing pipeline (curves, sharpening, vignette, …) works unchanged.

    Args:
        image:           float32 (H, W, 3) array in range 0–1.
        clip_percentile: percent of pixels clipped at each histogram end
                         during auto-stretch (default 0.1 %).

    Returns:
        float32 (H, W, 3) array in range 0–1 with R == G == B.
    """
    luma = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    inverted = 1.0 - luma

    p_lo = float(np.percentile(inverted, clip_percentile))
    p_hi = float(np.percentile(inverted, 100.0 - clip_percentile))
    if p_hi - p_lo < 1e-6:
        p_lo, p_hi = 0.0, 1.0

    stretched = np.clip(
        (inverted - p_lo) / (p_hi - p_lo), 0.0, 1.0
    ).astype(np.float32)

    log.debug("B&W inversion: p_lo=%.4f p_hi=%.4f", p_lo, p_hi)
    return np.stack([stretched, stretched, stretched], axis=2)


def suggest_auto_levels(
    image: np.ndarray,
    shadow_clip: float = 0.5,
    highlight_clip: float = 0.5,
) -> tuple[int, int, int]:
    """
    Suggest Belichtung, Schwarzpunkt, and Weißpunkt slider values from the image histogram.

    Analyses the luminance distribution of a float32 image (typically the
    WB-applied inverted preview) and returns slider integers that bring the
    tonal range into a well-exposed state.

    Pipeline context: the editing chain is WB → Exposure → Levels, so exposure
    is derived first (centering the mid-tone), then black / white points are
    computed on the simulated exposure-adjusted luma.

    Args:
        image:          float32 (H, W, 3) in range 0–1.
        shadow_clip:    percentile of dark pixels to clip, default 0.5 %.
        highlight_clip: percentile of bright pixels to clip, default 0.5 %.

    Returns:
        (exposure_int, black_int, white_int) ready for QSlider.setValue():
        - exposure_int: in [-30, +30]  (divide by 10 for EV stops)
        - black_int:    in [0, 49]
        - white_int:    in [51, 100]
    """
    luma = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).ravel()

    p_lo = float(np.percentile(luma, shadow_clip))
    p_hi = float(np.percentile(luma, 100.0 - highlight_clip))

    # Compute exposure so the mid-point of the content range lands at 0.5.
    midpoint = (p_lo + p_hi) / 2.0
    if midpoint > 1e-3:
        ev = math.log2(0.5 / midpoint)
        ev = max(-3.0, min(3.0, ev))
    else:
        ev = 0.0

    # Simulate the exposure gain and map the content endpoints to slider values.
    gain = 2.0 ** ev
    adj_lo = float(np.clip(p_lo * gain, 0.0, 0.49))
    adj_hi = float(np.clip(p_hi * gain, 0.51, 1.0))

    exposure_int = int(round(ev * 10.0))
    black_int = max(0, min(49, int(round(adj_lo * 100.0))))
    white_int = max(51, min(100, int(round(adj_hi * 100.0))))

    if black_int >= white_int:
        black_int = max(0, white_int - 5)

    log.debug(
        "Auto-levels: EV=%+.2f  black=%d  white=%d  (p_lo=%.3f p_hi=%.3f)",
        ev, black_int, white_int, p_lo, p_hi,
    )
    return exposure_int, black_int, white_int


def invert(
    image: np.ndarray,
    mask: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Invert a linear float32 negative image (negative → positive).

    Applies C-41 orange mask compensation automatically.
    Pass mask=(r, g, b) to use a pre-computed mask instead of auto-detection.
    """
    return invert_c41(image, mask=mask)
