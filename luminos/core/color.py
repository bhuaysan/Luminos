"""Colour correction utilities."""

import colorsys
import math

import numpy as np


# ── Internal helpers ──────────────────────────────────────────────────────────

_LUMA_R = np.float32(0.299)
_LUMA_G = np.float32(0.587)
_LUMA_B = np.float32(0.114)


def _luma(image: np.ndarray) -> np.ndarray:
    """BT.601 luminance, returns float32 (H, W). All three coefficients are
    float32 so no upcasting to float64 occurs."""
    return _LUMA_R * image[:, :, 0] + _LUMA_G * image[:, :, 1] + _LUMA_B * image[:, :, 2]


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separable Gaussian blur on a float32 (H, W, 3) array.

    Uses sliding-window views for fully vectorised row and column passes —
    no Pillow dependency, works on any numpy-supported platform.
    """
    radius = max(1, round(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    ks = len(k)
    out = np.empty_like(image)
    for ch in range(3):
        plane = image[:, :, ch]
        # Horizontal pass (along columns)
        p1 = np.pad(plane, ((0, 0), (radius, radius)), mode="reflect")
        w1 = np.lib.stride_tricks.sliding_window_view(p1, ks, axis=1)  # (H, W, ks)
        h_pass = (w1 * k).sum(axis=2)
        # Vertical pass (along rows)
        p2 = np.pad(h_pass, ((radius, radius), (0, 0)), mode="reflect")
        w2 = np.lib.stride_tricks.sliding_window_view(p2, ks, axis=0)  # (H, W, ks)
        out[:, :, ch] = (w2 * k).sum(axis=2)
    return out


# ── White-balance colour-temperature helpers ──────────────────────────────────


def _kelvin_to_srgb(T: float) -> tuple[float, float, float]:
    """
    CCT (K) → linear sRGB using the Kang et al. (2002) Planckian-locus
    polynomial.  Y is normalised to 1.  Negative primaries are clipped to 1e-6.
    Valid input range: 1667–25000 K.
    """
    T = max(1667.0, min(25000.0, float(T)))
    # xy chromaticity (Kang 2002)
    if T <= 4000.0:
        x = -0.2661239e9/T**3 - 0.2343580e6/T**2 + 0.8776956e3/T + 0.179910
    else:
        x = -3.0258469e9/T**3 + 2.1070379e6/T**2 + 0.2226347e3/T + 0.240390
    if T <= 2222.0:
        y = -1.1063814*x**3 - 1.34811020*x**2 + 2.18555832*x - 0.20219683
    elif T <= 4000.0:
        y = -0.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - 0.16748867
    else:
        y =  3.0817580*x**3 - 5.8733867 *x**2 + 3.75112997*x - 0.37001483
    # xy → XYZ (Y = 1) → linear sRGB (D65 matrix, IEC 61966-2-1)
    X, Z = x / y, (1.0 - x - y) / y
    r = max( 3.2406*X - 1.5372 - 0.4986*Z, 1e-6)
    g = max(-0.9689*X + 1.8758 + 0.0415*Z, 1e-6)
    b = max( 0.0557*X - 0.2040 + 1.0570*Z, 1e-6)
    return r, g, b


# Reference white at 6500 K (neutral point for the temperature slider).
_WB_REF = _kelvin_to_srgb(6500.0)


def temp_tint_to_rgb_multipliers(
    temp_k: float, tint: float
) -> tuple[float, float, float]:
    """
    Convert a colour-temperature / tint pair to per-channel RGB multipliers.

    Interpretation:
      - Higher temperature (warmer, e.g. 3200 K tungsten) → more red, less blue.
      - Positive tint → magenta cast (green reduced).
      - Negative tint → green cast (green boosted).

    Neutral point: temp_k = 6500, tint = 0 → (1.0, 1.0, 1.0).

    Args:
        temp_k: colour temperature in Kelvin (2000–12000).
        tint:   green/magenta tint in [-100, +100].  0 = no tint.

    Returns:
        (r, g, b) float multipliers suitable for apply_white_balance().
    """
    ref_r, ref_g, ref_b = _WB_REF
    tgt_r, tgt_g, tgt_b = _kelvin_to_srgb(temp_k)
    r_mult = tgt_r / ref_r
    b_mult = tgt_b / ref_b
    # Temperature contribution to green (before tint)
    g_mult = tgt_g / ref_g
    # Tint: positive = magenta (reduce G by up to 0.5 EV at ±100)
    g_mult *= math.pow(2.0, -tint / 100.0 * 0.5)
    return r_mult, g_mult, b_mult


def rgb_multipliers_to_temp_tint(
    r_mult: float, g_mult: float, b_mult: float
) -> tuple[int, int]:
    """
    Approximate inverse of temp_tint_to_rgb_multipliers.

    Finds the (temp_k, tint) pair that best reproduces the given RGB
    multipliers.  Uses binary search on the R/B ratio to estimate temperature,
    then derives tint from the residual green shift.

    Args:
        r_mult, g_mult, b_mult: positive RGB multipliers.

    Returns:
        (temp_k, tint) as integers (Kelvin, −100…+100).
    """
    ref_r, ref_g, ref_b = _WB_REF
    rb_target = r_mult / b_mult if b_mult > 1e-6 else 1.0

    # Binary search: as T rises, the r/b ratio falls (cooler = more blue).
    lo, hi = 2000.0, 12000.0
    for _ in range(50):
        mid = (lo + hi) * 0.5
        mr, _, mb = _kelvin_to_srgb(mid)
        mid_ratio = (mr / ref_r) / (mb / ref_b)
        if mid_ratio > rb_target:
            lo = mid   # ratio still too high → need higher T
        else:
            hi = mid   # ratio too low → need lower T
    temp = round((lo + hi) * 0.5)

    # Derive tint from green residual.
    tgt_g = _kelvin_to_srgb(float(temp))[1]
    g_mult_base = tgt_g / ref_g          # g multiplier at found temp, tint=0
    if g_mult_base > 1e-9 and g_mult > 1e-9:
        # g_mult = g_mult_base * 2^(-tint/100 * 0.5)
        tint_raw = -math.log2(g_mult / g_mult_base) * 200.0
    else:
        tint_raw = 0.0
    tint = max(-100, min(100, round(tint_raw)))
    return temp, tint


def apply_white_balance(image: np.ndarray, rgb_multipliers: tuple[float, float, float]) -> np.ndarray:
    """
    Apply per-channel gain (white balance).

    Args:
        image: float32 (H, W, 3) array in range 0–1.
        rgb_multipliers: (r, g, b) gains — e.g. (1.2, 1.0, 0.9).

    Returns:
        Clipped float32 array in range 0–1.
    """
    mults = np.array(rgb_multipliers, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image * mults, 0.0, 1.0)


def apply_exposure(image: np.ndarray, stops: float) -> np.ndarray:
    """
    Apply exposure in EV stops (linear gain = 2^stops).

    Args:
        image: float32 (H, W, 3) array.
        stops: positive = brighter, negative = darker.

    Returns:
        Clipped float32 array.
    """
    gain = 2.0 ** stops
    return np.clip(image * gain, 0.0, 1.0)


def apply_levels(image: np.ndarray, black: float, white: float) -> np.ndarray:
    """
    Remap input levels [black, white] → [0, 1] (input black / white point).

    Equivalent to dragging the black and white input sliders in a levels tool.
    Values below *black* are clipped to 0; values above *white* to 1.

    Args:
        image: float32 (H, W, 3) in range 0–1.
        black: input value mapped to 0 (0.0 = no change).
        white: input value mapped to 1 (1.0 = no change).

    Returns:
        Clipped float32 array.
    """
    if white <= black:
        return image
    return np.clip((image - black) / (white - black), 0.0, 1.0).astype(np.float32)


def apply_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust colour saturation.

    Uses a luma-weighted interpolation:
      - factor = 0 → fully desaturated (greyscale)
      - factor = 1 → no change
      - factor > 1 → boosted saturation (factor = 2 doubles the colour distance
                     from grey)

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        factor: saturation multiplier (≥ 0).

    Returns:
        float32 array, values may slightly exceed [0, 1] and are clipped.
    """
    if abs(factor - 1.0) < 1e-4:
        return image
    luma = _luma(image)[:, :, np.newaxis]
    return np.clip(luma + factor * (image - luma), 0.0, 1.0).astype(np.float32)


def apply_contrast(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply contrast using a smooth S-curve anchored at black, mid-grey, and white.

    Uses f(x) = x + amount * x*(1-x)*(x-0.5)*4, which always maps 0→0 and 1→1
    so no clipping occurs at the endpoints.

    amount > 0: S-curve (darks darker, brights brighter).
    amount < 0: inverse S-curve (reduced contrast / washed-out look).

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: contrast in [-1, 1]. 0 = no change.

    Returns:
        Clipped float32 array.
    """
    if abs(amount) < 1e-4:
        return image
    result = image + amount * image * (1.0 - image) * (image - 0.5) * 4.0
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_highlights(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Compress or expand the highlight region.

    Applies a luminance-weighted adjustment that targets bright pixels
    (luminance > 0.5) with a smooth quadratic roll-off toward midtones.

    amount > 0: boosts highlights.
    amount < 0: recovers / compresses highlights.

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: highlight adjustment in [-1, 1].

    Returns:
        Clipped float32 array.
    """
    if abs(amount) < 1e-4:
        return image
    luma = _luma(image)
    mask = np.clip((luma - 0.5) * 2.0, 0.0, 1.0) ** 2
    return np.clip(image + amount * 0.5 * mask[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)


def apply_shadows(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Lift or crush the shadow region.

    Applies a luminance-weighted adjustment that targets dark pixels
    (luminance < 0.5) with a smooth quadratic roll-off toward midtones.

    amount > 0: lifts shadows (brighter).
    amount < 0: crushes shadows (darker).

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: shadow adjustment in [-1, 1].

    Returns:
        Clipped float32 array.
    """
    if abs(amount) < 1e-4:
        return image
    luma = _luma(image)
    mask = np.clip((0.5 - luma) * 2.0, 0.0, 1.0) ** 2
    return np.clip(image + amount * 0.5 * mask[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)


def apply_vibrance(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Adjust vibrance — targeted saturation that boosts less-saturated colours more.

    Unlike ``apply_saturation``, vibrance weights the boost inversely to the
    per-pixel HSV saturation, so already-vivid colours are affected less and
    skin-tone clipping is reduced.

    amount > 0: increases vibrance.
    amount < 0: reduces vibrance (desaturates low-saturation areas first).

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: vibrance in [-1, 1]. 0 = no change.

    Returns:
        Clipped float32 array.
    """
    if abs(amount) < 1e-4:
        return image
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.divide(max_c - min_c, max_c, out=np.zeros_like(max_c), where=max_c > 1e-6)
    weight = (1.0 - sat)[:, :, np.newaxis]
    luma = _luma(image)[:, :, np.newaxis]
    result = luma + (1.0 + amount * weight) * (image - luma)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_sharpening(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Unsharp mask sharpening.

    Blurs the image with a Gaussian (radius ≈ 1.5 px) and adds back the
    high-frequency residual scaled by *amount*.

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: sharpening strength (0 = no effect, 1 = strong).

    Returns:
        Clipped float32 array.
    """
    if amount < 1e-4:
        return image
    blurred = _gaussian_blur(image, sigma=1.5)
    return np.clip(image + amount * (image - blurred), 0.0, 1.0).astype(np.float32)


def apply_noise_reduction(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Gaussian blur noise reduction.

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: reduction strength (0 = off, 1 = strong; Gaussian sigma up to 3 px).

    Returns:
        float32 array in range 0–1.
    """
    if amount < 1e-4:
        return image
    return _gaussian_blur(image, sigma=amount * 3.0)


def apply_vignette(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Radial vignette effect.

    amount > 0: darkens edges (classic vignette).
    amount < 0: brightens edges (reverse vignette).

    Uses a normalised distance-from-centre mask with a smooth quadratic
    roll-off so the effect fades gradually from the edges.

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: vignette strength in [-1, 1]. 0 = no change.

    Returns:
        Clipped float32 array.
    """
    if abs(amount) < 1e-4:
        return image
    h, w = image.shape[:2]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y = (np.arange(h, dtype=np.float32) - cy) / (cy if cy > 0 else 1.0)
    x = (np.arange(w, dtype=np.float32) - cx) / (cx if cx > 0 else 1.0)
    xx, yy = np.meshgrid(x, y)
    # Normalise so distance = 1 at the corner of the image
    dist = np.clip(np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2.0), 0.0, 1.0)
    mask = dist ** 2  # float32: dist derived from float32 arange
    factor = (1.0 - amount * mask)[:, :, np.newaxis]
    return np.clip(image * factor, 0.0, 1.0).astype(np.float32)


def apply_grain(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Film grain simulation — luminance-weighted Gaussian noise.

    Grain is stronger in shadows and midtones, weaker in highlights,
    mimicking silver-halide film behaviour.

    Args:
        image:  float32 (H, W, 3) in range 0–1.
        amount: grain strength (0 = off, 1 = strong; noise sigma up to 0.08).

    Returns:
        Clipped float32 array.
    """
    if amount < 1e-4:
        return image
    sigma = amount * 0.08
    noise = np.random.default_rng().standard_normal(image.shape).astype(np.float32) * sigma
    weight = (1.0 - np.float32(0.5) * _luma(image))[:, :, np.newaxis]
    return np.clip(image + noise * weight, 0.0, 1.0).astype(np.float32)


def apply_split_toning(
    image: np.ndarray,
    shadow_hue: float,
    shadow_sat: float,
    highlight_hue: float,
    highlight_sat: float,
    balance: float = 0.0,
) -> np.ndarray:
    """
    Apply split toning — different colour casts to shadows and highlights.

    Args:
        image:         float32 (H, W, 3) in range 0–1.
        shadow_hue:    shadow colour hue in degrees [0, 360].
        shadow_sat:    shadow toning saturation [0, 1].  0 = no toning.
        highlight_hue: highlight colour hue in degrees [0, 360].
        highlight_sat: highlight toning saturation [0, 1].  0 = no toning.
        balance:       shifts the shadow/highlight boundary [−1, +1].
                       0 = centred at midtones; +1 = more highlights,
                       −1 = more shadow area.

    Returns:
        Clipped float32 array.
    """
    if shadow_sat < 1e-4 and highlight_sat < 1e-4:
        return image

    luma = _luma(image)
    center = float(np.clip(0.5 + balance * 0.4, 0.1, 0.9))
    result = image.copy()

    if shadow_sat > 1e-4:
        r, g, b = colorsys.hls_to_rgb(shadow_hue / 360.0, 0.5, float(shadow_sat))
        delta = np.array([r - 0.5, g - 0.5, b - 0.5], dtype=np.float32)
        shadow_w = (np.clip(1.0 - luma / center, 0.0, 1.0) ** 2).astype(np.float32)
        result = result + shadow_w[:, :, np.newaxis] * delta

    if highlight_sat > 1e-4:
        r, g, b = colorsys.hls_to_rgb(highlight_hue / 360.0, 0.5, float(highlight_sat))
        delta = np.array([r - 0.5, g - 0.5, b - 0.5], dtype=np.float32)
        denom = max(1.0 - center, 1e-6)
        highlight_w = (np.clip((luma - center) / denom, 0.0, 1.0) ** 2).astype(np.float32)
        result = result + highlight_w[:, :, np.newaxis] * delta

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate *image* by *angle* degrees counter-clockwise (expand canvas).

    Uses per-channel float32 PIL mode 'F' rotation so no precision is lost
    during interpolation.  The background fill is black (0.0).

    Args:
        image: float32 (H, W, 3) in range 0–1.
        angle: rotation angle in degrees (positive = CCW).

    Returns:
        float32 (H', W', 3) — dimensions change if expand=True.
    """
    if abs(angle) < 0.01:
        return image
    from PIL import Image
    channels = []
    for ch in range(3):
        pil_ch = Image.fromarray(image[:, :, ch], mode="F")
        rotated = pil_ch.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0.0,
        )
        channels.append(np.asarray(rotated, dtype=np.float32))
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0)
