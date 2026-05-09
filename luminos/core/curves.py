"""
Tone curve utilities.

Provides monotone cubic spline interpolation (Fritsch-Carlson) for building
256-entry float32 look-up tables from user-defined control points, and
fast LUT-based application to float32 image arrays.
"""

from __future__ import annotations

import numpy as np

# Sentinel: identity LUT used when a channel has no curve applied.
_IDENTITY_LUT: np.ndarray = np.linspace(0.0, 1.0, 256, dtype=np.float32)

_DEFAULT_POINTS: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]


def make_lut(points: list[tuple[float, float]]) -> np.ndarray:
    """
    Build a 256-entry float32 LUT from control points using monotone cubic
    interpolation (Fritsch-Carlson).  Output is clamped to [0, 1].

    Args:
        points: list of (input, output) pairs in [0, 1].  Need not be sorted.
                At least two distinct x-values are required; otherwise the
                identity LUT is returned.

    Returns:
        float32 array of shape (256,), lut[i] = output for input i/255.
    """
    if len(points) < 2:
        return _IDENTITY_LUT.copy()

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Remove duplicates on x
    unique = np.concatenate([[True], np.diff(xs) > 1e-9])
    xs, ys = xs[unique], ys[unique]

    n = len(xs)
    if n < 2:
        return _IDENTITY_LUT.copy()
    if n == 2:
        t = np.linspace(0.0, 1.0, 256)
        return np.clip(np.interp(t, xs, ys), 0.0, 1.0).astype(np.float32)

    # Chord slopes
    h = np.diff(xs)
    s = np.diff(ys) / np.where(h > 1e-12, h, 1e-12)

    # Initial tangents (arithmetic mean of neighbouring slopes)
    m = np.empty(n)
    m[0]    = s[0]
    m[-1]   = s[-1]
    m[1:-1] = (s[:-1] + s[1:]) / 2.0

    # Fritsch-Carlson monotonicity constraints
    for i in range(n - 1):
        if abs(s[i]) < 1e-10:
            m[i] = m[i + 1] = 0.0
            continue
        alpha, beta = m[i] / s[i], m[i + 1] / s[i]
        mag = np.sqrt(alpha ** 2 + beta ** 2)
        if mag > 3.0:
            tau = 3.0 / mag
            m[i]     *= tau
            m[i + 1] *= tau

    # Evaluate cubic Hermite spline at 256 uniformly spaced input values
    t = np.linspace(0.0, 1.0, 256)
    idxs = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, n - 2)

    xi  = xs[idxs]
    xi1 = xs[idxs + 1]
    yi  = ys[idxs]
    yi1 = ys[idxs + 1]
    mi  = m[idxs]
    mi1 = m[idxs + 1]
    hi  = np.where((xi1 - xi) > 1e-12, xi1 - xi, 1e-12)
    tt  = (t - xi) / hi

    h00 =  2 * tt**3 - 3 * tt**2 + 1
    h10 =      tt**3 - 2 * tt**2 + tt
    h01 = -2 * tt**3 + 3 * tt**2
    h11 =      tt**3 -     tt**2

    out = h00 * yi + h10 * hi * mi + h01 * yi1 + h11 * hi * mi1
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def is_identity(lut: np.ndarray) -> bool:
    """Return True if *lut* is (approximately) the identity mapping."""
    return bool(np.allclose(lut, _IDENTITY_LUT, atol=1e-4))


def apply_curves(
    image: np.ndarray,
    luts: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Apply master + per-channel tone curves to a float32 (H, W, 3) image.

    The master LUT is composed with each channel LUT into a single combined
    LUT so only three vectorised index operations are needed.

    Args:
        image: float32 (H, W, 3) in range [0, 1].
        luts:  dict with keys 'master', 'r', 'g', 'b', each a 256-entry
               float32 LUT as returned by :func:`make_lut`.

    Returns:
        float32 (H, W, 3) in range [0, 1].
    """
    master = luts["master"]
    r_lut  = luts["r"]
    g_lut  = luts["g"]
    b_lut  = luts["b"]

    # Compose master + per-channel into three combined LUTs.
    m_idx = np.clip((master * 255.0).astype(np.int32), 0, 255)
    combined = [
        r_lut[m_idx],
        g_lut[m_idx],
        b_lut[m_idx],
    ]

    result = np.empty_like(image)
    for ch in range(3):
        idx = np.clip((image[:, :, ch] * 255.0).astype(np.int32), 0, 255)
        result[:, :, ch] = combined[ch][idx]
    return result


def all_identity(luts: dict[str, np.ndarray]) -> bool:
    """Return True if every LUT in *luts* is the identity mapping."""
    return all(is_identity(v) for v in luts.values())


def apply_curves_fast(
    image: np.ndarray,
    combined: list[np.ndarray],
) -> np.ndarray:
    """
    Apply pre-composed per-channel LUTs to a float32 (H, W, 3) image.

    *combined* must be a list of three 256-entry LUTs where each entry is
    already the composition of the master LUT with the per-channel LUT:
    ``combined[ch] = channel_lut[master_lut_indices]``.

    Use :py:attr:`_CurveWidget.combined_luts` which caches and invalidates
    this composition automatically, so the master+channel composition is
    recomputed only when the curve actually changes rather than every frame.

    Args:
        image:    float32 (H, W, 3) in range [0, 1].
        combined: list of three 256-entry float32 LUTs (R, G, B).

    Returns:
        float32 (H, W, 3) in range [0, 1].
    """
    result = np.empty_like(image)
    for ch in range(3):
        idx = np.clip((image[:, :, ch] * 255.0).astype(np.int32), 0, 255)
        result[:, :, ch] = combined[ch][idx]
    return result
