import numpy as np
import pytest

from luminos.core.color import (
    apply_exposure,
    apply_levels,
    apply_saturation,
    apply_white_balance,
    temp_tint_to_rgb_multipliers,
)
from luminos.core.curves import all_identity, apply_curves, make_lut
from luminos.core.inversion import detect_orange_mask, invert_bw
from luminos.core.pipeline import process


def test_basic_color_adjustments_keep_float_rgb_range():
    image = np.array(
        [
            [[0.2, 0.4, 0.6], [0.9, 0.8, 0.7]],
            [[0.0, 0.5, 1.0], [0.1, 0.1, 0.1]],
        ],
        dtype=np.float32,
    )

    result = apply_white_balance(image, (2.0, 1.0, 0.5))
    result = apply_exposure(result, 0.5)
    result = apply_levels(result, 0.1, 0.9)
    result = apply_saturation(result, 0.8)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_temp_tint_neutral_is_close_to_identity():
    rgb = temp_tint_to_rgb_multipliers(6500, 0)
    assert rgb == pytest.approx((1.0, 1.0, 1.0), abs=0.03)


def test_curves_identity_and_non_identity_application():
    identity_points = [(0.0, 0.0), (1.0, 1.0)]
    luts = {ch: make_lut(identity_points) for ch in ("master", "r", "g", "b")}
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(2, 2, 3)

    assert all_identity(luts)
    np.testing.assert_allclose(apply_curves(image, luts), image, atol=1 / 255)

    luts["master"] = make_lut([(0.0, 0.0), (1.0, 0.5)])
    adjusted = apply_curves(image, luts)
    assert adjusted.shape == image.shape
    assert float(adjusted.max()) <= 0.5


def test_invert_bw_returns_equal_channels():
    image = np.dstack(
        [
            np.linspace(0.1, 0.9, 9, dtype=np.float32).reshape(3, 3),
            np.linspace(0.2, 0.8, 9, dtype=np.float32).reshape(3, 3),
            np.linspace(0.3, 0.7, 9, dtype=np.float32).reshape(3, 3),
        ]
    )

    result = invert_bw(image)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, :, 0], result[:, :, 1])
    np.testing.assert_allclose(result[:, :, 1], result[:, :, 2])
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_detect_orange_mask_uses_bright_percentile_region():
    image = np.full((20, 20, 3), [0.2, 0.2, 0.2], dtype=np.float32)
    image[-4:, :, :] = [0.85, 0.55, 0.35]

    mask = detect_orange_mask(image)

    assert mask == pytest.approx((0.85, 0.55, 0.35), abs=0.02)


def test_process_bw_pipeline_smoke():
    image = np.linspace(0.05, 0.95, 75, dtype=np.float32).reshape(5, 5, 3)

    result = process(
        image,
        film_type="bw",
        exposure_stops=0.2,
        white_balance=(1.0, 1.0, 1.0),
        black_point=0.02,
        white_point=0.98,
        saturation=1.0,
        sharpening=0.0,
    )

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
