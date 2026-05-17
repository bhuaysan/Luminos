import numpy as np
import pytest

from luminos.core.curves import all_identity
from luminos.ui.export_params import (
    curve_luts_from_settings,
    processing_params_from_settings,
    split_toning_params_from_settings,
)
from luminos.ui.session import _EditSettings, _ImageEntry


def test_split_toning_params_from_settings_returns_none_when_inactive():
    settings = _EditSettings(split_shadow_sat=0, split_highlight_sat=0)

    assert split_toning_params_from_settings(settings) is None


def test_split_toning_params_from_settings_scales_slider_values():
    settings = _EditSettings(
        split_shadow_hue=35,
        split_shadow_sat=20,
        split_highlight_hue=210,
        split_highlight_sat=40,
        split_balance=-25,
    )

    assert split_toning_params_from_settings(settings) == (35.0, 0.2, 210.0, 0.4, -0.25)


def test_curve_luts_from_settings_builds_all_channels():
    settings = _EditSettings()
    settings.curve_points["master"] = [(0.0, 0.0), (1.0, 0.5)]

    luts = curve_luts_from_settings(settings)

    assert set(luts) == {"master", "r", "g", "b"}
    assert all(lut.shape == (256,) for lut in luts.values())
    assert all(lut.dtype == np.float32 for lut in luts.values())
    assert not all_identity(luts)


def test_processing_params_from_settings_maps_saved_entry_state():
    settings = _EditSettings(
        exposure=7,
        contrast=-20,
        highlights=15,
        shadows=-10,
        black_point=3,
        white_point=97,
        wb_temp=6500,
        wb_tint=10,
        vibrance=25,
        saturation=80,
        sharpening=12,
        noise_reduction=8,
        vignette=-30,
        grain=6,
        angle=45,
        split_shadow_hue=40,
        split_shadow_sat=20,
    )
    entry = _ImageEntry(
        path="/tmp/image.tif",
        orange_mask=(0.8, 0.5, 0.3),
        film_type="bw",
        exif_bytes=b"exif",
    )

    params = processing_params_from_settings(settings, entry)

    assert params.exposure_stops == 0.7
    assert params.mask == (0.8, 0.5, 0.3)
    assert params.black_point == 0.03
    assert params.white_point == 0.97
    assert params.saturation == 0.8
    assert params.sharpening == 0.12
    assert params.angle == 4.5
    assert params.contrast == -0.2
    assert params.highlights == 0.15
    assert params.shadows == -0.1
    assert params.vibrance == 0.25
    assert params.noise_reduction == 0.08
    assert params.vignette == -0.3
    assert params.grain == 0.06
    assert params.film_type == "bw"
    assert params.split_toning == (40.0, 0.2, 0.0, 0.0, 0.0)
    assert params.exif_bytes == b"exif"
    assert params.white_balance == pytest.approx((1.0, 1.0, 1.0), abs=0.1)
