from types import SimpleNamespace

from luminos.ui.export_controller import build_current_processing_params, ensure_export_extension
from luminos.ui.session import _ImageEntry


def test_ensure_export_extension_uses_selected_filter_when_missing_suffix():
    assert ensure_export_extension("/tmp/out", "TIFF 16-bit (*.tif *.tiff)") == "/tmp/out.tif"
    assert ensure_export_extension("/tmp/out", "PNG 8-bit (*.png)") == "/tmp/out.png"
    assert ensure_export_extension("/tmp/out", "JPEG 8-bit (*.jpg *.jpeg)") == "/tmp/out.jpg"


def test_ensure_export_extension_preserves_existing_suffix():
    assert ensure_export_extension("/tmp/out.custom", "PNG 8-bit (*.png)") == "/tmp/out.custom"


def test_build_current_processing_params_uses_entry_and_ui_state():
    window = SimpleNamespace(
        _curve_widget=SimpleNamespace(luts={"master": "lut"}),
        _exposure=lambda: 0.5,
        _white_balance=lambda: (1.1, 1.0, 0.9),
        _black_point=lambda: 0.02,
        _white_point=lambda: 0.98,
        _saturation=lambda: 0.8,
        _sharpening=lambda: 0.1,
        _angle=lambda: 1.5,
        _contrast=lambda: -0.2,
        _highlights=lambda: 0.3,
        _shadows=lambda: -0.4,
        _vibrance=lambda: 0.25,
        _noise_reduction=lambda: 0.05,
        _vignette=lambda: -0.1,
        _grain=lambda: 0.06,
        _split_toning_params=lambda: (10.0, 0.2, 200.0, 0.3, 0.0),
    )
    entry = _ImageEntry(
        path="/scan/a.tif",
        orange_mask=(0.8, 0.5, 0.3),
        film_type="bw",
        exif_bytes=b"exif",
    )

    params = build_current_processing_params(window, entry)

    assert params.exposure_stops == 0.5
    assert params.white_balance == (1.1, 1.0, 0.9)
    assert params.mask == (0.8, 0.5, 0.3)
    assert params.black_point == 0.02
    assert params.white_point == 0.98
    assert params.saturation == 0.8
    assert params.curve_luts == {"master": "lut"}
    assert params.sharpening == 0.1
    assert params.angle == 1.5
    assert params.contrast == -0.2
    assert params.highlights == 0.3
    assert params.shadows == -0.4
    assert params.vibrance == 0.25
    assert params.noise_reduction == 0.05
    assert params.vignette == -0.1
    assert params.grain == 0.06
    assert params.film_type == "bw"
    assert params.split_toning == (10.0, 0.2, 200.0, 0.3, 0.0)
    assert params.exif_bytes == b"exif"


def test_build_current_processing_params_allows_batch_overrides():
    window = SimpleNamespace(
        _curve_widget=SimpleNamespace(luts=None),
        _exposure=lambda: 0.0,
        _white_balance=lambda: (1.0, 1.0, 1.0),
        _black_point=lambda: 0.0,
        _white_point=lambda: 1.0,
        _saturation=lambda: 1.0,
        _sharpening=lambda: 0.0,
        _angle=lambda: 0.0,
        _contrast=lambda: 0.0,
        _highlights=lambda: 0.0,
        _shadows=lambda: 0.0,
        _vibrance=lambda: 0.0,
        _noise_reduction=lambda: 0.0,
        _vignette=lambda: 0.0,
        _grain=lambda: 0.0,
        _split_toning_params=lambda: None,
    )

    params = build_current_processing_params(window, None, mask=None, film_type="c41")

    assert params.mask is None
    assert params.film_type == "c41"
    assert params.exif_bytes is None
