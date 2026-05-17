from types import SimpleNamespace

from luminos.ui.slider_registry import _fmt_signed, build_slider_defs


class DummySlider:
    pass


class DummyLabel:
    pass


def _dummy_window():
    names = [
        "exposure",
        "contrast",
        "highlights",
        "shadows",
        "black",
        "white",
        "temp",
        "tint",
        "vibrance",
        "sat",
        "sharp",
        "noise",
        "vignette",
        "grain",
        "angle",
        "st_shadow_hue",
        "st_shadow_sat",
        "st_hi_hue",
        "st_hi_sat",
        "st_balance",
    ]
    values = {}
    for name in names:
        values[f"_{name}_slider"] = DummySlider()
        values[f"_{name}_label"] = DummyLabel()
    return SimpleNamespace(**values)


def test_fmt_signed_formats_zero_without_plus():
    assert _fmt_signed(0) == "0"
    assert _fmt_signed(5) == "+5"
    assert _fmt_signed(-5) == "-5"


def test_build_slider_defs_maps_all_edit_settings_attrs():
    defs = build_slider_defs(_dummy_window())

    assert [d.attr for d in defs] == [
        "exposure",
        "contrast",
        "highlights",
        "shadows",
        "black_point",
        "white_point",
        "wb_temp",
        "wb_tint",
        "vibrance",
        "saturation",
        "sharpening",
        "noise_reduction",
        "vignette",
        "grain",
        "angle",
        "split_shadow_hue",
        "split_shadow_sat",
        "split_highlight_hue",
        "split_highlight_sat",
        "split_balance",
    ]


def test_build_slider_defs_formatters_match_ui_labels():
    defs = {d.attr: d for d in build_slider_defs(_dummy_window())}

    assert defs["exposure"].fmt(12) == "+1.2 EV"
    assert defs["black_point"].fmt(3) == "0.03"
    assert defs["white_point"].fmt(97) == "0.97"
    assert defs["wb_temp"].fmt(5200) == "5200 K"
    assert defs["saturation"].fmt(85) == "0.85×"
    assert defs["angle"].fmt(-12) == "-1.2°"
    assert defs["split_shadow_hue"].fmt(45) == "45°"
