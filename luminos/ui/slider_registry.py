"""Slider registry and formatting helpers for edit settings."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable

_SliderDef = namedtuple("_SliderDef", ("slider", "label", "attr", "fmt"))


def _fmt_signed(value: int) -> str:
    """Format an int value with a leading +/- sign, returning '0' for zero."""
    return "0" if value == 0 else f"{value:+d}"


def build_slider_defs(window) -> list[_SliderDef]:
    """Build the canonical slider registry from a MainWindow instance."""
    return [
        _SliderDef(window._exposure_slider, window._exposure_label, "exposure", _fmt_ev),
        _SliderDef(window._contrast_slider, window._contrast_label, "contrast", _fmt_signed),
        _SliderDef(window._highlights_slider, window._highlights_label, "highlights", _fmt_signed),
        _SliderDef(window._shadows_slider, window._shadows_label, "shadows", _fmt_signed),
        _SliderDef(window._black_slider, window._black_label, "black_point", _fmt_percent_float),
        _SliderDef(window._white_slider, window._white_label, "white_point", _fmt_percent_float),
        _SliderDef(window._temp_slider, window._temp_label, "wb_temp", _fmt_kelvin),
        _SliderDef(window._tint_slider, window._tint_label, "wb_tint", _fmt_signed),
        _SliderDef(window._vibrance_slider, window._vibrance_label, "vibrance", _fmt_signed),
        _SliderDef(window._sat_slider, window._sat_label, "saturation", _fmt_saturation),
        _SliderDef(window._sharp_slider, window._sharp_label, "sharpening", str),
        _SliderDef(window._noise_slider, window._noise_label, "noise_reduction", str),
        _SliderDef(window._vignette_slider, window._vignette_label, "vignette", _fmt_signed),
        _SliderDef(window._grain_slider, window._grain_label, "grain", str),
        _SliderDef(window._angle_slider, window._angle_label, "angle", _fmt_angle),
        _SliderDef(
            window._st_shadow_hue_slider,
            window._st_shadow_hue_label,
            "split_shadow_hue",
            _fmt_degrees,
        ),
        _SliderDef(
            window._st_shadow_sat_slider,
            window._st_shadow_sat_label,
            "split_shadow_sat",
            str,
        ),
        _SliderDef(
            window._st_hi_hue_slider,
            window._st_hi_hue_label,
            "split_highlight_hue",
            _fmt_degrees,
        ),
        _SliderDef(
            window._st_hi_sat_slider,
            window._st_hi_sat_label,
            "split_highlight_sat",
            str,
        ),
        _SliderDef(
            window._st_balance_slider,
            window._st_balance_label,
            "split_balance",
            _fmt_signed,
        ),
    ]


def connect_slider_defs(
    slider_defs: list[_SliderDef],
    *,
    angle_slider,
    on_pressed: Callable[[], None],
    on_released: Callable[[], None],
    on_reset: Callable[[], None],
    on_angle_changed: Callable[[int], None],
    on_regular_changed: Callable[[], None],
) -> None:
    """Wire slider signals to history, label update, and preview callbacks."""
    for defn in slider_defs:
        defn.slider.sliderPressed.connect(on_pressed)
        defn.slider.sliderReleased.connect(on_released)
        defn.slider.about_to_reset.connect(on_reset)
        if defn.slider is angle_slider:
            defn.slider.valueChanged.connect(on_angle_changed)
        else:
            defn.slider.valueChanged.connect(
                lambda value, d=defn: (d.label.setText(d.fmt(value)), on_regular_changed())
            )


def _fmt_ev(value: int) -> str:
    return f"{value / 10.0:+.1f} EV"


def _fmt_percent_float(value: int) -> str:
    return f"{value / 100.0:.2f}"


def _fmt_kelvin(value: int) -> str:
    return f"{value} K"


def _fmt_saturation(value: int) -> str:
    return f"{value / 100.0:.2f}×"


def _fmt_angle(value: int) -> str:
    return f"{value / 10.0:+.1f}°"


def _fmt_degrees(value: int) -> str:
    return f"{value}°"
