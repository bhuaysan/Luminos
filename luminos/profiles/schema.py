"""
Profile JSON schema validation.

A valid profile must have:
  - "name":  non-empty string
  - "type":  "color_negative" or "bw_negative"

Optional keys:
  - "orange_mask":   [r, g, b]  — 3 positive floats overriding auto-detected mask
  - "tone_curve":    [[x, y], ...]  — ≥2 control points for the master tone curve
  - "white_balance": [r, g, b]  — 3 positive floats (1.0 = neutral multiplier)
"""
from __future__ import annotations

from typing import Any

_VALID_TYPES = {"color_negative", "bw_negative"}


def validate_profile(data: dict[str, Any]) -> None:
    """Raise ValueError if *data* does not describe a valid profile."""
    if not isinstance(data, dict):
        raise ValueError("Profile must be a JSON object")

    for key in ("name", "type"):
        if key not in data:
            raise ValueError(f"Profile is missing required key: {key!r}")

    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("Profile 'name' must be a non-empty string")

    if data["type"] not in _VALID_TYPES:
        raise ValueError(
            f"Profile 'type' must be one of {sorted(_VALID_TYPES)}, got {data['type']!r}"
        )

    if "orange_mask" in data:
        mask = data["orange_mask"]
        if not (
            isinstance(mask, list)
            and len(mask) == 3
            and all(isinstance(v, (int, float)) and v > 0 for v in mask)
        ):
            raise ValueError(
                "Profile 'orange_mask' must be a list of 3 positive numbers"
            )

    if "tone_curve" in data:
        curve = data["tone_curve"]
        if not isinstance(curve, list) or len(curve) < 2:
            raise ValueError(
                "Profile 'tone_curve' must be a list of at least 2 [x, y] pairs"
            )
        for pt in curve:
            if not (
                isinstance(pt, list)
                and len(pt) == 2
                and all(isinstance(v, (int, float)) for v in pt)
            ):
                raise ValueError(
                    "Each 'tone_curve' point must be [x, y] with numeric values"
                )

    if "white_balance" in data:
        wb = data["white_balance"]
        if not (
            isinstance(wb, list)
            and len(wb) == 3
            and all(isinstance(v, (int, float)) and v > 0 for v in wb)
        ):
            raise ValueError(
                "Profile 'white_balance' must be a list of 3 positive numbers"
            )
