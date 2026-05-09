"""
Profile discovery and loading.

Built-in profiles are shipped in ``luminos/profiles/data/``.
User-defined profiles are stored in ``~/.local/share/luminos/profiles/``.

User profiles with the same display name as a built-in override the built-in.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from luminos.profiles.schema import validate_profile

log = logging.getLogger(__name__)

# Built-in profiles shipped with Luminos.
_BUILTIN_DIR = Path(__file__).parent / "data"
# User profiles stored in the XDG data directory.
_USER_DIR = Path.home() / ".local" / "share" / "luminos" / "profiles"


def list_profiles() -> list[tuple[str, Path, str]]:
    """
    Return a sorted list of ``(display_name, path, type)`` for all discoverable profiles.

    ``type`` is one of ``"color_negative"`` or ``"bw_negative"``.

    Searches built-in profiles first, then the user directory.
    A user profile whose ``"name"`` matches a built-in overrides it.
    Profiles that fail validation are skipped with a warning.
    """
    found: dict[str, tuple[Path, str]] = {}

    for directory in (_BUILTIN_DIR, _USER_DIR):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = _read_json(path)
                validate_profile(data)
                found[data["name"]] = (path, data.get("type", "color_negative"))
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping invalid profile %s: %s", path, exc)

    return sorted(
        [(name, path, ptype) for name, (path, ptype) in found.items()],
        key=lambda t: t[0].lower(),
    )


def load_profile(path: Path | str) -> dict[str, Any]:
    """
    Load and validate a single profile JSON file.

    Returns the profile dict.
    Raises ``ValueError`` if the file is invalid or ``OSError`` if it cannot be read.
    """
    data = _read_json(Path(path))
    validate_profile(data)
    return data


# ── Internal ──────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
