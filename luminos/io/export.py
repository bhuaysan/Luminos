"""
Export processed float32 images to TIFF (16-bit), PNG (8-bit), or JPEG (8-bit).

All functions accept a float32 (H, W, 3) array in range 0–1 — the internal
representation used throughout Luminos.

16-bit TIFF backend priority
-----------------------------
1. pyvips   — fastest, handles 50 MP+ without RAM spikes (optional install)
2. tifffile — reliable, standard scientific TIFF (required install)
3. Pillow   — 8-bit fallback only; warns that precision is reduced

Metadata / colour management
------------------------------
All outputs embed an sRGB ICC profile so other applications render colours
correctly without manual profile assignment.

TIFF  — ICC via tag 34675 (tifffile path); software + datetime tags.
PNG   — ICC via Pillow icc_profile parameter; metadata as iTXt chunks.
JPEG  — ICC via Pillow icc_profile; EXIF from source file merged with
        Luminos software tag (requires piexif for merging; raw source
        bytes are passed through unchanged when piexif is absent).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

log = logging.getLogger(__name__)

_SOFTWARE = "Luminos"

# TIFF IFD tag numbers
_T_SOFTWARE   = 305
_T_DATETIME   = 306
_T_IMAGE_DESC = 270
_T_ICC_PROFILE = 34675  # tag for embedded ICC profile

# ── sRGB ICC profile (lazy-loaded, module-level cache) ────────────────────────

_srgb_icc_cache: bytes | None = None
_srgb_icc_loaded: bool = False


def _get_srgb_icc_bytes() -> bytes | None:
    """Return sRGB IEC61966-2.1 ICC profile bytes, or None on failure."""
    global _srgb_icc_cache, _srgb_icc_loaded
    if not _srgb_icc_loaded:
        try:
            from PIL import ImageCms
            profile = ImageCms.createProfile('sRGB')
            _srgb_icc_cache = ImageCms.ImageCmsProfile(profile).tobytes()
        except Exception:
            _srgb_icc_cache = None
        _srgb_icc_loaded = True
    return _srgb_icc_cache


# ── Public API ────────────────────────────────────────────────────────────────


def save_tiff(
    arr: np.ndarray,
    path: str | Path,
    metadata: dict | None = None,
    icc_profile: bytes | None = None,
) -> Path:
    """
    Save a float32 (H, W, 3) array as a lossless 16-bit TIFF.

    Args:
        arr:         float32 array in range 0–1.
        path:        Destination path (parent directories are created).
        metadata:    Optional dict with ``film_type``, ``description``, ``date``.
        icc_profile: Raw ICC profile bytes to embed (defaults to sRGB).

    Returns:
        Resolved Path of the written file.
    """
    path = _ensure_parent(path)
    arr = np.clip(arr, 0.0, 1.0)
    uint16 = (arr * 65535.0).astype(np.uint16)
    icc = icc_profile if icc_profile is not None else _get_srgb_icc_bytes()

    try:
        _save_tiff_pyvips(uint16, path)
        log.debug("TIFF saved via pyvips: %s", path)
        return path
    except ImportError:
        pass
    except Exception as exc:
        log.warning("pyvips TIFF save failed (%s), trying tifffile", exc)

    try:
        _save_tiff_tifffile(uint16, path, metadata, icc)
        log.debug("TIFF saved via tifffile: %s", path)
        return path
    except ImportError:
        pass
    except Exception as exc:
        log.warning("tifffile save failed (%s), falling back to Pillow 8-bit", exc)

    log.warning(
        "Saving %s as 8-bit TIFF — install tifffile or pyvips for 16-bit output",
        path,
    )
    _save_tiff_pillow_8bit(arr, path, icc)
    return path


def save_png(
    arr: np.ndarray,
    path: str | Path,
    metadata: dict | None = None,
    icc_profile: bytes | None = None,
) -> Path:
    """
    Save a float32 (H, W, 3) array as a lossless 8-bit PNG with sRGB ICC profile.

    Args:
        arr:         float32 array in range 0–1.
        path:        Destination path.
        metadata:    Optional dict (see save_tiff).
        icc_profile: Raw ICC profile bytes (defaults to sRGB).

    Returns:
        Resolved Path of the written file.
    """
    path = _ensure_parent(path)
    uint8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(uint8, mode="RGB")
    icc = icc_profile if icc_profile is not None else _get_srgb_icc_bytes()

    pnginfo = PngInfo()
    for key, value in _build_png_metadata(metadata).items():
        pnginfo.add_text(key, str(value))

    save_kwargs: dict = {"format": "PNG", "pnginfo": pnginfo}
    if icc:
        save_kwargs["icc_profile"] = icc

    img.save(str(path), **save_kwargs)
    return path


def save_jpeg(
    arr: np.ndarray,
    path: str | Path,
    quality: int = 95,
    metadata: dict | None = None,
    icc_profile: bytes | None = None,
    source_exif: bytes | None = None,
) -> Path:
    """
    Save a float32 (H, W, 3) array as an 8-bit JPEG with sRGB ICC profile.

    EXIF from the original file (``source_exif``) is merged with Luminos
    software / processing-date tags using piexif.  When piexif is absent the
    source bytes are passed through unchanged.

    Args:
        arr:         float32 array in range 0–1.
        path:        Destination path.
        quality:     JPEG quality, 1–95.
        metadata:    Optional dict (see save_tiff).
        icc_profile: Raw ICC profile bytes (defaults to sRGB).
        source_exif: Raw EXIF bytes from the original file to preserve.

    Returns:
        Resolved Path of the written file.
    """
    path = _ensure_parent(path)
    uint8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(uint8, mode="RGB")
    icc = icc_profile if icc_profile is not None else _get_srgb_icc_bytes()

    save_kwargs: dict = {"format": "JPEG", "quality": quality}
    if icc:
        save_kwargs["icc_profile"] = icc

    exif_bytes = _build_jpeg_exif(metadata, source_exif)
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes

    img.save(str(path), **save_kwargs)
    return path


# ── Backend implementations ───────────────────────────────────────────────────


def _save_tiff_pyvips(uint16: np.ndarray, path: Path) -> None:
    import pyvips
    img = pyvips.Image.new_from_array(uint16)
    img.tiffsave(str(path), compression="none", bigtiff=True)


def _save_tiff_tifffile(
    uint16: np.ndarray,
    path: Path,
    metadata: dict | None,
    icc_profile: bytes | None,
) -> None:
    import tifffile

    meta = metadata or {}
    description = _description_string(meta)
    now = _exif_datetime(meta.get("date"))

    extra_tags = []
    if icc_profile:
        extra_tags.append((_T_ICC_PROFILE, 7, None, icc_profile, True))

    tifffile.imwrite(
        str(path),
        uint16,
        photometric="rgb",
        software=_SOFTWARE,
        datetime=now,
        description=description or None,
        extratags=extra_tags or None,
    )


def _save_tiff_pillow_8bit(arr: np.ndarray, path: Path, icc_profile: bytes | None) -> None:
    uint8 = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(uint8, mode="RGB")
    save_kwargs: dict = {"format": "TIFF"}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    img.save(str(path), **save_kwargs)


# ── Metadata builders ─────────────────────────────────────────────────────────


def _build_png_metadata(metadata: dict | None) -> dict[str, str]:
    meta = metadata or {}
    out: dict[str, str] = {
        "Software": _SOFTWARE,
        "DateTime": _exif_datetime(meta.get("date")).strftime("%Y:%m:%d %H:%M:%S"),
    }
    if meta.get("film_type"):
        out["Film"] = str(meta["film_type"])
    if meta.get("description"):
        out["Description"] = str(meta["description"])
    return out


def _build_jpeg_exif(
    metadata: dict | None,
    source_exif: bytes | None = None,
) -> bytes | None:
    """
    Build a JPEG EXIF block.

    If *source_exif* is provided the existing tags are preserved and only
    ``Software``, ``DateTime``, and (optionally) ``ImageDescription`` are
    updated.  When piexif is not installed the source bytes are returned as-is.
    """
    try:
        import piexif
    except ImportError:
        return source_exif  # pass through unchanged

    meta = metadata or {}
    now_str = _exif_datetime(meta.get("date")).strftime("%Y:%m:%d %H:%M:%S").encode()

    if source_exif:
        try:
            exif_dict = piexif.load(source_exif)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "1st": {}}
    else:
        exif_dict = {"0th": {}, "Exif": {}, "1st": {}}

    ifd0 = exif_dict.setdefault("0th", {})
    exif_ifd = exif_dict.setdefault("Exif", {})

    ifd0[piexif.ImageIFD.Software] = _SOFTWARE.encode()
    ifd0[piexif.ImageIFD.DateTime] = now_str
    # Keep original capture time; set digitised time to processing time.
    exif_ifd.setdefault(piexif.ExifIFD.DateTimeOriginal, now_str)
    exif_ifd[piexif.ExifIFD.DateTimeDigitized] = now_str

    desc = _description_string(meta)
    if desc:
        ifd0[piexif.ImageIFD.ImageDescription] = desc.encode('utf-8', errors='replace')

    try:
        return piexif.dump(exif_dict)
    except Exception:
        return source_exif


# ── Utilities ─────────────────────────────────────────────────────────────────


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _description_string(meta: dict) -> str:
    parts = []
    if meta.get("film_type"):
        parts.append(f"Film: {meta['film_type']}")
    if meta.get("description"):
        parts.append(meta["description"])
    return "  |  ".join(parts)


def _exif_datetime(date_value=None) -> datetime.datetime:
    if date_value is None:
        return datetime.datetime.now()
    if isinstance(date_value, datetime.datetime):
        return date_value
    if isinstance(date_value, datetime.date):
        return datetime.datetime(date_value.year, date_value.month, date_value.day)
    raw = str(date_value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    log.warning("Could not parse EXIF date %r; original capture time will not be preserved", date_value)
    return datetime.datetime.now()
