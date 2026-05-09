"""Extract EXIF bytes from image files for preservation in exported copies."""

from __future__ import annotations

from pathlib import Path

_RAW_EXTENSIONS = frozenset({'.nef', '.cr2', '.cr3', '.arw', '.dng', '.orf', '.raf', '.rw2'})


def read_exif_bytes(path: str) -> bytes | None:
    """
    Return raw EXIF bytes from *path*, or None if unavailable.

    - TIFF / JPEG: read directly via Pillow.
    - Camera RAW: build a minimal block from rawpy metadata + piexif
      (make, model, ISO, shutter, aperture, focal length).
      Returns None if piexif is not installed.
    """
    suffix = Path(path).suffix.lower()

    if suffix in ('.tif', '.tiff', '.jpg', '.jpeg'):
        try:
            from PIL import Image
            with Image.open(path) as img:
                exif = img.info.get('exif')
                if exif:
                    return exif
        except Exception:
            pass
        return None

    if suffix in _RAW_EXTENSIONS:
        return _exif_from_raw(path)

    return None


def _exif_from_raw(path: str) -> bytes | None:
    try:
        import piexif
        import rawpy
        from fractions import Fraction
    except ImportError:
        return None

    try:
        with rawpy.imread(path) as raw:
            m = raw.metadata
    except Exception:
        return None

    ifd0: dict = {}
    exif_ifd: dict = {}

    if m.camera_manufacturer:
        ifd0[piexif.ImageIFD.Make] = m.camera_manufacturer.encode('ascii', errors='replace')
    if m.camera_model:
        ifd0[piexif.ImageIFD.Model] = m.camera_model.encode('ascii', errors='replace')
    if m.iso:
        exif_ifd[piexif.ExifIFD.ISOSpeedRatings] = int(m.iso)
    if m.shutter and m.shutter > 0:
        frac = Fraction(m.shutter).limit_denominator(100000)
        exif_ifd[piexif.ExifIFD.ExposureTime] = (frac.numerator, frac.denominator)
    if m.aperture and m.aperture > 0:
        frac = Fraction(float(m.aperture)).limit_denominator(100)
        exif_ifd[piexif.ExifIFD.FNumber] = (frac.numerator, frac.denominator)
    if m.focal_len and m.focal_len > 0:
        frac = Fraction(float(m.focal_len)).limit_denominator(100)
        exif_ifd[piexif.ExifIFD.FocalLength] = (frac.numerator, frac.denominator)

    if not ifd0 and not exif_ifd:
        return None

    try:
        return piexif.dump({"0th": ifd0, "Exif": exif_ifd, "1st": {}})
    except Exception:
        return None
