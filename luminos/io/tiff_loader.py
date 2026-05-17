"""Load TIFF files (8-bit or 16-bit) into float32 NumPy arrays."""

import numpy as np


def load_tiff(path: str) -> np.ndarray:
    """
    Load a TIFF and return a float32 array (H, W, 3), range 0–1.

    Tries pyvips first (fast, handles huge files); falls back to tifffile for
    precision-preserving 16-bit IO, then Pillow as a last resort.
    """
    try:
        return _load_pyvips(path)
    except Exception:
        pass
    try:
        return _load_tifffile(path)
    except Exception:
        return _load_pillow(path)


def _load_pyvips(path: str) -> np.ndarray:
    import pyvips

    img = pyvips.Image.new_from_file(path, access="sequential")
    # pyvips gives us a flat bytes buffer; reshape to (H, W, bands)
    data = img.numpy()
    arr = np.array(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr /= _max_for_format(img.format)
    return _ensure_rgb(arr)


def _load_tifffile(path: str) -> np.ndarray:
    import tifffile

    arr = tifffile.imread(path)
    if arr.ndim > 3:
        arr = arr[0]
    arr = _ensure_rgb(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    max_value = float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float32) / max_value


def _load_pillow(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    arr /= 65535.0 if arr.max() > 255 else 255.0
    return arr


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        return arr[:, :, :3]
    return arr


def _max_for_format(fmt: str) -> float:
    """Map a pyvips BandFormat name to the integer max value."""
    return {
        "uchar": 255.0,
        "ushort": 65535.0,
        "uint": 4294967295.0,
        "float": 1.0,
        "double": 1.0,
    }.get(fmt, 65535.0)
