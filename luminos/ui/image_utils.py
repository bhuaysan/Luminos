"""Image conversion helpers used by the Qt UI."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def array_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert a uint8 RGB ``(H, W, 3)`` array to a QPixmap."""
    h, w, _ = arr.shape
    qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def rotate_uint8(arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a uint8 RGB image by *angle* degrees counter-clockwise, expanding canvas."""
    pil = Image.fromarray(arr)
    rotated = pil.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(rotated)
