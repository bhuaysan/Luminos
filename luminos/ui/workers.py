"""QThread worker classes for import, full-load, and export operations."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from luminos.core.pipeline import load_image, process
from luminos.core.inversion import detect_orange_mask, invert
from luminos.core.color import rotate_image
from luminos.io.export import save_tiff, save_png, save_jpeg, _get_srgb_icc_bytes
from luminos.io.exif import read_exif_bytes

_PREVIEW_LONG_EDGE = 1500
_THUMB_LONG_EDGE = 160


@dataclasses.dataclass
class ProcessingParams:
    """All per-image pipeline parameters passed to export workers."""
    exposure_stops: float
    white_balance: tuple[float, float, float]
    mask: tuple[float, float, float] | None
    black_point: float = 0.0
    white_point: float = 1.0
    saturation: float = 1.0
    curve_luts: dict | None = None
    sharpening: float = 0.0
    angle: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    vibrance: float = 0.0
    noise_reduction: float = 0.0
    vignette: float = 0.0
    grain: float = 0.0
    film_type: str = "c41"
    split_toning: tuple[float, float, float, float, float] | None = None
    exif_bytes: bytes | None = None


def _downscale(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Nearest-neighbour downscale via step-based slice. Returns original if already small."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image
    # ceil so step is always >= 2 when long_edge > max_long_edge, guaranteeing
    # the result fits within the limit (floor-division gives step=1 for images
    # between max_long_edge+1 and 2*max_long_edge-1, which is no downscale at all).
    step = math.ceil(long_edge / max_long_edge)
    return image[::step, ::step].copy()


def _apply_crop(
    image: np.ndarray,
    crop_region_norm: tuple[float, float, float, float],
) -> np.ndarray:
    """Slice *image* to the normalised crop region; raises ValueError for empty results."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop_region_norm
    cropped = image[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if cropped.size == 0:
        raise ValueError(
            f"Crop region {crop_region_norm!r} produced an empty image "
            f"(source shape {image.shape})"
        )
    return cropped


def _run_process_and_save(
    image: np.ndarray,
    output_path: str,
    params: ProcessingParams,
    *,
    fmt: str,
    quality: int = 95,
) -> None:
    """Run the full pipeline on *image* and write the result to *output_path*."""
    result = process(
        image,
        exposure_stops=params.exposure_stops,
        white_balance=params.white_balance,
        mask=params.mask,
        black_point=params.black_point,
        white_point=params.white_point,
        saturation=params.saturation,
        curve_luts=params.curve_luts,
        sharpening=params.sharpening,
        contrast=params.contrast,
        highlights=params.highlights,
        shadows=params.shadows,
        vibrance=params.vibrance,
        noise_reduction=params.noise_reduction,
        vignette=params.vignette,
        grain=params.grain,
        film_type=params.film_type,
        split_toning=params.split_toning,
    )
    if abs(params.angle) > 0.05:
        result = rotate_image(result, params.angle)
    icc = _get_srgb_icc_bytes()
    if fmt == "tiff":
        save_tiff(result, output_path, icc_profile=icc)
    elif fmt == "png":
        save_png(result, output_path, icc_profile=icc)
    else:
        save_jpeg(result, output_path, quality=quality, icc_profile=icc, source_exif=params.exif_bytes)


class _ImportWorker(QThread):
    """
    Sequentially processes a list of image paths.

    For each path: load → downscale → detect mask → invert → thumbnail.
    Emits ``image_ready`` incrementally so the UI updates as each image finishes.
    Full image is freed immediately after downscaling to keep RAM bounded.
    """

    image_ready = Signal(str, object, object, object, object, object)  # + exif_bytes
    error = Signal(str, str)
    all_done = Signal()

    def __init__(self, paths: list[str], preview_long_edge: int = _PREVIEW_LONG_EDGE) -> None:
        super().__init__()
        self._paths = paths
        self._preview_long_edge = preview_long_edge

    def run(self) -> None:
        for path in self._paths:
            try:
                raw = load_image(path)
                preview = _downscale(raw, self._preview_long_edge)
                del raw

                mask = detect_orange_mask(preview)
                inverted = invert(preview, mask=mask)

                thumb_uint8 = (
                    np.clip(_downscale(inverted, _THUMB_LONG_EDGE), 0, 1) * 255
                ).astype(np.uint8)

                exif = read_exif_bytes(path)
                self.image_ready.emit(path, thumb_uint8, preview, inverted, mask, exif)
            except Exception as exc:
                self.error.emit(path, str(exc))
        self.all_done.emit()


class _FullImageLoader(QThread):
    """
    Loads the full-resolution image for the active entry.

    Signal named ``loaded`` (not ``finished``) to avoid shadowing QThread.finished.
    Checks isInterruptionRequested() before emitting so superseded loaders are silent.
    """

    loaded = Signal(str, object)
    error = Signal(str, str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        try:
            raw = load_image(self._path)
            if not self.isInterruptionRequested():
                self.loaded.emit(self._path, raw)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(self._path, str(exc))


class _ExportWorker(QThread):
    """Export the active full-resolution image with current pipeline settings."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        raw: np.ndarray,
        path: str,
        params: ProcessingParams,
        crop_region_norm: tuple[float, float, float, float] | None = None,
        rotation_steps: int = 0,
    ) -> None:
        super().__init__()
        self._raw = raw
        self._path = path
        self._params = params
        self._crop = crop_region_norm
        self._rotation_steps = rotation_steps

    def run(self) -> None:
        try:
            raw = self._raw
            if self._rotation_steps:
                raw = np.rot90(raw, k=self._rotation_steps)
            if self._crop is not None:
                raw = _apply_crop(raw, self._crop)
            suffix = self._path.rsplit(".", 1)[-1].lower()
            fmt = "tiff" if suffix in ("tif", "tiff") else ("png" if suffix == "png" else "jpeg")
            _run_process_and_save(raw, self._path, self._params, fmt=fmt)
            self.finished.emit(self._path)
        except Exception as exc:
            self.error.emit(str(exc))


class _BatchExportWorker(QThread):
    """
    Batch-export a list of images to an output directory.

    Each image is loaded fresh, processed with per-image settings, and saved
    as ``<stem><suffix>.<ext>``. Pre-computed orange masks are reused when available.
    """

    progress = Signal(int, int, str)
    finished = Signal(int)
    error = Signal(str, str)

    def __init__(
        self,
        paths: list[str],
        output_dir: str,
        fmt: str,
        quality: int,
        params: ProcessingParams,
        masks: dict[str, tuple[float, float, float]],
        crops: dict[str, tuple[float, float, float, float]] | None = None,
        rotations: dict[str, int] | None = None,
        film_types: dict[str, str] | None = None,
        exif_data: dict[str, bytes | None] | None = None,
        suffix: str = "_positive",
    ) -> None:
        super().__init__()
        self._paths = paths
        self._output_dir = output_dir
        self._fmt = fmt
        self._quality = quality
        self._params = params
        self._masks = masks
        self._crops = crops or {}
        self._rotations = rotations or {}
        self._film_types = film_types or {}
        self._exif_data = exif_data or {}
        self._suffix = suffix

    def run(self) -> None:
        _EXT = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg"}
        ext = _EXT.get(self._fmt, ".tif")

        total = len(self._paths)
        exported = 0
        for i, path in enumerate(self._paths):
            self.progress.emit(i, total, path)
            try:
                raw = load_image(path)
                rot = self._rotations.get(path, 0)
                if rot:
                    raw = np.rot90(raw, k=rot)
                crop = self._crops.get(path)
                if crop is not None:
                    raw = _apply_crop(raw, crop)
                out_path = Path(self._output_dir) / (Path(path).stem + self._suffix + ext)
                per_image = dataclasses.replace(
                    self._params,
                    mask=self._masks.get(path),
                    film_type=self._film_types.get(path, "c41"),
                    exif_bytes=self._exif_data.get(path),
                )
                _run_process_and_save(raw, str(out_path), per_image, fmt=self._fmt, quality=self._quality)
                del raw
                exported += 1
            except Exception as exc:
                self.error.emit(path, str(exc))
        self.finished.emit(exported)
