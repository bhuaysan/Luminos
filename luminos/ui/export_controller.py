"""Single-image and batch export orchestration for MainWindow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog

from luminos.ui.dialogs import _BatchExportDialog
from luminos.ui.export_params import processing_params_from_settings
from luminos.ui.workers import _BatchExportWorker, _ExportWorker, ProcessingParams

_EXPORT_FILTERS = (
    "TIFF 16-bit (*.tif *.tiff);;"
    "PNG 8-bit (*.png);;"
    "JPEG 8-bit (*.jpg *.jpeg)"
)


class _ExportController:
    """Coordinate export dialogs, worker creation, and export status updates."""

    def __init__(self, window) -> None:
        self._window = window

    def export_image(self) -> None:
        w = self._window
        if w._raw_image is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            w, "Export Image", "", _EXPORT_FILTERS
        )
        if not path:
            return

        path = ensure_export_extension(path, selected_filter)
        w._status.showMessage(f"Exporting {Path(path).name}…")
        w._set_export_enabled(False)
        w._set_batch_enabled(False)

        entry = w._entries.get(w._active_path)
        worker = _ExportWorker(
            raw=w._raw_image,
            path=path,
            params=build_current_processing_params(w, entry),
            crop_region_norm=entry.crop_region_norm if entry is not None else None,
            rotation_steps=entry.rotation_steps if entry is not None else 0,
        )
        worker.finished.connect(self.on_export_finished)
        worker.error.connect(self.on_export_error)
        w._export_worker = worker
        worker.start()

    def on_export_finished(self, path: str) -> None:
        w = self._window
        w._set_export_enabled(True)
        w._set_batch_enabled(w._filmstrip.count() > 0)
        w._status.showMessage(f"Saved: {path}")

    def on_export_error(self, msg: str) -> None:
        w = self._window
        w._set_export_enabled(True)
        w._set_batch_enabled(w._filmstrip.count() > 0)
        w._status.showMessage(f"Export error: {msg}")

    def batch_export(self) -> None:
        w = self._window
        selected = w._filmstrip.selectedItems()
        if not selected:
            w._status.showMessage("Keine Bilder im Filmstreifen ausgewählt.")
            return
        if w._active_path in w._entries:
            w._entries[w._active_path].settings = w._read_settings()

        settings = w._app_settings
        dialog = _BatchExportDialog(
            w,
            default_fmt=settings.default_export_format,
            default_quality=settings.default_jpeg_quality,
            default_output_dir=settings.default_output_dir,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        w._set_batch_enabled(False)
        w._set_export_enabled(False)
        w._status.showMessage(
            f"Batch export: 0 / {len(paths)} — {Path(dialog.output_dir).name}/"
        )

        worker = _BatchExportWorker(
            paths=paths,
            output_dir=dialog.output_dir,
            fmt=dialog.fmt,
            quality=dialog.quality,
            params=build_current_processing_params(w, None, mask=None, film_type="c41"),
            params_by_path={
                p: processing_params_from_settings(w._entries[p].settings, w._entries[p])
                for p in paths
                if p in w._entries
            },
            masks={
                p: w._entries[p].orange_mask
                for p in paths
                if p in w._entries and w._entries[p].orange_mask is not None
            },
            crops={
                p: w._entries[p].crop_region_norm
                for p in paths
                if p in w._entries and w._entries[p].crop_region_norm is not None
            },
            rotations={
                p: w._entries[p].rotation_steps
                for p in paths
                if p in w._entries
            },
            film_types={
                p: w._entries[p].film_type
                for p in paths
                if p in w._entries
            },
            exif_data={
                p: w._entries[p].exif_bytes
                for p in paths
                if p in w._entries
            },
            suffix=settings.export_suffix,
        )
        worker.progress.connect(self.on_batch_progress)
        worker.finished.connect(self.on_batch_finished)
        worker.error.connect(self.on_batch_error)
        w._batch_worker = worker
        worker.start()

    def on_batch_progress(self, current: int, total: int, path: str) -> None:
        self._window._status.showMessage(
            f"Batch-Export: {current + 1} / {total} — {Path(path).name}"
        )

    def on_batch_finished(self, count: int) -> None:
        w = self._window
        w._set_batch_enabled(True)
        w._set_export_enabled(w._raw_image is not None)
        w._status.showMessage(f"Batch export complete: {count} file(s) saved.")

    def on_batch_error(self, path: str, msg: str) -> None:
        self._window._status.showMessage(f"Batch error ({Path(path).name}): {msg}")


def ensure_export_extension(path: str, selected_filter: str) -> str:
    """Append an export extension when the user did not type one."""
    if "." in Path(path).name:
        return path
    if "TIFF" in selected_filter:
        return f"{path}.tif"
    if "PNG" in selected_filter:
        return f"{path}.png"
    return f"{path}.jpg"


def build_current_processing_params(
    window,
    entry,
    *,
    mask="__entry__",
    film_type: str | None = None,
) -> ProcessingParams:
    """Build ProcessingParams from the currently active UI controls."""
    return ProcessingParams(
        exposure_stops=window._exposure(),
        white_balance=window._white_balance(),
        mask=entry.orange_mask if mask == "__entry__" and entry is not None else mask,
        black_point=window._black_point(),
        white_point=window._white_point(),
        saturation=window._saturation(),
        curve_luts=window._curve_widget.luts,
        sharpening=window._sharpening(),
        angle=window._angle(),
        contrast=window._contrast(),
        highlights=window._highlights(),
        shadows=window._shadows(),
        vibrance=window._vibrance(),
        noise_reduction=window._noise_reduction(),
        vignette=window._vignette(),
        grain=window._grain(),
        film_type=film_type or (entry.film_type if entry is not None else "c41"),
        split_toning=window._split_toning_params(),
        exif_bytes=entry.exif_bytes if entry is not None else None,
    )
