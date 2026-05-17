"""Import and active-image loading helpers for MainWindow."""

from __future__ import annotations

from pathlib import Path

from luminos.ui.session import _ImageEntry
from luminos.ui.workers import _FullImageLoader, _ImportWorker


def new_import_paths(paths: list[str], entries: dict[str, _ImageEntry]) -> list[str]:
    """Return paths that are not already present in the entry map."""
    return [path for path in paths if path not in entries]


class _ActivationController:
    """Coordinate import workers and full-resolution image loader lifetime."""

    def __init__(self, window) -> None:
        self._window = window

    def start_import(self, paths: list[str]) -> None:
        w = self._window
        new_paths = new_import_paths(paths, w._entries)
        if not new_paths:
            w._status.showMessage("All selected images are already imported.")
            return

        for path in new_paths:
            w._entries[path] = _ImageEntry(
                path=path,
                film_type=w._app_settings.default_film_type,
            )

        w._status.showMessage(f"Importing {len(new_paths)} image(s)…")
        worker = _ImportWorker(new_paths, w._app_settings.preview_long_edge)
        worker.image_ready.connect(w._on_image_ready)
        worker.error.connect(w._on_import_error)
        worker.all_done.connect(w._on_import_done)
        w._import_worker = worker
        worker.start()

    def park_full_loader(self) -> None:
        """Disconnect and keep the current full loader alive until its thread exits."""
        w = self._window
        if w._full_loader is None:
            return

        w._full_loader.requestInterruption()
        try:
            w._full_loader.loaded.disconnect(w._on_full_image_loaded)
            w._full_loader.error.disconnect(w._on_load_error)
        except RuntimeError:
            pass

        old = w._full_loader
        w._old_loaders.append(old)
        old.finished.connect(
            lambda loader=old: w._old_loaders.remove(loader)
            if loader in w._old_loaders else None
        )
        w._full_loader = None

    def start_full_loader(self, path: str) -> None:
        """Start a full-resolution loader for the active path."""
        w = self._window
        w._status.showMessage(f"Loading: {Path(path).name}…")
        loader = _FullImageLoader(path)
        loader.loaded.connect(w._on_full_image_loaded)
        loader.error.connect(w._on_load_error)
        w._full_loader = loader
        loader.start()
