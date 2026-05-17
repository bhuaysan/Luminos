"""Undo/redo history for per-image edit settings."""

from __future__ import annotations

from collections import deque

from luminos.ui.session import _EditSettings


class _EditHistory:
    """Manage undo/redo stacks for slider and curve editing state."""

    def __init__(self, maxlen: int = 50) -> None:
        self.undo: deque[_EditSettings] = deque(maxlen=maxlen)
        self.redo: deque[_EditSettings] = deque(maxlen=maxlen)

    def push(self, settings: _EditSettings) -> None:
        self.undo.append(settings)
        self.redo.clear()

    def undo_step(self, current: _EditSettings) -> _EditSettings | None:
        if not self.undo:
            return None
        self.redo.append(current)
        return self.undo.pop()

    def redo_step(self, current: _EditSettings) -> _EditSettings | None:
        if not self.redo:
            return None
        self.undo.append(current)
        return self.redo.pop()

    def clear_all(self) -> None:
        self.undo.clear()
        self.redo.clear()

    def has_undo(self) -> bool:
        return bool(self.undo)

    def has_redo(self) -> bool:
        return bool(self.redo)
