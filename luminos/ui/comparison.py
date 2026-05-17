"""Before/after and split-view comparison helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap

from luminos.ui.image_utils import array_to_pixmap


class _ComparisonController:
    """Manage before/after and split-view state for the main viewer."""

    def __init__(
        self,
        before_button,
        split_button,
        before_action,
        refresh_display: Callable[[], None],
    ) -> None:
        self._before_button = before_button
        self._split_button = split_button
        self._before_action = before_action
        self._refresh_display = refresh_display
        self.before_after = False
        self.split_view = False
        self.before_pixmap: QPixmap | None = None

    def set_enabled(self, enabled: bool) -> None:
        self._before_button.setEnabled(enabled)
        self._before_action.setEnabled(enabled)

    def toggle_before_after(self) -> None:
        if self._before_button.isEnabled():
            self._before_button.setChecked(not self._before_button.isChecked())

    def reset(self) -> None:
        """Reset comparison modes without emitting toggle signals."""
        self.before_after = False
        self.before_pixmap = None
        self.split_view = False
        self._before_button.blockSignals(True)
        self._before_button.setChecked(False)
        self._before_button.setText("Vorher")
        self._before_button.blockSignals(False)
        self._split_button.blockSignals(True)
        self._split_button.setChecked(False)
        self._split_button.blockSignals(False)
        self._before_action.setChecked(False)

    def set_before_after(self, active: bool, inverted_preview: np.ndarray | None) -> None:
        """Show the unedited inverted image or the processed result."""
        self.before_after = active
        if active:
            self._rebuild_before_pixmap(inverted_preview)
        if active and self.split_view:
            self._split_button.blockSignals(True)
            self._split_button.setChecked(False)
            self._split_button.blockSignals(False)
            self.split_view = False
        self._before_button.setText("Nachher" if active else "Vorher")
        self._before_action.setChecked(active)
        self._refresh_display()

    def set_split_view(self, active: bool, inverted_preview: np.ndarray | None) -> None:
        """Activate/deactivate the side-by-side split-view comparison."""
        self.split_view = active
        if active:
            self._rebuild_before_pixmap(inverted_preview)
            if self.before_after:
                self._before_button.blockSignals(True)
                self._before_button.setChecked(False)
                self._before_button.setText("Vorher")
                self._before_button.blockSignals(False)
                self._before_action.setChecked(False)
                self.before_after = False
        self._refresh_display()

    def invalidate_before_pixmap(self, inverted_preview: np.ndarray | None) -> None:
        """Clear and rebuild the cached before pixmap when comparison is active."""
        self.before_pixmap = None
        if (self.before_after or self.split_view) and inverted_preview is not None:
            self._rebuild_before_pixmap(inverted_preview)

    def base_pixmap(self, after_pixmap: QPixmap) -> QPixmap:
        """Return the pixmap that should be scaled for the current comparison mode."""
        if self.before_after and self.before_pixmap is not None:
            return self.before_pixmap
        return after_pixmap

    def maybe_make_split_composite(self, scaled_after: QPixmap) -> QPixmap:
        """Wrap a scaled after pixmap in a split composite when split-view is active."""
        if self.split_view and self.before_pixmap is not None:
            return make_split_composite(scaled_after, self.before_pixmap)
        return scaled_after

    def _rebuild_before_pixmap(self, inverted_preview: np.ndarray | None) -> None:
        if inverted_preview is None:
            self.before_pixmap = None
            return
        uint8 = (np.clip(inverted_preview, 0, 1) * 255).astype(np.uint8)
        self.before_pixmap = array_to_pixmap(uint8)


def make_split_composite(after_scaled: QPixmap, before_pixmap: QPixmap | None) -> QPixmap:
    """
    Build a split-screen composite from *after_scaled* and ``before_pixmap``.

    Left half: before image; right half: after image.  A thin white divider
    line and text labels are drawn at the centre.
    """
    if before_pixmap is None:
        return after_scaled

    w, h = after_scaled.width(), after_scaled.height()
    before_scaled = before_pixmap.scaled(
        w,
        h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    before_y = (h - before_scaled.height()) // 2

    composite = QPixmap(w, h)
    composite.fill(QColor(0, 0, 0))
    painter = QPainter(composite)

    painter.setClipRect(0, 0, w // 2, h)
    painter.drawPixmap(0, before_y, before_scaled)

    painter.setClipRect(w // 2, 0, w - w // 2, h)
    painter.drawPixmap(0, 0, after_scaled)

    painter.setClipping(False)

    mid = w // 2
    painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.SolidLine))
    painter.drawLine(mid, 0, mid, h)

    font = QFont()
    font.setPointSize(8)
    font.setBold(True)
    painter.setFont(font)
    margin = 6
    label_h = 18
    for text, align_right in (("VORHER", False), ("NACHHER", True)):
        fm = painter.fontMetrics()
        label_w = fm.horizontalAdvance(text)
        label_x = mid + margin if align_right else mid - margin - label_w
        label_y = margin
        painter.setPen(QColor(0, 0, 0, 160))
        painter.drawText(
            label_x + 1,
            label_y + 1,
            label_w,
            label_h,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )
        painter.setPen(QColor(255, 255, 255, 220))
        painter.drawText(
            label_x,
            label_y,
            label_w,
            label_h,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    painter.end()
    return composite
