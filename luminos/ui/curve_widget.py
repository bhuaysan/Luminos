"""Interactive spline tone-curve editor widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from luminos.core.curves import make_lut

_CURVE_CHANNELS = ("master", "r", "g", "b")
_CURVE_CH_LABELS = ("Master", "Rot", "Grün", "Blau")
_CURVE_CH_COLORS = {
    "master": QColor(210, 210, 210),
    "r":      QColor(220, 65,  65),
    "g":      QColor(65,  195, 65),
    "b":      QColor(65,  100, 220),
}
_IDENTITY_POINTS = [(0.0, 0.0), (1.0, 1.0)]


class _CurveWidget(QWidget):
    """
    Interactive spline tone-curve editor.

    One curve per channel: Master, R, G, B.
    Left-click on empty area → add control point.
    Left-click + drag on existing point → move it.
    Right-click on point → remove it (minimum 2 points remain).
    Points at x=0 and x=1 can only move vertically.
    Active channel shown at full opacity; others drawn dimmed.
    Emits ``curve_changed`` after any modification.
    ``combined_luts`` caches master+channel LUT composition.
    """

    curve_changed = Signal()
    curve_editing_started = Signal()
    curve_editing_finished = Signal()

    _PAD = 10
    _POINT_R = 5

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._points: dict[str, list[tuple[float, float]]] = {
            ch: list(_IDENTITY_POINTS) for ch in _CURVE_CHANNELS
        }
        self._luts: dict[str, np.ndarray] = {
            ch: np.linspace(0.0, 1.0, 256, dtype=np.float32) for ch in _CURVE_CHANNELS
        }
        self._combined_cache: list[np.ndarray] | None = None
        self._active = "master"
        self._drag_idx: int | None = None
        self._edited_this_press: bool = False

        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(18, 18, 18))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def luts(self) -> dict[str, np.ndarray]:
        return self._luts

    @property
    def combined_luts(self) -> list[np.ndarray]:
        """Pre-composed master+channel LUTs; recomputed only when curves change."""
        if self._combined_cache is None:
            master = self._luts["master"]
            m_idx = np.clip((master * 255.0).astype(np.int32), 0, 255)
            self._combined_cache = [
                self._luts["r"][m_idx],
                self._luts["g"][m_idx],
                self._luts["b"][m_idx],
            ]
        return self._combined_cache

    def set_active_channel(self, ch: str) -> None:
        self._active = ch
        self.update()

    def reset_all(self) -> None:
        for ch in _CURVE_CHANNELS:
            self._points[ch] = list(_IDENTITY_POINTS)
            self._luts[ch] = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        self._combined_cache = None
        self.curve_changed.emit()
        self.update()

    def is_all_identity(self) -> bool:
        return all(pts == _IDENTITY_POINTS for pts in self._points.values())

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _canvas(self):
        p = self._PAD
        return p, p, self.width() - 2 * p, self.height() - 2 * p

    def _to_curve(self, pos: QPoint) -> tuple[float, float]:
        cx, cy, cw, ch = self._canvas()
        x = (pos.x() - cx) / max(cw, 1)
        y = 1.0 - (pos.y() - cy) / max(ch, 1)
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def _to_widget(self, px: float, py: float) -> QPoint:
        cx, cy, cw, ch = self._canvas()
        return QPoint(int(cx + px * cw), int(cy + (1.0 - py) * ch))

    def _hit(self, pos: QPoint) -> int | None:
        pts = self._points[self._active]
        for i, (px, py) in enumerate(pts):
            if (pos - self._to_widget(px, py)).manhattanLength() <= self._POINT_R * 2:
                return i
        return None

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._edited_this_press = False
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit(event.position().toPoint())
            if idx is not None:
                self.curve_editing_started.emit()
                self._drag_idx = idx
            else:
                self.curve_editing_started.emit()
                nx, ny = self._to_curve(event.position().toPoint())
                pts = self._points[self._active]
                pts.append((nx, ny))
                pts.sort(key=lambda p: p[0])
                self._drag_idx = next(i for i, p in enumerate(pts) if p == (nx, ny))
                self._recompute(self._active)
                self._edited_this_press = True
                self.curve_changed.emit()
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            idx = self._hit(event.position().toPoint())
            if idx is not None:
                pts = self._points[self._active]
                if len(pts) > 2:
                    self.curve_editing_started.emit()
                    pts.pop(idx)
                    self._recompute(self._active)
                    self._edited_this_press = True
                    self.curve_changed.emit()
                    self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_idx is None:
            event.accept()
            return
        nx, ny = self._to_curve(event.position().toPoint())
        pts = self._points[self._active]
        n = len(pts)
        if self._drag_idx == 0:
            nx = 0.0
        elif self._drag_idx == n - 1:
            nx = 1.0
        pts[self._drag_idx] = (nx, ny)
        if 0 < self._drag_idx < n - 1:
            pts.sort(key=lambda p: p[0])
            self._drag_idx = next(
                (i for i, p in enumerate(pts) if p == (nx, ny)),
                self._drag_idx,
            )
        self._recompute(self._active)
        self._edited_this_press = True
        self.curve_changed.emit()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._edited_this_press:
            self.curve_editing_finished.emit()
        self._drag_idx = None
        self._edited_this_press = False
        event.accept()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, cw, ch = self._canvas()

        p.fillRect(self.rect(), QColor(18, 18, 18))

        p.setPen(QPen(QColor(45, 45, 45), 1))
        for frac in (0.25, 0.5, 0.75):
            gx = cx + int(frac * cw)
            gy = cy + int(frac * ch)
            p.drawLine(gx, cy, gx, cy + ch)
            p.drawLine(cx, gy, cx + cw, gy)

        ref_pen = QPen(QColor(55, 55, 55), 1, Qt.PenStyle.DashLine)
        ref_pen.setDashPattern([4.0, 4.0])
        p.setPen(ref_pen)
        p.drawLine(cx, cy + ch, cx + cw, cy)

        p.setPen(QPen(QColor(55, 55, 55), 1))
        p.drawRect(cx, cy, cw, ch)

        for ch_key in (*[c for c in _CURVE_CHANNELS if c != self._active], self._active):
            lut = self._luts[ch_key]
            is_active = ch_key == self._active
            if not is_active:
                pts = self._points[ch_key]
                if pts == _IDENTITY_POINTS:
                    continue
            col = QColor(_CURVE_CH_COLORS[ch_key])
            col.setAlpha(220 if is_active else 55)
            pen = QPen(col, 1.5 if is_active else 1.0)
            p.setPen(pen)
            path = QPainterPath()
            for i in range(256):
                x = cx + (i / 255.0) * cw
                y = cy + (1.0 - float(lut[i])) * ch
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)

        pt_col = QColor(_CURVE_CH_COLORS[self._active])
        pt_col.setAlpha(230)
        p.setBrush(QBrush(pt_col))
        p.setPen(QPen(QColor(240, 240, 240, 200), 1.0))
        for px, py in self._points[self._active]:
            wp = self._to_widget(px, py)
            p.drawEllipse(wp, self._POINT_R, self._POINT_R)

        p.end()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _recompute(self, ch: str) -> None:
        self._luts[ch] = make_lut(self._points[ch])
        self._combined_cache = None
