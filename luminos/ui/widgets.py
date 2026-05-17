"""Reusable UI widgets: scroll area, filmstrip, crop overlay, navigator, etc."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QListWidget,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

_FILMSTRIP_ICON_W = 170
_FILMSTRIP_ICON_H = 110


# ── Zoom scroll area ──────────────────────────────────────────────────────────


class _ZoomScrollArea(QScrollArea):
    """QScrollArea with mouse-wheel zoom and drag-to-pan."""

    wheel_step = Signal(int, object)
    image_clicked = Signal(object)
    crop_drag = Signal(object, object)
    crop_end = Signal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pan_active = False
        self._pan_start_pos: QPoint | None = None
        self._pan_start_hval: int = 0
        self._pan_start_vval: int = 0
        self._pipette_active = False
        self._crop_active = False
        self._crop_start: QPoint | None = None
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_pipette_mode(self, active: bool) -> None:
        self._pipette_active = active
        cursor = Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(cursor)

    def set_crop_mode(self, active: bool) -> None:
        self._crop_active = active
        self._crop_start = None
        cursor = Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(cursor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._crop_active:
            event.accept()
            return
        delta = event.angleDelta().y()
        if delta:
            self.wheel_step.emit(1 if delta > 0 else -1, event.position().toPoint())
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._pipette_active:
                self.image_clicked.emit(event.position().toPoint())
                event.accept()
                return
            if self._crop_active:
                self._crop_start = event.position().toPoint()
                event.accept()
                return
            self._pan_active = True
            self._pan_start_pos = event.position().toPoint()
            self._pan_start_hval = self.horizontalScrollBar().value()
            self._pan_start_vval = self.verticalScrollBar().value()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._crop_active and self._crop_start is not None:
            self.crop_drag.emit(self._crop_start, event.position().toPoint())
            event.accept()
            return
        if self._pan_active and self._pan_start_pos is not None:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self._pan_start_hval - delta.x())
            self.verticalScrollBar().setValue(self._pan_start_vval - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._crop_active and self._crop_start is not None:
                self.crop_end.emit(self._crop_start, event.position().toPoint())
                self._crop_start = None
                event.accept()
                return
            if self._pan_active:
                self._pan_active = False
                self._pan_start_pos = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
        super().mouseReleaseEvent(event)


# ── Reset slider ──────────────────────────────────────────────────────────────


class _ResetSlider(QSlider):
    """QSlider that resets to its default value on double-click."""

    about_to_reset = Signal()

    def __init__(self, orientation, default: int, parent=None) -> None:
        super().__init__(orientation, parent)
        self._default = default

    @property
    def default(self) -> int:
        return self._default

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.about_to_reset.emit()
        self.setValue(self._default)
        event.accept()


# ── Collapsible section ───────────────────────────────────────────────────────


class _CollapsibleSection(QWidget):
    """Panel section with a clickable header that collapses/expands the body."""

    _HEADER_STYLE = (
        "QPushButton {"
        "  background-color: #252525;"
        "  color: #b8b8b8;"
        "  font-weight: bold;"
        "  font-size: 10px;"
        "  text-transform: uppercase;"
        "  letter-spacing: 0.5px;"
        "  text-align: left;"
        "  padding: 5px 10px;"
        "  border: none;"
        "  border-top: 1px solid #383838;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2f2f2f;"
        "  color: #d0d0d0;"
        "}"
    )

    def __init__(self, title: str, parent=None, expanded: bool = True) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._title = title
        self._expanded = expanded

        self._btn = QPushButton()
        self._btn.setFlat(True)
        self._btn.setStyleSheet(self._HEADER_STYLE)
        self._btn.setFixedHeight(28)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(self._toggle)
        root.addWidget(self._btn)

        self._body = QWidget()
        body_vbox = QVBoxLayout(self._body)
        body_vbox.setContentsMargins(8, 6, 6, 10)
        body_vbox.setSpacing(4)
        self._body_layout = body_vbox
        root.addWidget(self._body)

        self._body.setVisible(expanded)
        self._refresh_btn()

    def add_widget(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._refresh_btn()

    def _refresh_btn(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        self._btn.setText(f"  {arrow}   {self._title}")


# ── Filmstrip ─────────────────────────────────────────────────────────────────


class _CentredIconDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.decorationAlignment = (
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        option.displayAlignment = (
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
        )


class _FilmstripList(QListWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setItemDelegate(_CentredIconDelegate(self))
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.viewport() and event.type() == QEvent.Type.Resize:
            self._centre_column()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._centre_column()

    def _centre_column(self) -> None:
        vp_h = self.viewport().height()
        spacing = max(0, (vp_h - (_FILMSTRIP_ICON_H + 30)) // 2)
        self.setGridSize(QSize(_FILMSTRIP_ICON_W, _FILMSTRIP_ICON_H + 30))
        self.setSpacing(spacing)
        self.viewport().update()


# ── Crop overlay ──────────────────────────────────────────────────────────────


class _CropOverlay(QWidget):
    """
    Transparent overlay on top of the image viewport showing the crop selection.

    Darkened mask outside the rectangle, white border, rule-of-thirds grid.
    Mouse events pass through (WA_TransparentForMouseEvents).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setVisible(False)
        self._rect: tuple[int, int, int, int] | None = None

    def set_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        self._rect = rect
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._rect is None:
            return
        p = QPainter(self)
        x0, y0, x1, y1 = self._rect
        rx0, ry0 = min(x0, x1), min(y0, y1)
        rx1, ry1 = max(x0, x1), max(y0, y1)
        vw, vh = self.width(), self.height()

        dark = QColor(0, 0, 0, 140)
        p.setPen(Qt.PenStyle.NoPen)
        if ry0 > 0:
            p.fillRect(0, 0, vw, ry0, dark)
        if ry1 < vh:
            p.fillRect(0, ry1, vw, vh - ry1, dark)
        if rx0 > 0:
            p.fillRect(0, ry0, rx0, ry1 - ry0, dark)
        if rx1 < vw:
            p.fillRect(rx1, ry0, vw - rx1, ry1 - ry0, dark)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 210), 1.5))
        p.drawRect(rx0, ry0, rx1 - rx0, ry1 - ry0)

        sel_w = rx1 - rx0
        sel_h = ry1 - ry0
        if sel_w > 20 and sel_h > 20:
            pen_thirds = QPen(QColor(255, 255, 255, 70), 0.8, Qt.PenStyle.DashLine)
            pen_thirds.setDashPattern([4.0, 3.0])
            p.setPen(pen_thirds)
            for frac in (1 / 3.0, 2 / 3.0):
                gx = rx0 + int(frac * sel_w)
                gy = ry0 + int(frac * sel_h)
                p.drawLine(gx, ry0, gx, ry1)
                p.drawLine(rx0, gy, rx1, gy)

        p.end()


# ── Navigator widget ──────────────────────────────────────────────────────────


class _NavigatorWidget(QWidget):
    """
    Miniature image preview with a viewport rectangle.

    Shows the full processed image scaled to fit. Draws a rectangle for the
    visible portion when zoomed. Click/drag to pan the main view.
    """

    pan_requested = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._rect_norm: tuple[float, float, float, float] | None = None
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(20, 20, 20))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def update_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self.update()

    def update_rect(self, rect_norm: tuple[float, float, float, float] | None) -> None:
        self._rect_norm = rect_norm
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._rect_norm = None
        self.update()

    def _thumbnail_geometry(self) -> tuple[int, int, int, int]:
        if self._pixmap is None or self._pixmap.isNull():
            return 0, 0, 0, 0
        pm_w, pm_h = self._pixmap.width(), self._pixmap.height()
        w, h = self.width(), self.height()
        if pm_w == 0 or pm_h == 0:
            return 0, 0, 0, 0
        scale = min(w / pm_w, h / pm_h)
        img_w = max(1, round(pm_w * scale))
        img_h = max(1, round(pm_h * scale))
        off_x = (w - img_w) // 2
        off_y = (h - img_h) // 2
        return off_x, off_y, img_w, img_h

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))

        if self._pixmap is None or self._pixmap.isNull():
            font = QFont()
            font.setPointSize(8)
            p.setFont(font)
            p.setPen(QColor(100, 100, 100))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kein Bild")
            p.end()
            return

        off_x, off_y, img_w, img_h = self._thumbnail_geometry()
        p.drawPixmap(off_x, off_y, img_w, img_h, self._pixmap)

        if self._rect_norm is not None:
            nx, ny, nw, nh = self._rect_norm
            rx = off_x + round(nx * img_w)
            ry = off_y + round(ny * img_h)
            rw = max(2, round(nw * img_w))
            rh = max(2, round(nh * img_h))
            p.fillRect(rx, ry, rw, rh, QColor(255, 200, 50, 40))
            p.setPen(QPen(QColor(255, 200, 50), 1, Qt.PenStyle.SolidLine))
            p.drawRect(rx, ry, rw - 1, rh - 1)

        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_pan(event.position().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_pan(event.position().toPoint())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def _emit_pan(self, pos: QPoint) -> None:
        off_x, off_y, img_w, img_h = self._thumbnail_geometry()
        if img_w == 0 or img_h == 0:
            return
        frac_x = max(0.0, min(1.0, (pos.x() - off_x) / img_w))
        frac_y = max(0.0, min(1.0, (pos.y() - off_y) / img_h))
        self.pan_requested.emit(frac_x, frac_y)
