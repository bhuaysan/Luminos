"""Histogram computation helpers, background worker, and display widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QMutex, QRectF, QThread, QWaitCondition, Signal, Qt
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

# ── Smoothing kernel ──────────────────────────────────────────────────────────

_HIST_SMOOTH_SIGMA = 6.0
_HIST_SMOOTH_RADIUS = int(3 * _HIST_SMOOTH_SIGMA)
_HIST_SMOOTH_X = np.arange(-_HIST_SMOOTH_RADIUS, _HIST_SMOOTH_RADIUS + 1, dtype=np.float64)
_HIST_SMOOTH = np.exp(-_HIST_SMOOTH_X ** 2 / (2.0 * _HIST_SMOOTH_SIGMA ** 2))
_HIST_SMOOTH /= _HIST_SMOOTH.sum()
del _HIST_SMOOTH_SIGMA, _HIST_SMOOTH_RADIUS, _HIST_SMOOTH_X

# ── Display colours ───────────────────────────────────────────────────────────

_HIST_CH_FILL = (
    QColor(220, 55,  55,  110),
    QColor(55,  200, 55,  110),
    QColor(55,  80,  220, 110),
)
_HIST_CH_LINE = (
    QColor(220, 55,  55,  230),
    QColor(55,  200, 55,  230),
    QColor(55,  80,  220, 230),
)
_HIST_LUMA_COLOR  = QColor(230, 230, 230, 200)
_HIST_GRID_COLOR  = QColor(55,  55,  55,  180)
_HIST_BG_COLOR    = QColor(18,  18,  18)
_HIST_LABEL_COLOR = QColor(100, 100, 100, 255)


# ── Pure computation helpers ──────────────────────────────────────────────────


def _hist_compute(arr: np.ndarray) -> dict:
    """Compute smoothed per-channel histograms from a uint8 (H, W, 3) array."""
    r = np.convolve(
        np.bincount(arr[:, :, 0].ravel(), minlength=256).astype(np.float32),
        _HIST_SMOOTH, mode='same',
    ).astype(np.float32)
    g = np.convolve(
        np.bincount(arr[:, :, 1].ravel(), minlength=256).astype(np.float32),
        _HIST_SMOOTH, mode='same',
    ).astype(np.float32)
    b = np.convolve(
        np.bincount(arr[:, :, 2].ravel(), minlength=256).astype(np.float32),
        _HIST_SMOOTH, mode='same',
    ).astype(np.float32)
    luma_u8 = (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    ).astype(np.uint8)
    luma = np.convolve(
        np.bincount(luma_u8.ravel(), minlength=256).astype(np.float32),
        _HIST_SMOOTH, mode='same',
    ).astype(np.float32)
    return {"r": r, "g": g, "b": b, "luma": luma}


def _hist_normalise(counts: np.ndarray, global_max: float, log: bool) -> np.ndarray:
    """Map raw (smoothed) counts → [0, 1] for display."""
    if log:
        v = np.log10(counts.astype(np.float64) + 1.0)
        m = float(np.log10(global_max + 1.0)) if global_max > 0 else 1.0
    else:
        v = counts.astype(np.float64)
        m = float(global_max) if global_max > 0 else 1.0
    return np.clip(v / m, 0.0, 1.0).astype(np.float32)


def _hist_filled_path(vals: np.ndarray, w: float, h: float) -> QPainterPath:
    """Closed filled-area path for a histogram channel."""
    path = QPainterPath()
    dx = w / max(len(vals) - 1, 1)
    path.moveTo(0.0, h)
    for i, v in enumerate(vals):
        path.lineTo(i * dx, h - float(v) * h)
    path.lineTo(w, h)
    path.closeSubpath()
    return path


def _hist_line_path(vals: np.ndarray, w: float, h: float) -> QPainterPath:
    """Open outline path for a histogram channel (channel lines and luma)."""
    path = QPainterPath()
    dx = w / max(len(vals) - 1, 1)
    path.moveTo(0.0, h - float(vals[0]) * h)
    for i in range(1, len(vals)):
        path.lineTo(i * dx, h - float(vals[i]) * h)
    return path


# ── Background worker ─────────────────────────────────────────────────────────


class _HistogramWorker(QThread):
    """
    Runs histogram computation off the GUI thread.

    "Latest wins": if ``compute()`` is called while a job is already running,
    the new array replaces the pending job so stale frames never queue up.
    """

    result_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mutex   = QMutex()
        self._cond    = QWaitCondition()
        self._pending: np.ndarray | None = None
        self._running = True

    def compute(self, arr: np.ndarray) -> None:
        self._mutex.lock()
        self._pending = arr
        self._cond.wakeOne()
        self._mutex.unlock()
        if not self.isRunning():
            self.start()

    def shutdown(self) -> None:
        self._mutex.lock()
        self._running = False
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait(3000)

    def run(self) -> None:
        while True:
            self._mutex.lock()
            while self._pending is None and self._running:
                self._cond.wait(self._mutex)
            arr     = self._pending
            self._pending = None
            running = self._running
            self._mutex.unlock()
            if not running:
                break
            if arr is not None:
                self.result_ready.emit(_hist_compute(arr))


# ── Display widget ────────────────────────────────────────────────────────────


class _HistogramWidget(QWidget):
    """
    Live RGB + optional luminance histogram.

    Three semi-transparent filled areas (R/G/B), jointly normalised.
    Opaque channel outlines drawn on top of all fills.
    White dashed BT.601 luminance curve (independently normalised, optional).
    Subtle grid lines at 25 %, 50 %, 75 %.
    Linear ↔ log₁₀ vertical scale — right-click context menu or set_log_scale().
    Computation runs on _HistogramWorker so the GUI never blocks.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._log_scale  = False
        self._show_luma  = False
        self._data: dict | None = None

        self._worker = _HistogramWorker(self)
        self._worker.result_ready.connect(self._on_result)

        self.setFixedHeight(110)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        pal = self.palette()
        pal.setColor(self.backgroundRole(), _HIST_BG_COLOR)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def update_from_uint8(self, arr: np.ndarray) -> None:
        self._worker.compute(np.ascontiguousarray(arr, dtype=np.uint8))

    def clear(self) -> None:
        self._data = None
        self.update()

    def set_log_scale(self, enabled: bool) -> None:
        if self._log_scale != enabled:
            self._log_scale = enabled
            self.update()

    def set_show_luminance(self, enabled: bool) -> None:
        if self._show_luma != enabled:
            self._show_luma = enabled
            self.update()

    def shutdown_worker(self) -> None:
        self._worker.shutdown()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        p.fillRect(self.rect(), _HIST_BG_COLOR)

        if self._data is None:
            font = QFont()
            font.setPointSize(8)
            p.setFont(font)
            p.setPen(_HIST_LABEL_COLOR)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kein Bild")
            p.end()
            return

        p.setPen(QPen(_HIST_GRID_COLOR, 1, Qt.PenStyle.SolidLine))
        for frac in (0.25, 0.50, 0.75):
            x = int(frac * w)
            p.drawLine(x, 0, x, int(h))

        r, g, b = self._data["r"], self._data["g"], self._data["b"]
        rgb_max = float(max(r.max(), g.max(), b.max()))
        norm = [_hist_normalise(ch, rgb_max, self._log_scale) for ch in (r, g, b)]

        p.setPen(Qt.PenStyle.NoPen)
        for vals, fill in zip(norm, _HIST_CH_FILL):
            p.setBrush(QBrush(fill))
            p.drawPath(_hist_filled_path(vals, w, h))

        p.setBrush(Qt.BrushStyle.NoBrush)
        for vals, line_col in zip(norm, _HIST_CH_LINE):
            p.setPen(QPen(line_col, 1.0))
            p.drawPath(_hist_line_path(vals, w, h))

        if self._show_luma:
            luma = self._data["luma"]
            luma_vals = _hist_normalise(luma, float(luma.max()), self._log_scale)
            pen = QPen(_HIST_LUMA_COLOR, 1.5, Qt.PenStyle.DashLine)
            pen.setDashPattern([5.0, 3.0])
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(_hist_line_path(luma_vals, w, h))

        label = "LOG" if self._log_scale else "LIN"
        font = QFont("monospace")
        font.setPointSize(7)
        p.setFont(font)
        p.setPen(_HIST_LABEL_COLOR)
        p.drawText(
            QRectF(w - 34.0, 2.0, 32.0, 14.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            label,
        )
        p.end()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        act_log = QAction("Logarithmisch", self)
        act_log.setCheckable(True)
        act_log.setChecked(self._log_scale)
        act_log.toggled.connect(self.set_log_scale)
        menu.addAction(act_log)

        act_luma = QAction("Luminanz anzeigen", self)
        act_luma.setCheckable(True)
        act_luma.setChecked(self._show_luma)
        act_luma.toggled.connect(self.set_show_luminance)
        menu.addAction(act_luma)

        menu.exec(self.mapToGlobal(pos))

    def _on_result(self, data: dict) -> None:
        self._data = data
        self.update()
