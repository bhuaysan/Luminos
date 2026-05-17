"""Main application window."""

from __future__ import annotations

import copy
import os
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QEvent, QPoint, QSize, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from luminos.core.pipeline import load_image, process
from luminos.core.inversion import detect_orange_mask, invert, invert_bw, suggest_auto_levels
from luminos.core.color import (
    apply_white_balance,
    apply_exposure,
    apply_levels,
    apply_saturation,
    apply_sharpening,
    rotate_image,
    apply_contrast,
    apply_highlights,
    apply_shadows,
    apply_vibrance,
    apply_noise_reduction,
    apply_vignette,
    apply_grain,
    apply_split_toning,
    temp_tint_to_rgb_multipliers,
    rgb_multipliers_to_temp_tint,
)
from luminos.core.curves import apply_curves_fast
from luminos.io.export import save_tiff

from luminos.ui.session import (
    _EditSettings,
    _ImageEntry,
    _AppSettings,
    _SESSION_FILTER,
    _load_app_settings,
    _save_app_settings,
    SessionManager,
)
from luminos.ui.workers import (
    _PREVIEW_LONG_EDGE,
    _ImportWorker,
    _FullImageLoader,
    _ExportWorker,
    _BatchExportWorker,
    ProcessingParams,
)
from luminos.ui.histogram import _HistogramWidget
from luminos.ui.curve_widget import _CurveWidget, _CURVE_CHANNELS, _CURVE_CH_LABELS
from luminos.ui.widgets import (
    _FILMSTRIP_ICON_W,
    _FILMSTRIP_ICON_H,
    _ZoomScrollArea,
    _ResetSlider,
    _CollapsibleSection,
    _FilmstripList,
    _CropOverlay,
    _NavigatorWidget,
)
from luminos.ui.dialogs import _BatchExportDialog, _PreferencesDialog
from luminos.ui.export_params import processing_params_from_settings

_IMPORT_FILTER = (
    "Images (*.tif *.tiff *.nef *.cr2 *.cr3 *.arw *.dng *.orf *.raf *.rw2)"
)
_IMPORT_EXTENSIONS = frozenset({
    '.tif', '.tiff', '.nef', '.cr2', '.cr3', '.arw', '.dng', '.orf', '.raf', '.rw2'
})
_EXPORT_FILTERS = (
    "TIFF 16-bit (*.tif *.tiff);;"
    "PNG 8-bit (*.png);;"
    "JPEG 8-bit (*.jpg *.jpeg)"
)

_ZOOM_STEPS = [10, 15, 25, 33, 50, 67, 75, 100, 125, 150, 200, 300, 400]


def _fmt_signed(v: int) -> str:
    """Format an int value with a leading +/- sign, returning '0' for zero."""
    return "0" if v == 0 else f"{v:+d}"


from collections import namedtuple as _namedtuple
_SliderDef = _namedtuple("_SliderDef", ("slider", "label", "attr", "fmt"))


class _EditHistory:
    """Manages undo/redo stacks for slider and curve editing state."""

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


def _array_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert a uint8 (H, W, 3) array to QPixmap (main thread only)."""
    h, w, _ = arr.shape
    qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _rotate_uint8(arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a uint8 (H, W, 3) array by *angle* degrees (positive = CCW, expand)."""
    from PIL import Image
    pil = Image.fromarray(arr)
    rotated = pil.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(rotated)

# ── Main window ───────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Luminos — Film Negative Converter")
        self.resize(1300, 750)

        # ── App settings (loaded first — other init may read them) ────────
        self._app_settings: _AppSettings = _load_app_settings()

        # ── Per-image state ───────────────────────────────────────────────
        # dict[path → _ImageEntry] — populated as imports complete.
        self._entries: dict[str, _ImageEntry] = {}
        # Path of the image currently shown in the main viewer.
        self._active_path: str | None = None

        # Shortcuts to active entry data (mirrors entry fields for quick access)
        self._raw_image: np.ndarray | None = None
        self._inverted_preview: np.ndarray | None = None
        self._orange_mask: tuple[float, float, float] | None = None

        # Cached QPixmap of the last pipeline result (re-scaled on resize)
        self._processed_pixmap: QPixmap | None = None
        # Last fully-processed uint8 array (after rotation) — used for clipping overlay.
        self._display_uint8: np.ndarray | None = None
        # Clipping indicator mode: highlights → red, shadows → blue
        self._clipping_mode: bool = False

        # Before/After comparison state
        self._before_after: bool = False
        self._before_pixmap: QPixmap | None = None
        # Split-view (side-by-side before|after) state
        self._split_view: bool = False

        # Zoom state
        # True  → always scale to fit the viewport (default)
        # False → fixed percentage stored in _zoom_pct
        self._zoom_fit_mode: bool = True
        self._zoom_pct: int = 100  # integer percentage, 10–400

        # Crop state
        self._crop_mode: bool = False
        self._crop_sel_norm: tuple[float, float, float, float] | None = None
        self._crop_ratio: float | None = None
        # Zoom state saved when entering crop mode so it can be restored on exit
        self._pre_crop_zoom_fit: bool = True
        self._pre_crop_zoom_pct: int = 100

        # Undo / redo stacks for editing-slider changes (per active image).
        self._history = _EditHistory()
        self._session_manager = SessionManager()
        # Snapshot taken on sliderPressed; compared on sliderReleased.
        self._pre_drag_settings: _EditSettings | None = None

        # Worker references (kept alive to prevent premature GC)
        self._import_worker: _ImportWorker | None = None
        self._full_loader: _FullImageLoader | None = None
        self._export_worker: _ExportWorker | None = None
        self._batch_worker: _BatchExportWorker | None = None
        # Old _FullImageLoader instances that are still running but whose
        # results we no longer want.  Held here so Python does not GC them
        # while their threads are live (Qt ABRT if QObject destroyed in wrong
        # thread).  Each entry removes itself via QThread.finished.
        self._old_loaders: list[_FullImageLoader] = []

        # Crop overlay — initialised before layout so resizeEvent guards work.
        self._crop_overlay: _CropOverlay | None = None

        # Profile map: display name → Path and type (populated in _load_profiles)
        self._profile_paths: dict[str, object] = {}
        self._profile_types: dict[str, str] = {}

        # Session restore: maps path → serialized settings dict, applied after import
        self._pending_session_settings: dict[str, dict] | None = None
        self._pending_session_active: str | None = None

        # ── Menu bar ──────────────────────────────────────────────────────
        self._build_menu()

        # ── Layout ────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(8)
        left_col.addWidget(self._build_viewer(), stretch=1)
        left_col.addWidget(self._build_filmstrip())
        root.addLayout(left_col, stretch=1)
        root.addWidget(self._build_controls())
        self._setup_slider_defs()

        # ── Crop overlay (child of viewport, covers it entirely) ──────────
        self._crop_overlay = _CropOverlay(self._scroll.viewport())
        self._crop_overlay.resize(self._scroll.viewport().size())
        self._crop_overlay.raise_()

        # ── Drag & drop ───────────────────────────────────────────────────
        self.setAcceptDrops(True)

        # ── Status bar ────────────────────────────────────────────────────
        self._status = self.statusBar()
        self._status.showMessage("Ready")

        # ── Preview debounce timer ────────────────────────────────────────
        # Slider valueChanged fires 30+ times per second while dragging.
        # The timer coalesces rapid bursts: preview runs 50 ms after the last
        # change, not after every individual tick.
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(50)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)

        # ── Keyboard shortcuts ────────────────────────────────────────────
        self._setup_shortcuts()

        # ── Pipette signal ────────────────────────────────────────────────
        self._scroll.image_clicked.connect(self._on_pipette_click)

        # ── Crop signals ──────────────────────────────────────────────────
        self._scroll.crop_drag.connect(self._on_crop_drag)
        self._scroll.crop_end.connect(self._on_crop_end)

        # ── Navigator signals ─────────────────────────────────────────────
        self._scroll.horizontalScrollBar().valueChanged.connect(self._update_navigator)
        self._scroll.verticalScrollBar().valueChanged.connect(self._update_navigator)
        self._navigator.pan_requested.connect(self._on_navigator_pan)

        # ── Profiles ──────────────────────────────────────────────────────
        self._load_profiles()

        # ── Apply persistent preferences ───────────────────────────────────
        self._apply_app_settings(self._app_settings)

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # ── Datei ─────────────────────────────────────────────────────────
        file_menu = mb.addMenu("&Datei")
        file_menu.addAction("&Importieren…", self._import_images, QKeySequence("Ctrl+O"))
        file_menu.addSeparator()
        self._act_save_session = file_menu.addAction(
            "&Sitzung speichern…", self._save_session, QKeySequence("Ctrl+S")
        )
        self._act_save_session.setEnabled(False)
        file_menu.addAction(
            "Sitzung &öffnen…", self._open_session, QKeySequence("Ctrl+Shift+O")
        )
        self._recent_menu = file_menu.addMenu("&Zuletzt geöffnet")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        self._act_export = file_menu.addAction("&Exportieren…", self._export_image, QKeySequence("Ctrl+E"))
        self._act_batch  = file_menu.addAction("&Batch-Export…", self._batch_export, QKeySequence("Ctrl+Shift+E"))
        self._act_export.setEnabled(False)
        self._act_batch.setEnabled(False)
        file_menu.addSeparator()
        file_menu.addAction("&Einstellungen…", self._show_preferences, QKeySequence("Ctrl+,"))
        file_menu.addSeparator()
        file_menu.addAction("&Beenden", self.close, QKeySequence("Ctrl+Q"))

        # ── Bearbeiten ────────────────────────────────────────────────────
        edit_menu = mb.addMenu("&Bearbeiten")
        self._act_undo_menu = edit_menu.addAction("&Rückgängig", self._undo_settings, QKeySequence("Ctrl+Z"))
        self._act_redo_menu = edit_menu.addAction("&Wiederholen", self._redo_settings, QKeySequence("Ctrl+Y"))
        self._act_undo_menu.setEnabled(False)
        self._act_redo_menu.setEnabled(False)
        edit_menu.addSeparator()
        self._act_reset_menu = edit_menu.addAction("Einst. &zurücksetzen", self._reset_all_sliders, QKeySequence("R"))
        self._act_copy_menu  = edit_menu.addAction("Einst. &kopieren", self._copy_settings_to_selected)
        self._act_reset_menu.setEnabled(False)
        self._act_copy_menu.setEnabled(False)

        # ── Ansicht ───────────────────────────────────────────────────────
        view_menu = mb.addMenu("&Ansicht")
        view_menu.addAction("&Einpassen", self._zoom_fit, QKeySequence("F"))
        view_menu.addAction("100 %", self._zoom_100, QKeySequence("1"))
        view_menu.addSeparator()
        self._act_before_after = view_menu.addAction("&Vorher/Nachher", self._toggle_before_after, QKeySequence("\\"))
        self._act_before_after.setCheckable(True)
        self._act_before_after.setEnabled(False)

        # ── Hilfe ─────────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Hilfe")
        help_menu.addAction("Über &Luminos…", self._show_about)

    def _set_export_enabled(self, v: bool) -> None:
        self._export_btn.setEnabled(v)
        self._act_export.setEnabled(v)

    def _set_batch_enabled(self, v: bool) -> None:
        self._batch_btn.setEnabled(v)
        self._act_batch.setEnabled(v)

    def _set_before_enabled(self, v: bool) -> None:
        self._before_btn.setEnabled(v)
        self._act_before_after.setEnabled(v)

    def _set_save_session_enabled(self, v: bool) -> None:
        self._act_save_session.setEnabled(v)

    # ── Preferences ───────────────────────────────────────────────────────────

    def _show_preferences(self) -> None:
        dlg = _PreferencesDialog(self._app_settings, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._app_settings = dlg.result_settings()
        _save_app_settings(self._app_settings)
        self._apply_app_settings(self._app_settings)

    def _apply_app_settings(self, s: _AppSettings) -> None:
        """Apply all preferences to the running application immediately."""
        self._preview_timer.setInterval(s.preview_debounce_ms)
        self._histogram.set_log_scale(s.histogram_log_scale)

    # ── Widget builders ───────────────────────────────────────────────────────

    def _build_filmstrip(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(_FILMSTRIP_ICON_H + 30 + 30)  # grid row + scrollbar + button headroom
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── Button column (left of the strip) ────────────────────────────
        btn_col = QWidget()
        btn_col.setFixedWidth(130)
        btn_layout = QVBoxLayout(btn_col)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(2)

        import_btn = QPushButton("Importieren…")
        import_btn.setToolTip("Bilder importieren (RAW / TIFF)")
        import_btn.clicked.connect(self._import_images)
        btn_layout.addWidget(import_btn)

        self._select_all_btn = QPushButton("Alle wählen")
        self._select_all_btn.setEnabled(False)
        self._select_all_btn.setToolTip("Alle Bilder markieren / Auswahl aufheben")
        self._select_all_btn.clicked.connect(self._toggle_select_all)
        btn_layout.addWidget(self._select_all_btn)

        self._remove_btn = QPushButton("Entfernen")
        self._remove_btn.setEnabled(False)
        self._remove_btn.setToolTip("Markierte Bilder aus dem Filmstreifen entfernen")
        self._remove_btn.clicked.connect(self._remove_selected)
        btn_layout.addWidget(self._remove_btn)

        btn_layout.addStretch()

        self._reset_selected_btn = QPushButton("Einst. zurücksetzen")
        self._reset_selected_btn.setEnabled(False)
        self._reset_selected_btn.setToolTip(
            "Bearbeitungseinstellungen der markierten Bilder auf Standard zurücksetzen"
        )
        self._reset_selected_btn.clicked.connect(self._reset_settings_for_selected)
        btn_layout.addWidget(self._reset_selected_btn)

        self._copy_settings_btn = QPushButton("Einst. kopieren")
        self._copy_settings_btn.setEnabled(False)
        self._copy_settings_btn.setToolTip(
            "Aktuelle Einstellungen auf alle markierten Bilder übertragen"
        )
        self._copy_settings_btn.clicked.connect(self._copy_settings_to_selected)
        btn_layout.addWidget(self._copy_settings_btn)

        layout.addWidget(btn_col)

        # ── Horizontal filmstrip list ─────────────────────────────────────
        self._filmstrip = _FilmstripList()
        self._filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self._filmstrip.setFlow(QListView.Flow.LeftToRight)
        self._filmstrip.setWrapping(False)
        self._filmstrip.setMovement(QListView.Movement.Static)
        self._filmstrip.setResizeMode(QListView.ResizeMode.Fixed)
        self._filmstrip.setIconSize(QSize(_FILMSTRIP_ICON_W, _FILMSTRIP_ICON_H))
        self._filmstrip.setGridSize(QSize(_FILMSTRIP_ICON_W, _FILMSTRIP_ICON_H + 30))
        self._filmstrip.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._filmstrip.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._filmstrip.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._filmstrip.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._filmstrip.itemClicked.connect(self._on_filmstrip_clicked)
        self._filmstrip.itemSelectionChanged.connect(self._on_filmstrip_selection_changed)
        layout.addWidget(self._filmstrip, stretch=1)

        return w

    def _build_viewer(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)

        # Image label inside a zoom-aware scroll area
        self._image_label = QLabel("No image loaded")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(600, 400)

        self._scroll = _ZoomScrollArea()
        self._scroll.setWidget(self._image_label)
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.wheel_step.connect(self._on_wheel_zoom)
        vbox.addWidget(self._scroll, stretch=1)

        # ── Zoom bar ──────────────────────────────────────────────────────────
        zoom_bar = QWidget()
        zoom_layout = QHBoxLayout(zoom_bar)
        zoom_layout.setContentsMargins(6, 2, 6, 2)
        zoom_layout.setSpacing(6)

        fit_btn = QPushButton("Fit")
        fit_btn.setFixedWidth(42)
        fit_btn.setToolTip("Reset to fit-in-window view  |  F")
        fit_btn.clicked.connect(self._zoom_fit)
        zoom_layout.addWidget(fit_btn)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 400)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setToolTip("Zoom level (10 %–400 %)")
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        self._zoom_slider.setFixedWidth(160)
        zoom_layout.addWidget(self._zoom_slider)

        self._zoom_label = QLabel("Fit")
        self._zoom_label.setFixedWidth(46)
        self._zoom_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        zoom_layout.addWidget(self._zoom_label)

        zoom_layout.addStretch()

        self._clipping_btn = QPushButton("Clipping")
        self._clipping_btn.setFixedWidth(66)
        self._clipping_btn.setCheckable(True)
        self._clipping_btn.setEnabled(False)
        self._clipping_btn.setToolTip(
            "Beschnittene Lichter (rot) und Tiefen (blau) anzeigen"
        )
        self._clipping_btn.toggled.connect(self._set_clipping_mode)
        zoom_layout.addWidget(self._clipping_btn)

        self._before_btn = QPushButton("Vorher")
        self._before_btn.setFixedWidth(60)
        self._before_btn.setCheckable(True)
        self._set_before_enabled(False)
        self._before_btn.setToolTip("Vorher/Nachher-Vergleich  |  \\")
        self._before_btn.toggled.connect(self._set_before_after)
        zoom_layout.addWidget(self._before_btn)

        self._split_btn = QPushButton("Geteilt")
        self._split_btn.setFixedWidth(60)
        self._split_btn.setCheckable(True)
        self._split_btn.setEnabled(False)
        self._split_btn.setToolTip("Vorher/Nachher nebeneinander (Split-Ansicht)")
        self._split_btn.toggled.connect(self._set_split_view)
        zoom_layout.addWidget(self._split_btn)

        vbox.addWidget(zoom_bar)
        return container

    # ── Controls-panel helpers ────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        return lbl

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    @staticmethod
    def _make_slider_row(label_text: str, value_label: QLabel) -> QWidget:
        """Compact row: left-aligned label + right-aligned value label."""
        row = QWidget()
        hbox = QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        hbox.addWidget(QLabel(label_text), stretch=1)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        value_label.setFixedWidth(58)
        hbox.addWidget(value_label)
        return row

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Navigator (always visible, outside scroll) ────────────────────
        self._navigator = _NavigatorWidget()
        outer.addWidget(self._navigator)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep0)

        # ── Histogram (always visible, outside scroll) ────────────────────
        self._histogram = _HistogramWidget()
        outer.addWidget(self._histogram)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        # ── Scrollable controls area ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        tip = "Doppelklick zum Zurücksetzen  |  R = alle zurücksetzen"

        # ── Profil ────────────────────────────────────────────────────────
        sec_profile = _CollapsibleSection("Profil")
        layout.addWidget(sec_profile)

        # Film-type selector
        film_type_row = QWidget()
        film_type_layout = QHBoxLayout(film_type_row)
        film_type_layout.setContentsMargins(0, 0, 0, 0)
        film_type_layout.setSpacing(8)
        self._film_c41_btn = QRadioButton("C-41 Farbe")
        self._film_bw_btn = QRadioButton("Schwarzweiß")
        self._film_c41_btn.setChecked(True)
        self._film_type_group = QButtonGroup(self)
        self._film_type_group.addButton(self._film_c41_btn)
        self._film_type_group.addButton(self._film_bw_btn)
        self._film_c41_btn.toggled.connect(self._on_film_type_changed)
        film_type_layout.addWidget(self._film_c41_btn)
        film_type_layout.addWidget(self._film_bw_btn)
        film_type_layout.addStretch()
        sec_profile.add_widget(film_type_row)

        profile_row = QWidget()
        profile_layout = QHBoxLayout(profile_row)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(4)

        self._profile_combo = QComboBox()
        self._profile_combo.setToolTip(
            "Filmtyp-Profil auswählen und anwenden\n"
            "Setzt Weißabgleich und Tonwertkurve auf Filmstock-Vorgaben"
        )
        self._profile_combo.addItem("(Kein Profil)")
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        profile_layout.addWidget(self._profile_combo, stretch=1)

        self._profile_reload_btn = QPushButton("↺")
        self._profile_reload_btn.setFixedWidth(28)
        self._profile_reload_btn.setToolTip("Profilliste neu einlesen")
        self._profile_reload_btn.clicked.connect(self._load_profiles)
        profile_layout.addWidget(self._profile_reload_btn)

        sec_profile.add_widget(profile_row)

        # ── Belichtung ────────────────────────────────────────────────────
        sec_exp = _CollapsibleSection("Belichtung")
        layout.addWidget(sec_exp)

        self._exposure_label = QLabel("0.0 EV")
        self._exposure_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._exposure_slider.setRange(-30, 30)
        self._exposure_slider.setValue(0)
        self._exposure_slider.setToolTip(tip)
        sec_exp.add_widget(self._make_slider_row("Belichtung (EV)", self._exposure_label))
        sec_exp.add_widget(self._exposure_slider)

        self._contrast_label = QLabel("0")
        self._contrast_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._contrast_slider.setRange(-100, 100)
        self._contrast_slider.setValue(0)
        self._contrast_slider.setToolTip(tip)
        sec_exp.add_widget(self._make_slider_row("Kontrast", self._contrast_label))
        sec_exp.add_widget(self._contrast_slider)

        self._highlights_label = QLabel("0")
        self._highlights_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._highlights_slider.setRange(-100, 100)
        self._highlights_slider.setValue(0)
        self._highlights_slider.setToolTip(tip)
        sec_exp.add_widget(self._make_slider_row("Glanzlichter", self._highlights_label))
        sec_exp.add_widget(self._highlights_slider)

        self._shadows_label = QLabel("0")
        self._shadows_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._shadows_slider.setRange(-100, 100)
        self._shadows_slider.setValue(0)
        self._shadows_slider.setToolTip(tip)
        sec_exp.add_widget(self._make_slider_row("Tiefen", self._shadows_label))
        sec_exp.add_widget(self._shadows_slider)

        self._black_label = QLabel("0.00")
        self._black_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._black_slider.setRange(0, 49)
        self._black_slider.setValue(0)
        self._black_slider.setToolTip(tip)
        sec_exp.add_widget(self._make_slider_row("Schwarzpunkt", self._black_label))
        sec_exp.add_widget(self._black_slider)

        self._white_label = QLabel("1.00")
        self._white_slider = _ResetSlider(Qt.Orientation.Horizontal, default=100)
        self._white_slider.setRange(51, 100)
        self._white_slider.setValue(100)
        self._white_slider.setToolTip(tip)
        sec_exp.add_widget(self._make_slider_row("Weißpunkt", self._white_label))
        sec_exp.add_widget(self._white_slider)

        self._auto_levels_btn = QPushButton("Auto-Belichtung")
        self._auto_levels_btn.setEnabled(False)
        self._auto_levels_btn.setToolTip(
            "Belichtung, Schwarzpunkt und Weißpunkt automatisch aus dem Histogramm ermitteln"
        )
        self._auto_levels_btn.clicked.connect(self._on_auto_levels)
        sec_exp.add_widget(self._auto_levels_btn)

        # ── Farbe ─────────────────────────────────────────────────────────
        sec_color = _CollapsibleSection("Farbe")
        layout.addWidget(sec_color)

        self._temp_label = QLabel("6500 K")
        self._temp_slider = _ResetSlider(Qt.Orientation.Horizontal, default=6500)
        self._temp_slider.setRange(2000, 12000)
        self._temp_slider.setValue(6500)
        self._temp_slider.setToolTip(tip)
        sec_color.add_widget(self._make_slider_row("Temperatur", self._temp_label))
        sec_color.add_widget(self._temp_slider)

        self._tint_label = QLabel("0")
        self._tint_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._tint_slider.setRange(-100, 100)
        self._tint_slider.setValue(0)
        self._tint_slider.setToolTip(tip)
        sec_color.add_widget(self._make_slider_row("Tönung", self._tint_label))
        sec_color.add_widget(self._tint_slider)

        self._pipette_btn = QPushButton("WB Pipette")
        self._pipette_btn.setCheckable(True)
        self._pipette_btn.setEnabled(False)
        self._pipette_btn.setToolTip(
            "Neutralen Bereich anklicken zum Setzen des Weißabgleichs  |  P"
        )
        self._pipette_btn.toggled.connect(self._set_pipette_mode)
        sec_color.add_widget(self._pipette_btn)

        self._auto_wb_btn = QPushButton("Auto-Weißabgleich")
        self._auto_wb_btn.setEnabled(False)
        self._auto_wb_btn.setToolTip(
            "Weißabgleich automatisch anhand des Bildmittelwerts setzen (Gray-World)"
        )
        self._auto_wb_btn.clicked.connect(self._on_auto_wb)
        sec_color.add_widget(self._auto_wb_btn)

        self._vibrance_label = QLabel("0")
        self._vibrance_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._vibrance_slider.setRange(-100, 100)
        self._vibrance_slider.setValue(0)
        self._vibrance_slider.setToolTip(tip)
        sec_color.add_widget(self._make_slider_row("Dynamik", self._vibrance_label))
        sec_color.add_widget(self._vibrance_slider)

        self._sat_label = QLabel("1.00×")
        self._sat_slider = _ResetSlider(Qt.Orientation.Horizontal, default=100)
        self._sat_slider.setRange(0, 200)
        self._sat_slider.setValue(100)
        self._sat_slider.setToolTip(tip)
        sec_color.add_widget(self._make_slider_row("Sättigung", self._sat_label))
        sec_color.add_widget(self._sat_slider)

        # ── Tonwertkurve ──────────────────────────────────────────────────
        sec_curve = _CollapsibleSection("Tonwertkurve")
        layout.addWidget(sec_curve)

        curve_ch_row = QWidget()
        curve_ch_layout = QHBoxLayout(curve_ch_row)
        curve_ch_layout.setContentsMargins(0, 0, 0, 0)
        curve_ch_layout.setSpacing(4)
        curve_ch_layout.addWidget(QLabel("Kanal:"))
        self._curve_ch_combo = QComboBox()
        for _lbl in _CURVE_CH_LABELS:
            self._curve_ch_combo.addItem(_lbl)
        self._curve_ch_combo.currentIndexChanged.connect(self._on_curve_channel_changed)
        curve_ch_layout.addWidget(self._curve_ch_combo, stretch=1)
        sec_curve.add_widget(curve_ch_row)

        self._curve_widget = _CurveWidget()
        self._curve_widget.curve_changed.connect(self._schedule_preview)
        sec_curve.add_widget(self._curve_widget)

        # ── Details ───────────────────────────────────────────────────────
        sec_detail = _CollapsibleSection("Details")
        layout.addWidget(sec_detail)

        self._sharp_label = QLabel("0")
        self._sharp_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._sharp_slider.setRange(0, 100)
        self._sharp_slider.setValue(0)
        self._sharp_slider.setToolTip(tip)
        sec_detail.add_widget(self._make_slider_row("Schärfen", self._sharp_label))
        sec_detail.add_widget(self._sharp_slider)

        self._noise_label = QLabel("0")
        self._noise_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._noise_slider.setRange(0, 100)
        self._noise_slider.setValue(0)
        self._noise_slider.setToolTip(tip)
        sec_detail.add_widget(self._make_slider_row("Entrauschen", self._noise_label))
        sec_detail.add_widget(self._noise_slider)

        self._vignette_label = QLabel("0")
        self._vignette_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._vignette_slider.setRange(-100, 100)
        self._vignette_slider.setValue(0)
        self._vignette_slider.setToolTip(tip)
        sec_detail.add_widget(self._make_slider_row("Vignettierung", self._vignette_label))
        sec_detail.add_widget(self._vignette_slider)

        self._grain_label = QLabel("0")
        self._grain_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._grain_slider.setRange(0, 100)
        self._grain_slider.setValue(0)
        self._grain_slider.setToolTip(tip)
        sec_detail.add_widget(self._make_slider_row("Körnung", self._grain_label))
        sec_detail.add_widget(self._grain_slider)

        # ── Split Toning ──────────────────────────────────────────────────
        sec_split = _CollapsibleSection("Split Toning", expanded=False)
        layout.addWidget(sec_split)

        self._st_shadow_hue_label = QLabel("0°")
        self._st_shadow_hue_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._st_shadow_hue_slider.setRange(0, 360)
        self._st_shadow_hue_slider.setValue(0)
        self._st_shadow_hue_slider.setToolTip(tip)
        sec_split.add_widget(self._make_slider_row("Schatten Farbton", self._st_shadow_hue_label))
        sec_split.add_widget(self._st_shadow_hue_slider)

        self._st_shadow_sat_label = QLabel("0")
        self._st_shadow_sat_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._st_shadow_sat_slider.setRange(0, 100)
        self._st_shadow_sat_slider.setValue(0)
        self._st_shadow_sat_slider.setToolTip(tip)
        sec_split.add_widget(self._make_slider_row("Schatten Stärke", self._st_shadow_sat_label))
        sec_split.add_widget(self._st_shadow_sat_slider)

        self._st_hi_hue_label = QLabel("0°")
        self._st_hi_hue_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._st_hi_hue_slider.setRange(0, 360)
        self._st_hi_hue_slider.setValue(0)
        self._st_hi_hue_slider.setToolTip(tip)
        sec_split.add_widget(self._make_slider_row("Lichter Farbton", self._st_hi_hue_label))
        sec_split.add_widget(self._st_hi_hue_slider)

        self._st_hi_sat_label = QLabel("0")
        self._st_hi_sat_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._st_hi_sat_slider.setRange(0, 100)
        self._st_hi_sat_slider.setValue(0)
        self._st_hi_sat_slider.setToolTip(tip)
        sec_split.add_widget(self._make_slider_row("Lichter Stärke", self._st_hi_sat_label))
        sec_split.add_widget(self._st_hi_sat_slider)

        self._st_balance_label = QLabel("0")
        self._st_balance_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._st_balance_slider.setRange(-100, 100)
        self._st_balance_slider.setValue(0)
        self._st_balance_slider.setToolTip(tip)
        sec_split.add_widget(self._make_slider_row("Balance", self._st_balance_label))
        sec_split.add_widget(self._st_balance_slider)

        # ── Drehen ────────────────────────────────────────────────────────
        sec_rot = _CollapsibleSection("Drehen", expanded=False)
        layout.addWidget(sec_rot)

        rot_row = QWidget()
        rot_layout = QHBoxLayout(rot_row)
        rot_layout.setContentsMargins(0, 0, 0, 0)
        rot_layout.setSpacing(4)
        self._rot_ccw_btn = QPushButton("↺ 90°")
        self._rot_ccw_btn.setEnabled(False)
        self._rot_ccw_btn.setToolTip("90° gegen den Uhrzeigersinn drehen")
        self._rot_ccw_btn.clicked.connect(self._rotate_ccw)
        rot_layout.addWidget(self._rot_ccw_btn)
        self._rot_cw_btn = QPushButton("↻ 90°")
        self._rot_cw_btn.setEnabled(False)
        self._rot_cw_btn.setToolTip("90° im Uhrzeigersinn drehen")
        self._rot_cw_btn.clicked.connect(self._rotate_cw)
        rot_layout.addWidget(self._rot_cw_btn)
        sec_rot.add_widget(rot_row)

        self._angle_label = QLabel("0.0°")
        self._angle_slider = _ResetSlider(Qt.Orientation.Horizontal, default=0)
        self._angle_slider.setRange(-150, 150)
        self._angle_slider.setValue(0)
        self._angle_slider.setToolTip(tip)
        sec_rot.add_widget(self._make_slider_row("Begradigung (°)", self._angle_label))
        sec_rot.add_widget(self._angle_slider)

        self._rot_undo_btn = QPushButton("Drehung zurücksetzen")
        self._rot_undo_btn.setEnabled(False)
        self._rot_undo_btn.setToolTip("Alle Drehungen und Zuschnitte zurücksetzen — stellt das ursprünglich geladene Bild wieder her")
        self._rot_undo_btn.clicked.connect(self._reset_crop_to_initial)
        sec_rot.add_widget(self._rot_undo_btn)

        # ── Zuschnitt ─────────────────────────────────────────────────────
        sec_crop = _CollapsibleSection("Zuschnitt", expanded=False)
        layout.addWidget(sec_crop)

        self._crop_btn = QPushButton("Bereich auswählen")
        self._crop_btn.setCheckable(True)
        self._crop_btn.setEnabled(False)
        self._crop_btn.setToolTip("Bereich mit der Maus zeichnen  |  C")
        self._crop_btn.toggled.connect(self._toggle_crop_mode)
        sec_crop.add_widget(self._crop_btn)

        self._aspect_combo = QComboBox()
        self._aspect_combo.setEnabled(False)
        _RATIOS = [
            ("Frei",  None),
            ("1:1",   1.0),
            ("4:3",   4 / 3),
            ("3:2",   3 / 2),
            ("16:9",  16 / 9),
            ("3:4",   3 / 4),
            ("2:3",   2 / 3),
            ("9:16",  9 / 16),
        ]
        for _lbl, _ in _RATIOS:
            self._aspect_combo.addItem(_lbl)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        self._aspect_ratios = _RATIOS
        sec_crop.add_widget(self._aspect_combo)

        crop_row = QWidget()
        crop_row_layout = QHBoxLayout(crop_row)
        crop_row_layout.setContentsMargins(0, 0, 0, 0)
        crop_row_layout.setSpacing(4)
        self._crop_apply_btn = QPushButton("Anwenden")
        self._crop_apply_btn.setEnabled(False)
        self._crop_apply_btn.clicked.connect(self._apply_crop)
        crop_row_layout.addWidget(self._crop_apply_btn)
        self._crop_undo_btn = QPushButton("Zurücksetzen")
        self._crop_undo_btn.setEnabled(False)
        self._crop_undo_btn.setToolTip("Alle Zuschnitte und Drehungen zurücksetzen — stellt das ursprünglich geladene Bild wieder her")
        self._crop_undo_btn.clicked.connect(self._reset_crop_to_initial)
        crop_row_layout.addWidget(self._crop_undo_btn)
        sec_crop.add_widget(crop_row)

        layout.addStretch()

        # ── Export (outside scroll, always visible) ───────────────────────
        export_bar = QWidget()
        export_layout = QVBoxLayout(export_bar)
        export_layout.setContentsMargins(6, 4, 6, 4)
        export_layout.setSpacing(4)

        undo_redo_row = QWidget()
        undo_redo_layout = QHBoxLayout(undo_redo_row)
        undo_redo_layout.setContentsMargins(0, 0, 0, 0)
        undo_redo_layout.setSpacing(4)

        _undo_redo_style = (
            "QPushButton { font-size: 15px; }"
            "QPushButton:enabled  { color: #d0d0d0; }"
            "QPushButton:disabled { color: #444444; }"
        )

        self._undo_btn = QPushButton("↩")
        self._undo_btn.setToolTip("Rückgängig  |  Strg+Z")
        self._undo_btn.setEnabled(False)
        self._undo_btn.setStyleSheet(_undo_redo_style)
        self._undo_btn.clicked.connect(self._undo_settings)

        self._redo_btn = QPushButton("↪")
        self._redo_btn.setToolTip("Wiederholen  |  Strg+Y")
        self._redo_btn.setEnabled(False)
        self._redo_btn.setStyleSheet(_undo_redo_style)
        self._redo_btn.clicked.connect(self._redo_settings)

        undo_redo_layout.addWidget(self._undo_btn, stretch=1)
        undo_redo_layout.addWidget(self._redo_btn, stretch=1)
        export_layout.addWidget(undo_redo_row)

        reset_btn = QPushButton("Zurücksetzen")
        reset_btn.setToolTip("Alle Regler auf Standardwerte zurücksetzen  |  R")
        reset_btn.clicked.connect(self._reset_all_sliders)
        export_layout.addWidget(reset_btn)

        export_row = QWidget()
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        export_row_layout.setSpacing(4)

        self._export_btn = QPushButton("Export…")
        self._set_export_enabled(False)
        self._export_btn.clicked.connect(self._export_image)
        export_row_layout.addWidget(self._export_btn, stretch=1)

        self._batch_btn = QPushButton("Batch-Export…")
        self._set_batch_enabled(False)
        self._batch_btn.setToolTip(
            "Ausgewählte Bilder mit ihren gespeicherten Einstellungen exportieren"
        )
        self._batch_btn.clicked.connect(self._batch_export)
        export_row_layout.addWidget(self._batch_btn, stretch=1)

        export_layout.addWidget(export_row)
        outer.addWidget(export_bar)

        self._curve_widget.curve_editing_started.connect(self._push_undo_current)

        return panel

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _exposure(self) -> float:
        return self._exposure_slider.value() / 10.0

    def _white_balance(self) -> tuple[float, float, float]:
        return temp_tint_to_rgb_multipliers(
            self._temp_slider.value(),
            self._tint_slider.value(),
        )

    def _black_point(self) -> float:
        return self._black_slider.value() / 100.0

    def _white_point(self) -> float:
        return self._white_slider.value() / 100.0

    def _saturation(self) -> float:
        return self._sat_slider.value() / 100.0

    def _contrast(self) -> float:
        return self._contrast_slider.value() / 100.0

    def _highlights(self) -> float:
        return self._highlights_slider.value() / 100.0

    def _shadows(self) -> float:
        return self._shadows_slider.value() / 100.0

    def _vibrance(self) -> float:
        return self._vibrance_slider.value() / 100.0

    def _sharpening(self) -> float:
        return self._sharp_slider.value() / 100.0

    def _noise_reduction(self) -> float:
        return self._noise_slider.value() / 100.0

    def _vignette(self) -> float:
        return self._vignette_slider.value() / 100.0

    def _grain(self) -> float:
        return self._grain_slider.value() / 100.0

    def _angle(self) -> float:
        return self._angle_slider.value() / 10.0

    def _split_toning_params(self) -> tuple[float, float, float, float, float] | None:
        """Return (shadow_hue, shadow_sat, hi_hue, hi_sat, balance) or None if inactive."""
        sh_sat = self._st_shadow_sat_slider.value() / 100.0
        hi_sat = self._st_hi_sat_slider.value() / 100.0
        if sh_sat < 1e-4 and hi_sat < 1e-4:
            return None
        return (
            float(self._st_shadow_hue_slider.value()),
            sh_sat,
            float(self._st_hi_hue_slider.value()),
            hi_sat,
            self._st_balance_slider.value() / 100.0,
        )

    def _make_clipping_pixmap(self) -> QPixmap:
        """Build a display pixmap with red highlight / blue shadow clipping indicators."""
        arr = self._display_uint8.copy()
        # Any channel >= 252 → fully saturated highlight → red
        hl_mask = arr.max(axis=2) >= 252
        # All channels <= 3 → crushed shadow → blue
        sh_mask = arr.max(axis=2) <= 3
        arr[hl_mask] = [255, 0, 0]
        arr[sh_mask] = [0, 0, 255]
        h, w = arr.shape[:2]
        qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ── Import ────────────────────────────────────────────────────────────────

    def _import_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Film Negatives", "", _IMPORT_FILTER
        )
        if paths:
            self._start_import(paths)

    def _start_import(self, paths: list[str]) -> None:
        new_paths = [p for p in paths if p not in self._entries]
        if not new_paths:
            self._status.showMessage("All selected images are already imported.")
            return
        for p in new_paths:
            self._entries[p] = _ImageEntry(path=p, film_type=self._app_settings.default_film_type)
        self._status.showMessage(f"Importing {len(new_paths)} image(s)…")
        worker = _ImportWorker(new_paths, self._app_settings.preview_long_edge)
        worker.image_ready.connect(self._on_image_ready)
        worker.error.connect(self._on_import_error)
        worker.all_done.connect(self._on_import_done)
        self._import_worker = worker
        worker.start()

    # ── Drag & drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(
            u.isLocalFile()
            and Path(u.toLocalFile()).suffix.lower() in _IMPORT_EXTENSIONS
            for u in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [
            u.toLocalFile()
            for u in event.mimeData().urls()
            if u.isLocalFile()
            and Path(u.toLocalFile()).suffix.lower() in _IMPORT_EXTENSIONS
        ]
        if paths:
            self._start_import(paths)
            event.acceptProposedAction()

    def _on_image_ready(
        self,
        path: str,
        thumb_uint8: np.ndarray,
        preview: np.ndarray,
        inverted: np.ndarray,
        mask: tuple,
        exif_bytes: bytes | None,
    ) -> None:
        entry = self._entries[path]
        entry.raw_preview = preview
        entry.inverted_preview = inverted
        entry.orange_mask = mask
        entry.exif_bytes = exif_bytes

        thumb_pix = _array_to_pixmap(thumb_uint8)
        item = QListWidgetItem(QIcon(thumb_pix), Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._filmstrip.addItem(item)

        self._set_batch_enabled(True)
        self._copy_settings_btn.setEnabled(True)
        self._remove_btn.setEnabled(True)
        self._select_all_btn.setEnabled(True)
        self._reset_selected_btn.setEnabled(True)
        self._pipette_btn.setEnabled(True)
        self._auto_wb_btn.setEnabled(True)
        self._auto_levels_btn.setEnabled(True)
        self._set_before_enabled(True)
        self._split_btn.setEnabled(True)
        self._clipping_btn.setEnabled(True)
        self._crop_btn.setEnabled(True)
        self._aspect_combo.setEnabled(True)
        self._rot_ccw_btn.setEnabled(True)
        self._rot_cw_btn.setEnabled(True)
        self._set_save_session_enabled(True)
        self._status.showMessage(f"Loaded: {Path(path).name}")
        # Restore correct enabled-state for colour controls in case the active
        # image is B&W (unconditional setEnabled(True) above would override it).
        self._update_color_controls_enabled()

        # Apply pending session settings for this path (restored from file)
        if self._pending_session_settings and path in self._pending_session_settings:
            saved = self._pending_session_settings.pop(path)
            entry = self._entries[path]
            self._session_manager.apply_entry_data(entry, saved)
            # Re-invert if film type differs from default c41
            if entry.film_type == "bw" and entry.raw_preview is not None:
                entry.inverted_preview = invert_bw(entry.raw_preview)
                self._inverted_preview = entry.inverted_preview

        # Auto-activate the first imported image (or the saved active path)
        if self._active_path is None:
            activate_now = (
                self._pending_session_active == path
                if self._pending_session_active
                else True
            )
            if activate_now:
                self._filmstrip.setCurrentItem(item)
                self._activate_path(path)

    def _on_import_error(self, path: str, msg: str) -> None:
        self._status.showMessage(f"Import error ({Path(path).name}): {msg}")
        # Remove the reserved entry so the user can retry
        self._entries.pop(path, None)

    def _on_import_done(self) -> None:
        count = self._filmstrip.count()
        self._status.showMessage(f"{count} image(s) in filmstrip — ready.")
        # If session restore is complete, activate the saved active path
        if self._pending_session_active and self._pending_session_active in self._entries:
            active = self._pending_session_active
            self._pending_session_active = None
            self._pending_session_settings = None
            # Find and select the item in the filmstrip
            for i in range(self._filmstrip.count()):
                item = self._filmstrip.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == active:
                    self._filmstrip.setCurrentItem(item)
                    self._activate_path(active)
                    break
        else:
            self._pending_session_active = None
            self._pending_session_settings = None

    # ── Filmstrip interaction ─────────────────────────────────────────────────

    def _on_filmstrip_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        self._activate_path(path)

    def _on_filmstrip_selection_changed(self) -> None:
        """Keep the 'Alle wählen' button label in sync with the current selection."""
        count = self._filmstrip.count()
        if count > 0 and len(self._filmstrip.selectedItems()) == count:
            self._select_all_btn.setText("Auswahl löschen")
        else:
            self._select_all_btn.setText("Alle wählen")

    def _activate_path(self, path: str) -> None:
        """Switch the main viewer to the given image path."""
        if path == self._active_path:
            return

        # Save current settings to the outgoing entry before switching.
        if self._active_path and self._active_path in self._entries:
            self._entries[self._active_path].settings = self._read_settings()
            self._entries[self._active_path].raw_image = None

        self._active_path = path
        entry = self._entries.get(path)
        if entry is None:
            return

        self._raw_image = None          # not yet loaded for export
        self._inverted_preview = entry.inverted_preview
        self._orange_mask = entry.orange_mask
        self._set_export_enabled(False)

        # Reset before/after and split-view mode on image switch.
        # Block signals: the toggled() handlers call _fit_pixmap_to_label() /
        # _update_preview(), which must not fire here before the new image is ready.
        self._before_after = False
        self._before_btn.blockSignals(True)
        self._before_btn.setChecked(False)
        self._before_btn.blockSignals(False)
        self._before_btn.setText("Vorher")
        self._before_pixmap = None
        self._split_view = False
        self._split_btn.blockSignals(True)
        self._split_btn.setChecked(False)
        self._split_btn.blockSignals(False)

        # Clear editing undo/redo history — each image has its own independent history.
        self._history.clear_all()
        self._pre_drag_settings = None
        self._sync_undo_redo_btns()

        # Sync crop UI to the newly active entry
        self._crop_sel_norm = None
        self._crop_apply_btn.setEnabled(False)
        self._crop_undo_btn.setEnabled(bool(entry.crop_history))
        self._update_crop_overlay()

        # Sync film-type radio buttons to the incoming entry (block to avoid re-inversion).
        self._film_c41_btn.blockSignals(True)
        self._film_bw_btn.blockSignals(True)
        self._film_c41_btn.setChecked(entry.film_type == "c41")
        self._film_bw_btn.setChecked(entry.film_type == "bw")
        self._film_c41_btn.blockSignals(False)
        self._film_bw_btn.blockSignals(False)
        self._filter_profiles_by_type(entry.film_type)
        self._update_color_controls_enabled()

        # Apply the entry's saved settings and refresh the preview.
        if self._inverted_preview is not None:
            self._apply_settings(entry.settings)

        # Load full-res in background so Export can be enabled.
        # Before creating the new loader, disconnect and park the old one so it
        # is not GC'd while still running (Qt ABRT: QObject destroyed in wrong
        # thread).  QThread.finished fires on the main thread after run() exits,
        # at which point it is safe to release the reference.
        if self._full_loader is not None:
            self._full_loader.requestInterruption()
            try:
                self._full_loader.loaded.disconnect(self._on_full_image_loaded)
                self._full_loader.error.disconnect(self._on_load_error)
            except RuntimeError:
                pass  # already disconnected
            old = self._full_loader
            self._old_loaders.append(old)
            old.finished.connect(
                lambda o=old: self._old_loaders.remove(o)
                if o in self._old_loaders else None
            )
            self._full_loader = None

        self._status.showMessage(f"Loading: {Path(path).name}…")
        loader = _FullImageLoader(path)
        loader.loaded.connect(self._on_full_image_loaded)
        loader.error.connect(self._on_load_error)
        self._full_loader = loader
        loader.start()

    def _on_full_image_loaded(self, path: str, raw: np.ndarray) -> None:
        # Discard result if the user switched to another image in the meantime
        if path != self._active_path:
            return

        self._raw_image = raw
        entry = self._entries.get(path)
        if entry:
            entry.raw_image = raw

        self._set_export_enabled(True)
        self._status.showMessage(f"Ready: {Path(path).name}")

    def _on_load_error(self, path: str, msg: str) -> None:
        self._status.showMessage(f"Load error ({Path(path).name}): {msg}")

    # ── Slider slots ──────────────────────────────────────────────────────────

    def _on_auto_levels(self) -> None:
        """
        Analyse the current inverted preview (after applying current WB) and
        set Belichtung, Schwarzpunkt, and Weißpunkt to suggested values.
        """
        if self._inverted_preview is None:
            return

        # Apply current WB so the analysis sees the same colours as the viewer.
        analysis = apply_white_balance(self._inverted_preview, self._white_balance())
        exp_int, black_int, white_int = suggest_auto_levels(analysis)

        self._push_undo_current()

        # Block intermediate signals so _update_preview fires only once.
        for sl in (self._exposure_slider, self._black_slider, self._white_slider):
            sl.blockSignals(True)
        self._exposure_slider.setValue(exp_int)
        self._black_slider.setValue(black_int)
        self._white_slider.setValue(white_int)
        for sl in (self._exposure_slider, self._black_slider, self._white_slider):
            sl.blockSignals(False)

        # Sync labels and refresh once.
        self._exposure_label.setText(f"{exp_int / 10.0:+.1f} EV")
        self._black_label.setText(f"{black_int / 100.0:.2f}")
        self._white_label.setText(f"{white_int / 100.0:.2f}")
        self._update_preview()
        self._status.showMessage(
            f"Auto-Belichtung: {exp_int / 10.0:+.1f} EV  "
            f"Schwarz {black_int / 100.0:.2f}  Weiß {white_int / 100.0:.2f}"
        )

    def _on_angle_changed(self, value: int) -> None:
        self._angle_label.setText(f"{value / 10.0:+.1f}°")
        entry = self._entries.get(self._active_path)
        has_history = bool(entry.crop_history) if entry else False
        self._rot_undo_btn.setEnabled(value != 0 or has_history)
        self._sync_undo_redo_btns()
        self._schedule_preview()

    def _on_curve_channel_changed(self, index: int) -> None:
        self._curve_widget.set_active_channel(_CURVE_CHANNELS[index])

    def _on_wb_changed(self) -> None:
        temp = self._temp_slider.value()
        tint = self._tint_slider.value()
        self._temp_label.setText(f"{temp} K")
        self._tint_label.setText(f"{tint:+d}" if tint != 0 else "0")
        self._schedule_preview()

    def _on_contrast_changed(self, value: int) -> None:
        self._contrast_label.setText(f"{value:+d}" if value != 0 else "0")
        self._schedule_preview()

    def _on_highlights_changed(self, value: int) -> None:
        self._highlights_label.setText(f"{value:+d}" if value != 0 else "0")
        self._schedule_preview()

    def _on_shadows_changed(self, value: int) -> None:
        self._shadows_label.setText(f"{value:+d}" if value != 0 else "0")
        self._schedule_preview()

    def _on_vibrance_changed(self, value: int) -> None:
        self._vibrance_label.setText(f"{value:+d}" if value != 0 else "0")
        self._schedule_preview()

    def _on_noise_changed(self, value: int) -> None:
        self._noise_label.setText(str(value))
        self._schedule_preview()

    def _on_vignette_changed(self, value: int) -> None:
        self._vignette_label.setText(f"{value:+d}" if value != 0 else "0")
        self._schedule_preview()

    def _on_grain_changed(self, value: int) -> None:
        self._grain_label.setText(str(value))
        self._schedule_preview()

    def _on_st_changed(self) -> None:
        self._st_shadow_hue_label.setText(f"{self._st_shadow_hue_slider.value()}°")
        self._st_shadow_sat_label.setText(str(self._st_shadow_sat_slider.value()))
        self._st_hi_hue_label.setText(f"{self._st_hi_hue_slider.value()}°")
        self._st_hi_sat_label.setText(str(self._st_hi_sat_slider.value()))
        v = self._st_balance_slider.value()
        self._st_balance_label.setText(f"{v:+d}" if v != 0 else "0")
        self._schedule_preview()

    def _set_clipping_mode(self, active: bool) -> None:
        self._clipping_mode = active
        self._fit_pixmap_to_label()

    # ── Preview rendering ─────────────────────────────────────────────────────

    def _update_preview(self) -> None:
        """
        Apply the full editing chain to the cached inverted preview.

        Order: WB → Exposure → Levels → Saturation → Curves → Sharpening
               → (free angle rotation)

        Inversion is NOT re-run — it is cached in _inverted_preview.
        """
        if self._inverted_preview is None:
            return

        result = apply_white_balance(self._inverted_preview, self._white_balance())
        result = apply_exposure(result, self._exposure())

        contrast = self._contrast()
        if abs(contrast) > 1e-4:
            result = apply_contrast(result, contrast)

        highlights = self._highlights()
        if abs(highlights) > 1e-4:
            result = apply_highlights(result, highlights)

        shadows = self._shadows()
        if abs(shadows) > 1e-4:
            result = apply_shadows(result, shadows)

        bp, wp = self._black_point(), self._white_point()
        if bp > 0.0 or wp < 1.0:
            result = apply_levels(result, bp, wp)

        vibrance = self._vibrance()
        if abs(vibrance) > 1e-4:
            result = apply_vibrance(result, vibrance)

        sat = self._saturation()
        if abs(sat - 1.0) > 1e-4:
            result = apply_saturation(result, sat)

        st = self._split_toning_params()
        if st is not None:
            result = apply_split_toning(result, *st)

        if not self._curve_widget.is_all_identity():
            result = apply_curves_fast(result, self._curve_widget.combined_luts)

        noise_red = self._noise_reduction()
        if noise_red > 1e-4:
            result = apply_noise_reduction(result, noise_red)

        sharp = self._sharpening()
        if sharp > 1e-4:
            result = apply_sharpening(result, sharp)

        vig = self._vignette()
        if abs(vig) > 1e-4:
            result = apply_vignette(result, vig)

        grain = self._grain()
        if grain > 1e-4:
            result = apply_grain(result, grain)

        uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)

        angle = self._angle()
        if abs(angle) > 0.05:
            uint8 = _rotate_uint8(uint8, angle)

        self._display_uint8 = uint8
        h, w = uint8.shape[:2]
        qimg = QImage(uint8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._processed_pixmap = QPixmap.fromImage(qimg)
        self._histogram.update_from_uint8(uint8)
        self._fit_pixmap_to_label()

    def _fit_pixmap_to_label(self) -> None:
        """Render _processed_pixmap into the scroll area according to current zoom mode."""
        if self._processed_pixmap is None:
            return

        # ── Choose base pixmap / build split composite ────────────────────
        # "after" pixmap: clean or with clipping overlay
        after_pix = (
            self._make_clipping_pixmap()
            if self._clipping_mode and self._display_uint8 is not None
            else self._processed_pixmap
        )
        if self._split_view and self._before_pixmap is not None:
            pix = after_pix
        elif self._before_after and self._before_pixmap is not None:
            pix = self._before_pixmap
        else:
            pix = after_pix

        vp = self._scroll.viewport().size()

        if self._zoom_fit_mode:
            # Scale after-pixmap to fit the viewport.
            scaled = pix.scaled(
                vp,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            if self._split_view and self._before_pixmap is not None:
                scaled = self._make_split_composite(scaled)

            # Clear any fixed-size constraint left by fixed-zoom mode;
            # setWidgetResizable(True) cannot resize a widget past its fixed bounds.
            self._image_label.setMinimumSize(0, 0)
            self._image_label.setMaximumSize(16777215, 16777215)
            self._scroll.setWidgetResizable(True)
            self._image_label.setPixmap(scaled)

            eff_pct = round(scaled.width() / pix.width() * 100) if pix.width() > 0 else 100
            self._zoom_label.setText("Fit")
            # Move slider to reflect effective zoom without triggering _on_zoom_slider
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(max(10, min(400, eff_pct)))
            self._zoom_slider.blockSignals(False)
        else:
            # Fixed zoom: label is sized to the scaled image; scrollbars appear if needed.
            z = self._zoom_pct / 100.0
            scaled_w = max(1, round(pix.width() * z))
            scaled_h = max(1, round(pix.height() * z))
            scaled = pix.scaled(
                scaled_w,
                scaled_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            if self._split_view and self._before_pixmap is not None:
                scaled = self._make_split_composite(scaled)

            self._scroll.setWidgetResizable(False)
            self._image_label.setFixedSize(scaled.width(), scaled.height())
            self._image_label.setPixmap(scaled)
            self._zoom_label.setText(f"{self._zoom_pct} %")

        self._update_crop_overlay()
        self._update_navigator()

    # ── Zoom slots ────────────────────────────────────────────────────────────

    def _zoom_fit(self) -> None:
        """Reset to fit-in-window mode."""
        self._zoom_fit_mode = True
        self._zoom_label.setText("Fit")
        self._fit_pixmap_to_label()

    def _zoom_100(self) -> None:
        """Jump directly to 100% zoom."""
        self._zoom_slider.setValue(100)

    def _on_zoom_slider(self, value: int) -> None:
        """Slider moved by the user — switch to fixed-zoom mode."""
        self._zoom_fit_mode = False
        self._zoom_pct = value
        self._fit_pixmap_to_label()

    def _on_wheel_zoom(self, step: int, mouse_pos: QPoint) -> None:
        """Mouse-wheel zoom: step to the nearest preset level, anchored to the cursor."""
        pix = self._processed_pixmap
        if pix is None:
            return

        # In fit mode, compute the scaled size once — reused for both the
        # current-zoom calculation and the cursor-anchor fraction below.
        fitted_size: tuple[int, int] | None = None
        if self._zoom_fit_mode:
            vp = self._scroll.viewport().size()
            fitted = pix.scaled(
                vp,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            fitted_size = (fitted.width(), fitted.height())
            current = round(fitted_size[0] / pix.width() * 100) if pix.width() > 0 else 100
        else:
            current = self._zoom_pct

        # Find the next preset above/below current
        if step > 0:
            candidates = [z for z in _ZOOM_STEPS if z > current]
            new_pct = candidates[0] if candidates else _ZOOM_STEPS[-1]
        else:
            candidates = [z for z in _ZOOM_STEPS if z < current]
            new_pct = candidates[-1] if candidates else _ZOOM_STEPS[0]

        # Compute image-space fraction under the cursor before zooming, so we
        # can scroll to keep that point under the cursor after the zoom.
        frac_x, frac_y = 0.5, 0.5
        if pix.width() > 0 and pix.height() > 0:
            if self._zoom_fit_mode and fitted_size is not None:
                # Fit mode: image is centred inside the viewport with possible letterboxing.
                vp = self._scroll.viewport().size()
                img_w, img_h = fitted_size
                off_x = (vp.width() - img_w) / 2.0
                off_y = (vp.height() - img_h) / 2.0
                cx = mouse_pos.x() - off_x
                cy = mouse_pos.y() - off_y
                frac_x = max(0.0, min(1.0, cx / img_w)) if img_w > 0 else 0.5
                frac_y = max(0.0, min(1.0, cy / img_h)) if img_h > 0 else 0.5
            else:
                # Fixed mode: cursor position in image-label coordinates.
                hval = self._scroll.horizontalScrollBar().value()
                vval = self._scroll.verticalScrollBar().value()
                label_w = self._image_label.width()
                label_h = self._image_label.height()
                cx = hval + mouse_pos.x()
                cy = vval + mouse_pos.y()
                frac_x = cx / label_w if label_w > 0 else 0.5
                frac_y = cy / label_h if label_h > 0 else 0.5

        self._zoom_fit_mode = False
        self._zoom_pct = new_pct
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(new_pct)
        self._zoom_slider.blockSignals(False)
        self._fit_pixmap_to_label()  # sets _zoom_label to f"{new_pct} %"

        # After zoom, scroll so the image point that was under the cursor remains
        # under the cursor.  Use the pixmap dimensions to compute the new label
        # size directly (same formula as _fit_pixmap_to_label) so we don't have
        # to wait for a layout pass.
        if pix.width() > 0:
            new_label_w = max(1, round(pix.width() * new_pct / 100))
            new_label_h = max(1, round(pix.height() * new_pct / 100))
            self._scroll.horizontalScrollBar().setValue(
                max(0, round(frac_x * new_label_w - mouse_pos.x()))
            )
            self._scroll.verticalScrollBar().setValue(
                max(0, round(frac_y * new_label_h - mouse_pos.y()))
            )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_pixmap_to_label()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._histogram.shutdown_worker()
        super().closeEvent(event)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _setup_shortcuts(self) -> None:
        def _bind(key: str, slot) -> None:
            QShortcut(QKeySequence(key), self).activated.connect(slot)

        _bind("Left",        self._filmstrip_prev)
        _bind("Right",       self._filmstrip_next)
        _bind("P",           self._toggle_pipette)
        _bind("C",           self._shortcut_crop)
        _bind("Return",      self._shortcut_crop_apply)
        _bind("Enter",       self._shortcut_crop_apply)
        _bind("Z",           self._reset_crop_to_initial)
        _bind("+",           lambda: self._keyboard_zoom(+1))
        _bind("=",           lambda: self._keyboard_zoom(+1))  # unshifted + on most keyboards
        _bind("-",           lambda: self._keyboard_zoom(-1))
        _bind("Escape",      self._shortcut_escape)
        _bind("Ctrl+A",      self._toggle_select_all)
        _bind("Delete",      self._remove_selected)
        # Note: R, F, \, Ctrl+Z, Ctrl+Y, Ctrl+E, Ctrl+Shift+E are handled via menu QActions

    def _schedule_preview(self) -> None:
        """Start (or restart) the debounce timer — preview fires 50 ms after the last change."""
        self._preview_timer.start()

    def _filmstrip_prev(self) -> None:
        """Activate the previous image in the filmstrip."""
        count = self._filmstrip.count()
        if count == 0:
            return
        row = max(0, self._filmstrip.currentRow() - 1)
        self._filmstrip.setCurrentRow(row)
        path = self._filmstrip.item(row).data(Qt.ItemDataRole.UserRole)
        self._activate_path(path)

    def _filmstrip_next(self) -> None:
        """Activate the next image in the filmstrip."""
        count = self._filmstrip.count()
        if count == 0:
            return
        row = min(count - 1, self._filmstrip.currentRow() + 1)
        self._filmstrip.setCurrentRow(row)
        path = self._filmstrip.item(row).data(Qt.ItemDataRole.UserRole)
        self._activate_path(path)

    # ── Undo / Redo ───────────────────────────────────────────────────────────

    def _push_undo_current(self) -> None:
        if self._inverted_preview is None:
            return
        self._history.push(self._read_settings())
        self._sync_undo_redo_btns()

    def _on_any_slider_pressed(self) -> None:
        if self._inverted_preview is not None:
            self._pre_drag_settings = self._read_settings()

    def _on_any_slider_released(self) -> None:
        if self._pre_drag_settings is None:
            return
        if self._read_settings() != self._pre_drag_settings:
            self._history.push(self._pre_drag_settings)
            self._sync_undo_redo_btns()
        self._pre_drag_settings = None

    def _undo_settings(self) -> None:
        if self._history.has_undo() and self._inverted_preview is not None:
            prev = self._history.undo_step(self._read_settings())
            self._apply_settings(prev)
            self._sync_undo_redo_btns()
        else:
            self._undo_crop()

    def _redo_settings(self) -> None:
        if not self._history.has_redo() or self._inverted_preview is None:
            return
        nxt = self._history.redo_step(self._read_settings())
        self._apply_settings(nxt)
        self._sync_undo_redo_btns()

    def _sync_undo_redo_btns(self) -> None:
        entry = self._entries.get(self._active_path) if self._active_path else None
        has_crop = bool(entry.crop_history) if entry else False
        has_angle = self._angle_slider.value() != 0
        has_undo = self._history.has_undo() or has_crop or has_angle
        has_redo = self._history.has_redo()
        self._undo_btn.setEnabled(has_undo)
        self._redo_btn.setEnabled(has_redo)
        self._act_undo_menu.setEnabled(has_undo)
        self._act_redo_menu.setEnabled(has_redo)

    def _reset_all_sliders(self) -> None:
        """Reset every editing slider to its default value."""
        self._push_undo_current()
        for defn in self._slider_defs:
            defn.slider.setValue(defn.slider.default)
        self._curve_widget.reset_all()

    # ── Slider definitions ────────────────────────────────────────────────────

    def _setup_slider_defs(self) -> None:
        """Build the canonical slider registry and wire up all slider signals."""
        self._slider_defs: list[_SliderDef] = [
            _SliderDef(self._exposure_slider,      self._exposure_label,      "exposure",           lambda v: f"{v/10.0:+.1f} EV"),
            _SliderDef(self._contrast_slider,      self._contrast_label,      "contrast",           _fmt_signed),
            _SliderDef(self._highlights_slider,    self._highlights_label,    "highlights",         _fmt_signed),
            _SliderDef(self._shadows_slider,       self._shadows_label,       "shadows",            _fmt_signed),
            _SliderDef(self._black_slider,         self._black_label,         "black_point",        lambda v: f"{v/100.0:.2f}"),
            _SliderDef(self._white_slider,         self._white_label,         "white_point",        lambda v: f"{v/100.0:.2f}"),
            _SliderDef(self._temp_slider,          self._temp_label,          "wb_temp",            lambda v: f"{v} K"),
            _SliderDef(self._tint_slider,          self._tint_label,          "wb_tint",            _fmt_signed),
            _SliderDef(self._vibrance_slider,      self._vibrance_label,      "vibrance",           _fmt_signed),
            _SliderDef(self._sat_slider,           self._sat_label,           "saturation",         lambda v: f"{v/100.0:.2f}×"),
            _SliderDef(self._sharp_slider,         self._sharp_label,         "sharpening",         str),
            _SliderDef(self._noise_slider,         self._noise_label,         "noise_reduction",    str),
            _SliderDef(self._vignette_slider,      self._vignette_label,      "vignette",           _fmt_signed),
            _SliderDef(self._grain_slider,         self._grain_label,         "grain",              str),
            _SliderDef(self._angle_slider,         self._angle_label,         "angle",              lambda v: f"{v/10.0:+.1f}°"),
            _SliderDef(self._st_shadow_hue_slider, self._st_shadow_hue_label, "split_shadow_hue",   lambda v: f"{v}°"),
            _SliderDef(self._st_shadow_sat_slider, self._st_shadow_sat_label, "split_shadow_sat",   str),
            _SliderDef(self._st_hi_hue_slider,     self._st_hi_hue_label,     "split_highlight_hue", lambda v: f"{v}°"),
            _SliderDef(self._st_hi_sat_slider,     self._st_hi_sat_label,     "split_highlight_sat", str),
            _SliderDef(self._st_balance_slider,    self._st_balance_label,    "split_balance",      _fmt_signed),
        ]
        for defn in self._slider_defs:
            defn.slider.sliderPressed.connect(self._on_any_slider_pressed)
            defn.slider.sliderReleased.connect(self._on_any_slider_released)
            defn.slider.about_to_reset.connect(self._push_undo_current)
            if defn.slider is self._angle_slider:
                defn.slider.valueChanged.connect(self._on_angle_changed)
            else:
                defn.slider.valueChanged.connect(
                    lambda v, d=defn: (d.label.setText(d.fmt(v)), self._schedule_preview())
                )

    # ── Profiles ──────────────────────────────────────────────────────────────

    def _load_profiles(self) -> None:
        """Scan built-in and user profile directories and store all profiles."""
        from luminos.profiles.loader import list_profiles

        self._profile_paths = {}
        self._profile_types = {}
        for name, path, ptype in list_profiles():
            self._profile_paths[name] = path
            self._profile_types[name] = ptype
        self._filter_profiles_by_type()

    def _filter_profiles_by_type(self, film_type: str | None = None) -> None:
        """Repopulate the profile combo showing only profiles matching the current film type."""
        if film_type is None:
            film_type = "bw" if self._film_bw_btn.isChecked() else "c41"
        target_type = "bw_negative" if film_type == "bw" else "color_negative"

        self._profile_combo.blockSignals(True)
        current_text = self._profile_combo.currentText()
        self._profile_combo.clear()
        self._profile_combo.addItem("(Kein Profil)")
        for name in sorted(self._profile_paths.keys(), key=str.lower):
            if self._profile_types.get(name) == target_type:
                self._profile_combo.addItem(name)
        idx = self._profile_combo.findText(current_text)
        self._profile_combo.setCurrentIndex(max(0, idx))
        self._profile_combo.blockSignals(False)

    def _on_profile_selected(self, index: int) -> None:
        """Auto-apply the selected profile when the combo selection changes."""
        if index <= 0:
            return
        self._apply_profile(self._profile_combo.itemText(index))

    def _on_film_type_changed(self) -> None:
        """
        Re-invert the active image with the newly selected film process.

        C-41: orange-mask compensation → colour positive.
        B&W:  luminance inversion, no mask → greyscale positive (RGB equal channels).

        The raw_preview is always kept so switching is lossless.
        Crop history is cleared because the new inverted base is a different array.
        """
        new_type = "bw" if self._film_bw_btn.isChecked() else "c41"
        self._filter_profiles_by_type(new_type)

        entry = self._entries.get(self._active_path)
        if entry is None or entry.raw_preview is None:
            return

        if entry.film_type == new_type:
            return

        if new_type == "bw":
            new_inv = invert_bw(entry.raw_preview)
            entry.orange_mask = None
        else:
            mask = detect_orange_mask(entry.raw_preview)
            new_inv = invert(entry.raw_preview, mask=mask)
            entry.orange_mask = mask

        entry.film_type = new_type
        entry.inverted_preview = new_inv
        self._inverted_preview = new_inv
        self._orange_mask = entry.orange_mask

        # Crop history references the old inverted array — clear it.
        entry.crop_history.clear()
        self._crop_undo_btn.setEnabled(False)
        self._sync_undo_redo_btns()

        self._invalidate_before_pixmap()
        self._update_color_controls_enabled()
        self._update_preview()
        self._status.showMessage(
            f"Filmtyp: {'Schwarzweiß' if new_type == 'bw' else 'C-41 Farbe'}"
        )

    def _update_color_controls_enabled(self) -> None:
        """Grey out colour-specific controls when the active image is B&W."""
        entry = self._entries.get(self._active_path)
        is_bw = entry is not None and entry.film_type == "bw"
        for widget in (
            self._temp_slider,
            self._tint_slider,
            self._pipette_btn,
            self._auto_wb_btn,
            self._vibrance_slider,
            self._sat_slider,
        ):
            widget.setEnabled(not is_bw)

    def _apply_profile(self, name: str) -> None:
        """
        Apply a named profile to the active image.

        Steps:
          1. If the profile provides ``orange_mask``, re-invert the cached
             raw_preview with that mask and update the entry's inverted_preview.
          2. Apply ``white_balance`` to the WB sliders.
          3. Apply ``tone_curve`` control points to the master curve channel.
          4. Refresh the preview.
        """
        if self._inverted_preview is None:
            return

        path = self._profile_paths.get(name)
        if path is None:
            return

        try:
            from luminos.profiles.loader import load_profile
            profile = load_profile(path)
        except Exception as exc:  # noqa: BLE001
            self._status.showMessage(f"Profil-Fehler: {exc}")
            return

        self._push_undo_current()

        # ── Orange mask re-inversion (optional) ───────────────────────────
        if "orange_mask" in profile:
            entry = self._entries.get(self._active_path)
            if entry is not None and entry.raw_preview is not None:
                from luminos.core.inversion import invert_c41
                mask = tuple(float(v) for v in profile["orange_mask"])
                new_inv = invert_c41(entry.raw_preview, mask=mask)
                entry.inverted_preview = new_inv
                self._inverted_preview = new_inv
                self._invalidate_before_pixmap()

        # ── White balance ─────────────────────────────────────────────────
        if "white_balance" in profile:
            r, g, b = profile["white_balance"]
            temp, tint = rgb_multipliers_to_temp_tint(float(r), float(g), float(b))
            temp = max(2000, min(12000, temp))
            tint = max(-100, min(100, tint))
            self._temp_slider.blockSignals(True)
            self._tint_slider.blockSignals(True)
            self._temp_slider.setValue(temp)
            self._tint_slider.setValue(tint)
            self._temp_slider.blockSignals(False)
            self._tint_slider.blockSignals(False)
            self._temp_label.setText(f"{temp} K")
            self._tint_label.setText(f"{tint:+d}" if tint != 0 else "0")

        # ── Master tone curve ─────────────────────────────────────────────
        if "tone_curve" in profile:
            pts = [(float(p[0]), float(p[1])) for p in profile["tone_curve"]]
            self._curve_widget._points["master"] = pts
            self._curve_widget._recompute("master")
            self._curve_widget.update()

        self._update_preview()
        self._status.showMessage(f"Profil angewendet: {name}")

    # ── Settings capture / restore ────────────────────────────────────────────

    def _read_settings(self) -> _EditSettings:
        """Capture all current slider values and curve control points."""
        return _EditSettings(
            **{defn.attr: defn.slider.value() for defn in self._slider_defs},
            curve_points=copy.deepcopy(self._curve_widget._points),
        )

    def _apply_settings(self, s: _EditSettings) -> None:
        """Apply all slider values + curve points from *s*, then refresh the preview."""
        for defn in self._slider_defs:
            defn.slider.blockSignals(True)
            defn.slider.setValue(getattr(s, defn.attr))
            defn.slider.blockSignals(False)
        for defn in self._slider_defs:
            defn.label.setText(defn.fmt(getattr(s, defn.attr)))

        # Restore curve control points and recompute LUTs
        self._curve_widget._points = copy.deepcopy(s.curve_points)
        for ch in ("master", "r", "g", "b"):
            self._curve_widget._recompute(ch)
        self._curve_widget.update()

        self._update_preview()

    def _copy_settings_to_selected(self) -> None:
        """Copy current slider/curve state to all selected filmstrip images (except active)."""
        if self._active_path is None:
            return
        settings = self._read_settings()
        selected = self._filmstrip.selectedItems()
        count = 0
        for item in selected:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path == self._active_path:
                continue
            if path in self._entries:
                self._entries[path].settings = copy.deepcopy(settings)
                count += 1
        if count:
            self._status.showMessage(
                f"Einstellungen auf {count} Bild(er) kopiert."
            )
        else:
            self._status.showMessage("Keine anderen Bilder ausgewählt.")

    def _toggle_select_all(self) -> None:
        """Select all filmstrip images, or clear selection if all are already selected."""
        count = self._filmstrip.count()
        if count == 0:
            return
        if len(self._filmstrip.selectedItems()) == count:
            self._filmstrip.clearSelection()
            self._select_all_btn.setText("Alle wählen")
        else:
            self._filmstrip.selectAll()
            self._select_all_btn.setText("Auswahl löschen")

    def _reset_settings_for_selected(self) -> None:
        """Reset editing settings of all selected images to factory defaults."""
        selected = self._filmstrip.selectedItems()
        if not selected:
            self._status.showMessage("Keine Bilder ausgewählt.")
            return
        count = 0
        for item in selected:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path == self._active_path:
                # Active image: use the normal reset path so undo is recorded.
                self._reset_all_sliders()
                count += 1
            elif path in self._entries:
                self._entries[path].settings = _EditSettings()
                count += 1
        if count:
            self._status.showMessage(
                f"Einstellungen für {count} Bild(er) zurückgesetzt."
            )

    def _remove_selected(self) -> None:
        """Remove selected images from the filmstrip and release their memory."""
        selected = self._filmstrip.selectedItems()
        if not selected:
            self._status.showMessage("Keine Bilder ausgewählt.")
            return

        paths_to_remove = {item.data(Qt.ItemDataRole.UserRole) for item in selected}
        active_removed = self._active_path in paths_to_remove

        # Remember active row so we can land on the closest remaining image.
        active_row = 0
        if active_removed:
            for i in range(self._filmstrip.count()):
                if self._filmstrip.item(i).data(Qt.ItemDataRole.UserRole) == self._active_path:
                    active_row = i
                    break

        # takeItem() keeps row() accurate for remaining items as the list shrinks.
        for item in selected:
            self._filmstrip.takeItem(self._filmstrip.row(item))

        for path in paths_to_remove:
            self._entries.pop(path, None)

        remaining = self._filmstrip.count()
        count = len(paths_to_remove)

        if active_removed:
            # Stop any in-flight full-image loader for the removed image.
            if self._full_loader is not None:
                try:
                    self._full_loader.loaded.disconnect(self._on_full_image_loaded)
                    self._full_loader.error.disconnect(self._on_load_error)
                except RuntimeError:
                    pass
                old = self._full_loader
                self._old_loaders.append(old)
                old.finished.connect(
                    lambda o=old: self._old_loaders.remove(o)
                    if o in self._old_loaders else None
                )
                self._full_loader = None

            self._active_path = None  # clear so _activate_path won't short-circuit

            if remaining > 0:
                new_row = min(active_row, remaining - 1)
                new_item = self._filmstrip.item(new_row)
                self._filmstrip.setCurrentItem(new_item)
                self._activate_path(new_item.data(Qt.ItemDataRole.UserRole))
            else:
                # Filmstrip is empty — reset viewer and disable per-image controls.
                self._raw_image = None
                self._inverted_preview = None
                self._processed_pixmap = None
                self._before_pixmap = None
                self._split_view = False
                self._split_btn.setChecked(False)
                self._image_label.setPixmap(QPixmap())
                self._image_label.setText("No image loaded")
                self._navigator.clear()
                self._set_export_enabled(False)
                self._set_batch_enabled(False)
                self._set_before_enabled(False)
                for btn in (
                    self._copy_settings_btn, self._remove_btn, self._select_all_btn,
                    self._reset_selected_btn, self._split_btn, self._pipette_btn,
                    self._auto_wb_btn, self._auto_levels_btn, self._crop_btn,
                    self._rot_ccw_btn, self._rot_cw_btn,
                ):
                    btn.setEnabled(False)
                self._select_all_btn.setText("Alle wählen")

        self._status.showMessage(f"{count} Bild(er) entfernt.")

    # ── Before / After comparison ─────────────────────────────────────────────

    def _toggle_before_after(self) -> None:
        """Toggle before/after mode (keyboard shortcut \\)."""
        if self._before_btn.isEnabled():
            self._before_btn.setChecked(not self._before_btn.isChecked())

    def _set_before_after(self, active: bool) -> None:
        """Show the unedited inverted image (before) or the processed result (after)."""
        self._before_after = active
        if active and self._inverted_preview is not None:
            # Build the "before" pixmap from the raw inverted preview on demand.
            uint8 = (np.clip(self._inverted_preview, 0, 1) * 255).astype(np.uint8)
            h, w = uint8.shape[:2]
            qimg = QImage(uint8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            self._before_pixmap = QPixmap.fromImage(qimg)
        if active and self._split_view:
            # Mutual exclusion: before-only and split cannot both be on.
            self._split_btn.blockSignals(True)
            self._split_btn.setChecked(False)
            self._split_btn.blockSignals(False)
            self._split_view = False
        self._before_btn.setText("Nachher" if active else "Vorher")
        self._act_before_after.setChecked(active)
        self._fit_pixmap_to_label()

    def _set_split_view(self, active: bool) -> None:
        """Activate/deactivate the side-by-side split-view comparison."""
        self._split_view = active
        if active:
            # Always rebuild _before_pixmap so it matches the current _inverted_preview
            # (it may have changed due to crop/rotation/film-type since last activation).
            if self._inverted_preview is not None:
                uint8 = (np.clip(self._inverted_preview, 0, 1) * 255).astype(np.uint8)
                h, w = uint8.shape[:2]
                qimg = QImage(uint8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
                self._before_pixmap = QPixmap.fromImage(qimg)
            # Mutual exclusion: split and before-only cannot both be on.
            if self._before_after:
                self._before_btn.blockSignals(True)
                self._before_btn.setChecked(False)
                self._before_btn.setText("Vorher")
                self._before_btn.blockSignals(False)
                self._before_after = False
        self._fit_pixmap_to_label()

    def _make_split_composite(self, after_scaled: QPixmap) -> QPixmap:
        """
        Build a split-screen composite from *after_scaled* (already scaled to
        the desired display size) and ``_before_pixmap``.

        Left half: 'before' image; right half: 'after' image.
        A thin white divider line and text labels are drawn at the centre.
        """
        before_pix = self._before_pixmap
        if before_pix is None:
            return after_scaled

        w, h = after_scaled.width(), after_scaled.height()
        # Scale before to fit the same canvas.  KeepAspectRatio is safe here:
        # both images share the same source dimensions, so they produce identical
        # scaled sizes in the normal case.  When fine-angle rotation has expanded
        # _processed_pixmap slightly, KeepAspectRatio avoids stretching the before.
        before_scaled = before_pix.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Centre before vertically if it's shorter (rotation-expanded after).
        before_y = (h - before_scaled.height()) // 2

        composite = QPixmap(w, h)
        composite.fill(QColor(0, 0, 0))
        p = QPainter(composite)

        # Draw before on the left half.
        p.setClipRect(0, 0, w // 2, h)
        p.drawPixmap(0, before_y, before_scaled)

        # Draw after on the right half.
        p.setClipRect(w // 2, 0, w - w // 2, h)
        p.drawPixmap(0, 0, after_scaled)

        p.setClipping(False)

        # Centre divider line.
        mid = w // 2
        p.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.SolidLine))
        p.drawLine(mid, 0, mid, h)

        # Small labels.
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        margin = 6
        label_h = 18
        for text, align_right in (("VORHER", False), ("NACHHER", True)):
            fm = p.fontMetrics()
            lw = fm.horizontalAdvance(text)
            if align_right:
                lx = mid + margin
            else:
                lx = mid - margin - lw
            ly = margin
            # Shadow
            p.setPen(QColor(0, 0, 0, 160))
            p.drawText(lx + 1, ly + 1, lw, label_h,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       text)
            # Text
            p.setPen(QColor(255, 255, 255, 220))
            p.drawText(lx, ly, lw, label_h,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       text)

        p.end()
        return composite

    # ── Before-pixmap lifecycle ───────────────────────────────────────────────

    def _invalidate_before_pixmap(self) -> None:
        """
        Called whenever ``_inverted_preview`` is replaced in-place (film-type
        toggle, profile re-inversion, crop, rotation, undo).

        Clears the cached "before" pixmap and, if a comparison mode is currently
        active, immediately rebuilds it from the new ``_inverted_preview`` so the
        display stays consistent.
        """
        self._before_pixmap = None
        if (self._before_after or self._split_view) and self._inverted_preview is not None:
            uint8 = (np.clip(self._inverted_preview, 0, 1) * 255).astype(np.uint8)
            h, w = uint8.shape[:2]
            qimg = QImage(uint8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            self._before_pixmap = QPixmap.fromImage(qimg)

    # ── Navigator ─────────────────────────────────────────────────────────────

    def _update_navigator(self) -> None:
        """Refresh the navigator thumbnail and viewport rectangle."""
        if self._processed_pixmap is None:
            self._navigator.update_pixmap(None)
            self._navigator.update_rect(None)
            return

        self._navigator.update_pixmap(self._processed_pixmap)

        if self._zoom_fit_mode:
            # Entire image is visible — no rectangle needed.
            self._navigator.update_rect(None)
            return

        pix = self._processed_pixmap
        z = self._zoom_pct / 100.0
        img_w = max(1, round(pix.width() * z))
        img_h = max(1, round(pix.height() * z))

        vp = self._scroll.viewport().size()
        hval = self._scroll.horizontalScrollBar().value()
        vval = self._scroll.verticalScrollBar().value()

        vis_w = min(vp.width(), img_w)
        vis_h = min(vp.height(), img_h)

        nx = hval / img_w
        ny = vval / img_h
        nw = vis_w / img_w
        nh = vis_h / img_h

        self._navigator.update_rect((nx, ny, nw, nh))

    def _on_navigator_pan(self, frac_x: float, frac_y: float) -> None:
        """Pan the main view so that *frac_x/frac_y* (image-space) is centred."""
        if self._processed_pixmap is None or self._zoom_fit_mode:
            return
        pix = self._processed_pixmap
        z = self._zoom_pct / 100.0
        img_w = max(1, round(pix.width() * z))
        img_h = max(1, round(pix.height() * z))
        vp = self._scroll.viewport().size()
        hval = max(0, round(frac_x * img_w - vp.width() / 2))
        vval = max(0, round(frac_y * img_h - vp.height() / 2))
        self._scroll.horizontalScrollBar().setValue(hval)
        self._scroll.verticalScrollBar().setValue(vval)

    # ── 90° Rotation ──────────────────────────────────────────────────────────

    def _rotate_ccw(self) -> None:
        """Rotate the current image 90° counter-clockwise."""
        self._apply_rotation_90(steps=1)

    def _rotate_cw(self) -> None:
        """Rotate the current image 90° clockwise (= 3× CCW)."""
        self._apply_rotation_90(steps=3)

    def _apply_rotation_90(self, steps: int) -> None:
        """Apply *steps* × 90° CCW rotation (steps ∈ {1, 3}), with undo support."""
        if self._inverted_preview is None:
            return
        entry = self._entries.get(self._active_path)
        if entry is None:
            return

        # Push to shared undo stack
        entry.crop_history.append((
            entry.inverted_preview.copy() if entry.inverted_preview is not None else None,
            entry.raw_preview.copy() if entry.raw_preview is not None else None,
            entry.crop_region_norm,
            entry.rotation_steps,
        ))

        # Rotate preview arrays
        entry.inverted_preview = np.rot90(entry.inverted_preview, k=steps)
        if entry.raw_preview is not None:
            entry.raw_preview = np.rot90(entry.raw_preview, k=steps)

        # Track cumulative rotation for full-res export
        entry.rotation_steps = (entry.rotation_steps + steps) % 4

        self._inverted_preview = entry.inverted_preview
        self._invalidate_before_pixmap()
        self._crop_sel_norm = None
        self._crop_apply_btn.setEnabled(False)
        self._crop_undo_btn.setEnabled(True)
        self._rot_undo_btn.setEnabled(True)
        self._update_crop_overlay()
        self._update_preview()
        self._sync_undo_redo_btns()
        self._status.showMessage(
            f"Gedreht {'↺' if steps == 1 else '↻'} 90°"
        )

    def _keyboard_zoom(self, step: int) -> None:
        """Zoom one step in/out, anchored to the centre of the viewport."""
        vp = self._scroll.viewport()
        center = QPoint(vp.width() // 2, vp.height() // 2)
        self._on_wheel_zoom(step, center)

    # ── WB Pipette ────────────────────────────────────────────────────────────

    def _toggle_pipette(self) -> None:
        """Toggle pipette mode (keyboard shortcut P)."""
        self._pipette_btn.setChecked(not self._pipette_btn.isChecked())

    def _set_pipette_mode(self, active: bool) -> None:
        """Activate or deactivate pipette mode."""
        self._scroll.set_pipette_mode(active)
        if active:
            self._status.showMessage("Pipette: click a neutral (white/grey) area")
        else:
            self._status.showMessage("Ready")

    def _map_viewport_to_preview(self, pos: QPoint) -> tuple[int, int] | None:
        """
        Map a viewport-coordinate point to (px, py) in _inverted_preview space.

        Returns None if no image is loaded or the click is outside the image.
        """
        if self._inverted_preview is None or self._processed_pixmap is None:
            return None

        pix = self._processed_pixmap

        if self._zoom_fit_mode:
            vp = self._scroll.viewport().size()
            fitted = pix.scaled(
                vp,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            fit_w, fit_h = fitted.width(), fitted.height()
            if fit_w <= 0 or fit_h <= 0:
                return None
            off_x = (vp.width()  - fit_w) / 2.0
            off_y = (vp.height() - fit_h) / 2.0
            rel_x = (pos.x() - off_x) / fit_w
            rel_y = (pos.y() - off_y) / fit_h
        else:
            hval    = self._scroll.horizontalScrollBar().value()
            vval    = self._scroll.verticalScrollBar().value()
            label_w = self._image_label.width()
            label_h = self._image_label.height()
            if label_w <= 0 or label_h <= 0:
                return None
            rel_x   = (pos.x() + hval) / label_w
            rel_y   = (pos.y() + vval) / label_h

        if not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
            return None  # clicked outside the image area

        prev_h, prev_w = self._inverted_preview.shape[:2]
        px = int(np.clip(rel_x * prev_w, 0, prev_w - 1))
        py = int(np.clip(rel_y * prev_h, 0, prev_h - 1))
        return px, py

    def _on_pipette_click(self, pos: QPoint) -> None:
        """Sample the clicked region and apply its colour as the WB neutral point."""
        coords = self._map_viewport_to_preview(pos)
        if coords is None:
            self._set_pipette_mode(False)
            self._pipette_btn.setChecked(False)
            return

        px, py = coords
        prev = self._inverted_preview

        # Sample a 5×5 neighbourhood and average — more robust than a single pixel.
        h, w = prev.shape[:2]
        r0, r1 = max(0, py - 2), min(h, py + 3)
        c0, c1 = max(0, px - 2), min(w, px + 3)
        patch = prev[r0:r1, c0:c1]           # float32 (H', W', 3) in [0, 1]
        r, g, b = float(patch[:, :, 0].mean()), float(patch[:, :, 1].mean()), float(patch[:, :, 2].mean())

        if min(r, g, b) < 0.01:
            # Sampled area is too dark — can't reliably compute WB.
            self._status.showMessage("Pipette: sampled area is too dark, try another spot")
            self._set_pipette_mode(False)
            self._pipette_btn.setChecked(False)
            return

        # Compute multipliers so the sampled patch becomes neutral grey:
        #   r * wb_r = g * wb_g = b * wb_b = target
        target = (r + g + b) / 3.0
        wb_r, wb_g, wb_b = target / r, target / g, target / b

        temp, tint = self._set_wb_from_multipliers(wb_r, wb_g, wb_b)

        # Deactivate pipette after a successful pick.
        self._set_pipette_mode(False)
        self._pipette_btn.setChecked(False)
        self._status.showMessage(
            f"WB set from pixel ({px}, {py}) — {temp} K  Tönung {tint:+d}"
        )

    def _on_auto_wb(self) -> None:
        """Gray-World auto white balance: scale channels so their means are equal."""
        if self._inverted_preview is None:
            return

        prev = self._inverted_preview          # float32 (H, W, 3), pre-WB
        r = float(prev[:, :, 0].mean())
        g = float(prev[:, :, 1].mean())
        b = float(prev[:, :, 2].mean())

        if min(r, g, b) < 0.01:
            self._status.showMessage("Auto-WB: Bild zu dunkel.")
            return

        target = (r + g + b) / 3.0
        wb_r, wb_g, wb_b = target / r, target / g, target / b

        temp, tint = self._set_wb_from_multipliers(wb_r, wb_g, wb_b)
        self._status.showMessage(f"Auto-WB: {temp} K  Tönung {tint:+d}")

    def _set_wb_from_multipliers(
        self, wb_r: float, wb_g: float, wb_b: float
    ) -> tuple[int, int]:
        """Convert RGB multipliers to temp/tint, push undo, and set sliders.

        Returns (temp_k, tint) for callers that want to display the values.
        """
        temp, tint = rgb_multipliers_to_temp_tint(wb_r, wb_g, wb_b)
        temp = max(2000, min(12000, temp))
        tint = max(-100, min(100, tint))
        self._push_undo_current()
        self._temp_slider.setValue(temp)
        self._tint_slider.setValue(tint)
        return temp, tint

    # ── Crop ──────────────────────────────────────────────────────────────────

    def _shortcut_crop(self) -> None:
        """Toggle crop mode via keyboard shortcut C."""
        if self._crop_btn.isEnabled():
            self._crop_btn.setChecked(not self._crop_btn.isChecked())

    def _shortcut_crop_apply(self) -> None:
        """Confirm crop selection via Enter/Return — only acts when a selection exists."""
        if self._crop_apply_btn.isEnabled():
            self._apply_crop()
            self._crop_btn.setChecked(False)

    def _shortcut_escape(self) -> None:
        """Escape: exit crop mode or pipette mode, whichever is active."""
        if self._crop_btn.isChecked():
            self._crop_btn.setChecked(False)
        elif self._pipette_btn.isChecked():
            self._pipette_btn.setChecked(False)

    def _toggle_crop_mode(self, active: bool) -> None:
        """Activate or deactivate crop-draw mode."""
        self._crop_mode = active
        self._scroll.set_crop_mode(active)
        # Deactivate pipette when entering crop mode
        if active and self._pipette_btn.isChecked():
            self._pipette_btn.setChecked(False)
        if not active:
            # Clear in-progress selection when leaving crop mode without applying
            self._crop_sel_norm = None
            self._crop_overlay.set_rect(None)
            self._crop_overlay.setVisible(False)
            self._crop_apply_btn.setEnabled(False)
            self._status.showMessage("Ready")
            # Restore zoom state that was saved on entry
            self._zoom_fit_mode = self._pre_crop_zoom_fit
            self._zoom_pct = self._pre_crop_zoom_pct
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(self._zoom_pct)
            self._zoom_slider.blockSignals(False)
            if self._zoom_fit_mode:
                self._zoom_label.setText("Fit")
            else:
                self._zoom_label.setText(f"{self._zoom_pct} %")
            self._fit_pixmap_to_label()
        else:
            # Save current zoom state and force fit-to-window so the full image
            # is always visible during crop selection (fixes coordinate clamping
            # at viewport edge in fixed-zoom mode).
            self._pre_crop_zoom_fit = self._zoom_fit_mode
            self._pre_crop_zoom_pct = self._zoom_pct
            if not self._zoom_fit_mode:
                self._zoom_fit()
            self._crop_overlay.setVisible(True)
            self._crop_overlay.raise_()
            self._status.showMessage(
                "Crop: Bereich mit der Maus zeichnen — Anwenden um zuzuschneiden"
            )

    def _viewport_to_preview_norm(self, pos: QPoint) -> tuple[float, float] | None:
        """Map a viewport-coordinate point to (x_norm, y_norm) ∈ [0,1] in preview space."""
        if self._processed_pixmap is None:
            return None
        pix = self._processed_pixmap
        if self._zoom_fit_mode:
            vp = self._scroll.viewport().size()
            fitted = pix.scaled(
                vp,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            fit_w, fit_h = fitted.width(), fitted.height()
            off_x = (vp.width() - fit_w) / 2.0
            off_y = (vp.height() - fit_h) / 2.0
            rel_x = (pos.x() - off_x) / fit_w if fit_w > 0 else 0.5
            rel_y = (pos.y() - off_y) / fit_h if fit_h > 0 else 0.5
        else:
            hval = self._scroll.horizontalScrollBar().value()
            vval = self._scroll.verticalScrollBar().value()
            label_w = self._image_label.width()
            label_h = self._image_label.height()
            rel_x = (pos.x() + hval) / label_w if label_w > 0 else 0.5
            rel_y = (pos.y() + vval) / label_h if label_h > 0 else 0.5
        return (float(np.clip(rel_x, 0.0, 1.0)), float(np.clip(rel_y, 0.0, 1.0)))

    def _preview_norm_to_viewport(self, xn: float, yn: float) -> QPoint | None:
        """Convert normalised preview coords [0,1] to a QPoint in viewport space."""
        if self._processed_pixmap is None:
            return None
        pix = self._processed_pixmap
        if self._zoom_fit_mode:
            vp = self._scroll.viewport().size()
            fitted = pix.scaled(
                vp,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            fit_w, fit_h = fitted.width(), fitted.height()
            off_x = (vp.width() - fit_w) / 2.0
            off_y = (vp.height() - fit_h) / 2.0
            return QPoint(int(off_x + xn * fit_w), int(off_y + yn * fit_h))
        else:
            hval = self._scroll.horizontalScrollBar().value()
            vval = self._scroll.verticalScrollBar().value()
            label_w = self._image_label.width()
            label_h = self._image_label.height()
            return QPoint(int(xn * label_w - hval), int(yn * label_h - vval))

    def _constrain_aspect_norm(
        self,
        x0: float, y0: float, x1: float, y1: float,
        aspect: float,
    ) -> tuple[float, float, float, float]:
        """Constrain selection so width/height = aspect (in image-pixel space)."""
        if self._inverted_preview is None:
            return x0, y0, x1, y1
        prev_h, prev_w = self._inverted_preview.shape[:2]
        dx_pix = abs(x1 - x0) * prev_w
        dy_pix = abs(y1 - y0) * prev_h
        if dy_pix < 1:
            dy_pix = 1.0
        if dx_pix / dy_pix > aspect:
            # Constrain by height → adjust width
            new_dx = (dy_pix * aspect) / prev_w
            x1 = x0 + new_dx if x1 >= x0 else x0 - new_dx
        else:
            # Constrain by width → adjust height
            new_dy = (dx_pix / aspect) / prev_h
            y1 = y0 + new_dy if y1 >= y0 else y0 - new_dy
        return x0, y0, float(np.clip(x1, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))

    def _update_crop_overlay(self) -> None:
        """Resize the overlay to the current viewport and repaint the selection."""
        if self._crop_overlay is None:
            return
        vp_size = self._scroll.viewport().size()
        self._crop_overlay.resize(vp_size)

        if self._crop_sel_norm is None or not self._crop_mode:
            self._crop_overlay.set_rect(None)
            return

        x0n, y0n, x1n, y1n = self._crop_sel_norm
        p0 = self._preview_norm_to_viewport(x0n, y0n)
        p1 = self._preview_norm_to_viewport(x1n, y1n)
        if p0 is not None and p1 is not None:
            self._crop_overlay.set_rect((p0.x(), p0.y(), p1.x(), p1.y()))
        else:
            self._crop_overlay.set_rect(None)

    def _on_aspect_changed(self, index: int) -> None:
        """Aspect ratio combo changed — update current selection if any."""
        self._crop_ratio = self._aspect_ratios[index][1]
        # Re-constrain the existing selection
        if self._crop_sel_norm is not None and self._crop_ratio is not None:
            x0, y0, x1, y1 = self._crop_sel_norm
            x0, y0, x1, y1 = self._constrain_aspect_norm(x0, y0, x1, y1, self._crop_ratio)
            self._crop_sel_norm = (x0, y0, x1, y1)
            self._update_crop_overlay()

    def _on_crop_drag(self, start: QPoint, current: QPoint) -> None:
        """Update the crop selection rectangle while the user drags."""
        n_start = self._viewport_to_preview_norm(start)
        n_curr = self._viewport_to_preview_norm(current)
        if n_start is None or n_curr is None:
            return
        x0, y0 = n_start
        x1, y1 = n_curr
        if self._crop_ratio is not None:
            x0, y0, x1, y1 = self._constrain_aspect_norm(x0, y0, x1, y1, self._crop_ratio)
        self._crop_sel_norm = (x0, y0, x1, y1)
        self._crop_apply_btn.setEnabled(True)
        self._update_crop_overlay()

    def _on_crop_end(self, start: QPoint, end: QPoint) -> None:
        """Finalise the crop selection on mouse release."""
        self._on_crop_drag(start, end)

    def _apply_crop(self) -> None:
        """Crop the image to the current selection and push undo entry."""
        if self._crop_sel_norm is None or self._inverted_preview is None:
            return

        x0n, y0n, x1n, y1n = self._crop_sel_norm
        cx0, cy0 = min(x0n, x1n), min(y0n, y1n)
        cx1, cy1 = max(x0n, x1n), max(y0n, y1n)

        if (cx1 - cx0) < 0.01 or (cy1 - cy0) < 0.01:
            self._status.showMessage("Crop: Auswahl zu klein")
            return

        entry = self._entries.get(self._active_path)
        if entry is None:
            return

        # Push current state to undo stack
        entry.crop_history.append((
            entry.inverted_preview.copy() if entry.inverted_preview is not None else None,
            entry.raw_preview.copy() if entry.raw_preview is not None else None,
            entry.crop_region_norm,
        ))

        # Compose new crop_region_norm in original-image space
        old = entry.crop_region_norm
        if old is None:
            entry.crop_region_norm = (cx0, cy0, cx1, cy1)
        else:
            ox0, oy0, ox1, oy1 = old
            entry.crop_region_norm = (
                ox0 + cx0 * (ox1 - ox0),
                oy0 + cy0 * (oy1 - oy0),
                ox0 + cx1 * (ox1 - ox0),
                oy0 + cy1 * (oy1 - oy0),
            )

        # Crop the preview arrays in-place
        prev_h, prev_w = entry.inverted_preview.shape[:2]
        r0, r1 = int(cy0 * prev_h), int(cy1 * prev_h)
        c0, c1 = int(cx0 * prev_w), int(cx1 * prev_w)
        entry.inverted_preview = entry.inverted_preview[r0:r1, c0:c1].copy()

        if entry.raw_preview is not None:
            rp_h, rp_w = entry.raw_preview.shape[:2]
            entry.raw_preview = entry.raw_preview[
                int(cy0 * rp_h):int(cy1 * rp_h),
                int(cx0 * rp_w):int(cx1 * rp_w),
            ].copy()

        # Update active shortcuts
        self._inverted_preview = entry.inverted_preview
        self._invalidate_before_pixmap()

        # Reset crop selection
        self._crop_sel_norm = None
        self._crop_apply_btn.setEnabled(False)
        self._crop_undo_btn.setEnabled(True)
        self._update_crop_overlay()
        self._update_preview()
        self._sync_undo_redo_btns()
        self._status.showMessage("Zuschnitt angewendet")

    def _reset_crop_to_initial(self) -> None:
        """Reset crop and rotation to the state the image was in when first loaded."""
        entry = self._entries.get(self._active_path)
        if entry is None:
            return

        has_history = bool(entry.crop_history)
        has_angle = self._angle_slider.value() != 0
        if not has_history and not has_angle:
            return

        if has_history:
            # The first entry is the snapshot taken before any crop or rotation.
            snap = entry.crop_history[0]
            if len(snap) == 4:
                inv_snap, raw_snap, crop_snap, rot_snap = snap
                entry.rotation_steps = rot_snap
            else:
                inv_snap, raw_snap, crop_snap = snap
            entry.inverted_preview = inv_snap
            entry.raw_preview = raw_snap
            entry.crop_region_norm = crop_snap
            entry.crop_history.clear()
            self._inverted_preview = entry.inverted_preview
            self._invalidate_before_pixmap()

        # Reset fine-angle slider without triggering a second undo entry.
        if has_angle:
            self._angle_slider.blockSignals(True)
            self._angle_slider.setValue(0)
            self._angle_slider.blockSignals(False)
            self._angle_label.setText("0.0°")

        self._crop_sel_norm = None
        self._crop_apply_btn.setEnabled(False)
        self._crop_undo_btn.setEnabled(False)
        self._rot_undo_btn.setEnabled(False)
        self._update_crop_overlay()
        self._update_preview()
        self._sync_undo_redo_btns()
        self._status.showMessage("Zuschnitt auf Original zurückgesetzt")

    def _undo_crop(self) -> None:
        """Restore the previous crop / rotation state.

        Fine angle (Begradigung) is treated as the outermost undo step: if the
        angle slider is non-zero, the first call resets it to 0.  Subsequent
        calls pop structural history entries (90° rotations, crops).
        """
        entry = self._entries.get(self._active_path)
        if entry is None:
            return

        # Undo fine angle first — it is non-destructive so no history entry is
        # needed; just reset the slider.
        if self._angle_slider.value() != 0:
            self._angle_slider.setValue(0)
            # _on_angle_changed fires and updates the undo button state; done.
            return

        if not entry.crop_history:
            return

        snap = entry.crop_history.pop()
        # Support both old 3-tuple (crop only) and new 4-tuple (crop+rotation)
        if len(snap) == 4:
            inv_snap, raw_snap, crop_snap, rot_snap = snap
            entry.rotation_steps = rot_snap
        else:
            inv_snap, raw_snap, crop_snap = snap

        entry.inverted_preview = inv_snap
        entry.raw_preview = raw_snap
        entry.crop_region_norm = crop_snap

        self._inverted_preview = entry.inverted_preview
        self._invalidate_before_pixmap()
        self._crop_sel_norm = None
        has_history = bool(entry.crop_history)
        self._crop_apply_btn.setEnabled(False)
        self._crop_undo_btn.setEnabled(has_history)
        self._rot_undo_btn.setEnabled(has_history)
        self._update_crop_overlay()
        self._update_preview()
        self._sync_undo_redo_btns()
        self._status.showMessage("Rückgängig gemacht")

    # ── Single-image export ───────────────────────────────────────────────────

    def _export_image(self) -> None:
        if self._raw_image is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Image", "", _EXPORT_FILTERS
        )
        if not path:
            return

        if "." not in Path(path).name:
            if "TIFF" in selected_filter:
                path += ".tif"
            elif "PNG" in selected_filter:
                path += ".png"
            else:
                path += ".jpg"

        self._status.showMessage(f"Exporting {Path(path).name}…")
        self._set_export_enabled(False)
        self._set_batch_enabled(False)

        entry = self._entries.get(self._active_path)
        crop = entry.crop_region_norm if entry is not None else None
        rot  = entry.rotation_steps   if entry is not None else 0
        film_type = entry.film_type   if entry is not None else "c41"
        params = ProcessingParams(
            exposure_stops=self._exposure(),
            white_balance=self._white_balance(),
            mask=self._orange_mask,
            black_point=self._black_point(),
            white_point=self._white_point(),
            saturation=self._saturation(),
            curve_luts=self._curve_widget.luts,
            sharpening=self._sharpening(),
            angle=self._angle(),
            contrast=self._contrast(),
            highlights=self._highlights(),
            shadows=self._shadows(),
            vibrance=self._vibrance(),
            noise_reduction=self._noise_reduction(),
            vignette=self._vignette(),
            grain=self._grain(),
            film_type=film_type,
            split_toning=self._split_toning_params(),
            exif_bytes=entry.exif_bytes if entry is not None else None,
        )
        worker = _ExportWorker(
            raw=self._raw_image,
            path=path,
            params=params,
            crop_region_norm=crop,
            rotation_steps=rot,
        )
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_export_error)
        self._export_worker = worker
        worker.start()

    def _on_export_finished(self, path: str) -> None:
        self._set_export_enabled(True)
        self._set_batch_enabled(self._filmstrip.count() > 0)
        self._status.showMessage(f"Saved: {path}")

    def _on_export_error(self, msg: str) -> None:
        self._set_export_enabled(True)
        self._set_batch_enabled(self._filmstrip.count() > 0)
        self._status.showMessage(f"Export error: {msg}")

    # ── Batch export ──────────────────────────────────────────────────────────

    def _batch_export(self) -> None:
        selected = self._filmstrip.selectedItems()
        if not selected:
            self._status.showMessage("Keine Bilder im Filmstreifen ausgewählt.")
            return
        if self._active_path in self._entries:
            self._entries[self._active_path].settings = self._read_settings()

        s = self._app_settings
        dlg = _BatchExportDialog(
            self,
            default_fmt=s.default_export_format,
            default_quality=s.default_jpeg_quality,
            default_output_dir=s.default_output_dir,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        masks = {
            p: self._entries[p].orange_mask
            for p in paths
            if p in self._entries and self._entries[p].orange_mask is not None
        }
        crops = {
            p: self._entries[p].crop_region_norm
            for p in paths
            if p in self._entries and self._entries[p].crop_region_norm is not None
        }

        self._set_batch_enabled(False)
        self._set_export_enabled(False)
        self._status.showMessage(
            f"Batch export: 0 / {len(paths)} — {Path(dlg.output_dir).name}/"
        )

        rotations = {
            p: self._entries[p].rotation_steps
            for p in paths
            if p in self._entries
        }
        film_types = {
            p: self._entries[p].film_type
            for p in paths
            if p in self._entries
        }
        exif_data = {
            p: self._entries[p].exif_bytes
            for p in paths
            if p in self._entries
        }
        params_by_path = {
            p: processing_params_from_settings(self._entries[p].settings, self._entries[p])
            for p in paths
            if p in self._entries
        }
        params = ProcessingParams(
            exposure_stops=self._exposure(),
            white_balance=self._white_balance(),
            mask=None,
            black_point=self._black_point(),
            white_point=self._white_point(),
            saturation=self._saturation(),
            curve_luts=self._curve_widget.luts,
            sharpening=self._sharpening(),
            angle=self._angle(),
            contrast=self._contrast(),
            highlights=self._highlights(),
            shadows=self._shadows(),
            vibrance=self._vibrance(),
            noise_reduction=self._noise_reduction(),
            vignette=self._vignette(),
            grain=self._grain(),
            film_type="c41",
            split_toning=self._split_toning_params(),
        )
        worker = _BatchExportWorker(
            paths=paths,
            output_dir=dlg.output_dir,
            fmt=dlg.fmt,
            quality=dlg.quality,
            params=params,
            params_by_path=params_by_path,
            masks=masks,
            crops=crops,
            rotations=rotations,
            film_types=film_types,
            exif_data=exif_data,
            suffix=self._app_settings.export_suffix,
        )
        worker.progress.connect(self._on_batch_progress)
        worker.finished.connect(self._on_batch_finished)
        worker.error.connect(self._on_batch_error)
        self._batch_worker = worker
        worker.start()

    def _on_batch_progress(self, current: int, total: int, path: str) -> None:
        self._status.showMessage(
            f"Batch-Export: {current + 1} / {total} — {Path(path).name}"
        )

    def _on_batch_finished(self, count: int) -> None:
        self._set_batch_enabled(True)
        self._set_export_enabled(self._raw_image is not None)
        self._status.showMessage(f"Batch export complete: {count} file(s) saved.")

    def _on_batch_error(self, path: str, msg: str) -> None:
        # Non-fatal: log to status and continue — worker proceeds with next image
        self._status.showMessage(f"Batch error ({Path(path).name}): {msg}")

    # ── Hilfe ─────────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Über Luminos",
            "<b>Luminos</b><br>"
            "Film Negative Converter<br><br>"
            "Konvertiert analoge Filmnegative (RAW/TIFF)<br>"
            "in hochwertige Positive.",
        )

    # ── Sitzung speichern / öffnen ────────────────────────────────────────────

    def _save_session(self) -> None:
        if not self._entries:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sitzung speichern", "", _SESSION_FILTER
        )
        if not path:
            return
        if not path.endswith(".luminos"):
            path += ".luminos"

        if self._active_path and self._active_path in self._entries:
            self._entries[self._active_path].settings = self._read_settings()

        err = self._session_manager.save_file(path, self._active_path, self._entries)
        if err:
            QMessageBox.critical(self, "Fehler", f"Sitzung konnte nicht gespeichert werden:\n{err}")
            return

        self._status.showMessage(f"Sitzung gespeichert: {Path(path).name}")
        self._rebuild_recent_menu()

    def _open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Sitzung öffnen", "", _SESSION_FILTER
        )
        if not path:
            return
        self._load_session_file(path)

    def _load_session_file(self, path: str) -> None:
        data, err = self._session_manager.load_file(path)
        if err:
            QMessageBox.critical(self, "Fehler", f"Sitzung konnte nicht geladen werden:\n{err}")
            return

        images = data.get("images", [])
        if not images:
            return

        existing_paths = [img["path"] for img in images if img["path"] in self._entries]
        new_entries = [img for img in images if img["path"] not in self._entries]

        for img_path in existing_paths:
            saved = next(i for i in images if i["path"] == img_path)
            self._session_manager.apply_entry_data(self._entries[img_path], saved)

        active_path = data.get("active_path")
        if new_entries:
            self._pending_session_settings = {img["path"]: img for img in new_entries}
            self._pending_session_active = active_path

            new_paths = [img["path"] for img in new_entries]
            missing = [p for p in new_paths if not os.path.exists(p)]
            if missing:
                names = "\n".join(Path(p).name for p in missing[:5])
                suffix = f"\n… und {len(missing) - 5} weitere" if len(missing) > 5 else ""
                QMessageBox.warning(
                    self, "Dateien nicht gefunden",
                    f"Folgende Dateien wurden nicht gefunden und werden übersprungen:\n{names}{suffix}"
                )
                for p in missing:
                    self._pending_session_settings.pop(p, None)
                new_paths = [p for p in new_paths if p not in missing]

            if new_paths:
                for p in new_paths:
                    self._entries[p] = _ImageEntry(path=p)
                worker = _ImportWorker(new_paths, self._app_settings.preview_long_edge)
                worker.image_ready.connect(self._on_image_ready)
                worker.error.connect(self._on_import_error)
                worker.all_done.connect(self._on_import_done)
                self._import_worker = worker
                worker.start()
                self._status.showMessage(f"Sitzung wird geladen: {len(new_paths)} Bild(er)…")
        else:
            if active_path and active_path in self._entries:
                for i in range(self._filmstrip.count()):
                    item = self._filmstrip.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == active_path:
                        self._filmstrip.setCurrentItem(item)
                        self._activate_path(active_path)
                        break
            self._status.showMessage("Sitzung geladen.")

        self._rebuild_recent_menu()

    # ── Zuletzt geöffnet ──────────────────────────────────────────────────────

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        recents = self._session_manager.load_recent()
        if not recents:
            no_act = self._recent_menu.addAction("(Keine kürzlichen Sitzungen)")
            no_act.setEnabled(False)
            return
        for p in recents:
            label = f"{Path(p).name}  —  {Path(p).parent}"
            act = self._recent_menu.addAction(label)
            act.setData(p)
            act.triggered.connect(lambda checked=False, fp=p: self._load_session_file(fp))
        self._recent_menu.addSeparator()
        self._recent_menu.addAction("Liste leeren", self._clear_recent_sessions)

    def _clear_recent_sessions(self) -> None:
        self._session_manager.clear_recent()
        self._rebuild_recent_menu()
