"""Application dialogs: batch export and preferences."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from luminos.ui.session import _AppSettings


# ── Batch Export Dialog ───────────────────────────────────────────────────────


class _BatchExportDialog(QDialog):
    """
    Format + quality + output directory selection for batch export.

    Read ``.fmt``, ``.quality``, and ``.output_dir`` after ``exec()`` returns Accepted.
    """

    _FORMATS = [
        ("TIFF 16-bit (lossless)", "tiff", ".tif"),
        ("PNG 8-bit  (lossless)", "png",  ".png"),
        ("JPEG 8-bit (lossy)",    "jpeg", ".jpg"),
    ]

    def __init__(
        self,
        parent=None,
        default_fmt: str = "tiff",
        default_quality: int = 95,
        default_output_dir: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Export Options")
        self.setMinimumWidth(420)

        self.fmt: str = default_fmt
        self.quality: int = default_quality
        self.output_dir: str = default_output_dir

        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setVerticalSpacing(10)

        self._fmt_combo = QComboBox()
        for label, _, _ in self._FORMATS:
            self._fmt_combo.addItem(label)
        fmt_idx = next((i for i, (_, k, _) in enumerate(self._FORMATS) if k == default_fmt), 0)
        self._fmt_combo.setCurrentIndex(fmt_idx)
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._fmt_combo)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 95)
        self._quality_spin.setValue(default_quality)
        self._quality_spin.setSuffix("  (1 = smallest, 95 = best)")
        self._quality_spin.setEnabled(default_fmt == "jpeg")
        form.addRow("JPEG quality:", self._quality_spin)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_label_text = default_output_dir if default_output_dir else "(none selected)"
        self._dir_label = QLabel(dir_label_text)
        self._dir_label.setWordWrap(True)
        dir_btn = QPushButton("Choose…")
        dir_btn.setFixedWidth(80)
        dir_btn.clicked.connect(self._pick_directory)
        dir_layout.addWidget(self._dir_label, stretch=1)
        dir_layout.addWidget(dir_btn)
        form.addRow("Output folder:", dir_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(bool(default_output_dir))
        form.addRow(buttons)

    def _on_format_changed(self, index: int) -> None:
        self._quality_spin.setEnabled(self._FORMATS[index][1] == "jpeg")

    def _pick_directory(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if d:
            self.output_dir = d
            parts = Path(d).parts
            short = str(Path(*parts[-2:])) if len(parts) >= 2 else d
            self._dir_label.setText(short)
            self._ok_btn.setEnabled(True)

    def _on_accept(self) -> None:
        idx = self._fmt_combo.currentIndex()
        self.fmt = self._FORMATS[idx][1]
        self.quality = self._quality_spin.value()
        self.accept()


# ── Preferences Dialog ────────────────────────────────────────────────────────


class _PreferencesDialog(QDialog):
    """Application preferences dialog with Import / Export / Oberfläche tabs."""

    def __init__(self, settings: _AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.setMinimumWidth(460)
        self._s = dataclasses.replace(settings)

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._build_import_tab(), "Import")
        tabs.addTab(self._build_export_tab(), "Export")
        tabs.addTab(self._build_ui_tab(), "Oberfläche")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_import_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(10)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self._film_combo = QComboBox()
        self._film_combo.addItem("C-41 Farbe", "c41")
        self._film_combo.addItem("Schwarzweiß", "bw")
        self._film_combo.setCurrentIndex(0 if self._s.default_film_type == "c41" else 1)
        form.addRow("Standard-Filmtyp:", self._film_combo)

        self._preview_combo = QComboBox()
        options = [("Niedrig  (800 px)", 800), ("Mittel  (1200 px)", 1200),
                   ("Hoch  (1500 px)", 1500), ("Sehr hoch  (2000 px)", 2000)]
        for label, val in options:
            self._preview_combo.addItem(label, val)
        idx = max(0, next(
            (i for i, (_, v) in enumerate(options) if v == self._s.preview_long_edge), 2
        ))
        self._preview_combo.setCurrentIndex(idx)
        note = QLabel("Gilt erst für neu importierte Bilder.")
        note.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow("Vorschau-Qualität:", self._preview_combo)
        form.addRow("", note)

        return w

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(10)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self._fmt_combo = QComboBox()
        fmt_options = [("TIFF 16-bit (lossless)", "tiff"),
                       ("PNG 8-bit  (lossless)", "png"),
                       ("JPEG 8-bit (lossy)",    "jpeg")]
        for label, val in fmt_options:
            self._fmt_combo.addItem(label, val)
        fmt_idx = next((i for i, (_, v) in enumerate(fmt_options)
                        if v == self._s.default_export_format), 0)
        self._fmt_combo.setCurrentIndex(fmt_idx)
        self._fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        form.addRow("Standard-Format:", self._fmt_combo)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 95)
        self._quality_spin.setValue(self._s.default_jpeg_quality)
        self._quality_spin.setSuffix("  (1 = kleinste, 95 = beste)")
        self._quality_spin.setEnabled(self._s.default_export_format == "jpeg")
        form.addRow("JPEG-Qualität:", self._quality_spin)

        self._suffix_edit = QLineEdit(self._s.export_suffix)
        self._suffix_edit.setPlaceholderText("_positive")
        form.addRow("Dateiname-Suffix:", self._suffix_edit)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self._dir_label = QLabel(self._s.default_output_dir or "(nicht gesetzt)")
        self._dir_label.setWordWrap(True)
        dir_btn = QPushButton("Wählen…")
        dir_btn.setFixedWidth(80)
        dir_btn.clicked.connect(self._pick_output_dir)
        clr_btn = QPushButton("✕")
        clr_btn.setFixedWidth(28)
        clr_btn.setToolTip("Zurücksetzen")
        clr_btn.clicked.connect(self._clear_output_dir)
        dir_layout.addWidget(self._dir_label, stretch=1)
        dir_layout.addWidget(dir_btn)
        dir_layout.addWidget(clr_btn)
        form.addRow("Standard-Ausgabeordner:", dir_row)

        return w

    def _build_ui_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(10)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self._debounce_combo = QComboBox()
        debounce_opts = [("Sehr schnell  (20 ms)", 20), ("Schnell  (50 ms)", 50),
                         ("Normal  (100 ms)", 100), ("Langsam  (200 ms)", 200)]
        for label, val in debounce_opts:
            self._debounce_combo.addItem(label, val)
        db_idx = max(0, next(
            (i for i, (_, v) in enumerate(debounce_opts) if v == self._s.preview_debounce_ms), 1
        ))
        self._debounce_combo.setCurrentIndex(db_idx)
        form.addRow("Vorschau-Reaktion:", self._debounce_combo)

        self._hist_log_cb = QCheckBox("Logarithmische Skala")
        self._hist_log_cb.setChecked(self._s.histogram_log_scale)
        form.addRow("Histogramm:", self._hist_log_cb)

        self._recent_spin = QSpinBox()
        self._recent_spin.setRange(1, 30)
        self._recent_spin.setValue(self._s.recent_sessions_max)
        form.addRow("Zuletzt geöffnet (max.):", self._recent_spin)

        return w

    def _on_fmt_changed(self, idx: int) -> None:
        self._quality_spin.setEnabled(self._fmt_combo.itemData(idx) == "jpeg")

    def _pick_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Ausgabeordner wählen")
        if d:
            self._s.default_output_dir = d
            self._dir_label.setText(d)

    def _clear_output_dir(self) -> None:
        self._s.default_output_dir = ""
        self._dir_label.setText("(nicht gesetzt)")

    def result_settings(self) -> _AppSettings:
        """Call after exec() == Accepted to get the edited settings."""
        self._s.default_film_type = self._film_combo.currentData()
        self._s.preview_long_edge = self._preview_combo.currentData()
        self._s.default_export_format = self._fmt_combo.currentData()
        self._s.default_jpeg_quality = self._quality_spin.value()
        self._s.export_suffix = self._suffix_edit.text().strip() or "_positive"
        self._s.preview_debounce_ms = self._debounce_combo.currentData()
        self._s.histogram_log_scale = self._hist_log_cb.isChecked()
        self._s.recent_sessions_max = self._recent_spin.value()
        return self._s
