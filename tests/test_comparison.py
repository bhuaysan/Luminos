import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton

from luminos.ui.comparison import _ComparisonController, make_split_composite
from luminos.ui.image_utils import array_to_pixmap


def _app():
    return QApplication.instance() or QApplication([])


def test_comparison_controller_before_after_and_split_are_mutually_exclusive():
    _app()
    refresh_count = 0

    def refresh():
        nonlocal refresh_count
        refresh_count += 1

    before_btn = QPushButton()
    before_btn.setCheckable(True)
    split_btn = QPushButton()
    split_btn.setCheckable(True)
    action = QAction()
    action.setCheckable(True)
    controller = _ComparisonController(before_btn, split_btn, action, refresh)
    preview = np.ones((2, 3, 3), dtype=np.float32)

    controller.set_before_after(True, preview)
    assert controller.before_after
    assert controller.before_pixmap is not None
    assert before_btn.text() == "Nachher"
    assert action.isChecked()

    controller.set_split_view(True, preview)
    assert controller.split_view
    assert not controller.before_after
    assert not before_btn.isChecked()
    assert before_btn.text() == "Vorher"
    assert not action.isChecked()
    assert refresh_count == 2


def test_comparison_controller_reset_clears_modes_without_refresh():
    _app()
    before_btn = QPushButton()
    before_btn.setCheckable(True)
    split_btn = QPushButton()
    split_btn.setCheckable(True)
    action = QAction()
    action.setCheckable(True)
    controller = _ComparisonController(before_btn, split_btn, action, lambda: None)
    preview = np.ones((2, 2, 3), dtype=np.float32)

    controller.set_split_view(True, preview)
    controller.reset()

    assert not controller.before_after
    assert not controller.split_view
    assert controller.before_pixmap is None
    assert not before_btn.isChecked()
    assert not split_btn.isChecked()
    assert before_btn.text() == "Vorher"


def test_make_split_composite_preserves_after_size():
    _app()
    before = array_to_pixmap(np.zeros((3, 4, 3), dtype=np.uint8))
    after = array_to_pixmap(np.full((3, 4, 3), 255, dtype=np.uint8))

    composite = make_split_composite(after, before)

    assert isinstance(composite, QPixmap)
    assert composite.size() == after.size()
