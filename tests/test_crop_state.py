import numpy as np
import pytest

from luminos.ui.crop_state import (
    apply_crop_selection,
    apply_rotation_90,
    reset_crop_to_initial,
    undo_crop_or_rotation,
)
from luminos.ui.session import _ImageEntry


def _entry() -> _ImageEntry:
    return _ImageEntry(
        path="/scan/a.tif",
        inverted_preview=np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3),
        raw_preview=np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3) + 1000,
    )


def test_apply_rotation_90_rotates_previews_and_pushes_history():
    entry = _entry()
    original = entry.inverted_preview.copy()

    apply_rotation_90(entry, 1)

    assert entry.rotation_steps == 1
    assert entry.inverted_preview.shape == (6, 4, 3)
    np.testing.assert_array_equal(entry.inverted_preview, np.rot90(original, k=1))
    assert len(entry.crop_history) == 1


def test_apply_crop_selection_crops_previews_and_sets_region():
    entry = _entry()

    apply_crop_selection(entry, (0.0, 0.0, 0.5, 0.5))

    assert entry.inverted_preview.shape == (2, 3, 3)
    assert entry.raw_preview.shape == (2, 3, 3)
    assert entry.crop_region_norm == (0.0, 0.0, 0.5, 0.5)
    assert len(entry.crop_history) == 1


def test_apply_crop_selection_composes_existing_region():
    entry = _entry()
    entry.crop_region_norm = (0.2, 0.2, 0.8, 0.8)

    apply_crop_selection(entry, (0.25, 0.25, 0.75, 0.75))

    assert entry.crop_region_norm == pytest.approx((0.35, 0.35, 0.65, 0.65))


def test_apply_crop_selection_rejects_tiny_selection():
    entry = _entry()

    with pytest.raises(ValueError):
        apply_crop_selection(entry, (0.1, 0.1, 0.105, 0.2))


def test_undo_crop_or_rotation_restores_previous_snapshot():
    entry = _entry()
    original = entry.inverted_preview.copy()
    apply_crop_selection(entry, (0.0, 0.0, 0.5, 0.5))

    assert undo_crop_or_rotation(entry)

    np.testing.assert_array_equal(entry.inverted_preview, original)
    assert entry.crop_region_norm is None
    assert not entry.crop_history


def test_reset_crop_to_initial_uses_first_snapshot_and_clears_history():
    entry = _entry()
    original = entry.inverted_preview.copy()
    apply_rotation_90(entry, 1)
    apply_crop_selection(entry, (0.0, 0.0, 0.5, 0.5))

    assert reset_crop_to_initial(entry)

    np.testing.assert_array_equal(entry.inverted_preview, original)
    assert entry.rotation_steps == 0
    assert entry.crop_region_norm is None
    assert not entry.crop_history


def test_reset_crop_to_initial_returns_false_without_history():
    assert not reset_crop_to_initial(_entry())
