"""Pure crop and 90-degree rotation state operations for image entries."""

from __future__ import annotations

import numpy as np

from luminos.ui.session import _ImageEntry


def apply_rotation_90(entry: _ImageEntry, steps: int) -> None:
    """Apply *steps* x 90-degree CCW rotation to preview arrays and push history."""
    entry.crop_history.append((
        entry.inverted_preview.copy() if entry.inverted_preview is not None else None,
        entry.raw_preview.copy() if entry.raw_preview is not None else None,
        entry.crop_region_norm,
        entry.rotation_steps,
    ))
    entry.inverted_preview = np.rot90(entry.inverted_preview, k=steps)
    if entry.raw_preview is not None:
        entry.raw_preview = np.rot90(entry.raw_preview, k=steps)
    entry.rotation_steps = (entry.rotation_steps + steps) % 4


def apply_crop_selection(
    entry: _ImageEntry,
    selection_norm: tuple[float, float, float, float],
) -> None:
    """Crop preview arrays to a normalized selection and push history."""
    x0n, y0n, x1n, y1n = selection_norm
    cx0, cy0 = min(x0n, x1n), min(y0n, y1n)
    cx1, cy1 = max(x0n, x1n), max(y0n, y1n)

    if (cx1 - cx0) < 0.01 or (cy1 - cy0) < 0.01:
        raise ValueError("selection too small")

    entry.crop_history.append((
        entry.inverted_preview.copy() if entry.inverted_preview is not None else None,
        entry.raw_preview.copy() if entry.raw_preview is not None else None,
        entry.crop_region_norm,
    ))

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

    prev_h, prev_w = entry.inverted_preview.shape[:2]
    r0, r1 = int(cy0 * prev_h), int(cy1 * prev_h)
    c0, c1 = int(cx0 * prev_w), int(cx1 * prev_w)
    entry.inverted_preview = entry.inverted_preview[r0:r1, c0:c1].copy()

    if entry.raw_preview is not None:
        raw_h, raw_w = entry.raw_preview.shape[:2]
        entry.raw_preview = entry.raw_preview[
            int(cy0 * raw_h):int(cy1 * raw_h),
            int(cx0 * raw_w):int(cx1 * raw_w),
        ].copy()


def reset_crop_to_initial(entry: _ImageEntry) -> bool:
    """Restore the first crop/rotation history snapshot."""
    if not entry.crop_history:
        return False

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
    return True


def undo_crop_or_rotation(entry: _ImageEntry) -> bool:
    """Pop one crop/rotation history entry."""
    if not entry.crop_history:
        return False

    snap = entry.crop_history.pop()
    if len(snap) == 4:
        inv_snap, raw_snap, crop_snap, rot_snap = snap
        entry.rotation_steps = rot_snap
    else:
        inv_snap, raw_snap, crop_snap = snap

    entry.inverted_preview = inv_snap
    entry.raw_preview = raw_snap
    entry.crop_region_norm = crop_snap
    return True
