"""Load camera RAW files into linear float32 NumPy arrays."""

import numpy as np
import rawpy


def load_raw(path: str) -> np.ndarray:
    """
    Load a RAW file and return a linear float32 array (H, W, 3), range 0–1.

    rawpy is told to:
    - disable auto-bright / auto-WB so we get true linear data
    - no_auto_scale is intentionally omitted so rawpy auto-scales the
      sensor output to the full 16-bit range, making the /65535 division correct
    - output as 16-bit, which we then normalize ourselves
    """
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            output_color=rawpy.ColorSpace.raw,   # stay in camera colour space
            output_bps=16,
            no_auto_bright=True,
            use_auto_wb=False,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],       # flat multipliers
            gamma=(1, 1),                         # linear (no gamma)
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    return rgb.astype(np.float32) / 65535.0
