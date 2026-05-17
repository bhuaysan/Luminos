import numpy as np

from luminos.ui.image_utils import rotate_uint8


def test_rotate_uint8_returns_expanded_uint8_rgb_image():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[1, 2] = [255, 128, 64]

    rotated = rotate_uint8(image, 45)

    assert rotated.dtype == np.uint8
    assert rotated.ndim == 3
    assert rotated.shape[2] == 3
    assert rotated.shape[0] > image.shape[0]
    assert rotated.shape[1] > image.shape[1]


def test_rotate_uint8_zero_angle_preserves_shape():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    rotated = rotate_uint8(image, 0)

    assert rotated.shape == image.shape
