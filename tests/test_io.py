import numpy as np
from PIL import Image

from luminos.io.export import save_jpeg, save_png, save_tiff
from luminos.io import tiff_loader
from luminos.io.tiff_loader import load_tiff


def test_load_tiff_preserves_16_bit_rgb_range(tmp_path):
    import tifffile

    path = tmp_path / "input.tif"
    source = np.array(
        [
            [[0, 32768, 65535], [1000, 2000, 3000]],
            [[4000, 5000, 6000], [65000, 64000, 63000]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(path, source, photometric="rgb")

    loaded = load_tiff(str(path))

    assert loaded.shape == source.shape
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded[0, 0], [0.0, 32768 / 65535, 1.0], rtol=0, atol=1e-6)


def test_load_tiff_expands_grayscale_to_rgb(tmp_path):
    import tifffile

    path = tmp_path / "gray.tif"
    source = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    tifffile.imwrite(path, source)

    loaded = load_tiff(str(path))

    assert loaded.shape == (2, 2, 3)
    np.testing.assert_allclose(loaded[:, :, 0], loaded[:, :, 1])
    np.testing.assert_allclose(loaded[:, :, 1], loaded[:, :, 2])


def test_load_tiff_falls_back_to_tifffile_after_pyvips_error(monkeypatch):
    expected = np.ones((1, 2, 3), dtype=np.float32)

    def fail_pyvips(path):
        raise RuntimeError("pyvips unavailable")

    def load_with_tifffile(path):
        return expected

    def fail_pillow(path):
        raise AssertionError("Pillow fallback should not be used")

    monkeypatch.setattr(tiff_loader, "_load_pyvips", fail_pyvips)
    monkeypatch.setattr(tiff_loader, "_load_tifffile", load_with_tifffile)
    monkeypatch.setattr(tiff_loader, "_load_pillow", fail_pillow)

    assert load_tiff("image.tif") is expected


def test_load_tiff_falls_back_to_pillow_after_pyvips_and_tifffile_errors(monkeypatch):
    expected = np.zeros((1, 2, 3), dtype=np.float32)

    def fail(path):
        raise RuntimeError("backend unavailable")

    def load_with_pillow(path):
        return expected

    monkeypatch.setattr(tiff_loader, "_load_pyvips", fail)
    monkeypatch.setattr(tiff_loader, "_load_tifffile", fail)
    monkeypatch.setattr(tiff_loader, "_load_pillow", load_with_pillow)

    assert load_tiff("image.tif") is expected


def test_save_tiff_writes_16_bit_rgb(tmp_path):
    import tifffile

    path = tmp_path / "out.tif"
    image = np.zeros((3, 4, 3), dtype=np.float32)
    image[:, :, 0] = 1.0
    image[:, :, 1] = 0.5

    save_tiff(image, path, metadata={"film_type": "test", "description": "pytest"})
    written = tifffile.imread(path)

    assert written.shape == image.shape
    assert written.dtype == np.uint16
    assert int(written[0, 0, 0]) == 65535
    assert 32767 <= int(written[0, 0, 1]) <= 32768
    assert int(written[0, 0, 2]) == 0


def test_save_png_writes_rgb_and_metadata(tmp_path):
    path = tmp_path / "out.png"
    image = np.ones((2, 3, 3), dtype=np.float32) * 0.25

    save_png(image, path, metadata={"film_type": "test", "description": "pytest"})

    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert img.size == (3, 2)
        assert img.info["Software"] == "Luminos"
        assert img.info["Film"] == "test"
        assert img.info["Description"] == "pytest"


def test_save_jpeg_writes_rgb_file(tmp_path):
    path = tmp_path / "out.jpg"
    image = np.ones((2, 3, 3), dtype=np.float32) * 0.75

    save_jpeg(image, path, quality=90, metadata={"description": "pytest"})

    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert img.size == (3, 2)
