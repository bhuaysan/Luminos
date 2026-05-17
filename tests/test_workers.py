from pathlib import Path

import numpy as np

from luminos.ui.workers import ProcessingParams, _BatchExportWorker


def test_batch_export_uses_per_image_processing_params(monkeypatch, tmp_path):
    raw = np.ones((2, 2, 3), dtype=np.float32) * 0.5
    saved: list[tuple[str, str, float, str, tuple[float, float, float] | None]] = []

    def fake_load_image(path):
        return raw.copy()

    def fake_run_process_and_save(image, output_path, params, *, fmt, quality=95):
        saved.append((Path(output_path).name, fmt, params.exposure_stops, params.film_type, params.mask))

    monkeypatch.setattr("luminos.ui.workers.load_image", fake_load_image)
    monkeypatch.setattr("luminos.ui.workers._run_process_and_save", fake_run_process_and_save)

    paths = ["/input/first.tif", "/input/second.tif"]
    fallback = ProcessingParams(
        exposure_stops=0.0,
        white_balance=(1.0, 1.0, 1.0),
        mask=None,
    )
    first_params = ProcessingParams(
        exposure_stops=1.0,
        white_balance=(1.0, 1.0, 1.0),
        mask=(0.1, 0.2, 0.3),
        film_type="c41",
    )
    second_params = ProcessingParams(
        exposure_stops=-0.5,
        white_balance=(1.0, 1.0, 1.0),
        mask=None,
        film_type="bw",
    )

    worker = _BatchExportWorker(
        paths=paths,
        output_dir=str(tmp_path),
        fmt="png",
        quality=90,
        params=fallback,
        masks={paths[0]: (0.1, 0.2, 0.3)},
        params_by_path={
            paths[0]: first_params,
            paths[1]: second_params,
        },
        film_types={
            paths[0]: "c41",
            paths[1]: "bw",
        },
        suffix="_done",
    )

    worker.run()

    assert saved == [
        ("first_done.png", "png", 1.0, "c41", (0.1, 0.2, 0.3)),
        ("second_done.png", "png", -0.5, "bw", None),
    ]
