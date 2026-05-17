import json

from luminos.ui import session
from luminos.ui.session import (
    SessionManager,
    _EditSettings,
    _ImageEntry,
    _settings_from_dict,
    _settings_to_dict,
)


def test_edit_settings_dict_roundtrip_preserves_values_and_curve_points():
    settings = _EditSettings(
        exposure=12,
        contrast=-8,
        highlights=4,
        shadows=-5,
        black_point=3,
        white_point=96,
        wb_temp=5200,
        wb_tint=-12,
        vibrance=20,
        saturation=85,
        sharpening=9,
        noise_reduction=7,
        vignette=-15,
        grain=11,
        angle=25,
        split_shadow_hue=45,
        split_shadow_sat=30,
        split_highlight_hue=210,
        split_highlight_sat=25,
        split_balance=-10,
        curve_points={
            "master": [(0.0, 0.0), (0.5, 0.45), (1.0, 1.0)],
            "r": [(0.0, 0.0), (1.0, 0.9)],
            "g": [(0.0, 0.1), (1.0, 1.0)],
            "b": [(0.0, 0.0), (1.0, 1.0)],
        },
    )

    restored = _settings_from_dict(_settings_to_dict(settings))

    assert restored == settings


def test_edit_settings_from_partial_dict_uses_defaults():
    restored = _settings_from_dict({"exposure": 5})

    assert restored.exposure == 5
    assert restored.white_point == 100
    assert restored.wb_temp == 6500
    assert restored.curve_points == _EditSettings().curve_points


def test_session_manager_serialize_includes_entry_state():
    manager = SessionManager()
    entry = _ImageEntry(
        path="/scan/a.tif",
        film_type="bw",
        crop_region_norm=(0.1, 0.2, 0.8, 0.9),
        rotation_steps=2,
        settings=_EditSettings(exposure=7),
    )

    data = manager.serialize("/scan/a.tif", {entry.path: entry})

    assert data["version"] == session._SESSION_VERSION
    assert data["active_path"] == "/scan/a.tif"
    assert data["images"] == [
        {
            "path": "/scan/a.tif",
            "film_type": "bw",
            "crop_region_norm": [0.1, 0.2, 0.8, 0.9],
            "rotation_steps": 2,
            "settings": _settings_to_dict(entry.settings),
        }
    ]


def test_session_manager_apply_entry_data_restores_saved_fields():
    manager = SessionManager()
    entry = _ImageEntry(path="/scan/a.tif")
    saved = {
        "film_type": "bw",
        "crop_region_norm": [0.2, 0.3, 0.7, 0.8],
        "rotation_steps": 3,
        "settings": {"exposure": -4, "saturation": 75},
    }

    manager.apply_entry_data(entry, saved)

    assert entry.film_type == "bw"
    assert entry.crop_region_norm == (0.2, 0.3, 0.7, 0.8)
    assert entry.rotation_steps == 3
    assert entry.settings.exposure == -4
    assert entry.settings.saturation == 75
    assert entry.settings.white_point == 100


def test_session_manager_save_and_load_file_roundtrip(tmp_path, monkeypatch):
    recent_paths: list[str] = []
    monkeypatch.setattr(session, "_add_recent_session", recent_paths.append)

    manager = SessionManager()
    path = tmp_path / "project.luminos"
    entry = _ImageEntry(path="/scan/a.tif", settings=_EditSettings(exposure=3))

    error = manager.save_file(str(path), entry.path, {entry.path: entry})
    loaded, load_error = manager.load_file(str(path))

    assert error is None
    assert load_error is None
    assert loaded == manager.serialize(entry.path, {entry.path: entry})
    assert recent_paths == [str(path), str(path)]


def test_session_manager_load_file_reports_json_errors(tmp_path):
    manager = SessionManager()
    path = tmp_path / "broken.luminos"
    path.write_text("{not json", encoding="utf-8")

    data, error = manager.load_file(str(path))

    assert data is None
    assert error


def test_session_manager_clear_recent_writes_empty_list(tmp_path, monkeypatch):
    recent_path = tmp_path / "recent_sessions.json"
    monkeypatch.setattr(session, "_RECENT_SESSIONS_PATH", recent_path)

    SessionManager().clear_recent()

    assert json.loads(recent_path.read_text(encoding="utf-8")) == []
