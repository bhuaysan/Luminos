from luminos.ui.activation_controller import new_import_paths
from luminos.ui.session import _ImageEntry


def test_new_import_paths_filters_existing_entries():
    entries = {
        "/scan/a.tif": _ImageEntry(path="/scan/a.tif"),
        "/scan/b.tif": _ImageEntry(path="/scan/b.tif"),
    }

    assert new_import_paths(
        ["/scan/a.tif", "/scan/c.tif", "/scan/b.tif", "/scan/d.tif"],
        entries,
    ) == ["/scan/c.tif", "/scan/d.tif"]


def test_new_import_paths_preserves_duplicates_not_already_imported():
    assert new_import_paths(["/scan/a.tif", "/scan/a.tif"], {}) == [
        "/scan/a.tif",
        "/scan/a.tif",
    ]
