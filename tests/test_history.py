from luminos.ui.history import _EditHistory
from luminos.ui.session import _EditSettings


def test_edit_history_push_undo_redo_cycle():
    history = _EditHistory()
    initial = _EditSettings(exposure=0)
    changed = _EditSettings(exposure=10)

    history.push(initial)

    assert history.has_undo()
    assert not history.has_redo()
    assert history.undo_step(changed) == initial
    assert not history.has_undo()
    assert history.has_redo()
    assert history.redo_step(initial) == changed


def test_edit_history_push_clears_redo_stack():
    history = _EditHistory()
    first = _EditSettings(exposure=1)
    second = _EditSettings(exposure=2)
    current = _EditSettings(exposure=3)

    history.push(first)
    assert history.undo_step(current) == first
    assert history.has_redo()

    history.push(second)

    assert history.has_undo()
    assert not history.has_redo()


def test_edit_history_maxlen_discards_oldest_entries():
    history = _EditHistory(maxlen=2)
    history.push(_EditSettings(exposure=1))
    history.push(_EditSettings(exposure=2))
    history.push(_EditSettings(exposure=3))

    assert [s.exposure for s in history.undo] == [2, 3]


def test_edit_history_clear_all_removes_both_stacks():
    history = _EditHistory()
    current = _EditSettings(exposure=2)

    history.push(_EditSettings(exposure=1))
    history.undo_step(current)
    history.clear_all()

    assert not history.has_undo()
    assert not history.has_redo()
