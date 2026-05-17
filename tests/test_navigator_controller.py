from types import SimpleNamespace

from luminos.ui.navigator_controller import _NavigatorController


class FakePixmap:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class FakeBar:
    def __init__(self, value: int = 0) -> None:
        self._value = value

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:  # noqa: N802
        self._value = value


class FakeScroll:
    def __init__(self, width: int, height: int, hval: int = 0, vval: int = 0) -> None:
        self._size = SimpleNamespace(width=lambda: width, height=lambda: height)
        self._hbar = FakeBar(hval)
        self._vbar = FakeBar(vval)

    def viewport(self):
        return SimpleNamespace(size=lambda: self._size)

    def horizontalScrollBar(self):  # noqa: N802
        return self._hbar

    def verticalScrollBar(self):  # noqa: N802
        return self._vbar


class FakeNavigator:
    def __init__(self) -> None:
        self.pixmap = "unset"
        self.rect = "unset"

    def update_pixmap(self, pixmap) -> None:
        self.pixmap = pixmap

    def update_rect(self, rect) -> None:
        self.rect = rect


def test_navigator_update_clears_when_pixmap_missing():
    navigator = FakeNavigator()
    controller = _NavigatorController(navigator, FakeScroll(100, 50))

    controller.update(None, zoom_fit_mode=False, zoom_pct=100)

    assert navigator.pixmap is None
    assert navigator.rect is None


def test_navigator_update_hides_rect_in_fit_mode():
    navigator = FakeNavigator()
    pixmap = FakePixmap(200, 100)
    controller = _NavigatorController(navigator, FakeScroll(100, 50))

    controller.update(pixmap, zoom_fit_mode=True, zoom_pct=100)

    assert navigator.pixmap is pixmap
    assert navigator.rect is None


def test_navigator_update_computes_visible_rect_for_fixed_zoom():
    navigator = FakeNavigator()
    pixmap = FakePixmap(200, 100)
    controller = _NavigatorController(navigator, FakeScroll(100, 50, hval=50, vval=25))

    controller.update(pixmap, zoom_fit_mode=False, zoom_pct=200)

    assert navigator.rect == (0.125, 0.125, 0.25, 0.25)


def test_navigator_pan_centres_requested_fraction():
    scroll = FakeScroll(100, 50)
    controller = _NavigatorController(FakeNavigator(), scroll)

    controller.pan_to(0.5, 0.5, FakePixmap(200, 100), zoom_fit_mode=False, zoom_pct=200)

    assert scroll.horizontalScrollBar().value() == 150
    assert scroll.verticalScrollBar().value() == 75


def test_navigator_pan_ignores_fit_mode():
    scroll = FakeScroll(100, 50)
    controller = _NavigatorController(FakeNavigator(), scroll)

    controller.pan_to(0.5, 0.5, FakePixmap(200, 100), zoom_fit_mode=True, zoom_pct=200)

    assert scroll.horizontalScrollBar().value() == 0
    assert scroll.verticalScrollBar().value() == 0
