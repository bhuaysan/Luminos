"""Controller for the viewer navigator thumbnail and pan rectangle."""

from __future__ import annotations


class _NavigatorController:
    """Synchronise navigator thumbnail/viewport state with the main scroll area."""

    def __init__(self, navigator, scroll_area) -> None:
        self._navigator = navigator
        self._scroll = scroll_area

    def update(self, pixmap, *, zoom_fit_mode: bool, zoom_pct: int) -> None:
        """Refresh the navigator thumbnail and viewport rectangle."""
        if pixmap is None:
            self._navigator.update_pixmap(None)
            self._navigator.update_rect(None)
            return

        self._navigator.update_pixmap(pixmap)

        if zoom_fit_mode:
            self._navigator.update_rect(None)
            return

        img_w, img_h = self._scaled_image_size(pixmap, zoom_pct)
        viewport = self._scroll.viewport().size()
        hval = self._scroll.horizontalScrollBar().value()
        vval = self._scroll.verticalScrollBar().value()

        vis_w = min(viewport.width(), img_w)
        vis_h = min(viewport.height(), img_h)

        self._navigator.update_rect((
            hval / img_w,
            vval / img_h,
            vis_w / img_w,
            vis_h / img_h,
        ))

    def pan_to(self, frac_x: float, frac_y: float, pixmap, *, zoom_fit_mode: bool, zoom_pct: int) -> None:
        """Pan the main view so that *frac_x/frac_y* in image-space is centred."""
        if pixmap is None or zoom_fit_mode:
            return

        img_w, img_h = self._scaled_image_size(pixmap, zoom_pct)
        viewport = self._scroll.viewport().size()
        hval = max(0, round(frac_x * img_w - viewport.width() / 2))
        vval = max(0, round(frac_y * img_h - viewport.height() / 2))
        self._scroll.horizontalScrollBar().setValue(hval)
        self._scroll.verticalScrollBar().setValue(vval)

    @staticmethod
    def _scaled_image_size(pixmap, zoom_pct: int) -> tuple[int, int]:
        zoom = zoom_pct / 100.0
        return (
            max(1, round(pixmap.width() * zoom)),
            max(1, round(pixmap.height() * zoom)),
        )
