"""
RangeSlider — a horizontal dual-handle slider widget.

Emits range_changed(lo: int, hi: int) when either handle moves.
Both handles are draggable; the selected range is highlighted between them.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QRect


class RangeSlider(QWidget):
    """Horizontal slider with two draggable handles for selecting a numeric range."""

    range_changed = pyqtSignal(int, int)  # (lo, hi)

    _HANDLE_W = 10
    _HANDLE_H = 14
    _TRACK_H  = 4

    def __init__(self, minimum: int = 0, maximum: int = 100,
                 lo: int = 0, hi: int = 100, parent=None) -> None:
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._lo  = max(minimum, min(lo, maximum))
        self._hi  = max(minimum, min(hi, maximum))
        self._drag: str | None = None  # 'lo' or 'hi'
        self.setMinimumHeight(22)
        self.setMinimumWidth(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lo(self) -> int:
        return self._lo

    def hi(self) -> int:
        return self._hi

    def set_lo(self, value: int) -> None:
        self._lo = max(self._min, min(value, self._hi))
        self.update()

    def set_hi(self, value: int) -> None:
        self._hi = max(self._lo, min(value, self._max))
        self.update()

    def set_range(self, lo: int, hi: int) -> None:
        self._lo = max(self._min, min(lo, self._max))
        self._hi = max(self._min, min(hi, self._max))
        self.update()

    def set_minimum(self, v: int) -> None:
        self._min = v
        self.update()

    def set_maximum(self, v: int) -> None:
        self._max = v
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w  = self.width()
        h  = self.height()
        cx = w // 2     # centre‑x  (unused but kept for clarity)
        cy = h // 2

        hw  = self._HANDLE_W
        hh  = self._HANDLE_H
        th  = self._TRACK_H
        pad = hw // 2   # left/right padding so handles don't go off-edge

        # Background track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(190, 190, 190))
        p.drawRect(QRect(pad, cy - th // 2, w - hw, th))

        # Active range highlight
        lx = self._val_to_x(self._lo)
        hx = self._val_to_x(self._hi)
        p.setBrush(QColor(80, 140, 240))
        p.drawRect(QRect(lx, cy - th // 2, hx - lx, th))

        # Handles
        p.setBrush(QColor(50, 110, 210))
        p.setPen(QPen(QColor(30, 80, 180), 1))
        p.drawRoundedRect(QRect(lx - hw // 2, cy - hh // 2, hw, hh), 2, 2)
        p.drawRoundedRect(QRect(hx - hw // 2, cy - hh // 2, hw, hh), 2, 2)

        p.end()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        x  = event.position().x()
        lx = self._val_to_x(self._lo)
        hx = self._val_to_x(self._hi)
        # Grab whichever handle is closer
        self._drag = 'lo' if abs(x - lx) <= abs(x - hx) else 'hi'

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            return
        val = self._x_to_val(event.position().x())
        if self._drag == 'lo':
            new_lo = max(self._min, min(val, self._hi - 1))
            if new_lo != self._lo:
                self._lo = new_lo
                self.update()
                self.range_changed.emit(self._lo, self._hi)
        else:
            new_hi = max(self._lo + 1, min(val, self._max))
            if new_hi != self._hi:
                self._hi = new_hi
                self.update()
                self.range_changed.emit(self._lo, self._hi)

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _val_to_x(self, val: int) -> int:
        span = self._max - self._min
        if span == 0:
            return self._HANDLE_W // 2
        frac   = (val - self._min) / span
        usable = self.width() - self._HANDLE_W
        return int(self._HANDLE_W // 2 + frac * usable)

    def _x_to_val(self, x: float) -> int:
        usable = self.width() - self._HANDLE_W
        if usable <= 0:
            return self._min
        frac = (x - self._HANDLE_W // 2) / usable
        frac = max(0.0, min(1.0, frac))
        return int(round(self._min + frac * (self._max - self._min)))
