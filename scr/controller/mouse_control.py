"""
MouseControl — translates Qt mouse/wheel events into renderer state changes.

Mirrors the Java clearvolume/renderer/cleargl/MouseControl.java behaviour:

  Left-drag              → arcball rotation
  Right-drag             → XY pan
  Scroll                 → Z zoom
  Ctrl + left-drag       → transfer function range (min=X, max=Y)
  Shift + left-drag      → gamma adjustment
  Shift+Ctrl + left-drag → brightness adjustment
  Double-click           → request fullscreen toggle
"""

from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent

from ..utils.math_utils import (
    arcball_vector, arcball_rotation,
    quaternion_multiply, quaternion_normalise,
)
from ..renderer.volume_renderer import VolumeRenderer


class MouseControl:
    """
    Stateful mouse handler.  Attach to a QOpenGLWidget and forward events.

    Example::

        class GLViewport(QOpenGLWidget):
            def __init__(self):
                self.mouse = MouseControl(self.renderer)

            def mousePressEvent(self, event):
                self.mouse.press(event, self.width(), self.height())

            def mouseMoveEvent(self, event):
                self.mouse.move(event, self.width(), self.height())
                self.update()

            def mouseReleaseEvent(self, event):
                self.mouse.release(event)

            def wheelEvent(self, event):
                self.mouse.wheel(event)
                self.update()
    """

    ZOOM_SPEED: float = 0.005
    PAN_SPEED: float = 0.002

    def __init__(self, renderer: VolumeRenderer) -> None:
        self._renderer = renderer
        self._last_pos: QPointF | None = None
        self._button: Qt.MouseButton | None = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def press(self, event: QMouseEvent, w: int, h: int) -> None:
        self._last_pos = event.position()
        self._button = event.button()

    def release(self, event: QMouseEvent) -> None:  # noqa: ARG002
        self._last_pos = None
        self._button = None

    def move(self, event: QMouseEvent, w: int, h: int) -> None:
        if self._last_pos is None:
            return

        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos

        mods = event.modifiers()
        btn = self._button

        # --- Ctrl + left → transfer function range -----------------------
        if btn == Qt.MouseButton.LeftButton and (mods & Qt.KeyboardModifier.ControlModifier):
            x_ndc = pos.x() / max(w, 1)
            y_ndc = pos.y() / max(h, 1)
            rng_min = float(np.clip(x_ndc - 0.05, 0.0, 0.95))
            rng_max = float(np.clip(x_ndc + 0.05, 0.05, 1.0))
            self._renderer.range_min = rng_min
            self._renderer.range_max = max(rng_max, rng_min + 0.01)
            return

        # --- Shift+Ctrl + left → brightness ------------------------------
        if btn == Qt.MouseButton.LeftButton and (
            mods & Qt.KeyboardModifier.ShiftModifier
            and mods & Qt.KeyboardModifier.ControlModifier
        ):
            self._renderer.brightness = float(
                np.clip(self._renderer.brightness - dy * 0.02, 0.1, 10.0)
            )
            return

        # --- Shift + left → gamma ----------------------------------------
        if btn == Qt.MouseButton.LeftButton and (mods & Qt.KeyboardModifier.ShiftModifier):
            self._renderer.gamma = float(
                np.clip(self._renderer.gamma - dy * 0.02, 0.01, 5.0)
            )
            return

        # --- Left → arcball rotation -------------------------------------
        if btn == Qt.MouseButton.LeftButton:
            self._rotate(pos, dx, dy, w, h)
            return

        # --- Right or Middle → pan ----------------------------------------
        # Pan speed scales with camera distance so that one pixel of drag
        # moves the volume by one pixel-worth of world space at the current zoom.
        if btn in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            pan_scale = self._renderer._camera_distance * self.PAN_SPEED
            t = self._renderer.translation.copy()
            t[0] += dx * pan_scale
            t[1] -= dy * pan_scale  # Y is flipped between screen and world
            self._renderer.translation = t
            return

    def wheel(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        # Scroll up (positive delta) → zoom in (reduce camera distance)
        new_dist = self._renderer._camera_distance - delta * self.ZOOM_SPEED
        self._renderer._camera_distance = float(np.clip(new_dist, 0.3, 20.0))

    def double_click(self) -> None:
        """Signal to the parent widget to toggle full-screen."""
        # Handled by GLViewport.mouseDoubleClickEvent
        pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rotate(self, pos: QPointF, dx: float, dy: float,
                w: int, h: int) -> None:
        """Update renderer rotation via arcball."""
        # Map current and previous pixel to NDC [-1, 1]
        cx = 2.0 * pos.x() / max(w, 1) - 1.0
        cy = 1.0 - 2.0 * pos.y() / max(h, 1)
        px = cx - 2.0 * dx / max(w, 1)
        py = cy + 2.0 * dy / max(h, 1)

        v1 = arcball_vector(px, py)
        v2 = arcball_vector(cx, cy)

        delta_q = arcball_rotation(v1, v2)
        new_q = quaternion_multiply(delta_q, self._renderer.rotation)
        self._renderer.rotation = quaternion_normalise(new_q)
