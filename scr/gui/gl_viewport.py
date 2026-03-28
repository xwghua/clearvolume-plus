"""
GLViewport — QOpenGLWidget that owns the VolumeRenderer and MouseControl.

This is the central render window.  It:
  - Initialises OpenGL on first show
  - Runs the render loop via paintGL
  - Forwards mouse/keyboard events to MouseControl
  - Paints axis overlays via QPainter on top of the GL content
"""

from __future__ import annotations
import math
import numpy as np

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal

from OpenGL.GL import glFinish

from ..renderer.volume_renderer import VolumeRenderer
from ..renderer.render_algorithm import RenderAlgorithm
from ..controller.mouse_control import MouseControl
from ..volume.volume import Volume
from ..utils.math_utils import (
    quaternion_to_matrix4, perspective_matrix, look_at_matrix,
)


class GLViewport(QOpenGLWidget):
    """OpenGL render surface for volume visualisation."""

    # Emitted when the timepoint changes via keyboard ([ / ]) so the
    # ControlPanel time slider stays in sync.
    timepoint_changed = pyqtSignal(int)

    # Emitted whenever rotation, translation, or camera distance changes
    # (mouse drag / scroll / keyboard) so the AxisPanel transform spinboxes
    # can stay in sync.
    transform_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(512, 512)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._renderer = VolumeRenderer()
        self._mouse = MouseControl(self._renderer)

        # Axis overlay settings (modified by AxisPanel)
        self.axis_visible: dict[str, bool] = {"X": True, "Y": True, "Z": True}
        self.axis_labels: dict[str, str] = {"X": "X", "Y": "Y", "Z": "Z"}
        self.axis_colors: dict[str, QColor] = {
            "X": QColor(220, 50, 50),
            "Y": QColor(50, 200, 50),
            "Z": QColor(50, 100, 220),
        }
        self.axis_ticks: int = 5
        self.axis_tick_unit: float = 1.0   # physical unit per tick
        self.axis_tick_show_labels: bool = True  # show numeric labels on ticks

    # ------------------------------------------------------------------
    # Public API (delegated to renderer)
    # ------------------------------------------------------------------

    @property
    def renderer(self) -> VolumeRenderer:
        return self._renderer

    def set_volume(self, volume: Volume) -> None:
        self._renderer.set_volume(volume)
        self.update()

    # ------------------------------------------------------------------
    # QOpenGLWidget overrides
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        self._renderer.initGL()

    def resizeGL(self, w: int, h: int) -> None:
        pass  # viewport set each frame inside render()

    def paintGL(self) -> None:
        # Qt6 QOpenGLWidget creates a physical-pixel-sized FBO (width * dpr × height * dpr).
        # We must pass physical dimensions to glViewport so GL renders into the full FBO.
        # QPainter uses logical coordinates and auto-scales for DPR, so _paint_axes() is unchanged.
        dpr = self.devicePixelRatio()
        self._renderer.render(int(self.width() * dpr), int(self.height() * dpr))
        # Ensure GL commands finish before QPainter takes over
        glFinish()
        # Paint axis labels on top via QPainter
        self._paint_axes()

    def closeEvent(self, event) -> None:
        self._renderer.cleanup()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Mouse / keyboard events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._mouse.press(event, self.width(), self.height())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._mouse.release(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse.move(event, self.width(), self.height())
        self.update()
        self.transform_changed.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._mouse.wheel(event)
        self.update()
        self.transform_changed.emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        renderer = self._renderer
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        step = 0.1 if shift else 0.01

        if key == Qt.Key.Key_I:
            renderer.set_algorithm(renderer._algorithm.next())
        elif key == Qt.Key.Key_R:
            from ..utils.math_utils import quaternion_identity
            renderer.rotation = quaternion_identity()
            renderer.translation = np.zeros(2, dtype=np.float32)
            # Re-fit camera distance to the loaded volume (or default if none loaded).
            if renderer._volume is not None:
                asp = renderer._volume.aspect_ratio
                half = max(float(asp[0]), float(asp[1])) * 0.5
                import math as _math
                renderer._camera_distance = half / _math.tan(_math.radians(renderer._fov_deg * 0.5)) * 1.25
            else:
                renderer._camera_distance = 2.0
        elif key in (Qt.Key.Key_Left,):
            t = renderer.translation.copy(); t[0] -= step; renderer.translation = t
        elif key in (Qt.Key.Key_Right,):
            t = renderer.translation.copy(); t[0] += step; renderer.translation = t
        elif key in (Qt.Key.Key_Up,):
            t = renderer.translation.copy(); t[1] += step; renderer.translation = t
        elif key in (Qt.Key.Key_Down,):
            t = renderer.translation.copy(); t[1] -= step; renderer.translation = t
        elif key == Qt.Key.Key_BracketLeft:
            # Step time-lapse backward by 1 frame
            if renderer.n_timepoints > 1:
                new_t = max(0, renderer.current_timepoint - 1)
                if new_t != renderer.current_timepoint:
                    renderer.set_timepoint(new_t)
                    self.timepoint_changed.emit(new_t)
        elif key == Qt.Key.Key_BracketRight:
            # Step time-lapse forward by 1 frame
            if renderer.n_timepoints > 1:
                new_t = min(renderer.n_timepoints - 1, renderer.current_timepoint + 1)
                if new_t != renderer.current_timepoint:
                    renderer.set_timepoint(new_t)
                    self.timepoint_changed.emit(new_t)
        elif key == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()

        self.update()
        self.transform_changed.emit()

    # ------------------------------------------------------------------
    # Axis overlay (QPainter on top of GL)
    # ------------------------------------------------------------------

    def _paint_axes(self) -> None:
        """Draw X/Y/Z axis lines + tick marks + labels using QPainter."""
        if not any(self.axis_visible.values()):
            return

        w = self.width()
        h = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Project axis endpoints from 3D model space to 2D screen
        proj = perspective_matrix(self._renderer._fov_deg, w / max(h, 1), 0.01, 100.0)
        view = self._renderer._build_view_matrix()
        model = self._renderer._build_model_matrix()
        mvp = proj @ view @ model

        # Axis extents in model space (half the volume aspect)
        if self._renderer._volume is not None:
            asp = self._renderer._volume.aspect_ratio
        else:
            asp = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        half_x, half_y, half_z = asp * 0.5

        # Convert ROI texture-space bounds [0,1] to model space [-half, +half].
        # texCoord=0 → model=-half, texCoord=1 → model=+half.
        roi_min = self._renderer.roi_min  # shape (3,), values in [0,1]
        roi_max = self._renderer.roi_max
        roi_start = np.array([
            (2.0 * roi_min[0] - 1.0) * half_x,
            (2.0 * roi_min[1] - 1.0) * half_y,
            (2.0 * roi_min[2] - 1.0) * half_z,
        ], dtype=np.float32)
        roi_end = np.array([
            (2.0 * roi_max[0] - 1.0) * half_x,
            (2.0 * roi_max[1] - 1.0) * half_y,
            (2.0 * roi_max[2] - 1.0) * half_z,
        ], dtype=np.float32)

        # Origin: ROI min corner.  Three axis lines run along the box edges
        # that meet at this corner, spanning only the active ROI region.
        origin = roi_start

        axes = {
            "X": (origin, np.array([roi_end[0],  origin[1],   origin[2]  ])),
            "Y": (origin, np.array([origin[0],   roi_end[1],  origin[2]  ])),
            "Z": (origin, np.array([origin[0],   origin[1],   roi_end[2] ])),
        }

        font = QFont("Arial", 9, QFont.Weight.Bold)
        painter.setFont(font)

        for axis_name, (p0, p1) in axes.items():
            if not self.axis_visible.get(axis_name, True):
                continue

            color = self.axis_colors[axis_name]
            pen = QPen(color, 1)
            painter.setPen(pen)

            s0 = self._project(p0, mvp, w, h)
            s1 = self._project(p1, mvp, w, h)
            if s0 is None or s1 is None:
                continue

            painter.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))

            # --- Tick marks ---
            n_ticks = max(1, self.axis_ticks)
            for i in range(1, n_ticks + 1):
                t_frac = i / n_ticks
                pt = p0 + (p1 - p0) * t_frac
                sp = self._project(pt, mvp, w, h)
                if sp is None:
                    continue
                # Small perpendicular tick (5 px in screen space)
                ax = s1 - s0
                ax_len = np.linalg.norm(ax)
                if ax_len < 1e-4:
                    continue
                perp = np.array([-ax[1], ax[0]]) / ax_len * 5
                tx0 = sp + perp
                tx1 = sp - perp
                painter.drawLine(int(tx0[0]), int(tx0[1]),
                                 int(tx1[0]), int(tx1[1]))
                # Tick label
                if self.axis_tick_show_labels:
                    label_val = i * self.axis_tick_unit
                    tick_label = f"{label_val:.4g}"
                    painter.setPen(QPen(color.lighter(160), 1))
                    painter.drawText(QRectF(sp[0]+4, sp[1]-10, 50, 16), tick_label)
                    painter.setPen(pen)

            # --- Axis label at far end ---
            painter.setPen(QPen(color, 1))
            label = self.axis_labels.get(axis_name, axis_name)
            if s1 is not None:
                painter.drawText(QRectF(s1[0] + 6, s1[1] - 8, 60, 16), label)

        painter.end()

    def _project(self, p3: np.ndarray, mvp: np.ndarray,
                 w: int, h: int) -> np.ndarray | None:
        """Project a 3D model-space point to 2D screen pixels."""
        v = np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float32)
        clip = mvp @ v
        if abs(clip[3]) < 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if abs(ndc[0]) > 2.0 or abs(ndc[1]) > 2.0:
            return None  # behind camera or far off-screen
        sx = (ndc[0] + 1.0) * 0.5 * w
        sy = (1.0 - ndc[1]) * 0.5 * h  # flip Y
        return np.array([sx, sy], dtype=np.float32)
