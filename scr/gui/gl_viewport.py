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
from PyQt6.QtGui import (
    QMouseEvent, QWheelEvent, QKeyEvent,
    QPainter, QPen, QColor, QFont, QBrush, QPainterPath,
)
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
        self.axis_font_size: int = 9        # axis label / tick font size in pt

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
        """Draw X/Y/Z axis lines + tick marks + labels using QPainter.

        The axis origin is dynamically placed at the bounding-box corner that is
        front-bottom-left from the current camera angle, so X always extends
        rightward, Y upward, and Z deepens into the volume regardless of rotation.
        Tick marks are drawn even when the axis line itself is hidden.
        """
        any_visible = any(self.axis_visible.values())
        has_ticks   = self.axis_ticks > 0
        if not any_visible and not has_ticks:
            return

        w = self.width()
        h = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        proj  = perspective_matrix(self._renderer._fov_deg, w / max(h, 1), 0.01, 100.0)
        view  = self._renderer._build_view_matrix()
        model = self._renderer._build_model_matrix()
        vm    = view @ model          # world → camera space (no projection)
        mvp   = proj @ vm

        # Axis extents in model space
        if self._renderer._volume is not None:
            asp = self._renderer._volume.aspect_ratio
        else:
            asp = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        half_x, half_y, half_z = float(asp[0]) * 0.5, float(asp[1]) * 0.5, float(asp[2]) * 0.5

        # ROI bounds in model space
        roi_min = self._renderer.roi_min
        roi_max = self._renderer.roi_max
        x0 = float((2.0 * roi_min[0] - 1.0) * half_x)
        y0 = float((2.0 * roi_min[1] - 1.0) * half_y)
        z0 = float((2.0 * roi_min[2] - 1.0) * half_z)
        x1 = float((2.0 * roi_max[0] - 1.0) * half_x)
        y1 = float((2.0 * roi_max[1] - 1.0) * half_y)
        z1 = float((2.0 * roi_max[2] - 1.0) * half_z)

        # Build all 8 bounding-box corners.
        # Bit encoding: bit2=x (0→x0,1→x1), bit1=y, bit0=z.
        # This means XOR-ing with 4 flips x, 2 flips y, 1 flips z.
        xs, ys, zs = (x0, x1), (y0, y1), (z0, z1)
        corners = np.array([
            [xs[(i >> 2) & 1], ys[(i >> 1) & 1], zs[i & 1], 1.0]
            for i in range(8)
        ], dtype=np.float32)  # (8, 4)

        # Transform to camera space and pick front-bottom-left corner.
        # In camera space: +X=right, +Y=up, camera looks in +Z (objects at -Z).
        # Front = max cz (least negative), left = min cx, bottom = min cy.
        cam = (vm @ corners.T).T     # (8, 4)
        cw  = cam[:, 3]
        cx  = cam[:, 0] / cw
        cy  = cam[:, 1] / cw
        cz  = cam[:, 2] / cw
        best = int(np.argmax(cz - cx - cy))

        origin = corners[best,     :3]
        p1_x   = corners[best ^ 4, :3]   # adjacent corner along model X
        p1_y   = corners[best ^ 2, :3]   # adjacent corner along model Y
        p1_z   = corners[best ^ 1, :3]   # adjacent corner along model Z

        axes_data = {
            "X": (origin, p1_x),
            "Y": (origin, p1_y),
            "Z": (origin, p1_z),
        }

        # Build font — regular (not bold) weight; construct QFontMetrics directly
        # so metrics are reliable regardless of the painting context.
        from PyQt6.QtGui import QFontMetrics, QFontMetricsF
        font = QFont("Arial", self.axis_font_size)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        fm = QFontMetricsF(font)
        # Vertical baseline adjustment: centres the cap-height on the anchor point.
        baseline_up = (fm.ascent() - fm.descent()) * 0.5
        # Gap distances that scale with font size.
        # axis_gap: space between axis tip and the axis letter label.
        # tick_gap: space between tick mark end and tick number label.
        axis_gap  = fm.ascent() * 1.2   # ~1 full cap-height clearance
        tick_gap  = fm.ascent() * 0.8   # slightly smaller for tick numbers

        # Screen-space centre of the bounding box — used to determine which
        # perpendicular direction is "outward" for tick labels.
        box_center_model = np.mean(corners[:, :3], axis=0)
        center_s = self._project(box_center_model, mvp, w, h)

        for axis_name, (p0, p1) in axes_data.items():
            axis_vis = self.axis_visible.get(axis_name, True)
            if not axis_vis and not has_ticks:
                continue

            color = self.axis_colors[axis_name]
            pen   = QPen(color, 1)

            s0 = self._project(p0, mvp, w, h)
            s1 = self._project(p1, mvp, w, h)
            if s0 is None or s1 is None:
                continue

            # Axis screen direction (unit vector) and its left perpendicular
            ax_vec  = s1 - s0
            ax_len  = float(np.linalg.norm(ax_vec))
            if ax_len < 1e-4:
                continue
            ax_unit   = ax_vec / ax_len
            perp_unit = np.array([-ax_unit[1], ax_unit[0]], dtype=np.float32)

            # Choose the perpendicular direction that points AWAY from the box
            # centre so that tick labels always appear outside the bounding box.
            if center_s is not None:
                mid_s     = (s0 + s1) * 0.5
                to_center = center_s - mid_s
                if float(np.dot(perp_unit, to_center)) > 0:
                    perp_unit = -perp_unit

            # --- Axis line (only when visible) ---
            if axis_vis:
                painter.setPen(pen)
                painter.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))

            # --- Axis label: placed BEYOND s1, outside the bounding box ---
            if axis_vis:
                label = self.axis_labels.get(axis_name, axis_name)
                lpos  = s1 + ax_unit * axis_gap
                _draw_text_path(painter, font, QBrush(color),
                                lpos[0], lpos[1] + baseline_up, label)

            # --- Tick marks (drawn even when axis line is hidden) ---
            if has_ticks:
                tick_half = 4                         # half tick length in screen px
                label_gap = tick_gap
                n_ticks   = max(1, self.axis_ticks)
                for i in range(1, n_ticks + 1):
                    pt = p0 + (p1 - p0) * (i / n_ticks)
                    sp = self._project(pt, mvp, w, h)
                    if sp is None:
                        continue

                    # Tick line (crosses axis, half inside / half outside)
                    tk0 = sp + perp_unit * tick_half
                    tk1 = sp - perp_unit * tick_half
                    painter.setPen(pen)
                    painter.drawLine(
                        int(tk0[0]), int(tk0[1]),
                        int(tk1[0]), int(tk1[1]),
                    )

                    # Tick label — offset in outward perpendicular direction
                    if self.axis_tick_show_labels:
                        tick_str = f"{i * self.axis_tick_unit:.4g}"
                        lpos     = sp + perp_unit * (tick_half + label_gap)
                        _draw_text_path(painter, font, QBrush(color.lighter(160)),
                                        lpos[0], lpos[1] + baseline_up, tick_str)

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


# ---------------------------------------------------------------------------
# Text rendering helper
# ---------------------------------------------------------------------------

def _draw_text_path(
    painter: QPainter,
    font: QFont,
    brush: QBrush,
    x: float,
    y: float,
    text: str,
) -> None:
    """Draw *text* as a filled vector path at baseline (x, y).

    Using QPainterPath.addText() converts glyphs to filled outlines before
    any OpenGL interaction, completely bypassing Qt's GL glyph-texture cache.
    This eliminates the corrupted / 'tilted-lines' rendering that occurs when
    QPainter.drawText() is called inside QOpenGLWidget.paintGL().
    """
    path = QPainterPath()
    path.addText(QPointF(x, y), font, text)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(brush)
    painter.drawPath(path)
