"""
AxisPanel — QDockWidget for axis overlays, volume ROI clipping,
bounding-box / mesh overlay, coordinate transform options, and camera
transform fields (rotation, translation, camera distance).

Controls:
  - Axis Visibility & Labels group: show/hide, label text, colour per axis
  - Tick Marks group: count, physical unit spacing
  - ROI Clipping group: per-axis dual-handle range sliders (texture-space [0, 1])
  - Overlay group: bounding box, mesh grid, mesh divisions
  - Transform group: flip coordinates
  - Camera & Transform group: Euler angles (Rx/Ry/Rz), translation (Tx/Ty),
    camera distance — all bidirectionally synced with mouse operations
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QGroupBox, QColorDialog,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal

from .gl_viewport import GLViewport
from .range_slider import RangeSlider
from ..utils.math_utils import (
    quaternion_to_euler_deg, quaternion_from_euler_deg, quaternion_identity,
)


class AxisPanel(QDockWidget):
    """Axis overlay + ROI clipping + overlay + transform + camera panel (right dock)."""

    changed = pyqtSignal()

    def __init__(self, viewport: GLViewport, parent=None) -> None:
        super().__init__("Axes & ROI", parent)
        self._vp = viewport
        self._color_buttons: dict[str, QPushButton] = {}

        # Camera spinbox refs (set in _make_camera_group)
        self._spin_rx: QDoubleSpinBox | None = None
        self._spin_ry: QDoubleSpinBox | None = None
        self._spin_rz: QDoubleSpinBox | None = None
        self._spin_tx: QDoubleSpinBox | None = None
        self._spin_ty: QDoubleSpinBox | None = None
        self._spin_dist: QDoubleSpinBox | None = None

        # Guard to avoid feedback loops when updating spinboxes from signals
        self._updating_camera: bool = False

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._make_axes_group())
        layout.addWidget(self._make_ticks_group())
        layout.addWidget(self._make_roi_group())
        layout.addWidget(self._make_overlay_group())
        layout.addWidget(self._make_transform_group())
        layout.addWidget(self._make_camera_group())
        layout.addStretch()

        self.setWidget(widget)
        self.setMinimumWidth(240)

        # Sync camera spinboxes whenever the viewport emits transform_changed
        self._vp.transform_changed.connect(self._sync_camera_spinboxes)

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _make_axes_group(self) -> QGroupBox:
        box = QGroupBox("Axis Visibility & Labels")
        grid = QGridLayout(box)
        grid.setSpacing(4)

        headers = ["Axis", "Show", "Label", "Colour"]
        for col, h in enumerate(headers):
            lbl = QLabel(f"<b>{h}</b>")
            grid.addWidget(lbl, 0, col)

        for row_idx, axis in enumerate(["X", "Y", "Z"], start=1):
            grid.addWidget(QLabel(axis), row_idx, 0)

            cb = QCheckBox()
            cb.setChecked(self._vp.axis_visible.get(axis, True))
            cb.toggled.connect(lambda checked, a=axis: self._on_vis(a, checked))
            grid.addWidget(cb, row_idx, 1)

            le = QLineEdit(self._vp.axis_labels.get(axis, axis))
            le.setMaximumWidth(70)
            le.textChanged.connect(lambda text, a=axis: self._on_label(a, text))
            grid.addWidget(le, row_idx, 2)

            btn = QPushButton()
            btn.setFixedSize(28, 22)
            color = self._vp.axis_colors.get(axis, QColor(200, 200, 200))
            self._set_button_color(btn, color)
            btn.clicked.connect(lambda _checked, a=axis: self._on_color(a))
            self._color_buttons[axis] = btn
            grid.addWidget(btn, row_idx, 3)

        return box

    def _make_ticks_group(self) -> QGroupBox:
        box = QGroupBox("Tick Marks")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Count"))
        self._spin_ticks = QSpinBox()
        self._spin_ticks.setRange(0, 20)
        self._spin_ticks.setValue(self._vp.axis_ticks)
        self._spin_ticks.valueChanged.connect(self._on_tick_count)
        row.addWidget(self._spin_ticks)
        vl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Unit spacing"))
        self._spin_unit = QDoubleSpinBox()
        self._spin_unit.setRange(0.001, 1e6)
        self._spin_unit.setDecimals(3)
        self._spin_unit.setValue(self._vp.axis_tick_unit)
        self._spin_unit.valueChanged.connect(self._on_unit)
        row2.addWidget(self._spin_unit)
        vl.addLayout(row2)

        self._chk_tick_labels = QCheckBox("Show tick numbers")
        self._chk_tick_labels.setChecked(self._vp.axis_tick_show_labels)
        self._chk_tick_labels.toggled.connect(self._on_tick_labels)
        vl.addWidget(self._chk_tick_labels)

        return box

    def _make_roi_group(self) -> QGroupBox:
        """ROI Clipping — three dual-handle range sliders (x/y/z) in texture-space [0,1]."""
        box = QGroupBox("ROI Clipping")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        self._rs_x = _range_slider(0, 100, 0, 100, "X", vl, self._on_roi_x)
        self._rs_y = _range_slider(0, 100, 0, 100, "Y", vl, self._on_roi_y)
        self._rs_z = _range_slider(0, 100, 0, 100, "Z", vl, self._on_roi_z)

        return box

    def _make_overlay_group(self) -> QGroupBox:
        box = QGroupBox("Overlay")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        r = self._vp.renderer
        self._chk_box = QCheckBox("Show bounding box")
        self._chk_box.setChecked(r.show_box)
        self._chk_box.toggled.connect(self._on_show_box)
        vl.addWidget(self._chk_box)

        self._chk_mesh = QCheckBox("Show mesh grid")
        self._chk_mesh.setChecked(r.show_mesh)
        self._chk_mesh.toggled.connect(self._on_show_mesh)
        vl.addWidget(self._chk_mesh)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mesh divisions"))
        self._spin_mesh = QSpinBox()
        self._spin_mesh.setRange(2, 20)
        self._spin_mesh.setValue(r.mesh_divisions)
        self._spin_mesh.valueChanged.connect(self._on_mesh_divisions)
        row.addWidget(self._spin_mesh)
        vl.addLayout(row)

        return box

    def _make_transform_group(self) -> QGroupBox:
        box = QGroupBox("Transform")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        self._chk_flip = QCheckBox("Flip coordinates (mirror all axes)")
        self._chk_flip.setChecked(self._vp.renderer.flip_coords)
        self._chk_flip.toggled.connect(self._on_flip_coords)
        vl.addWidget(self._chk_flip)
        return box

    def _make_camera_group(self) -> QGroupBox:
        """Camera & Transform — Euler angles, translation, distance with precise control."""
        box = QGroupBox("Camera & Transform")
        grid = QGridLayout(box)
        grid.setSpacing(4)

        r = self._vp.renderer
        rx, ry, rz = quaternion_to_euler_deg(r.rotation)
        tx, ty = float(r.translation[0]), float(r.translation[1])
        dist = float(r._camera_distance)

        def _spin(row, label, value, lo, hi, decimals, callback):
            grid.addWidget(QLabel(label), row, 0)
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(decimals)
            sb.setValue(value)
            sb.setSingleStep(1.0 if decimals == 1 else 0.01)
            sb.valueChanged.connect(callback)
            grid.addWidget(sb, row, 1)
            return sb

        self._spin_rx   = _spin(0, "Rx (°)",  rx,   -360.0, 360.0, 1, self._on_rx)
        self._spin_ry   = _spin(1, "Ry (°)",  ry,   -360.0, 360.0, 1, self._on_ry)
        self._spin_rz   = _spin(2, "Rz (°)",  rz,   -360.0, 360.0, 1, self._on_rz)
        self._spin_tx   = _spin(3, "Tx",       tx,   -10.0,  10.0,  3, self._on_tx)
        self._spin_ty   = _spin(4, "Ty",       ty,   -10.0,  10.0,  3, self._on_ty)
        self._spin_dist = _spin(5, "Distance", dist,  0.1,   20.0,  2, self._on_dist)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._on_reset_camera)
        grid.addWidget(btn_reset, 6, 0, 1, 2)

        return box

    # ------------------------------------------------------------------
    # Callbacks — axes
    # ------------------------------------------------------------------

    def _on_vis(self, axis: str, checked: bool) -> None:
        self._vp.axis_visible[axis] = checked
        self._vp.update()
        self.changed.emit()

    def _on_label(self, axis: str, text: str) -> None:
        self._vp.axis_labels[axis] = text
        self._vp.update()
        self.changed.emit()

    def _on_color(self, axis: str) -> None:
        current = self._vp.axis_colors.get(axis, QColor(200, 200, 200))
        color = QColorDialog.getColor(current, self, f"Choose {axis} axis colour")
        if color.isValid():
            self._vp.axis_colors[axis] = color
            self._set_button_color(self._color_buttons[axis], color)
            self._vp.update()
            self.changed.emit()

    def _on_tick_count(self, value: int) -> None:
        self._vp.axis_ticks = value
        self._vp.update()
        self.changed.emit()

    def _on_unit(self, value: float) -> None:
        self._vp.axis_tick_unit = value
        self._vp.update()
        self.changed.emit()

    def _on_tick_labels(self, checked: bool) -> None:
        self._vp.axis_tick_show_labels = checked
        self._vp.update()
        self.changed.emit()

    # ------------------------------------------------------------------
    # Callbacks — ROI (dual-handle range sliders)
    # ------------------------------------------------------------------

    def _on_roi_x(self, lo: int, hi: int) -> None:
        r = self._vp.renderer
        r.roi_min[0] = lo / 100.0
        r.roi_max[0] = hi / 100.0
        r.mark_box_dirty()
        self._vp.update()

    def _on_roi_y(self, lo: int, hi: int) -> None:
        r = self._vp.renderer
        r.roi_min[1] = lo / 100.0
        r.roi_max[1] = hi / 100.0
        r.mark_box_dirty()
        self._vp.update()

    def _on_roi_z(self, lo: int, hi: int) -> None:
        r = self._vp.renderer
        r.roi_min[2] = lo / 100.0
        r.roi_max[2] = hi / 100.0
        r.mark_box_dirty()
        self._vp.update()

    # ------------------------------------------------------------------
    # Callbacks — overlay
    # ------------------------------------------------------------------

    def _on_show_box(self, checked: bool) -> None:
        self._vp.renderer.show_box = checked
        self._vp.update()
        self.changed.emit()

    def _on_show_mesh(self, checked: bool) -> None:
        self._vp.renderer.show_mesh = checked
        self._vp.update()
        self.changed.emit()

    def _on_mesh_divisions(self, value: int) -> None:
        r = self._vp.renderer
        r.mesh_divisions = value
        r.mark_box_dirty()
        self._vp.update()
        self.changed.emit()

    def _on_flip_coords(self, checked: bool) -> None:
        self._vp.renderer.flip_coords = checked
        self._vp.update()
        self.changed.emit()

    # ------------------------------------------------------------------
    # Callbacks — camera spinboxes
    # ------------------------------------------------------------------

    def _on_rx(self, value: float) -> None:
        if self._updating_camera:
            return
        r = self._vp.renderer
        _, ry, rz = quaternion_to_euler_deg(r.rotation)
        r.rotation = quaternion_from_euler_deg(value, ry, rz)
        self._vp.update()

    def _on_ry(self, value: float) -> None:
        if self._updating_camera:
            return
        r = self._vp.renderer
        rx, _, rz = quaternion_to_euler_deg(r.rotation)
        r.rotation = quaternion_from_euler_deg(rx, value, rz)
        self._vp.update()

    def _on_rz(self, value: float) -> None:
        if self._updating_camera:
            return
        r = self._vp.renderer
        rx, ry, _ = quaternion_to_euler_deg(r.rotation)
        r.rotation = quaternion_from_euler_deg(rx, ry, value)
        self._vp.update()

    def _on_tx(self, value: float) -> None:
        if self._updating_camera:
            return
        r = self._vp.renderer
        t = r.translation.copy()
        t[0] = value
        r.translation = t
        self._vp.update()

    def _on_ty(self, value: float) -> None:
        if self._updating_camera:
            return
        r = self._vp.renderer
        t = r.translation.copy()
        t[1] = value
        r.translation = t
        self._vp.update()

    def _on_dist(self, value: float) -> None:
        if self._updating_camera:
            return
        self._vp.renderer._camera_distance = value
        self._vp.update()

    def _on_reset_camera(self) -> None:
        r = self._vp.renderer
        r.rotation = quaternion_identity()
        r.translation = np.zeros(2, dtype=np.float32)
        if r._volume is not None:
            import math
            asp = r._volume.aspect_ratio
            half = max(float(asp[0]), float(asp[1])) * 0.5
            r._camera_distance = half / math.tan(math.radians(r._fov_deg * 0.5)) * 1.25
        else:
            r._camera_distance = 2.0
        self._vp.update()
        self._sync_camera_spinboxes()

    # ------------------------------------------------------------------
    # Camera sync (called by transform_changed signal)
    # ------------------------------------------------------------------

    def _sync_camera_spinboxes(self) -> None:
        if (self._spin_rx is None or self._updating_camera):
            return
        self._updating_camera = True
        try:
            r = self._vp.renderer
            rx, ry, rz = quaternion_to_euler_deg(r.rotation)
            self._spin_rx.setValue(rx)
            self._spin_ry.setValue(ry)
            self._spin_rz.setValue(rz)
            self._spin_tx.setValue(float(r.translation[0]))
            self._spin_ty.setValue(float(r.translation[1]))
            self._spin_dist.setValue(float(r._camera_distance))
        finally:
            self._updating_camera = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_button_color(btn: QPushButton, color: QColor) -> None:
        btn.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #888;"
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _range_slider(minimum: int, maximum: int, lo: int, hi: int,
                  label: str, layout: QVBoxLayout,
                  callback) -> RangeSlider:
    """Create a labelled dual-handle range slider and add it to *layout*."""
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setFixedWidth(20)
    row.addWidget(lbl)
    rs = RangeSlider(minimum, maximum, lo, hi)
    rs.range_changed.connect(callback)
    row.addWidget(rs)
    layout.addLayout(row)
    return rs
