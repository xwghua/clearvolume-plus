"""
ControlPanel — QDockWidget mirroring Java's ControlJPanel.

Single-channel mode:
  Groups: Render, Intensity (gamma/brightness/range), Quality

Multi-channel mode:
  Groups: Render, Channels (per-channel: visibility/TF/color/brightness/gamma/range),
          Quality
  The global Intensity group is hidden; each channel has its own sliders.

Time-lapse:
  Adds a Time group with a frame slider and Record button.

ROI Clipping, Overlay, and Transform have been moved to the right AxisPanel.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QGroupBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QColorDialog, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..renderer.volume_renderer import VolumeRenderer
from ..renderer.transfer_function import TransferFunction1D
from ..renderer.render_algorithm import RenderAlgorithm


_DEFAULT_CHANNEL_COLORS = [
    QColor(0,   220,  80),
    QColor(220,   0, 180),
    QColor(0,   180, 255),
    QColor(255, 220,   0),
]
_DEFAULT_TF_NAMES = ["Green", "Magenta", "Cyan", "Yellow"]


class ControlPanel(QDockWidget):
    """Control panel dock widget."""

    parameter_changed = pyqtSignal()
    record_video_requested = pyqtSignal()
    record_stop_requested = pyqtSignal()

    def __init__(self, renderer: VolumeRenderer, parent=None) -> None:
        super().__init__("Controls", parent)
        self._renderer = renderer
        self._tf_presets = TransferFunction1D.all_presets()
        self._building = False

        # Dynamic groups
        self._ch_group: QGroupBox | None = None
        self._time_group: QGroupBox | None = None

        # Per-channel slider lists (populated by _make_channel_group)
        self._ch_checkboxes:    list[QCheckBox] = []
        self._ch_tf_combos:     list[QComboBox] = []
        self._ch_color_btns:    list[QPushButton] = []
        self._ch_sl_brightness: list[QSlider] = []
        self._ch_sl_gamma:      list[QSlider] = []
        self._ch_sl_range_min:  list[QSlider] = []
        self._ch_sl_range_max:  list[QSlider] = []

        self._sl_time:    QSlider   | None = None
        self._time_label: QLabel   | None = None
        self._btn_record: QPushButton | None = None
        self._is_recording: bool = False
        self._is_rgb: bool = False

        widget = QWidget()
        self._main_layout = QVBoxLayout(widget)
        self._main_layout.setContentsMargins(8, 8, 8, 8)
        self._main_layout.setSpacing(6)

        self._main_layout.addWidget(self._make_render_group())
        self._intensity_group = self._make_intensity_group()
        self._main_layout.addWidget(self._intensity_group)
        self._main_layout.addWidget(self._make_quality_group())
        self._main_layout.addStretch()

        self.setWidget(widget)
        self.setMinimumWidth(260)

    # ------------------------------------------------------------------
    # Dynamic stack-aware update
    # ------------------------------------------------------------------

    def update_for_stack(self, stack) -> None:
        """Show/rebuild channel and time controls based on the loaded stack."""
        from ..volume.volume import VolumeStack
        if not isinstance(stack, VolumeStack):
            return

        # Remove old dynamic groups
        if self._ch_group is not None:
            self._main_layout.removeWidget(self._ch_group)
            self._ch_group.deleteLater()
            self._ch_group = None
        if self._time_group is not None:
            self._main_layout.removeWidget(self._time_group)
            self._time_group.deleteLater()
            self._time_group = None

        # For RGB volumes: show global intensity group but hide gamma and range
        # controls — only brightness applies (scales all R/G/B channels equally).
        # For multi-channel: hide global intensity group, show per-channel group.
        if stack.is_rgb:
            self._is_rgb = True
            self._intensity_group.setVisible(True)
            self._combo_tf.setEnabled(False)
            self._combo_tf.setToolTip("Transfer function is disabled for RGB volumes — original colours are preserved.")
            self._gamma_widget.setVisible(False)
            self._range_min_widget.setVisible(False)
            self._range_max_widget.setVisible(False)
        elif stack.n_channels > 1:
            self._is_rgb = False
            self._intensity_group.setVisible(False)
            self._combo_tf.setEnabled(False)
            self._combo_tf.setToolTip("Per-channel transfer functions are set in the Channels group above.")
        else:
            self._is_rgb = False
            self._intensity_group.setVisible(True)
            self._combo_tf.setEnabled(True)
            self._combo_tf.setToolTip("")
            self._gamma_widget.setVisible(True)
            self._range_min_widget.setVisible(True)
            self._range_max_widget.setVisible(True)

        # Stretch is the last item; insert before it
        stretch_pos = self._main_layout.count() - 1

        if stack.n_channels > 1 and not stack.is_rgb:
            self._ch_group = self._make_channel_group(stack.n_channels)
            # Insert after Render group (index 1)
            self._main_layout.insertWidget(1, self._ch_group)
            stretch_pos += 1
            # Push default TFs into renderer
            for c in range(stack.n_channels):
                name = _DEFAULT_TF_NAMES[c % len(_DEFAULT_TF_NAMES)]
                idx = next(
                    (i for i, (n, _) in enumerate(self._tf_presets) if n == name), 0
                )
                _, tf = self._tf_presets[idx]
                self._renderer.set_channel_tf(c, tf)

        if stack.n_timepoints > 1:
            self._time_group = self._make_time_group(stack.n_timepoints)
            self._main_layout.insertWidget(self._main_layout.count() - 1,
                                           self._time_group)

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _make_render_group(self) -> QGroupBox:
        box = QGroupBox("Render")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mode"))
        self._combo_mode = QComboBox()
        for algo in RenderAlgorithm:
            self._combo_mode.addItem(algo.label(), userData=algo)
        self._combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(self._combo_mode)
        vl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Transfer fn"))
        self._combo_tf = QComboBox()
        for name, _ in self._tf_presets:
            self._combo_tf.addItem(name)
        self._combo_tf.currentIndexChanged.connect(self._on_tf_changed)
        row2.addWidget(self._combo_tf)
        vl.addLayout(row2)

        return box

    def _make_intensity_group(self) -> QGroupBox:
        box = QGroupBox("Intensity")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        # Gamma, range_min, and range_max are hidden for RGB volumes; wrap each
        # in a QWidget container so the whole row (label + slider) can be toggled.
        self._gamma_widget, self._sl_gamma = _slider_row(
            1, 500, int(self._renderer.gamma * 100), "Gamma", vl, self._on_gamma)
        self._sl_brightness = _slider(10, 1000, int(self._renderer.brightness * 100),
                                       "Brightness", vl, self._on_brightness)
        self._range_min_widget, self._sl_range_min = _slider_row(
            0, 100, int(self._renderer.range_min * 100), "Range min", vl, self._on_range_min)
        self._range_max_widget, self._sl_range_max = _slider_row(
            0, 200, int(self._renderer.range_max * 100), "Range max", vl, self._on_range_max)
        return box

    def _make_quality_group(self) -> QGroupBox:
        box = QGroupBox("Quality")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        self._sl_quality = _slider(10, 100, int(self._renderer.quality * 100),
                                    "Quality", vl, self._on_quality)
        self._sl_dithering = _slider(0, 100, int(self._renderer.dithering * 100),
                                      "Dithering", vl, self._on_dithering)

        row = QHBoxLayout()
        row.addWidget(QLabel("Iso value"))
        self._spin_iso = QDoubleSpinBox()
        self._spin_iso.setRange(0.0, 1.0)
        self._spin_iso.setSingleStep(0.01)
        self._spin_iso.setDecimals(3)
        self._spin_iso.setValue(self._renderer.iso_value)
        self._spin_iso.valueChanged.connect(self._on_iso_value)
        row.addWidget(self._spin_iso)
        vl.addLayout(row)

        return box

    def _make_channel_group(self, n_channels: int) -> QGroupBox:
        """Per-channel: visibility checkbox, colour picker, TF dropdown, and sliders."""
        box = QGroupBox("Channels")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        self._ch_checkboxes    = []
        self._ch_tf_combos     = []
        self._ch_color_btns    = []
        self._ch_sl_brightness = []
        self._ch_sl_gamma      = []
        self._ch_sl_range_min  = []
        self._ch_sl_range_max  = []

        for c in range(n_channels):
            # --- separator before each channel (except first) ---
            if c > 0:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                vl.addWidget(line)

            # --- Header row: checkbox + colour button + TF dropdown ---
            hdr = QHBoxLayout()

            chk = QCheckBox(f"Ch {c}")
            chk.setChecked(True)
            chk.toggled.connect(lambda v, ch=c: self._on_channel_visible(ch, v))
            self._ch_checkboxes.append(chk)
            hdr.addWidget(chk)

            default_color = _DEFAULT_CHANNEL_COLORS[c % len(_DEFAULT_CHANNEL_COLORS)]
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            btn.setToolTip(f"Pick colour for channel {c}")
            _set_btn_color(btn, default_color)
            btn.clicked.connect(lambda _, ch=c: self._on_channel_color(ch))
            self._ch_color_btns.append(btn)
            hdr.addWidget(btn)

            combo = QComboBox()
            for name, _ in self._tf_presets:
                combo.addItem(name)
            default_name = _DEFAULT_TF_NAMES[c % len(_DEFAULT_TF_NAMES)]
            default_idx = next(
                (i for i, (n, _) in enumerate(self._tf_presets) if n == default_name), 0
            )
            combo.setCurrentIndex(default_idx)
            combo.currentIndexChanged.connect(
                lambda idx, ch=c: self._on_channel_tf(ch, idx)
            )
            self._ch_tf_combos.append(combo)
            hdr.addWidget(combo)
            vl.addLayout(hdr)

            # --- Per-channel intensity sliders ---
            sl_b = _slider(10, 1000, int(self._renderer.get_channel_brightness(c) * 100),
                           "Brightness", vl, lambda v, ch=c: self._on_ch_brightness(ch, v))
            sl_g = _slider(1, 500, int(self._renderer.get_channel_gamma(c) * 100),
                           "Gamma", vl, lambda v, ch=c: self._on_ch_gamma(ch, v))
            sl_rn = _slider(0, 100, int(self._renderer.get_channel_range_min(c) * 100),
                            "Range min", vl, lambda v, ch=c: self._on_ch_range_min(ch, v))
            sl_rx = _slider(0, 200, int(self._renderer.get_channel_range_max(c) * 100),
                            "Range max", vl, lambda v, ch=c: self._on_ch_range_max(ch, v))
            self._ch_sl_brightness.append(sl_b)
            self._ch_sl_gamma.append(sl_g)
            self._ch_sl_range_min.append(sl_rn)
            self._ch_sl_range_max.append(sl_rx)

        return box

    def _make_time_group(self, n_timepoints: int) -> QGroupBox:
        box = QGroupBox("Time")
        vl = QVBoxLayout(box)
        vl.setSpacing(4)

        self._time_label = QLabel(f"Frame: 0 / {n_timepoints - 1}")
        vl.addWidget(self._time_label)

        self._sl_time = QSlider(Qt.Orientation.Horizontal)
        self._sl_time.setMinimum(0)
        self._sl_time.setMaximum(n_timepoints - 1)
        self._sl_time.setValue(0)
        self._sl_time.valueChanged.connect(self._on_time_changed)
        vl.addWidget(self._sl_time)

        row = QHBoxLayout()
        self._btn_record = QPushButton("● Rec")
        self._btn_record.setToolTip("Record time-lapse to video file")
        self._btn_record.clicked.connect(self._on_record_clicked)
        row.addWidget(self._btn_record)
        row.addStretch()
        vl.addLayout(row)

        return box

    # ------------------------------------------------------------------
    # Slot callbacks — render mode and global TF
    # ------------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        algo = self._combo_mode.itemData(idx)
        if algo is not None:
            self._renderer.set_algorithm(algo)
            self.parameter_changed.emit()

    def _on_tf_changed(self, idx: int) -> None:
        _, tf = self._tf_presets[idx]
        self._renderer.set_transfer_function(tf)
        self.parameter_changed.emit()

    # ------------------------------------------------------------------
    # Slot callbacks — global intensity (single-channel)
    # ------------------------------------------------------------------

    def _on_gamma(self, value: int) -> None:
        self._renderer.gamma = value / 100.0
        self.parameter_changed.emit()

    def _on_brightness(self, value: int) -> None:
        v = value / 100.0
        if self._is_rgb:
            # RGB: apply brightness uniformly to all three colour channels.
            for c in range(3):
                self._renderer.set_channel_brightness(c, v)
        else:
            self._renderer.brightness = v
        self.parameter_changed.emit()

    def _on_range_min(self, value: int) -> None:
        new_min = value / 100.0
        if new_min < self._renderer.range_max:
            self._renderer.range_min = new_min
            self.parameter_changed.emit()

    def _on_range_max(self, value: int) -> None:
        new_max = value / 100.0
        if new_max > self._renderer.range_min:
            self._renderer.range_max = new_max
            self.parameter_changed.emit()

    # ------------------------------------------------------------------
    # Slot callbacks — quality
    # ------------------------------------------------------------------

    def _on_quality(self, value: int) -> None:
        self._renderer.quality = value / 100.0
        self.parameter_changed.emit()

    def _on_dithering(self, value: int) -> None:
        self._renderer.dithering = value / 100.0
        self.parameter_changed.emit()

    def _on_iso_value(self, value: float) -> None:
        self._renderer.iso_value = float(value)
        self.parameter_changed.emit()

    # ------------------------------------------------------------------
    # Slot callbacks — per-channel intensity
    # ------------------------------------------------------------------

    def _on_ch_brightness(self, c: int, value: int) -> None:
        self._renderer.set_channel_brightness(c, value / 100.0)
        self.parameter_changed.emit()

    def _on_ch_gamma(self, c: int, value: int) -> None:
        self._renderer.set_channel_gamma(c, value / 100.0)
        self.parameter_changed.emit()

    def _on_ch_range_min(self, c: int, value: int) -> None:
        v = value / 100.0
        if v < self._renderer.get_channel_range_max(c):
            self._renderer.set_channel_range_min(c, v)
            self.parameter_changed.emit()

    def _on_ch_range_max(self, c: int, value: int) -> None:
        v = value / 100.0
        if v > self._renderer.get_channel_range_min(c):
            self._renderer.set_channel_range_max(c, v)
            self.parameter_changed.emit()

    # ------------------------------------------------------------------
    # Slot callbacks — channel TF, colour, visibility
    # ------------------------------------------------------------------

    def _on_channel_visible(self, c: int, visible: bool) -> None:
        self._renderer.set_channel_visible(c, visible)
        self.parameter_changed.emit()

    def _on_channel_tf(self, c: int, idx: int) -> None:
        _, tf = self._tf_presets[idx]
        self._renderer.set_channel_tf(c, tf)
        self.parameter_changed.emit()

    def _on_channel_color(self, c: int) -> None:
        current_color = _DEFAULT_CHANNEL_COLORS[c % len(_DEFAULT_CHANNEL_COLORS)]
        if c < len(self._ch_color_btns):
            style = self._ch_color_btns[c].styleSheet()
            try:
                hex_str = style.split("background-color:")[-1].strip().rstrip(";").strip()
                current_color = QColor(hex_str)
            except Exception:
                pass

        color = QColorDialog.getColor(current_color, self)
        if not color.isValid():
            return

        if c < len(self._ch_color_btns):
            _set_btn_color(self._ch_color_btns[c], color)

        tf = TransferFunction1D.from_color(
            color.redF(), color.greenF(), color.blueF()
        )
        self._renderer.set_channel_tf(c, tf)

        if c < len(self._ch_tf_combos):
            self._ch_tf_combos[c].blockSignals(True)
            self._ch_tf_combos[c].setCurrentIndex(-1)
            self._ch_tf_combos[c].blockSignals(False)

        self.parameter_changed.emit()

    # ------------------------------------------------------------------
    # Slot callbacks — time and recording
    # ------------------------------------------------------------------

    def _on_time_changed(self, value: int) -> None:
        self._renderer.set_timepoint(value)
        n = self._renderer.n_timepoints
        if self._time_label is not None:
            self._time_label.setText(f"Frame: {value} / {n - 1}")
        self.parameter_changed.emit()

    def _on_record_clicked(self) -> None:
        if self._is_recording:
            self.record_stop_requested.emit()
        else:
            self.record_video_requested.emit()

    def set_recording(self, recording: bool) -> None:
        """Called by MainWindow to update button state."""
        self._is_recording = recording
        if self._btn_record is not None:
            self._btn_record.setText("■ Stop" if recording else "● Rec")

    # ------------------------------------------------------------------
    # Sync from renderer
    # ------------------------------------------------------------------

    def sync_from_renderer(self) -> None:
        """Refresh all slider values from renderer state."""
        self._building = True

        # Global sliders (channel 0)
        self._sl_gamma.setValue(int(self._renderer.gamma * 100))
        self._sl_brightness.setValue(int(self._renderer.brightness * 100))
        self._sl_range_min.setValue(int(self._renderer.range_min * 100))
        self._sl_range_max.setValue(int(self._renderer.range_max * 100))
        self._sl_quality.setValue(int(self._renderer.quality * 100))
        self._sl_dithering.setValue(int(self._renderer.dithering * 100))
        self._spin_iso.setValue(self._renderer.iso_value)

        # Per-channel sliders
        for c, (sl_b, sl_g, sl_rn, sl_rx) in enumerate(zip(
            self._ch_sl_brightness, self._ch_sl_gamma,
            self._ch_sl_range_min,  self._ch_sl_range_max,
        )):
            sl_b.setValue(int(self._renderer.get_channel_brightness(c) * 100))
            sl_g.setValue(int(self._renderer.get_channel_gamma(c) * 100))
            sl_rn.setValue(int(self._renderer.get_channel_range_min(c) * 100))
            sl_rx.setValue(int(self._renderer.get_channel_range_max(c) * 100))

        if self._sl_time is not None:
            self._sl_time.setValue(self._renderer.current_timepoint)

        self._building = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slider(minimum: int, maximum: int, value: int,
            label: str, layout: QVBoxLayout,
            callback) -> QSlider:
    """Create a labelled horizontal slider and add it to *layout*."""
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setFixedWidth(70)
    row.addWidget(lbl)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    slider.valueChanged.connect(callback)
    row.addWidget(slider)
    layout.addLayout(row)
    return slider


def _slider_row(minimum: int, maximum: int, value: int,
                label: str, layout: QVBoxLayout,
                callback) -> tuple:
    """Create a labelled slider inside a QWidget container.

    Returns (container_widget, slider).  Calling container_widget.setVisible(False)
    hides both the label and slider as a unit — used for rows that are
    conditionally hidden (e.g. gamma/range are hidden for RGB volumes).
    """
    from PyQt6.QtWidgets import QWidget
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setFixedWidth(70)
    row.addWidget(lbl)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    slider.valueChanged.connect(callback)
    row.addWidget(slider)
    layout.addWidget(container)
    return container, slider


def _set_btn_color(btn: QPushButton, color: QColor) -> None:
    btn.setStyleSheet(
        f"background-color: {color.name()}; "
        "border: 1px solid #555; border-radius: 2px;"
    )
