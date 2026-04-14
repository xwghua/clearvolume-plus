"""
MainWindow — QMainWindow with menu bar and dock layout.

Layout:
  +---------+------------------+--------+
  | Controls| GL Viewport      |  Axes  |
  |  (left) |   (centre)       | (right)|
  +---------+------------------+--------+

File → Open (TIFF / RAW), Quit
View → Toggle Controls, Toggle Axes, Full-screen
"""

from __future__ import annotations
import os
import datetime

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QApplication,
    QDialog, QRadioButton, QDialogButtonBox, QButtonGroup, QVBoxLayout, QLabel,
)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QKeySequence, QImage, QDragEnterEvent, QDropEvent

from .gl_viewport import GLViewport
from .control_panel import ControlPanel
from .axis_panel import AxisPanel
from ..volume.loader import load_stack, _KIND_LABELS
from ..utils.math_utils import quaternion_from_euler_deg
from .axis_panel import _CAMERA_PRESETS


class MainWindow(QMainWindow):
    """Top-level application window."""

    TITLE = "ClearVolume-plus"

    def __init__(self, initial_file: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle(self.TITLE)
        self.resize(1100, 720)

        # Central GL viewport
        self._viewport = GLViewport()
        self.setCentralWidget(self._viewport)

        # Control panel (left dock)
        self._ctrl = ControlPanel(self._viewport, self)
        self._ctrl.parameter_changed.connect(self._viewport.update)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._ctrl)

        # Axis panel (right dock)
        self._axes = AxisPanel(self._viewport, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._axes)

        # Current file state (for Reinterpret)
        self._current_file_path: str = ""
        self._reinterpret_act: QAction | None = None  # set in _build_menu

        # Video recording state
        self._video_writer = None
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._on_video_frame)

        # Wire up viewport timepoint signal → control panel slider
        self._viewport.timepoint_changed.connect(self._on_viewport_timepoint)

        # Wire up recording signals
        self._ctrl.record_video_requested.connect(self._on_record_requested)
        self._ctrl.record_stop_requested.connect(self._on_record_stop)

        self._build_menu()
        self.setAcceptDrops(True)

        if initial_file:
            self._load_file(initial_file)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")

        open_act = QAction("&Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        export_act = QAction("&Export Visualization…", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._on_export)
        file_menu.addAction(export_act)

        file_menu.addSeparator()

        self._reinterpret_act = QAction("&Reinterpret Volume Type…", self)
        self._reinterpret_act.setEnabled(False)
        self._reinterpret_act.triggered.connect(self._on_reinterpret)
        file_menu.addAction(self._reinterpret_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = mb.addMenu("&View")

        toggle_ctrl = QAction("Toggle &Controls", self)
        toggle_ctrl.setShortcut("Ctrl+Shift+C")
        toggle_ctrl.triggered.connect(lambda: _toggle_dock(self._ctrl))
        view_menu.addAction(toggle_ctrl)

        toggle_axes = QAction("Toggle &Axes Panel", self)
        toggle_axes.setShortcut("Ctrl+Shift+A")
        toggle_axes.triggered.connect(lambda: _toggle_dock(self._axes))
        view_menu.addAction(toggle_axes)

        view_menu.addSeparator()

        # Camera preset submenu (mirrors the quick-set combo in AxisPanel)
        cam_menu = view_menu.addMenu("Camera &Preset")
        for name, angles in _CAMERA_PRESETS[1:]:   # skip placeholder row
            act = QAction(name, self)
            act.triggered.connect(
                lambda _checked, a=angles: self._apply_camera_preset(*a)
            )
            cam_menu.addAction(act)

        view_menu.addSeparator()

        fs_act = QAction("&Full Screen", self)
        fs_act.setShortcut(Qt.Key.Key_F11)
        fs_act.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fs_act)

        # Help menu
        help_menu = mb.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

        shortcuts_act = QAction("&Keyboard Shortcuts", self)
        shortcuts_act.triggered.connect(self._on_shortcuts)
        help_menu.addAction(shortcuts_act)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Volume File",
            "",
            "TIFF stacks (*.tif *.tiff);;Raw binary (*.raw);;All files (*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str, reinterpret_as: str | None = None) -> None:
        self.setWindowTitle(f"{self.TITLE} — Loading…")
        QApplication.processEvents()
        try:
            stack = load_stack(path, parent_widget=self, reinterpret_as=reinterpret_as)
            self._viewport.renderer.set_stack(stack)
            # Apply the Off-axis Front default view on every fresh load
            _off_axis = next(
                (a for name, a in _CAMERA_PRESETS if name == "Off-axis Front"), None
            )
            if _off_axis is not None:
                self._viewport.renderer.rotation = quaternion_from_euler_deg(*_off_axis)
            self._viewport.update()
            self._ctrl.update_for_stack(stack)
            self._ctrl.sync_from_renderer()
            self._current_file_path = path
            if self._reinterpret_act is not None:
                self._reinterpret_act.setEnabled(True)
            fname = os.path.basename(path)
            if stack.is_rgb:
                type_info = "RGB"
            elif stack.n_channels > 1:
                type_info = f"{stack.n_channels}ch"
            else:
                type_info = ""
            t_info = f"{stack.n_timepoints}t" if stack.n_timepoints > 1 else ""
            extras = ", ".join(x for x in [type_info, t_info] if x)
            extras_str = f"  [{extras}]" if extras else ""
            self.setWindowTitle(
                f"{self.TITLE} — {fname}  "
                f"({stack.width}×{stack.height}×{stack.depth}){extras_str}"
            )
            lazy_str = "  (lazy)" if stack.is_lazy else ""
            type_str = "RGB" if stack.is_rgb else f"ch={stack.n_channels}"
            self.statusBar().showMessage(
                f"Loaded: {fname}  "
                f"shape=({stack.depth},{stack.height},{stack.width})  "
                f"{type_str}  t={stack.n_timepoints}"
                f"{lazy_str}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            self.setWindowTitle(self.TITLE)

    def _on_reinterpret(self) -> None:
        """Let the user pick a volume type and reload the current file."""
        if not self._current_file_path:
            return

        stack = self._viewport.renderer._stack
        raw_shape = stack._raw_shape if stack is not None else ()
        raw_shape_str = " × ".join(str(d) for d in raw_shape) if raw_shape else "unknown"

        # Build dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Reinterpret Volume Type")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel(f"<b>Raw data shape:</b> {raw_shape_str}"))
        vl.addWidget(QLabel("Choose how to interpret the axes:"))

        _LABELS_DISPLAY = {
            "MultiCH":  "Multi-channel (Z C Y X)",
            "T":        "Time-lapse (T Z Y X)",
            "MultiCHT": "Time-lapse multi-channel (T Z C Y X)",
            "RGB":      "RGB volume (Z Y X 3)",
            "RGBT":     "Time-lapse RGB (T Z Y X 3)",
        }

        grp = QButtonGroup(dlg)
        radios: dict[str, QRadioButton] = {}
        for key in _KIND_LABELS:
            rb = QRadioButton(_LABELS_DISPLAY[key])
            vl.addWidget(rb)
            grp.addButton(rb)
            radios[key] = rb

        # Pre-select based on current data_kind
        current_kind = stack._data_kind if stack is not None else ""
        kind_to_label = {
            "ZCYX": "MultiCH", "TZCYX": "MultiCHT", "TCZYX": "MultiCHT",
            "TZYX": "T", "ZYXC": "RGB", "TZYXC": "RGBT",
        }
        pre = kind_to_label.get(current_kind, "")
        if pre in radios:
            radios[pre].setChecked(True)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        choice = next((k for k, rb in radios.items() if rb.isChecked()), None)
        if choice is None:
            return

        # Try to reload with the selected interpretation
        try:
            self._load_file(self._current_file_path, reinterpret_as=choice)
        except Exception as exc:
            # Reload with auto-detection and show the error
            self.statusBar().showMessage(
                f"Reinterpret failed ({exc}) — reverting to auto-detected type."
            )
            try:
                self._load_file(self._current_file_path)
            except Exception:
                pass

    def _on_export(self) -> None:
        # Grab the GL viewport framebuffer (renders axes overlay too)
        self._viewport.makeCurrent()
        image = self._viewport.grabFramebuffer()
        self._viewport.doneCurrent()

        if image.isNull():
            QMessageBox.warning(self, "Export", "Nothing to export — no volume loaded.")
            return

        # Determine export directory: 3 levels up from this file → project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        export_dir = os.path.join(project_root, "export")
        os.makedirs(export_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(export_dir, f"clearvolume_{ts}.png")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Visualization",
            default_path,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;TIFF image (*.tif *.tiff)",
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            self._save_tiff(image, path)
        elif image.save(path):
            self.statusBar().showMessage(f"Exported: {path}")
        else:
            QMessageBox.critical(self, "Export error", f"Failed to save image to:\n{path}")

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag if it carries exactly one supported file."""
        mime = event.mimeData()
        if mime.hasUrls():
            paths = [u.toLocalFile() for u in mime.urls()]
            if len(paths) == 1 and _is_supported(paths[0]):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Load the dropped file."""
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        if paths and _is_supported(paths[0]):
            event.acceptProposedAction()
            self._load_file(paths[0])
        else:
            event.ignore()

    def _apply_camera_preset(self, rx: float, ry: float, rz: float) -> None:
        r = self._viewport.renderer
        r.rotation = quaternion_from_euler_deg(rx, ry, rz)
        self._viewport.update()
        self._viewport.transform_changed.emit()

    def _save_tiff(self, image: "QImage", path: str) -> None:
        """Save a QImage as a TIFF using tifffile (preferred) or Qt."""
        try:
            import tifffile
            from PyQt6.QtGui import QImage as _QImage
            img_rgb = image.convertToFormat(_QImage.Format.Format_RGB888)
            ptr = img_rgb.constBits()
            ptr.setsize(img_rgb.sizeInBytes())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                img_rgb.height(), img_rgb.width(), 3
            ).copy()
            tifffile.imwrite(path, arr)
            self.statusBar().showMessage(f"Exported: {path}")
        except ImportError:
            # tifffile not installed — fall back to Qt's built-in TIFF plugin
            if image.save(path):
                self.statusBar().showMessage(f"Exported: {path}")
            else:
                QMessageBox.critical(
                    self, "Export error",
                    "TIFF export requires the tifffile package.\n"
                    "Install with:  pip install tifffile",
                )
        except Exception as exc:
            QMessageBox.critical(self, "Export error", f"Failed to save TIFF:\n{exc}")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About ClearVolume-plus",
            "<b>ClearVolume-plus</b><br>"
            "Python/PyOpenGL 3D volume renderer.<br><br>"
            "Rewrite of the Fiji ClearVolume plugin (Java/JOGL/OpenCL).<br><br>"
            "<b>Developer:</b> Xuanwen Hua<br>"
            "<b>Email:</b> xuanwen@alumni.emory.edu<br>"
            "<b>Repository:</b> https://github.com/xwghua/clearvolume-plus<br><br>"
            "<b>Citation:</b><br>"
            "Xuanwen Hua, <i>ClearVolume-plus</i>,<br>"
            "https://github.com/xwghua/clearvolume-plus, 2026.<br><br>"
            "<b>Mouse controls:</b><br>"
            "Left-drag: rotate &nbsp;&nbsp; Right-drag: pan<br>"
            "Scroll: zoom &nbsp;&nbsp; Ctrl+drag: TF range<br>"
            "Shift+drag: gamma &nbsp;&nbsp; Double-click: full-screen",
        )

    def _on_viewport_timepoint(self, t: int) -> None:
        """Keep the ControlPanel time slider in sync when keyboard changes timepoint."""
        if self._ctrl._sl_time is not None:
            self._ctrl._sl_time.blockSignals(True)
            self._ctrl._sl_time.setValue(t)
            self._ctrl._sl_time.blockSignals(False)
        if self._ctrl._time_label is not None:
            n = self._viewport.renderer.n_timepoints
            self._ctrl._time_label.setText(f"Frame: {t} / {n - 1}")
        self._viewport.update()

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def _on_record_requested(self) -> None:
        """Open a file dialog then start recording frame-by-frame."""
        try:
            import imageio  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "Missing dependency",
                "Video recording requires imageio and imageio-ffmpeg.\n"
                "Install with:  pip install imageio imageio-ffmpeg"
            )
            return

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        export_dir = os.path.join(project_root, "export")
        os.makedirs(export_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(export_dir, f"timelapse_{ts}.mp4")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video",
            default_path,
            "MP4 video (*.mp4);;AVI video (*.avi);;GIF animation (*.gif)",
        )
        if not path:
            return

        import imageio
        fps = 10
        try:
            self._video_writer = imageio.get_writer(path, fps=fps)
        except Exception as exc:
            QMessageBox.critical(self, "Video error", f"Could not open video writer:\n{exc}")
            return

        self._ctrl.set_recording(True)
        self.statusBar().showMessage("Recording… press ■ Stop to finish early.")

        # Write the current frame first, then start the timer
        self._write_current_frame()

        r = self._viewport.renderer
        if r.current_timepoint >= r.n_timepoints - 1:
            # Only one frame — wrap up immediately
            self._finalize_video(path)
        else:
            self._video_output_path = path
            self._video_timer.start(100)  # ~10 fps playback pace

    def _on_video_frame(self) -> None:
        """Advance one timepoint, render, and write frame."""
        r = self._viewport.renderer
        next_t = r.current_timepoint + 1
        if next_t >= r.n_timepoints:
            self._finalize_video(self._video_output_path)
            return

        # Advance timepoint and sync the slider
        r.set_timepoint(next_t)
        if self._ctrl._sl_time is not None:
            self._ctrl._sl_time.blockSignals(True)
            self._ctrl._sl_time.setValue(next_t)
            self._ctrl._sl_time.blockSignals(False)
        if self._ctrl._time_label is not None:
            self._ctrl._time_label.setText(f"Frame: {next_t} / {r.n_timepoints - 1}")

        self._viewport.repaint()   # synchronous render so FBO is ready
        self._write_current_frame()

    def _write_current_frame(self) -> None:
        """Grab the current GL framebuffer and append it to the video."""
        self._viewport.makeCurrent()
        qimg = self._viewport.grabFramebuffer()
        self._viewport.doneCurrent()

        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        ptr = qimg.constBits()
        ptr.setsize(qimg.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            qimg.height(), qimg.width(), 3
        ).copy()

        if self._video_writer is not None:
            self._video_writer.append_data(arr)

    def _on_record_stop(self) -> None:
        """Stop recording early at the current frame."""
        path = getattr(self, "_video_output_path", "")
        self._finalize_video(path)

    def _finalize_video(self, path: str) -> None:
        self._video_timer.stop()
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception:
                pass
            self._video_writer = None
        self._ctrl.set_recording(False)
        self.statusBar().showMessage(f"Video saved: {path}" if path else "Recording stopped.")

    def _on_shortcuts(self) -> None:
        msg = (
            "<b>Keyboard Shortcuts</b><br><br>"
            "<b>I</b>: Cycle render mode<br>"
            "<b>R</b>: Reset rotation/translation<br>"
            "<b>Arrow keys</b>: Pan<br>"
            "<b>[</b> / <b>]</b>: Step time-lapse ±1 frame<br>"
            "<b>Esc</b>: Exit full-screen<br>"
            "<b>F11</b>: Toggle full-screen<br>"
            "<b>Ctrl+O</b>: Open file<br>"
            "<b>Ctrl+E</b>: Export visualization to PNG<br>"
            "<b>Ctrl+Q</b>: Quit<br>"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", msg)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _toggle_dock(dock) -> None:
    dock.setVisible(not dock.isVisible())


def _is_supported(path: str) -> bool:
    """Return True if *path* has a file extension the loader can handle."""
    return os.path.splitext(path)[1].lower() in ('.tif', '.tiff', '.raw')
