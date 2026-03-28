"""
ClearVolume-plus — Entry point.

Usage
-----
macOS / Linux:
    python3 scr/main.py [path/to/stack.tif]

Windows:
    py scr/main.py [path/to/stack.tif]

If a file path is provided on the command line it is loaded on startup;
otherwise the application opens a blank window and you can use File → Open.
"""

from __future__ import annotations
import sys
import os


def main() -> None:
    """Create QApplication, show MainWindow, enter event loop."""
    # On Windows, Qt defaults to ANGLE (OpenGL ES via D3D11) which does not
    # support glBlendEquation(GL_MAX).  QT_OPENGL=desktop is the Qt 6
    # supported mechanism to force the native hardware driver; must be set
    # before QApplication is created.
    if sys.platform == "win32":
        os.environ.setdefault("QT_OPENGL", "desktop")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # anti-aliasing
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("ClearVolume-plus")
    app.setOrganizationName("ClearVolumePlus")

    # Optional: load file from CLI argument
    initial_file: str | None = None
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.isfile(candidate):
            initial_file = candidate
        else:
            print(f"Warning: file not found: {candidate}", file=sys.stderr)

    from .gui.main_window import MainWindow
    window = MainWindow(initial_file=initial_file)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
