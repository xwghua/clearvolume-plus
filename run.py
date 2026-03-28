#!/usr/bin/env python3
"""
Top-level launcher for ClearVolume-plus.

Usage
-----
macOS / Linux:
    python3 run.py [path/to/stack.tif]

Windows:
    py run.py [path/to/stack.tif]

Run from the project root directory (the folder containing this file).
"""
import sys
import os

# Ensure the project root is on the path so the package is importable.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scr.main import main

if __name__ == "__main__":
    main()
