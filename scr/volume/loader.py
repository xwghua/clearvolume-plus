"""Volume loader — reads TIFF stacks and raw binary files into Volume / VolumeStack objects."""

from __future__ import annotations
import os
import numpy as np
from .volume import Volume, VolumeStack

# Threshold (bytes, as float32) above which time-lapse stacks are loaded lazily.
MEMORY_LIMIT_BYTES = 1 * 1024 ** 3  # 1 GB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(file_path: str, **kwargs) -> Volume:
    """
    Load a single-frame volume from *file_path* (legacy API, single channel).

    Returns a normalised Volume.  For multi-channel / time-lapse files, only
    channel 0 / timepoint 0 is returned.  Use ``load_stack()`` for full data.
    """
    stack = load_stack(file_path, **kwargs)
    frame = stack.get_frame(0)
    return Volume(
        data=frame[0],
        voxel_size_x=stack.voxel_size_x,
        voxel_size_y=stack.voxel_size_y,
        voxel_size_z=stack.voxel_size_z,
        file_path=file_path,
    )


def load_stack(file_path: str, parent_widget=None,
               reinterpret_as: str | None = None, **kwargs) -> VolumeStack:
    """
    Load a volume file and return a VolumeStack.

    Automatically detects:
    - Single-channel 3D stacks
    - Multi-channel (C) stacks
    - Time-lapse (T) stacks
    - Multi-channel time-lapse stacks

    If the dimensionality is ambiguous, *parent_widget* (a QWidget) is used as
    the parent for a Qt disambiguation dialog.  If None, a heuristic is used.

    Large stacks (> MEMORY_LIMIT_BYTES float32) are loaded lazily: only one
    timepoint is in RAM at a time.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Volume file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".tif", ".tiff"):
        return _load_tiff_stack(file_path, parent_widget, reinterpret_as)
    elif ext == ".raw":
        vol = _load_raw(file_path, **kwargs)
        return _volume_to_stack(vol.normalise())
    else:
        raise ValueError(f"Unsupported volume format: {ext!r}")


# ---------------------------------------------------------------------------
# TIFF stack loader
# ---------------------------------------------------------------------------

def _load_tiff_stack(path: str, parent_widget=None,
                     reinterpret_as: str | None = None) -> VolumeStack:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "tifffile is required.  Install with: pip install tifffile"
        ) from exc

    # --- Probe shape & voxel sizes without loading all data ----------------
    with tifffile.TiffFile(path) as tif:
        if tif.series:
            raw_shape = tif.series[0].shape
            raw_dtype = tif.series[0].dtype
        else:
            raw_shape = tif.asarray().shape
            raw_dtype = tif.pages[0].dtype

        voxel_x, voxel_y, voxel_z = 1.0, 1.0, 1.0
        try:
            tags = tif.pages[0].tags
            xres = tags.get("XResolution")
            yres = tags.get("YResolution")
            if xres is not None:
                v = xres.value
                voxel_x = v[1] / v[0] if isinstance(v, tuple) and v[0] != 0 else 1.0
            if yres is not None:
                v = yres.value
                voxel_y = v[1] / v[0] if isinstance(v, tuple) and v[0] != 0 else 1.0
        except Exception:
            pass
        try:
            ij = tif.imagej_metadata
            if ij and "spacing" in ij:
                voxel_z = float(ij["spacing"])
        except Exception:
            pass

    # --- Infer (or force) data_kind ----------------------------------------
    if reinterpret_as:
        data_kind, n_t, n_c, Z, Y, X = _force_kind(raw_shape, reinterpret_as)
    else:
        data_kind, n_t, n_c, Z, Y, X = _infer_kind(raw_shape, parent_widget)

    # --- Memory check ------------------------------------------------------
    total_float32_bytes = n_t * n_c * Z * Y * X * 4
    is_lazy = total_float32_bytes > MEMORY_LIMIT_BYTES

    # --- Normalisation bounds ----------------------------------------------
    norm_lo, norm_hi = _compute_norm_bounds(
        path, data_kind, raw_shape, n_c, is_lazy
    )

    is_rgb = data_kind in ("ZYXC", "TZYXC")

    if is_lazy:
        return VolumeStack(
            n_channels=n_c,
            n_timepoints=n_t,
            depth=Z, height=Y, width=X,
            voxel_size_x=voxel_x,
            voxel_size_y=voxel_y,
            voxel_size_z=voxel_z,
            file_path=path,
            is_lazy=True,
            is_rgb=is_rgb,
            data=None,
            norm_lo=norm_lo,
            norm_hi=norm_hi,
            _raw_shape=raw_shape,
            _raw_dtype=np.dtype(raw_dtype),
            _data_kind=data_kind,
        )

    # --- Eager load --------------------------------------------------------
    import tifffile as tf2
    raw = tf2.imread(path)
    data: list = []
    for t in range(n_t):
        frame = []
        for c in range(n_c):
            arr = _extract_channel(raw, t, c, data_kind).astype(np.float32)
            lo, hi = norm_lo[c], norm_hi[c]
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            frame.append(np.clip(arr, 0.0, 1.0))
        data.append(frame)

    return VolumeStack(
        n_channels=n_c,
        n_timepoints=n_t,
        depth=Z, height=Y, width=X,
        voxel_size_x=voxel_x,
        voxel_size_y=voxel_y,
        voxel_size_z=voxel_z,
        file_path=path,
        is_lazy=False,
        is_rgb=is_rgb,
        data=data,
        norm_lo=norm_lo,
        norm_hi=norm_hi,
        _raw_shape=raw_shape,
        _raw_dtype=np.dtype(raw_dtype),
        _data_kind=data_kind,
    )


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

def _infer_kind(
    raw_shape: tuple,
    parent_widget=None,
) -> tuple[str, int, int, int, int, int]:
    """
    Return (data_kind, n_timepoints, n_channels, Z, Y, X).

    Heuristic rules:
      2D (Y, X)           → ZYX (single slice)
      3D (Z, Y, X)        → ZYX
      4D, shape[1] ≤ 4    → ZCYX  (Z-C-Y-X, multi-channel)
      4D, shape[1] > 4    → TZYX  (T-Z-Y-X, time-lapse)
      5D, shape[2] ≤ 4    → TZCYX (T-Z-C-Y-X)
      5D, shape[1] ≤ 4    → TCZYX (T-C-Z-Y-X)

    For ambiguous 4D cases (shape[1] == 4) a dialog is shown.
    """
    ndim = len(raw_shape)

    # --- RGB detection: last dimension == 3 means colour channels (R/G/B).
    # Takes priority over other interpretations.
    if ndim == 4 and raw_shape[-1] == 3:
        Z, Y, X, C = raw_shape
        return "ZYXC", 1, C, Z, Y, X

    if ndim == 5 and raw_shape[-1] == 3:
        T, Z, Y, X, C = raw_shape
        return "TZYXC", T, C, Z, Y, X

    if ndim == 2:
        Y, X = raw_shape
        return "ZYX", 1, 1, 1, Y, X

    if ndim == 3:
        Z, Y, X = raw_shape
        return "ZYX", 1, 1, Z, Y, X

    if ndim == 4:
        # Ambiguous: shape[1] == 4 could be 4 channels or 4 Z-slices
        if raw_shape[1] < 4:
            # Clearly channels (1, 2 or 3 channels)
            Z, C, Y, X = raw_shape
            return "ZCYX", 1, C, Z, Y, X
        elif raw_shape[1] > 4:
            # Clearly Z-stack or time
            # Could be (T, Z, Y, X) or (C, Z, Y, X) with C > 4 (unusual).
            # Treat as time-lapse.
            T, Z, Y, X = raw_shape
            return "TZYX", T, 1, Z, Y, X
        else:
            # shape[1] == 4: ambiguous → ask user
            kind = _ask_4d_kind(raw_shape, parent_widget)
            if kind == "ZCYX":
                Z, C, Y, X = raw_shape
                return "ZCYX", 1, C, Z, Y, X
            else:  # TZYX
                T, Z, Y, X = raw_shape
                return "TZYX", T, 1, Z, Y, X

    if ndim == 5:
        if raw_shape[2] <= 4:
            # (T, Z, C, Y, X)
            T, Z, C, Y, X = raw_shape
            return "TZCYX", T, C, Z, Y, X
        elif raw_shape[1] <= 4:
            # (T, C, Z, Y, X)
            T, C, Z, Y, X = raw_shape
            return "TCZYX", T, C, Z, Y, X
        else:
            # Both shape[1] and shape[2] > 4: ambiguous
            kind = _ask_5d_kind(raw_shape, parent_widget)
            if kind == "TZCYX":
                T, Z, C, Y, X = raw_shape
                return "TZCYX", T, C, Z, Y, X
            else:
                T, C, Z, Y, X = raw_shape
                return "TCZYX", T, C, Z, Y, X

    raise ValueError(
        f"Cannot interpret array of shape {raw_shape} as a volume stack."
    )


def _ask_4d_kind(raw_shape: tuple, parent_widget) -> str:
    """Show a dialog for ambiguous 4D TIFF (shape[1] == 4)."""
    try:
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QPushButton, QHBoxLayout
        dlg = QDialog(parent_widget)
        dlg.setWindowTitle("Ambiguous TIFF dimensions")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel(
            f"TIFF shape {raw_shape} could be interpreted as:\n"
            f"  (Z={raw_shape[0]}, C={raw_shape[1]}, Y={raw_shape[2]}, X={raw_shape[3]})  — multi-channel\n"
            f"  (T={raw_shape[0]}, Z={raw_shape[1]}, Y={raw_shape[2]}, X={raw_shape[3]})  — time-lapse\n"
            "Please choose:"
        ))
        rb_ch = QRadioButton(f"Multi-channel  (C={raw_shape[1]} channels)")
        rb_t  = QRadioButton(f"Time-lapse     (T={raw_shape[0]} timepoints)")
        rb_ch.setChecked(True)
        vl.addWidget(rb_ch)
        vl.addWidget(rb_t)
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(dlg.accept)
        btns.addWidget(ok)
        vl.addLayout(btns)
        dlg.exec()
        return "ZCYX" if rb_ch.isChecked() else "TZYX"
    except Exception:
        # Fallback: treat as channels if shape[1] <= 4
        return "ZCYX" if raw_shape[1] <= 4 else "TZYX"


def _ask_5d_kind(raw_shape: tuple, parent_widget) -> str:
    """Show a dialog for ambiguous 5D TIFF."""
    try:
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QPushButton, QHBoxLayout
        dlg = QDialog(parent_widget)
        dlg.setWindowTitle("Ambiguous TIFF dimensions")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel(
            f"TIFF shape {raw_shape}.\n"
            "Please choose axis order:"
        ))
        rb_tzcyx = QRadioButton(
            f"T={raw_shape[0]}, Z={raw_shape[1]}, C={raw_shape[2]}, Y={raw_shape[3]}, X={raw_shape[4]}"
        )
        rb_tczyx = QRadioButton(
            f"T={raw_shape[0]}, C={raw_shape[1]}, Z={raw_shape[2]}, Y={raw_shape[3]}, X={raw_shape[4]}"
        )
        rb_tzcyx.setChecked(True)
        vl.addWidget(rb_tzcyx)
        vl.addWidget(rb_tczyx)
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(dlg.accept)
        btns.addWidget(ok)
        vl.addLayout(btns)
        dlg.exec()
        return "TZCYX" if rb_tzcyx.isChecked() else "TCZYX"
    except Exception:
        return "TZCYX"


# ---------------------------------------------------------------------------
# Normalisation bounds
# ---------------------------------------------------------------------------

def _compute_norm_bounds(
    path: str,
    data_kind: str,
    raw_shape: tuple,
    n_c: int,
    is_lazy: bool,
) -> tuple[list[float], list[float]]:
    """
    Compute per-channel [1st-pct, 99th-pct] bounds for normalisation.

    For lazy stacks: uses only the first timepoint (fast, consistent enough).
    For eager stacks: uses the full dataset.
    """
    import tifffile

    if is_lazy:
        # Load only first timepoint from pages
        with tifffile.TiffFile(path) as tif:
            channels_raw = _load_first_frame_pages(tif, data_kind, raw_shape)
    else:
        raw = tifffile.imread(path)
        channels_raw = [
            _extract_all_timepoints_channel(raw, c, data_kind) for c in range(n_c)
        ]

    # For RGB data, normalise all colour channels together so that white
    # balance is preserved (same lo/hi for R, G, B).
    if data_kind in ("ZYXC", "TZYXC") and len(channels_raw) == 3:
        all_f = np.concatenate([ch.astype(np.float32).ravel() for ch in channels_raw])
        if channels_raw[0].dtype == np.uint8:
            shared_lo, shared_hi = 0.0, 255.0
        else:
            shared_lo = 0.0
            shared_hi = float(all_f.max())
            if shared_hi < 1e-9:
                shared_hi = 1.0
        return [shared_lo] * 3, [shared_hi] * 3

    norm_lo, norm_hi = [], []
    for ch in channels_raw:
        f = ch.astype(np.float32)
        if ch.dtype == np.uint8:
            # 8-bit data uses the full 0-255 bit-depth range; no clipping.
            lo = 0.0
            hi = 255.0
        else:
            # For 16-bit and higher: normalise by the actual data maximum so
            # that NO signal is clipped before it reaches the GPU texture.
            # Percentile-based clipping causes irreversible saturation for
            # typical sparse fluorescence volumes where signal voxels are
            # <5% of the total — the 99th percentile falls in the background,
            # and every bright structure above it is permanently clamped to 1.0.
            # The renderer's rangeMin/rangeMax sliders provide contrast control.
            lo = 0.0
            hi = float(f.max())
            if hi < 1e-9:
                hi = 1.0
        norm_lo.append(lo)
        norm_hi.append(hi)
    return norm_lo, norm_hi


def _load_first_frame_pages(tif, data_kind: str, raw_shape: tuple) -> list:
    """Load pages for timepoint 0 only, returning raw channel arrays."""
    if data_kind == "TZCYX":
        _T, Z, C, Y, X = raw_shape
        pages = np.stack([tif.pages[i].asarray() for i in range(Z * C)])
        pages = pages.reshape(Z, C, Y, X)
        return [pages[:, c, :, :] for c in range(C)]
    elif data_kind == "TCZYX":
        _T, C, Z, Y, X = raw_shape
        pages = np.stack([tif.pages[i].asarray() for i in range(C * Z)])
        pages = pages.reshape(C, Z, Y, X)
        return [pages[c, :, :, :] for c in range(C)]
    elif data_kind == "TZYX":
        _T, Z, Y, X = raw_shape
        pages = np.stack([tif.pages[i].asarray() for i in range(Z)])
        return [pages]
    elif data_kind == "TZYXC":
        _T, Z, Y, X, C = raw_shape
        pages = np.stack([tif.pages[i].asarray() for i in range(Z)])
        # pages shape: (Z, Y, X, C)
        return [pages[:, :, :, c] for c in range(C)]

    elif data_kind == "ZYXC":
        raw = tif.asarray()  # (Z, Y, X, C)
        return [raw[:, :, :, c] for c in range(raw.shape[-1])]

    else:
        # Non-lazy kinds: load everything
        raw = tif.asarray()
        n_c = raw_shape[1] if data_kind in ("ZCYX",) else 1
        return [_extract_channel(raw, 0, c, data_kind) for c in range(n_c)]


# ---------------------------------------------------------------------------
# Channel extraction helpers
# ---------------------------------------------------------------------------

def _extract_channel(raw: np.ndarray, t: int, c: int, data_kind: str) -> np.ndarray:
    """Return (Z, Y, X) for given timepoint t and channel c."""
    if data_kind == "ZYX":
        return raw                     # (Z, Y, X)
    elif data_kind == "ZCYX":
        return raw[:, c, :, :]         # (Z, C, Y, X)
    elif data_kind == "TZYX":
        return raw[t]                  # (T, Z, Y, X)
    elif data_kind == "TZCYX":
        return raw[t, :, c, :, :]      # (T, Z, C, Y, X)
    elif data_kind == "TCZYX":
        return raw[t, c, :, :, :]      # (T, C, Z, Y, X)
    elif data_kind == "ZYXC":
        return raw[:, :, :, c]         # (Z, Y, X, C) → (Z, Y, X)
    elif data_kind == "TZYXC":
        return raw[t, :, :, :, c]      # (T, Z, Y, X, C) → (Z, Y, X)
    else:
        raise ValueError(f"Unknown data_kind: {data_kind!r}")


def _extract_all_timepoints_channel(raw: np.ndarray, c: int, data_kind: str) -> np.ndarray:
    """Return all timepoints for channel c (for computing global norm bounds)."""
    if data_kind == "ZYX":
        return raw
    elif data_kind == "ZCYX":
        return raw[:, c, :, :]
    elif data_kind == "TZYX":
        return raw                     # single channel, all T
    elif data_kind == "TZCYX":
        return raw[:, :, c, :, :]      # all T and Z for channel c
    elif data_kind == "TCZYX":
        return raw[:, c, :, :, :]      # all T and Z for channel c
    elif data_kind == "ZYXC":
        return raw[:, :, :, c]
    elif data_kind == "TZYXC":
        return raw[:, :, :, :, c]      # all T for colour channel c
    else:
        raise ValueError(f"Unknown data_kind: {data_kind!r}")


# ---------------------------------------------------------------------------
# Forced reinterpretation
# ---------------------------------------------------------------------------

_KIND_LABELS = ("MultiCH", "T", "MultiCHT", "RGB", "RGBT")


def _force_kind(
    raw_shape: tuple,
    kind_label: str,
) -> tuple[str, int, int, int, int, int]:
    """
    Force a specific data-kind, raising ValueError if the shape is incompatible.

    kind_label must be one of: 'MultiCH', 'T', 'MultiCHT', 'RGB', 'RGBT'.
    Returns (data_kind, n_timepoints, n_channels, Z, Y, X).
    """
    ndim = len(raw_shape)

    if kind_label == "RGB":
        if ndim == 4 and raw_shape[-1] == 3:
            Z, Y, X, C = raw_shape
            return "ZYXC", 1, 3, Z, Y, X
        raise ValueError(
            f"Cannot interpret shape {raw_shape} as RGB: "
            "expected 4-D with last dimension = 3."
        )

    if kind_label == "RGBT":
        if ndim == 5 and raw_shape[-1] == 3:
            T, Z, Y, X, C = raw_shape
            return "TZYXC", T, 3, Z, Y, X
        raise ValueError(
            f"Cannot interpret shape {raw_shape} as time-lapse RGB: "
            "expected 5-D with last dimension = 3."
        )

    if kind_label == "MultiCH":
        if ndim == 4 and raw_shape[-1] != 3:
            # (Z, C, Y, X) — axis 1 is channels
            Z, C, Y, X = raw_shape
            return "ZCYX", 1, C, Z, Y, X
        raise ValueError(
            f"Cannot interpret shape {raw_shape} as MultiCH: "
            "expected 4-D (Z, C, Y, X) array where axis-1 is the channel count."
        )

    if kind_label == "T":
        if ndim == 4:
            T, Z, Y, X = raw_shape
            return "TZYX", T, 1, Z, Y, X
        raise ValueError(
            f"Cannot interpret shape {raw_shape} as time-lapse: "
            "expected 4-D (T, Z, Y, X) array."
        )

    if kind_label == "MultiCHT":
        if ndim == 5 and raw_shape[-1] != 3:
            # Default: (T, Z, C, Y, X)
            T, Z, C, Y, X = raw_shape
            return "TZCYX", T, C, Z, Y, X
        raise ValueError(
            f"Cannot interpret shape {raw_shape} as MultiCHT: "
            "expected 5-D (T, Z, C, Y, X) array."
        )

    raise ValueError(f"Unknown kind label: {kind_label!r}. "
                     f"Valid options: {_KIND_LABELS}")


# ---------------------------------------------------------------------------
# RAW loader (unchanged)
# ---------------------------------------------------------------------------

def _load_raw(
    file_path: str,
    shape: tuple[int, int, int] = (1, 1, 1),
    dtype: type = np.uint8,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Volume:
    """Read a headerless binary raw volume."""
    data = np.fromfile(file_path, dtype=dtype)
    if data.size != shape[0] * shape[1] * shape[2]:
        raise ValueError(
            f"RAW file size ({data.size} elements) does not match "
            f"requested shape {shape}."
        )
    data = data.reshape(shape).astype(np.float32)
    return Volume(
        data=data,
        voxel_size_x=float(voxel_size[0]),
        voxel_size_y=float(voxel_size[1]),
        voxel_size_z=float(voxel_size[2]),
        file_path=file_path,
    )


def _volume_to_stack(vol: Volume) -> VolumeStack:
    """Wrap a single Volume in a VolumeStack (n_channels=1, n_timepoints=1)."""
    return VolumeStack(
        n_channels=1,
        n_timepoints=1,
        depth=vol.depth,
        height=vol.height,
        width=vol.width,
        voxel_size_x=vol.voxel_size_x,
        voxel_size_y=vol.voxel_size_y,
        voxel_size_z=vol.voxel_size_z,
        file_path=vol.file_path,
        is_lazy=False,
        data=[[vol.data]],
        norm_lo=[float(vol.data.min())],
        norm_hi=[float(vol.data.max())],
        _data_kind="ZYX",
    )
