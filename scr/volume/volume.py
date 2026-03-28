"""Volume data containers — Volume (single frame) and VolumeStack (multi-channel/time-lapse)."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Volume:
    """
    Container for a 3D volumetric dataset.

    Attributes
    ----------
    data : np.ndarray
        3-D float32 array with shape (depth, height, width) == (Z, Y, X).
        Values are normalised to [0, 1] when loaded via VolumeLoader.
    voxel_size_x : float
        Physical size of one voxel along X in arbitrary units (e.g. µm).
    voxel_size_y : float
        Physical size of one voxel along Y.
    voxel_size_z : float
        Physical size of one voxel along Z.
    channel_id : int
        Channel index (0-based).
    channel_name : str
        Human-readable channel label.
    time_index : int
        Time-point index (0-based).
    file_path : str
        Source file path, for display purposes.
    """

    data: np.ndarray
    voxel_size_x: float = 1.0
    voxel_size_y: float = 1.0
    voxel_size_z: float = 1.0
    channel_id: int = 0
    channel_name: str = ""
    time_index: int = 0
    file_path: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of slices (Z axis)."""
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        """Number of rows (Y axis)."""
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Number of columns (X axis)."""
        return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        """(depth, height, width) == (Z, Y, X)."""
        return self.data.shape  # type: ignore[return-value]

    @property
    def num_voxels(self) -> int:
        """Total number of voxels."""
        return self.depth * self.height * self.width

    @property
    def aspect_ratio(self) -> np.ndarray:
        """
        Physical aspect ratio as a float32 array [rx, ry, rz] normalised so
        that the largest dimension is 1.0.  Used by the shader to scale the
        bounding box.
        """
        sizes = np.array(
            [self.width * self.voxel_size_x,
             self.height * self.voxel_size_y,
             self.depth * self.voxel_size_z],
            dtype=np.float32,
        )
        return sizes / sizes.max()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def normalise(self) -> "Volume":
        """Return a copy with data rescaled to [0, 1] float32."""
        d = self.data.astype(np.float32)
        lo, hi = d.min(), d.max()
        if hi > lo:
            d = (d - lo) / (hi - lo)
        return Volume(
            data=d,
            voxel_size_x=self.voxel_size_x,
            voxel_size_y=self.voxel_size_y,
            voxel_size_z=self.voxel_size_z,
            channel_id=self.channel_id,
            channel_name=self.channel_name,
            time_index=self.time_index,
            file_path=self.file_path,
        )

    def __repr__(self) -> str:
        return (
            f"Volume(shape={self.shape}, dtype={self.data.dtype}, "
            f"voxel=({self.voxel_size_x}, {self.voxel_size_y}, {self.voxel_size_z}), "
            f"ch={self.channel_id}, t={self.time_index})"
        )


# ---------------------------------------------------------------------------
# VolumeStack — multi-channel and/or time-lapse container
# ---------------------------------------------------------------------------

@dataclass
class VolumeStack:
    """
    Multi-channel and/or time-lapse volume container.

    Supports:
    - Single-channel, single-timepoint  (equivalent to Volume)
    - Multi-channel:  n_channels > 1,  n_timepoints = 1
    - Time-lapse:     n_channels = 1,  n_timepoints > 1
    - Multi-channel time-lapse: both > 1

    For stacks whose uncompressed float32 size exceeds MEMORY_LIMIT_BYTES
    (set in loader.py), is_lazy=True and data is loaded per-frame on demand.

    Layout of eagerly-loaded data
    ------------------------------
    data[t][c]  →  np.ndarray  shape (Z, Y, X), dtype float32, values in [0, 1]

    Layout for lazy loading
    -----------------------
    data is None.  get_frame(t) opens the TIFF each time and reads only the
    pages for that timepoint (uses no extra RAM beyond one frame).

    data_kind values
    ----------------
    "ZYX"    : (Z, Y, X)               single channel, 1 timepoint
    "ZCYX"   : (Z, C, Y, X)            multi-channel,  1 timepoint
    "TZYX"   : (T, Z, Y, X)            single channel, T timepoints
    "TZCYX"  : (T, Z, C, Y, X)         multi-channel,  T timepoints
    "TCZYX"  : (T, C, Z, Y, X)         multi-channel,  T timepoints (C first)
    "ZYXC"   : (Z, Y, X, C)            RGB, 1 timepoint  (C=3)
    "TZYXC"  : (T, Z, Y, X, C)         RGB, T timepoints (C=3)
    """

    n_channels: int
    n_timepoints: int
    depth: int
    height: int
    width: int
    voxel_size_x: float = 1.0
    voxel_size_y: float = 1.0
    voxel_size_z: float = 1.0
    file_path: str = ""
    is_lazy: bool = False

    # Eager: data[t][c] = (Z, Y, X) float32, already normalised to [0, 1]
    data: Optional[List] = None

    # Per-channel normalisation bounds (1st/99th percentile of raw values).
    # Used to normalise lazy-loaded frames to [0, 1].
    norm_lo: List[float] = field(default_factory=list)
    norm_hi: List[float] = field(default_factory=list)

    # True when data is a true-color RGB volume (C=3 colour channels,
    # not independent fluorescence channels).
    is_rgb: bool = False

    # For lazy page-level loading
    _raw_shape: tuple = field(default_factory=tuple)
    _raw_dtype: object = field(default=None)
    _data_kind: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def aspect_ratio(self) -> np.ndarray:
        """Physical aspect ratio [rx, ry, rz] normalised to max=1.0."""
        sizes = np.array(
            [self.width  * self.voxel_size_x,
             self.height * self.voxel_size_y,
             self.depth  * self.voxel_size_z],
            dtype=np.float32,
        )
        return sizes / sizes.max()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_frame(self, t: int) -> List[np.ndarray]:
        """
        Return a list of n_channels arrays for timepoint *t*.

        Each array has shape (Z, Y, X), dtype float32, values in [0, 1].
        """
        if not self.is_lazy:
            if self.data is None:
                raise RuntimeError("VolumeStack has no data (is_lazy=False but data is None).")
            return self.data[t]
        return _lazy_load_frame(
            self.file_path, t,
            self._data_kind, self._raw_shape,
            self.norm_lo, self.norm_hi,
        )

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def to_volume(self, t: int = 0, c: int = 0) -> Volume:
        """Extract a single Volume from the stack (for axis overlay / aspect ratio)."""
        frame = self.get_frame(t)
        return Volume(
            data=frame[c],
            voxel_size_x=self.voxel_size_x,
            voxel_size_y=self.voxel_size_y,
            voxel_size_z=self.voxel_size_z,
            channel_id=c,
            time_index=t,
            file_path=self.file_path,
        )

    def __repr__(self) -> str:
        return (
            f"VolumeStack(n_ch={self.n_channels}, n_t={self.n_timepoints}, "
            f"shape=({self.depth},{self.height},{self.width}), "
            f"lazy={self.is_lazy}, kind={self._data_kind!r})"
        )


# ---------------------------------------------------------------------------
# Lazy page-level TIFF loader (no full-file load)
# ---------------------------------------------------------------------------

def _lazy_load_frame(
    file_path: str,
    t: int,
    data_kind: str,
    raw_shape: tuple,
    norm_lo: List[float],
    norm_hi: List[float],
) -> List[np.ndarray]:
    """
    Open the TIFF file, read only the pages for timepoint *t*, and return
    a list of normalised (Z, Y, X) float32 arrays (one per channel).

    Pages are assumed to be in the natural row-major order that tifffile
    writes for standard scientific TIFFs (outer axes first).
    """
    import tifffile

    with tifffile.TiffFile(file_path) as tif:
        if data_kind == "TZCYX":
            _T, Z, C, Y, X = raw_shape
            pages_per_t = Z * C
            base = t * pages_per_t
            pages = np.stack(
                [tif.pages[base + i].asarray() for i in range(pages_per_t)]
            )
            pages = pages.reshape(Z, C, Y, X)
            channels_raw = [pages[:, c, :, :] for c in range(C)]

        elif data_kind == "TCZYX":
            _T, C, Z, Y, X = raw_shape
            pages_per_t = C * Z
            base = t * pages_per_t
            pages = np.stack(
                [tif.pages[base + i].asarray() for i in range(pages_per_t)]
            )
            pages = pages.reshape(C, Z, Y, X)
            channels_raw = [pages[c, :, :, :] for c in range(C)]

        elif data_kind == "TZYX":
            _T, Z, Y, X = raw_shape
            base = t * Z
            pages = np.stack(
                [tif.pages[base + i].asarray() for i in range(Z)]
            )
            channels_raw = [pages]  # single channel

        elif data_kind == "TZYXC":
            # Time-lapse RGB: each page is an RGB image (Y, X, 3)
            _T, Z, Y, X, C = raw_shape
            base = t * Z
            pages = np.stack(
                [tif.pages[base + i].asarray() for i in range(Z)]
            )
            # pages shape: (Z, Y, X, C)
            channels_raw = [pages[:, :, :, c] for c in range(C)]

        elif data_kind == "ZYXC":
            # Single-timepoint RGB (unlikely to be lazy, but handle it)
            raw = tif.asarray()
            channels_raw = [raw[:, :, :, c] for c in range(raw.shape[-1])]

        else:
            raise ValueError(
                f"Lazy loading not implemented for data_kind={data_kind!r}. "
                "Load the full stack (set a higher MEMORY_LIMIT_BYTES)."
            )

    result: List[np.ndarray] = []
    for c, ch_raw in enumerate(channels_raw):
        f = ch_raw.astype(np.float32)
        lo, hi = norm_lo[c], norm_hi[c]
        if hi > lo:
            f = (f - lo) / (hi - lo)
        result.append(np.clip(f, 0.0, 1.0))
    return result
