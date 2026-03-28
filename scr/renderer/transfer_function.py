"""Transfer function — 1-D colour LUT, analogous to TransferFunction1D.java."""

from __future__ import annotations
import numpy as np
from typing import Optional


class TransferFunction1D:
    """
    A 1-D transfer function mapping normalised intensity [0, 1] to RGBA [0, 1].

    Internally stored as a (256, 4) float32 array that is uploaded once to
    a 1-D GL texture.  The shader samples it with a normalised + gamma-
    corrected intensity value.
    """

    LUT_SIZE: int = 256

    def __init__(self, lut: Optional[np.ndarray] = None) -> None:
        """
        Parameters
        ----------
        lut : np.ndarray, optional
            Shape (N, 4) float32 array of RGBA values.  If N != 256 it will
            be resampled to 256 entries.  If None, defaults to a grey ramp.
        """
        if lut is None:
            self._lut = _make_grey()
        else:
            self._lut = _resample(lut, self.LUT_SIZE)
        self._dirty: bool = True  # needs GL upload

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lut(self) -> np.ndarray:
        """(256, 4) float32 RGBA array."""
        return self._lut

    @lut.setter
    def lut(self, value: np.ndarray) -> None:
        self._lut = _resample(value, self.LUT_SIZE)
        self._dirty = True

    @property
    def dirty(self) -> bool:
        """True if the LUT has changed since the last GL upload."""
        return self._dirty

    def mark_clean(self) -> None:
        """Call after uploading to GL to clear the dirty flag."""
        self._dirty = False

    # ------------------------------------------------------------------
    # Factory methods (mirrors TransferFunctions.java presets)
    # ------------------------------------------------------------------

    @classmethod
    def grey(cls) -> "TransferFunction1D":
        """Linear grey ramp: black → white."""
        return cls(_make_grey())

    @classmethod
    def hot(cls) -> "TransferFunction1D":
        """Hot (black → red → yellow → white)."""
        return cls(_make_hot())

    @classmethod
    def cool_warm(cls) -> "TransferFunction1D":
        """Cool-warm diverging (blue → white → red)."""
        return cls(_make_cool_warm())

    @classmethod
    def rainbow(cls) -> "TransferFunction1D":
        """Rainbow / jet (blue → cyan → green → yellow → red)."""
        return cls(_make_rainbow())

    @classmethod
    def green(cls) -> "TransferFunction1D":
        """Black → green ramp."""
        return cls(_make_single_channel(1))

    @classmethod
    def red(cls) -> "TransferFunction1D":
        """Black → red ramp."""
        return cls(_make_single_channel(0))

    @classmethod
    def blue(cls) -> "TransferFunction1D":
        """Black → blue ramp."""
        return cls(_make_single_channel(2))

    @classmethod
    def cyan(cls) -> "TransferFunction1D":
        """Black → cyan ramp (equal green + blue)."""
        return cls(_make_dual_channel(1, 2))

    @classmethod
    def yellow(cls) -> "TransferFunction1D":
        """Black → yellow ramp (equal red + green)."""
        return cls(_make_dual_channel(0, 1))

    @classmethod
    def magenta(cls) -> "TransferFunction1D":
        """Black → magenta ramp (equal red + blue)."""
        return cls(_make_dual_channel(0, 2))

    @classmethod
    def fire(cls) -> "TransferFunction1D":
        """Fire: black → dark red → orange → bright yellow → white."""
        return cls(_make_fire())

    @classmethod
    def ice(cls) -> "TransferFunction1D":
        """Ice: black → dark blue → cyan → white."""
        return cls(_make_ice())

    @classmethod
    def from_color(cls, r: float, g: float, b: float) -> "TransferFunction1D":
        """Create a TF ramp from black to the given RGB color (alpha = intensity)."""
        n = cls.LUT_SIZE
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        lut = np.zeros((n, 4), dtype=np.float32)
        lut[:, 0] = float(r) * t
        lut[:, 1] = float(g) * t
        lut[:, 2] = float(b) * t
        lut[:, 3] = t
        return cls(lut)

    @classmethod
    def all_presets(cls) -> list[tuple[str, "TransferFunction1D"]]:
        """Return a list of (name, TransferFunction1D) for all presets."""
        return [
            ("Grey",      cls.grey()),
            ("Green",     cls.green()),
            ("Red",       cls.red()),
            ("Blue",      cls.blue()),
            ("Cyan",      cls.cyan()),
            ("Yellow",    cls.yellow()),
            ("Magenta",   cls.magenta()),
            ("Hot",       cls.hot()),
            ("Fire",      cls.fire()),
            ("Ice",       cls.ice()),
            ("Cool-Warm", cls.cool_warm()),
            ("Rainbow",   cls.rainbow()),
        ]

    def __repr__(self) -> str:
        return f"TransferFunction1D(lut.shape={self._lut.shape})"


# ---------------------------------------------------------------------------
# Internal LUT builders
# ---------------------------------------------------------------------------

def _make_lut(n: int = 256) -> np.ndarray:
    """Allocate an (N, 4) float32 RGBA array."""
    return np.zeros((n, 4), dtype=np.float32)


def _resample(lut: np.ndarray, n: int) -> np.ndarray:
    """Resample *lut* (shape M×4) to shape N×4 via linear interpolation."""
    lut = np.asarray(lut, dtype=np.float32)
    if lut.ndim != 2 or lut.shape[1] != 4:
        raise ValueError(f"LUT must be shape (M, 4), got {lut.shape}")
    if lut.shape[0] == n:
        return lut.copy()
    m = lut.shape[0]
    x_old = np.linspace(0.0, 1.0, m)
    x_new = np.linspace(0.0, 1.0, n)
    out = np.stack(
        [np.interp(x_new, x_old, lut[:, c]) for c in range(4)],
        axis=1,
    ).astype(np.float32)
    return out


def _t(n: int = 256) -> np.ndarray:
    """Normalised position array [0, 1] of length n."""
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def _make_grey(n: int = 256) -> np.ndarray:
    t = _t(n)
    lut = _make_lut(n)
    lut[:, 0] = t
    lut[:, 1] = t
    lut[:, 2] = t
    lut[:, 3] = t
    return lut


def _make_hot(n: int = 256) -> np.ndarray:
    """Black → red → yellow → white, alpha ramps linearly."""
    t = _t(n)
    lut = _make_lut(n)
    # Red: 0 → 1 in first 1/3
    lut[:, 0] = np.clip(t * 3.0, 0.0, 1.0)
    # Green: 0 → 1 in second 1/3
    lut[:, 1] = np.clip(t * 3.0 - 1.0, 0.0, 1.0)
    # Blue: 0 → 1 in last 1/3
    lut[:, 2] = np.clip(t * 3.0 - 2.0, 0.0, 1.0)
    lut[:, 3] = t
    return lut


def _make_cool_warm(n: int = 256) -> np.ndarray:
    """Blue → white → red diverging map."""
    t = _t(n)
    lut = _make_lut(n)
    # Red increases from 0 to 1
    lut[:, 0] = t
    # Green peaks at 0.5 (white centre)
    lut[:, 1] = 1.0 - np.abs(2.0 * t - 1.0)
    # Blue decreases from 1 to 0
    lut[:, 2] = 1.0 - t
    lut[:, 3] = t
    return lut


def _make_rainbow(n: int = 256) -> np.ndarray:
    """Jet-style rainbow."""
    t = _t(n)
    lut = _make_lut(n)
    lut[:, 0] = np.clip(np.abs(t * 4.0 - 3.0) - 0.5, 0.0, 1.0)  # red
    lut[:, 1] = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0.0, 1.0)  # green
    lut[:, 2] = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0.0, 1.0)  # blue
    lut[:, 3] = t
    return lut


def _make_single_channel(channel: int, n: int = 256) -> np.ndarray:
    """Black → single colour ramp (channel 0=R, 1=G, 2=B)."""
    t = _t(n)
    lut = _make_lut(n)
    lut[:, channel] = t
    lut[:, 3] = t
    return lut


def _make_dual_channel(ch1: int, ch2: int, n: int = 256) -> np.ndarray:
    """Black → dual colour ramp (two channels equal, e.g. cyan = G+B)."""
    t = _t(n)
    lut = _make_lut(n)
    lut[:, ch1] = t
    lut[:, ch2] = t
    lut[:, 3]   = t
    return lut


def _make_fire(n: int = 256) -> np.ndarray:
    """Fire: black → dark red → orange → yellow → white."""
    t = _t(n)
    lut = _make_lut(n)
    # Red: full from 0.2 onward
    lut[:, 0] = np.clip(t / 0.4, 0.0, 1.0)
    # Green: starts at 0.4
    lut[:, 1] = np.clip((t - 0.4) / 0.4, 0.0, 1.0)
    # Blue: starts at 0.8 (white tip)
    lut[:, 2] = np.clip((t - 0.8) / 0.2, 0.0, 1.0)
    lut[:, 3] = t
    return lut


def _make_ice(n: int = 256) -> np.ndarray:
    """Ice: black → dark blue → cyan → white."""
    t = _t(n)
    lut = _make_lut(n)
    # Blue: full from the start
    lut[:, 2] = np.clip(t / 0.5, 0.0, 1.0)
    # Green (→ cyan): starts at 0.4
    lut[:, 1] = np.clip((t - 0.4) / 0.4, 0.0, 1.0)
    # Red (→ white): starts at 0.8
    lut[:, 0] = np.clip((t - 0.8) / 0.2, 0.0, 1.0)
    lut[:, 3] = t
    return lut
