"""Rendering algorithm enum — mirrors clearvolume/renderer/RenderAlgorithm.java."""

from enum import IntEnum


class RenderAlgorithm(IntEnum):
    """Supported volume rendering algorithms."""

    MaxProjection = 0
    """Maximum Intensity Projection (MIP).  Each pixel shows the highest
    voxel intensity encountered along the corresponding ray."""

    IsoSurface = 1
    """Iso-surface rendering.  Finds the first voxel crossing a user-defined
    threshold and shades it with Phong lighting."""

    def next(self) -> "RenderAlgorithm":
        """Cycle to the next algorithm."""
        members = list(RenderAlgorithm)
        return members[(self.value + 1) % len(members)]

    def label(self) -> str:
        """Human-readable label for GUI display."""
        return {
            RenderAlgorithm.MaxProjection: "Max Projection",
            RenderAlgorithm.IsoSurface: "Iso-Surface",
        }[self]
