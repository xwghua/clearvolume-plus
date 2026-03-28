"""Tests for volume/loader.py, volume/volume.py, and VolumeStack."""

import os
import pytest
import numpy as np

from scr.volume.volume import Volume, VolumeStack
from scr.volume.loader import load, load_stack, _infer_kind


# Path to the test TIFFs relative to repo root
_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "testdata")
TESTDATA_SINGLE = os.path.join(_TESTDATA_DIR, "FLFM_stack_00022_mito_z1_52.tif")
TESTDATA_MULTICHANNEL = os.path.join(_TESTDATA_DIR, "FLFM_stack_merged_z1_52.tif")
TESTDATA_TIMELAPSE = os.path.join(_TESTDATA_DIR, "FLFM_stack_mg_merged_z1_64.tif")


# ---------------------------------------------------------------------------
# Volume data container
# ---------------------------------------------------------------------------

class TestVolume:
    def test_shape_properties(self):
        data = np.zeros((10, 20, 30), dtype=np.float32)
        vol = Volume(data=data)
        assert vol.depth == 10
        assert vol.height == 20
        assert vol.width == 30
        assert vol.shape == (10, 20, 30)
        assert vol.num_voxels == 6000

    def test_aspect_ratio_max_one(self):
        data = np.zeros((10, 50, 100), dtype=np.float32)
        vol = Volume(data=data, voxel_size_x=1.0, voxel_size_y=1.0, voxel_size_z=1.0)
        asp = vol.aspect_ratio
        assert float(asp.max()) == pytest.approx(1.0)
        assert asp[0] == pytest.approx(1.0)   # X is largest (100)
        assert asp[1] == pytest.approx(0.5)   # Y = 50
        assert asp[2] == pytest.approx(0.1)   # Z = 10

    def test_normalise_range(self):
        data = np.array([[[0.0, 100.0], [50.0, 200.0]]], dtype=np.float32)
        vol = Volume(data=data).normalise()
        assert float(vol.data.min()) == pytest.approx(0.0)
        assert float(vol.data.max()) == pytest.approx(1.0)

    def test_normalise_uniform_volume(self):
        data = np.ones((4, 4, 4), dtype=np.float32) * 42.0
        vol = Volume(data=data).normalise()
        assert vol.data.min() == vol.data.max()


# ---------------------------------------------------------------------------
# _infer_kind shape detection
# ---------------------------------------------------------------------------

class TestInferKind:
    def test_2d(self):
        kind, n_t, n_c, Z, Y, X = _infer_kind((64, 128))
        assert kind == "ZYX" and n_t == 1 and n_c == 1

    def test_3d(self):
        kind, n_t, n_c, Z, Y, X = _infer_kind((10, 20, 30))
        assert kind == "ZYX" and n_t == 1 and n_c == 1
        assert Z == 10 and Y == 20 and X == 30

    def test_4d_multichannel(self):
        # (Z, C, Y, X) with small C
        kind, n_t, n_c, Z, Y, X = _infer_kind((52, 2, 512, 512))
        assert kind == "ZCYX" and n_t == 1 and n_c == 2
        assert Z == 52 and Y == 512 and X == 512

    def test_4d_timelapse(self):
        # (T, Z, Y, X) with large Z
        kind, n_t, n_c, Z, Y, X = _infer_kind((100, 64, 400, 400))
        assert kind == "TZYX" and n_c == 1 and n_t == 100
        assert Z == 64

    def test_5d_tzcyx(self):
        kind, n_t, n_c, Z, Y, X = _infer_kind((100, 64, 2, 400, 400))
        assert kind == "TZCYX" and n_t == 100 and n_c == 2
        assert Z == 64 and Y == 400 and X == 400


# ---------------------------------------------------------------------------
# VolumeStack
# ---------------------------------------------------------------------------

class TestVolumeStack:
    def _make_stack(self, n_t=2, n_c=2, Z=4, Y=8, X=8):
        data = [
            [np.random.rand(Z, Y, X).astype(np.float32) for c in range(n_c)]
            for t in range(n_t)
        ]
        return VolumeStack(
            n_channels=n_c, n_timepoints=n_t,
            depth=Z, height=Y, width=X,
            is_lazy=False,
            data=data,
            norm_lo=[0.0] * n_c,
            norm_hi=[1.0] * n_c,
            _data_kind="TZCYX",
        )

    def test_get_frame(self):
        stack = self._make_stack()
        frame = stack.get_frame(0)
        assert len(frame) == 2
        assert frame[0].shape == (4, 8, 8)

    def test_aspect_ratio_max_one(self):
        stack = self._make_stack()
        asp = stack.aspect_ratio
        assert float(asp.max()) == pytest.approx(1.0)

    def test_to_volume(self):
        stack = self._make_stack()
        vol = stack.to_volume(t=0, c=1)
        assert isinstance(vol, Volume)
        assert vol.depth == 4 and vol.height == 8 and vol.width == 8

    def test_missing_data_raises(self):
        stack = VolumeStack(
            n_channels=1, n_timepoints=1,
            depth=4, height=8, width=8,
            is_lazy=False, data=None,
            norm_lo=[0.0], norm_hi=[1.0],
        )
        with pytest.raises(RuntimeError):
            stack.get_frame(0)


# ---------------------------------------------------------------------------
# TIFF loading — single channel (requires test file)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isfile(TESTDATA_SINGLE), reason="test TIFF not found")
class TestTIFFLoaderSingle:
    def test_load_returns_volume(self):
        vol = load(TESTDATA_SINGLE)
        assert isinstance(vol, Volume)
        assert vol.depth == 52
        assert vol.height == 512
        assert vol.width == 512

    def test_dtype_float32(self):
        vol = load(TESTDATA_SINGLE)
        assert vol.data.dtype == np.float32

    def test_range_normalised(self):
        vol = load(TESTDATA_SINGLE)
        assert float(vol.data.min()) >= 0.0
        assert float(vol.data.max()) <= 1.0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path/volume.tif")

    def test_unsupported_format(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy")
            name = f.name
        try:
            with pytest.raises(ValueError):
                load(name)
        finally:
            os.unlink(name)


# ---------------------------------------------------------------------------
# TIFF loading — multi-channel
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isfile(TESTDATA_MULTICHANNEL), reason="multi-ch TIFF not found")
class TestTIFFLoaderMultiChannel:
    def test_stack_shape(self):
        stack = load_stack(TESTDATA_MULTICHANNEL)
        assert stack.n_channels == 2
        assert stack.n_timepoints == 1
        assert stack.depth == 52
        assert stack.height == 512
        assert stack.width == 512

    def test_not_lazy(self):
        stack = load_stack(TESTDATA_MULTICHANNEL)
        assert not stack.is_lazy

    def test_channels_normalised(self):
        stack = load_stack(TESTDATA_MULTICHANNEL)
        for c in range(stack.n_channels):
            ch = stack.get_frame(0)[c]
            assert ch.dtype == np.float32
            assert float(ch.min()) >= 0.0
            assert float(ch.max()) <= 1.0


# ---------------------------------------------------------------------------
# TIFF loading — time-lapse multi-channel (lazy)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isfile(TESTDATA_TIMELAPSE), reason="time-lapse TIFF not found")
class TestTIFFLoaderTimeLapse:
    def test_stack_shape(self):
        stack = load_stack(TESTDATA_TIMELAPSE)
        assert stack.n_channels == 2
        assert stack.n_timepoints == 100
        assert stack.depth == 64
        assert stack.height == 400
        assert stack.width == 400

    def test_is_lazy(self):
        stack = load_stack(TESTDATA_TIMELAPSE)
        assert stack.is_lazy

    def test_lazy_frame_load(self):
        stack = load_stack(TESTDATA_TIMELAPSE)
        frame = stack.get_frame(0)
        assert len(frame) == 2
        assert frame[0].shape == (64, 400, 400)
        assert frame[0].dtype == np.float32

    def test_different_timepoints(self):
        stack = load_stack(TESTDATA_TIMELAPSE)
        f0 = stack.get_frame(0)
        f99 = stack.get_frame(99)
        # Different timepoints should have data (not guarantee different values
        # for all files, but they should load without error)
        assert f0[0].shape == f99[0].shape
