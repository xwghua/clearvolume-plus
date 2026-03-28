"""Tests for renderer/transfer_function.py."""

import pytest
import numpy as np

from scr.renderer.transfer_function import (
    TransferFunction1D, _resample,
)


class TestTransferFunction1D:
    def test_default_grey(self):
        tf = TransferFunction1D()
        assert tf.lut.shape == (256, 4)
        assert tf.lut.dtype == np.float32
        # Grey: R == G == B at any point
        np.testing.assert_allclose(tf.lut[:, 0], tf.lut[:, 1], atol=1e-5)
        np.testing.assert_allclose(tf.lut[:, 0], tf.lut[:, 2], atol=1e-5)

    def test_dirty_flag(self):
        tf = TransferFunction1D()
        assert tf.dirty
        tf.mark_clean()
        assert not tf.dirty
        tf.lut = np.zeros((256, 4), dtype=np.float32)
        assert tf.dirty

    def test_lut_range(self):
        for preset in ["grey", "hot", "cool_warm", "rainbow", "green", "red", "blue"]:
            tf = getattr(TransferFunction1D, preset)()
            assert tf.lut.min() >= 0.0, f"{preset} lut has value < 0"
            assert tf.lut.max() <= 1.0, f"{preset} lut has value > 1"

    def test_all_presets_names(self):
        presets = TransferFunction1D.all_presets()
        names = [n for n, _ in presets]
        assert "Grey" in names
        assert "Hot" in names
        assert "Rainbow" in names
        assert len(presets) >= 7  # may grow as new presets are added

    def test_resample_identity(self):
        lut = np.random.rand(256, 4).astype(np.float32)
        out = _resample(lut, 256)
        np.testing.assert_allclose(out, lut, atol=1e-5)

    def test_resample_upscale(self):
        lut = np.zeros((4, 4), dtype=np.float32)
        lut[:, 0] = [0.0, 0.33, 0.66, 1.0]
        out = _resample(lut, 256)
        assert out.shape == (256, 4)
        assert out[0, 0] == pytest.approx(0.0, abs=0.01)
        assert out[-1, 0] == pytest.approx(1.0, abs=0.01)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            _resample(np.zeros((256, 3)), 256)  # need 4 channels


class TestRenderAlgorithm:
    def test_enum_values(self):
        from scr.renderer.render_algorithm import RenderAlgorithm
        assert int(RenderAlgorithm.MaxProjection) == 0
        assert int(RenderAlgorithm.IsoSurface) == 1

    def test_cycle(self):
        from scr.renderer.render_algorithm import RenderAlgorithm
        assert RenderAlgorithm.MaxProjection.next() == RenderAlgorithm.IsoSurface
        assert RenderAlgorithm.IsoSurface.next() == RenderAlgorithm.MaxProjection


class TestMathUtils:
    def test_quaternion_identity(self):
        from scr.utils.math_utils import quaternion_identity
        q = quaternion_identity()
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_quaternion_multiply_identity(self):
        from scr.utils.math_utils import (
            quaternion_identity, quaternion_multiply, quaternion_normalise
        )
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        q = quaternion_normalise(q)
        result = quaternion_multiply(quaternion_identity(), q)
        np.testing.assert_allclose(result, q, atol=1e-5)

    def test_rotation_matrix_identity(self):
        from scr.utils.math_utils import (
            quaternion_identity, quaternion_to_matrix4
        )
        m = quaternion_to_matrix4(quaternion_identity())
        np.testing.assert_allclose(m, np.eye(4), atol=1e-5)

    def test_arcball_unit_sphere(self):
        from scr.utils.math_utils import arcball_vector
        v = arcball_vector(0.0, 0.0)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_perspective_matrix(self):
        from scr.utils.math_utils import perspective_matrix
        proj = perspective_matrix(90.0, 1.0, 0.1, 100.0)
        assert proj.shape == (4, 4)
        assert abs(proj[0, 0]) == pytest.approx(abs(proj[1, 1]), rel=0.01)
