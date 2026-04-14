"""
VolumeRenderer — manages all OpenGL state for volume ray-casting.

Supports:
  - Single-channel single-timepoint volumes
  - Multi-channel volumes (up to MAX_CHANNELS, additive blending)
  - Time-lapse volumes (per-frame GL texture re-upload)
  - Multi-channel time-lapse (both)

Designed to live inside a QOpenGLWidget subclass:
  - initGL()   called once from initializeGL()
  - render()   called each frame from paintGL()
  - cleanup()  called when the widget is destroyed
"""

from __future__ import annotations
import os
import math
import numpy as np
import ctypes

from OpenGL.GL import (
    glGetError, GL_NO_ERROR,
    glGenTextures, glBindTexture, glTexImage2D, glTexImage3D,
    glTexParameteri, glPixelStorei,
    GL_TEXTURE_2D, GL_TEXTURE_3D,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_CLAMP_TO_EDGE, GL_LINEAR,
    GL_RGBA, GL_RGBA32F, GL_RED, GL_R32F, GL_FLOAT,
    GL_UNPACK_ALIGNMENT,
    glGenVertexArrays, glBindVertexArray,
    glGenBuffers, glBindBuffer, glBufferData,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    glEnableVertexAttribArray, glVertexAttribPointer,
    GL_FLOAT as GL_FLOAT_TYPE,
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv,
    glGetProgramInfoLog, GL_LINK_STATUS,
    glUseProgram, glGetUniformLocation, glDeleteProgram,
    glUniform1i, glUniform1f, glUniform3f, glUniformMatrix4fv,
    glActiveTexture, GL_TEXTURE0,
    glDrawArrays, GL_TRIANGLES,
    glDeleteTextures, glDeleteBuffers, glDeleteVertexArrays,
    glViewport, glClear, glClearColor, GL_COLOR_BUFFER_BIT,
    glDisable, glEnable, GL_DEPTH_TEST, GL_BLEND,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
    glBlendEquation, GL_FUNC_ADD, GL_MAX,
    glGetIntegerv, GL_MAX_3D_TEXTURE_SIZE,
)

from ..volume.volume import Volume, VolumeStack
from ..renderer.transfer_function import TransferFunction1D
from ..renderer.render_algorithm import RenderAlgorithm
from ..utils.math_utils import (
    quaternion_to_matrix4, perspective_matrix, look_at_matrix,
)


_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
MAX_CHANNELS = 4  # maximum number of simultaneously-loaded channels


class VolumeRenderer:
    """
    Encapsulates all OpenGL resources and the render loop for a VolumeStack.

    Usage (inside a QOpenGLWidget subclass)::

        def initializeGL(self):
            self.renderer = VolumeRenderer()
            self.renderer.initGL()

        def paintGL(self):
            self.renderer.render(self.width(), self.height())

        def closeEvent(self, event):
            self.renderer.cleanup()
    """

    def __init__(self) -> None:
        # --- Scene state --------------------------------------------------
        self._stack: VolumeStack | None = None
        self._volume: Volume | None = None          # representative volume (ch 0, t 0)
        self._current_timepoint: int = 0

        self._algorithm = RenderAlgorithm.MaxProjection

        # --- Per-channel state --------------------------------------------
        # Default TF colors: green / magenta / cyan / yellow
        self._channel_tfs: list[TransferFunction1D] = [
            TransferFunction1D.green(),
            TransferFunction1D.magenta(),
            TransferFunction1D.cyan(),
            TransferFunction1D.yellow(),
        ]
        self._channel_visible: list[bool] = [True] * MAX_CHANNELS
        self._tf_dirty: list[bool] = [True] * MAX_CHANNELS

        # Per-channel intensity / contrast parameters (index = channel).
        # For single-channel, only index 0 is used.
        # Properties 'brightness', 'gamma', 'range_min', 'range_max',
        # 'iso_value' alias index 0 for backward compatibility.
        self._ch_brightness: list[float] = [1.0] * MAX_CHANNELS
        self._ch_gamma:      list[float] = [1.0] * MAX_CHANNELS
        self._ch_range_min:  list[float] = [0.0] * MAX_CHANNELS
        self._ch_range_max:  list[float] = [1.0] * MAX_CHANNELS
        self._ch_iso_value:  list[float] = [0.5] * MAX_CHANNELS

        # Global params (same for every channel)
        self.quality:   float = 0.5
        self.dithering: float = 0.3

        # ROI clipping (texture coordinate space, [0,1] per axis)
        self.roi_min: np.ndarray = np.zeros(3, dtype=np.float32)
        self.roi_max: np.ndarray = np.ones(3, dtype=np.float32)
        # Coordinate flip (swap origin to opposite corner)
        self.flip_coords: bool = True

        # --- Camera -------------------------------------------------------
        self._rotation_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._translation: np.ndarray = np.zeros(2, dtype=np.float32)
        self._camera_distance: float = 2.0
        self._fov_deg: float = 45.0

        # --- Light --------------------------------------------------------
        self._light_dir: np.ndarray = np.array([0.5, 0.5, 1.0], dtype=np.float32)

        # --- Box / mesh overlay -------------------------------------------
        self.show_box: bool = False
        self.show_mesh: bool = False
        self.mesh_divisions: int = 4
        self._box_overlay = None

        # --- GL handles ---------------------------------------------------
        self._program: int = 0
        self._vao: int = 0
        self._vbo: int = 0
        self._vol_textures: list[int] = [0] * MAX_CHANNELS   # one per channel
        self._tf_textures: list[int] = [0] * MAX_CHANNELS    # one per channel
        self._initialised: bool = False
        self._noise_seed: float = 0.0
        self._vol_dirty: bool = False   # True → re-upload vol textures next frame
        self._box_dirty: bool = False   # True → rebuild box/mesh geometry next frame
        self._use_gl_max: bool = True   # False on drivers that reject GL_MAX (ANGLE)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def set_stack(self, stack: VolumeStack) -> None:
        """Load a VolumeStack (multi-channel and/or time-lapse)."""
        self._stack = stack
        self._current_timepoint = 0

        # Representative volume (for aspect ratio, axis overlay, None-checks)
        self._volume = _make_representative_volume(stack)

        # Auto-fit camera
        asp = stack.aspect_ratio
        half_max_xy = max(float(asp[0]), float(asp[1])) * 0.5
        self._camera_distance = half_max_xy / math.tan(math.radians(self._fov_deg * 0.5)) * 1.25
        self._translation = np.zeros(2, dtype=np.float32)
        self._rotation_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Auto-set per-channel params.
        # range_max is set to 1.0 (the full normalised range) so that no
        # signal is clipped on first display.  The user can narrow the window
        # via the range sliders for contrast adjustment.
        for c in range(stack.n_channels):
            self._ch_range_max[c] = 0.5   # half of normalised max → brighter initial view
            self._ch_range_min[c] = 0.0
            self._ch_brightness[c] = 1.0
            self._ch_gamma[c] = 1.0
            self._ch_iso_value[c] = 0.5

        # Reset channel visibility for actual n_channels
        for c in range(MAX_CHANNELS):
            self._channel_visible[c] = (c < stack.n_channels)

        # For RGB volumes use fixed R/G/B transfer functions so that the
        # three colour channels composite correctly via the multi-channel path.
        if stack.is_rgb:
            self._channel_tfs[0] = TransferFunction1D.red()
            self._channel_tfs[1] = TransferFunction1D.green()
            self._channel_tfs[2] = TransferFunction1D.blue()
            self._tf_dirty = [True] * MAX_CHANNELS

        self._vol_dirty = True
        self._box_dirty = True
        self._tf_dirty = [True] * MAX_CHANNELS

        if self._initialised:
            self._upload_stack_frame(0)
            self._upload_all_tfs()

    def set_volume(self, volume: Volume) -> None:
        """Backward-compat wrapper: wrap a single Volume in a 1-ch, 1-t stack."""
        from ..volume.volume import VolumeStack
        stack = VolumeStack(
            n_channels=1,
            n_timepoints=1,
            depth=volume.depth,
            height=volume.height,
            width=volume.width,
            voxel_size_x=volume.voxel_size_x,
            voxel_size_y=volume.voxel_size_y,
            voxel_size_z=volume.voxel_size_z,
            file_path=volume.file_path,
            is_lazy=False,
            data=[[volume.data]],
            norm_lo=[0.0],
            norm_hi=[1.0],
            _data_kind="ZYX",
        )
        self.set_stack(stack)

    def set_timepoint(self, t: int) -> None:
        """Change to timepoint *t* without touching camera state."""
        if self._stack is None:
            return
        t = max(0, min(t, self._stack.n_timepoints - 1))
        if t == self._current_timepoint:
            return
        self._current_timepoint = t
        self._vol_dirty = True
        # Actual re-upload happens at the start of the next render() call.

    def set_channel_visible(self, c: int, visible: bool) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._channel_visible[c] = visible

    def set_channel_tf(self, c: int, tf: TransferFunction1D) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._channel_tfs[c] = tf
            self._tf_dirty[c] = True

    def set_transfer_function(self, tf: TransferFunction1D) -> None:
        """Set transfer function for channel 0 (backward compat)."""
        self.set_channel_tf(0, tf)

    def set_algorithm(self, algo: RenderAlgorithm) -> None:
        self._algorithm = algo

    # --- Channel-0 aliases for backward compatibility -------------------

    @property
    def brightness(self) -> float:
        return self._ch_brightness[0]

    @brightness.setter
    def brightness(self, v: float) -> None:
        self._ch_brightness[0] = float(v)

    @property
    def gamma(self) -> float:
        return self._ch_gamma[0]

    @gamma.setter
    def gamma(self, v: float) -> None:
        self._ch_gamma[0] = float(v)

    @property
    def range_min(self) -> float:
        return self._ch_range_min[0]

    @range_min.setter
    def range_min(self, v: float) -> None:
        self._ch_range_min[0] = float(v)

    @property
    def range_max(self) -> float:
        return self._ch_range_max[0]

    @range_max.setter
    def range_max(self, v: float) -> None:
        self._ch_range_max[0] = float(v)

    @property
    def iso_value(self) -> float:
        return self._ch_iso_value[0]

    @iso_value.setter
    def iso_value(self, v: float) -> None:
        self._ch_iso_value[0] = float(v)

    # --- Per-channel parameter setters/getters ---------------------------

    def set_channel_brightness(self, c: int, v: float) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._ch_brightness[c] = float(v)

    def set_channel_gamma(self, c: int, v: float) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._ch_gamma[c] = float(v)

    def set_channel_range_min(self, c: int, v: float) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._ch_range_min[c] = float(v)

    def set_channel_range_max(self, c: int, v: float) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._ch_range_max[c] = float(v)

    def set_channel_iso(self, c: int, v: float) -> None:
        if 0 <= c < MAX_CHANNELS:
            self._ch_iso_value[c] = float(v)

    def get_channel_brightness(self, c: int) -> float:
        return self._ch_brightness[c] if 0 <= c < MAX_CHANNELS else 1.0

    def get_channel_gamma(self, c: int) -> float:
        return self._ch_gamma[c] if 0 <= c < MAX_CHANNELS else 1.0

    def get_channel_range_min(self, c: int) -> float:
        return self._ch_range_min[c] if 0 <= c < MAX_CHANNELS else 0.0

    def get_channel_range_max(self, c: int) -> float:
        return self._ch_range_max[c] if 0 <= c < MAX_CHANNELS else 1.0

    def get_channel_iso(self, c: int) -> float:
        return self._ch_iso_value[c] if 0 <= c < MAX_CHANNELS else 0.5

    @property
    def n_channels(self) -> int:
        return self._stack.n_channels if self._stack is not None else 0

    @property
    def n_timepoints(self) -> int:
        return self._stack.n_timepoints if self._stack is not None else 0

    @property
    def current_timepoint(self) -> int:
        return self._current_timepoint

    def mark_box_dirty(self) -> None:
        """Schedule a box/mesh geometry rebuild on the next render() call."""
        self._box_dirty = True

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation_quat

    @rotation.setter
    def rotation(self, q: np.ndarray) -> None:
        self._rotation_quat = np.asarray(q, dtype=np.float32)

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @translation.setter
    def translation(self, t: np.ndarray) -> None:
        self._translation = np.asarray(t, dtype=np.float32)

    # -----------------------------------------------------------------------
    # GL lifecycle
    # -----------------------------------------------------------------------

    def initGL(self) -> None:
        """Compile shaders, build quad VAO, allocate textures."""
        while glGetError() != GL_NO_ERROR:
            pass

        glClearColor(0.05, 0.05, 0.05, 1.0)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._program = self._compile_program()
        self._vao, self._vbo = self._build_quad()

        # Pin sampler uniforms to fixed texture units immediately after linking.
        #   unit 0 → volumeTex   (per-channel greyscale volume)
        #   unit 1 → transferTex  (per-channel 1-D LUT)
        glUseProgram(self._program)
        glUniform1i(glGetUniformLocation(self._program, "volumeTex"),   0)
        glUniform1i(glGetUniformLocation(self._program, "transferTex"), 1)
        glUseProgram(0)

        # Probe GL_MAX blend equation support.
        # ANGLE on Windows may not honour GL_MAX even when glGetError() returns
        # GL_NO_ERROR.  Test it now; fall back to additive (GL_ONE+GL_ONE) if
        # unsupported.  Clear stale errors first so the probe is clean.
        while glGetError() != GL_NO_ERROR:
            pass
        glBlendEquation(GL_MAX)
        self._use_gl_max = (glGetError() == GL_NO_ERROR)
        glBlendEquation(GL_FUNC_ADD)   # restore default
        while glGetError() != GL_NO_ERROR:
            pass

        # Allocate per-channel textures
        self._vol_textures = list(glGenTextures(MAX_CHANNELS))
        self._tf_textures  = list(glGenTextures(MAX_CHANNELS))

        self._upload_all_tfs()

        if self._stack is not None:
            self._upload_stack_frame(self._current_timepoint)

        from ..overlay.box_overlay import BoxOverlay
        self._box_overlay = BoxOverlay()
        self._box_overlay.initGL()
        self._box_dirty = True  # will be built on the first render() call

        self._initialised = True

    def render(self, viewport_w: int, viewport_h: int) -> None:
        """Draw one frame.  Must be called with the GL context current."""
        glViewport(0, 0, viewport_w, viewport_h)
        glClear(GL_COLOR_BUFFER_BIT)

        if not self._initialised or self._stack is None:
            return

        # Re-assert critical GL state every frame.
        # On Windows, Qt's QPainter (used for the axis overlay) interacts with
        # the ANGLE/D3D11 backend and can silently reset GL state between frames —
        # including disabling GL_BLEND.  When blend is disabled the second channel
        # draw call overwrites the first, so only the last visible channel appears.
        # Setting state here (rather than only in initGL) is the standard defensive
        # pattern and fixes the Windows multi-channel compositing bug.
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        # Deferred uploads / geometry rebuilds (GL context is current here)
        if self._vol_dirty:
            self._upload_stack_frame(self._current_timepoint)
            self._vol_dirty = False

        if self._box_dirty and self._box_overlay is not None and self._stack is not None:
            self._box_overlay.update_geometry(
                self._stack.aspect_ratio, self.mesh_divisions,
                self.roi_min, self.roi_max,
            )
            self._box_dirty = False

        for c in range(self._stack.n_channels):
            if self._tf_dirty[c] or self._channel_tfs[c].dirty:
                self._upload_transfer_function(c)
                self._tf_dirty[c] = False

        # Advance dithering seed
        self._noise_seed = (self._noise_seed + 0.1) % 1.0

        # Determine visible channels
        visible = [c for c in range(self._stack.n_channels)
                   if self._channel_visible[c]]
        if not visible:
            return

        glUseProgram(self._program)
        self._upload_uniforms_global(viewport_w, viewport_h)

        # Unified multi-channel path — works for single-channel, multi-channel,
        # and RGB (which is stored as 3 R/G/B greyscale channels + fixed TFs).
        # For multi-channel (including RGB) switch to (GL_ONE, GL_ONE) factors
        # before optionally enabling GL_MAX so that D3D11/ANGLE computes
        # max(1*src, 1*dst) = per-component max, identical to desktop OpenGL.
        if len(visible) > 1:
            glBlendFunc(GL_ONE, GL_ONE)
            if self._use_gl_max:
                glBlendEquation(GL_MAX)

        for c in visible:
            self._upload_uniforms_channel(c)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self._vol_textures[c])

            glActiveTexture(GL_TEXTURE0 + 1)
            glBindTexture(GL_TEXTURE_2D, self._tf_textures[c])

            glBindVertexArray(self._vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

        if len(visible) > 1:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            if self._use_gl_max:
                glBlendEquation(GL_FUNC_ADD)

        glUseProgram(0)

        # Box / mesh overlay
        if self._box_overlay is not None and (self.show_box or self.show_mesh):
            aspect_ratio = viewport_w / max(viewport_h, 1)
            proj  = perspective_matrix(self._fov_deg, aspect_ratio, 0.01, 100.0)
            view  = self._build_view_matrix()
            model = self._build_model_matrix()
            mvp   = (proj @ view @ model).astype(np.float32)
            self._box_overlay.render(mvp, self.show_box, self.show_mesh)

    def cleanup(self) -> None:
        """Delete all GL objects."""
        if not self._initialised:
            return
        for tex in self._vol_textures + self._tf_textures:
            if tex:
                glDeleteTextures(1, [tex])
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        if self._program:
            glDeleteProgram(self._program)
        if self._box_overlay is not None:
            self._box_overlay.cleanup()
        self._initialised = False

    # -----------------------------------------------------------------------
    # Private — GL helpers
    # -----------------------------------------------------------------------

    def _upload_uniforms_global(self, viewport_w: int, viewport_h: int) -> None:
        """Upload camera, geometry, and global rendering uniforms (same for every channel)."""
        aspect_ratio = viewport_w / max(viewport_h, 1)
        glUniform1f(glGetUniformLocation(self._program, "viewportW"), float(viewport_w))
        glUniform1f(glGetUniformLocation(self._program, "viewportH"), float(viewport_h))

        tan_half_fov = math.tan(math.radians(self._fov_deg * 0.5))
        glUniform1f(glGetUniformLocation(self._program, "cameraDistance"), self._camera_distance)
        glUniform1f(glGetUniformLocation(self._program, "tanHalfFov"),     tan_half_fov)
        glUniform1f(glGetUniformLocation(self._program, "aspectRatio"),    aspect_ratio)

        model     = self._build_model_matrix()
        inv_model = np.linalg.inv(model).astype(np.float32)
        _set_uniform_mat4(self._program, "modelMatrix",        model)
        _set_uniform_mat4(self._program, "inverseModelMatrix", inv_model)

        asp = self._stack.aspect_ratio
        glUniform3f(glGetUniformLocation(self._program, "volumeAspect"),
                    asp[0], asp[1], asp[2])

        glUniform1f(glGetUniformLocation(self._program, "quality"),   self.quality)
        glUniform1f(glGetUniformLocation(self._program, "dithering"), self.dithering)
        glUniform1f(glGetUniformLocation(self._program, "noiseSeed"), self._noise_seed)
        glUniform3f(glGetUniformLocation(self._program, "lightDir"),  *self._light_dir.tolist())

        glUniform3f(glGetUniformLocation(self._program, "roiMin"), *self.roi_min.tolist())
        glUniform3f(glGetUniformLocation(self._program, "roiMax"), *self.roi_max.tolist())

    def _upload_uniforms_channel(self, c: int) -> None:
        """Upload per-channel intensity/contrast uniforms for channel *c*."""
        glUniform1f(glGetUniformLocation(self._program, "brightness"), self._ch_brightness[c])
        glUniform1f(glGetUniformLocation(self._program, "gamma"),      self._ch_gamma[c])
        glUniform1f(glGetUniformLocation(self._program, "rangeMin"),   self._ch_range_min[c])
        glUniform1f(glGetUniformLocation(self._program, "rangeMax"),   self._ch_range_max[c])
        glUniform1i(glGetUniformLocation(self._program, "renderMode"), int(self._algorithm))
        glUniform1f(glGetUniformLocation(self._program, "isoValue"),   self._ch_iso_value[c])

    def _upload_stack_frame(self, t: int) -> None:
        """Upload 3D textures for all channels at timepoint *t*.

        RGB volumes are stored as 3 separate R/G/B greyscale channels so that
        they go through the same single-sampler multi-channel render path as
        ordinary fluorescence volumes.  This avoids the need for a second
        sampler3D in the shader, which caused draw-call failures on Windows
        D3D11/ANGLE when the second sampler pointed to an uninitialized texture.
        """
        if self._stack is None:
            return
        frame = self._stack.get_frame(t)
        for c, channel_data in enumerate(frame):
            self._upload_channel_volume(c, channel_data)
        # Keep _volume pointing to ch 0 for aspect ratio / None-check
        if frame:
            self._volume = Volume(
                data=frame[0],
                voxel_size_x=self._stack.voxel_size_x,
                voxel_size_y=self._stack.voxel_size_y,
                voxel_size_z=self._stack.voxel_size_z,
            )

    def _upload_channel_volume(self, c: int, data: np.ndarray) -> None:
        """Upload a single (Z, Y, X) float32 array to vol_textures[c]."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        Z, Y, X = data.shape

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_3D, self._vol_textures[c])
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, X, Y, Z, 0, GL_RED, GL_FLOAT, data)
        glBindTexture(GL_TEXTURE_3D, 0)

    def _upload_transfer_function(self, c: int) -> None:
        """Upload channel c's TF as a single-row 2D texture."""
        lut = np.ascontiguousarray(self._channel_tfs[c].lut, dtype=np.float32)
        n = lut.shape[0]

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self._tf_textures[c])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, n, 1, 0, GL_RGBA, GL_FLOAT, lut)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._channel_tfs[c].mark_clean()

    def _upload_all_tfs(self) -> None:
        for c in range(MAX_CHANNELS):
            self._upload_transfer_function(c)
        self._tf_dirty = [False] * MAX_CHANNELS

    def _compile_program(self) -> int:
        vert_src = _read_shader("volume_vert.glsl")
        frag_src = _read_shader("volume_frag.glsl")
        vert = _compile_shader(vert_src, GL_VERTEX_SHADER)
        frag = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glLinkProgram(prog)
        status = glGetProgramiv(prog, GL_LINK_STATUS)
        if not status:
            log = glGetProgramInfoLog(prog).decode()
            raise RuntimeError(f"Shader program link error:\n{log}")
        return prog

    def _build_quad(self) -> tuple[int, int]:
        """Return (VAO, VBO) for a full-screen triangle pair."""
        verts = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0, -1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        stride = 4 * verts.itemsize
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT_TYPE, False, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT_TYPE, False, stride,
                              ctypes.c_void_p(2 * verts.itemsize))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return vao, vbo

    def _build_view_matrix(self) -> np.ndarray:
        cam_pos = np.array([0.0, 0.0, self._camera_distance], dtype=np.float32)
        target  = np.zeros(3, dtype=np.float32)
        up      = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at_matrix(cam_pos, target, up)

    def _build_model_matrix(self) -> np.ndarray:
        rot = quaternion_to_matrix4(self._rotation_quat)
        tx, ty = float(self._translation[0]), float(self._translation[1])
        trans = np.array([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        model = (trans @ rot).astype(np.float32)
        if self.flip_coords:
            # Post-multiply by a −1 scale on all three spatial axes.
            # This reflects the volume through its own centre in object space,
            # so the box overlay, axis overlay, and ray-marching all see the
            # same (consistently flipped) transform — eliminating misalignment.
            flip = np.diag([-1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            model = (model @ flip).astype(np.float32)
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_representative_volume(stack: VolumeStack) -> Volume:
    """Create a minimal Volume from a stack for aspect-ratio / None-check use."""
    frame = stack.get_frame(0)
    return Volume(
        data=frame[0],
        voxel_size_x=stack.voxel_size_x,
        voxel_size_y=stack.voxel_size_y,
        voxel_size_z=stack.voxel_size_z,
        file_path=stack.file_path,
    )


def _read_shader(filename: str) -> str:
    path = os.path.join(_SHADER_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not status:
        log = glGetShaderInfoLog(shader).decode()
        kind = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        raise RuntimeError(f"{kind} shader compile error:\n{log}")
    return shader


def _set_uniform_mat4(program: int, name: str, mat: np.ndarray) -> None:
    loc = glGetUniformLocation(program, name)
    if loc == -1:
        return
    glUniformMatrix4fv(loc, 1, True, mat.astype(np.float32))
