"""
BoxOverlay — draws bounding box edges and mesh grid lines via GL_LINES.

Uses a dedicated GLSL program (line_vert / line_frag) uploaded at initGL.
Geometry is rebuilt whenever the volume aspect ratio, mesh_divisions, or
ROI clipping bounds change.
"""

from __future__ import annotations
import ctypes
import numpy as np

from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glDeleteVertexArrays,
    glGenBuffers, glBindBuffer, glBufferData, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW,
    glEnableVertexAttribArray, glVertexAttribPointer,
    GL_FLOAT,
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv,
    glGetProgramInfoLog, GL_LINK_STATUS,
    glUseProgram, glGetUniformLocation, glUniformMatrix4fv, glUniform4f,
    glDrawArrays, GL_LINES,
    glDeleteProgram,
    glEnable, GL_BLEND,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
)


_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
uniform vec4 lineColor;
out vec4 fragColor;
void main() {
    fragColor = lineColor;
}
"""


class BoxOverlay:
    """
    Renders bounding-box edges and/or mesh grid lines over the volume.

    The geometry respects the current ROI clipping region: when ROI is
    narrower than the full volume, the box and mesh shrink accordingly.

    Call order:
        initGL()                                      — once, inside an active GL context
        update_geometry(aspect, divs, roi_min, roi_max) — whenever volume/roi/divs changes
        render(mvp, show_box, show_mesh)              — each frame
        cleanup()                                     — on widget destroy
    """

    BOX_COLOR  = (1.0, 1.0, 1.0, 0.7)   # white, semi-transparent
    MESH_COLOR = (0.7, 0.7, 0.7, 0.35)  # grey, subtle

    def __init__(self) -> None:
        self._program: int = 0
        self._vao: int = 0
        self._vbo: int = 0
        self._n_box: int = 0       # number of vertices for box edges (GL_LINES)
        self._n_mesh: int = 0      # number of vertices for mesh lines
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def initGL(self) -> None:
        self._program = _compile_program(_VERT_SRC, _FRAG_SRC)
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)

        # Bind VAO + VBO layout (position: vec3, location 0)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        # Allocate a large enough buffer; fill later in update_geometry
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._initialised = True

    def update_geometry(
        self,
        aspect: np.ndarray,
        mesh_divisions: int,
        roi_min: np.ndarray | None = None,
        roi_max: np.ndarray | None = None,
    ) -> None:
        """
        Rebuild VAO data for the given volume aspect, grid resolution, and ROI.

        *roi_min* / *roi_max* are texture-space coordinates in [0, 1] per axis.
        Default is [0,0,0] / [1,1,1] (full volume).  The box/mesh corners are
        mapped to model space using:  coord = (2*tex - 1) * half_extent.
        """
        if not self._initialised:
            return

        hx = float(aspect[0]) * 0.5
        hy = float(aspect[1]) * 0.5
        hz = float(aspect[2]) * 0.5

        # ROI in texture-space [0,1] → model-space corners
        rmin = roi_min if roi_min is not None else np.zeros(3, dtype=np.float32)
        rmax = roi_max if roi_max is not None else np.ones(3, dtype=np.float32)

        x0 = float((2.0 * rmin[0] - 1.0) * hx)
        y0 = float((2.0 * rmin[1] - 1.0) * hy)
        z0 = float((2.0 * rmin[2] - 1.0) * hz)
        x1 = float((2.0 * rmax[0] - 1.0) * hx)
        y1 = float((2.0 * rmax[1] - 1.0) * hy)
        z1 = float((2.0 * rmax[2] - 1.0) * hz)

        box_verts  = _box_lines(x0, y0, z0, x1, y1, z1)
        mesh_verts = _mesh_lines(x0, y0, z0, x1, y1, z1, max(1, mesh_divisions))

        self._n_box  = len(box_verts)  // 3
        self._n_mesh = len(mesh_verts) // 3

        all_verts = np.concatenate([box_verts, mesh_verts]).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, all_verts.nbytes, all_verts, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self, mvp: np.ndarray, show_box: bool, show_mesh: bool) -> None:
        if not self._initialised:
            return
        if not show_box and not show_mesh:
            return

        glUseProgram(self._program)
        loc_mvp   = glGetUniformLocation(self._program, "mvp")
        loc_color = glGetUniformLocation(self._program, "lineColor")

        glUniformMatrix4fv(loc_mvp, 1, True, mvp.astype(np.float32))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self._vao)

        if show_box and self._n_box > 0:
            glUniform4f(loc_color, *self.BOX_COLOR)
            glDrawArrays(GL_LINES, 0, self._n_box)

        if show_mesh and self._n_mesh > 0:
            glUniform4f(loc_color, *self.MESH_COLOR)
            glDrawArrays(GL_LINES, self._n_box, self._n_mesh)

        glBindVertexArray(0)
        glUseProgram(0)

    def cleanup(self) -> None:
        if not self._initialised:
            return
        glDeleteBuffers(1, [self._vbo])
        glDeleteVertexArrays(1, [self._vao])
        glDeleteProgram(self._program)
        self._initialised = False


# ---------------------------------------------------------------------------
# Geometry builders — take explicit (x0,y0,z0)→(x1,y1,z1) ROI corners
# ---------------------------------------------------------------------------

def _box_lines(
    x0: float, y0: float, z0: float,
    x1: float, y1: float, z1: float,
) -> np.ndarray:
    """Return flat float32 array of vertex pairs for 12 box edges."""
    corners = np.array([
        [x0, y0, z0],  # 0  min-min-min
        [x1, y0, z0],  # 1  max-min-min
        [x1, y1, z0],  # 2  max-max-min
        [x0, y1, z0],  # 3  min-max-min
        [x0, y0, z1],  # 4  min-min-max
        [x1, y0, z1],  # 5  max-min-max
        [x1, y1, z1],  # 6  max-max-max
        [x0, y1, z1],  # 7  min-max-max
    ], dtype=np.float32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face  (-z)
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face (+z)
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]

    verts = []
    for a, b in edges:
        verts.append(corners[a])
        verts.append(corners[b])

    return np.array(verts, dtype=np.float32).ravel()


def _mesh_lines(
    x0: float, y0: float, z0: float,
    x1: float, y1: float, z1: float,
    divs: int,
) -> np.ndarray:
    """
    Return flat float32 array of grid lines on all 6 faces of the ROI box.

    Each face gets (divs-1) interior lines along each face axis.
    """
    verts = []

    def grid_lines(p0, u_unit, v_unit, u_len, v_len, u_divs, v_divs):
        for i in range(1, v_divs):
            t = i / v_divs
            start = p0 + v_unit * v_len * t
            verts.append(start)
            verts.append(start + u_unit * u_len)
        for i in range(1, u_divs):
            t = i / u_divs
            start = p0 + u_unit * u_len * t
            verts.append(start)
            verts.append(start + v_unit * v_len)

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    p000 = np.array([x0, y0, z0], dtype=np.float32)
    ux = np.array([1, 0, 0], dtype=np.float32)
    uy = np.array([0, 1, 0], dtype=np.float32)
    uz = np.array([0, 0, 1], dtype=np.float32)

    faces = [
        (p000,                       ux, uy, dx, dy),  # back   (z = z0)
        (p000 + uz * dz,             ux, uy, dx, dy),  # front  (z = z1)
        (p000,                       uz, uy, dz, dy),  # left   (x = x0)
        (p000 + ux * dx,             uz, uy, dz, dy),  # right  (x = x1)
        (p000,                       ux, uz, dx, dz),  # bottom (y = y0)
        (p000 + uy * dy,             ux, uz, dx, dz),  # top    (y = y1)
    ]

    for corner, u_ax, v_ax, u_len, v_len in faces:
        grid_lines(corner, u_ax, v_ax, u_len, v_len, divs, divs)

    if not verts:
        return np.array([], dtype=np.float32)

    return np.array(verts, dtype=np.float32).ravel()


# ---------------------------------------------------------------------------
# Shader helpers
# ---------------------------------------------------------------------------

def _compile_program(vert_src: str, frag_src: str) -> int:
    vert = _compile_shader(vert_src, GL_VERTEX_SHADER)
    frag = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vert)
    glAttachShader(prog, frag)
    glLinkProgram(prog)
    status = glGetProgramiv(prog, GL_LINK_STATUS)
    if not status:
        log = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"BoxOverlay shader link error:\n{log}")
    return prog


def _compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not status:
        log = glGetShaderInfoLog(shader).decode()
        kind = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        raise RuntimeError(f"BoxOverlay {kind} shader error:\n{log}")
    return shader
