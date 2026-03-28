"""Math utilities — quaternions, matrices, arcball."""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers  (storage order: [w, x, y, z])
# ---------------------------------------------------------------------------

def quaternion_identity() -> np.ndarray:
    """Return the identity quaternion [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quaternion_normalise(q: np.ndarray) -> np.ndarray:
    """Return unit quaternion."""
    n = np.linalg.norm(q)
    if n < 1e-8:
        return quaternion_identity()
    return (q / n).astype(np.float32)


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Quaternion from rotation *axis* (unit vector) and *angle* in radians."""
    axis = np.asarray(axis, dtype=np.float32)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return quaternion_identity()
    axis = axis / n
    s = math.sin(angle_rad * 0.5)
    c = math.cos(angle_rad * 0.5)
    return np.array([c, axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float32)


def quaternion_to_matrix4(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 4×4 rotation matrix (row-major)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),  0.0],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x),  0.0],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y),  0.0],
        [              0.0,             0.0,               0.0,  1.0],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Arcball
# ---------------------------------------------------------------------------

def arcball_vector(x_ndc: float, y_ndc: float) -> np.ndarray:
    """
    Map NDC coordinates (in [-1, 1]) to a point on the unit sphere.
    If the point is outside the sphere's equatorial disk, it is projected
    onto the hyperbolic sheet (Shoemake 1992).
    """
    p = np.array([x_ndc, y_ndc, 0.0], dtype=np.float64)
    sq = p[0]*p[0] + p[1]*p[1]
    if sq <= 1.0:
        p[2] = math.sqrt(1.0 - sq)
    else:
        p /= math.sqrt(sq)  # normalise to unit circle
    return p.astype(np.float32)


def arcball_rotation(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Quaternion that rotates *v1* to *v2* on the arcball sphere.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Unit 3-vectors (output of :func:`arcball_vector`).
    """
    angle = math.acos(min(1.0, float(np.dot(v1, v2))))
    if abs(angle) < 1e-6:
        return quaternion_identity()
    axis = np.cross(v1, v2)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return quaternion_identity()
    axis /= n
    return quaternion_from_axis_angle(axis, angle)


# ---------------------------------------------------------------------------
# Standard 4×4 matrices (row-major, matching numpy @ operator)
# ---------------------------------------------------------------------------

def perspective_matrix(fov_deg: float, aspect: float,
                       near: float, far: float) -> np.ndarray:
    """OpenGL-style perspective projection (row-major, right-handed)."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    nf = 1.0 / (near - far)
    return np.array([
        [f / aspect, 0.0,             0.0,            0.0],
        [       0.0,   f,             0.0,            0.0],
        [       0.0, 0.0, (far+near)*nf,  2*far*near*nf],
        [       0.0, 0.0,            -1.0,            0.0],
    ], dtype=np.float32)


def look_at_matrix(eye: np.ndarray, center: np.ndarray,
                   up: np.ndarray) -> np.ndarray:
    """OpenGL-style look-at view matrix (row-major, right-handed)."""
    f = center - eye
    f /= np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float32)
    s = np.cross(f, u)
    s_n = np.linalg.norm(s)
    if s_n < 1e-8:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        s /= s_n
    u = np.cross(s, f)
    return np.array([
        [ s[0],  s[1],  s[2], -float(np.dot(s, eye))],
        [ u[0],  u[1],  u[2], -float(np.dot(u, eye))],
        [-f[0], -f[1], -f[2],  float(np.dot(f, eye))],
        [  0.0,   0.0,   0.0,                     1.0],
    ], dtype=np.float32)


def quaternion_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Convert unit quaternion [w,x,y,z] to intrinsic ZYX Euler angles in degrees.

    Returns (rx_deg, ry_deg, rz_deg).  Rotation order: Rz applied first, then Ry, then Rx.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    ry = math.degrees(math.asin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return rx, ry, rz


def quaternion_from_euler_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Convert intrinsic ZYX Euler angles (degrees) to unit quaternion [w,x,y,z]."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=np.float32)


def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m
