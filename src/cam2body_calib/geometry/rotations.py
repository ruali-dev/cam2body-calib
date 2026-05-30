"""Rotation conversions: rotation matrix <-> Euler angles, frame transforms.

Convention: Fixed-axis RPY (extrinsic XYZ = intrinsic ZYX).
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

Two camera frames:
    optical: OpenCV convention, x=right, y=down, z=forward
    link:    ROS REP-103 convention, x=forward, y=left, z=up

The static rotation between them:
    optical z (forward) -> link x (forward)
    optical x (right)   -> link -y (right = -left)
    optical y (down)    -> link -z (down = -up)
"""

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

# ── Frame transforms ─────────────────────────────────────────────

# R_optical_to_link: maps vectors FROM optical TO link frame.
# X_link = R_optical_to_link @ X_optical
R_OPTICAL_TO_LINK = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)

# R_link_to_optical: inverse of above. X_optical = R_link_to_optical @ X_link
R_LINK_TO_OPTICAL = R_OPTICAL_TO_LINK.T  # [[0,-1,0],[0,0,-1],[1,0,0]]


def convert_optical_to_link(T_body_optical: np.ndarray) -> np.ndarray:
    """Convert T_body_camera_optical to T_body_camera_link.

    Both transforms map points FROM camera TO body, just with different
    camera-frame conventions (optical vs link).

    X_body = R_body_optical @ X_optical + t
           = R_body_link @ X_link + t

    Since X_link = R_optical_to_link @ X_optical:
        R_body_link = R_body_optical @ R_optical_to_link^T
        t is unchanged (same physical point, same position)
    """
    R_body_optical = T_body_optical[:3, :3]
    t = T_body_optical[:3, 3].copy()
    R_body_link = R_body_optical @ R_LINK_TO_OPTICAL
    T_body_link = np.eye(4, dtype=np.float64)
    T_body_link[:3, :3] = R_body_link
    T_body_link[:3, 3] = t
    return T_body_link


# ── RPY extraction ───────────────────────────────────────────────

def rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to [roll, pitch, yaw] in radians.

    Uses fixed-axis (extrinsic XYZ) R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Equivalent to scipy.spatial.transform.Rotation.as_euler("xyz").
    """
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    if np.abs(cos_pitch) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def matrix_to_rpy_scipy(R: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Convert 3x3 rotation matrix to RPY using scipy.

    Uses extrinsic "xyz" (fixed-axis) convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    rpy = ScipyRotation.from_matrix(R).as_euler("xyz", degrees=degrees)
    return np.asarray(rpy, dtype=np.float64)


def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert [roll, pitch, yaw] to 3x3 rotation matrix.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rpy_to_degrees(rpy: np.ndarray) -> np.ndarray:
    """Convert RPY from radians to degrees."""
    return np.degrees(rpy)
