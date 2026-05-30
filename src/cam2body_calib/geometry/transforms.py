"""Transform utilities for 4x4 homogeneous matrices and solvePnP conventions.

CRITICAL CONVENTION:
- solvePnP outputs rvec, tvec such that: X_cam = R * X_obj + t
  where X_obj are points in the object/world/body coordinate system.
- The 4x4 matrix [R t; 0 1] transforms points FROM body TO camera frame: T_cam_body.
- To get camera pose in body frame: T_body_cam = inv(T_cam_body).
"""

import cv2
import numpy as np


def rvec_tvec_to_4x4(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector + translation to 4x4 homogeneous transform.

    Returns T = [[R, t], [0, 0, 0, 1]] where R = Rodrigues(rvec).

    For solvePnP output with body-frame object points, this is T_cam_body:
        X_cam_h = T_cam_body @ X_body_h
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def pose_4x4_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 homogeneous transform to (rvec, tvec)."""
    R = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform.

    If T = [R t; 0 1], then T_inv = [R^T  -R^T t; 0 1].

    solvePnP gives T_cam_body.
        invert_pose(T_cam_body) = T_body_cam  (camera pose in body frame).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform T to 3D points.

    Args:
        T: 4x4 homogeneous transform.
        points: (N, 3) array of 3D points.

    Returns:
        (N, 3) transformed points.
    """
    N = points.shape[0]
    pts_h = np.hstack([points, np.ones((N, 1), dtype=points.dtype)])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]
