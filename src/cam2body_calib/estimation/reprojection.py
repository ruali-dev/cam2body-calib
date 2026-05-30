"""Reprojection error computation utilities."""

import cv2
import numpy as np

from .result import ReprojectionStats


def compute_reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> ReprojectionStats:
    """Compute reprojection errors between observed and projected points.

    Projects 3D object points (in body frame) using the given pose and camera,
    then computes per-point Euclidean distance to observed image points.

    Args:
        object_points: 3D points, shape (N, 3), in body coordinate system.
        image_points: Observed 2D points, shape (N, 2), in image coordinates.
        rvec: Rodrigues rotation vector (body->camera).
        tvec: Translation vector (body->camera).
        K: Camera intrinsic matrix (3, 3).
        D: Distortion coefficients.
        inlier_mask: Optional boolean mask of RANSAC inliers, shape (N,).

    Returns:
        ReprojectionStats with mean, max, and per-point errors.
    """
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)

    errors = np.linalg.norm(image_points - projected, axis=1)

    n_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else len(object_points)

    return ReprojectionStats(
        mean_error=float(np.mean(errors)),
        max_error=float(np.max(errors)),
        per_point_errors=errors,
        inlier_count=n_inliers,
        total_points=len(object_points),
    )
