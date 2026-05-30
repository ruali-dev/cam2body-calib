"""Result types for pose estimation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReprojectionStats:
    """Reprojection error statistics for a PnP solution.

    Attributes:
        mean_error: Mean reprojection error in pixels.
        max_error: Maximum reprojection error in pixels.
        per_point_errors: Per-point reprojection errors, shape (N,).
        inlier_count: Number of RANSAC inliers.
        total_points: Total number of point correspondences.
    """

    mean_error: float
    max_error: float
    per_point_errors: np.ndarray
    inlier_count: int
    total_points: int


@dataclass
class PoseResult:
    """Result of PnP pose estimation.

    TWO KEY TRANSFORMS:
        T_cam_body: body -> camera. Direct output of solvePnP.
            X_cam_h = T_cam_body @ X_body_h
        T_body_cam: camera -> body. Camera pose in body frame.
            X_body_h = T_body_cam @ X_cam_h
            T_body_cam = inv(T_cam_body)

    Attributes:
        T_cam_body: 4x4, body->camera transform in OpenCV optical frame
            (solvePnP direct output). X_optical = T_cam_body @ X_body_h.
        T_body_camera_optical: 4x4, camera_optical->body transform
            (= inv(T_cam_body)). Camera optical frame: x=right, y=down, z=forward.
        T_body_camera_link: 4x4, camera_link->body transform.
            Camera link frame: x=forward, y=left, z=up (same convention as body).
            Same position as optical, rotation differs by static 90-degree transform.
        position_body: [x, y, z] camera position in body coordinates (meters).
            x=forward, y=left, z=up.
        rpy_optical_body_cam: [roll, pitch, yaw] camera_optical orientation
            in body frame (radians). Extrinsic XYZ: R = Rz(yaw)@Ry(pitch)@Rx(roll).
        rpy_link_body_cam: [roll, pitch, yaw] camera_link orientation
            in body frame (radians). When camera_link is perfectly aligned with body,
            this is (0, 0, 0).
        reprojection_stats: Detailed reprojection error statistics.
        num_points: Total 3D-2D correspondences used.
        success: Whether pose estimation succeeded.
        message: Human-readable status or error message.
    """

    T_cam_body: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_body_camera_optical: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_body_camera_link: np.ndarray = field(default_factory=lambda: np.eye(4))
    position_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy_optical_body_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy_link_body_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reprojection_stats: ReprojectionStats | None = None
    num_points: int = 0
    success: bool = False
    message: str = ""
