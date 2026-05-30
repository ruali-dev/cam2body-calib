"""Visualization: draw detections, reprojections, and coordinate axes."""

import cv2
import numpy as np

from ..camera.model import CameraModel
from ..estimation.result import PoseResult
from ..fiducials.base import DetectionResult
from ..geometry.transforms import pose_4x4_to_rvec_tvec


def create_visualization(
    image: np.ndarray,
    detection: DetectionResult,
    object_points_body: np.ndarray,
    image_points: np.ndarray,
    pose_result: PoseResult,
    camera: CameraModel,
) -> np.ndarray:
    """Create a comprehensive visualization of calibration results.

    Draws:
        - Detected marker outlines and IDs (green).
        - Observed corners (red filled circles).
        - Reprojected corners (blue crosses).
        - Error lines connecting observed and reprojected (cyan).
        - Body coordinate axes at the body frame origin (if visible).
        - Pose and error info as text overlay.

    Args:
        image: Original BGR image.
        detection: DetectionResult with marker_ids and corners_2d.
        object_points_body: (N, 3) 3D points in body frame used for PnP.
        image_points: (N, 2) observed 2D points used for PnP.
        pose_result: PoseResult from PnPSolver.
        camera: CameraModel with intrinsics.

    Returns:
        Annotated BGR image as numpy array.
    """
    vis = image.copy()

    # --- 1. Draw detected marker outlines and IDs ---
    if detection.num_detected > 0:
        for corners, m_id in zip(detection.corners_2d, detection.marker_ids):
            pts = corners.reshape(-1, 2).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            # Label with marker ID near the first corner
            label_pos = tuple(pts[0])
            cv2.putText(
                vis, f"ID:{m_id}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )

    # --- 2. Draw reprojections if pose was solved ---
    if pose_result.success and len(object_points_body) > 0:
        rvec, tvec = pose_4x4_to_rvec_tvec(pose_result.T_cam_body)

        # Project ALL object points (not just inliers) for full comparison
        projected, _ = cv2.projectPoints(
            object_points_body.reshape(-1, 3), rvec, tvec,
            camera.K, camera.D,
        )
        projected = projected.reshape(-1, 2)

        # Draw per-point: observed (red circle) + reprojected (blue cross) + error line (cyan)
        for obs, proj in zip(image_points, projected):
            obs_i = tuple(obs.astype(int))
            proj_i = tuple(proj.astype(int))
            error = np.linalg.norm(obs - proj)
            # Size scales with error but clamped
            radius = max(3, min(8, int(error * 2)))
            cv2.circle(vis, obs_i, radius, (0, 0, 255), -1)  # red filled
            cv2.drawMarker(
                vis, proj_i, (255, 0, 0),
                cv2.MARKER_CROSS, max(6, radius + 2), 1,
            )  # blue cross
            cv2.line(vis, obs_i, proj_i, (255, 255, 0), 1)  # cyan error line

        # --- 3. Draw body coordinate axes ---
        # drawFrameAxes draws the object/world/body frame at its origin.
        # The body origin is at position given by rvec, tvec.
        # If the body origin is outside the image FOV, axes may not be visible.
        axis_length = _estimate_axis_length(object_points_body)
        cv2.drawFrameAxes(
            vis, camera.K, camera.D, rvec, tvec, axis_length, thickness=2,
        )
        # Legend for axes
        cv2.putText(
            vis, "Body axes: X=red(fwd) Y=green(left) Z=blue(up)",
            (10, vis.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

        # --- 4. Text overlay with results ---
        pos = pose_result.position_body
        rpy_deg = np.degrees(pose_result.rpy_optical_body_cam)
        stats = pose_result.reprojection_stats

        lines = [
            "Camera pose in body frame:",
            f"  Pos [m]: x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}",
            f"  RPY [deg]: roll={rpy_deg[0]:.2f}  pitch={rpy_deg[1]:.2f}  yaw={rpy_deg[2]:.2f}",
        ]
        if stats is not None:
            lines += [
                f"Reproj err [px]: mean={stats.mean_error:.3f}  max={stats.max_error:.3f}",
                f"Inliers: {stats.inlier_count}/{stats.total_points}",
            ]

        # Draw semi-transparent background for text
        text_h = len(lines) * 22 + 16
        text_w = 580
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (text_w, text_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        for i, line in enumerate(lines):
            y = 22 + i * 22
            cv2.putText(
                vis, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

    elif not pose_result.success:
        # Show error message
        cv2.putText(
            vis, f"Pose estimation FAILED: {pose_result.message}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    return vis


def _estimate_axis_length(object_points: np.ndarray) -> float:
    """Estimate a reasonable axis length based on scene scale.

    Uses 20% of the bounding sphere diameter of the object points,
    clamped to [0.05, 1.0] meters.
    """
    if len(object_points) == 0:
        return 0.2
    centroid = np.mean(object_points, axis=0)
    max_dist = np.max(np.linalg.norm(object_points - centroid, axis=1))
    length = max(0.05, min(1.0, max_dist * 0.2))
    return float(length)
