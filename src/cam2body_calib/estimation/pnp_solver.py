"""PnP-based camera extrinsics estimation.

Solves for T_cam_body (body->camera transform) from 3D-2D correspondences,
then inverts to get T_body_cam (camera pose in body frame).
"""

import cv2
import numpy as np

from ..camera.model import CameraModel
from ..geometry.rotations import (
    convert_optical_to_link,
    matrix_to_rpy_scipy,
)
from ..geometry.transforms import invert_pose, rvec_tvec_to_4x4
from .reprojection import compute_reprojection_errors
from .result import PoseResult


class PnPSolver:
    """Estimate camera extrinsics from 3D-2D point correspondences.

    Pipeline:
        1. solvePnPRansac for robust initial estimate.
        2. (Optional) solvePnPRefineLM on inliers for refinement.
        3. Compute T_cam_body and T_body_cam.
        4. Compute reprojection errors.

    Key transforms:
        T_cam_body: transforms body-frame points to camera frame.
            X_cam = T_cam_body @ X_body_h    (solvePnP output)
        T_body_cam: transforms camera-frame points to body frame.
            X_body = T_body_cam @ X_cam_h    (inverse, camera pose in body)
    """

    def __init__(
        self,
        camera: CameraModel,
        ransac_threshold: float = 3.0,
        refine: bool = True,
    ):
        """
        Args:
            camera: CameraModel with intrinsics and distortion params.
            ransac_threshold: RANSAC reprojection error threshold in pixels.
            refine: Whether to apply LM refinement on RANSAC inliers.
        """
        self.camera = camera
        self.ransac_threshold = ransac_threshold
        self.refine = refine

    def solve(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
    ) -> PoseResult:
        """Solve for camera pose from 3D-2D correspondences.

        Args:
            object_points: 3D points in body frame, shape (N, 3).
            image_points: 2D image points, shape (N, 2).

        Returns:
            PoseResult with both T_cam_body and T_body_cam, plus reprojection stats.
            On failure, PoseResult.success == False with message.
        """
        N = len(object_points)

        if N < 4:
            return PoseResult(
                success=False,
                message=(
                    f"Need at least 4 point correspondences for PnP, got {N}. "
                    "Ensure at least one marker with 4 corners is detected "
                    "and present in the layout."
                ),
            )

        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

        K = self.camera.K
        D = self.camera.D

        # Step 1: Robust PnP with RANSAC
        # solvePnPRansac solves: X_cam = R * X_body + t  => gives T_cam_body
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            D,
            reprojectionError=self.ransac_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 4:
            n_inliers = len(inliers) if inliers is not None else 0
            return PoseResult(
                success=False,
                message=(
                    f"solvePnPRansac failed: only {n_inliers} inliers from "
                    f"{N} points. Check marker layout correctness, image quality, "
                    "or increase --ransac-threshold."
                ),
            )

        # Step 2: Refine on inliers (LM or VVS if available)
        if self.refine:
            obj_inliers = object_points[inliers.ravel()]
            img_inliers = image_points[inliers.ravel()]
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_inliers, img_inliers, K, D, rvec, tvec
                )
            elif hasattr(cv2, "solvePnPRefineVVS"):
                rvec, tvec = cv2.solvePnPRefineVVS(
                    obj_inliers, img_inliers, K, D, rvec, tvec
                )

        # Step 3: Build transforms
        # T_cam_body: body -> camera_optical (solvePnP direct output).
        #   Camera optical frame: x=right, y=down, z=forward.
        T_cam_body = rvec_tvec_to_4x4(rvec, tvec)

        # T_body_camera_optical: camera_optical -> body (INVERSE).
        T_body_optical = invert_pose(T_cam_body)

        # T_body_camera_link: camera_link -> body.
        #   Camera link frame: x=forward, y=left, z=up (same convention as body).
        #   When camera_link is perfectly aligned with body, RPY = (0, 0, 0).
        T_body_link = convert_optical_to_link(T_body_optical)

        # Camera position in body frame (same for both optical and link).
        position_body = T_body_optical[:3, 3].copy()

        # RPY: extrinsic XYZ, R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        rpy_optical = matrix_to_rpy_scipy(T_body_optical[:3, :3])
        rpy_link = matrix_to_rpy_scipy(T_body_link[:3, :3])

        # Step 4: Reprojection errors
        inlier_mask = np.zeros(N, dtype=bool)
        inlier_mask[inliers.ravel()] = True
        reproj_stats = compute_reprojection_errors(
            object_points, image_points, rvec, tvec, K, D, inlier_mask
        )

        return PoseResult(
            T_cam_body=T_cam_body,
            T_body_camera_optical=T_body_optical,
            T_body_camera_link=T_body_link,
            position_body=position_body,
            rpy_optical_body_cam=rpy_optical,
            rpy_link_body_cam=rpy_link,
            reprojection_stats=reproj_stats,
            num_points=N,
            success=True,
            message=f"Pose estimated from {N} points ({reproj_stats.inlier_count} inliers).",
        )
