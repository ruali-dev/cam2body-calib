"""Camera model holding intrinsics and distortion parameters.

Supports both pinhole (standard OpenCV) and fisheye (cv2.fisheye) models.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraModel:
    """Camera intrinsic parameters.

    Attributes:
        K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        D: Distortion coefficients.
            - pinhole: [k1, k2, p1, p2, k3(, k4, k5, k6)]
            - fisheye: [k1, k2, k3, k4]
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        camera_name: Human-readable camera identifier.
        model: Camera model type — "pinhole" or "fisheye".
    """

    K: np.ndarray
    D: np.ndarray
    image_width: int
    image_height: int
    camera_name: str = ""
    model: str = "pinhole"

    @classmethod
    def from_config(cls, config: dict) -> "CameraModel":
        """Create CameraModel from a configuration dictionary.

        Expected keys: camera_name, image_width, image_height,
                      K (3x3 list), D (list), model (optional str).
        """
        K = np.array(config["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"Camera matrix K must be 3x3, got {K.shape}")

        D = np.array(config["D"], dtype=np.float64)
        if D.ndim != 1:
            raise ValueError(f"Distortion D must be a 1D list, got shape {D.shape}")

        if D.size == 0:
            D = np.zeros(5, dtype=np.float64)

        model = str(config.get("model", "pinhole")).lower()
        if model not in ("pinhole", "fisheye"):
            raise ValueError(
                f"Unknown camera model '{model}'. Expected 'pinhole' or 'fisheye'."
            )

        return cls(
            K=K,
            D=D,
            image_width=int(config["image_width"]),
            image_height=int(config["image_height"]),
            camera_name=str(config.get("camera_name", "")),
            model=model,
        )

    def undistort_image(
        self,
        image: np.ndarray,
        balance: float = 0.0,
        new_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, "CameraModel"]:
        """Undistort an image, returning the rectified image and updated CameraModel.

        For fisheye: uses cv2.fisheye.initUndistortRectifyMap + remap.
            The new K is computed via estimateNewCameraMatrixForUndistortRectify.
            D is set to zeros (the rectified image has no distortion).

        For pinhole: uses cv2.undistort with the original K.
            D is set to zeros.

        If D is already all zeros, returns the image unchanged.

        Args:
            image: Input BGR image (H, W, 3).
            balance: Fisheye undistortion balance [0, 1].
                0 = maximum crop to valid pixels (no black borders).
                1 = retain all original pixels (with black borders).
                Only used for fisheye model.
            new_size: Optional output size (width, height). Defaults to input size.

        Returns:
            (undistorted_image, updated_camera_model) where the camera model
            has the rectified K and D=zeros.
        """
        if not np.any(self.D != 0):
            # Already distortion-free
            return image.copy(), CameraModel(
                K=self.K.copy(),
                D=np.zeros_like(self.D),
                image_width=self.image_width,
                image_height=self.image_height,
                camera_name=self.camera_name,
                model="pinhole",
            )

        h, w = image.shape[:2]
        if new_size is None:
            new_size = (w, h)

        if self.model == "fisheye":
            return self._undistort_fisheye(image, balance, new_size)
        else:
            return self._undistort_pinhole(image, new_size)

    def _undistort_fisheye(
        self, image: np.ndarray, balance: float, new_size: tuple[int, int]
    ) -> tuple[np.ndarray, "CameraModel"]:
        """Undistort using OpenCV fisheye model."""
        h, w = image.shape[:2]

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (w, h), np.eye(3), balance=balance,
        )

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, new_size, cv2.CV_16SC2,
        )
        undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        return undistorted, CameraModel(
            K=new_K,
            D=np.zeros(4, dtype=np.float64),
            image_width=new_size[0],
            image_height=new_size[1],
            camera_name=self.camera_name,
            model="pinhole",  # after rectification, projection is pinhole
        )

    def _undistort_pinhole(
        self, image: np.ndarray, new_size: tuple[int, int]
    ) -> tuple[np.ndarray, "CameraModel"]:
        """Undistort using standard pinhole (radial+tangential) model."""
        undistorted = cv2.undistort(image, self.K, self.D, None, self.K)

        return undistorted, CameraModel(
            K=self.K.copy(),
            D=np.zeros_like(self.D),
            image_width=new_size[0],
            image_height=new_size[1],
            camera_name=self.camera_name,
            model="pinhole",
        )
