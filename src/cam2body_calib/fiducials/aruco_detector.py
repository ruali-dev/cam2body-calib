"""ArUco marker detector using OpenCV contrib.

Supports both new (cv2.aruco.ArucoDetector, OpenCV 4.7+) and
old (cv2.aruco.detectMarkers) APIs.
"""

import cv2
import numpy as np

from .base import DetectionResult, FiducialDetector


class ArucoDetector(FiducialDetector):
    """Detect ArUco markers in images.

    Corner order (OpenCV convention):
        corners[i][0] = top-left
        corners[i][1] = top-right
        corners[i][2] = bottom-right
        corners[i][3] = bottom-left
    All in clockwise order when viewed from the front of the marker.
    """

    def __init__(self, dictionary_name: str = "DICT_4X4_50"):
        """
        Args:
            dictionary_name: OpenCV ArUco dictionary name, e.g. "DICT_4X4_50",
                "DICT_6X6_250", etc.

        Raises:
            ValueError: If the dictionary name is not recognized.
        """
        self.dictionary_name = dictionary_name
        dict_id = getattr(cv2.aruco, dictionary_name, None)
        if dict_id is None:
            available = sorted(
                d for d in dir(cv2.aruco) if d.startswith("DICT_")
            )
            raise ValueError(
                f"Unknown ArUco dictionary: {dictionary_name}. "
                f"Available: {available}"
            )
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.params = cv2.aruco.DetectorParameters()

        # OpenCV 4.7+ has the ArucoDetector class
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._cv_detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)
        else:
            self._cv_detector = None

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect ArUco markers in an image.

        Args:
            image: BGR or grayscale image (H, W, C) or (H, W).

        Returns:
            DetectionResult with marker IDs and 2D corners.
            If no markers detected, returns empty result.
        """
        if self._cv_detector is not None:
            corners, ids, _rejected = self._cv_detector.detectMarkers(image)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                image, self.dictionary, parameters=self.params
            )

        if ids is None or len(ids) == 0:
            return DetectionResult(
                marker_ids=np.array([], dtype=np.int32),
                corners_2d=np.empty((0, 4, 2), dtype=np.float64),
            )

        ids = ids.ravel()
        # corners from OpenCV: list of shape-(1,4,2) or (4,1,2) arrays
        corners_array = np.array(
            [c.reshape(4, 2).astype(np.float64) for c in corners]
        )
        return DetectionResult(marker_ids=ids, corners_2d=corners_array)
