"""ChArUco board detector (future implementation).

ChArUco boards combine chessboard corners with ArUco markers for
improved corner localization accuracy.
"""

import numpy as np

from .base import DetectionResult, FiducialDetector


class CharucoDetector(FiducialDetector):
    """Placeholder for ChArUco board detection.

    Not yet implemented. Will detect ChArUco board corners and return
    marker IDs with their 2D corner positions.
    """

    def detect(self, image: np.ndarray) -> DetectionResult:
        raise NotImplementedError("ChArUco detector not yet implemented")
