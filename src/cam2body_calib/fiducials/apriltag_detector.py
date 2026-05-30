"""AprilTag detector (future implementation).

Will use apriltag Python bindings or pupil-apriltags for detection.
"""

import numpy as np

from .base import DetectionResult, FiducialDetector


class AprilTagDetector(FiducialDetector):
    """Placeholder for AprilTag detection.

    Not yet implemented. Will detect AprilTag markers and return
    tag IDs with their 2D corner positions.
    """

    def detect(self, image: np.ndarray) -> DetectionResult:
        raise NotImplementedError("AprilTag detector not yet implemented")
