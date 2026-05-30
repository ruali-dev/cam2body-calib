"""Abstract base class for fiducial marker detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """Result of fiducial marker detection.

    Attributes:
        marker_ids: Array of detected marker IDs, shape (N,). Empty if none detected.
        corners_2d: Array of detected corners, shape (N, 4, 2).
            corners_2d[i] contains 4 corners (x, y) for marker with marker_ids[i].
            Corner order: clockwise from top-left (OpenCV ArUco convention).
            This ordering MUST match the 3D corner ordering in the layout.
    """

    marker_ids: np.ndarray
    corners_2d: np.ndarray

    def __post_init__(self):
        if self.marker_ids is None:
            self.marker_ids = np.array([], dtype=np.int32)
        if self.corners_2d is None:
            self.corners_2d = np.empty((0, 4, 2), dtype=np.float64)
        self.corners_2d = np.asarray(self.corners_2d, dtype=np.float64)

    @property
    def num_detected(self) -> int:
        return len(self.marker_ids)


class FiducialDetector(ABC):
    """Abstract base for fiducial marker detectors.

    Subclasses implement detect() for specific marker types
    (ArUco, AprilTag, ChArUco, etc.).
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect fiducial markers in the given image.

        Args:
            image: BGR or grayscale image as numpy array.

        Returns:
            DetectionResult with detected marker IDs and 2D corner positions.
        """
        ...
