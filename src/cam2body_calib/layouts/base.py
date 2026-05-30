"""Abstract base class for marker layout providers."""

from abc import ABC, abstractmethod

import numpy as np


class LayoutProvider(ABC):
    """Provides 3D marker corner positions in the body/base_link coordinate system.

    Given a marker ID, returns the 3D coordinates of its four corners
    in the body frame. The corner order MUST match the OpenCV detection order
    (clockwise from top-left in the marker's canonical orientation).
    """

    @abstractmethod
    def get_corners_body(self, marker_id: int) -> np.ndarray | None:
        """Get 3D corners for a marker in body frame.

        Args:
            marker_id: Integer marker ID.

        Returns:
            np.ndarray of shape (4, 3) with corners in body frame (meters),
            or None if the marker ID is not in this layout.

        IMPORTANT:
            Corner order must match OpenCV detection order:
            [top-left, top-right, bottom-right, bottom-left] (clockwise).
        """
        ...

    @abstractmethod
    def known_ids(self) -> set[int]:
        """Return the set of all marker IDs in this layout."""
        ...
