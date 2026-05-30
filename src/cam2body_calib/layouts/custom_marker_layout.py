"""Custom marker layout: explicit per-marker corner positions in body frame.

Loaded from YAML, each marker ID maps to its four corner 3D positions.
Corner order in YAML MUST match OpenCV ArUco detection order
(clockwise from top-left in the marker's canonical orientation).

Example YAML:
    dictionary: DICT_4X4_50
    markers:
      0:
        corners_body:
          - [1.0,  0.5, 0.2]   # corner 0: top-left
          - [1.0,  0.6, 0.2]   # corner 1: top-right
          - [1.0,  0.6, 0.1]   # corner 2: bottom-right
          - [1.0,  0.5, 0.1]   # corner 3: bottom-left
"""

import numpy as np

from .base import LayoutProvider


class CustomMarkerLayout(LayoutProvider):
    """Layout where each marker's four corners are explicitly specified in body frame.

    Corner ordering rule:
        The i-th corner in get_corners_body(marker_id) corresponds to
        the i-th corner in ArucoDetector.detect() for that marker.
        OpenCV returns corners clockwise from top-left.
        The YAML must list corners in the SAME ORDER.
    """

    def __init__(self, layout_config: dict):
        """
        Args:
            layout_config: Parsed YAML dict with keys 'dictionary' and 'markers'.

        Raises:
            ValueError: If any marker has wrong corner dimensions.
        """
        self._corners: dict[int, np.ndarray] = {}
        markers_dict = layout_config.get("markers", {})

        if not markers_dict:
            raise ValueError(
                "Marker layout contains no markers. "
                "Add marker entries under the 'markers' key."
            )

        for marker_id_str, entry in markers_dict.items():
            marker_id = int(marker_id_str)
            corners = np.array(entry["corners_body"], dtype=np.float64)
            if corners.shape != (4, 3):
                raise ValueError(
                    f"Marker {marker_id}: expected 4 corners with 3 coordinates each "
                    f"(shape (4,3)), got shape {corners.shape}"
                )
            self._corners[marker_id] = corners

        self._dictionary = layout_config.get("dictionary", "DICT_4X4_50")

    @property
    def dictionary_name(self) -> str:
        """The ArUco dictionary name specified in the layout config."""
        return self._dictionary

    def get_corners_body(self, marker_id: int) -> np.ndarray | None:
        """Get 3D corners for a marker in body frame (meters).

        Args:
            marker_id: Integer marker ID.

        Returns:
            (4, 3) array of corner positions in body frame,
            or None if this marker ID is not in the layout.

        Corner order (must match OpenCV):
            [0] top-left, [1] top-right, [2] bottom-right, [3] bottom-left.
        """
        return self._corners.get(marker_id)

    def known_ids(self) -> set[int]:
        """Return the set of all known marker IDs."""
        return set(self._corners.keys())
