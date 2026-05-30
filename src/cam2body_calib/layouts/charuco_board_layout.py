"""ChArUco board layout provider (future implementation).

Will generate 3D corner positions from board geometry parameters
(square size, marker size, board dimensions).
"""

import numpy as np

from .base import LayoutProvider


class CharucoBoardLayout(LayoutProvider):
    """Placeholder for ChArUco board layout.

    Not yet implemented. Will generate 3D corner positions from
    board parameters like squares_x, squares_y, square_length, marker_length.
    """

    def get_corners_body(self, marker_id: int) -> np.ndarray | None:
        raise NotImplementedError("ChArUco board layout not yet implemented")

    def known_ids(self) -> set[int]:
        raise NotImplementedError("ChArUco board layout not yet implemented")
