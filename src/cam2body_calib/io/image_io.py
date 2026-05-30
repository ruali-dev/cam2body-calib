"""Image I/O utilities."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read image from file.

    Args:
        path: Path to image file.

    Returns:
        BGR image as numpy array.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If image cannot be read (corrupt or unsupported format).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image (corrupt or unsupported format): {path}")
    return img


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write image to file, creating parent directories if needed.

    Args:
        path: Output path.
        image: Image as numpy array.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise OSError(f"Failed to write image to: {path}")
