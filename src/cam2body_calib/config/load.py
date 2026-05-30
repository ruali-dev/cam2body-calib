"""Configuration loading with validation."""

from ..camera.model import CameraModel
from ..io.yaml_io import read_yaml
from .schemas import CameraConfig, MarkerLayoutConfig


def load_camera(path: str) -> CameraModel:
    """Load and validate camera configuration from YAML.

    Args:
        path: Path to camera YAML config file.

    Returns:
        CameraModel with validated intrinsics and distortion parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        pydantic.ValidationError: If config structure or values are invalid.
    """
    data = read_yaml(path)
    # Validate with Pydantic
    CameraConfig(**data)
    # Build CameraModel from validated data
    return CameraModel.from_config(data)


def load_marker_layout(path: str) -> dict:
    """Load and validate marker layout configuration from YAML.

    Args:
        path: Path to marker layout YAML config file.

    Returns:
        Validated configuration dict ready for LayoutProvider construction.

    Raises:
        FileNotFoundError: If config file does not exist.
        pydantic.ValidationError: If config structure or values are invalid.
    """
    data = read_yaml(path)
    # Validate with Pydantic
    MarkerLayoutConfig(**data)
    return data
