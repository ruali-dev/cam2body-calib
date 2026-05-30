"""Pydantic schemas for configuration validation."""

from pydantic import BaseModel, Field, field_validator

CAMERA_MODELS = ("pinhole", "fisheye")


class CameraConfig(BaseModel):
    """Camera intrinsics configuration schema."""

    camera_name: str = ""
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    K: list[list[float]]  # 3x3 intrinsic matrix
    D: list[float]  # distortion coefficients
    model: str = "pinhole"  # "pinhole" or "fisheye"

    @field_validator("K")
    @classmethod
    def k_must_be_3x3(cls, v: list) -> list:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError(f"K must be a 3x3 matrix, got rows: {[len(row) for row in v]}")
        return v

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v: str) -> str:
        if v.lower() not in CAMERA_MODELS:
            raise ValueError(f"Unknown camera model '{v}'. Expected: {CAMERA_MODELS}")
        return v.lower()


class MarkerEntry(BaseModel):
    """Single marker layout entry."""

    corners_body: list[list[float]]  # 4x3

    @field_validator("corners_body")
    @classmethod
    def must_have_4_corners(cls, v: list) -> list:
        if len(v) != 4:
            raise ValueError(f"Expected 4 corners per marker, got {len(v)}")
        for i, corner in enumerate(v):
            if len(corner) != 3:
                raise ValueError(
                    f"Corner {i}: expected 3 coordinates (x, y, z), got {len(corner)}"
                )
        return v


HANDEDNESS_VALUES = ("right", "left")


class BodyFrameConfig(BaseModel):
    """Body frame convention declaration."""

    name: str = "body"
    convention: str = "x_forward_y_left_z_up"
    handedness: str = "right"

    @field_validator("handedness")
    @classmethod
    def handedness_must_be_valid(cls, v: str) -> str:
        if v.lower() not in HANDEDNESS_VALUES:
            raise ValueError(f"handedness must be 'right' or 'left', got '{v}'")
        return v.lower()


class MarkerLayoutConfig(BaseModel):
    """Marker layout configuration schema.

    Keys are marker IDs as strings (YAML limitation), values are MarkerEntry objects.
    """

    dictionary: str = "DICT_4X4_50"
    body_frame: BodyFrameConfig | None = None
    markers: dict[str, MarkerEntry]
