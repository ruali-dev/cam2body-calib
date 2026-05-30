"""Base types for export profiles."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Pose6Result:
    """6-DoF pose exported in a specific coordinate convention.

    Attributes:
        x, y, z: Position in the export frame (meters).
        roll, pitch, yaw: Orientation in the export frame (degrees).
        T_parent_child: 4x4 homogeneous transform (parent <- child).
        profile_name: Name of the export profile.
        parent_frame: Description of the parent (body) frame convention.
        child_frame: Description of the child (camera) frame convention.
        euler_order: Euler angle extraction order.
        yaw_convention: Description of yaw positive direction.
        pitch_convention: Description of pitch positive direction.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    T_parent_child: np.ndarray = field(default_factory=lambda: np.eye(4))
    profile_name: str = ""
    parent_frame: str = ""
    child_frame: str = ""
    euler_order: str = "xyz"
    yaw_convention: str = ""
    pitch_convention: str = ""

    def save_extrinsic_yaml(
        self,
        output_path: str | Path,
        parent_frame_id: str = "body",
        child_frame_id: str = "camera",
    ) -> Path:
        """Save pose to extrinsic YAML format.

        Format:
            child_frame_id: <child>
            header:
              frame_id: <parent>
              stamp: {sec: 0, nanosec: 0}
            rotation:
              roll: <rad>
              pitch: <rad>
              yaw: <rad>
            translation:
              x: <m>
              y: <m>
              z: <m>

        Args:
            output_path: Path to save the YAML file.
            parent_frame_id: Name of the parent (body) frame.
            child_frame_id: Name of the child (camera) frame.

        Returns:
            Path to the saved file.
        """
        import yaml

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "child_frame_id": child_frame_id,
            "header": {
                "frame_id": parent_frame_id,
                "stamp": {"sec": 0, "nanosec": 0},
            },
            "rotation": {
                "roll": float(np.radians(self.roll)),
                "pitch": float(np.radians(self.pitch)),
                "yaw": float(np.radians(self.yaw)),
            },
            "translation": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return path


class ExportProfile(ABC):
    """Abstract base for coordinate convention export profiles.

    Each profile converts a canonical PoseResult (body_rh, right-handed)
    into a custom coordinate convention for downstream consumers.
    """

    @abstractmethod
    def export(self, T_body_rh_camera_link: np.ndarray) -> Pose6Result:
        """Convert canonical pose to custom convention.

        Args:
            T_body_rh_camera_link: 4x4, camera_link_rh->body_rh (right-handed).
                Camera_link_rh: x=fwd, y=left, z=up.

        Returns:
            Pose6Result in the custom convention.
        """
        ...

    @abstractmethod
    def validate_rotation(self, R: np.ndarray) -> bool:
        """Check that rotation matrix has det ~= +1 for RPY extraction."""
        ...
