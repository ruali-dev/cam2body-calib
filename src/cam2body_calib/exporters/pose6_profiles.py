"""Pose6 export profiles for different coordinate conventions.

Provides two profiles:
- left_handed:  x=fwd, y=right, z=up (left-handed), yaw positive = right
- right_handed: x=fwd, y=left,  z=up (right-handed, ROS REP-103), yaw positive = left
"""

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .base import ExportProfile, Pose6Result

# Mirror matrix: flips y-axis (left <-> right)
S = np.diag([1.0, -1.0, 1.0])


class LeftHandedPose6Exporter(ExportProfile):
    """Export canonical right-handed pose to left-handed convention.

    x=fwd, y=right, z=up (LEFT-HANDED)
    yaw > 0 = right turn

    Suitable for vehicle coordinate systems where the y-axis points
    to the right side of the vehicle.
    """

    PROFILE_NAME = "left_handed"
    PARENT_FRAME = "vehicle_lh (x=fwd, y=right, z=up, left-handed)"
    CHILD_FRAME = "camera_link_lh (x=fwd, y=right, z=up, left-handed)"
    EULER_ORDER = "xyz"
    YAW_CONVENTION = "yaw > 0 = right"
    PITCH_CONVENTION = "pitch > 0 = up"
    SHORT_DESC = "x=fwd, y=right, z=up (left-handed)"

    def export(self, T_body_rh_camera_link: np.ndarray) -> Pose6Result:
        R_rh = T_body_rh_camera_link[:3, :3]
        t_rh = T_body_rh_camera_link[:3, 3]

        p_lh = S @ t_rh
        R_lh = S @ R_rh @ S

        if not self.validate_rotation(R_lh):
            raise ValueError(f"Rotation det={np.linalg.det(R_lh):.6f}, expected +1.")

        fwd = R_lh[:, 0]
        fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
        yaw_deg = float(np.degrees(np.arctan2(fy, fx)))
        elevation_deg = float(np.degrees(np.arctan2(fz, np.sqrt(fx ** 2 + fy ** 2))))
        pitch_deg = elevation_deg

        rpy_rad = ScipyRotation.from_matrix(R_lh).as_euler(self.EULER_ORDER, degrees=False)
        roll_deg = float(np.degrees(rpy_rad[0]))

        T_lh = np.eye(4, dtype=np.float64)
        T_lh[:3, :3] = R_lh
        T_lh[:3, 3] = p_lh

        return Pose6Result(
            x=float(p_lh[0]), y=float(p_lh[1]), z=float(p_lh[2]),
            roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
            T_parent_child=T_lh,
            profile_name=self.PROFILE_NAME,
            parent_frame=self.PARENT_FRAME,
            child_frame=self.CHILD_FRAME,
            euler_order=self.EULER_ORDER,
            yaw_convention=self.YAW_CONVENTION,
            pitch_convention=self.PITCH_CONVENTION,
        )

    def validate_rotation(self, R: np.ndarray) -> bool:
        return abs(np.linalg.det(R) - 1.0) < 1e-6


class RightHandedPose6Exporter(ExportProfile):
    """Pass-through exporter for right-handed convention.

    x=fwd, y=left, z=up (RIGHT-HANDED, ROS REP-103 compatible)
    yaw > 0 = left turn

    This is the canonical internal frame — no transformation needed.
    Suitable for ROS, standard robotics applications.
    """

    PROFILE_NAME = "right_handed"
    PARENT_FRAME = "body_rh (x=fwd, y=left, z=up, right-handed, ROS REP-103)"
    CHILD_FRAME = "camera_link_rh (x=fwd, y=left, z=up, right-handed)"
    EULER_ORDER = "xyz"
    YAW_CONVENTION = "yaw > 0 = left"
    PITCH_CONVENTION = "pitch > 0 = up"
    SHORT_DESC = "x=fwd, y=left, z=up (right-handed)"

    def export(self, T_body_rh_camera_link: np.ndarray) -> Pose6Result:
        R = T_body_rh_camera_link[:3, :3]
        t = T_body_rh_camera_link[:3, 3]

        if not self.validate_rotation(R):
            raise ValueError(f"Rotation det={np.linalg.det(R):.6f}, expected +1.")

        rpy_rad = ScipyRotation.from_matrix(R).as_euler(self.EULER_ORDER, degrees=False)
        rpy_deg = np.degrees(rpy_rad)

        return Pose6Result(
            x=float(t[0]), y=float(t[1]), z=float(t[2]),
            roll=float(rpy_deg[0]), pitch=float(rpy_deg[1]), yaw=float(rpy_deg[2]),
            T_parent_child=T_body_rh_camera_link.copy(),
            profile_name=self.PROFILE_NAME,
            parent_frame=self.PARENT_FRAME,
            child_frame=self.CHILD_FRAME,
            euler_order=self.EULER_ORDER,
            yaw_convention=self.YAW_CONVENTION,
            pitch_convention=self.PITCH_CONVENTION,
        )

    def validate_rotation(self, R: np.ndarray) -> bool:
        return abs(np.linalg.det(R) - 1.0) < 1e-6


# ── Registry ────────────────────────────────────────────────────────

_exporters = {
    LeftHandedPose6Exporter.PROFILE_NAME: LeftHandedPose6Exporter,
    RightHandedPose6Exporter.PROFILE_NAME: RightHandedPose6Exporter,
    # legacy aliases
    "vehicle_lh_pose6": LeftHandedPose6Exporter,
    "company_vehicle_lh_pose6": LeftHandedPose6Exporter,
}


def list_profiles() -> list[dict]:
    """Return metadata for all available export profiles."""
    seen = set()
    profiles = []
    for cls in [LeftHandedPose6Exporter, RightHandedPose6Exporter]:
        if cls.PROFILE_NAME not in seen:
            seen.add(cls.PROFILE_NAME)
            profiles.append({
                "name": cls.PROFILE_NAME,
                "short_desc": cls.SHORT_DESC,
                "parent_frame": cls.PARENT_FRAME,
                "yaw_convention": cls.YAW_CONVENTION,
            })
    return profiles


def get_exporter(name: str) -> ExportProfile:
    """Get an exporter instance by profile name."""
    cls = _exporters.get(name)
    if cls is None:
        available = [k for k in _exporters if not k.startswith("company_") and not k.startswith("vehicle_lh")]
        raise ValueError(f"Unknown export profile '{name}'. Available: {available}")
    return cls()
