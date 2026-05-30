"""Left-handed vehicle pose6 export profile.

Converts canonical right-handed PoseResult to vehicle_lh convention.

Vehicle coordinate system:
    vehicle_lh: x=forward, y=right, z=up (LEFT-HANDED)
    camera_link_lh: x=forward, y=right, z=up (LEFT-HANDED)

The 6-DoF output (x, y, z, roll, pitch, yaw):
    - parent: vehicle_lh, x=fwd, y=right, z=up
    - child:  camera_link_lh, x=fwd, y=right, z=up
    - units: meters, degrees

    Yaw computed from forward axis:
        yaw = atan2(fwd_y, fwd_x), yaw > 0 = right.

    Pitch computed from forward axis (NOT from scipy Euler):
        elevation = atan2(fwd_z, sqrt(fwd_x^2 + fwd_y^2))
        pitch = elevation
        pitch > 0 = look upward, pitch < 0 = look downward.

    Roll kept from scipy decomposition of S@R@S (det=+1).

Rotation:
    R_lh = S @ R_rh @ S,  where S = diag([1, -1, 1]).
    Mirrors BOTH source and target, det = +1.
    DO NOT use R_lh = S @ R_rh (det = -1).
"""

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .base import ExportProfile, Pose6Result

S = np.diag([1.0, -1.0, 1.0])


class VehicleLHPose6Exporter(ExportProfile):
    """Export canonical body_rh pose to vehicle_lh pose6."""

    PROFILE_NAME = "vehicle_lh_pose6"
    PARENT_FRAME = "vehicle_lh (x=fwd, y=right, z=up, left-handed)"
    CHILD_FRAME = "camera_link_lh (x=fwd, y=right, z=up, left-handed)"
    EULER_ORDER = "xyz"
    YAW_CONVENTION = "yaw > 0 = right"
    PITCH_CONVENTION = "pitch > 0 = look upward, pitch < 0 = look downward"

    def export(self, T_body_rh_camera_link: np.ndarray) -> Pose6Result:
        """Convert to vehicle_lh pose6.

        Args:
            T_body_rh_camera_link: 4x4, camera_link_rh -> body_rh.
        """
        R_rh = T_body_rh_camera_link[:3, :3]
        t_rh = T_body_rh_camera_link[:3, 3]

        # Position: flip y
        p_lh = S @ t_rh

        # Rotation: mirror both source and target
        R_lh = S @ R_rh @ S

        if not self.validate_rotation(R_lh):
            raise ValueError(
                f"Rotation matrix det={np.linalg.det(R_lh):.6f}, expected +1."
            )

        # Forward axis in vehicle_lh = camera_link_lh x-axis = column 0
        fwd = R_lh[:, 0]
        fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])

        # Yaw from forward axis (horizontal direction)
        yaw_deg = float(np.degrees(np.arctan2(fy, fx)))

        # Pitch from forward axis (elevation)
        elevation_deg = float(np.degrees(
            np.arctan2(fz, np.sqrt(fx ** 2 + fy ** 2))
        ))
        pitch_deg = elevation_deg

        # Roll from scipy
        rpy_rad = ScipyRotation.from_matrix(R_lh).as_euler(
            self.EULER_ORDER, degrees=False,
        )
        roll_deg = float(np.degrees(rpy_rad[0]))

        T_lh = np.eye(4, dtype=np.float64)
        T_lh[:3, :3] = R_lh
        T_lh[:3, 3] = p_lh

        return Pose6Result(
            x=float(p_lh[0]),
            y=float(p_lh[1]),
            z=float(p_lh[2]),
            roll=roll_deg,
            pitch=pitch_deg,
            yaw=yaw_deg,
            T_parent_child=T_lh,
            profile_name=self.PROFILE_NAME,
            parent_frame=self.PARENT_FRAME,
            child_frame=self.CHILD_FRAME,
            euler_order=self.EULER_ORDER,
            yaw_convention=self.YAW_CONVENTION,
            pitch_convention=self.PITCH_CONVENTION,
        )

    def validate_rotation(self, R: np.ndarray) -> bool:
        det = np.linalg.det(R)
        return abs(det - 1.0) < 1e-6


def build_exporter_from_config(config: dict) -> VehicleLHPose6Exporter | None:
    if config is None:
        return None
    t_ok = config.get("type") == "pose6"
    n_ok = config.get("name") in (
        VehicleLHPose6Exporter.PROFILE_NAME,
        "company_vehicle_lh_pose6",  # backward compat for old configs
    )
    if t_ok and n_ok:
        return VehicleLHPose6Exporter()
    return None
