"""Frame convention adapter: left-handed <-> right-handed body frame.

Vehicle frame (vehicle_lh):       x=forward, y=right, z=up (LEFT-HANDED).
Canonical internal frame (body_rh): x=forward, y=left,  z=up (RIGHT-HANDED).

Conversion: S_lh_to_rh = diag([1, -1, 1])
    P_body_rh = S_lh_to_rh @ P_vehicle_lh
    P_vehicle_lh = S_lh_to_rh @ P_body_rh   (S is its own inverse)

All PnP computation is done in body_rh (right-handed).
Output is given in both body_rh and vehicle_lh.
"""

from dataclasses import dataclass

import numpy as np

# Scaling matrix: vehicle_lh -> body_rh.
# Flips the y-axis (right -> left). S = S^-1.
S_LH_TO_RH = np.diag([1.0, -1.0, 1.0])


@dataclass
class FrameConvention:
    """Body frame convention descriptor."""

    name: str = "body"
    handedness: str = "right"  # "right" or "left"
    x_axis: str = "forward"
    y_axis: str = "left"
    z_axis: str = "up"

    @classmethod
    def from_config(cls, config: dict | None) -> "FrameConvention":
        """Create from YAML body_frame section."""
        if config is None:
            return cls()
        return cls(
            name=config.get("name", "body"),
            handedness=config.get("handedness", "right"),
            x_axis=config.get("x_axis", "forward"),
            y_axis=config.get("y_axis", "left"),
            z_axis=config.get("z_axis", "up"),
        )

    @property
    def is_left_handed(self) -> bool:
        return self.handedness.lower() == "left"

    @property
    def lh_label(self) -> str:
        """Human-readable label for the left-handed frame."""
        return f"{self.name}_lh" if self.is_left_handed else self.name


# ── Point / vector conversion ─────────────────────────────────────

def points_lh_to_rh(points_lh: np.ndarray) -> np.ndarray:
    """Convert points from left-handed vehicle frame to right-handed body frame.
    points_lh: (N, 3)
    """
    return points_lh @ np.diag([1.0, -1.0, 1.0])


def points_rh_to_lh(points_rh: np.ndarray) -> np.ndarray:
    """Convert points from right-handed body frame to left-handed vehicle frame.
    points_rh: (N, 3)
    """
    return points_rh @ np.diag([1.0, -1.0, 1.0])  # S = S^-1


# ── Position conversion ───────────────────────────────────────────

def position_rh_to_lh(pos_rh: np.ndarray) -> np.ndarray:
    """Convert position from body_rh to vehicle_lh.
    pos_rh: (3,) array [x_fwd, y_left, z_up]
    Returns: (3,) array [x_fwd, y_right, z_up]
    """
    return S_LH_TO_RH @ pos_rh


def position_lh_to_rh(pos_lh: np.ndarray) -> np.ndarray:
    """Convert position from vehicle_lh to body_rh."""
    return S_LH_TO_RH @ pos_lh  # same, S = S^-1


# ── Rotation / transform conversion ───────────────────────────────

def rotation_rh_to_lh(R_body_rh: np.ndarray) -> np.ndarray:
    """Convert rotation matrix from body_rh to vehicle_lh.

    R_body_rh maps camera->body_rh:  X_body_rh = R_body_rh @ X_cam
    R_vehicle maps camera->vehicle:  X_vehicle = S @ X_body_rh = S @ R_body_rh @ X_cam
    So R_vehicle = S @ R_body_rh
    """
    return S_LH_TO_RH @ R_body_rh


def rotation_lh_to_rh(R_vehicle: np.ndarray) -> np.ndarray:
    """Convert rotation matrix from vehicle_lh to body_rh."""
    return S_LH_TO_RH @ R_vehicle  # S = S^-1


def transform_rh_to_lh(T_body_rh: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform from body_rh to vehicle_lh.

    T_body_rh = [R, t] maps camera->body_rh.
    T_vehicle = [S@R, S@t] maps camera->vehicle_lh.
    """
    T_lh = T_body_rh.copy()
    T_lh[:3, :3] = S_LH_TO_RH @ T_body_rh[:3, :3]
    T_lh[:3, 3] = S_LH_TO_RH @ T_body_rh[:3, 3]
    return T_lh


# ── Camera axis direction interpretation ──────────────────────────

def camera_forward_interpretation_lh(fwd_vehicle: np.ndarray) -> str:
    """Interpret camera forward axis direction in vehicle_lh frame.

    vehicle_lh: x=forward, y=right, z=up.
    fwd_vehicle: (3,) unit vector of camera z-axis in vehicle frame.
    """
    fx, fy, fz = fwd_vehicle
    parts = []
    if abs(fx) > 0.05:
        parts.append("前" if fx > 0 else "后")
    if abs(fy) > 0.05:
        parts.append("右" if fy > 0 else "左")
    if abs(fz) > 0.05:
        parts.append("上" if fz > 0 else "下")
    return "".join(parts) if parts else "—"


def yaw_interpretation_lh(yaw_deg: float) -> str:
    """Interpret yaw in vehicle_lh frame (x=forward, y=right, z=up).

    yaw > 0: camera rotates toward +y (right), so it looks right/右前方.
    yaw < 0: camera rotates toward -y (left), so it looks left/左前方.
    """
    if abs(yaw_deg) < 2.0:
        return "正前方"
    if yaw_deg > 0:
        return f"右前方 ({yaw_deg:.1f} deg)"
    else:
        return f"左前方 ({abs(yaw_deg):.1f} deg)"
