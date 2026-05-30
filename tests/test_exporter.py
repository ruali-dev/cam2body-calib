"""Tests for the vehicle LH pose6 exporter."""

import numpy as np

from cam2body_calib.exporters.vehicle_lh_pose6 import (
    VehicleLHPose6Exporter,
    S,
)


def make_exporter():
    return VehicleLHPose6Exporter()


def identity_rh_pose():
    """Camera perfectly aligned with body_rh: T_body_rh_camera_link = I."""
    return np.eye(4)


def test_det_s_r_s_is_positive_one():
    """S @ R @ S has det = +1 for any proper rotation R."""
    from scipy.spatial.transform import Rotation as R
    rng = np.random.RandomState(42)
    for _ in range(10):
        R_mat = R.random(random_state=rng).as_matrix()
        R_lh = S @ R_mat @ S
        det = np.linalg.det(R_lh)
        assert abs(det - 1.0) < 1e-10, f"det(S@R@S) = {det}, expected +1"


def test_det_s_r_is_negative_one():
    """S @ R has det = -1, cannot extract RPY."""
    from scipy.spatial.transform import Rotation as R
    rng = np.random.RandomState(42)
    for _ in range(10):
        R_mat = R.random(random_state=rng).as_matrix()
        R_wrong = S @ R_mat  # det = -1
        det = np.linalg.det(R_wrong)
        assert abs(det + 1.0) < 1e-10, f"det(S@R) = {det}, expected -1"


def test_identity_pose_exports_zero():
    """Identity pose in body_rh -> zero position and RPY in company LH."""
    exporter = make_exporter()
    T = identity_rh_pose()
    pose6 = exporter.export(T)

    assert abs(pose6.x) < 1e-10
    assert abs(pose6.y) < 1e-10
    assert abs(pose6.z) < 1e-10
    assert abs(pose6.roll) < 1e-10
    assert abs(pose6.pitch) < 1e-10
    assert abs(pose6.yaw) < 1e-10
    # Also verify the exported T is identity
    np.testing.assert_array_almost_equal(pose6.T_parent_child, np.eye(4))


def test_position_y_flips():
    """Camera at body_rh (1, 2, 3) -> vehicle_lh (1, -2, 3)."""
    exporter = make_exporter()
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    pose6 = exporter.export(T)

    assert abs(pose6.x - 1.0) < 1e-10   # x unchanged
    assert abs(pose6.y - (-2.0)) < 1e-10  # y flipped
    assert abs(pose6.z - 3.0) < 1e-10   # z unchanged


def test_yaw_left_in_rh_becomes_right_in_lh():
    """body_rh yaw +30 (left) -> vehicle_lh yaw should be flipped sign.

    In body_rh: yaw +30 means camera looks LEFT.
    In vehicle_lh: y=right, so looking LEFT = negative yaw.
    The S@R@S transform should produce yaw ~ -30 deg.
    """
    from cam2body_calib.geometry.rotations import rpy_to_rotation_matrix

    exporter = make_exporter()
    rpy_rh = np.radians([0.0, 0.0, 30.0])  # yaw +30 = look left
    R_rh = rpy_to_rotation_matrix(rpy_rh)
    T = np.eye(4)
    T[:3, :3] = R_rh

    pose6 = exporter.export(T)

    # In LH: left is negative y, so looking left = negative yaw
    assert pose6.yaw < 0, f"Expected negative yaw (left), got {pose6.yaw}"
    assert abs(pose6.yaw - (-30.0)) < 0.01


def test_yaw_right_in_rh_becomes_right_in_lh():
    """body_rh yaw -30 (right) -> vehicle_lh yaw ~ +30 (right)."""
    from cam2body_calib.geometry.rotations import rpy_to_rotation_matrix

    exporter = make_exporter()
    rpy_rh = np.radians([0.0, 0.0, -30.0])  # yaw -30 = look right
    R_rh = rpy_to_rotation_matrix(rpy_rh)
    T = np.eye(4)
    T[:3, :3] = R_rh

    pose6 = exporter.export(T)

    # In LH: right is positive y, so looking right = positive yaw
    assert pose6.yaw > 0, f"Expected positive yaw (right), got {pose6.yaw}"
    assert abs(pose6.yaw - 30.0) < 0.01


def _make_T_from_R(R_lh):
    """Helper: create a 4x4 with given R (already in LH frame) and zero translation."""
    T = np.eye(4)
    T[:3, :3] = R_lh
    return T


def _R_lh_from_yaw_pitch(yaw_deg, pitch_deg):
    """Build R_lh = Rz(yaw) @ Ry(pitch) in vehicle_lh frame."""
    from scipy.spatial.transform import Rotation as R
    return R.from_euler("zy", [yaw_deg, pitch_deg], degrees=True).as_matrix()


def test_forward_straight_ahead():
    """fwd=[1,0,0] -> yaw=0, pitch=0."""
    R_lh = np.eye(3)  # identity → fwd = [1,0,0]
    # Need to pass through the exporter. But exporter expects T_body_rh, not R_lh.
    # R_lh = S @ R_rh @ S → R_rh = S @ R_lh @ S (same transform, S=S^-1)
    R_rh = S @ R_lh @ S
    T_body = _make_T_from_R(R_rh)
    pose6 = make_exporter().export(T_body)
    assert abs(pose6.yaw) < 1e-10
    assert abs(pose6.pitch) < 1e-10


def test_yaw_10_right():
    """fwd=[cos10, sin10, 0] -> yaw=+10, pitch=0."""
    R_lh = _R_lh_from_yaw_pitch(10, 0)  # yaw +10 = right
    R_rh = S @ R_lh @ S
    pose6 = make_exporter().export(_make_T_from_R(R_rh))
    assert abs(pose6.yaw - 10.0) < 1e-10
    assert abs(pose6.pitch) < 1e-10


def test_pitch_look_up_10():
    """fwd=[cos10, 0, sin10] -> look up 10 deg -> company pitch = +10.

    Ry(-10) in LH: forward z = sin(10) > 0 = looking UP.
    """
    R_lh = _R_lh_from_yaw_pitch(0, -10)  # Ry(-10): forward tilts UP
    R_rh = S @ R_lh @ S
    pose6 = make_exporter().export(_make_T_from_R(R_rh))
    assert abs(pose6.yaw) < 1e-10
    assert abs(pose6.pitch - 10.0) < 1e-10  # pitch = elevation = +10


def test_pitch_look_down_10():
    """fwd=[cos10, 0, -sin10] -> look down 10 deg -> company pitch = -10.

    Ry(+10) in LH: forward z = -sin(10) < 0 = looking DOWN.
    """
    R_lh = _R_lh_from_yaw_pitch(0, 10)  # Ry(+10): forward tilts DOWN
    R_rh = S @ R_lh @ S
    pose6 = make_exporter().export(_make_T_from_R(R_rh))
    assert abs(pose6.yaw) < 1e-10
    assert abs(pose6.pitch - (-10.0)) < 1e-10  # pitch = elevation = -10


def test_validate_rejects_left_handed():
    """validate_rotation should reject S @ R (det = -1)."""
    exporter = make_exporter()
    R = np.eye(3)
    R_wrong = S @ R  # det = -1
    assert not exporter.validate_rotation(R_wrong)


def test_validate_accepts_right_handed():
    """validate_rotation should accept S @ R @ S (det = +1)."""
    exporter = make_exporter()
    R = np.eye(3)
    R_good = S @ R @ S  # det = +1
    assert exporter.validate_rotation(R_good)
