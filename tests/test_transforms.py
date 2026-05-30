"""Tests for geometry transform utilities.

Verifies that 4x4 homogeneous transforms are correctly inverted
and that the round-trip is consistent.
"""

import numpy as np

from cam2body_calib.geometry.transforms import (
    invert_pose,
    pose_4x4_to_rvec_tvec,
    rvec_tvec_to_4x4,
    transform_points,
)


def test_invert_pose_identity():
    """Inverting identity should yield identity."""
    T = np.eye(4)
    T_inv = invert_pose(T)
    np.testing.assert_array_almost_equal(T_inv, np.eye(4))


def test_invert_pose_roundtrip():
    """T_inv @ T = I and T @ T_inv = I."""
    # Create a random rigid transform
    rng = np.random.RandomState(42)

    # Random rotation via random rvec
    axis = rng.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = rng.uniform(0, np.pi)
    rvec = (axis * angle).reshape(3, 1)

    # Random translation
    tvec = rng.uniform(-5, 5, size=(3, 1))

    # Build 4x4
    T = rvec_tvec_to_4x4(rvec, tvec)
    T_inv = invert_pose(T)

    # Check T_inv @ T = I
    prod1 = T_inv @ T
    np.testing.assert_array_almost_equal(prod1, np.eye(4), decimal=10)

    # Check T @ T_inv = I
    prod2 = T @ T_inv
    np.testing.assert_array_almost_equal(prod2, np.eye(4), decimal=10)


def test_invert_pose_semantics():
    """If T maps A->B, then T_inv maps B->A. Verify with point transform."""
    # Known rvec, tvec representing some T_cam_body
    axis = np.array([0.0, 0.0, 1.0])  # rotation around z
    angle = np.pi / 4  # 45 degrees
    rvec = (axis * angle).reshape(3, 1)
    tvec = np.array([1.0, 2.0, 3.0]).reshape(3, 1)

    T_cam_body = rvec_tvec_to_4x4(rvec, tvec)
    T_body_cam = invert_pose(T_cam_body)

    # A point in body frame
    X_body = np.array([5.0, 0.0, 0.0])  # 5m forward

    # Transform to camera frame
    X_cam = transform_points(T_cam_body, X_body.reshape(1, 3)).ravel()

    # Transform back to body frame
    X_body_back = transform_points(T_body_cam, X_cam.reshape(1, 3)).ravel()

    np.testing.assert_array_almost_equal(X_body, X_body_back, decimal=10)


def test_rvec_tvec_roundtrip():
    """pose_4x4_to_rvec_tvec should be inverse of rvec_tvec_to_4x4 (up to rotation)."""
    rng = np.random.RandomState(99)
    axis = rng.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = rng.uniform(0.1, np.pi - 0.1)  # avoid 0 and pi
    rvec = (axis * angle).reshape(3, 1)
    tvec = rng.uniform(-10, 10, size=(3, 1))

    T = rvec_tvec_to_4x4(rvec, tvec)
    rvec2, tvec2 = pose_4x4_to_rvec_tvec(T)

    # Rebuild from recovered params
    T2 = rvec_tvec_to_4x4(rvec2, tvec2)

    np.testing.assert_array_almost_equal(T, T2, decimal=10)
