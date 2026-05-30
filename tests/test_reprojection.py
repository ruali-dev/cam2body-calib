"""Integration test for PnP solver with synthetic data.

Creates a known T_cam_body (body->camera transform), generates virtual 3D
marker corners in body frame, projects them to 2D, then runs PnPSolver to
recover the pose. Verifies the recovered pose matches ground truth.

IMPORTANT COORDINATE NOTE:
- Body frame: x=forward, y=left, z=up
- Camera frame: x=right, y=down, z=forward
- The base rotation between them is:
  R_cam_body = [[0,-1,0], [0,0,-1], [1,0,0]]
  (body x→cam z, body y→cam -x, body z→cam -y)
"""

import cv2
import numpy as np

from cam2body_calib.camera.model import CameraModel
from cam2body_calib.estimation.pnp_solver import PnPSolver
from cam2body_calib.estimation.reprojection import compute_reprojection_errors
from cam2body_calib.geometry.rotations import rotation_matrix_to_rpy
from cam2body_calib.geometry.transforms import (
    invert_pose,
    pose_4x4_to_rvec_tvec,
)


def make_test_camera():
    """Create a realistic camera model for testing."""
    return CameraModel(
        K=np.array([
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
        D=np.array([-0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float64),
        image_width=1280,
        image_height=720,
        camera_name="test_camera",
    )


def make_object_points_body():
    """Create synthetic 3D points in body frame.

    Body frame: x forward, y left, z up.

    Three markers placed at body x=2.0m (2 meters ahead), spanning left/right,
    at approximately camera eye level to ensure projections fall within image.
    Camera is at body (0.5, 0, 1.5), so markers are ~1.5m in front of camera
    and near the camera's vertical level (z=1.2-1.8, spanning ~0.6m vertically).
    """
    markers_corners = []

    # Marker 0: center, (x=2.0, y=-0.4..0.4, z=1.3..1.8)
    markers_corners.append(np.array([
        [2.0,  0.4, 1.8],  # top-left
        [2.0, -0.4, 1.8],  # top-right
        [2.0, -0.4, 1.3],  # bottom-right
        [2.0,  0.4, 1.3],  # bottom-left
    ], dtype=np.float64))

    # Marker 1: left side (body y positive = left)
    markers_corners.append(np.array([
        [2.0,  1.1, 1.8],
        [2.0,  0.5, 1.8],
        [2.0,  0.5, 1.3],
        [2.0,  1.1, 1.3],
    ], dtype=np.float64))

    # Marker 2: right side (body y negative = right)
    markers_corners.append(np.array([
        [2.0, -0.5, 1.8],
        [2.0, -1.1, 1.8],
        [2.0, -1.1, 1.3],
        [2.0, -0.5, 1.3],
    ], dtype=np.float64))

    # Marker 3: farther wall for depth variation (x=2.8, improves PnP stability)
    markers_corners.append(np.array([
        [2.8,  0.3, 1.6],
        [2.8, -0.3, 1.6],
        [2.8, -0.3, 1.2],
        [2.8,  0.3, 1.2],
    ], dtype=np.float64))

    return np.vstack(markers_corners)


def make_ground_truth_pose():
    """Create a physically meaningful camera extrinsics.

    Camera is mounted at body position (0.5, 0, 1.5) — 0.5m forward of
    body origin, centered laterally, 1.5m above ground.

    Camera orientation: looking forward (body +x) with a slight downward
    pitch of ~0.15 rad so the markers (at z=1.2-1.8) are in the image center.

    Construction:
    1. R_base: the static 90-degree body→camera rotation
    2. R_pitch: extra camera tilt about camera x-axis
    3. R_cam_body = R_pitch @ R_base
    4. t_cam_body = -R_cam_body @ camera_position_in_body

    Returns:
        T_cam_body, T_body_cam
    """
    # Base rotation: body-frame axes → camera-frame axes
    #   body x (forward) → camera z (forward)
    #   body y (left)    → camera -x (right)
    #   body z (up)      → camera -y (down)
    R_base = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Extra pitch: camera tilted down by 0.15 rad (rotation about camera x-axis)
    # Positive pitch = camera looks UP, so negative = camera looks DOWN
    pitch = -0.15
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cp, -sp],
        [0.0, sp, cp],
    ], dtype=np.float64)

    # Compose: first apply R_base (body→cam), then R_pitch (cam→tilted cam)
    R_cam_body = R_pitch @ R_base

    # Camera position in body frame
    cam_pos_body = np.array([0.5, 0.0, 1.5], dtype=np.float64)

    # t_cam_body = position of body origin in camera frame
    t_cam_body = -R_cam_body @ cam_pos_body

    T_cam_body = np.eye(4, dtype=np.float64)
    T_cam_body[:3, :3] = R_cam_body
    T_cam_body[:3, 3] = t_cam_body

    # T_body_cam = camera pose in body frame (the output we ultimately want)
    T_body_cam = invert_pose(T_cam_body)

    return T_cam_body, T_body_cam


def project_points(object_points_body, T_cam_body, camera):
    """Project 3D body-frame points into 2D image using known T_cam_body."""
    rvec, tvec = pose_4x4_to_rvec_tvec(T_cam_body)
    image_points, _ = cv2.projectPoints(
        object_points_body.reshape(-1, 3), rvec, tvec, camera.K, camera.D,
    )
    return image_points.reshape(-1, 2)


# ── Tests ────────────────────────────────────────────────────────


def test_pnp_solver_recovers_known_pose():
    """Noise-free: recovered pose should match ground truth exactly."""
    camera = make_test_camera()
    object_points_body = make_object_points_body()
    T_cam_body_gt, T_body_cam_gt = make_ground_truth_pose()

    # Project points to 2D (noise-free)
    image_points = project_points(object_points_body, T_cam_body_gt, camera)

    # Verify projections are within image bounds
    assert np.all(image_points[:, 0] >= 0), f"min u = {image_points[:, 0].min()}"
    assert np.all(image_points[:, 0] < camera.image_width), f"max u = {image_points[:, 0].max()}"
    assert np.all(image_points[:, 1] >= 0), f"min v = {image_points[:, 1].min()}"
    assert np.all(image_points[:, 1] < camera.image_height), f"max v = {image_points[:, 1].max()}"

    # Run PnP solver
    solver = PnPSolver(camera=camera, ransac_threshold=1.0, refine=True)
    result = solver.solve(object_points_body, image_points)

    assert result.success, f"PnP solver failed: {result.message}"

    # Recovered transforms should match ground truth (high precision, noise-free)
    np.testing.assert_array_almost_equal(
        result.T_cam_body, T_cam_body_gt, decimal=4,
    )
    np.testing.assert_array_almost_equal(
        result.T_body_camera_optical, T_body_cam_gt, decimal=4,
    )

    # Camera position in body frame
    gt_position = T_body_cam_gt[:3, 3]
    np.testing.assert_array_almost_equal(result.position_body, gt_position, decimal=4)

    # RPY should match
    gt_rpy = rotation_matrix_to_rpy(T_body_cam_gt[:3, :3])
    np.testing.assert_array_almost_equal(result.rpy_optical_body_cam, gt_rpy, decimal=3)

    # Reprojection errors should be negligible (noise-free)
    assert result.reprojection_stats is not None
    assert result.reprojection_stats.mean_error < 0.1, (
        f"Mean reprojection error too high: {result.reprojection_stats.mean_error:.4f}"
    )
    assert result.reprojection_stats.inlier_count == len(object_points_body)


def test_pnp_solver_with_noise():
    """Test robustness: pose should still be reasonable with 1px Gaussian noise."""
    rng = np.random.RandomState(7)

    camera = make_test_camera()
    object_points_body = make_object_points_body()
    T_cam_body_gt, T_body_cam_gt = make_ground_truth_pose()

    image_points = project_points(object_points_body, T_cam_body_gt, camera)
    image_points_noisy = (
        image_points + rng.randn(*image_points.shape).astype(np.float64) * 1.0
    )

    solver = PnPSolver(camera=camera, ransac_threshold=3.0, refine=True)
    result = solver.solve(object_points_body, image_points_noisy)

    assert result.success, f"PnP solver failed on noisy data: {result.message}"

    # Position should be within ~3 cm with 1px noise
    gt_position = T_body_cam_gt[:3, 3]
    position_error = np.linalg.norm(result.position_body - gt_position)
    assert position_error < 0.05, (
        f"Position error: {position_error:.4f} m"
    )

    # Rotation should be within ~3 degrees
    gt_rpy = rotation_matrix_to_rpy(T_body_cam_gt[:3, :3])
    rpy_error = np.linalg.norm(result.rpy_optical_body_cam - gt_rpy)
    assert rpy_error < 0.06, (
        f"Rotation error: {np.degrees(rpy_error):.2f} deg"
    )

    # Mean reprojection error should be on the order of the noise
    assert result.reprojection_stats is not None
    assert result.reprojection_stats.mean_error < 2.0, (
        f"Mean reprojection error: {result.reprojection_stats.mean_error:.3f} px"
    )


def test_reprojection_error_computation():
    """compute_reprojection_errors should return near-zero for noise-free data."""
    camera = make_test_camera()
    object_points_body = make_object_points_body()
    T_cam_body_gt, _ = make_ground_truth_pose()

    rvec, tvec = pose_4x4_to_rvec_tvec(T_cam_body_gt)
    image_points = project_points(object_points_body, T_cam_body_gt, camera)

    stats = compute_reprojection_errors(
        object_points_body, image_points, rvec, tvec,
        camera.K, camera.D,
    )

    assert stats.mean_error < 1e-6
    assert stats.max_error < 1e-6
    assert stats.total_points == len(object_points_body)
    assert stats.inlier_count == stats.total_points


def test_t_body_cam_position_is_reasonable():
    """Sanity check: T_body_cam position should recover camera mounting position."""
    _, T_body_cam_gt = make_ground_truth_pose()
    pos = T_body_cam_gt[:3, 3]

    # Camera should be at approximately (0.5, 0, 1.5)
    np.testing.assert_allclose(pos[0], 0.5, atol=0.01)  # forward
    np.testing.assert_allclose(pos[1], 0.0, atol=0.01)  # lateral
    np.testing.assert_allclose(pos[2], 1.5, atol=0.01)  # height


# ── Fisheye camera tests ─────────────────────────────────────────


def make_fisheye_camera():
    """Create a synthetic fisheye camera model for testing."""
    return CameraModel(
        K=np.array([
            [260.0, 0.0, 480.0],
            [0.0, 370.0, 380.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
        D=np.array([0.12, -0.03, -0.005, 0.002], dtype=np.float64),
        image_width=960,
        image_height=768,
        camera_name="test_fisheye",
        model="fisheye",
    )


def test_fisheye_camera_from_config():
    """CameraModel.from_config should parse model=fisheye correctly."""
    config = {
        "camera_name": "test",
        "image_width": 960,
        "image_height": 768,
        "K": [[260.0, 0.0, 480.0], [0.0, 370.0, 380.0], [0.0, 0.0, 1.0]],
        "D": [0.12, -0.03, -0.005, 0.002],
        "model": "fisheye",
    }
    cam = CameraModel.from_config(config)
    assert cam.model == "fisheye"
    assert len(cam.D) == 4
    np.testing.assert_array_equal(cam.K[2], [0.0, 0.0, 1.0])


def test_fisheye_undistort_returns_pinhole_model():
    """After fisheye undistortion, the returned model should be pinhole with D=zeros."""
    cam = make_fisheye_camera()
    # Create a synthetic image (colored gradient)
    rng = np.random.RandomState(42)
    img = (rng.rand(cam.image_height, cam.image_width, 3) * 255).astype(np.uint8)

    undistorted, new_cam = cam.undistort_image(img, balance=0.0)

    assert isinstance(undistorted, np.ndarray)
    assert undistorted.ndim == 3
    assert new_cam.model == "pinhole"
    assert np.all(new_cam.D == 0.0)
    assert new_cam.K.shape == (3, 3)
    # The rectified K should differ from the original fisheye K
    assert not np.allclose(new_cam.K, cam.K), "Rectified K should differ from fisheye K"


def test_fisheye_undistort_balance():
    """balance=0 should produce a smaller valid area than balance=1."""
    cam = make_fisheye_camera()
    rng = np.random.RandomState(42)
    img = (rng.rand(cam.image_height, cam.image_width, 3) * 255).astype(np.uint8)

    _, cam0 = cam.undistort_image(img, balance=0.0)
    _, cam1 = cam.undistort_image(img, balance=1.0)

    # balance=0 crops to valid pixels, resulting in smaller/same image or different K
    # balance=1 preserves all pixels, typically with different K parameters
    # The focal lengths and principal points will differ between the two
    assert not np.allclose(cam0.K, cam1.K), (
        "balance=0 and balance=1 should produce different rectified K matrices"
    )


def test_fisheye_project_then_undistort():
    """3D points projected with fisheye → undistort → re-project with new K."""
    cam = make_fisheye_camera()

    # Define 3D points in front of the camera (in camera frame)
    # Camera looks along +z, points spread in x,y at z=2.0
    rng = np.random.RandomState(99)
    object_points = np.zeros((10, 3), dtype=np.float64)
    object_points[:, 0] = rng.uniform(-0.5, 0.5, 10)  # x: -0.5 to 0.5
    object_points[:, 1] = rng.uniform(-0.4, 0.4, 10)  # y: -0.4 to 0.4
    object_points[:, 2] = 2.0  # z: 2m ahead

    # Project with fisheye model (points are in camera frame, rvec=0, tvec=0)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    distorted_pts, _ = cv2.fisheye.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, cam.K, cam.D,
    )
    distorted_pts = distorted_pts.reshape(-1, 2)

    # All distorted points should be within the image
    assert np.all(distorted_pts[:, 0] >= 0)
    assert np.all(distorted_pts[:, 0] < cam.image_width)
    assert np.all(distorted_pts[:, 1] >= 0)
    assert np.all(distorted_pts[:, 1] < cam.image_height)

    # Now simulate what happens after image undistortion:
    # Create an image, undistort it, get new K.
    img = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
    _, new_cam = cam.undistort_image(img, balance=0.0)

    # After rectification, the projection should use new_K and D=zeros (pinhole).
    # The same 3D points should re-project to valid image coordinates.
    rectified_pts, _ = cv2.projectPoints(
        object_points.reshape(-1, 3), rvec, tvec, new_cam.K, None,
    )
    rectified_pts = rectified_pts.reshape(-1, 2)

    assert np.all(rectified_pts[:, 0] >= 0)
    assert np.all(rectified_pts[:, 0] < new_cam.image_width)
    assert np.all(rectified_pts[:, 1] >= 0)
    assert np.all(rectified_pts[:, 1] < new_cam.image_height)


def test_pinhole_no_distortion_is_noop():
    """Pinhole camera with D=zeros should return the same K after undistortion."""
    cam = CameraModel(
        K=np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        image_width=1280,
        image_height=720,
        camera_name="zero_dist",
        model="pinhole",
    )
    rng = np.random.RandomState(1)
    img = (rng.rand(720, 1280, 3) * 255).astype(np.uint8)
    undistorted, new_cam = cam.undistort_image(img)

    assert new_cam.model == "pinhole"
    assert np.all(new_cam.D == 0.0)
    np.testing.assert_array_almost_equal(new_cam.K, cam.K)


# ── Camera link frame tests ─────────────────────────────────────


def test_link_frame_rpy_zero_when_aligned():
    """When camera_link is perfectly aligned with body, RPY should be (0,0,0).

    "Perfectly aligned" means camera optical axes map to body axes with
    NO extra tilt/pan/roll — only the static 90° optical→link rotation.

    T_cam_body has R = [[0,-1,0],[0,0,-1],[1,0,0]] (body→optical).
    T_body_camera_optical = inv(T_cam_body).
    T_body_camera_link = convert_optical_to_link(T_body_camera_optical).
    Camera_link RPY extracted from T_body_camera_link should be (0,0,0).
    """
    from cam2body_calib.geometry.rotations import (
        convert_optical_to_link,
        matrix_to_rpy_scipy,
    )
    from cam2body_calib.geometry.transforms import invert_pose

    # body→optical for a perfectly aligned camera:
    #   body x (fwd)    → optical z (fwd)
    #   body y (left)   → optical -x (right = -left)
    #   body z (up)     → optical -y (down = -up)
    R_cam_body = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    t_cam_body = np.array([0.0, 0.0, 5.0])  # body origin 5m ahead of camera

    T_cam_body = np.eye(4)
    T_cam_body[:3, :3] = R_cam_body
    T_cam_body[:3, 3] = t_cam_body

    T_body_optical = invert_pose(T_cam_body)
    T_body_link = convert_optical_to_link(T_body_optical)

    rpy_link = matrix_to_rpy_scipy(T_body_link[:3, :3], degrees=True)

    # camera_link RPY should be ~(0, 0, 0) when perfectly aligned
    np.testing.assert_array_almost_equal(rpy_link, [0.0, 0.0, 0.0], decimal=6)

    # Position should be camera at body (0, 0, 5) — the inverse of t_cam_body
    expected_pos = -R_cam_body.T @ t_cam_body
    np.testing.assert_array_almost_equal(T_body_link[:3, 3], expected_pos)
    np.testing.assert_array_almost_equal(T_body_optical[:3, 3], expected_pos)


def test_link_frame_rpy_with_mount_rotation():
    """Camera tilted down 10° about its own right axis: link RPY should show it."""
    from cam2body_calib.geometry.rotations import (
        convert_optical_to_link,
        matrix_to_rpy_scipy,
    )
    from cam2body_calib.geometry.transforms import invert_pose

    # Base body→optical rotation
    R_cam_body_base = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Camera tilted down 10°: rotation about camera's own x (right) axis
    tilt_rad = np.radians(-10.0)  # Rx(-10): camera looks DOWN
    ct, st = np.cos(tilt_rad), np.sin(tilt_rad)
    R_tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, ct, -st],
        [0.0, st, ct],
    ], dtype=np.float64)

    # Compose: R_cam_body = R_tilt @ R_cam_body_base
    R_cam_body = R_tilt @ R_cam_body_base
    t = np.array([1.0, 0.0, 2.0])

    T_cam_body = np.eye(4)
    T_cam_body[:3, :3] = R_cam_body
    T_cam_body[:3, 3] = t

    T_body_optical = invert_pose(T_cam_body)
    T_body_link = convert_optical_to_link(T_body_optical)

    rpy_link = matrix_to_rpy_scipy(T_body_link[:3, :3], degrees=True)

    # With a 10° tilt about camera x (right), the link RPY roll should be ~10°
    # (roll is about body x = forward, which in link frame is the same axis)
    # Optical x (right) -> link -y. 10 deg about optical x = -10 deg pitch in link.
    np.testing.assert_allclose(rpy_link[0], 0.0, atol=0.01)   # roll ~0
    np.testing.assert_allclose(rpy_link[1], -10.0, atol=0.01)  # pitch ~-10
    np.testing.assert_allclose(rpy_link[2], 0.0, atol=0.01)   # yaw ~0


# ── Left-handed frame tests ─────────────────────────────────────


def test_lh_to_rh_point_conversion():
    """Vehicle_lh (x=fwd,y=right,z=up) -> body_rh (x=fwd,y=left,z=up)."""
    from cam2body_calib.geometry.frames import points_lh_to_rh, points_rh_to_lh

    pts_lh = np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.0]])
    pts_rh = points_lh_to_rh(pts_lh)
    # y should be flipped
    np.testing.assert_array_equal(pts_rh[:, 0], pts_lh[:, 0])   # x unchanged
    np.testing.assert_array_equal(pts_rh[:, 1], -pts_lh[:, 1])  # y flipped
    np.testing.assert_array_equal(pts_rh[:, 2], pts_lh[:, 2])   # z unchanged
    # Round-trip
    pts_back = points_rh_to_lh(pts_rh)
    np.testing.assert_array_almost_equal(pts_back, pts_lh)


def test_lh_rh_rotation():
    """R_lh = S @ R_rh for a simple case."""
    from cam2body_calib.geometry.frames import rotation_lh_to_rh, rotation_rh_to_lh

    # Identity rotation: camera perfectly aligned with body_rh
    R_rh = np.eye(3)
    R_lh = rotation_rh_to_lh(R_rh)
    expected = np.diag([1.0, -1.0, 1.0])  # S matrix
    np.testing.assert_array_almost_equal(R_lh, expected)
    # Round-trip
    np.testing.assert_array_almost_equal(rotation_lh_to_rh(R_lh), R_rh)


def test_lh_rpy_yaw_positive_is_right():
    """In vehicle_lh (x=fwd,y=right,z=up), yaw>0 means looking right."""
    from cam2body_calib.geometry.frames import (
        rotation_rh_to_lh,
    )
    from cam2body_calib.geometry.rotations import (
        rpy_to_rotation_matrix,
    )

    # body_rh: camera yaw +30 deg (looking left in body_rh convention)
    rpy_rh = np.radians([0.0, 0.0, 30.0])
    R_rh = rpy_to_rotation_matrix(rpy_rh)
    R_lh = rotation_rh_to_lh(R_rh)

    # R_lh = S @ R_rh has det = -1 (left-handed). Cannot extract RPY via scipy.
    assert np.linalg.det(R_lh) < 0  # LH rotation matrix has negative determinant

    # Camera forward = camera_link x-axis = column 0.
    # body_rh yaw +30: forward = [cos30, sin30, 0] = looks forward-left.
    # vehicle_lh: S @ [cos30, sin30, 0] = [cos30, -sin30, 0].
    # In LH (y=right): negative y = left. Camera looks forward-left. Correct.
    fwd_lh = R_lh[:, 0]
    assert fwd_lh[0] > 0.8     # mainly forward
    assert fwd_lh[1] < 0       # negative y in LH = left
    assert abs(fwd_lh[2]) < 0.1


def test_frame_convention_from_config():
    """FrameConvention.from_config parses body_frame section correctly."""
    from cam2body_calib.geometry.frames import FrameConvention

    fc = FrameConvention.from_config({
        "name": "vehicle",
        "convention": "x_forward_y_right_z_up",
        "handedness": "left",
    })
    assert fc.is_left_handed
    assert fc.name == "vehicle"
    assert fc.lh_label == "vehicle_lh"

    fc2 = FrameConvention.from_config(None)
    assert not fc2.is_left_handed
    assert fc2.handedness == "right"
