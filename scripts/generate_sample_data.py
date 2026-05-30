"""Generate synthetic sample data for testing and demo purposes.

Creates a pinhole camera image with four clean (unwarped) ArUco markers
placed at positions computed from perspective projection of known 3D
corner coordinates. This ensures:
- Markers are detected reliably (no warping artifacts).
- 2D positions roughly match the 3D layout under a valid camera pose.

Usage:
    python scripts/generate_sample_data.py
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIGS_DIR = ROOT / "configs"


def _save_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Config saved: {path}")


def _build_optical_rotation():
    return np.array(
        [[0, -1, 0],
         [0, 0, -1],
         [1, 0, 0]],
        dtype=np.float64,
    )


def main():
    img_w, img_h = 1600, 1200

    # ── Camera config ──
    K = np.array(
        [[1000, 0, img_w / 2],
         [0, 1000, img_h / 2],
         [0, 0, 1]],
        dtype=np.float64,
    )
    camera_cfg = {
        "camera_name": "sample_camera",
        "model": "pinhole",
        "image_width": img_w,
        "image_height": img_h,
        "K": K.tolist(),
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    _save_yaml(camera_cfg, CONFIGS_DIR / "camera.sample.yaml")

    # ── Ground-truth camera pose ──
    c_body = np.array([0.1, 0.0, 1.1])
    R_body_to_optical = _build_optical_rotation()
    t_body_in_optical = -R_body_to_optical @ c_body
    rvec, _ = cv2.Rodrigues(R_body_to_optical)

    # ── Marker 3D layout (body frame: x=fwd, y=left, z=up) ──
    s = 0.30   # marker side length (m)
    x_wall = 3.0

    markers_layout = {}
    centers = {
        0: (x_wall,  0.5, 0.8),
        1: (x_wall, -0.5, 0.8),
        2: (x_wall,  0.5, 0.0),
        3: (x_wall, -0.5, 0.0),
    }
    for m_id, (cx, cy, cz) in centers.items():
        markers_layout[m_id] = {
            "corners_body": [
                [cx,       cy + s/2, cz + s/2],
                [cx + s,   cy + s/2, cz + s/2],
                [cx + s,   cy - s/2, cz - s/2],
                [cx,       cy - s/2, cz - s/2],
            ]
        }

    marker_layout_cfg = {
        "body_frame": {
            "name": "vehicle",
            "convention": "x_forward_y_left_z_up",
            "handedness": "right",
        },
        "dictionary": "DICT_4X4_50",
        "markers": markers_layout,
    }
    _save_yaml(marker_layout_cfg, CONFIGS_DIR / "marker_layout.sample.yaml")

    # ── Render markers at projected positions ──
    marker_px = 160
    quiet = 16
    total = marker_px + 2 * quiet

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    for m_id, m_cfg in markers_layout.items():
        pts_3d = np.array(m_cfg["corners_body"], dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t_body_in_optical, K, None)
        pts_2d = pts_2d.reshape(4, 2)

        # Center of projected quad
        center = pts_2d.mean(axis=0)
        cx_i, cy_i = int(round(center[0])), int(round(center[1]))

        # Expected size at this depth
        depth = np.linalg.norm(pts_3d.mean(axis=0) - c_body)
        expected_px = K[0, 0] * s / depth
        scale = expected_px / marker_px

        # Place flat marker centered at projected position, scaled to approx size
        marker = cv2.aruco.generateImageMarker(aruco_dict, m_id, marker_px)
        marker = np.pad(marker, quiet, constant_values=255)

        if scale != 1.0:
            new_sz = int(total * scale)
            marker = cv2.resize(marker, (new_sz, new_sz), interpolation=cv2.INTER_NEAREST)

        h, w = marker.shape
        x0 = cx_i - w // 2
        y0 = cy_i - h // 2

        # Clamp to canvas
        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(img_w, x0 + w)
        y1c = min(img_h, y0 + h)
        mx0 = x0c - x0
        my0 = y0c - y0
        mx1 = mx0 + (x1c - x0c)
        my1 = my0 + (y1c - y0c)

        patch = marker[my0:my1, mx0:mx1]
        for ch in range(3):
            canvas[y0c:y1c, x0c:x1c, ch] = np.where(patch < 128, 0, canvas[y0c:y1c, x0c:x1c, ch])

    out_path = DATA_DIR / "sample_markers.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"  Sample image saved: {out_path}")

    print("\n  Sample data ready. Run:")
    print(f"    uv run cam2body-calib estimate -i data/sample_markers.png "
          f"-c configs/camera.sample.yaml -l configs/marker_layout.sample.yaml")


if __name__ == "__main__":
    main()
