# cam2body-calib

Monocular camera-to-body extrinsics calibration using visual fiducial markers.

Estimates the 6-DoF pose (x/y/z + roll/pitch/yaw) of a camera in the vehicle body frame by solving the Perspective-n-Point (PnP) problem with known 3D marker positions and detected 2D corners.

## How It Works

Given known camera intrinsics (K/D), 3D positions of fiducial markers in the body frame, and their 2D pixel coordinates in an image, the tool uses `solvePnPRansac` + LM refinement to solve:

```
P_camera = R · P_body + t
```

The pose is then inverted to get the camera position in body frame: `T_body_cam = inv(T_cam_body)`. Note that the `tvec` from `solvePnP` is the body origin in camera frame, not the camera position.

## Coordinate Frames

### Body Frame (base_link)

| Axis | Direction |
|------|-----------|
| X | Forward |
| Y | Left |
| Z | Up |

Right-handed by default. Left-handed convention (`x=fwd, y=right, z=up`) is also supported via config.

### OpenCV Camera Frame

| Axis | Direction |
|----|-----------|
| X | Right |
| Y | Down |
| Z | Forward |

The tool outputs three transformation matrices:
- `T_cam_body` (optical frame, direct solvePnP output)
- `T_body_camera_optical` (inverted)
- `T_body_camera_link` (camera_link frame, axes aligned with body — **recommended**)

### RPY Convention

```
R = Rz(yaw) · Ry(pitch) · Rx(roll)   (fixed-axis XYZ extrinsic)

roll:  about body x (forward)
pitch: about body y (left)
yaw:   about body z (up), positive = left turn
```

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/your-username/cam2body-calib.git
cd cam2body-calib
uv sync
```

## Quick Start

### 1. Prepare config files

**camera.yaml** — Camera intrinsics:

```yaml
camera_name: "my_camera"
model: pinhole              # pinhole or fisheye
image_width: 1920
image_height: 1080
K:
  - [800.0, 0.0, 960.0]
  - [0.0, 800.0, 540.0]
  - [0.0, 0.0, 1.0]
D: [-0.3, 0.1, 0.0, 0.0, 0.0]  # pinhole: 5 coeffs; fisheye: 4 coeffs
```

**marker_layout.yaml** — Marker 3D positions in body frame (meters):

```yaml
body_frame:
  name: vehicle
  convention: x_forward_y_left_z_up
  handedness: right

dictionary: DICT_4X4_50

markers:
  0:
    corners_body:
      - [1.0,  0.25, 0.2]   # top-left
      - [1.0, -0.25, 0.2]   # top-right
      - [1.0, -0.25, 0.1]   # bottom-right
      - [1.0,  0.25, 0.1]   # bottom-left
```

> Corner order must match OpenCV's ArUco detection order (clockwise from marker top-left). Wrong ordering produces incorrect poses even if reprojection error looks fine.

### 2. Run calibration

```bash
uv run cam2body-calib estimate \
  -i data/image.jpg \
  -c configs/camera.yaml \
  -l configs/marker_layout.yaml \
  -o outputs/result.png
```

### 3. Interpret results

Terminal output includes:
- Three 4×4 transformation matrices
- Camera position (xyz) and orientation (roll/pitch/yaw) in body frame
- Reprojection error statistics (mean / max / inlier count)
- Quality rating

The visualization image shows:
- Green borders: detected markers
- Red circles: detected corners
- Blue crosses: reprojected corners
- Yellow lines: per-point error vectors

## Manual Annotation Mode

For cases where automatic ArUco detection is unavailable or higher precision is needed:

```bash
uv run cam2body-calib annotate \
  -i data/image.jpg \
  -c configs/camera.yaml \
  -o outputs/annotations.yaml
```

Controls:
| Action | Key |
|--------|-----|
| Add point | Left click |
| Undo | Right click |
| Zoom | Scroll wheel |
| Pan | Middle-drag |
| Save & exit | S |
| Clear all | C |
| Reset view | R |
| Quit | Q / ESC |

After saving, edit the YAML file to fill in 3D body-frame coordinates, then solve via Python using `PnPSolver`.

## Export Profiles

Different downstream systems expect different pose6 conventions. Use export profiles in `marker_layout.yaml`:

```yaml
export_profiles:
  - name: vehicle_lh_pose6
    type: pose6
```

| Profile | Frame Convention | Yaw |
|---------|-----------------|-----|
| (default) | x=fwd, y=left, z=up (RH) | positive = left |
| `vehicle_lh_pose6` | x=fwd, y=right, z=up (LH) | positive = right |

All PnP computation runs in right-handed frame. Left-handed exports use `S@R@S` (det=+1) for safe RPY extraction.

## Quality Assessment

| Mean Reprojection Error | Rating |
|------------------------|--------|
| < 0.5 px | Excellent |
| < 1.5 px | Good |
| < 3.0 px | Fair — verify with multiple images |
| > 3.0 px | Unreliable |

Also check:
- Is the camera position physically plausible?
- Does the camera forward axis match the expected viewing direction?
- Are results consistent across multiple views?

## Running Tests

```bash
uv sync --dev
uv run pytest tests/ -v
```

## Project Structure

```
cam2body-calib/
├── pyproject.toml
├── configs/                      # Camera intrinsics + marker layout examples
│   ├── camera.example.yaml
│   └── marker_layout.example.yaml
├── data/                         # Place calibration images here
├── outputs/                      # Results and visualizations
├── scripts/
│   ├── generate_sample_data.py   # Synthetic sample data generator
│   ├── blur_faces.py             # Batch face blurring
│   └── interactive_blur.py       # Interactive region blurring
├── src/cam2body_calib/
│   ├── cli.py                    # CLI entry point (estimate / annotate)
│   ├── camera/model.py           # CameraModel (pinhole/fisheye + undistort)
│   ├── config/                   # YAML loading + Pydantic validation
│   ├── estimation/               # PnP solver, reprojection errors
│   ├── exporters/                # Coordinate convention exporters
│   ├── fiducials/                # Marker detection (ArUco)
│   ├── geometry/                 # 4×4 transforms, RPY, frame conventions
│   ├── interactive/              # Manual point annotation tool
│   ├── io/                       # Image / YAML I/O
│   ├── layouts/                  # Marker 3D layout providers
│   └── visualization/            # Result visualization
└── tests/
```

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| High reprojection error | Inaccurate intrinsics / wrong 3D coords | Check per-point error distribution |
| Unreasonable position/orientation | Wrong coordinate system mapping | Verify 3D coordinate axis definitions |
| PnP fails (0 inliers) | 3D-2D correspondence mismatch | Check axis direction and point ordering |
| Fisheye detection failure | Image not undistorted | Both CLI commands auto-undistort |
| Marker too small/far | Poor corner localization | Keep markers > 30px in image |

## License

MIT
