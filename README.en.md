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

## Usage

### 1. Prepare camera intrinsics

**camera.yaml** (example: `configs/camera.example.yaml`):

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

### 2. Annotate corner points

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

If the image is from a fisheye camera, the tool automatically undistorts it when `-c` is provided.

### 3. Fill in 3D coordinates and solve PnP

Edit `outputs/annotations.yaml` to add body-frame 3D coordinates for each point:

```yaml
image_points:
  - u: 367.0
    v: 507.0
    x: 2.5        # body x (forward)
    y: 0.6        # body y (left)
    z: 0.8        # body z (up)
  - u: 463.0
    v: 462.0
    x: 2.5
    y: -0.6
    z: 0.8
  # ...
```

Then run PnP:

```python
import numpy as np
from cam2body_calib.config.load import load_camera
from cam2body_calib.estimation.pnp_solver import PnPSolver
from cam2body_calib.io.yaml_io import read_yaml

cam = load_camera("configs/camera.yaml")
anno = read_yaml("outputs/annotations.yaml")

obj_pts = np.array([[p["x"], p["y"], p["z"]] for p in anno["image_points"]],
                   dtype=np.float64)
img_pts = np.array([[p["u"], p["v"]] for p in anno["image_points"]],
                   dtype=np.float64)

result = PnPSolver(cam).solve(obj_pts, img_pts)
print(f"Position: x={result.position_body[0]:.4f}, "
      f"y={result.position_body[1]:.4f}, "
      f"z={result.position_body[2]:.4f}")
print(f"RPY (link): roll={result.rpy_link_body_cam[0]:.4f}, "
      f"pitch={result.rpy_link_body_cam[1]:.4f}, "
      f"yaw={result.rpy_link_body_cam[2]:.4f}")
print(f"Reprojection error: mean={result.reprojection_stats.mean_error:.4f} px")
```

### 4. Check quality

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

## ArUco Auto-Detection (optional)

If your fiducials are ArUco markers, use the `estimate` command for automatic corner detection:

```bash
uv run cam2body-calib estimate \
  -i data/image.jpg \
  -c configs/camera.yaml \
  -l configs/marker_layout.yaml \
  -o outputs/result.png
```

Prepare `marker_layout.yaml` (example: `configs/marker_layout.example.yaml`) with 3D positions of each marker's four corners in body frame. Corner order must match OpenCV's detection order (clockwise from marker top-left).

## Export Profiles

Different downstream systems expect different pose6 conventions. Declare in `marker_layout.yaml`:

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
