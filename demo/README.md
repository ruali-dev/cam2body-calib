# Demo — NE Fisheye Camera Calibration

Reproducible end-to-end example.

## Files

| File | Purpose |
|------|---------|
| `intrinsics.yaml` | NE camera intrinsics (fisheye, 960x768) |
| `image.png` | Blurred calibration image with marker on the ground |

## Steps

1. `uv run cam2body-calib ui`
2. Load intrinsics: `demo/intrinsics.yaml`
3. Load image: `demo/image.png`
4. Click the 4 marker corners in order (top-left → top-right → bottom-right → bottom-left)
5. Enter the 3D coordinates from the table below
6. Solve PnP

## 3D Corner Coordinates

Body frame (right-handed): x=forward, y=left, z=up. Unit: meters.

| Point | u (px) | v (px) | x (m) | y (m) | z (m) |
|-------|--------|--------|-------|-------|-------|
| 1 | 368.0 | 507.0 | 1.50 | 1.80 | -1.40 |
| 2 | 467.0 | 462.0 | 2.50 | 1.80 | -1.40 |
| 3 | 409.0 | 629.0 | 1.50 | 0.80 | -1.40 |
| 4 | 533.0 | 536.0 | 2.50 | 0.80 | -1.40 |

The 4 corners form a rectangular marker on the ground (~1m x ~1m, ~1.4m below body origin).

## Expected Result

With `intrinsics.yaml` loaded, you should get:

```
Position: x≈1.13m  y≈-0.40m  z≈-0.07m
RPY: roll≈-0.8°  pitch≈15.3°  yaw≈55.3°
Reprojection: mean≈1.0px  inliers=4/4
```
