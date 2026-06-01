"""REST API for cam2body-calib web UI."""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/api")

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "configs"


# ── Pydantic models ────────────────────────────────────────────────


class CameraInfo(BaseModel):
    name: str
    model: str
    image_width: int
    image_height: int


class PointData(BaseModel):
    u: float
    v: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class SolveRequest(BaseModel):
    camera_name: str
    points: list[PointData]


class ExportRequest(BaseModel):
    pose_matrix: list[list[float]]
    profile: str = "vehicle_lh_pose6"


# ── Camera listing ──────────────────────────────────────────────────


@router.get("/cameras")
def list_cameras() -> list[CameraInfo]:
    """List available camera config files under configs/."""
    from cam2body_calib.config.load import load_camera

    cameras: list[CameraInfo] = []
    for path in sorted(CONFIGS_DIR.glob("camera*.yaml")):
        try:
            cam = load_camera(str(path))
            cameras.append(CameraInfo(
                name=cam.camera_name,
                model=cam.model,
                image_width=cam.image_width,
                image_height=cam.image_height,
            ))
        except Exception:
            continue
    return cameras


# ── Image upload + undistort ────────────────────────────────────────


@router.post("/undistort")
async def undistort_image(image: UploadFile, camera_name: str, balance: float = 0.0):
    """Upload an image, apply camera undistortion, return base64 PNG + K."""
    from cam2body_calib.config.load import load_camera
    from cam2body_calib.camera.model import CameraModel

    cam = _load_named_camera(camera_name)
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Failed to decode image")

    if np.any(cam.D != 0):
        img, cam = cam.undistort_image(img, balance=balance)

    _, buf = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(buf).decode()

    return {
        "image_base64": img_b64,
        "width": cam.image_width,
        "height": cam.image_height,
        "K": cam.K.tolist(),
        "fx": float(cam.K[0, 0]),
        "fy": float(cam.K[1, 1]),
        "cx": float(cam.K[0, 2]),
        "cy": float(cam.K[1, 2]),
        "model": cam.model,
        "camera_name": cam.camera_name,
    }


# ── PnP solve ───────────────────────────────────────────────────────


@router.post("/solve")
def solve_pnp(req: SolveRequest):
    """Run PnP with annotated 2D points and 3D coordinates."""
    from cam2body_calib.estimation.pnp_solver import PnPSolver

    cam = _load_named_camera(req.camera_name)

    obj_pts = np.array([[p.x, p.y, p.z] for p in req.points], dtype=np.float64)
    img_pts = np.array([[p.u, p.v] for p in req.points], dtype=np.float64)

    solver = PnPSolver(cam)
    result = solver.solve(obj_pts, img_pts)

    if not result.success:
        raise HTTPException(400, result.message)

    stats = result.reprojection_stats
    pos = result.position_body
    rpy_link = np.degrees(result.rpy_link_body_cam)
    rpy_optical = np.degrees(result.rpy_optical_body_cam)

    response: dict = {
        "success": True,
        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "rpy_link": {"roll": float(rpy_link[0]), "pitch": float(rpy_link[1]), "yaw": float(rpy_link[2])},
        "rpy_optical": {"roll": float(rpy_optical[0]), "pitch": float(rpy_optical[1]), "yaw": float(rpy_optical[2])},
        "reprojection": {
            "mean_error": float(stats.mean_error),
            "max_error": float(stats.max_error),
            "inliers": stats.inlier_count,
            "total": stats.total_points,
            "per_point": [float(e) for e in stats.per_point_errors],
        },
        "T_body_camera_link": result.T_body_camera_link.tolist(),
    }
    return response


# ── Export profile ──────────────────────────────────────────────────


@router.post("/export")
def export_pose(req: ExportRequest):
    """Export a pose matrix in a custom coordinate convention."""
    from cam2body_calib.exporters.vehicle_lh_pose6 import VehicleLHPose6Exporter

    T = np.array(req.pose_matrix, dtype=np.float64)
    if T.shape != (4, 4):
        raise HTTPException(400, "pose_matrix must be 4x4")

    exporter = VehicleLHPose6Exporter()
    pose6 = exporter.export(T)
    return {
        "x": pose6.x,
        "y": pose6.y,
        "z": pose6.z,
        "roll": pose6.roll,
        "pitch": pose6.pitch,
        "yaw": pose6.yaw,
        "T_parent_child": pose6.T_parent_child.tolist(),
        "profile": pose6.profile_name,
    }


# ── Helpers ─────────────────────────────────────────────────────────


def _load_named_camera(name: str):
    """Load a camera config by its camera_name field."""
    from cam2body_calib.config.load import load_camera

    for path in sorted(CONFIGS_DIR.glob("camera*.yaml")):
        try:
            cam = load_camera(str(path))
            if cam.camera_name == name:
                return cam
        except Exception:
            continue
    raise HTTPException(404, f"Camera '{name}' not found in configs/")
