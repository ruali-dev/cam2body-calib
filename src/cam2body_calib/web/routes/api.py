"""REST API for cam2body-calib web UI."""

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel

from cam2body_calib.camera.model import CameraModel

router = APIRouter(prefix="/api")

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "configs"


# ── Pydantic models ────────────────────────────────────────────────


class CameraParams(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    model: str = "pinhole"
    D: list[float] = []

    def to_camera_model(self) -> CameraModel:
        K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )
        d = np.array(self.D, dtype=np.float64)
        return CameraModel(
            K=K, D=d,
            image_width=self.width,
            image_height=self.height,
            camera_name="",
            model=self.model,
        )


class PointData(BaseModel):
    u: float
    v: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class SolveRequest(BaseModel):
    camera: CameraParams
    points: list[PointData]


class ExportRequest(BaseModel):
    pose_matrix: list[list[float]]
    profile: str = "vehicle_lh_pose6"


# ── Camera intrinsics upload ────────────────────────────────────────


@router.post("/camera/upload-yaml")
async def upload_camera_yaml(file: UploadFile):
    """Upload a camera intrinsics YAML file, return parsed parameters."""
    import yaml

    contents = await file.read()
    try:
        data = yaml.safe_load(contents)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    K = data.get("K")
    if K is None or not isinstance(K, list) or len(K) != 3:
        raise HTTPException(400, "YAML must contain a 3x3 'K' matrix")

    return {
        "fx": float(K[0][0]),
        "fy": float(K[1][1]),
        "cx": float(K[0][2]),
        "cy": float(K[1][2]),
        "width": int(data.get("image_width", 0)),
        "height": int(data.get("image_height", 0)),
        "model": str(data.get("model", "pinhole")),
        "D": [float(d) for d in data.get("D", [])],
        "camera_name": str(data.get("camera_name", "")),
    }


# ── Image upload + undistort ────────────────────────────────────────


@router.post("/undistort-with-camera")
async def undistort_with_camera(
    image: UploadFile,
    camera: str = Form(...),
    balance: float = Form(0.0),
):
    """Upload an image and undistort it with camera params (JSON string)."""
    import json

    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Failed to decode image")

    cam_data = json.loads(camera)
    cam = _camera_from_dict(cam_data)

    if np.any(cam.D != 0):
        img, cam = cam.undistort_image(img, balance=balance)

    _, buf = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(buf).decode()

    return {
        "image_base64": img_b64,
        "width": cam.image_width,
        "height": cam.image_height,
        "fx": float(cam.K[0, 0]),
        "fy": float(cam.K[1, 1]),
        "cx": float(cam.K[0, 2]),
        "cy": float(cam.K[1, 2]),
        "model": cam.model,
    }


# ── PnP solve ───────────────────────────────────────────────────────


@router.post("/solve")
def solve_pnp(req: SolveRequest):
    """Run PnP with annotated 2D points and 3D coordinates."""
    from cam2body_calib.estimation.pnp_solver import PnPSolver

    cam = req.camera.to_camera_model()

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

    return {
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


def _camera_from_dict(d: dict) -> CameraModel:
    K = np.array(
        [[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )
    D = np.array(d.get("D", []), dtype=np.float64)
    return CameraModel(
        K=K, D=D,
        image_width=int(d.get("width", 0)),
        image_height=int(d.get("height", 0)),
        camera_name="",
        model=str(d.get("model", "pinhole")),
    )
