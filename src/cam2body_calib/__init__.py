"""cam2body-calib: Camera-to-body extrinsics calibration using visual fiducial markers.

Estimates T_cam_body (body->camera transform) via solvePnP from ArUco marker
detections, then inverts to get T_body_cam (camera pose in body frame).
"""

__version__ = "0.1.0"
