"""Fiducial marker detection module."""
from .aruco_detector import ArucoDetector
from .base import DetectionResult, FiducialDetector

__all__ = ["FiducialDetector", "DetectionResult", "ArucoDetector"]
