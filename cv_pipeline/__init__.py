"""
Krishi Sahyog - Computer Vision Pipeline

Frozen CropGuard v0.1 research pipeline.
"""

from .detection import detect_crop

from .disease_area import analyze_disease

from .severity import classify_severity


__all__ = [
    "detect_crop",
    "analyze_disease",
    "classify_severity",
]