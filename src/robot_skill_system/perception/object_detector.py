"""Object-detection contracts.

Semantic image detections remain image-space evidence. They are never accepted as
metric robot targets without an :class:`ObjectPoseEstimator`.
"""

from .interfaces import ObjectDetection2D, ObjectDetector

__all__ = ["ObjectDetection2D", "ObjectDetector"]
