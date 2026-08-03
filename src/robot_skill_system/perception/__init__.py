"""Local perception interfaces, deterministic mocks, and optional adapters."""

from .hand_pose import MediaPipeHandPoseEstimator
from .interfaces import (
    HandPoseEstimate,
    HandPoseEstimator,
    MarkerDetection,
    MarkerPoseEstimator,
    MarkerToolPoseEstimator,
    ObjectDetection2D,
    ObjectDetector,
    ObjectPoseEstimator,
    PerceptionPipeline,
    PerceptionResult,
    SurfaceEstimator,
    WorkspaceBuilder,
)
from .mock_perception import (
    MockMarkerPoseEstimator,
    MockObjectDetector,
    MockObjectPoseEstimator,
    MockPerception,
    MockPerceptionConfig,
    MockSurfaceEstimator,
    MockToolPoseEstimator,
    MockWorkspaceBuilder,
)
from .object_pose import DepthCentroidObjectPoseEstimator
from .tool_pose import (
    ArucoMarkerPoseEstimator,
    ArucoMarkerToolPoseEstimator,
    MarkerToolDefinition,
)

__all__ = [
    "ArucoMarkerPoseEstimator",
    "ArucoMarkerToolPoseEstimator",
    "DepthCentroidObjectPoseEstimator",
    "HandPoseEstimate",
    "HandPoseEstimator",
    "MarkerDetection",
    "MarkerPoseEstimator",
    "MarkerToolPoseEstimator",
    "MarkerToolDefinition",
    "MediaPipeHandPoseEstimator",
    "MockMarkerPoseEstimator",
    "MockObjectDetector",
    "MockObjectPoseEstimator",
    "MockPerception",
    "MockPerceptionConfig",
    "MockSurfaceEstimator",
    "MockToolPoseEstimator",
    "MockWorkspaceBuilder",
    "ObjectDetection2D",
    "ObjectDetector",
    "ObjectPoseEstimator",
    "PerceptionPipeline",
    "PerceptionResult",
    "SurfaceEstimator",
    "WorkspaceBuilder",
]
