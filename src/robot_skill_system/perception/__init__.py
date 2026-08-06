"""Local perception interfaces, deterministic mocks, and optional adapters."""

from .hand_pose import MediaPipeHandPoseEstimator
from .hand_tracking import (
    LocalHandLandmark,
    LocalHandTrackingFrame,
    LocalHandTrackingSummary,
    LocalTrackedHand,
    MediaPipeHandLandmarkTracker,
    NormalizedHandPoint,
    unavailable_hand_tracking_summary,
)
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
    "LocalHandLandmark",
    "LocalHandTrackingFrame",
    "LocalHandTrackingSummary",
    "LocalTrackedHand",
    "MarkerDetection",
    "MarkerPoseEstimator",
    "MarkerToolPoseEstimator",
    "MarkerToolDefinition",
    "MediaPipeHandPoseEstimator",
    "MediaPipeHandLandmarkTracker",
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
    "NormalizedHandPoint",
    "PerceptionPipeline",
    "PerceptionResult",
    "SurfaceEstimator",
    "WorkspaceBuilder",
    "unavailable_hand_tracking_summary",
]
