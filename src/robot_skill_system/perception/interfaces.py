"""Typed local-perception outputs and estimator protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from robot_skill_system.capture.interfaces import CaptureResult, SynchronizedRGBDFrame
from robot_skill_system.scene.models import (
    BoundingBox2D,
    CameraMetadata,
    ConfidenceSummary,
    ObjectInstance,
    Pose,
    SceneSnapshot,
    StrictModel,
    SurfaceInstance,
    ToolInstance,
    Vector3,
    WorkspaceRegion,
)


class MarkerDetection(StrictModel):
    """Calibrated local fiducial pose; not a semantic object guess."""

    marker_id: int = Field(ge=0)
    pose: Pose
    side_length_m: float = Field(gt=0.0)
    reprojection_error_px: float | None = Field(default=None, ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class HandPoseEstimate(StrictModel):
    """Virtual gripper observation derived from a locally detected hand."""

    pose: Pose
    gripper_width_m: float = Field(ge=0.0)
    direction: Vector3
    palm_normal: Vector3
    confidence: float = Field(ge=0.0, le=1.0)


class ObjectDetection2D(StrictModel):
    """Image-space object evidence that still requires local pose estimation."""

    detection_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    bounding_box: BoundingBox2D
    confidence: float = Field(ge=0.0, le=1.0)


class PerceptionResult(StrictModel):
    """Complete typed outputs from one local RGB-D perception trigger."""

    timestamp_ns: int = Field(ge=0)
    reference_frame: str
    markers: list[MarkerDetection] = Field(default_factory=list)
    hands: list[HandPoseEstimate] = Field(default_factory=list)
    tools: list[ToolInstance] = Field(default_factory=list)
    objects: list[ObjectInstance] = Field(default_factory=list)
    surfaces: list[SurfaceInstance] = Field(default_factory=list)
    workspace_regions: list[WorkspaceRegion] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    def to_scene_snapshot(
        self,
        *,
        scene_id: str,
        valid_for_ms: int = 2000,
        calibration_id: str = "mock_calibration",
        camera_metadata: CameraMetadata | None = None,
    ) -> SceneSnapshot:
        """Convert already-localized outputs to the shared scene schema."""

        return SceneSnapshot(
            schema_version="1.0",
            scene_id=scene_id,
            timestamp_ns=self.timestamp_ns,
            reference_frame=self.reference_frame,
            valid_for_ms=valid_for_ms,
            objects=self.objects,
            tools=self.tools,
            surfaces=self.surfaces,
            workspace_regions=self.workspace_regions,
            calibration_id=calibration_id,
            camera_metadata=camera_metadata,
            confidence_summary=ConfidenceSummary(
                overall=self.confidence,
                perception=self.confidence,
                geometry=self.confidence,
                calibration=1.0 if calibration_id else 0.0,
            ),
        )


@runtime_checkable
class HandPoseEstimator(Protocol):
    def estimate(self, frame: SynchronizedRGBDFrame) -> HandPoseEstimate | None:
        """Estimate a virtual hand TCP using local RGB-D data."""


@runtime_checkable
class MarkerPoseEstimator(Protocol):
    def estimate(self, frame: SynchronizedRGBDFrame) -> Sequence[MarkerDetection]:
        """Estimate calibrated marker poses."""


@runtime_checkable
class MarkerToolPoseEstimator(Protocol):
    def estimate(self, frame: SynchronizedRGBDFrame) -> Sequence[ToolInstance]:
        """Map calibrated marker poses to configured tool TCP poses."""


@runtime_checkable
class ObjectDetector(Protocol):
    def detect(self, frame: SynchronizedRGBDFrame) -> Sequence[ObjectDetection2D]:
        """Return local image-space object detections."""


@runtime_checkable
class ObjectPoseEstimator(Protocol):
    def estimate(
        self,
        frame: SynchronizedRGBDFrame,
        detections: Sequence[ObjectDetection2D] = (),
    ) -> Sequence[ObjectInstance]:
        """Calculate metric object poses from local RGB-D evidence."""


@runtime_checkable
class SurfaceEstimator(Protocol):
    def estimate(self, frame: SynchronizedRGBDFrame) -> Sequence[SurfaceInstance]:
        """Estimate local planes, normals, and boundaries."""


@runtime_checkable
class WorkspaceBuilder(Protocol):
    def build(
        self,
        frame: SynchronizedRGBDFrame,
        surfaces: Sequence[SurfaceInstance] = (),
    ) -> Sequence[WorkspaceRegion]:
        """Build conservative free/task/unknown workspace regions."""


@runtime_checkable
class PerceptionPipeline(Protocol):
    def analyze(
        self, capture: CaptureResult | SynchronizedRGBDFrame
    ) -> PerceptionResult:
        """Run all configured local perception stages."""


__all__ = [
    "HandPoseEstimate",
    "HandPoseEstimator",
    "MarkerDetection",
    "MarkerPoseEstimator",
    "MarkerToolPoseEstimator",
    "ObjectDetection2D",
    "ObjectDetector",
    "ObjectPoseEstimator",
    "PerceptionPipeline",
    "PerceptionResult",
    "SurfaceEstimator",
    "WorkspaceBuilder",
]
