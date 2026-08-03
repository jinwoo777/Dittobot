"""Deterministic local perception outputs for mock and test flows."""

from __future__ import annotations

from dataclasses import dataclass

from robot_skill_system.capture.interfaces import CaptureResult, SynchronizedRGBDFrame
from robot_skill_system.scene.models import (
    AccessPolicy,
    BoundingBox2D,
    BoundingBox3D,
    ObjectInstance,
    Pose,
    Quaternion,
    SurfaceInstance,
    SurfaceRole,
    ToolInstance,
    Vector3,
    WorkspaceRegion,
    WorkspaceRole,
)

from .interfaces import HandPoseEstimate, MarkerDetection, ObjectDetection2D, PerceptionResult


@dataclass(frozen=True)
class MockPerceptionConfig:
    confidence: float = 0.98
    reference_frame: str = "camera_color_optical_frame"
    marker_id: int = 7
    marker_side_length_m: float = 0.04
    tool_attached: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("mock perception confidence must be in [0, 1]")


def _pose(
    timestamp_ns: int,
    position: tuple[float, float, float],
    confidence: float,
    reference_frame: str,
    source: str = "mock_perception",
) -> Pose:
    return Pose(
        frame_id=reference_frame,
        position_m=Vector3(x=position[0], y=position[1], z=position[2]),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        timestamp_ns=timestamp_ns,
        source=source,
        confidence=confidence,
    )


class MockPerception:
    """Produces marker, hand, tool, object, surface, and workspace observations."""

    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.config = config or MockPerceptionConfig()

    @staticmethod
    def _frame(capture: CaptureResult | SynchronizedRGBDFrame) -> SynchronizedRGBDFrame:
        return capture.representative_frame if isinstance(capture, CaptureResult) else capture

    def detect(self, frame: SynchronizedRGBDFrame) -> list[ObjectDetection2D]:
        height, width = frame.color_image_rgb.shape[:2]
        return [
            ObjectDetection2D(
                detection_id="cloth_detection",
                class_name="cloth",
                bounding_box=BoundingBox2D(
                    x_min_px=width * 0.30,
                    y_min_px=height * 0.35,
                    x_max_px=width * 0.55,
                    y_max_px=height * 0.60,
                ),
                confidence=self.config.confidence,
            )
        ]

    def analyze(self, capture: CaptureResult | SynchronizedRGBDFrame) -> PerceptionResult:
        frame = self._frame(capture)
        confidence = self.config.confidence
        timestamp_ns = frame.timestamp_ns
        reference_frame = self.config.reference_frame
        marker_pose = _pose(timestamp_ns, (0.12, -0.04, 0.65), confidence, reference_frame)
        tool_pose = _pose(
            timestamp_ns,
            (0.12, -0.04, 0.61),
            confidence,
            reference_frame,
            source="robot_tf_mock" if self.config.tool_attached else "mock_marker_tool_pose",
        )
        object_pose = _pose(timestamp_ns, (0.20, 0.08, 0.74), confidence, reference_frame)
        marker = MarkerDetection(
            marker_id=self.config.marker_id,
            pose=marker_pose,
            side_length_m=self.config.marker_side_length_m,
            reprojection_error_px=0.2,
            confidence=confidence,
        )
        hand = HandPoseEstimate(
            pose=_pose(timestamp_ns, (0.10, -0.03, 0.62), confidence, reference_frame),
            gripper_width_m=0.055,
            direction=Vector3(x=1.0, y=0.0, z=0.0),
            palm_normal=Vector3(x=0.0, y=0.0, z=-1.0),
            confidence=confidence,
        )
        tool = ToolInstance(
            instance_id="wiper_01",
            tool_class="wiper",
            attached=self.config.tool_attached,
            tcp_frame="wiper_01_tcp",
            pose=tool_pose,
            compatible_skills=["wipe_surface"],
            verification_confidence=confidence,
        )
        object_instance = ObjectInstance(
            instance_id="cloth_01",
            class_name="cloth",
            attributes={"color": "blue"},
            pose=object_pose,
            bounding_box_2d=self.detect(frame)[0].bounding_box,
            bounding_box_3d=BoundingBox3D(
                center_pose=object_pose,
                size_m=Vector3(x=0.18, y=0.14, z=0.008),
            ),
            confidence=confidence,
            visible_fraction=1.0,
            pose_source="mock_depth_centroid",
        )
        boundary = [
            Vector3(x=-0.35, y=-0.25, z=0.75),
            Vector3(x=0.35, y=-0.25, z=0.75),
            Vector3(x=0.35, y=0.25, z=0.75),
            Vector3(x=-0.35, y=0.25, z=0.75),
        ]
        surface = SurfaceInstance(
            instance_id="table_surface_01",
            role=SurfaceRole.CONTACT_TARGET,
            center_m=Vector3(x=0.0, y=0.0, z=0.75),
            normal=Vector3(x=0.0, y=0.0, z=-1.0),
            boundary_m=boundary,
            confidence=confidence,
            allowed_contact_links=["wiper_01_tcp"],
            material="laminate",
        )
        task_region = WorkspaceRegion(
            region_id="table_task_region",
            role=WorkspaceRole.TASK_REGION,
            geometry={
                "type": "box",
                "center_m": [0.0, 0.0, 0.70],
                "size_m": [0.70, 0.50, 0.20],
            },
            frame_id=reference_frame,
            minimum_clearance_m=0.02,
            access_policy=AccessPolicy.SUPERVISED,
            confidence=confidence,
        )
        free_region = WorkspaceRegion(
            region_id="camera_free_space",
            role=WorkspaceRole.FREE_SPACE,
            geometry={
                "type": "box",
                "center_m": [0.0, 0.0, 0.50],
                "size_m": [0.80, 0.60, 0.35],
            },
            frame_id=reference_frame,
            minimum_clearance_m=0.03,
            access_policy=AccessPolicy.ALLOWED,
            confidence=confidence,
        )
        unknown_region = WorkspaceRegion(
            region_id="unobserved_space",
            role=WorkspaceRole.UNKNOWN_REGION,
            geometry={"type": "complement_of_observed_frustum"},
            frame_id=reference_frame,
            minimum_clearance_m=0.05,
            access_policy=AccessPolicy.FORBIDDEN,
            confidence=1.0 - confidence,
        )
        return PerceptionResult(
            timestamp_ns=timestamp_ns,
            reference_frame=reference_frame,
            markers=[marker],
            hands=[hand],
            tools=[tool],
            objects=[object_instance],
            surfaces=[surface],
            workspace_regions=[task_region, free_region, unknown_region],
            confidence=confidence,
        )

    # Protocol-specific convenience methods let this single deterministic mock
    # stand in for each estimator during focused unit tests.
    def estimate_marker_poses(self, frame: SynchronizedRGBDFrame) -> list[MarkerDetection]:
        return self.analyze(frame).markers

    def estimate_tool_poses(self, frame: SynchronizedRGBDFrame) -> list[ToolInstance]:
        return self.analyze(frame).tools

    def estimate_object_poses(self, frame: SynchronizedRGBDFrame) -> list[ObjectInstance]:
        return self.analyze(frame).objects

    def estimate_surfaces(self, frame: SynchronizedRGBDFrame) -> list[SurfaceInstance]:
        return self.analyze(frame).surfaces

    def build_workspace(self, frame: SynchronizedRGBDFrame) -> list[WorkspaceRegion]:
        return self.analyze(frame).workspace_regions


class MockMarkerPoseEstimator:
    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.pipeline = MockPerception(config)

    def estimate(self, frame: SynchronizedRGBDFrame) -> list[MarkerDetection]:
        return self.pipeline.analyze(frame).markers


class MockToolPoseEstimator:
    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.pipeline = MockPerception(config)

    def estimate(self, frame: SynchronizedRGBDFrame) -> list[ToolInstance]:
        return self.pipeline.analyze(frame).tools


class MockObjectDetector:
    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.pipeline = MockPerception(config)

    def detect(self, frame: SynchronizedRGBDFrame) -> list[ObjectDetection2D]:
        return self.pipeline.detect(frame)


class MockObjectPoseEstimator:
    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.pipeline = MockPerception(config)

    def estimate(
        self,
        frame: SynchronizedRGBDFrame,
        detections: list[ObjectDetection2D] | tuple[ObjectDetection2D, ...] = (),
    ) -> list[ObjectInstance]:
        del detections
        return self.pipeline.analyze(frame).objects


class MockSurfaceEstimator:
    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.pipeline = MockPerception(config)

    def estimate(self, frame: SynchronizedRGBDFrame) -> list[SurfaceInstance]:
        return self.pipeline.analyze(frame).surfaces


class MockWorkspaceBuilder:
    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.pipeline = MockPerception(config)

    def build(
        self,
        frame: SynchronizedRGBDFrame,
        surfaces: list[SurfaceInstance] | tuple[SurfaceInstance, ...] = (),
    ) -> list[WorkspaceRegion]:
        del surfaces
        return self.pipeline.analyze(frame).workspace_regions


__all__ = [
    "MockMarkerPoseEstimator",
    "MockObjectDetector",
    "MockObjectPoseEstimator",
    "MockPerception",
    "MockPerceptionConfig",
    "MockSurfaceEstimator",
    "MockToolPoseEstimator",
    "MockWorkspaceBuilder",
]
