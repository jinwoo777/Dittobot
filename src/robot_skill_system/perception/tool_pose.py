"""Marker/tool pose interfaces and an optional local ArUco implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.capture.interfaces import NotConfiguredError, SynchronizedRGBDFrame
from robot_skill_system.scene.models import Pose, Quaternion, ToolInstance, Vector3

from ._geometry import rotation_matrix_to_quaternion_xyzw
from .interfaces import MarkerDetection, MarkerPoseEstimator, MarkerToolPoseEstimator

FloatArray = NDArray[np.float64]


def _load_cv2() -> ModuleType:
    try:
        import cv2
    except ImportError as exc:
        raise NotConfiguredError(
            "ArUco pose estimation requires the optional opencv-python package"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise NotConfiguredError("installed OpenCV build does not provide the aruco module")
    return cv2


class ArucoMarkerPoseEstimator:
    """Detect configured square markers and solve metric pose locally."""

    def __init__(
        self,
        marker_sizes_m: Mapping[int, float],
        *,
        dictionary_name: str = "DICT_4X4_50",
        maximum_reprojection_error_px: float = 4.0,
    ) -> None:
        if not marker_sizes_m or any(
            marker_id < 0 or size_m <= 0.0 for marker_id, size_m in marker_sizes_m.items()
        ):
            raise ValueError("marker_sizes_m must contain positive configured marker sizes")
        self.marker_sizes_m = dict(marker_sizes_m)
        self.dictionary_name = dictionary_name
        self.maximum_reprojection_error_px = maximum_reprojection_error_px

    def estimate(self, frame: SynchronizedRGBDFrame) -> Sequence[MarkerDetection]:
        cv2 = _load_cv2()
        dictionary_id = getattr(cv2.aruco, self.dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"unknown fixed ArUco dictionary: {self.dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        gray = cv2.cvtColor(frame.color_image_rgb, cv2.COLOR_RGB2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary)
            corners, identifiers, _ = detector.detectMarkers(gray)
        else:
            corners, identifiers, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if identifiers is None:
            return ()
        intrinsics = frame.color_intrinsics
        camera_matrix = np.asarray(
            (
                (intrinsics.fx_px, 0.0, intrinsics.cx_px),
                (0.0, intrinsics.fy_px, intrinsics.cy_px),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
        output: list[MarkerDetection] = []
        for image_corners, identifier_array in zip(corners, identifiers, strict=True):
            marker_id = int(identifier_array[0])
            size_m = self.marker_sizes_m.get(marker_id)
            if size_m is None:
                continue
            half = size_m / 2.0
            object_points = np.asarray(
                ((-half, half, 0.0), (half, half, 0.0), (half, -half, 0.0), (-half, -half, 0.0)),
                dtype=np.float64,
            )
            flattened_corners = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points,
                flattened_corners,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            quaternion = rotation_matrix_to_quaternion_xyzw(
                np.asarray(rotation_matrix, dtype=np.float64)
            )
            projected, _ = cv2.projectPoints(
                object_points, rotation_vector, translation_vector, camera_matrix, distortion
            )
            reprojection_error = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(projected).reshape(4, 2) - flattened_corners
                        )
                    )
                )
            )
            if reprojection_error > self.maximum_reprojection_error_px:
                continue
            confidence = max(
                0.0, 1.0 - reprojection_error / self.maximum_reprojection_error_px
            )
            translation = np.asarray(translation_vector).reshape(3)
            pose = Pose(
                frame_id=frame.reference_frame,
                position_m=Vector3(
                    x=float(translation[0]),
                    y=float(translation[1]),
                    z=float(translation[2]),
                ),
                orientation_xyzw=Quaternion(
                    x=quaternion[0],
                    y=quaternion[1],
                    z=quaternion[2],
                    w=quaternion[3],
                ),
                timestamp_ns=frame.timestamp_ns,
                source="aruco_solvepnp",
                confidence=confidence,
            )
            output.append(
                MarkerDetection(
                    marker_id=marker_id,
                    pose=pose,
                    side_length_m=size_m,
                    reprojection_error_px=reprojection_error,
                    confidence=confidence,
                )
            )
        return tuple(output)


@dataclass(frozen=True)
class MarkerToolDefinition:
    """Approved calibration from one marker frame to a tool TCP."""

    instance_id: str
    tool_class: str
    tcp_frame: str
    marker_to_tcp_translation_m: tuple[float, float, float]
    marker_to_tcp_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    compatible_skills: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.instance_id or not self.tool_class or not self.tcp_frame:
            raise ValueError("marker tool identifiers must be non-empty")
        norm = math.sqrt(sum(value * value for value in self.marker_to_tcp_orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=1.0e-4, abs_tol=1.0e-4):
            raise ValueError("marker_to_tcp_orientation_xyzw must be normalized")


def _quaternion_multiply(first: FloatArray, second: FloatArray) -> FloatArray:
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return np.asarray(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dtype=np.float64,
    )


def _rotate_vector(quaternion: FloatArray, vector: FloatArray) -> FloatArray:
    pure = np.asarray((vector[0], vector[1], vector[2], 0.0), dtype=np.float64)
    conjugate = np.asarray((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]))
    return _quaternion_multiply(_quaternion_multiply(quaternion, pure), conjugate)[:3]


class ArucoMarkerToolPoseEstimator:
    """Apply approved marker-to-TCP calibrations to local marker estimates."""

    def __init__(
        self,
        marker_estimator: MarkerPoseEstimator,
        tool_definitions: Mapping[int, MarkerToolDefinition],
    ) -> None:
        self.marker_estimator = marker_estimator
        self.tool_definitions = dict(tool_definitions)

    def estimate(self, frame: SynchronizedRGBDFrame) -> Sequence[ToolInstance]:
        tools: list[ToolInstance] = []
        for marker in self.marker_estimator.estimate(frame):
            definition = self.tool_definitions.get(marker.marker_id)
            if definition is None:
                continue
            marker_quaternion = np.asarray(
                marker.pose.orientation_xyzw.as_tuple(), dtype=np.float64
            )
            calibration_quaternion = np.asarray(
                definition.marker_to_tcp_orientation_xyzw, dtype=np.float64
            )
            tcp_quaternion = _quaternion_multiply(marker_quaternion, calibration_quaternion)
            tcp_quaternion /= np.linalg.norm(tcp_quaternion)
            marker_position = np.asarray(marker.pose.position_m.as_tuple(), dtype=np.float64)
            offset = _rotate_vector(
                marker_quaternion,
                np.asarray(definition.marker_to_tcp_translation_m, dtype=np.float64),
            )
            tcp_position = marker_position + offset
            pose = Pose(
                frame_id=marker.pose.frame_id,
                position_m=Vector3(
                    x=float(tcp_position[0]),
                    y=float(tcp_position[1]),
                    z=float(tcp_position[2]),
                ),
                orientation_xyzw=Quaternion(
                    x=float(tcp_quaternion[0]),
                    y=float(tcp_quaternion[1]),
                    z=float(tcp_quaternion[2]),
                    w=float(tcp_quaternion[3]),
                ),
                timestamp_ns=marker.pose.timestamp_ns,
                source="aruco_marker_tcp_calibration",
                confidence=marker.confidence,
            )
            tools.append(
                ToolInstance(
                    instance_id=definition.instance_id,
                    tool_class=definition.tool_class,
                    attached=False,
                    tcp_frame=definition.tcp_frame,
                    pose=pose,
                    compatible_skills=list(definition.compatible_skills),
                    verification_confidence=marker.confidence,
                )
            )
        return tuple(tools)


__all__ = [
    "ArucoMarkerPoseEstimator",
    "ArucoMarkerToolPoseEstimator",
    "MarkerDetection",
    "MarkerPoseEstimator",
    "MarkerToolDefinition",
    "MarkerToolPoseEstimator",
]
