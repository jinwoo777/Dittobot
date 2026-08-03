"""Hand-pose interface and optional MediaPipe RGB-D implementation."""

from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import numpy as np

from robot_skill_system.capture.interfaces import NotConfiguredError, SynchronizedRGBDFrame
from robot_skill_system.scene.models import Pose, Quaternion, Vector3

from ._geometry import rotation_matrix_to_quaternion_xyzw
from .interfaces import HandPoseEstimate, HandPoseEstimator


def _load_mediapipe() -> ModuleType:
    try:
        import mediapipe  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotConfiguredError(
            "MediaPipe hand estimation requires the optional mediapipe package"
        ) from exc
    return cast(ModuleType, mediapipe)


def _depth_at(frame: SynchronizedRGBDFrame, pixel_x: int, pixel_y: int) -> float | None:
    height, width = frame.depth_image_m.shape
    if not 0 <= pixel_x < width or not 0 <= pixel_y < height:
        return None
    lower_x, upper_x = max(0, pixel_x - 1), min(width, pixel_x + 2)
    lower_y, upper_y = max(0, pixel_y - 1), min(height, pixel_y + 2)
    patch = frame.depth_image_m[lower_y:upper_y, lower_x:upper_x]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(valid)) if len(valid) else None


def _deproject(
    frame: SynchronizedRGBDFrame, normalized_x: float, normalized_y: float
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    intrinsics = frame.color_intrinsics
    pixel_x = int(round(normalized_x * (intrinsics.width_px - 1)))
    pixel_y = int(round(normalized_y * (intrinsics.height_px - 1)))
    depth_m = _depth_at(frame, pixel_x, pixel_y)
    if depth_m is None:
        return None
    x_m = (pixel_x - intrinsics.cx_px) / intrinsics.fx_px * depth_m
    y_m = (pixel_y - intrinsics.cy_px) / intrinsics.fy_px * depth_m
    return np.asarray((x_m, y_m, depth_m), dtype=np.float64)


class MediaPipeHandPoseEstimator:
    """Estimate a virtual TCP from local MediaPipe landmarks and aligned depth."""

    def __init__(
        self,
        *,
        minimum_detection_confidence: float = 0.6,
        minimum_tracking_confidence: float = 0.6,
    ) -> None:
        if not 0.0 <= minimum_detection_confidence <= 1.0:
            raise ValueError("minimum_detection_confidence must be in [0, 1]")
        if not 0.0 <= minimum_tracking_confidence <= 1.0:
            raise ValueError("minimum_tracking_confidence must be in [0, 1]")
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_tracking_confidence = minimum_tracking_confidence

    def estimate(self, frame: SynchronizedRGBDFrame) -> HandPoseEstimate | None:
        mediapipe = _load_mediapipe()
        with mediapipe.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=self.minimum_detection_confidence,
            min_tracking_confidence=self.minimum_tracking_confidence,
        ) as estimator:
            result = estimator.process(frame.color_image_rgb)
        if not result.multi_hand_landmarks:
            return None
        landmarks = result.multi_hand_landmarks[0].landmark
        indices = {"wrist": 0, "thumb_tip": 4, "index_mcp": 5, "index_tip": 8, "pinky_mcp": 17}
        points: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for name, index in indices.items():
            point = _deproject(frame, float(landmarks[index].x), float(landmarks[index].y))
            if point is None:
                return None
            points[name] = point
        virtual_tcp = (points["thumb_tip"] + points["index_tip"]) / 2.0
        direction = points["index_mcp"] - points["wrist"]
        palm_edge = points["pinky_mcp"] - points["wrist"]
        palm_normal = np.cross(direction, palm_edge)
        direction_norm = float(np.linalg.norm(direction))
        normal_norm = float(np.linalg.norm(palm_normal))
        if direction_norm < 1.0e-8 or normal_norm < 1.0e-8:
            return None
        x_axis = direction / direction_norm
        z_axis = palm_normal / normal_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
        classification_confidence = self.minimum_detection_confidence
        if result.multi_handedness:
            classification_confidence = float(
                result.multi_handedness[0].classification[0].score
            )
        confidence = min(1.0, max(0.0, classification_confidence))
        gripper_width_m = float(
            np.linalg.norm(points["thumb_tip"] - points["index_tip"])
        )
        pose = Pose(
            frame_id=frame.reference_frame,
            position_m=Vector3(
                x=float(virtual_tcp[0]),
                y=float(virtual_tcp[1]),
                z=float(virtual_tcp[2]),
            ),
            orientation_xyzw=Quaternion(
                x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
            ),
            timestamp_ns=frame.timestamp_ns,
            source="mediapipe_rgbd",
            confidence=confidence,
        )
        return HandPoseEstimate(
            pose=pose,
            gripper_width_m=gripper_width_m,
            direction=Vector3(x=float(x_axis[0]), y=float(x_axis[1]), z=float(x_axis[2])),
            palm_normal=Vector3(
                x=float(z_axis[0]), y=float(z_axis[1]), z=float(z_axis[2])
            ),
            confidence=confidence,
        )


__all__ = ["HandPoseEstimate", "HandPoseEstimator", "MediaPipeHandPoseEstimator"]
