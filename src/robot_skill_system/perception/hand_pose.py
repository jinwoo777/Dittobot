"""Local MediaPipe RGB-D fingertip tracking and gripper-state inference.

MediaPipe is used only to locate image-space landmarks.  Metric geometry and
the gripper-state decision are deterministic local operations over aligned
depth and calibrated colour intrinsics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from robot_skill_system.capture.interfaces import NotConfiguredError, SynchronizedRGBDFrame
from robot_skill_system.scene.models import Pose, Quaternion, StrictModel, Vector3

from ._geometry import rotation_matrix_to_quaternion_xyzw
from .deprojection import deproject_color_pixel
from .interfaces import HandPoseEstimate, HandPoseEstimator

THUMB_TIP_LANDMARK_INDEX: Literal[4] = 4
INDEX_TIP_LANDMARK_INDEX: Literal[8] = 8
DEFAULT_FINGER_CLOSE_THRESHOLD_M = 0.03
DEFAULT_FINGER_STATE_STABLE_FRAMES = 3
DEFAULT_DEPTH_PATCH_SIZE_PX = 5
DEFAULT_MAXIMUM_TIMESTAMP_SKEW_NS = 20_000_000


class FingerGripperState(str, Enum):
    """Stable or per-frame gripper state inferred from metric fingertip distance."""

    OPEN = "open"
    CLOSED = "closed"


class FingerObservation(StrictModel):
    """Auditable local RGB-D evidence and state-machine output for one frame."""

    frame_number: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    reference_frame: str = Field(min_length=1)
    status: Literal["valid", "uncertain"]
    thumb_landmark_index: Literal[4] = THUMB_TIP_LANDMARK_INDEX
    index_landmark_index: Literal[8] = INDEX_TIP_LANDMARK_INDEX
    thumb_normalized_xy: tuple[float, float] | None = None
    index_normalized_xy: tuple[float, float] | None = None
    thumb_pixel_xy: tuple[int, int] | None = None
    index_pixel_xy: tuple[int, int] | None = None
    thumb_depth_m: float | None = Field(default=None, gt=0.0)
    index_depth_m: float | None = Field(default=None, gt=0.0)
    thumb_point_camera_m: Vector3 | None = None
    index_point_camera_m: Vector3 | None = None
    midpoint_camera_m: Vector3 | None = None
    distance_m: float | None = Field(default=None, ge=0.0)
    candidate_state: FingerGripperState | None = None
    stabilization_progress_frames: int = Field(ge=0)
    required_stable_frames: int = Field(ge=1)
    stable_state: FingerGripperState | None = None
    invalid_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_observation_coherence(self) -> FingerObservation:
        if self.stabilization_progress_frames > self.required_stable_frames:
            raise ValueError("stabilization progress cannot exceed the required frames")
        if self.status == "valid":
            required = (
                self.thumb_normalized_xy,
                self.index_normalized_xy,
                self.thumb_pixel_xy,
                self.index_pixel_xy,
                self.thumb_depth_m,
                self.index_depth_m,
                self.thumb_point_camera_m,
                self.index_point_camera_m,
                self.midpoint_camera_m,
                self.distance_m,
                self.candidate_state,
            )
            if any(value is None for value in required):
                raise ValueError("valid finger observations require complete metric evidence")
            if self.invalid_reason is not None:
                raise ValueError("valid finger observations cannot have an invalid reason")
        elif self.invalid_reason is None:
            raise ValueError("uncertain finger observations require an invalid reason")
        return self

    def to_compact_landmark_trace(self) -> dict[str, object]:
        """Return only the two image-space tip traces suitable for an LLM hint."""

        thumb_detected = (
            self.thumb_normalized_xy is not None and self.thumb_pixel_xy is not None
        )
        index_detected = (
            self.index_normalized_xy is not None and self.index_pixel_xy is not None
        )
        return {
            "frame_index": self.frame_number,
            "timestamp_ns": self.timestamp_ns,
            "thumb_tip": {
                "landmark_index": self.thumb_landmark_index,
                "normalized_xy": self.thumb_normalized_xy if thumb_detected else None,
                "pixel_xy": self.thumb_pixel_xy if thumb_detected else None,
            },
            "index_tip": {
                "landmark_index": self.index_landmark_index,
                "normalized_xy": self.index_normalized_xy if index_detected else None,
                "pixel_xy": self.index_pixel_xy if index_detected else None,
            },
            "status": "valid" if thumb_detected and index_detected else "uncertain",
        }


class FingerStateTransition(StrictModel):
    """A first stable state or a subsequent stable open/closed transition."""

    frame_number: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    previous_state: FingerGripperState | None
    state: FingerGripperState
    distance_m: float = Field(ge=0.0)
    midpoint_camera_m: Vector3 | None = None


class FingerTrackingResult(StrictModel):
    """Combined frame evidence, optional hand pose, and optional transition."""

    observation: FingerObservation
    hand_pose: HandPoseEstimate | None = None
    transition: FingerStateTransition | None = None


@dataclass(frozen=True)
class _StabilizationDecision:
    stable_state: FingerGripperState | None
    progress_frames: int
    transition: FingerStateTransition | None


@dataclass(frozen=True)
class _ResolvedLandmark:
    normalized_xy: tuple[float, float] | None
    pixel_xy: tuple[int, int] | None
    depth_m: float | None
    point_camera_m: np.ndarray[Any, np.dtype[np.float64]] | None
    invalid_reason: str | None


def _load_mediapipe() -> ModuleType:
    try:
        import mediapipe  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotConfiguredError(
            "MediaPipe hand estimation requires the optional mediapipe package"
        ) from exc
    return cast(ModuleType, mediapipe)


def normalized_landmark_to_pixel(
    normalized_x: float,
    normalized_y: float,
    *,
    width_px: int,
    height_px: int,
) -> tuple[int, int] | None:
    """Map an in-frame normalized RGB landmark to an integer colour pixel."""

    if width_px <= 0 or height_px <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(normalized_x) or not math.isfinite(normalized_y):
        return None
    if not 0.0 <= normalized_x <= 1.0 or not 0.0 <= normalized_y <= 1.0:
        return None
    return (
        int(round(normalized_x * (width_px - 1))),
        int(round(normalized_y * (height_px - 1))),
    )


def depth_patch_median_m(
    frame: SynchronizedRGBDFrame,
    pixel_x: int,
    pixel_y: int,
    *,
    patch_size_px: int = DEFAULT_DEPTH_PATCH_SIZE_PX,
) -> float | None:
    """Return the median valid metric depth in an odd square pixel patch."""

    if patch_size_px <= 0 or patch_size_px % 2 == 0:
        raise ValueError("patch_size_px must be a positive odd integer")
    height, width = frame.depth_image_m.shape
    if not 0 <= pixel_x < width or not 0 <= pixel_y < height:
        return None
    radius = patch_size_px // 2
    lower_x, upper_x = max(0, pixel_x - radius), min(width, pixel_x + radius + 1)
    lower_y, upper_y = max(0, pixel_y - radius), min(height, pixel_y + radius + 1)
    patch = frame.depth_image_m[lower_y:upper_y, lower_x:upper_x]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(valid)) if len(valid) else None


def deproject_aligned_pixel(
    frame: SynchronizedRGBDFrame,
    pixel_x: int,
    pixel_y: int,
    depth_m: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Deproject a colour pixel and its aligned depth into the camera frame."""

    if not frame.aligned_depth_to_color:
        raise ValueError("depth must be aligned to colour before deprojection")
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be finite and positive")
    intrinsics = frame.color_intrinsics
    if not 0 <= pixel_x < intrinsics.width_px or not 0 <= pixel_y < intrinsics.height_px:
        raise ValueError("pixel is outside colour image bounds")
    return deproject_color_pixel(intrinsics, pixel_x, pixel_y, depth_m)


def classify_gripper_state(
    distance_m: float,
    *,
    close_threshold_m: float = DEFAULT_FINGER_CLOSE_THRESHOLD_M,
) -> FingerGripperState:
    """Classify ``<= threshold`` as closed and every larger metric distance as open."""

    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and non-negative")
    if not math.isfinite(close_threshold_m) or close_threshold_m <= 0.0:
        raise ValueError("close_threshold_m must be finite and positive")
    if distance_m <= close_threshold_m:
        return FingerGripperState.CLOSED
    return FingerGripperState.OPEN


class FingerStateStabilizer:
    """Require consecutive identical valid observations before changing state."""

    def __init__(self, *, stable_frames: int = DEFAULT_FINGER_STATE_STABLE_FRAMES) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        self.stable_frames = stable_frames
        self.stable_state: FingerGripperState | None = None
        self._run_state: FingerGripperState | None = None
        self._run_frames = 0

    def reset(self) -> None:
        """Reset all history, including the currently stable state."""

        self.stable_state = None
        self._run_state = None
        self._run_frames = 0

    def update(
        self,
        candidate_state: FingerGripperState | None,
        *,
        frame_number: int,
        timestamp_ns: int,
        distance_m: float | None = None,
        midpoint_camera_m: Vector3 | None = None,
    ) -> _StabilizationDecision:
        """Consume one state; ``None`` clears pending progress but preserves stability."""

        if frame_number < 0 or timestamp_ns < 0:
            raise ValueError("frame number and timestamp must be non-negative")
        if candidate_state is None:
            self._run_state = None
            self._run_frames = 0
            return _StabilizationDecision(self.stable_state, 0, None)
        if distance_m is None or not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("a valid candidate requires a non-negative finite distance_m")
        if candidate_state == self._run_state:
            self._run_frames = min(self.stable_frames, self._run_frames + 1)
        else:
            self._run_state = candidate_state
            self._run_frames = 1

        transition = None
        if self._run_frames >= self.stable_frames and candidate_state != self.stable_state:
            previous_state = self.stable_state
            self.stable_state = candidate_state
            transition = FingerStateTransition(
                frame_number=frame_number,
                timestamp_ns=timestamp_ns,
                previous_state=previous_state,
                state=candidate_state,
                distance_m=distance_m,
                midpoint_camera_m=midpoint_camera_m,
            )
        return _StabilizationDecision(self.stable_state, self._run_frames, transition)


def compact_fingertip_trace(
    observations: Sequence[FingerObservation],
) -> list[dict[str, object]]:
    """Serialize chronological thumb/index image coordinates without RGB or depth."""

    return [observation.to_compact_landmark_trace() for observation in observations]


class MediaPipeHandPoseEstimator:
    """Track a virtual TCP and stable gripper state from MediaPipe plus aligned depth.

    The optional MediaPipe ``Hands`` object is lazily created once and retained
    for this estimator's recording lifetime.  Call :meth:`close`, or use the
    estimator as a context manager, when the recording has been processed.
    """

    def __init__(
        self,
        *,
        minimum_detection_confidence: float = 0.6,
        minimum_tracking_confidence: float = 0.6,
        close_threshold_m: float = DEFAULT_FINGER_CLOSE_THRESHOLD_M,
        stable_frames: int = DEFAULT_FINGER_STATE_STABLE_FRAMES,
        maximum_timestamp_skew_ns: int = DEFAULT_MAXIMUM_TIMESTAMP_SKEW_NS,
    ) -> None:
        if not 0.0 <= minimum_detection_confidence <= 1.0:
            raise ValueError("minimum_detection_confidence must be in [0, 1]")
        if not 0.0 <= minimum_tracking_confidence <= 1.0:
            raise ValueError("minimum_tracking_confidence must be in [0, 1]")
        if not math.isfinite(close_threshold_m) or close_threshold_m <= 0.0:
            raise ValueError("close_threshold_m must be finite and positive")
        if maximum_timestamp_skew_ns < 0:
            raise ValueError("maximum_timestamp_skew_ns must be non-negative")
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_tracking_confidence = minimum_tracking_confidence
        self.close_threshold_m = close_threshold_m
        self.maximum_timestamp_skew_ns = maximum_timestamp_skew_ns
        self.state_stabilizer = FingerStateStabilizer(stable_frames=stable_frames)
        self._hands: Any | None = None

    @property
    def stable_state(self) -> FingerGripperState | None:
        """Return the currently confirmed gripper state, if one exists."""

        return self.state_stabilizer.stable_state

    def __enter__(self) -> MediaPipeHandPoseEstimator:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the optional MediaPipe graph without importing it at module load."""

        if self._hands is None:
            return
        close = getattr(self._hands, "close", None)
        if callable(close):
            close()
        self._hands = None

    def _get_hands(self) -> Any:
        if self._hands is None:
            mediapipe = _load_mediapipe()
            self._hands = mediapipe.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self.minimum_detection_confidence,
                min_tracking_confidence=self.minimum_tracking_confidence,
            )
        return self._hands

    def estimate(self, frame: SynchronizedRGBDFrame) -> HandPoseEstimate | None:
        """Backward-compatible hand-pose API; state evidence is available via ``track``."""

        return self.track(frame).hand_pose

    def track(self, frame: SynchronizedRGBDFrame) -> FingerTrackingResult:
        """Run persistent MediaPipe tracking and deterministic local RGB-D inference."""

        result = self._get_hands().process(frame.color_image_rgb)
        multi_hand_landmarks = getattr(result, "multi_hand_landmarks", None)
        if not multi_hand_landmarks:
            return self._uncertain_result(frame, invalid_reason="hand_not_detected")
        landmarks = multi_hand_landmarks[0].landmark
        required_highest_index = max(17, INDEX_TIP_LANDMARK_INDEX)
        if len(landmarks) <= required_highest_index:
            return self._uncertain_result(frame, invalid_reason="landmarks_incomplete")
        normalized_landmarks = {
            index: (float(landmarks[index].x), float(landmarks[index].y))
            for index in (0, THUMB_TIP_LANDMARK_INDEX, 5, INDEX_TIP_LANDMARK_INDEX, 17)
        }
        confidence = self.minimum_detection_confidence
        multi_handedness = getattr(result, "multi_handedness", None)
        if multi_handedness:
            try:
                confidence = float(multi_handedness[0].classification[0].score)
            except (AttributeError, IndexError, TypeError, ValueError):
                confidence = self.minimum_detection_confidence
        return self.track_normalized_landmarks(
            frame,
            normalized_landmarks,
            confidence=min(1.0, max(0.0, confidence)),
        )

    def track_normalized_landmarks(
        self,
        frame: SynchronizedRGBDFrame,
        normalized_landmarks: Mapping[int, tuple[float, float]],
        *,
        confidence: float = 1.0,
    ) -> FingerTrackingResult:
        """Process already-detected RGB landmarks without requiring MediaPipe import.

        This seam keeps geometry/state tests deterministic and is also useful for
        replaying stored semantic landmark evidence.  Only local aligned depth is
        authoritative for the metric state.
        """

        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        precondition_reason = self._frame_precondition_reason(frame)
        if precondition_reason is not None:
            thumb = self._resolve_image_landmark(
                frame,
                normalized_landmarks.get(THUMB_TIP_LANDMARK_INDEX),
                label="thumb_tip",
            )
            index = self._resolve_image_landmark(
                frame,
                normalized_landmarks.get(INDEX_TIP_LANDMARK_INDEX),
                label="index_tip",
            )
            return self._uncertain_result(
                frame,
                invalid_reason=precondition_reason,
                confidence=confidence,
                thumb=thumb,
                index=index,
            )
        thumb = self._resolve_landmark(
            frame,
            normalized_landmarks.get(THUMB_TIP_LANDMARK_INDEX),
            label="thumb_tip",
        )
        index = self._resolve_landmark(
            frame,
            normalized_landmarks.get(INDEX_TIP_LANDMARK_INDEX),
            label="index_tip",
        )
        invalid_reason = thumb.invalid_reason or index.invalid_reason
        if invalid_reason is not None:
            return self._uncertain_result(
                frame,
                invalid_reason=invalid_reason,
                confidence=confidence,
                thumb=thumb,
                index=index,
            )
        assert thumb.point_camera_m is not None
        assert index.point_camera_m is not None
        assert thumb.depth_m is not None
        assert index.depth_m is not None
        midpoint = (thumb.point_camera_m + index.point_camera_m) / 2.0
        distance_m = float(np.linalg.norm(thumb.point_camera_m - index.point_camera_m))
        candidate_state = classify_gripper_state(
            distance_m, close_threshold_m=self.close_threshold_m
        )
        midpoint_vector = _vector3(midpoint)
        decision = self.state_stabilizer.update(
            candidate_state,
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
            distance_m=distance_m,
            midpoint_camera_m=midpoint_vector,
        )
        hand_pose = self._build_hand_pose(
            frame,
            normalized_landmarks,
            thumb.point_camera_m,
            index.point_camera_m,
            distance_m,
            confidence,
        )
        observation = FingerObservation(
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
            reference_frame=frame.reference_frame,
            status="valid",
            thumb_normalized_xy=thumb.normalized_xy,
            index_normalized_xy=index.normalized_xy,
            thumb_pixel_xy=thumb.pixel_xy,
            index_pixel_xy=index.pixel_xy,
            thumb_depth_m=thumb.depth_m,
            index_depth_m=index.depth_m,
            thumb_point_camera_m=_vector3(thumb.point_camera_m),
            index_point_camera_m=_vector3(index.point_camera_m),
            midpoint_camera_m=midpoint_vector,
            distance_m=distance_m,
            candidate_state=candidate_state,
            stabilization_progress_frames=decision.progress_frames,
            required_stable_frames=self.state_stabilizer.stable_frames,
            stable_state=decision.stable_state,
            confidence=confidence,
        )
        return FingerTrackingResult(
            observation=observation,
            hand_pose=hand_pose,
            transition=decision.transition,
        )

    def _frame_precondition_reason(self, frame: SynchronizedRGBDFrame) -> str | None:
        if not frame.aligned_depth_to_color:
            return "depth_not_aligned_to_color"
        if (
            abs(frame.color_timestamp_ns - frame.depth_timestamp_ns)
            > self.maximum_timestamp_skew_ns
        ):
            return "timestamp_mismatch"
        return None

    def _resolve_landmark(
        self,
        frame: SynchronizedRGBDFrame,
        normalized_xy: tuple[float, float] | None,
        *,
        label: str,
    ) -> _ResolvedLandmark:
        image_landmark = self._resolve_image_landmark(
            frame, normalized_xy, label=label
        )
        if image_landmark.invalid_reason is not None:
            return image_landmark
        assert image_landmark.pixel_xy is not None
        depth_m = depth_patch_median_m(
            frame, image_landmark.pixel_xy[0], image_landmark.pixel_xy[1]
        )
        if depth_m is None:
            return _ResolvedLandmark(
                normalized_xy,
                image_landmark.pixel_xy,
                None,
                None,
                f"{label}_depth_unavailable",
            )
        try:
            point = deproject_aligned_pixel(
                frame, image_landmark.pixel_xy[0], image_landmark.pixel_xy[1], depth_m
            )
        except ValueError as exc:
            return _ResolvedLandmark(
                normalized_xy,
                image_landmark.pixel_xy,
                depth_m,
                None,
                f"{label}_metric_deprojection_unavailable: {exc}",
            )
        return _ResolvedLandmark(
            normalized_xy, image_landmark.pixel_xy, depth_m, point, None
        )

    def _resolve_image_landmark(
        self,
        frame: SynchronizedRGBDFrame,
        normalized_xy: tuple[float, float] | None,
        *,
        label: str,
    ) -> _ResolvedLandmark:
        if normalized_xy is None:
            return _ResolvedLandmark(None, None, None, None, f"{label}_missing")
        pixel_xy = normalized_landmark_to_pixel(
            normalized_xy[0],
            normalized_xy[1],
            width_px=frame.color_intrinsics.width_px,
            height_px=frame.color_intrinsics.height_px,
        )
        if pixel_xy is None:
            return _ResolvedLandmark(normalized_xy, None, None, None, f"{label}_out_of_bounds")
        return _ResolvedLandmark(normalized_xy, pixel_xy, None, None, None)

    def _uncertain_result(
        self,
        frame: SynchronizedRGBDFrame,
        *,
        invalid_reason: str,
        confidence: float = 0.0,
        thumb: _ResolvedLandmark | None = None,
        index: _ResolvedLandmark | None = None,
    ) -> FingerTrackingResult:
        decision = self.state_stabilizer.update(
            None,
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
        )
        observation = FingerObservation(
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
            reference_frame=frame.reference_frame,
            status="uncertain",
            thumb_normalized_xy=thumb.normalized_xy if thumb else None,
            index_normalized_xy=index.normalized_xy if index else None,
            thumb_pixel_xy=thumb.pixel_xy if thumb else None,
            index_pixel_xy=index.pixel_xy if index else None,
            thumb_depth_m=thumb.depth_m if thumb else None,
            index_depth_m=index.depth_m if index else None,
            thumb_point_camera_m=(
                _vector3(thumb.point_camera_m)
                if thumb is not None and thumb.point_camera_m is not None
                else None
            ),
            index_point_camera_m=(
                _vector3(index.point_camera_m)
                if index is not None and index.point_camera_m is not None
                else None
            ),
            stabilization_progress_frames=decision.progress_frames,
            required_stable_frames=self.state_stabilizer.stable_frames,
            stable_state=decision.stable_state,
            invalid_reason=invalid_reason,
            confidence=confidence,
        )
        return FingerTrackingResult(observation=observation)

    def _build_hand_pose(
        self,
        frame: SynchronizedRGBDFrame,
        normalized_landmarks: Mapping[int, tuple[float, float]],
        thumb_point: np.ndarray[Any, np.dtype[np.float64]],
        index_point: np.ndarray[Any, np.dtype[np.float64]],
        distance_m: float,
        confidence: float,
    ) -> HandPoseEstimate | None:
        supporting_points: dict[int, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for landmark_index, label in ((0, "wrist"), (5, "index_mcp"), (17, "pinky_mcp")):
            resolved = self._resolve_landmark(
                frame, normalized_landmarks.get(landmark_index), label=label
            )
            if resolved.point_camera_m is None:
                return None
            supporting_points[landmark_index] = resolved.point_camera_m
        virtual_tcp = (thumb_point + index_point) / 2.0
        direction = supporting_points[5] - supporting_points[0]
        palm_edge = supporting_points[17] - supporting_points[0]
        palm_normal = np.cross(direction, palm_edge)
        direction_norm = float(np.linalg.norm(direction))
        normal_norm = float(np.linalg.norm(palm_normal))
        if direction_norm < 1.0e-8 or normal_norm < 1.0e-8:
            return None
        x_axis = direction / direction_norm
        z_axis = palm_normal / normal_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = float(np.linalg.norm(y_axis))
        if y_axis_norm < 1.0e-8:
            return None
        y_axis /= y_axis_norm
        z_axis = np.cross(x_axis, y_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
        pose = Pose(
            frame_id=frame.reference_frame,
            position_m=_vector3(virtual_tcp),
            orientation_xyzw=Quaternion(
                x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
            ),
            timestamp_ns=frame.timestamp_ns,
            source="mediapipe_rgbd",
            confidence=confidence,
        )
        return HandPoseEstimate(
            pose=pose,
            gripper_width_m=distance_m,
            direction=_vector3(x_axis),
            palm_normal=_vector3(z_axis),
            confidence=confidence,
        )


def _vector3(point: np.ndarray[Any, np.dtype[np.float64]]) -> Vector3:
    return Vector3(x=float(point[0]), y=float(point[1]), z=float(point[2]))


__all__ = [
    "DEFAULT_DEPTH_PATCH_SIZE_PX",
    "DEFAULT_FINGER_CLOSE_THRESHOLD_M",
    "DEFAULT_FINGER_STATE_STABLE_FRAMES",
    "FingerGripperState",
    "FingerObservation",
    "FingerStateStabilizer",
    "FingerStateTransition",
    "FingerTrackingResult",
    "HandPoseEstimate",
    "HandPoseEstimator",
    "INDEX_TIP_LANDMARK_INDEX",
    "MediaPipeHandPoseEstimator",
    "THUMB_TIP_LANDMARK_INDEX",
    "classify_gripper_state",
    "compact_fingertip_trace",
    "deproject_aligned_pixel",
    "depth_patch_median_m",
    "normalized_landmark_to_pixel",
]
