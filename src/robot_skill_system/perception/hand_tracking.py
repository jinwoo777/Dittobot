"""Bounded local MediaPipe landmark evidence for semantic skill prompts."""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from robot_skill_system.capture.interfaces import ColorImage, NotConfiguredError
from robot_skill_system.scene.models import StrictModel

TrackedLandmarkName = Literal[
    "wrist",
    "thumb_tip",
    "index_mcp",
    "index_tip",
    "pinky_mcp",
]

_SELECTED_LANDMARKS: tuple[tuple[TrackedLandmarkName, int], ...] = (
    ("wrist", 0),
    ("thumb_tip", 4),
    ("index_mcp", 5),
    ("index_tip", 8),
    ("pinky_mcp", 17),
)


def _load_mediapipe() -> ModuleType:
    try:
        import mediapipe  # type: ignore
    except ImportError as exc:
        raise NotConfiguredError(
            "local hand tracking requires the optional mediapipe package"
        ) from exc
    return cast(ModuleType, mediapipe)


class NormalizedHandPoint(StrictModel):
    """A bounded image point; it is not a camera or robot coordinate."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class LocalHandLandmark(StrictModel):
    """One selected MediaPipe landmark with non-metric relative depth."""

    name: TrackedLandmarkName
    x_normalized: float = Field(ge=0.0, le=1.0)
    y_normalized: float = Field(ge=0.0, le=1.0)
    z_relative: float


class LocalTrackedHand(StrictModel):
    """Prompt-sized evidence for one detected hand."""

    handedness: Literal["left", "right", "unknown"]
    handedness_confidence: float = Field(ge=0.0, le=1.0)
    selected_landmarks: list[LocalHandLandmark] = Field(min_length=5, max_length=5)
    thumb_index_midpoint_normalized: NormalizedHandPoint

    @model_validator(mode="after")
    def _contains_expected_landmarks(self) -> LocalTrackedHand:
        names = [item.name for item in self.selected_landmarks]
        expected = [name for name, _index in _SELECTED_LANDMARKS]
        if names != expected:
            raise ValueError("selected hand landmarks must use the fixed semantic order")
        return self


class LocalHandTrackingFrame(StrictModel):
    """Local hand detections for exactly one recording keyframe."""

    frame_index: int = Field(ge=0)
    hands: list[LocalTrackedHand] = Field(default_factory=list, max_length=2)


class LocalHandTrackingSummary(StrictModel):
    """Auditable MediaPipe evidence added to the semantic model input."""

    backend: Literal["mediapipe_hands_0_10"] = "mediapipe_hands_0_10"
    status: Literal["completed", "unavailable"]
    coordinate_space: Literal[
        "normalized_image_xy_with_non_metric_relative_z"
    ] = "normalized_image_xy_with_non_metric_relative_z"
    semantic_only: Literal[True] = True
    requested_frame_count: int = Field(ge=1, le=300)
    processed_frame_count: int = Field(ge=0, le=300)
    detected_frame_count: int = Field(ge=0, le=300)
    frames: list[LocalHandTrackingFrame] = Field(default_factory=list, max_length=300)
    failure_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def _counts_and_status_match(self) -> LocalHandTrackingSummary:
        if self.processed_frame_count != len(self.frames):
            raise ValueError("processed hand-tracking count does not match frames")
        detected_count = sum(bool(frame.hands) for frame in self.frames)
        if self.detected_frame_count != detected_count:
            raise ValueError("detected hand-tracking count does not match frames")
        if len({frame.frame_index for frame in self.frames}) != len(self.frames):
            raise ValueError("hand-tracking frame indices must be unique")
        if self.status == "completed":
            if self.processed_frame_count != self.requested_frame_count:
                raise ValueError("completed hand tracking must cover every requested frame")
            if self.failure_reason is not None:
                raise ValueError("completed hand tracking cannot contain a failure reason")
        elif self.frames or self.failure_reason is None:
            raise ValueError("unavailable hand tracking requires a reason and no frame claims")
        return self


class MediaPipeHandLandmarkTracker:
    """Reuse one local MediaPipe graph across live frames or bounded keyframes."""

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        minimum_detection_confidence: float = 0.5,
        minimum_tracking_confidence: float = 0.5,
    ) -> None:
        if not 1 <= max_num_hands <= 2:
            raise ValueError("max_num_hands must be one or two")
        for name, value in (
            ("minimum_detection_confidence", minimum_detection_confidence),
            ("minimum_tracking_confidence", minimum_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_tracking_confidence = minimum_tracking_confidence
        self._mediapipe: ModuleType | None = None
        self._estimator: Any | None = None
        self._last_result: Any | None = None

    def __enter__(self) -> MediaPipeHandLandmarkTracker:
        if self._estimator is not None:
            raise RuntimeError("MediaPipe hand tracker is already open")
        self._mediapipe = _load_mediapipe()
        self._estimator = self._mediapipe.solutions.hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.minimum_detection_confidence,
            min_tracking_confidence=self.minimum_tracking_confidence,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._estimator is not None:
            self._estimator.close()
        self._estimator = None
        self._last_result = None
        self._mediapipe = None

    def track_rgb(
        self, frame_index: int, color_image_rgb: ColorImage
    ) -> LocalHandTrackingFrame:
        """Track selected landmarks in one RGB image without metric claims."""

        if self._estimator is None:
            raise RuntimeError("MediaPipe hand tracker must be opened before use")
        if color_image_rgb.dtype != np.uint8:
            raise ValueError("hand-tracking RGB image must use uint8")
        if color_image_rgb.ndim != 3 or color_image_rgb.shape[2] != 3:
            raise ValueError("hand-tracking RGB image must have shape (height, width, 3)")
        self._last_result = self._estimator.process(
            np.ascontiguousarray(color_image_rgb)
        )
        return _frame_from_result(frame_index, self._last_result)

    def track_sequence(
        self,
        frames: Iterable[tuple[int, ColorImage]],
        *,
        requested_frame_count: int,
    ) -> LocalHandTrackingSummary:
        """Track a bounded sequence while loading only one source frame at a time."""

        tracked: list[LocalHandTrackingFrame] = []
        with self:
            for frame_index, image in frames:
                tracked.append(self.track_rgb(frame_index, image))
        return LocalHandTrackingSummary(
            status="completed",
            requested_frame_count=requested_frame_count,
            processed_frame_count=len(tracked),
            detected_frame_count=sum(bool(frame.hands) for frame in tracked),
            frames=tracked,
        )

    def draw_last_result(self, color_image_bgr: ColorImage) -> None:
        """Draw the last result for the standalone preview; never used for prompts."""

        if self._mediapipe is None or self._last_result is None:
            return
        for hand_landmarks in self._last_result.multi_hand_landmarks or []:
            self._mediapipe.solutions.drawing_utils.draw_landmarks(
                color_image_bgr,
                hand_landmarks,
                self._mediapipe.solutions.hands.HAND_CONNECTIONS,
            )


def unavailable_hand_tracking_summary(
    requested_frame_count: int, reason: str
) -> LocalHandTrackingSummary:
    """Represent a missing optional backend without inventing detections."""

    return LocalHandTrackingSummary(
        status="unavailable",
        requested_frame_count=requested_frame_count,
        processed_frame_count=0,
        detected_frame_count=0,
        failure_reason=reason,
    )


def _frame_from_result(frame_index: int, result: Any) -> LocalHandTrackingFrame:
    landmarks_by_hand = result.multi_hand_landmarks or []
    handedness_by_hand = result.multi_handedness or []
    hands: list[LocalTrackedHand] = []
    for hand_index, raw_hand in enumerate(landmarks_by_hand[:2]):
        handedness = "unknown"
        handedness_confidence = 0.0
        if hand_index < len(handedness_by_hand):
            classifications = handedness_by_hand[hand_index].classification or []
            if classifications:
                label = str(classifications[0].label).lower()
                if label in {"left", "right"}:
                    handedness = label
                handedness_confidence = min(
                    1.0, max(0.0, float(classifications[0].score))
                )
        selected = [
            LocalHandLandmark(
                name=name,
                x_normalized=min(1.0, max(0.0, float(raw_hand.landmark[index].x))),
                y_normalized=min(1.0, max(0.0, float(raw_hand.landmark[index].y))),
                z_relative=float(raw_hand.landmark[index].z),
            )
            for name, index in _SELECTED_LANDMARKS
        ]
        by_name = {item.name: item for item in selected}
        thumb_tip = by_name["thumb_tip"]
        index_tip = by_name["index_tip"]
        hands.append(
            LocalTrackedHand(
                handedness=cast(Literal["left", "right", "unknown"], handedness),
                handedness_confidence=handedness_confidence,
                selected_landmarks=selected,
                thumb_index_midpoint_normalized=NormalizedHandPoint(
                    x=(thumb_tip.x_normalized + index_tip.x_normalized) / 2.0,
                    y=(thumb_tip.y_normalized + index_tip.y_normalized) / 2.0,
                ),
            )
        )
    return LocalHandTrackingFrame(frame_index=frame_index, hands=hands)


__all__ = [
    "LocalHandLandmark",
    "LocalHandTrackingFrame",
    "LocalHandTrackingSummary",
    "LocalTrackedHand",
    "MediaPipeHandLandmarkTracker",
    "NormalizedHandPoint",
    "unavailable_hand_tracking_summary",
]
