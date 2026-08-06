from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot_skill_system.perception import hand_tracking
from robot_skill_system.perception.hand_tracking import (
    MediaPipeHandLandmarkTracker,
    unavailable_hand_tracking_summary,
)


def _result(*, detected: bool) -> SimpleNamespace:
    if not detected:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    landmarks = [
        SimpleNamespace(x=index / 20.0, y=(20 - index) / 20.0, z=-index / 100.0)
        for index in range(21)
    ]
    return SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[
            SimpleNamespace(
                classification=[SimpleNamespace(label="Right", score=0.91)]
            )
        ],
    )


def _fake_mediapipe(results: list[SimpleNamespace]) -> SimpleNamespace:
    class FakeEstimator:
        def __init__(self, **_kwargs: object) -> None:
            self.closed = False

        def process(self, image: np.ndarray[Any, np.dtype[np.uint8]]) -> SimpleNamespace:
            assert image.flags.c_contiguous
            return results.pop(0)

        def close(self) -> None:
            self.closed = True

    return SimpleNamespace(
        solutions=SimpleNamespace(
            hands=SimpleNamespace(Hands=FakeEstimator, HAND_CONNECTIONS=()),
            drawing_utils=SimpleNamespace(draw_landmarks=lambda *_args: None),
        )
    )


def test_mediapipe_tracker_builds_bounded_prompt_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_mediapipe([_result(detected=True), _result(detected=False)])
    monkeypatch.setattr(hand_tracking, "_load_mediapipe", lambda: fake)
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    summary = MediaPipeHandLandmarkTracker().track_sequence(
        [(3, image), (9, image)], requested_frame_count=2
    )

    assert summary.status == "completed"
    assert summary.detected_frame_count == 1
    assert [frame.frame_index for frame in summary.frames] == [3, 9]
    [tracked_hand] = summary.frames[0].hands
    assert tracked_hand.handedness == "right"
    assert [item.name for item in tracked_hand.selected_landmarks] == [
        "wrist",
        "thumb_tip",
        "index_mcp",
        "index_tip",
        "pinky_mcp",
    ]
    assert tracked_hand.thumb_index_midpoint_normalized.x == pytest.approx(0.3)
    assert summary.semantic_only is True
    assert "non_metric_relative_z" in summary.coordinate_space


def test_unavailable_tracker_summary_makes_no_detection_claims() -> None:
    summary = unavailable_hand_tracking_summary(4, "mediapipe is not installed")

    assert summary.status == "unavailable"
    assert summary.processed_frame_count == 0
    assert summary.detected_frame_count == 0
    assert summary.frames == []
