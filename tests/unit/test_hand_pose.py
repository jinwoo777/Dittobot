from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.perception import deprojection, hand_pose
from robot_skill_system.perception.hand_pose import (
    FingerGripperState,
    FingerStateStabilizer,
    MediaPipeHandPoseEstimator,
    classify_gripper_state,
    compact_fingertip_trace,
    depth_patch_median_m,
    normalized_landmark_to_pixel,
)
from robot_skill_system.scene.models import Vector3
from robot_skill_system.settings import Settings


def _frame(
    *,
    depth_image_m: np.ndarray | None = None,
    frame_number: int = 0,
    aligned_depth_to_color: bool = True,
    color_timestamp_ns: int = 1_000_000_000,
    depth_timestamp_ns: int = 1_000_000_000,
    maximum_timestamp_skew_ns: int = 20_000_000,
    color_intrinsics: CameraIntrinsics | None = None,
) -> SynchronizedRGBDFrame:
    size = 101
    depth = (
        np.ones((size, size), dtype=np.float32)
        if depth_image_m is None
        else np.asarray(depth_image_m, dtype=np.float32)
    )
    return SynchronizedRGBDFrame(
        color_image_rgb=np.zeros((size, size, 3), dtype=np.uint8),
        depth_image_m=depth,
        color_timestamp_ns=color_timestamp_ns,
        depth_timestamp_ns=depth_timestamp_ns,
        color_intrinsics=color_intrinsics
        or CameraIntrinsics(
            width_px=size,
            height_px=size,
            fx_px=100.0,
            fy_px=100.0,
            cx_px=50.0,
            cy_px=50.0,
        ),
        frame_number=frame_number,
        aligned_depth_to_color=aligned_depth_to_color,
        maximum_timestamp_skew_ns=maximum_timestamp_skew_ns,
    )


@pytest.mark.parametrize(
    ("distance_m", "expected"),
    [
        (0.0299, FingerGripperState.CLOSED),
        (0.03, FingerGripperState.CLOSED),
        (0.0301, FingerGripperState.OPEN),
    ],
)
def test_metric_threshold_has_closed_inclusive_boundary(
    distance_m: float, expected: FingerGripperState
) -> None:
    assert classify_gripper_state(distance_m) is expected


def test_normalized_landmarks_use_separate_five_by_five_depth_patches() -> None:
    depth = np.zeros((101, 101), dtype=np.float32)
    depth[48:53, 28:33] = 1.0
    depth[48:53, 68:73] = 2.0
    depth[50, 30] = 50.0
    depth[50, 70] = np.nan
    tracker = MediaPipeHandPoseEstimator()

    result = tracker.track_normalized_landmarks(
        _frame(depth_image_m=depth),
        {4: (0.3, 0.5), 8: (0.7, 0.5)},
    )

    observation = result.observation
    assert observation.status == "valid"
    assert observation.thumb_normalized_xy == (0.3, 0.5)
    assert observation.index_normalized_xy == (0.7, 0.5)
    assert observation.thumb_pixel_xy == (30, 50)
    assert observation.index_pixel_xy == (70, 50)
    assert observation.thumb_depth_m == pytest.approx(1.0)
    assert observation.index_depth_m == pytest.approx(2.0)
    assert observation.thumb_point_camera_m == Vector3(x=-0.2, y=0.0, z=1.0)
    assert observation.index_point_camera_m == Vector3(x=0.4, y=0.0, z=2.0)
    assert observation.distance_m == pytest.approx(math.sqrt(1.36))
    assert result.hand_pose is None  # palm landmarks were intentionally omitted

    trace = compact_fingertip_trace([observation])
    assert trace == [observation.to_compact_landmark_trace()]
    assert json.loads(json.dumps(trace, separators=(",", ":"))) == [
        {
            "frame_index": 0,
            "timestamp_ns": 1_000_000_000,
            "thumb_tip": {
                "landmark_index": 4,
                "normalized_xy": [0.3, 0.5],
                "pixel_xy": [30, 50],
            },
            "index_tip": {
                "landmark_index": 8,
                "normalized_xy": [0.7, 0.5],
                "pixel_xy": [70, 50],
            },
            "status": "valid",
        }
    ]
    assert "depth" not in repr(trace)
    assert "distance" not in repr(trace)


def test_depth_patch_is_five_by_five_and_ignores_invalid_values() -> None:
    depth = np.zeros((101, 101), dtype=np.float32)
    depth[48:53, 48:53] = 0.8
    depth[48, 48] = np.inf
    depth[49, 49] = -1.0
    depth[50, 50] = 9.0

    assert depth_patch_median_m(_frame(depth_image_m=depth), 50, 50) == pytest.approx(0.8)
    assert normalized_landmark_to_pixel(0.0, 1.0, width_px=101, height_px=101) == (
        0,
        100,
    )
    assert normalized_landmark_to_pixel(-0.01, 0.5, width_px=101, height_px=101) is None


def test_three_consecutive_frames_are_required_and_uncertain_resets_pending() -> None:
    stabilizer = FingerStateStabilizer(stable_frames=3)
    midpoint = Vector3(x=0.0, y=0.0, z=1.0)

    first = stabilizer.update(
        FingerGripperState.CLOSED,
        frame_number=0,
        timestamp_ns=0,
        distance_m=0.02,
        midpoint_camera_m=midpoint,
    )
    second = stabilizer.update(
        FingerGripperState.CLOSED,
        frame_number=1,
        timestamp_ns=1,
        distance_m=0.02,
        midpoint_camera_m=midpoint,
    )
    third = stabilizer.update(
        FingerGripperState.CLOSED,
        frame_number=2,
        timestamp_ns=2,
        distance_m=0.02,
        midpoint_camera_m=midpoint,
    )
    assert (first.progress_frames, second.progress_frames, third.progress_frames) == (1, 2, 3)
    assert first.stable_state is None and second.stable_state is None
    assert third.transition is not None
    assert third.transition.previous_state is None
    assert third.transition.state is FingerGripperState.CLOSED

    for frame_number in (3, 4):
        pending = stabilizer.update(
            FingerGripperState.OPEN,
            frame_number=frame_number,
            timestamp_ns=frame_number,
            distance_m=0.05,
        )
        assert pending.transition is None
        assert pending.stable_state is FingerGripperState.CLOSED
    uncertain = stabilizer.update(None, frame_number=5, timestamp_ns=5)
    assert uncertain.progress_frames == 0
    assert uncertain.stable_state is FingerGripperState.CLOSED

    transitions = []
    for frame_number in (6, 7, 8):
        decision = stabilizer.update(
            FingerGripperState.OPEN,
            frame_number=frame_number,
            timestamp_ns=frame_number,
            distance_m=0.05,
        )
        if decision.transition is not None:
            transitions.append(decision.transition)
    assert len(transitions) == 1
    assert transitions[0].previous_state is FingerGripperState.CLOSED
    assert transitions[0].state is FingerGripperState.OPEN


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (_frame(aligned_depth_to_color=False), "depth_not_aligned_to_color"),
        (
            _frame(
                color_timestamp_ns=1_000_000_000,
                depth_timestamp_ns=1_030_000_000,
                maximum_timestamp_skew_ns=100_000_000,
            ),
            "timestamp_mismatch",
        ),
    ],
)
def test_unusable_rgbd_frame_is_uncertain_without_a_transition(
    frame: SynchronizedRGBDFrame, reason: str
) -> None:
    result = MediaPipeHandPoseEstimator().track_normalized_landmarks(
        frame, {4: (0.4, 0.5), 8: (0.6, 0.5)}
    )
    assert result.observation.status == "uncertain"
    assert result.observation.invalid_reason == reason
    assert result.observation.thumb_normalized_xy == (0.4, 0.5)
    assert result.observation.index_normalized_xy == (0.6, 0.5)
    assert result.observation.thumb_pixel_xy == (40, 50)
    assert result.observation.index_pixel_xy == (60, 50)
    assert result.transition is None
    assert result.observation.to_compact_landmark_trace()["status"] == "valid"


def test_compact_trace_sanitizes_out_of_bounds_landmarks() -> None:
    result = MediaPipeHandPoseEstimator().track_normalized_landmarks(
        _frame(), {4: (-0.01, 0.5), 8: (0.6, 0.5)}
    )

    assert result.observation.status == "uncertain"
    assert result.observation.invalid_reason == "thumb_tip_out_of_bounds"
    trace = result.observation.to_compact_landmark_trace()
    assert trace["thumb_tip"] == {
        "landmark_index": 4,
        "normalized_xy": None,
        "pixel_xy": None,
    }
    assert trace["index_tip"] == {
        "landmark_index": 8,
        "normalized_xy": (0.6, 0.5),
        "pixel_xy": (60, 50),
    }
    assert trace["status"] == "uncertain"


def test_depth_hole_clears_progress_but_preserves_confirmed_state() -> None:
    tracker = MediaPipeHandPoseEstimator(stable_frames=1)
    landmarks = {4: (0.49, 0.5), 8: (0.51, 0.5)}
    confirmed = tracker.track_normalized_landmarks(_frame(), landmarks)
    assert confirmed.observation.stable_state is FingerGripperState.CLOSED

    no_depth = np.zeros((101, 101), dtype=np.float32)
    uncertain = tracker.track_normalized_landmarks(
        _frame(depth_image_m=no_depth, frame_number=1), landmarks
    )
    assert uncertain.observation.invalid_reason == "thumb_tip_depth_unavailable"
    assert uncertain.observation.stabilization_progress_frames == 0
    assert uncertain.observation.stable_state is FingerGripperState.CLOSED
    assert uncertain.transition is None


def test_missing_fingertip_resets_pending_without_state_or_transition() -> None:
    tracker = MediaPipeHandPoseEstimator(stable_frames=2)
    closed_landmarks = {4: (0.49, 0.5), 8: (0.51, 0.5)}

    first = tracker.track_normalized_landmarks(_frame(frame_number=0), closed_landmarks)
    missing = tracker.track_normalized_landmarks(
        _frame(frame_number=1),
        {4: (0.49, 0.5)},
    )
    restarted = tracker.track_normalized_landmarks(
        _frame(frame_number=2), closed_landmarks
    )

    assert first.observation.stabilization_progress_frames == 1
    assert first.transition is None
    assert missing.observation.status == "uncertain"
    assert missing.observation.invalid_reason == "index_tip_missing"
    assert missing.observation.candidate_state is None
    assert missing.observation.stable_state is None
    assert missing.observation.stabilization_progress_frames == 0
    assert missing.transition is None
    assert restarted.observation.stabilization_progress_frames == 1
    assert restarted.observation.stable_state is None
    assert restarted.transition is None


def test_distorted_fingertips_are_uncertain_without_realsense_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> object:
        raise ValueError(
            "distortion-aware metric deprojection requires optional pyrealsense2"
        )

    monkeypatch.setattr(deprojection, "_load_pyrealsense2", unavailable)
    intrinsics = CameraIntrinsics(
        width_px=101,
        height_px=101,
        fx_px=100.0,
        fy_px=100.0,
        cx_px=50.0,
        cy_px=50.0,
        distortion_model="inverse_brown_conrady",
        distortion_coefficients=(0.1, 0.01, 0.0, 0.0, 0.0),
    )

    result = MediaPipeHandPoseEstimator().track_normalized_landmarks(
        _frame(color_intrinsics=intrinsics),
        {4: (0.4, 0.5), 8: (0.6, 0.5)},
    )

    assert result.observation.status == "uncertain"
    assert result.observation.candidate_state is None
    assert result.transition is None
    assert result.observation.invalid_reason is not None
    assert "requires optional pyrealsense2" in result.observation.invalid_reason


def test_mediapipe_hands_graph_is_persistent_and_configured_for_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    landmarks[0] = SimpleNamespace(x=0.5, y=0.7)
    landmarks[4] = SimpleNamespace(x=0.4, y=0.5)
    landmarks[5] = SimpleNamespace(x=0.5, y=0.4)
    landmarks[8] = SimpleNamespace(x=0.6, y=0.5)
    landmarks[17] = SimpleNamespace(x=0.7, y=0.7)
    fake_result = SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(score=0.9)])],
    )

    class FakeHands:
        instances: list[FakeHands] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.process_count = 0
            self.closed = False
            self.instances.append(self)

        def process(self, image: np.ndarray) -> object:
            assert image.dtype == np.uint8
            self.process_count += 1
            return fake_result

        def close(self) -> None:
            self.closed = True

    fake_module = SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(Hands=FakeHands))
    )
    monkeypatch.setattr(hand_pose, "_load_mediapipe", lambda: fake_module)
    tracker = MediaPipeHandPoseEstimator(
        minimum_detection_confidence=0.7,
        minimum_tracking_confidence=0.8,
    )

    first = tracker.track(_frame(frame_number=0))
    second = tracker.track(_frame(frame_number=1))

    assert len(FakeHands.instances) == 1
    instance = FakeHands.instances[0]
    assert instance.kwargs == {
        "static_image_mode": False,
        "max_num_hands": 1,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.8,
    }
    assert instance.process_count == 2
    assert first.observation.thumb_landmark_index == 4
    assert first.observation.index_landmark_index == 8
    assert second.hand_pose is not None
    tracker.close()
    assert instance.closed is True


def test_finger_tracking_settings_are_environment_backed(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "FINGER_CLOSE_THRESHOLD_M": "0.025",
            "FINGER_STATE_STABLE_FRAMES": "4",
            "MEDIAPIPE_MINIMUM_DETECTION_CONFIDENCE": "0.75",
            "MEDIAPIPE_MINIMUM_TRACKING_CONFIDENCE": "0.8",
        },
        root=tmp_path,
    )
    assert settings.finger_close_threshold_m == pytest.approx(0.025)
    assert settings.finger_state_stable_frames == 4
    assert settings.mediapipe_minimum_detection_confidence == pytest.approx(0.75)
    assert settings.mediapipe_minimum_tracking_confidence == pytest.approx(0.8)

    with pytest.raises(ValueError):
        Settings.from_env(
            {"ARTIFACT_ROOT": str(tmp_path), "FINGER_CLOSE_THRESHOLD_M": "0"},
            root=tmp_path,
        )
