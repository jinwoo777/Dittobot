from __future__ import annotations

import pytest

from robot_skill_system.demonstrations.segmentation import SegmentationConfig
from robot_skill_system.demonstrations.stage_segmentation import (
    DEFAULT_IDLE_SPEED_MPS,
    DemonstrationStage,
    GripActionEndSegmentationConfig,
    GripActionEndSegmentationError,
    StageSegmentationFailureCode,
    StageSegmentationWarningCode,
    TransitionRole,
    segment_grip_action_end,
)
from robot_skill_system.perception.hand_pose import (
    DEFAULT_FINGER_CLOSE_THRESHOLD_M,
    DEFAULT_FINGER_STATE_STABLE_FRAMES,
    FingerGripperState,
    FingerObservation,
)
from robot_skill_system.scene.models import Vector3

_PERIOD_NS = 250_000_000


def _valid_observation(
    frame_index: int,
    state: FingerGripperState,
    *,
    period_ns: int = _PERIOD_NS,
    midpoint_x_m: float | None = None,
) -> FingerObservation:
    distance_m = 0.02 if state is FingerGripperState.CLOSED else 0.05
    midpoint_x = frame_index * 0.001 if midpoint_x_m is None else midpoint_x_m
    half_distance = distance_m / 2.0
    return FingerObservation(
        frame_number=frame_index,
        timestamp_ns=1_000_000_000 + frame_index * period_ns,
        reference_frame="camera_color_optical_frame",
        status="valid",
        thumb_normalized_xy=(0.4, 0.5),
        index_normalized_xy=(0.6, 0.5),
        thumb_pixel_xy=(40, 50),
        index_pixel_xy=(60, 50),
        thumb_depth_m=1.0,
        index_depth_m=1.0,
        thumb_point_camera_m=Vector3(
            x=midpoint_x - half_distance,
            y=0.0,
            z=1.0,
        ),
        index_point_camera_m=Vector3(
            x=midpoint_x + half_distance,
            y=0.0,
            z=1.0,
        ),
        midpoint_camera_m=Vector3(x=midpoint_x, y=0.0, z=1.0),
        distance_m=distance_m,
        candidate_state=state,
        stabilization_progress_frames=1,
        required_stable_frames=DEFAULT_FINGER_STATE_STABLE_FRAMES,
        confidence=0.9,
    )


def _uncertain_observation(
    frame_index: int,
    *,
    period_ns: int = _PERIOD_NS,
    reason: str = "depth_missing",
) -> FingerObservation:
    return FingerObservation(
        frame_number=frame_index,
        timestamp_ns=1_000_000_000 + frame_index * period_ns,
        reference_frame="camera_color_optical_frame",
        status="uncertain",
        stabilization_progress_frames=0,
        required_stable_frames=DEFAULT_FINGER_STATE_STABLE_FRAMES,
        invalid_reason=reason,
        confidence=0.0,
    )


def _observations(states: list[FingerGripperState]) -> list[FingerObservation]:
    return [_valid_observation(index, state) for index, state in enumerate(states)]


def test_first_close_last_open_and_post_open_idle_define_three_stages() -> None:
    states = (
        [FingerGripperState.OPEN] * 3
        + [FingerGripperState.CLOSED] * 3
        + [FingerGripperState.OPEN] * 3
        + [FingerGripperState.CLOSED] * 8
        + [FingerGripperState.OPEN] * 6
    )
    observations = _observations(states)
    speeds = [0.02] * len(observations)
    speeds[20:23] = [0.004, 0.003, 0.002]

    result = segment_grip_action_end(observations, speed_mps=speeds)

    assert [interval.stage for interval in result.intervals] == [
        DemonstrationStage.GRIP,
        DemonstrationStage.ACTION,
        DemonstrationStage.END_MOTION,
    ]
    assert (result.grip.start_frame_index, result.grip.end_frame_index) == (0, 5)
    # Final stable open is frame 19; its one-second preroll is frame 15.
    assert (result.action.start_frame_index, result.action.end_frame_index) == (5, 15)
    assert (result.end_motion.start_frame_index, result.end_motion.end_frame_index) == (
        15,
        22,
    )
    assert result.evidence.first_stable_close.frame_index == 5
    assert result.evidence.final_stable_open.frame_index == 19
    assert result.evidence.idle_stop is not None
    assert result.evidence.idle_stop.frame_indices == (20, 21, 22)
    assert result.warnings == ()

    intermediate = result.evidence.action_intermediate_transitions
    assert [(item.frame_index, item.transition.state) for item in intermediate] == [
        (8, FingerGripperState.OPEN),
        (11, FingerGripperState.CLOSED),
    ]
    assert all(item.role is TransitionRole.ACTION_INTERMEDIATE for item in intermediate)


def test_short_action_clamps_end_preroll_to_first_valid_frame_after_close() -> None:
    states = [FingerGripperState.CLOSED] * 3 + [FingerGripperState.OPEN] * 6
    observations = _observations(states)
    speeds = [0.02] * len(observations)
    speeds[6:9] = [0.004, 0.004, 0.004]

    result = segment_grip_action_end(observations, speed_mps=speeds)

    assert result.grip.end_frame_index == 2
    assert result.action.start_frame_index == 2
    assert result.action.end_frame_index == 3
    assert result.end_motion.start_frame_index == 3
    assert result.end_motion.end_frame_index == 8
    assert [warning.code for warning in result.warnings] == [
        StageSegmentationWarningCode.END_PREROLL_CLAMPED_AFTER_CLOSE
    ]


def test_idle_run_requires_three_consecutive_valid_samples_and_preserves_invalid() -> None:
    states = [FingerGripperState.CLOSED] * 3 + [FingerGripperState.OPEN] * 10
    observations = _observations(states)
    observations[7] = _uncertain_observation(7, reason="mediapipe_landmarks_missing")
    speeds = [0.02] * len(observations)
    speeds[6:13] = [0.003, 0.003, 0.003, 0.02, 0.004, 0.003, 0.002]

    result = segment_grip_action_end(observations, speed_mps=speeds)

    assert result.evidence.idle_stop is not None
    # The uncertain frame at 7 invalidates the first low-speed run.  A fast
    # frame at 9 then resets it again, so confirmation is 10..12.
    assert result.evidence.idle_stop.frame_indices == (10, 11, 12)
    assert result.end_motion.end_frame_index == 12
    assert result.evidence.invalid_observations == (observations[7],)
    preserved = result.evidence.frames[7]
    assert preserved.observation.invalid_reason == "mediapipe_landmarks_missing"
    assert preserved.speed_mps == pytest.approx(0.003)
    assert preserved.speed_valid_for_idle_detection is False
    assert preserved.speed_invalid_reason == (
        "finger_observation_uncertain:mediapipe_landmarks_missing"
    )


def test_missing_stable_close_hard_fails_with_invalid_evidence() -> None:
    observations = _observations([FingerGripperState.OPEN] * 5)
    observations.insert(2, _uncertain_observation(2, reason="sparse_depth"))
    # Keep frame numbers and timestamps strictly increasing after insertion.
    observations = [
        observation.model_copy(
            update={
                "frame_number": index,
                "timestamp_ns": 1_000_000_000 + index * _PERIOD_NS,
            }
        )
        for index, observation in enumerate(observations)
    ]

    with pytest.raises(GripActionEndSegmentationError) as captured:
        segment_grip_action_end(observations, speed_mps=[0.01] * len(observations))

    failure = captured.value.failure
    assert failure.code is StageSegmentationFailureCode.STABLE_CLOSE_MISSING
    assert failure.invalid_observations[0].invalid_reason == "sparse_depth"
    assert len(failure.frames) == len(observations)


def test_missing_stable_open_after_close_hard_fails() -> None:
    observations = _observations([FingerGripperState.CLOSED] * 8)

    with pytest.raises(GripActionEndSegmentationError) as captured:
        segment_grip_action_end(observations, speed_mps=[0.01] * len(observations))

    assert (
        captured.value.failure.code
        is StageSegmentationFailureCode.STABLE_OPEN_AFTER_CLOSE_MISSING
    )


def test_recording_end_is_candidate_when_post_open_idle_is_absent() -> None:
    states = [FingerGripperState.CLOSED] * 3 + [FingerGripperState.OPEN] * 7
    observations = _observations(states)

    result = segment_grip_action_end(observations, speed_mps=[0.02] * len(observations))

    assert result.evidence.idle_stop is None
    assert result.end_motion.end_frame_index == len(observations) - 1
    assert StageSegmentationWarningCode.POST_OPEN_IDLE_STOP_NOT_FOUND in {
        warning.code for warning in result.warnings
    }


def test_defaults_share_local_hand_and_trajectory_segmentation_policy() -> None:
    config = GripActionEndSegmentationConfig()

    assert config.finger_close_threshold_m == DEFAULT_FINGER_CLOSE_THRESHOLD_M
    assert config.finger_state_stable_frames == DEFAULT_FINGER_STATE_STABLE_FRAMES
    assert config.idle_stable_samples == DEFAULT_FINGER_STATE_STABLE_FRAMES
    assert config.idle_speed_mps == SegmentationConfig().idle_speed_mps
    assert DEFAULT_IDLE_SPEED_MPS == 0.004


def test_custom_thresholds_and_stability_counts_are_applied_locally() -> None:
    states = [FingerGripperState.CLOSED] * 2 + [FingerGripperState.OPEN] * 4
    observations = _observations(states)
    config = GripActionEndSegmentationConfig(
        finger_state_stable_frames=2,
        idle_stable_samples=2,
        idle_speed_mps=0.01,
        end_motion_preroll_s=0.0,
    )
    speeds = [0.02] * len(observations)
    speeds[4:6] = [0.01, 0.009]

    result = segment_grip_action_end(
        observations,
        speed_mps=speeds,
        config=config,
    )

    assert result.evidence.first_stable_close.frame_index == 1
    assert result.evidence.final_stable_open.frame_index == 3
    assert result.evidence.idle_stop is not None
    assert result.evidence.idle_stop.frame_indices == (4, 5)


def test_speed_is_derived_from_consecutive_metric_midpoints_when_not_supplied() -> None:
    states = [FingerGripperState.CLOSED] * 3 + [FingerGripperState.OPEN] * 7
    observations = [
        _valid_observation(index, state, midpoint_x_m=0.0)
        for index, state in enumerate(states)
    ]

    result = segment_grip_action_end(observations)

    assert result.evidence.idle_stop is not None
    assert result.evidence.idle_stop.frame_indices == (6, 7, 8)
    assert result.evidence.frames[0].speed_source == "unavailable"
    assert result.evidence.frames[1].speed_source == "derived_midpoint"
    assert result.evidence.frames[1].speed_mps == pytest.approx(0.0)
