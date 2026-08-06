"""Deterministic Grip -> Action -> End demonstration boundaries.

The boundary decisions in this module are derived only from local metric
fingertip observations and local motion speeds.  Semantic labels and model
output are deliberately not accepted as inputs.

Stage intervals share their boundary frame.  This keeps the closing pose at
the end of Grip and the beginning of Action, and the pre-release pose at the
end of Action and the beginning of End.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robot_skill_system.perception.hand_pose import (
    DEFAULT_FINGER_CLOSE_THRESHOLD_M,
    DEFAULT_FINGER_STATE_STABLE_FRAMES,
    FingerGripperState,
    FingerObservation,
    FingerStateStabilizer,
    FingerStateTransition,
    classify_gripper_state,
)

from .segmentation import SegmentationConfig

DEFAULT_END_MOTION_PREROLL_S = 1.0
DEFAULT_IDLE_SPEED_MPS = SegmentationConfig().idle_speed_mps
DEFAULT_IDLE_STABLE_SAMPLES = DEFAULT_FINGER_STATE_STABLE_FRAMES


class _FrozenStrictModel(BaseModel):
    """Immutable, finite, extra-forbidding contract for segmentation artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class DemonstrationStage(str, Enum):
    """The three independently learned parts of a demonstration."""

    GRIP = "grip"
    ACTION = "action"
    END_MOTION = "end_motion"


class TransitionRole(str, Enum):
    """How a locally derived stable transition participates in segmentation."""

    PRE_GRIP_CONTEXT = "pre_grip_context"
    FIRST_STABLE_CLOSE = "first_stable_close"
    ACTION_INTERMEDIATE = "action_intermediate"
    FINAL_STABLE_OPEN = "final_stable_open"
    POST_FINAL_OPEN = "post_final_open"


class StageSegmentationWarningCode(str, Enum):
    """Non-structural fallback decisions retained in the result."""

    END_PREROLL_CLAMPED_AFTER_CLOSE = "end_preroll_clamped_after_close"
    POST_OPEN_IDLE_STOP_NOT_FOUND = "post_open_idle_stop_not_found"


class StageSegmentationFailureCode(str, Enum):
    """Structural failures for which three stages cannot be produced."""

    STABLE_CLOSE_MISSING = "stable_close_missing"
    STABLE_OPEN_AFTER_CLOSE_MISSING = "stable_open_after_close_missing"


class GripActionEndSegmentationConfig(_FrozenStrictModel):
    """Thresholds for deterministic local stage segmentation.

    Finger defaults are shared with :mod:`perception.hand_pose`; the idle
    threshold is shared with the existing local trajectory segmenter.
    """

    finger_close_threshold_m: float = Field(
        default=DEFAULT_FINGER_CLOSE_THRESHOLD_M, gt=0.0
    )
    finger_state_stable_frames: int = Field(
        default=DEFAULT_FINGER_STATE_STABLE_FRAMES, ge=1
    )
    end_motion_preroll_s: float = Field(default=DEFAULT_END_MOTION_PREROLL_S, ge=0.0)
    idle_speed_mps: float = Field(default=DEFAULT_IDLE_SPEED_MPS, ge=0.0)
    idle_stable_samples: int = Field(default=DEFAULT_IDLE_STABLE_SAMPLES, ge=1)


class StageFrameEvidence(_FrozenStrictModel):
    """One preserved recording frame and its speed eligibility."""

    frame_index: int = Field(ge=0)
    observation: FingerObservation
    speed_mps: float | None = Field(default=None, ge=0.0)
    speed_source: Literal["provided", "derived_midpoint", "unavailable"]
    speed_valid_for_idle_detection: bool
    speed_invalid_reason: str | None = None

    @model_validator(mode="after")
    def _validate_speed_evidence(self) -> StageFrameEvidence:
        if self.speed_source == "unavailable":
            if self.speed_mps is not None:
                raise ValueError("unavailable speed evidence cannot carry speed_mps")
            if self.speed_valid_for_idle_detection:
                raise ValueError("unavailable speed evidence cannot be valid for idle detection")
            if self.speed_invalid_reason is None:
                raise ValueError("unavailable speed evidence requires a reason")
            return self
        if self.speed_mps is None:
            raise ValueError("provided or derived speed evidence requires speed_mps")
        if self.speed_valid_for_idle_detection and self.speed_invalid_reason is not None:
            raise ValueError("valid speed evidence cannot carry an invalid reason")
        if not self.speed_valid_for_idle_detection and self.speed_invalid_reason is None:
            raise ValueError("excluded speed evidence requires an invalid reason")
        return self


class StableTransitionEvidence(_FrozenStrictModel):
    """A stable local gripper transition with its recording-frame role."""

    frame_index: int = Field(ge=0)
    transition: FingerStateTransition
    role: TransitionRole


class StableIdleStopEvidence(_FrozenStrictModel):
    """The consecutive low-speed samples that confirm the End terminal."""

    start_frame_index: int = Field(ge=0)
    confirmation_frame_index: int = Field(ge=0)
    start_timestamp_ns: int = Field(ge=0)
    confirmation_timestamp_ns: int = Field(ge=0)
    frame_indices: tuple[int, ...] = Field(min_length=1)
    speed_mps: tuple[float, ...] = Field(min_length=1)
    threshold_mps: float = Field(ge=0.0)
    required_sample_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_idle_run(self) -> StableIdleStopEvidence:
        if self.confirmation_frame_index < self.start_frame_index:
            raise ValueError("idle confirmation cannot precede idle start")
        if self.confirmation_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("idle confirmation timestamp cannot precede idle start")
        if len(self.frame_indices) != self.required_sample_count:
            raise ValueError("idle frame count must equal required_sample_count")
        if len(self.speed_mps) != self.required_sample_count:
            raise ValueError("idle speed count must equal required_sample_count")
        if self.frame_indices[0] != self.start_frame_index:
            raise ValueError("first idle frame must match start_frame_index")
        if self.frame_indices[-1] != self.confirmation_frame_index:
            raise ValueError("last idle frame must match confirmation_frame_index")
        if any(speed > self.threshold_mps for speed in self.speed_mps):
            raise ValueError("idle evidence contains a speed above threshold")
        return self


class StageInterval(_FrozenStrictModel):
    """Inclusive frame and timestamp range for one learned stage."""

    stage: DemonstrationStage
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(ge=0)
    start_frame_number: int = Field(ge=0)
    end_frame_number: int = Field(ge=0)
    start_timestamp_ns: int = Field(ge=0)
    end_timestamp_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_interval(self) -> StageInterval:
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("stage end frame cannot precede its start frame")
        if self.end_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("stage end timestamp cannot precede its start timestamp")
        return self


class StageSegmentationWarning(_FrozenStrictModel):
    """Auditable explanation for a deterministic fallback."""

    code: StageSegmentationWarningCode
    message: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)


class StageSegmentationEvidence(_FrozenStrictModel):
    """All local evidence, including frames that were unusable."""

    frames: tuple[StageFrameEvidence, ...] = Field(min_length=1)
    stable_transitions: tuple[StableTransitionEvidence, ...]
    invalid_observations: tuple[FingerObservation, ...]
    first_valid_frame_index: int = Field(ge=0)
    first_stable_close: StableTransitionEvidence
    final_stable_open: StableTransitionEvidence
    action_intermediate_transitions: tuple[StableTransitionEvidence, ...]
    idle_stop: StableIdleStopEvidence | None = None


class GripActionEndSegmentationResult(_FrozenStrictModel):
    """Successful deterministic decomposition into exactly three stages."""

    grip: StageInterval
    action: StageInterval
    end_motion: StageInterval
    evidence: StageSegmentationEvidence
    warnings: tuple[StageSegmentationWarning, ...] = ()

    @model_validator(mode="after")
    def _validate_stage_order(self) -> GripActionEndSegmentationResult:
        if self.grip.stage is not DemonstrationStage.GRIP:
            raise ValueError("grip interval must have the grip stage")
        if self.action.stage is not DemonstrationStage.ACTION:
            raise ValueError("action interval must have the action stage")
        if self.end_motion.stage is not DemonstrationStage.END_MOTION:
            raise ValueError("end_motion interval must have the end_motion stage")
        if self.grip.end_frame_index != self.action.start_frame_index:
            raise ValueError("Grip and Action must share the stable-close boundary")
        if self.action.end_frame_index != self.end_motion.start_frame_index:
            raise ValueError("Action and End must share the pre-release boundary")
        if self.end_motion.end_frame_index < self.end_motion.start_frame_index:
            raise ValueError("End terminal cannot precede its start")
        return self

    @property
    def intervals(self) -> tuple[StageInterval, StageInterval, StageInterval]:
        """Return the stages in their only executable order."""

        return self.grip, self.action, self.end_motion


class StageSegmentationFailure(_FrozenStrictModel):
    """Typed failure artifact retained when a required transition is absent."""

    code: StageSegmentationFailureCode
    message: str = Field(min_length=1)
    frames: tuple[StageFrameEvidence, ...] = Field(min_length=1)
    stable_transitions: tuple[StableTransitionEvidence, ...]
    invalid_observations: tuple[FingerObservation, ...]


class GripActionEndSegmentationError(ValueError):
    """Raised when local evidence cannot structurally produce all three stages."""

    def __init__(self, failure: StageSegmentationFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


def _validate_inputs(
    observations: Sequence[FingerObservation],
    speed_mps: Sequence[float | None] | None,
) -> tuple[FingerObservation, ...]:
    preserved = tuple(observations)
    if not preserved:
        raise ValueError("stage segmentation requires at least one finger observation")
    for previous, current in zip(preserved, preserved[1:], strict=False):
        if current.timestamp_ns <= previous.timestamp_ns:
            raise ValueError("finger observation timestamps must be strictly increasing")
        if current.frame_number <= previous.frame_number:
            raise ValueError("finger observation frame numbers must be strictly increasing")
    if speed_mps is not None:
        if len(speed_mps) != len(preserved):
            raise ValueError("speed_mps must contain one value per finger observation")
        for speed in speed_mps:
            if speed is not None and (not math.isfinite(speed) or speed < 0.0):
                raise ValueError("speed_mps values must be finite and non-negative")
    return preserved


def _midpoint_speed_mps(
    previous: FingerObservation,
    current: FingerObservation,
) -> float | None:
    if previous.status != "valid" or current.status != "valid":
        return None
    if previous.midpoint_camera_m is None or current.midpoint_camera_m is None:
        return None
    delta_s = (current.timestamp_ns - previous.timestamp_ns) / 1.0e9
    if delta_s <= 0.0:
        return None
    delta_x = current.midpoint_camera_m.x - previous.midpoint_camera_m.x
    delta_y = current.midpoint_camera_m.y - previous.midpoint_camera_m.y
    delta_z = current.midpoint_camera_m.z - previous.midpoint_camera_m.z
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) / delta_s


def _build_frame_evidence(
    observations: tuple[FingerObservation, ...],
    supplied_speed_mps: Sequence[float | None] | None,
) -> tuple[StageFrameEvidence, ...]:
    frames: list[StageFrameEvidence] = []
    for frame_index, observation in enumerate(observations):
        if supplied_speed_mps is not None:
            speed = supplied_speed_mps[frame_index]
            source: Literal["provided", "derived_midpoint", "unavailable"] = (
                "provided" if speed is not None else "unavailable"
            )
            unavailable_reason = "provided_speed_missing" if speed is None else None
        elif frame_index == 0:
            speed = None
            source = "unavailable"
            unavailable_reason = "previous_frame_unavailable"
        else:
            speed = _midpoint_speed_mps(observations[frame_index - 1], observation)
            source = "derived_midpoint" if speed is not None else "unavailable"
            unavailable_reason = (
                None if speed is not None else "consecutive_valid_midpoints_unavailable"
            )

        valid_for_idle = speed is not None and observation.status == "valid"
        invalid_reason = unavailable_reason
        if observation.status != "valid":
            invalid_reason = f"finger_observation_uncertain:{observation.invalid_reason}"
        frames.append(
            StageFrameEvidence(
                frame_index=frame_index,
                observation=observation,
                speed_mps=speed,
                speed_source=source,
                speed_valid_for_idle_detection=valid_for_idle,
                speed_invalid_reason=None if valid_for_idle else invalid_reason,
            )
        )
    return tuple(frames)


def _derive_stable_transitions(
    observations: tuple[FingerObservation, ...],
    config: GripActionEndSegmentationConfig,
) -> list[tuple[int, FingerStateTransition]]:
    stabilizer = FingerStateStabilizer(stable_frames=config.finger_state_stable_frames)
    transitions: list[tuple[int, FingerStateTransition]] = []
    for frame_index, observation in enumerate(observations):
        if observation.status != "valid" or observation.distance_m is None:
            decision = stabilizer.update(
                None,
                frame_number=observation.frame_number,
                timestamp_ns=observation.timestamp_ns,
            )
        else:
            state = classify_gripper_state(
                observation.distance_m,
                close_threshold_m=config.finger_close_threshold_m,
            )
            decision = stabilizer.update(
                state,
                frame_number=observation.frame_number,
                timestamp_ns=observation.timestamp_ns,
                distance_m=observation.distance_m,
                midpoint_camera_m=observation.midpoint_camera_m,
            )
        if decision.transition is not None:
            transitions.append((frame_index, decision.transition))
    return transitions


def _failure(
    code: StageSegmentationFailureCode,
    message: str,
    frames: tuple[StageFrameEvidence, ...],
    transitions: Sequence[tuple[int, FingerStateTransition]],
    invalid_observations: tuple[FingerObservation, ...],
) -> GripActionEndSegmentationError:
    return GripActionEndSegmentationError(
        StageSegmentationFailure(
            code=code,
            message=message,
            frames=frames,
            stable_transitions=tuple(
                StableTransitionEvidence(
                    frame_index=frame_index,
                    transition=transition,
                    role=TransitionRole.PRE_GRIP_CONTEXT,
                )
                for frame_index, transition in transitions
            ),
            invalid_observations=invalid_observations,
        )
    )


def _transition_evidence(
    transitions: Sequence[tuple[int, FingerStateTransition]],
    close_index: int,
    final_open_index: int,
) -> tuple[StableTransitionEvidence, ...]:
    evidence: list[StableTransitionEvidence] = []
    for frame_index, transition in transitions:
        if frame_index == close_index and transition.state is FingerGripperState.CLOSED:
            role = TransitionRole.FIRST_STABLE_CLOSE
        elif frame_index == final_open_index and transition.state is FingerGripperState.OPEN:
            role = TransitionRole.FINAL_STABLE_OPEN
        elif close_index < frame_index < final_open_index:
            role = TransitionRole.ACTION_INTERMEDIATE
        elif frame_index > final_open_index:
            role = TransitionRole.POST_FINAL_OPEN
        else:
            role = TransitionRole.PRE_GRIP_CONTEXT
        evidence.append(
            StableTransitionEvidence(
                frame_index=frame_index,
                transition=transition,
                role=role,
            )
        )
    return tuple(evidence)


def _find_end_start_index(
    frames: tuple[StageFrameEvidence, ...],
    close_index: int,
    final_open_timestamp_ns: int,
    config: GripActionEndSegmentationConfig,
) -> tuple[int, StageSegmentationWarning | None]:
    preroll_ns = int(round(config.end_motion_preroll_s * 1.0e9))
    target_timestamp_ns = max(0, final_open_timestamp_ns - preroll_ns)
    candidates = [
        frame.frame_index
        for frame in frames
        if frame.frame_index > close_index
        and frame.observation.status == "valid"
        and frame.observation.timestamp_ns <= target_timestamp_ns
    ]
    if candidates:
        return candidates[-1], None

    # A confirmed final open is itself valid, so a valid frame after close must
    # exist.  Keeping this assertion local makes the clamp rule explicit.
    end_start_index = next(
        frame.frame_index
        for frame in frames
        if frame.frame_index > close_index and frame.observation.status == "valid"
    )
    selected = frames[end_start_index]
    warning = StageSegmentationWarning(
        code=StageSegmentationWarningCode.END_PREROLL_CLAMPED_AFTER_CLOSE,
        message=(
            "the one-second End preroll preceded the stable close and was clamped "
            "to the first valid frame after close"
        ),
        frame_index=end_start_index,
        timestamp_ns=selected.observation.timestamp_ns,
    )
    return end_start_index, warning


def _find_idle_stop(
    frames: tuple[StageFrameEvidence, ...],
    final_open_index: int,
    config: GripActionEndSegmentationConfig,
) -> StableIdleStopEvidence | None:
    run: list[StageFrameEvidence] = []
    for frame in frames[final_open_index + 1 :]:
        if (
            frame.speed_valid_for_idle_detection
            and frame.speed_mps is not None
            and frame.speed_mps <= config.idle_speed_mps
        ):
            run.append(frame)
            if len(run) == config.idle_stable_samples:
                return StableIdleStopEvidence(
                    start_frame_index=run[0].frame_index,
                    confirmation_frame_index=run[-1].frame_index,
                    start_timestamp_ns=run[0].observation.timestamp_ns,
                    confirmation_timestamp_ns=run[-1].observation.timestamp_ns,
                    frame_indices=tuple(item.frame_index for item in run),
                    speed_mps=tuple(item.speed_mps for item in run if item.speed_mps is not None),
                    threshold_mps=config.idle_speed_mps,
                    required_sample_count=config.idle_stable_samples,
                )
        else:
            run.clear()
    return None


def _interval(
    stage: DemonstrationStage,
    frames: tuple[StageFrameEvidence, ...],
    start_index: int,
    end_index: int,
) -> StageInterval:
    start = frames[start_index].observation
    end = frames[end_index].observation
    return StageInterval(
        stage=stage,
        start_frame_index=start_index,
        end_frame_index=end_index,
        start_frame_number=start.frame_number,
        end_frame_number=end.frame_number,
        start_timestamp_ns=start.timestamp_ns,
        end_timestamp_ns=end.timestamp_ns,
    )


def segment_grip_action_end(
    observations: Sequence[FingerObservation],
    *,
    speed_mps: Sequence[float | None] | None = None,
    config: GripActionEndSegmentationConfig | None = None,
) -> GripActionEndSegmentationResult:
    """Split local metric evidence into Grip, Action, and End intervals.

    The first stable close closes Grip.  Only the last stable open after that
    close is the final release; every earlier transition after close remains
    Action evidence.  Missing either required transition raises
    :class:`GripActionEndSegmentationError` with all invalid input observations
    retained on its typed ``failure`` artifact.

    If ``speed_mps`` is omitted, speed is derived from consecutive valid metric
    fingertip midpoints.  A supplied speed for an uncertain finger observation
    is preserved but excluded from idle-stop confirmation.
    """

    effective_config = config or GripActionEndSegmentationConfig()
    preserved = _validate_inputs(observations, speed_mps)
    frames = _build_frame_evidence(preserved, speed_mps)
    invalid_observations = tuple(
        observation for observation in preserved if observation.status != "valid"
    )
    transitions = _derive_stable_transitions(preserved, effective_config)

    close = next(
        (
            item
            for item in transitions
            if item[1].state is FingerGripperState.CLOSED
        ),
        None,
    )
    if close is None:
        raise _failure(
            StageSegmentationFailureCode.STABLE_CLOSE_MISSING,
            "a stable close was not observed; Grip, Action, and End cannot be separated",
            frames,
            transitions,
            invalid_observations,
        )
    close_index, _ = close

    opens_after_close = [
        item
        for item in transitions
        if item[0] > close_index and item[1].state is FingerGripperState.OPEN
    ]
    if not opens_after_close:
        raise _failure(
            StageSegmentationFailureCode.STABLE_OPEN_AFTER_CLOSE_MISSING,
            "a stable open after close was not observed; the final release is unknown",
            frames,
            transitions,
            invalid_observations,
        )
    final_open_index, final_open_transition = opens_after_close[-1]

    stable_transitions = _transition_evidence(
        transitions,
        close_index,
        final_open_index,
    )
    first_close_evidence = next(
        item
        for item in stable_transitions
        if item.role is TransitionRole.FIRST_STABLE_CLOSE
    )
    final_open_evidence = next(
        item
        for item in stable_transitions
        if item.role is TransitionRole.FINAL_STABLE_OPEN
    )
    action_intermediate = tuple(
        item
        for item in stable_transitions
        if item.role is TransitionRole.ACTION_INTERMEDIATE
    )

    first_valid_index = next(
        frame.frame_index for frame in frames if frame.observation.status == "valid"
    )
    end_start_index, clamp_warning = _find_end_start_index(
        frames,
        close_index,
        final_open_transition.timestamp_ns,
        effective_config,
    )
    idle_stop = _find_idle_stop(frames, final_open_index, effective_config)
    warnings: list[StageSegmentationWarning] = []
    if clamp_warning is not None:
        warnings.append(clamp_warning)
    if idle_stop is None:
        end_index = len(frames) - 1
        terminal = frames[end_index].observation
        warnings.append(
            StageSegmentationWarning(
                code=StageSegmentationWarningCode.POST_OPEN_IDLE_STOP_NOT_FOUND,
                message=(
                    "no consecutive post-open idle run was found; End uses the "
                    "recording terminal frame as a candidate boundary"
                ),
                frame_index=end_index,
                timestamp_ns=terminal.timestamp_ns,
            )
        )
    else:
        end_index = idle_stop.confirmation_frame_index

    evidence = StageSegmentationEvidence(
        frames=frames,
        stable_transitions=stable_transitions,
        invalid_observations=invalid_observations,
        first_valid_frame_index=first_valid_index,
        first_stable_close=first_close_evidence,
        final_stable_open=final_open_evidence,
        action_intermediate_transitions=action_intermediate,
        idle_stop=idle_stop,
    )
    return GripActionEndSegmentationResult(
        grip=_interval(
            DemonstrationStage.GRIP,
            frames,
            first_valid_index,
            close_index,
        ),
        action=_interval(
            DemonstrationStage.ACTION,
            frames,
            close_index,
            end_start_index,
        ),
        end_motion=_interval(
            DemonstrationStage.END_MOTION,
            frames,
            end_start_index,
            end_index,
        ),
        evidence=evidence,
        warnings=tuple(warnings),
    )


__all__ = [
    "DEFAULT_END_MOTION_PREROLL_S",
    "DEFAULT_IDLE_SPEED_MPS",
    "DEFAULT_IDLE_STABLE_SAMPLES",
    "DemonstrationStage",
    "GripActionEndSegmentationConfig",
    "GripActionEndSegmentationError",
    "GripActionEndSegmentationResult",
    "StableIdleStopEvidence",
    "StableTransitionEvidence",
    "StageFrameEvidence",
    "StageInterval",
    "StageSegmentationEvidence",
    "StageSegmentationFailure",
    "StageSegmentationFailureCode",
    "StageSegmentationWarning",
    "StageSegmentationWarningCode",
    "TransitionRole",
    "segment_grip_action_end",
]
