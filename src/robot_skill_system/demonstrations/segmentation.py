"""Local, feature-driven demonstration segmentation.

Semantic labels may reinforce a locally observed state, but never create a
contact or gripper transition without corresponding local evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .models import (
    DemonstrationTrajectory,
    PoseSample,
    ProcessedTrajectory,
    SegmentationResult,
    SegmentState,
    TimeInterval,
    TrajectorySegment,
)
from .preprocessing import (
    PreprocessingConfig,
    TrajectoryPreprocessingError,
    preprocess_trajectory,
)
from .quality import QualityConfig, assess_trajectory_quality


@dataclass(frozen=True)
class SegmentationConfig:
    """Thresholds for observation-state classification."""

    idle_speed_mps: float = 0.004
    gripper_width_rate_mps: float = 0.008
    approach_distance_rate_mps: float = 0.004
    contact_search_distance_m: float = 0.018
    periodic_minimum_direction_reversals: int = 3
    periodic_minimum_path_to_displacement_ratio: float = 2.5
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self) -> None:
        numeric = (
            self.idle_speed_mps,
            self.gripper_width_rate_mps,
            self.approach_distance_rate_mps,
            self.contact_search_distance_m,
            self.periodic_minimum_path_to_displacement_ratio,
        )
        if any(value < 0.0 for value in numeric):
            raise ValueError("segmentation thresholds must be non-negative")
        if self.periodic_minimum_direction_reversals < 1:
            raise ValueError("periodic_minimum_direction_reversals must be positive")


def _is_periodic_motion(
    trajectory: ProcessedTrajectory, config: SegmentationConfig
) -> bool:
    if len(trajectory.samples) < 10:
        return False
    positions = np.asarray(
        [sample.position_m for sample in trajectory.samples], dtype=np.float64
    )
    centered = positions - np.mean(positions, axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinate = centered @ right_vectors[0]
    difference = np.diff(coordinate)
    noise_floor = max(1.0e-5, float(np.max(np.abs(difference))) * 0.03)
    signs = np.sign(difference[np.abs(difference) > noise_floor])
    if len(signs) < 2:
        return False
    direction_reversals = int(np.count_nonzero(signs[1:] != signs[:-1]))
    step_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(np.sum(step_length))
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    path_ratio = path_length / max(displacement, 1.0e-6)
    return bool(
        direction_reversals >= config.periodic_minimum_direction_reversals
        and path_ratio >= config.periodic_minimum_path_to_displacement_ratio
    )


def _optional_rate(
    first: float | None, second: float | None, delta_s: float
) -> float | None:
    if first is None or second is None or delta_s <= 0.0:
        return None
    return (second - first) / delta_s


def _local_states(
    trajectory: ProcessedTrajectory, config: SegmentationConfig
) -> list[SegmentState]:
    sample_count = len(trajectory.samples)
    periodic = _is_periodic_motion(trajectory, config)
    states: list[SegmentState] = []
    contact_seen = False
    for index, sample in enumerate(trajectory.samples):
        previous = trajectory.samples[index - 1] if index > 0 else sample
        delta_s = max(0.0, (sample.timestamp_ns - previous.timestamp_ns) / 1.0e9)
        gripper_rate = _optional_rate(
            previous.gripper_width_m, sample.gripper_width_m, delta_s
        )
        distance_rate = _optional_rate(
            previous.surface_distance_m, sample.surface_distance_m, delta_s
        )
        in_contact = trajectory.contact_candidate[index]
        was_in_contact = trajectory.contact_candidate[index - 1] if index > 0 else False
        speed = trajectory.speed_mps[index]

        if gripper_rate is not None and gripper_rate > config.gripper_width_rate_mps:
            state = SegmentState.GRIPPER_OPEN
        elif gripper_rate is not None and gripper_rate < -config.gripper_width_rate_mps:
            state = SegmentState.GRIPPER_CLOSE
        elif in_contact and not was_in_contact:
            state = SegmentState.CONTACT_SEARCH
            contact_seen = True
        elif in_contact:
            contact_seen = True
            state = (
                SegmentState.CONTACT_MOVE
                if speed > config.idle_speed_mps
                else SegmentState.WAIT
            )
        elif was_in_contact:
            state = SegmentState.CONTACT_RELEASE
        elif periodic and speed > config.idle_speed_mps:
            state = SegmentState.PERIODIC_MOVE
        elif (
            distance_rate is not None
            and distance_rate < -config.approach_distance_rate_mps
        ):
            state = (
                SegmentState.CONTACT_SEARCH
                if sample.surface_distance_m is not None
                and sample.surface_distance_m <= config.contact_search_distance_m
                else SegmentState.APPROACH
            )
        elif (
            contact_seen
            and distance_rate is not None
            and distance_rate > config.approach_distance_rate_mps
        ):
            state = SegmentState.RETRACT
        elif speed <= config.idle_speed_mps:
            if index == sample_count - 1:
                state = SegmentState.END
            elif index == 0:
                state = SegmentState.IDLE
            else:
                state = SegmentState.WAIT
        else:
            state = SegmentState.FREE_SPACE_MOVE

        # A semantic label is supporting evidence only.  Adopt it when it agrees
        # with the broad local motion/contact category; otherwise keep the local
        # result and leave semantic resolution to the orchestration layer.
        semantic = sample.semantic_label
        if semantic is not None:
            motion_states = {
                SegmentState.APPROACH,
                SegmentState.FREE_SPACE_MOVE,
                SegmentState.RETRACT,
                SegmentState.PERIODIC_MOVE,
            }
            contact_states = {
                SegmentState.CONTACT_SEARCH,
                SegmentState.CONTACT_MOVE,
                SegmentState.CONTACT_RELEASE,
            }
            gripper_states = {SegmentState.GRIPPER_OPEN, SegmentState.GRIPPER_CLOSE}
            for family in (motion_states, contact_states, gripper_states):
                if state in family and semantic in family:
                    state = semantic
                    break
        states.append(state)
    return states


def _overlaps(interval: TimeInterval, start_ns: int, end_ns: int) -> bool:
    return interval.start_timestamp_ns <= end_ns and interval.end_timestamp_ns >= start_ns


def _build_segments(
    trajectory: ProcessedTrajectory,
    states: Sequence[SegmentState],
    uncertain_intervals: Sequence[TimeInterval],
) -> tuple[TrajectorySegment, ...]:
    if not states:
        return ()
    ranges: list[tuple[int, int, SegmentState]] = []
    start_index = 0
    for index in range(1, len(states)):
        if states[index] != states[start_index]:
            ranges.append((start_index, index - 1, states[start_index]))
            start_index = index
    ranges.append((start_index, len(states) - 1, states[start_index]))

    output: list[TrajectorySegment] = []
    for first_index, last_index, state in ranges:
        samples = trajectory.samples[first_index : last_index + 1]
        start_ns = samples[0].timestamp_ns
        end_ns = samples[-1].timestamp_ns
        uncertain = any(
            _overlaps(interval, start_ns, end_ns) for interval in uncertain_intervals
        )
        observation_confidence = sum(sample.confidence for sample in samples) / len(samples)
        if uncertain:
            observation_confidence *= 0.5
        output.append(
            TrajectorySegment(
                state=state,
                start_index=first_index,
                end_index=last_index,
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                confidence=max(0.0, min(1.0, observation_confidence)),
                mean_speed_mps=sum(
                    trajectory.speed_mps[first_index : last_index + 1]
                )
                / len(samples),
                mean_curvature_per_m=sum(
                    trajectory.curvature_per_m[first_index : last_index + 1]
                )
                / len(samples),
                uncertain=uncertain,
            )
        )
    return tuple(output)


def _raw_samples(
    trajectory: ProcessedTrajectory | DemonstrationTrajectory | Sequence[PoseSample],
) -> list[PoseSample] | None:
    if isinstance(trajectory, ProcessedTrajectory):
        return None
    if isinstance(trajectory, DemonstrationTrajectory):
        return list(trajectory.samples)
    return list(trajectory)


def segment_trajectory(
    trajectory: ProcessedTrajectory | DemonstrationTrajectory | Sequence[PoseSample],
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Segment a trajectory and return an explicit uncertainty/retake result."""

    effective_config = config or SegmentationConfig()
    if isinstance(trajectory, ProcessedTrajectory):
        processed = trajectory
    else:
        try:
            processed = preprocess_trajectory(trajectory, effective_config.preprocessing)
        except TrajectoryPreprocessingError:
            # Preserve an analyzable shape for the teaching UI even when every
            # sample is low-confidence.  This fallback never hides the retake:
            # its intervals are reconstructed below and the result is forced.
            fallback = replace(effective_config.preprocessing, confidence_threshold=0.0)
            processed = preprocess_trajectory(trajectory, fallback)

    quality = assess_trajectory_quality(processed, effective_config.quality)
    uncertain_intervals = list(quality.uncertain_intervals)
    raw = _raw_samples(trajectory)
    forced_reason: str | None = None
    if raw is not None and not processed.report.low_confidence_intervals:
        threshold = effective_config.preprocessing.confidence_threshold
        low = [sample for sample in raw if sample.confidence < threshold]
        if low:
            interval = TimeInterval(
                start_timestamp_ns=min(sample.timestamp_ns for sample in low),
                end_timestamp_ns=max(sample.timestamp_ns for sample in low),
                reason="pose confidence below preprocessing threshold",
                minimum_confidence=min(sample.confidence for sample in low),
            )
            uncertain_intervals.append(interval)
            forced_reason = "all usable observations were below the confidence threshold"

    states = _local_states(processed, effective_config)
    segments = _build_segments(processed, states, uncertain_intervals)
    reteach_required = quality.reteach_required or forced_reason is not None
    reason = quality.reason
    retake_range = quality.recommended_retake_range
    if forced_reason is not None:
        reason = f"{reason}; {forced_reason}" if reason else forced_reason
        largest = max(
            uncertain_intervals,
            key=lambda interval: interval.end_timestamp_ns - interval.start_timestamp_ns,
        )
        retake_range = (largest.start_timestamp_ns, largest.end_timestamp_ns)
    return SegmentationResult(
        segments=segments,
        reteach_required=reteach_required,
        uncertain_intervals=tuple(uncertain_intervals),
        reason=reason,
        recommended_retake_range=retake_range,
        confidence=quality.confidence if forced_reason is None else min(quality.confidence, 0.2),
    )


__all__ = ["SegmentationConfig", "segment_trajectory"]
