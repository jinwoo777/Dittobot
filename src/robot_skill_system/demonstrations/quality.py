"""Quality and retake decisions for locally observed demonstrations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import (
    DemonstrationTrajectory,
    PoseSample,
    ProcessedTrajectory,
    TimeInterval,
)
from .preprocessing import preprocess_trajectory
from .trajectory import path_length_m


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for requesting more observation, not robot motion limits."""

    maximum_low_confidence_fraction: float = 0.10
    maximum_outlier_fraction: float = 0.10
    maximum_uncertain_interval_s: float = 0.25
    critical_confidence: float = 0.25

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_low_confidence_fraction", self.maximum_low_confidence_fraction),
            ("maximum_outlier_fraction", self.maximum_outlier_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.maximum_uncertain_interval_s < 0.0:
            raise ValueError("maximum_uncertain_interval_s must be non-negative")
        if not 0.0 <= self.critical_confidence <= 1.0:
            raise ValueError("critical_confidence must be in [0, 1]")


class QualityAssessment(BaseModel):
    """Auditable quality result consumed by segmentation and teaching UIs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: float = Field(ge=0.0, le=1.0)
    reteach_required: bool
    uncertain_intervals: tuple[TimeInterval, ...] = ()
    reason: str | None = None
    recommended_retake_range: tuple[int, int] | None = None
    low_confidence_fraction: float = Field(ge=0.0, le=1.0)
    outlier_fraction: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class RepeatConsistencyConfig:
    """Thresholds for validating multiple demonstrations of the same skill."""

    maximum_rms_position_error_m: float = 0.015
    maximum_position_error_m: float = 0.035
    maximum_duration_fraction_difference: float = 0.35
    maximum_path_length_fraction_difference: float = 0.30
    comparison_sample_count: int = 101

    def __post_init__(self) -> None:
        if min(
            self.maximum_rms_position_error_m,
            self.maximum_position_error_m,
            self.maximum_duration_fraction_difference,
            self.maximum_path_length_fraction_difference,
        ) <= 0.0:
            raise ValueError("repeat consistency thresholds must be positive")
        if self.comparison_sample_count < 3:
            raise ValueError("comparison_sample_count must be at least three")


class ConsistencyIssue(BaseModel):
    """A repeat inconsistency with an original observation timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    demonstration_index: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    reason: str
    position_error_m: float | None = Field(default=None, ge=0.0)


class RepeatConsistencyComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    demonstration_index: int = Field(ge=1)
    rms_position_error_m: float = Field(ge=0.0)
    maximum_position_error_m: float = Field(ge=0.0)
    duration_fraction_difference: float = Field(ge=0.0)
    path_length_fraction_difference: float = Field(ge=0.0)
    consistent: bool


class RepeatConsistencyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    consistent: bool
    confidence: float = Field(ge=0.0, le=1.0)
    comparisons: tuple[RepeatConsistencyComparison, ...]
    issues: tuple[ConsistencyIssue, ...]
    reason: str | None = None


def _interval_duration_s(interval: TimeInterval) -> float:
    return (interval.end_timestamp_ns - interval.start_timestamp_ns) / 1.0e9


def assess_trajectory_quality(
    trajectory: ProcessedTrajectory, config: QualityConfig | None = None
) -> QualityAssessment:
    """Assess whether filtered evidence is adequate or should be re-recorded."""

    effective_config = config or QualityConfig()
    report = trajectory.report
    input_count = max(1, report.input_sample_count + report.interpolated_sample_count)
    low_fraction = min(1.0, report.dropped_low_confidence_count / input_count)
    outlier_fraction = min(1.0, report.dropped_outlier_count / input_count)
    intervals = list(report.low_confidence_intervals)
    intervals.extend(
        TimeInterval(
            start_timestamp_ns=timestamp_ns,
            end_timestamp_ns=timestamp_ns,
            reason="isolated pose outlier removed",
        )
        for timestamp_ns in report.outlier_timestamps_ns
    )
    reasons: list[str] = []
    if low_fraction > effective_config.maximum_low_confidence_fraction:
        reasons.append(
            f"low-confidence fraction {low_fraction:.3f} exceeds "
            f"{effective_config.maximum_low_confidence_fraction:.3f}"
        )
    if outlier_fraction > effective_config.maximum_outlier_fraction:
        reasons.append(
            f"outlier fraction {outlier_fraction:.3f} exceeds "
            f"{effective_config.maximum_outlier_fraction:.3f}"
        )
    if any(
        _interval_duration_s(interval) > effective_config.maximum_uncertain_interval_s
        for interval in report.low_confidence_intervals
    ):
        reasons.append("a low-confidence interval is too long for reliable interpolation")
    if any(
        interval.minimum_confidence is not None
        and interval.minimum_confidence < effective_config.critical_confidence
        for interval in report.low_confidence_intervals
    ):
        reasons.append("critically low pose confidence observed")

    clean_confidence = sum(sample.confidence for sample in trajectory.samples) / len(
        trajectory.samples
    )
    confidence = max(
        0.0,
        min(1.0, clean_confidence * (1.0 - low_fraction) * (1.0 - outlier_fraction)),
    )
    retake_range: tuple[int, int] | None = None
    if intervals:
        largest = max(
            intervals,
            key=lambda interval: interval.end_timestamp_ns - interval.start_timestamp_ns,
        )
        padding_ns = report.nominal_period_ns or 0
        trajectory_start = trajectory.samples[0].timestamp_ns
        trajectory_end = trajectory.samples[-1].timestamp_ns
        retake_range = (
            max(trajectory_start, largest.start_timestamp_ns - padding_ns),
            min(trajectory_end, largest.end_timestamp_ns + padding_ns),
        )
    return QualityAssessment(
        confidence=confidence,
        reteach_required=bool(reasons),
        uncertain_intervals=tuple(intervals),
        reason="; ".join(reasons) if reasons else None,
        recommended_retake_range=retake_range,
        low_confidence_fraction=low_fraction,
        outlier_fraction=outlier_fraction,
    )


def _processed(
    trajectory: ProcessedTrajectory | DemonstrationTrajectory | Sequence[PoseSample],
) -> ProcessedTrajectory:
    if isinstance(trajectory, ProcessedTrajectory):
        return trajectory
    return preprocess_trajectory(trajectory)


def _normalized_positions(
    trajectory: ProcessedTrajectory, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    timestamps_ns = np.asarray(
        [sample.timestamp_ns for sample in trajectory.samples], dtype=np.int64
    )
    if len(timestamps_ns) == 1:
        phase = np.asarray((0.0,), dtype=np.float64)
    else:
        duration_ns = timestamps_ns[-1] - timestamps_ns[0]
        phase = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / max(duration_ns, 1)
    positions = np.asarray(
        [sample.position_m for sample in trajectory.samples], dtype=np.float64
    )
    positions -= positions[0]
    common_phase = np.linspace(0.0, 1.0, sample_count)
    if len(positions) == 1:
        interpolated = np.repeat(positions, sample_count, axis=0)
    else:
        interpolated = np.column_stack(
            [np.interp(common_phase, phase, positions[:, axis]) for axis in range(3)]
        )
    return common_phase, interpolated


def validate_repeat_consistency(
    demonstrations: Sequence[
        ProcessedTrajectory | DemonstrationTrajectory | Sequence[PoseSample]
    ],
    config: RepeatConsistencyConfig | None = None,
) -> RepeatConsistencyResult:
    """Compare repeated, anchor-relative paths and report timestamped deviations."""

    effective_config = config or RepeatConsistencyConfig()
    if len(demonstrations) < 2:
        issue = ConsistencyIssue(
            demonstration_index=0,
            timestamp_ns=0,
            reason="at least two successful demonstrations are required",
        )
        return RepeatConsistencyResult(
            consistent=False,
            confidence=0.0,
            comparisons=(),
            issues=(issue,),
            reason=issue.reason,
        )
    processed = tuple(_processed(trajectory) for trajectory in demonstrations)
    common_phase, reference_positions = _normalized_positions(
        processed[0], effective_config.comparison_sample_count
    )
    reference_duration_s = processed[0].duration_s
    reference_path_length_m = path_length_m(processed[0].samples)
    comparisons: list[RepeatConsistencyComparison] = []
    issues: list[ConsistencyIssue] = []
    confidence_factors: list[float] = []
    for demonstration_index, trajectory in enumerate(processed[1:], start=1):
        _, positions = _normalized_positions(
            trajectory, effective_config.comparison_sample_count
        )
        error_m = np.linalg.norm(positions - reference_positions, axis=1)
        rms_error_m = float(np.sqrt(np.mean(np.square(error_m))))
        maximum_error_m = float(np.max(error_m))
        maximum_index = int(np.argmax(error_m))
        duration_fraction = abs(trajectory.duration_s - reference_duration_s) / max(
            reference_duration_s, 1.0e-9
        )
        current_path_length_m = path_length_m(trajectory.samples)
        path_fraction = abs(current_path_length_m - reference_path_length_m) / max(
            reference_path_length_m, 1.0e-9
        )
        comparison_consistent = (
            rms_error_m <= effective_config.maximum_rms_position_error_m
            and maximum_error_m <= effective_config.maximum_position_error_m
            and duration_fraction <= effective_config.maximum_duration_fraction_difference
            and path_fraction <= effective_config.maximum_path_length_fraction_difference
        )
        comparisons.append(
            RepeatConsistencyComparison(
                demonstration_index=demonstration_index,
                rms_position_error_m=rms_error_m,
                maximum_position_error_m=maximum_error_m,
                duration_fraction_difference=duration_fraction,
                path_length_fraction_difference=path_fraction,
                consistent=comparison_consistent,
            )
        )
        timestamp_index = min(
            len(trajectory.samples) - 1,
            int(round(float(common_phase[maximum_index]) * (len(trajectory.samples) - 1))),
        )
        timestamp_ns = trajectory.samples[timestamp_index].timestamp_ns
        if rms_error_m > effective_config.maximum_rms_position_error_m:
            issues.append(
                ConsistencyIssue(
                    demonstration_index=demonstration_index,
                    timestamp_ns=timestamp_ns,
                    reason="repeat path RMS position deviation exceeds threshold",
                    position_error_m=rms_error_m,
                )
            )
        if maximum_error_m > effective_config.maximum_position_error_m:
            issues.append(
                ConsistencyIssue(
                    demonstration_index=demonstration_index,
                    timestamp_ns=timestamp_ns,
                    reason="repeat path maximum position deviation exceeds threshold",
                    position_error_m=maximum_error_m,
                )
            )
        if duration_fraction > effective_config.maximum_duration_fraction_difference:
            issues.append(
                ConsistencyIssue(
                    demonstration_index=demonstration_index,
                    timestamp_ns=trajectory.samples[-1].timestamp_ns,
                    reason="repeat duration differs beyond threshold",
                )
            )
        if path_fraction > effective_config.maximum_path_length_fraction_difference:
            issues.append(
                ConsistencyIssue(
                    demonstration_index=demonstration_index,
                    timestamp_ns=timestamp_ns,
                    reason="repeat path length differs beyond threshold",
                )
            )
        confidence_factors.append(
            max(
                0.0,
                min(
                    1.0,
                    1.0
                    - 0.5
                    * rms_error_m
                    / effective_config.maximum_rms_position_error_m
                    - 0.2
                    * duration_fraction
                    / effective_config.maximum_duration_fraction_difference
                    - 0.2
                    * path_fraction
                    / effective_config.maximum_path_length_fraction_difference,
                ),
            )
        )
    consistent = all(comparison.consistent for comparison in comparisons)
    reason = "; ".join(dict.fromkeys(issue.reason for issue in issues)) if issues else None
    return RepeatConsistencyResult(
        consistent=consistent,
        confidence=sum(confidence_factors) / len(confidence_factors),
        comparisons=tuple(comparisons),
        issues=tuple(issues),
        reason=reason,
    )


validate_repeat_demonstrations = validate_repeat_consistency


__all__ = [
    "ConsistencyIssue",
    "QualityAssessment",
    "QualityConfig",
    "RepeatConsistencyComparison",
    "RepeatConsistencyConfig",
    "RepeatConsistencyResult",
    "assess_trajectory_quality",
    "validate_repeat_consistency",
    "validate_repeat_demonstrations",
]
