"""Deterministic preprocessing for timestamped pose demonstrations.

The ordering in :func:`preprocess_trajectory` is intentional: timestamps are
sorted, short gaps are interpolated, low-confidence and isolated outlier samples
are removed, position/orientation are smoothed separately, and only then are
derivatives calculated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import (
    DemonstrationTrajectory,
    PoseSample,
    PreprocessingReport,
    ProcessedTrajectory,
    TimeInterval,
    Vector3,
)

FloatArray = NDArray[np.float64]


class TrajectoryPreprocessingError(ValueError):
    """Raised when no usable, temporally valid trajectory can be produced."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Local, non-robot thresholds for cleaning camera observations."""

    confidence_threshold: float = 0.55
    maximum_interpolation_gap_s: float = 0.35
    nominal_period_ns: int | None = None
    outlier_absolute_threshold_m: float = 0.025
    outlier_mad_scale: float = 6.0
    smoothing_window_samples: int = 7
    smoothing_polynomial_order: int = 2
    contact_distance_threshold_m: float = 0.006

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.maximum_interpolation_gap_s < 0.0:
            raise ValueError("maximum_interpolation_gap_s must be non-negative")
        if self.nominal_period_ns is not None and self.nominal_period_ns <= 0:
            raise ValueError("nominal_period_ns must be positive")
        if self.outlier_absolute_threshold_m <= 0.0:
            raise ValueError("outlier_absolute_threshold_m must be positive")
        if self.outlier_mad_scale <= 0.0:
            raise ValueError("outlier_mad_scale must be positive")
        if self.smoothing_window_samples < 1:
            raise ValueError("smoothing_window_samples must be positive")
        if self.smoothing_polynomial_order < 0:
            raise ValueError("smoothing_polynomial_order must be non-negative")


def _coerce_samples(
    trajectory: DemonstrationTrajectory | Sequence[PoseSample],
) -> list[PoseSample]:
    if isinstance(trajectory, DemonstrationTrajectory):
        return list(trajectory.samples)
    return list(trajectory)


def _sort_and_deduplicate(samples: Sequence[PoseSample]) -> tuple[list[PoseSample], int]:
    """Sort samples and keep the highest-confidence observation per timestamp."""

    by_timestamp: dict[int, PoseSample] = {}
    duplicate_count = 0
    for sample in samples:
        current = by_timestamp.get(sample.timestamp_ns)
        if current is None:
            by_timestamp[sample.timestamp_ns] = sample
            continue
        duplicate_count += 1
        if sample.confidence > current.confidence:
            by_timestamp[sample.timestamp_ns] = sample
    return [by_timestamp[key] for key in sorted(by_timestamp)], duplicate_count


def sort_samples(samples: Sequence[PoseSample]) -> list[PoseSample]:
    """Public convenience wrapper for timestamp sort/deduplication."""

    sorted_samples, _ = _sort_and_deduplicate(samples)
    return sorted_samples


def _nominal_period_ns(samples: Sequence[PoseSample], configured: int | None) -> int | None:
    if configured is not None:
        return configured
    if len(samples) < 2:
        return None
    timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    differences = np.diff(timestamps)
    positive = differences[differences > 0]
    if len(positive) == 0:
        return None
    return max(1, int(round(float(np.median(positive)))))


def _nlerp_quaternion(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    value = (1.0 - fraction) * q0 + fraction * q1
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        value = q0
        norm = float(np.linalg.norm(value))
    normalized = value / norm
    return tuple(float(component) for component in normalized)  # type: ignore[return-value]


def _interpolate_optional_float(
    first: float | None, second: float | None, fraction: float
) -> float | None:
    if first is None or second is None:
        return None
    return float((1.0 - fraction) * first + fraction * second)


def _interpolate_pair(first: PoseSample, second: PoseSample, timestamp_ns: int) -> PoseSample:
    fraction = (timestamp_ns - first.timestamp_ns) / (
        second.timestamp_ns - first.timestamp_ns
    )
    first_position = np.asarray(first.position_m, dtype=np.float64)
    second_position = np.asarray(second.position_m, dtype=np.float64)
    position = (1.0 - fraction) * first_position + fraction * second_position
    label = first.semantic_label if first.semantic_label == second.semantic_label else None
    expected_contact = (
        first.expected_contact
        if first.expected_contact == second.expected_contact
        else None
    )
    return PoseSample(
        timestamp_ns=timestamp_ns,
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        orientation_xyzw=_nlerp_quaternion(
            first.orientation_xyzw, second.orientation_xyzw, fraction
        ),
        frame_id=first.frame_id,
        source="interpolated",
        confidence=min(first.confidence, second.confidence),
        gripper_width_m=_interpolate_optional_float(
            first.gripper_width_m, second.gripper_width_m, fraction
        ),
        surface_distance_m=_interpolate_optional_float(
            first.surface_distance_m, second.surface_distance_m, fraction
        ),
        expected_contact=expected_contact,
        semantic_label=label,
    )


def _interpolate_missing_samples(
    samples: Sequence[PoseSample], nominal_period_ns: int | None, maximum_gap_s: float
) -> tuple[list[PoseSample], int]:
    if len(samples) < 2 or nominal_period_ns is None:
        return list(samples), 0
    maximum_gap_ns = int(maximum_gap_s * 1.0e9)
    output: list[PoseSample] = []
    inserted = 0
    for first, second in zip(samples[:-1], samples[1:], strict=True):
        output.append(first)
        gap_ns = second.timestamp_ns - first.timestamp_ns
        if gap_ns <= int(1.5 * nominal_period_ns) or gap_ns > maximum_gap_ns:
            continue
        interval_count = max(1, int(round(gap_ns / nominal_period_ns)))
        for interval_index in range(1, interval_count):
            timestamp_ns = first.timestamp_ns + int(
                round(interval_index * gap_ns / interval_count)
            )
            output.append(_interpolate_pair(first, second, timestamp_ns))
            inserted += 1
    output.append(samples[-1])
    return output, inserted


def interpolate_missing_samples(
    samples: Sequence[PoseSample],
    *,
    nominal_period_ns: int | None = None,
    maximum_gap_s: float = 0.35,
) -> list[PoseSample]:
    """Fill only short timestamp gaps; long unobserved spans stay explicit."""

    sorted_samples = sort_samples(samples)
    period_ns = _nominal_period_ns(sorted_samples, nominal_period_ns)
    interpolated, _ = _interpolate_missing_samples(sorted_samples, period_ns, maximum_gap_s)
    return interpolated


def _contiguous_low_confidence_intervals(
    samples: Sequence[PoseSample], threshold: float, nominal_period_ns: int | None
) -> tuple[TimeInterval, ...]:
    low = [sample for sample in samples if sample.confidence < threshold]
    if not low:
        return ()
    maximum_contiguous_gap = (
        int(1.75 * nominal_period_ns) if nominal_period_ns is not None else 0
    )
    groups: list[list[PoseSample]] = [[low[0]]]
    for sample in low[1:]:
        previous = groups[-1][-1]
        within_contiguous_gap = (
            sample.timestamp_ns - previous.timestamp_ns <= maximum_contiguous_gap
        )
        if maximum_contiguous_gap and within_contiguous_gap:
            groups[-1].append(sample)
        else:
            groups.append([sample])
    return tuple(
        TimeInterval(
            start_timestamp_ns=group[0].timestamp_ns,
            end_timestamp_ns=group[-1].timestamp_ns,
            reason="pose confidence below preprocessing threshold",
            minimum_confidence=min(sample.confidence for sample in group),
        )
        for group in groups
    )


def drop_low_confidence_samples(
    samples: Sequence[PoseSample], threshold: float = 0.55
) -> list[PoseSample]:
    """Remove observations below ``threshold`` while preserving time gaps."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return [sample for sample in samples if sample.confidence >= threshold]


def _outlier_mask(
    samples: Sequence[PoseSample], absolute_threshold_m: float, mad_scale: float
) -> NDArray[np.bool_]:
    count = len(samples)
    mask = np.zeros(count, dtype=np.bool_)
    if count < 5:
        return mask
    positions = np.asarray([sample.position_m for sample in samples], dtype=np.float64)
    residual = np.zeros(count, dtype=np.float64)
    for index in range(1, count - 1):
        first_time = samples[index - 1].timestamp_ns
        last_time = samples[index + 1].timestamp_ns
        if last_time == first_time:
            continue
        fraction = (samples[index].timestamp_ns - first_time) / (last_time - first_time)
        prediction = positions[index - 1] + fraction * (
            positions[index + 1] - positions[index - 1]
        )
        residual[index] = float(np.linalg.norm(positions[index] - prediction))
    interior = residual[1:-1]
    median = float(np.median(interior))
    mad = float(np.median(np.abs(interior - median)))
    robust_sigma = max(1.4826 * mad, 1.0e-9)
    threshold = max(absolute_threshold_m, median + mad_scale * robust_sigma)
    for index in range(1, count - 1):
        # Requiring a local residual maximum prevents the two neighbours of a
        # single spike from being discarded along with the spike itself.
        local_maximum = (
            residual[index] >= residual[index - 1]
            and residual[index] >= residual[index + 1]
        )
        neighbour_span = float(np.linalg.norm(positions[index + 1] - positions[index - 1]))
        isolated = neighbour_span < max(2.0 * threshold, residual[index])
        mask[index] = bool(residual[index] > threshold and local_maximum and isolated)
    return mask


def remove_outliers(
    samples: Sequence[PoseSample],
    *,
    absolute_threshold_m: float = 0.025,
    mad_scale: float = 6.0,
) -> list[PoseSample]:
    """Remove isolated spatial spikes using a robust local prediction residual."""

    mask = _outlier_mask(samples, absolute_threshold_m, mad_scale)
    return [sample for index, sample in enumerate(samples) if not mask[index]]


def _local_polynomial_smooth(
    values: FloatArray,
    timestamps_ns: NDArray[np.int64],
    window_samples: int,
    polynomial_order: int,
) -> FloatArray:
    count = len(values)
    if count < 3 or window_samples <= 1:
        return values.copy()
    window = min(count, window_samples)
    if window % 2 == 0 and window > 1:
        window -= 1
    order = min(polynomial_order, window - 1)
    if window <= order or window < 3:
        return values.copy()
    one_dimensional = values.ndim == 1
    matrix = values[:, None] if one_dimensional else values
    result = np.empty_like(matrix)
    half_window = window // 2
    for index in range(count):
        lower = max(0, min(index - half_window, count - window))
        upper = lower + window
        relative_s = (timestamps_ns[lower:upper] - timestamps_ns[index]).astype(
            np.float64
        ) / 1.0e9
        design = np.vander(relative_s, N=order + 1, increasing=True)
        coefficients, _, _, _ = np.linalg.lstsq(design, matrix[lower:upper], rcond=None)
        result[index] = coefficients[0]
    return result[:, 0] if one_dimensional else result


def smooth_samples(
    samples: Sequence[PoseSample],
    *,
    window_samples: int = 7,
    polynomial_order: int = 2,
) -> list[PoseSample]:
    """Smooth position and quaternion component series independently."""

    if not samples:
        return []
    timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    positions = np.asarray([sample.position_m for sample in samples], dtype=np.float64)
    quaternions = np.asarray(
        [sample.orientation_xyzw for sample in samples], dtype=np.float64
    )
    # q and -q encode the same rotation.  Keep neighbouring samples in one
    # hemisphere before component-wise smoothing.
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    smoothed_positions = _local_polynomial_smooth(
        positions, timestamps, window_samples, polynomial_order
    )
    smoothed_quaternions = _local_polynomial_smooth(
        quaternions, timestamps, window_samples, polynomial_order
    )
    norms = np.linalg.norm(smoothed_quaternions, axis=1)
    invalid = norms < 1.0e-12
    smoothed_quaternions[invalid] = quaternions[invalid]
    norms = np.linalg.norm(smoothed_quaternions, axis=1)
    smoothed_quaternions /= norms[:, None]

    output: list[PoseSample] = []
    for index, sample in enumerate(samples):
        output.append(
            sample.model_copy(
                update={
                    "position_m": tuple(
                        float(component) for component in smoothed_positions[index]
                    ),
                    "orientation_xyzw": tuple(
                        float(component) for component in smoothed_quaternions[index]
                    ),
                    "source": f"{sample.source}:smoothed",
                }
            )
        )
    return output


def _differentiate(values: FloatArray, timestamps_s: FloatArray) -> FloatArray:
    if len(values) <= 1:
        return np.zeros_like(values)
    if len(values) >= 3:
        differentiated = np.gradient(values, timestamps_s, axis=0, edge_order=2)
    else:
        differentiated = np.gradient(values, timestamps_s, axis=0, edge_order=1)
    return np.asarray(differentiated, dtype=np.float64)


def _vectors(values: FloatArray) -> tuple[Vector3, ...]:
    return tuple(
        (float(row[0]), float(row[1]), float(row[2]))
        for row in values
    )


def _calculate_kinematics(
    samples: Sequence[PoseSample], contact_distance_threshold_m: float
) -> tuple[
    tuple[Vector3, ...],
    tuple[Vector3, ...],
    tuple[Vector3, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[bool, ...],
]:
    positions = np.asarray([sample.position_m for sample in samples], dtype=np.float64)
    timestamps_s = np.asarray(
        [sample.timestamp_ns for sample in samples], dtype=np.float64
    ) / 1.0e9
    velocity = _differentiate(positions, timestamps_s)
    acceleration = _differentiate(velocity, timestamps_s)
    jerk = _differentiate(acceleration, timestamps_s)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    cross_product = np.cross(velocity, acceleration)
    denominator = np.power(speed, 3)
    curvature = np.divide(
        np.linalg.norm(cross_product, axis=1),
        denominator,
        out=np.zeros_like(speed),
        where=denominator > 1.0e-12,
    )
    contact_candidate = tuple(
        bool(
            sample.expected_contact
            or (
                sample.surface_distance_m is not None
                and sample.surface_distance_m <= contact_distance_threshold_m
            )
        )
        for sample in samples
    )
    return (
        _vectors(velocity),
        _vectors(acceleration),
        _vectors(jerk),
        tuple(float(value) for value in speed),
        tuple(float(value) for value in acceleration_magnitude),
        tuple(float(value) for value in jerk_magnitude),
        tuple(float(value) for value in curvature),
        contact_candidate,
    )


def preprocess_trajectory(
    trajectory: DemonstrationTrajectory | Sequence[PoseSample],
    config: PreprocessingConfig | None = None,
) -> ProcessedTrajectory:
    """Clean and derive a pose series without any hardware or network access."""

    effective_config = config or PreprocessingConfig()
    raw_samples = _coerce_samples(trajectory)
    if not raw_samples:
        raise TrajectoryPreprocessingError("trajectory contains no samples")
    sorted_samples, duplicates = _sort_and_deduplicate(raw_samples)
    nominal_period_ns = _nominal_period_ns(
        sorted_samples, effective_config.nominal_period_ns
    )
    interpolated, interpolated_count = _interpolate_missing_samples(
        sorted_samples,
        nominal_period_ns,
        effective_config.maximum_interpolation_gap_s,
    )
    low_confidence_intervals = _contiguous_low_confidence_intervals(
        interpolated, effective_config.confidence_threshold, nominal_period_ns
    )
    confidence_filtered = drop_low_confidence_samples(
        interpolated, effective_config.confidence_threshold
    )
    low_confidence_count = len(interpolated) - len(confidence_filtered)
    outlier_mask = _outlier_mask(
        confidence_filtered,
        effective_config.outlier_absolute_threshold_m,
        effective_config.outlier_mad_scale,
    )
    outlier_timestamps = tuple(
        sample.timestamp_ns
        for index, sample in enumerate(confidence_filtered)
        if outlier_mask[index]
    )
    filtered = [
        sample
        for index, sample in enumerate(confidence_filtered)
        if not outlier_mask[index]
    ]
    if not filtered:
        raise TrajectoryPreprocessingError(
            "trajectory has no samples after confidence and outlier filtering"
        )
    smoothed = smooth_samples(
        filtered,
        window_samples=effective_config.smoothing_window_samples,
        polynomial_order=effective_config.smoothing_polynomial_order,
    )
    (
        velocity,
        acceleration,
        jerk,
        speed,
        acceleration_magnitude,
        jerk_magnitude,
        curvature,
        contact_candidate,
    ) = _calculate_kinematics(
        smoothed, effective_config.contact_distance_threshold_m
    )
    report = PreprocessingReport(
        input_sample_count=len(raw_samples),
        output_sample_count=len(smoothed),
        duplicate_timestamp_count=duplicates,
        interpolated_sample_count=interpolated_count,
        dropped_low_confidence_count=low_confidence_count,
        dropped_outlier_count=len(outlier_timestamps),
        low_confidence_intervals=low_confidence_intervals,
        outlier_timestamps_ns=outlier_timestamps,
        nominal_period_ns=nominal_period_ns,
    )
    return ProcessedTrajectory(
        samples=tuple(smoothed),
        velocity_mps=velocity,
        acceleration_mps2=acceleration,
        jerk_mps3=jerk,
        speed_mps=speed,
        acceleration_magnitude_mps2=acceleration_magnitude,
        jerk_magnitude_mps3=jerk_magnitude,
        curvature_per_m=curvature,
        contact_candidate=contact_candidate,
        report=report,
    )


# A concise alias useful for callers that already deal specifically in poses.
preprocess_pose_samples = preprocess_trajectory
