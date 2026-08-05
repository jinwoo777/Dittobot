"""Deterministic geometry fitting and Local Motion Skill recommendations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import (
    FitResiduals,
    PoseSample,
    PrimitiveFit,
    PrimitiveRecommendation,
    ProcessedTrajectory,
    SegmentState,
    Vector3,
)
from .trajectory import path_length_m

FloatArray = NDArray[np.float64]


class PrimitiveFittingError(ValueError):
    """Raised when a requested fit has insufficient or degenerate geometry."""


@dataclass(frozen=True, slots=True)
class PeriodicPrimitiveGeometry:
    """Locally verified arguments for the fixed-centre periodic primitive schema."""

    center_m: Vector3
    amplitude_vector_m: Vector3
    repetitions: int
    observed_period_s: float
    observed_cycle_count: float
    residuals: FitResiduals


@dataclass(frozen=True)
class PrimitiveFitterConfig:
    """Geometry classification thresholds, independent of robot profiles."""

    line_maximum_rms_residual_m: float = 0.004
    arc_maximum_rms_residual_m: float = 0.004
    arc_minimum_sweep_rad: float = 0.35
    arc_maximum_radius_m: float = 10.0
    periodic_minimum_cycles: float = 1.5
    periodic_minimum_spectral_fraction: float = 0.35
    periodic_minimum_amplitude_m: float = 0.004
    periodic_maximum_relative_rms: float = 0.50
    minimum_segment_length_m: float = 0.002

    def __post_init__(self) -> None:
        positive = (
            self.line_maximum_rms_residual_m,
            self.arc_maximum_rms_residual_m,
            self.arc_minimum_sweep_rad,
            self.arc_maximum_radius_m,
            self.periodic_minimum_cycles,
            self.periodic_minimum_amplitude_m,
            self.periodic_maximum_relative_rms,
            self.minimum_segment_length_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("primitive fitting thresholds must be positive")
        if not 0.0 <= self.periodic_minimum_spectral_fraction <= 1.0:
            raise ValueError("periodic_minimum_spectral_fraction must be in [0, 1]")


def _samples(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
) -> tuple[PoseSample, ...]:
    if isinstance(trajectory, ProcessedTrajectory):
        return trajectory.samples
    return tuple(trajectory)


def _positions(trajectory: ProcessedTrajectory | Sequence[PoseSample]) -> FloatArray:
    samples = _samples(trajectory)
    if len(samples) < 2:
        raise PrimitiveFittingError("at least two pose samples are required")
    return np.asarray([sample.position_m for sample in samples], dtype=np.float64)


def _vector3(value: NDArray[np.float64]) -> Vector3:
    return float(value[0]), float(value[1]), float(value[2])


def _residuals(distances_m: FloatArray) -> FitResiduals:
    distances = np.abs(distances_m)
    return FitResiduals(
        root_mean_square_m=float(np.sqrt(np.mean(np.square(distances)))),
        mean_m=float(np.mean(distances)),
        maximum_m=float(np.max(distances)),
    )


def _mean_discrete_curvature(positions: FloatArray) -> float:
    if len(positions) < 3:
        return 0.0
    values: list[float] = []
    for first, middle, last in zip(
        positions[:-2], positions[1:-1], positions[2:], strict=True
    ):
        first_edge = middle - first
        second_edge = last - middle
        first_length = float(np.linalg.norm(first_edge))
        second_length = float(np.linalg.norm(second_edge))
        if first_length < 1.0e-9 or second_length < 1.0e-9:
            continue
        cosine = float(
            np.clip(
                np.dot(first_edge, second_edge) / (first_length * second_length),
                -1.0,
                1.0,
            )
        )
        angle = float(np.arccos(cosine))
        values.append(angle / max((first_length + second_length) / 2.0, 1.0e-9))
    return float(np.mean(values)) if values else 0.0


def fit_line(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
    config: PrimitiveFitterConfig | None = None,
) -> PrimitiveFit:
    """Fit an orthogonal least-squares 3-D line using PCA."""

    effective_config = config or PrimitiveFitterConfig()
    samples = _samples(trajectory)
    positions = _positions(trajectory)
    centroid = np.mean(positions, axis=0)
    _, _, right_vectors = np.linalg.svd(positions - centroid, full_matrices=False)
    direction = right_vectors[0]
    if float(np.dot(direction, positions[-1] - positions[0])) < 0.0:
        direction *= -1.0
    parameters = (positions - centroid) @ direction
    projected = centroid + parameters[:, None] * direction
    distances = np.linalg.norm(positions - projected, axis=1)
    residual = _residuals(distances)
    length = path_length_m(samples)
    relative_error = residual.root_mean_square_m / max(
        effective_config.line_maximum_rms_residual_m, 1.0e-12
    )
    confidence = 1.0 / (1.0 + relative_error * relative_error)
    middle_index = len(samples) // 2
    return PrimitiveFit(
        primitive_id="motion.move_l",
        fit_type="line",
        residuals=residual,
        confidence=confidence,
        score=confidence,
        segment_length_m=length,
        mean_curvature_per_m=_mean_discrete_curvature(projected),
        start_m=_vector3(projected[0]),
        via_m=_vector3(projected[middle_index]),
        end_m=_vector3(projected[-1]),
        rationale="orthogonal PCA line fit",
    )


@dataclass(frozen=True)
class _ArcGeometry:
    center_2d: FloatArray
    radius_m: float
    plane_origin: FloatArray
    basis_u: FloatArray
    basis_v: FloatArray
    projected_3d: FloatArray
    distances_m: FloatArray
    sweep_rad: float


def _arc_geometry(positions: FloatArray) -> _ArcGeometry:
    if len(positions) < 3:
        raise PrimitiveFittingError("at least three samples are required for an arc fit")
    origin = np.mean(positions, axis=0)
    centered = positions - origin
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] < 1.0e-10:
        raise PrimitiveFittingError("arc fit is degenerate for collinear samples")
    basis_u = right_vectors[0]
    basis_v = right_vectors[1]
    plane_normal = right_vectors[2]
    x = centered @ basis_u
    y = centered @ basis_v
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(x))))
    target = np.square(x) + np.square(y)
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise PrimitiveFittingError("circle fit is rank deficient")
    center_2d = solution[:2]
    radius_squared = float(solution[2] + np.dot(center_2d, center_2d))
    if radius_squared <= 1.0e-12:
        raise PrimitiveFittingError("circle fit produced a non-positive radius")
    radius = float(np.sqrt(radius_squared))
    radial_distance = np.sqrt(np.square(x - center_2d[0]) + np.square(y - center_2d[1]))
    plane_distance = centered @ plane_normal
    distances = np.sqrt(np.square(radial_distance - radius) + np.square(plane_distance))
    angles = np.unwrap(np.arctan2(y - center_2d[1], x - center_2d[0]))
    sweep = float(abs(angles[-1] - angles[0]))
    fitted_x = center_2d[0] + radius * np.cos(angles)
    fitted_y = center_2d[1] + radius * np.sin(angles)
    projected = origin + fitted_x[:, None] * basis_u + fitted_y[:, None] * basis_v
    return _ArcGeometry(
        center_2d=center_2d,
        radius_m=radius,
        plane_origin=origin,
        basis_u=basis_u,
        basis_v=basis_v,
        projected_3d=projected,
        distances_m=distances,
        sweep_rad=sweep,
    )


def fit_arc(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
    config: PrimitiveFitterConfig | None = None,
) -> PrimitiveFit:
    """Fit a best plane and algebraic circle, reporting full 3-D residuals."""

    effective_config = config or PrimitiveFitterConfig()
    samples = _samples(trajectory)
    positions = _positions(trajectory)
    geometry = _arc_geometry(positions)
    residual = _residuals(geometry.distances_m)
    residual_factor = 1.0 / (
        1.0
        + (
            residual.root_mean_square_m
            / max(effective_config.arc_maximum_rms_residual_m, 1.0e-12)
        )
        ** 2
    )
    sweep_factor = min(1.0, geometry.sweep_rad / effective_config.arc_minimum_sweep_rad)
    radius_factor = min(1.0, effective_config.arc_maximum_radius_m / geometry.radius_m)
    confidence = residual_factor * sweep_factor * radius_factor
    middle_index = len(samples) // 2
    return PrimitiveFit(
        primitive_id="motion.move_c",
        fit_type="arc",
        residuals=residual,
        confidence=confidence,
        score=confidence,
        segment_length_m=path_length_m(samples),
        mean_curvature_per_m=1.0 / geometry.radius_m,
        radius_m=geometry.radius_m,
        start_m=_vector3(geometry.projected_3d[0]),
        via_m=_vector3(geometry.projected_3d[middle_index]),
        end_m=_vector3(geometry.projected_3d[-1]),
        rationale=f"best-plane circle fit with {geometry.sweep_rad:.3f} rad sweep",
    )


@dataclass(frozen=True)
class _PeriodicGeometry:
    frequency_hz: float
    cycle_count: float
    spectral_fraction: float
    amplitude_m: float
    fitted: FloatArray
    residual_m: FloatArray


def _periodic_geometry(samples: Sequence[PoseSample], positions: FloatArray) -> _PeriodicGeometry:
    if len(samples) < 8:
        raise PrimitiveFittingError("at least eight samples are required for a periodic fit")
    timestamps_s = np.asarray(
        [sample.timestamp_ns for sample in samples], dtype=np.float64
    ) / 1.0e9
    relative_time = timestamps_s - timestamps_s[0]
    duration_s = float(relative_time[-1])
    if duration_s <= 0.0:
        raise PrimitiveFittingError("periodic fit requires increasing timestamps")
    # Use linear detrending so a periodic wiping stroke may translate slowly
    # along a surface without losing its repeated component.
    trend_design = np.column_stack((np.ones(len(relative_time)), relative_time))
    trend_coefficients, _, _, _ = np.linalg.lstsq(trend_design, positions, rcond=None)
    detrended = positions - trend_design @ trend_coefficients
    timestep_s = float(np.median(np.diff(timestamps_s)))
    if timestep_s <= 0.0:
        raise PrimitiveFittingError("periodic fit requires unique timestamps")
    spectrum = np.fft.rfft(detrended, axis=0)
    frequencies = np.fft.rfftfreq(len(detrended), d=timestep_s)
    if len(frequencies) <= 1:
        raise PrimitiveFittingError("periodic spectrum has no non-zero bins")
    power_by_bin = np.sum(np.square(np.abs(spectrum)), axis=1)
    power_by_bin[0] = 0.0
    dominant_index = int(np.argmax(power_by_bin))
    dominant_power = float(power_by_bin[dominant_index])
    total_power = float(np.sum(power_by_bin))
    spectral_fraction = dominant_power / max(total_power, 1.0e-18)
    seeded_frequency_hz = float(frequencies[dominant_index])
    if seeded_frequency_hz <= 0.0:
        raise PrimitiveFittingError("no non-zero periodic frequency found")
    # FFT bins are coarse for non-integral cycle counts. Refine the fixed seed
    # over one neighbouring-bin width by minimizing the 3-D harmonic residual.
    bin_width_hz = 1.0 / max(len(detrended) * timestep_s, 1.0e-12)
    lower_hz = max(0.5 / duration_s, seeded_frequency_hz - bin_width_hz)
    upper_hz = seeded_frequency_hz + bin_width_hz
    best_error = float("inf")
    best_frequency_hz = seeded_frequency_hz
    best_design: FloatArray | None = None
    best_coefficients: FloatArray | None = None
    for candidate_hz in np.linspace(lower_hz, upper_hz, 81):
        angular = 2.0 * np.pi * candidate_hz * relative_time
        design = np.column_stack(
            (
                np.ones(len(relative_time)),
                relative_time,
                np.sin(angular),
                np.cos(angular),
            )
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        error = float(np.sum(np.square(positions - design @ coefficients)))
        if error < best_error:
            best_error = error
            best_frequency_hz = float(candidate_hz)
            best_design = design
            best_coefficients = coefficients
    if best_design is None or best_coefficients is None:
        raise PrimitiveFittingError("periodic frequency refinement failed")
    frequency_hz = best_frequency_hz
    harmonic_design = best_design
    coefficients = best_coefficients
    fitted = harmonic_design @ coefficients
    residual = np.linalg.norm(positions - fitted, axis=1)
    sine_vector = coefficients[2]
    cosine_vector = coefficients[3]
    phases = np.linspace(0.0, 2.0 * np.pi, 181)
    harmonic_points = (
        np.sin(phases)[:, None] * sine_vector
        + np.cos(phases)[:, None] * cosine_vector
    )
    amplitude = float(np.max(np.linalg.norm(harmonic_points, axis=1)))
    return _PeriodicGeometry(
        frequency_hz=frequency_hz,
        cycle_count=frequency_hz * duration_s,
        spectral_fraction=spectral_fraction,
        amplitude_m=amplitude,
        fitted=fitted,
        residual_m=residual,
    )


def fit_periodic(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
    config: PrimitiveFitterConfig | None = None,
) -> PrimitiveFit:
    """Fit a translating first-harmonic path using timestamps and FFT seeding."""

    effective_config = config or PrimitiveFitterConfig()
    samples = _samples(trajectory)
    positions = _positions(trajectory)
    geometry = _periodic_geometry(samples, positions)
    residual = _residuals(geometry.residual_m)
    relative_rms = residual.root_mean_square_m / max(geometry.amplitude_m, 1.0e-12)
    cycle_factor = min(1.0, geometry.cycle_count / effective_config.periodic_minimum_cycles)
    spectral_factor = min(
        1.0,
        geometry.spectral_fraction
        / max(effective_config.periodic_minimum_spectral_fraction, 1.0e-12),
    )
    residual_factor = 1.0 / (
        1.0 + (relative_rms / effective_config.periodic_maximum_relative_rms) ** 2
    )
    amplitude_factor = min(
        1.0,
        geometry.amplitude_m / effective_config.periodic_minimum_amplitude_m,
    )
    confidence = cycle_factor * spectral_factor * residual_factor * amplitude_factor
    middle_index = len(samples) // 2
    return PrimitiveFit(
        primitive_id="motion.move_periodic",
        fit_type="periodic",
        residuals=residual,
        confidence=confidence,
        score=confidence,
        segment_length_m=path_length_m(samples),
        mean_curvature_per_m=_mean_discrete_curvature(geometry.fitted),
        start_m=_vector3(geometry.fitted[0]),
        via_m=_vector3(geometry.fitted[middle_index]),
        end_m=_vector3(geometry.fitted[-1]),
        period_s=1.0 / geometry.frequency_hz,
        amplitude_m=geometry.amplitude_m,
        cycle_count=geometry.cycle_count,
        rationale=(
            f"timestamped harmonic fit; spectral fraction "
            f"{geometry.spectral_fraction:.3f}"
        ),
    )


def fit_periodic_primitive_geometry(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
    fit: PrimitiveFit,
    *,
    maximum_error_m: float = 0.004,
    maximum_cycle_rounding_error: float = 0.10,
) -> PeriodicPrimitiveGeometry:
    """Fit the stricter geometry representable by ``motion.move_periodic``.

    The general periodic detector permits a translating trend and multi-axis
    harmonics.  The runtime primitive has only a fixed centre, one amplitude
    vector, and an integer repetition count.  This second local fit therefore
    rejects demonstrations that the runtime schema cannot reproduce within the
    requested positional error.  No model-supplied value participates.
    """

    if not np.isfinite(maximum_error_m) or maximum_error_m <= 0.0:
        raise ValueError("maximum_error_m must be finite and positive")
    if not np.isfinite(maximum_cycle_rounding_error) or not (
        0.0 <= maximum_cycle_rounding_error <= 0.25
    ):
        raise ValueError("maximum_cycle_rounding_error must be in [0, 0.25]")
    if fit.primitive_id != "motion.move_periodic":
        raise PrimitiveFittingError("selected local fit is not periodic")
    if (
        fit.period_s is None
        or fit.amplitude_m is None
        or fit.cycle_count is None
        or fit.confidence < 0.45
    ):
        raise PrimitiveFittingError("periodic fit lacks verified local parameters")

    samples = _samples(trajectory)
    positions = _positions(trajectory)
    if len(samples) < 8:
        raise PrimitiveFittingError("periodic primitive requires at least eight samples")
    timestamps_s = np.asarray(
        [sample.timestamp_ns for sample in samples], dtype=np.float64
    ) / 1.0e9
    if np.any(np.diff(timestamps_s) <= 0.0):
        raise PrimitiveFittingError("periodic primitive requires increasing timestamps")
    relative_time_s = timestamps_s - timestamps_s[0]
    angular_frequency = 2.0 * np.pi / fit.period_s
    phase = angular_frequency * relative_time_s
    design = np.column_stack(
        (np.ones(len(relative_time_s)), np.sin(phase), np.cos(phase))
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
    if rank < 3:
        raise PrimitiveFittingError("fixed-centre periodic fit is rank deficient")

    center = coefficients[0]
    harmonic_coefficients = coefficients[1:]
    _, singular_values, right_vectors = np.linalg.svd(
        harmonic_coefficients, full_matrices=False
    )
    if not len(singular_values) or singular_values[0] <= 1.0e-12:
        raise PrimitiveFittingError("fixed-centre periodic amplitude is degenerate")
    direction = right_vectors[0]
    dominant_component_index = int(np.argmax(np.abs(direction)))
    if direction[dominant_component_index] < 0.0:
        direction = -direction
    scalar_coefficients = harmonic_coefficients @ direction
    amplitude_m = float(np.linalg.norm(scalar_coefficients))
    if amplitude_m <= 1.0e-9 or float(np.max(np.abs(direction * amplitude_m))) > 1.0:
        raise PrimitiveFittingError("fixed-centre periodic amplitude is outside schema bounds")

    scalar_harmonic = design[:, 1:] @ scalar_coefficients
    reconstructed = center + scalar_harmonic[:, None] * direction
    residual = _residuals(np.linalg.norm(positions - reconstructed, axis=1))
    numerical_tolerance_m = 1.0e-9
    if residual.maximum_m > maximum_error_m + numerical_tolerance_m:
        raise PrimitiveFittingError(
            "fixed-centre periodic representation exceeds the positional error bound"
        )
    if abs(amplitude_m - fit.amplitude_m) > maximum_error_m + numerical_tolerance_m:
        raise PrimitiveFittingError(
            "fixed-centre amplitude disagrees with the selected periodic fit"
        )

    repetitions = int(round(fit.cycle_count))
    if not 1 <= repetitions <= 100 or (
        abs(fit.cycle_count - repetitions) > maximum_cycle_rounding_error
    ):
        raise PrimitiveFittingError(
            "observed periodic cycle count is not safely representable as repetitions"
        )
    amplitude_vector = direction * amplitude_m
    return PeriodicPrimitiveGeometry(
        center_m=_vector3(center),
        amplitude_vector_m=_vector3(amplitude_vector),
        repetitions=repetitions,
        observed_period_s=fit.period_s,
        observed_cycle_count=fit.cycle_count,
        residuals=residual,
    )


def _fixed_state_fit(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
    primitive_id: str,
    rationale: str,
) -> PrimitiveFit:
    samples = _samples(trajectory)
    positions = _positions(trajectory)
    middle_index = len(samples) // 2
    return PrimitiveFit(
        primitive_id=primitive_id,
        fit_type="state_mapping",
        residuals=FitResiduals(root_mean_square_m=0.0, mean_m=0.0, maximum_m=0.0),
        confidence=sum(sample.confidence for sample in samples) / len(samples),
        score=sum(sample.confidence for sample in samples) / len(samples),
        segment_length_m=path_length_m(samples),
        mean_curvature_per_m=_mean_discrete_curvature(positions),
        start_m=samples[0].position_m,
        via_m=samples[middle_index].position_m,
        end_m=samples[-1].position_m,
        rationale=rationale,
    )


def recommend_primitive(
    trajectory: ProcessedTrajectory | Sequence[PoseSample],
    *,
    state: SegmentState | None = None,
    config: PrimitiveFitterConfig | None = None,
) -> PrimitiveRecommendation:
    """Return deterministic candidates and the selected Local Skill ID."""

    effective_config = config or PrimitiveFitterConfig()
    fixed_mapping = {
        SegmentState.GRIPPER_OPEN: ("gripper.open", "observed increasing gripper width"),
        SegmentState.GRIPPER_CLOSE: ("gripper.close", "observed decreasing gripper width"),
        SegmentState.CONTACT_SEARCH: (
            "contact.search_surface",
            "locally observed transition into surface contact",
        ),
        SegmentState.CONTACT_RELEASE: (
            "contact.disable_force",
            "locally observed release from surface contact",
        ),
        SegmentState.RETRACT: (
            "recovery.safe_retract",
            "locally observed motion away from the contact surface",
        ),
    }
    if state in fixed_mapping:
        primitive_id, rationale = fixed_mapping[state]
        fixed_fit = _fixed_state_fit(trajectory, primitive_id, rationale)
        return PrimitiveRecommendation(
            recommended_primitive_id=primitive_id,
            selected_fit=fixed_fit,
            candidates=(fixed_fit,),
        )

    line = fit_line(trajectory, effective_config)
    candidates: list[PrimitiveFit] = [line]
    arc: PrimitiveFit | None = None
    periodic: PrimitiveFit | None = None
    try:
        arc = fit_arc(trajectory, effective_config)
        candidates.append(arc)
    except PrimitiveFittingError:
        pass
    try:
        periodic = fit_periodic(trajectory, effective_config)
        candidates.append(periodic)
    except PrimitiveFittingError:
        pass

    selected: PrimitiveFit
    if (
        periodic is not None
        and periodic.cycle_count is not None
        and periodic.amplitude_m is not None
        and periodic.cycle_count >= effective_config.periodic_minimum_cycles
        and periodic.amplitude_m >= effective_config.periodic_minimum_amplitude_m
        and periodic.residuals.root_mean_square_m
        <= effective_config.periodic_maximum_relative_rms * periodic.amplitude_m
        and "spectral fraction" in periodic.rationale
        and periodic.confidence >= 0.45
    ):
        selected = periodic
    elif (
        arc is not None
        and arc.radius_m is not None
        and arc.radius_m <= effective_config.arc_maximum_radius_m
        and arc.residuals.root_mean_square_m
        <= effective_config.arc_maximum_rms_residual_m
        and arc.mean_curvature_per_m * arc.segment_length_m
        >= effective_config.arc_minimum_sweep_rad
        and arc.residuals.root_mean_square_m
        < max(line.residuals.root_mean_square_m * 0.85, 1.0e-7)
    ):
        selected = arc
    elif (
        line.residuals.root_mean_square_m
        <= effective_config.line_maximum_rms_residual_m
        and line.segment_length_m >= effective_config.minimum_segment_length_m
    ):
        selected = line
    else:
        selected = line.model_copy(
            update={
                "primitive_id": "motion.move_spline",
                "fit_type": "spline_fallback",
                "confidence": max(0.25, line.confidence),
                "score": max(0.25, line.confidence),
                "rationale": "no line, arc, or periodic model met its local thresholds",
            }
        )
        candidates.append(selected)
    ordered = tuple(sorted(candidates, key=lambda fit: fit.score, reverse=True))
    return PrimitiveRecommendation(
        recommended_primitive_id=selected.primitive_id,
        selected_fit=selected,
        candidates=ordered,
    )


# Naming aliases used by orchestration and older fixtures.
fit_primitive = recommend_primitive
fit_trajectory = recommend_primitive

__all__ = [
    "PeriodicPrimitiveGeometry",
    "PrimitiveFitterConfig",
    "PrimitiveFittingError",
    "fit_arc",
    "fit_line",
    "fit_periodic",
    "fit_periodic_primitive_geometry",
    "fit_primitive",
    "fit_trajectory",
    "recommend_primitive",
]
