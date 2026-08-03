"""Typed models used by the offline demonstration-learning pipeline.

The models in this module deliberately contain no robot commands.  A demonstration
is evidence from which a local, deterministic pipeline derives motion candidates;
it is never executable robot code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import Enum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def _finite_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    """Convert a sequence to a finite, fixed-size tuple."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric sequence")
    if not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a numeric sequence")
    try:
        converted = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if len(converted) != length:
        raise ValueError(f"{name} must contain {length} values")
    if not all(math.isfinite(component) for component in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


class SegmentState(str, Enum):
    """Locally observable phases in a teaching demonstration."""

    IDLE = "IDLE"
    APPROACH = "APPROACH"
    GRIPPER_OPEN = "GRIPPER_OPEN"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    FREE_SPACE_MOVE = "FREE_SPACE_MOVE"
    CONTACT_SEARCH = "CONTACT_SEARCH"
    CONTACT_MOVE = "CONTACT_MOVE"
    CONTACT_RELEASE = "CONTACT_RELEASE"
    RETRACT = "RETRACT"
    PERIODIC_MOVE = "PERIODIC_MOVE"
    WAIT = "WAIT"
    END = "END"
    UNKNOWN = "UNKNOWN"


class PoseSample(BaseModel):
    """One timestamped 6-D pose observation.

    Orientation is normalized at the ingestion boundary.  A zero quaternion is
    rejected because it cannot represent an orientation.  Samples may be low
    confidence: the quality and preprocessing stages need to see those values in
    order to recommend a retake rather than silently losing the evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_ns: int = Field(ge=0)
    position_m: Vector3
    orientation_xyzw: QuaternionXYZW = (0.0, 0.0, 0.0, 1.0)
    frame_id: str = Field(default="camera", min_length=1, max_length=128)
    source: str = Field(default="demonstration", min_length=1, max_length=128)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    gripper_width_m: float | None = Field(default=None, ge=0.0)
    surface_distance_m: float | None = None
    expected_contact: bool | None = None
    semantic_label: SegmentState | None = None

    @field_validator("position_m", mode="before")
    @classmethod
    def validate_position(cls, value: object) -> Vector3:
        converted = _finite_tuple(value, 3, "position_m")
        return converted[0], converted[1], converted[2]

    @field_validator("orientation_xyzw", mode="before")
    @classmethod
    def normalize_orientation(cls, value: object) -> QuaternionXYZW:
        converted = _finite_tuple(value, 4, "orientation_xyzw")
        norm = math.sqrt(sum(component * component for component in converted))
        if norm < 1.0e-12:
            raise ValueError("orientation_xyzw must be a non-zero quaternion")
        return cast(QuaternionXYZW, tuple(component / norm for component in converted))

    @field_validator("frame_id", "source")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or any(ord(character) < 32 for character in stripped):
            raise ValueError("identifier must be non-empty and contain no control characters")
        return stripped

    @field_validator("surface_distance_m")
    @classmethod
    def validate_optional_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("surface_distance_m must be finite")
        return value


class DemonstrationTrajectory(BaseModel):
    """Raw pose observations for one teaching session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: tuple[PoseSample, ...] = Field(min_length=1)
    session_id: str | None = None
    operator_id: str | None = None
    operator_role: str | None = None
    successful: bool | None = None
    notes: str | None = None


class TimeInterval(BaseModel):
    """Inclusive timestamp range carrying a quality explanation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_timestamp_ns: int = Field(ge=0)
    end_timestamp_ns: int = Field(ge=0)
    reason: str
    minimum_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> TimeInterval:
        if self.end_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("interval end must not precede its start")
        return self


class PreprocessingReport(BaseModel):
    """Audit information retained while samples are filtered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sample_count: int = Field(ge=0)
    output_sample_count: int = Field(ge=0)
    duplicate_timestamp_count: int = Field(default=0, ge=0)
    interpolated_sample_count: int = Field(default=0, ge=0)
    dropped_low_confidence_count: int = Field(default=0, ge=0)
    dropped_outlier_count: int = Field(default=0, ge=0)
    low_confidence_intervals: tuple[TimeInterval, ...] = ()
    outlier_timestamps_ns: tuple[int, ...] = ()
    nominal_period_ns: int | None = Field(default=None, gt=0)
    smoothing_method: str = "local_polynomial"


class ProcessedTrajectory(BaseModel):
    """Smoothed samples and position derivatives in SI units."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: tuple[PoseSample, ...] = Field(min_length=1)
    velocity_mps: tuple[Vector3, ...]
    acceleration_mps2: tuple[Vector3, ...]
    jerk_mps3: tuple[Vector3, ...]
    speed_mps: tuple[float, ...]
    acceleration_magnitude_mps2: tuple[float, ...]
    jerk_magnitude_mps3: tuple[float, ...]
    curvature_per_m: tuple[float, ...]
    contact_candidate: tuple[bool, ...]
    report: PreprocessingReport

    @model_validator(mode="after")
    def validate_lengths(self) -> ProcessedTrajectory:
        expected = len(self.samples)
        named_values = {
            "velocity_mps": self.velocity_mps,
            "acceleration_mps2": self.acceleration_mps2,
            "jerk_mps3": self.jerk_mps3,
            "speed_mps": self.speed_mps,
            "acceleration_magnitude_mps2": self.acceleration_magnitude_mps2,
            "jerk_magnitude_mps3": self.jerk_magnitude_mps3,
            "curvature_per_m": self.curvature_per_m,
            "contact_candidate": self.contact_candidate,
        }
        for name, values in named_values.items():
            if len(values) != expected:
                raise ValueError(f"{name} must have one value per sample")
        return self

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return (self.samples[-1].timestamp_ns - self.samples[0].timestamp_ns) / 1.0e9


class TrajectorySegment(BaseModel):
    """A contiguous locally classified demonstration phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: SegmentState
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    start_timestamp_ns: int = Field(ge=0)
    end_timestamp_ns: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    mean_speed_mps: float = Field(ge=0.0)
    mean_curvature_per_m: float = Field(ge=0.0)
    uncertain: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> TrajectorySegment:
        if self.end_index < self.start_index:
            raise ValueError("segment end_index must not precede start_index")
        if self.end_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("segment end timestamp must not precede start timestamp")
        return self


class SegmentationResult(BaseModel):
    """Segments plus an explicit retake decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: tuple[TrajectorySegment, ...]
    reteach_required: bool
    uncertain_intervals: tuple[TimeInterval, ...] = ()
    reason: str | None = None
    recommended_retake_range: tuple[int, int] | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class FitResiduals(BaseModel):
    """Distance errors for a deterministic geometry fit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_mean_square_m: float = Field(ge=0.0)
    mean_m: float = Field(ge=0.0)
    maximum_m: float = Field(ge=0.0)


class PrimitiveFit(BaseModel):
    """A scored Local Motion Skill recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primitive_id: str
    fit_type: str
    residuals: FitResiduals
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0)
    segment_length_m: float = Field(ge=0.0)
    mean_curvature_per_m: float = Field(ge=0.0)
    start_m: Vector3
    via_m: Vector3
    end_m: Vector3
    radius_m: float | None = Field(default=None, gt=0.0)
    period_s: float | None = Field(default=None, gt=0.0)
    amplitude_m: float | None = Field(default=None, ge=0.0)
    cycle_count: float | None = Field(default=None, ge=0.0)
    rationale: str


class PrimitiveRecommendation(BaseModel):
    """All viable geometry fits and the deterministic winner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommended_primitive_id: str
    selected_fit: PrimitiveFit
    candidates: tuple[PrimitiveFit, ...]


# Descriptive compatibility aliases for callers that use trajectory terminology.
TrajectorySample = PoseSample
SegmentationState = SegmentState
