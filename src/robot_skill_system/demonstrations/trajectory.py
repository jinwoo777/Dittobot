"""Trajectory-facing API.

This module keeps the conventional import path while the implementation lives in
``preprocessing``.  It also provides small, unit-explicit summary helpers used by
semantic analyzers without sending raw recordings to a model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .models import PoseSample, ProcessedTrajectory, Vector3
from .preprocessing import PreprocessingConfig, preprocess_trajectory


class TrajectorySummary(BaseModel):
    """Compact local summary safe to pass to a semantic orchestration layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=1)
    duration_s: float = Field(ge=0.0)
    path_length_m: float = Field(ge=0.0)
    start_m: Vector3
    end_m: Vector3
    mean_speed_mps: float = Field(ge=0.0)
    maximum_speed_mps: float = Field(ge=0.0)
    mean_curvature_per_m: float = Field(ge=0.0)
    minimum_confidence: float = Field(ge=0.0, le=1.0)


def path_length_m(samples: Sequence[PoseSample]) -> float:
    """Return Cartesian polyline length in metres."""

    return sum(
        math.dist(first.position_m, second.position_m)
        for first, second in zip(samples[:-1], samples[1:], strict=True)
    )


def summarize_trajectory(trajectory: ProcessedTrajectory) -> TrajectorySummary:
    """Build a compact numeric summary from locally derived features."""

    return TrajectorySummary(
        sample_count=len(trajectory.samples),
        duration_s=trajectory.duration_s,
        path_length_m=path_length_m(trajectory.samples),
        start_m=trajectory.samples[0].position_m,
        end_m=trajectory.samples[-1].position_m,
        mean_speed_mps=sum(trajectory.speed_mps) / len(trajectory.speed_mps),
        maximum_speed_mps=max(trajectory.speed_mps),
        mean_curvature_per_m=(
            sum(trajectory.curvature_per_m) / len(trajectory.curvature_per_m)
        ),
        minimum_confidence=min(sample.confidence for sample in trajectory.samples),
    )


__all__ = [
    "PreprocessingConfig",
    "TrajectorySummary",
    "path_length_m",
    "preprocess_trajectory",
    "summarize_trajectory",
]
