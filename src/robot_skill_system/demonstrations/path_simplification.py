"""Deterministic simplification of one anchor-relative 3-D motion segment.

Callers must split a path at contact and gripper-state boundaries before using
this helper.  Both endpoints are always retained, so separately simplified
segments preserve those semantic boundaries exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M = 0.004

FloatArray = NDArray[np.float64]
AnchorRelativePointM = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PathSimplificationProvenance:
    """Auditable metrics for a deterministic local simplification."""

    original_sample_count: int
    simplified_sample_count: int
    maximum_error_m: float
    tolerance_m: float
    chosen_primitive_id: str

    def as_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-compatible provenance mapping."""

        return {
            "original_sample_count": self.original_sample_count,
            "simplified_sample_count": self.simplified_sample_count,
            "maximum_error_m": self.maximum_error_m,
            "tolerance_m": self.tolerance_m,
            "chosen_primitive_id": self.chosen_primitive_id,
        }


@dataclass(frozen=True, slots=True)
class SimplifiedAnchorRelativePath:
    """A reduced 3-D path and the evidence supporting the reduction."""

    positions_m: tuple[AnchorRelativePointM, ...]
    retained_sample_indices: tuple[int, ...]
    provenance: PathSimplificationProvenance


def simplify_anchor_relative_path(
    positions_m: Sequence[Sequence[float]],
    *,
    maximum_error_m: float = DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M,
) -> SimplifiedAnchorRelativePath:
    """Simplify one semantic path segment within a hard 4 mm error ceiling.

    A segment whose samples all lie within ``maximum_error_m`` of its endpoint
    chord is represented by its start and end as ``motion.move_l``.  Otherwise,
    ordered Ramer-Douglas-Peucker reduction is applied and the result is marked
    as the spline fallback.  Verified arcs and periodic motions must be selected
    by the local primitive fitter before this fallback helper is called.
    """

    if not math.isfinite(maximum_error_m) or not (
        0.0 < maximum_error_m <= DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M
    ):
        raise ValueError("maximum_error_m must be finite and in (0, 0.004]")
    points = _coerce_positions(positions_m)
    if len(points) < 2:
        raise ValueError("path simplification requires at least two 3-D samples")
    if float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) <= 1.0e-12:
        raise ValueError("path simplification requires non-zero motion")

    endpoint_distances = _distances_to_segment(points, points[0], points[-1])
    endpoint_maximum_error_m = float(np.max(endpoint_distances))
    retained_indices: tuple[int, ...]
    if endpoint_maximum_error_m <= maximum_error_m:
        retained_indices = (0, len(points) - 1)
        chosen_primitive_id = "motion.move_l"
        actual_maximum_error_m = endpoint_maximum_error_m
    else:
        retained_indices = _rdp_indices(points, maximum_error_m)
        chosen_primitive_id = "motion.move_spline"
        actual_maximum_error_m = _maximum_ordered_polyline_error(
            points, retained_indices
        )

    simplified_positions = tuple(
        _point_tuple(points[index]) for index in retained_indices
    )
    return SimplifiedAnchorRelativePath(
        positions_m=simplified_positions,
        retained_sample_indices=retained_indices,
        provenance=PathSimplificationProvenance(
            original_sample_count=len(points),
            simplified_sample_count=len(simplified_positions),
            maximum_error_m=actual_maximum_error_m,
            tolerance_m=maximum_error_m,
            chosen_primitive_id=chosen_primitive_id,
        ),
    )


def _coerce_positions(positions_m: Sequence[Sequence[float]]) -> FloatArray:
    try:
        points = np.asarray(positions_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions_m must contain numeric 3-D samples") from exc
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("positions_m must have shape (sample_count, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions_m must contain only finite metre values")
    return points


def _distances_to_segment(
    points: FloatArray,
    start: FloatArray,
    end: FloatArray,
) -> FloatArray:
    direction = end - start
    squared_length = float(np.dot(direction, direction))
    if squared_length <= 1.0e-24:
        return cast(FloatArray, np.linalg.norm(points - start, axis=1))
    parameters = np.clip(((points - start) @ direction) / squared_length, 0.0, 1.0)
    projections = start + parameters[:, None] * direction
    return cast(FloatArray, np.linalg.norm(points - projections, axis=1))


def _rdp_indices(points: FloatArray, tolerance_m: float) -> tuple[int, ...]:
    retained = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start_index, end_index = pending.pop()
        if end_index - start_index <= 1:
            continue
        interior = points[start_index + 1 : end_index]
        distances = _distances_to_segment(
            interior, points[start_index], points[end_index]
        )
        relative_index = int(np.argmax(distances))
        maximum_distance_m = float(distances[relative_index])
        if maximum_distance_m <= tolerance_m:
            continue
        split_index = start_index + relative_index + 1
        retained.add(split_index)
        pending.append((split_index, end_index))
        pending.append((start_index, split_index))
    return tuple(sorted(retained))


def _maximum_ordered_polyline_error(
    points: FloatArray,
    retained_indices: tuple[int, ...],
) -> float:
    maximum_error_m = 0.0
    for start_index, end_index in zip(
        retained_indices, retained_indices[1:], strict=False
    ):
        span = points[start_index : end_index + 1]
        distances = _distances_to_segment(
            span, points[start_index], points[end_index]
        )
        maximum_error_m = max(maximum_error_m, float(np.max(distances)))
    return maximum_error_m


def _point_tuple(point: FloatArray) -> AnchorRelativePointM:
    return float(point[0]), float(point[1]), float(point[2])


__all__ = [
    "AnchorRelativePointM",
    "DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M",
    "PathSimplificationProvenance",
    "SimplifiedAnchorRelativePath",
    "simplify_anchor_relative_path",
]
