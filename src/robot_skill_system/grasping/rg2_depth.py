"""Fail-closed RG2 width and depth-compensation diagnostics.

The public RG2 specification describes 110 mm as the *total stroke*.  It does
not establish a 110 mm finger-arm radius.  Consequently this module can compare
geometric hypotheses, but it deliberately does not return an executable robot
Z target.  Hardware depth compensation requires either a vendor-supported
adapter or an empirically verified width-to-Z calibration for the installed
fingertips and active TCP.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

RG2_TOTAL_STROKE_M = 0.110


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class GripWidthStatistics:
    """Summary of metric thumb-to-index distances at learned contact frames."""

    samples_m: tuple[float, ...]
    sample_count: int
    mean_m: float
    median_m: float
    population_stddev_m: float
    minimum_m: float
    maximum_m: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible payload."""

        return asdict(self)


def summarize_grip_widths(
    widths_m: Iterable[float],
    *,
    maximum_supported_width_m: float = RG2_TOTAL_STROKE_M,
) -> GripWidthStatistics:
    """Validate and summarize human fingertip distances as advisory widths.

    The returned mean is an example grip width, not an approved RG2 command.
    Mapping a human contact span to an RG2 target depends on installed fingertip
    thickness, contact geometry, compliance, and the gripper's work-range setup.
    """

    maximum = _finite_nonnegative(maximum_supported_width_m, "maximum_supported_width_m")
    if maximum <= 0.0:
        raise ValueError("maximum_supported_width_m must be greater than zero")
    samples = tuple(_finite_nonnegative(value, "width_m") for value in widths_m)
    if not samples:
        raise ValueError("at least one grip width sample is required")
    if any(value > maximum for value in samples):
        raise ValueError("grip width sample exceeds the configured width envelope")
    count = len(samples)
    mean = math.fsum(samples) / count
    ordered = sorted(samples)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    )
    variance = math.fsum((value - mean) ** 2 for value in samples) / count
    return GripWidthStatistics(
        samples_m=samples,
        sample_count=count,
        mean_m=mean,
        median_m=median,
        population_stddev_m=math.sqrt(variance),
        minimum_m=min(samples),
        maximum_m=max(samples),
    )


def circular_sagitta_m(
    total_width_change_m: float,
    *,
    calibrated_arc_radius_m: float,
) -> float:
    """Return circular sagitta for a symmetric two-finger width change.

    Each finger is assumed to move laterally by half of the *total* requested
    width change.  This is only useful after the radius and sign have been
    established for the installed fingertips; it is not the RG2 vendor model.
    """

    width = _finite_nonnegative(total_width_change_m, "total_width_change_m")
    radius = _finite_nonnegative(calibrated_arc_radius_m, "calibrated_arc_radius_m")
    if radius <= 0.0:
        raise ValueError("calibrated_arc_radius_m must be greater than zero")
    half_width = 0.5 * width
    if half_width > radius:
        raise ValueError("half of the total width change exceeds the arc radius")
    return radius - math.sqrt(radius * radius - half_width * half_width)


@dataclass(frozen=True, slots=True)
class Rg2DepthDiagnostic:
    """Non-executable comparison around a measured closed-gripper reference."""

    closed_reference_descent_m: float
    table_clearance_m: float
    baseline_descent_m: float
    example_grip_width_m: float
    rg2_total_stroke_m: float
    corrected_sagitta_if_stroke_were_radius_m: float
    mock_descent_with_corrected_hypothesis_m: float
    executable_hardware_descent_m: None
    hardware_compatible: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible payload."""

        return asdict(self)


def build_rg2_depth_diagnostic(
    *,
    closed_reference_descent_m: float,
    table_clearance_m: float,
    example_grip_width_m: float,
) -> Rg2DepthDiagnostic:
    """Compare the proposed formula without authorizing a hardware Z target."""

    reference = _finite_nonnegative(
        closed_reference_descent_m, "closed_reference_descent_m"
    )
    clearance = _finite_nonnegative(table_clearance_m, "table_clearance_m")
    width = _finite_nonnegative(example_grip_width_m, "example_grip_width_m")
    if clearance >= reference:
        raise ValueError("table_clearance_m must be smaller than the reference descent")
    if width > RG2_TOTAL_STROKE_M:
        raise ValueError("example_grip_width_m exceeds the RG2 total stroke")
    baseline = reference - clearance
    # These two values intentionally treat the specified total stroke as a radius
    # solely to make the user's hypothesis numerically inspectable.  They are not
    # vendor geometry and cannot be selected for hardware execution.
    sagitta = circular_sagitta_m(
        width, calibrated_arc_radius_m=RG2_TOTAL_STROKE_M
    )
    return Rg2DepthDiagnostic(
        closed_reference_descent_m=reference,
        table_clearance_m=clearance,
        baseline_descent_m=baseline,
        example_grip_width_m=width,
        rg2_total_stroke_m=RG2_TOTAL_STROKE_M,
        corrected_sagitta_if_stroke_were_radius_m=sagitta,
        mock_descent_with_corrected_hypothesis_m=baseline + sagitta,
        executable_hardware_descent_m=None,
        hardware_compatible=False,
        warnings=(
            "110 mm is the RG2 total stroke, not a verified finger arc radius",
            "depth-compensation magnitude and sign require installed-fingertip calibration",
            "184 mm is pose/TCP/calibration-specific and is not a transferable workspace Z",
        ),
    )
