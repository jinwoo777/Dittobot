from __future__ import annotations

import math

import pytest

from robot_skill_system.grasping.rg2_depth import (
    build_rg2_depth_diagnostic,
    circular_sagitta_m,
    summarize_grip_widths,
)


def test_summarize_grip_widths_uses_euclidean_samples_for_example_mean() -> None:
    summary = summarize_grip_widths([0.0244, 0.0188])

    assert summary.sample_count == 2
    assert summary.mean_m == pytest.approx(0.0216)
    assert summary.median_m == pytest.approx(0.0216)
    assert summary.minimum_m == pytest.approx(0.0188)
    assert summary.maximum_m == pytest.approx(0.0244)


def test_summarize_grip_widths_rejects_values_outside_rg2_envelope() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        summarize_grip_widths([0.111])


def test_circular_sagitta_uses_half_of_symmetric_total_width() -> None:
    radius_m = 0.110
    width_m = 0.030
    expected_m = radius_m - math.sqrt(radius_m**2 - (width_m / 2.0) ** 2)

    assert circular_sagitta_m(
        width_m, calibrated_arc_radius_m=radius_m
    ) == pytest.approx(expected_m)


def test_depth_diagnostic_keeps_hardware_target_unresolved() -> None:
    diagnostic = build_rg2_depth_diagnostic(
        closed_reference_descent_m=0.184,
        table_clearance_m=0.005,
        example_grip_width_m=0.030,
    )

    assert diagnostic.baseline_descent_m == pytest.approx(0.179)
    assert diagnostic.executable_hardware_descent_m is None
    assert diagnostic.hardware_compatible is False
    assert (
        diagnostic.mock_descent_with_corrected_hypothesis_m
        == pytest.approx(
            diagnostic.baseline_descent_m
            - diagnostic.corrected_sagitta_if_stroke_were_radius_m
        )
    )
