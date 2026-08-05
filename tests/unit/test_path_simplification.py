from __future__ import annotations

import pytest

from robot_skill_system.demonstrations.path_simplification import (
    DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M,
    simplify_anchor_relative_path,
)


def test_straight_path_at_four_mm_uses_only_move_l_endpoints() -> None:
    points = [
        (0.00, 0.000, 0.0),
        (0.05, 0.004, 0.0),
        (0.10, 0.000, 0.0),
    ]

    result = simplify_anchor_relative_path(points)

    assert result.positions_m == (points[0], points[-1])
    assert result.retained_sample_indices == (0, 2)
    assert result.provenance.original_sample_count == 3
    assert result.provenance.simplified_sample_count == 2
    assert result.provenance.maximum_error_m == pytest.approx(0.004)
    assert result.provenance.tolerance_m == DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M
    assert result.provenance.chosen_primitive_id == "motion.move_l"


def test_non_straight_path_uses_ordered_rdp_spline_with_bounded_error() -> None:
    points = [
        (0.00, 0.000, 0.0),
        (0.02, 0.001, 0.0),
        (0.04, 0.020, 0.0),
        (0.06, 0.039, 0.0),
        (0.08, 0.040, 0.0),
        (0.10, 0.040, 0.0),
    ]

    result = simplify_anchor_relative_path(points)

    assert result.positions_m[0] == points[0]
    assert result.positions_m[-1] == points[-1]
    assert result.retained_sample_indices == tuple(sorted(result.retained_sample_indices))
    assert 2 < len(result.positions_m) < len(points)
    assert result.provenance.maximum_error_m <= 0.004
    assert result.provenance.chosen_primitive_id == "motion.move_spline"
    assert result.provenance.as_dict()["simplified_sample_count"] == len(
        result.positions_m
    )


def test_separately_simplified_semantic_segments_keep_shared_boundary() -> None:
    before_gripper_transition = [
        (0.00, 0.000, 0.0),
        (0.02, 0.001, 0.0),
        (0.04, 0.000, 0.0),
    ]
    after_gripper_transition = [
        (0.04, 0.000, 0.0),
        (0.06, -0.001, 0.0),
        (0.08, 0.000, 0.0),
    ]

    first = simplify_anchor_relative_path(before_gripper_transition)
    second = simplify_anchor_relative_path(after_gripper_transition)

    assert first.positions_m[-1] == before_gripper_transition[-1]
    assert second.positions_m[0] == after_gripper_transition[0]
    assert first.positions_m[-1] == second.positions_m[0]


@pytest.mark.parametrize(
    "positions_m, error",
    [
        ([(0.0, 0.0, 0.0)], "at least two"),
        ([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], "non-zero motion"),
        ([(0.0, 0.0), (0.1, 0.0)], "shape"),
        ([(0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)], "finite"),
    ],
)
def test_simplification_rejects_invalid_geometry(
    positions_m: list[tuple[float, ...]], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        simplify_anchor_relative_path(positions_m)


def test_simplification_tolerance_cannot_exceed_four_mm() -> None:
    with pytest.raises(ValueError, match="0.004"):
        simplify_anchor_relative_path(
            [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)],
            maximum_error_m=0.0041,
        )
