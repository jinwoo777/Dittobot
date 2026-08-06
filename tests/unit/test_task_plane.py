from __future__ import annotations

import math

import numpy as np
import pytest

from robot_skill_system.calibration.task_plane import (
    camera_stationarity_diagnostics,
    compose_base_task_plane,
    rigid_transform_from_matrix,
    rigid_transform_to_matrix,
    task_plane_normal_hint_diagnostics,
)
from robot_skill_system.scene.models import Quaternion, Vector3
from robot_skill_system.scene.transforms import RigidTransform


def _transform(
    translation_m: tuple[float, float, float], *, rotation_z_deg: float = 0.0
) -> np.ndarray:
    angle = math.radians(rotation_z_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    matrix[:3, 3] = translation_m
    return matrix


def test_task_plane_chain_uses_base_flange_camera_plane_order() -> None:
    base_to_flange = rigid_transform_from_matrix(_transform((0.3, 0.0, 0.2)))
    flange_to_camera = rigid_transform_from_matrix(_transform((0.0, 0.0, 0.1)))
    camera_to_plane = rigid_transform_from_matrix(_transform((0.2, 0.1, 0.5)))

    result = compose_base_task_plane(
        base_to_flange=base_to_flange,
        flange_to_camera=flange_to_camera,
        camera_to_task_plane=camera_to_plane,
    )

    assert result.translation_m.as_tuple() == pytest.approx((0.5, 0.1, 0.8))
    assert rigid_transform_to_matrix(result) == pytest.approx(
        _transform((0.5, 0.1, 0.8))
    )


def test_camera_stationarity_uses_two_mm_and_quarter_degree_limits() -> None:
    start = _transform((0.0, 0.0, 0.0))
    within = _transform((0.002, 0.0, 0.0), rotation_z_deg=0.25)
    moved = _transform((0.0021, 0.0, 0.0), rotation_z_deg=0.251)

    passed = camera_stationarity_diagnostics(start, within)
    failed = camera_stationarity_diagnostics(start, moved)

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_task_plane_matrix_rejects_non_rigid_input() -> None:
    invalid = np.eye(4, dtype=np.float64)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        rigid_transform_from_matrix(invalid)

    transform = RigidTransform(
        translation_m=Vector3(x=0.0, y=0.0, z=0.0),
        rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    assert rigid_transform_to_matrix(transform) == pytest.approx(np.eye(4))


def test_ransac_normal_hint_verifies_plane_without_owning_normal_sign() -> None:
    manual = rigid_transform_from_matrix(_transform((0.0, 0.0, 0.5)))
    opposite = _transform((0.0, 0.0, 0.5), rotation_z_deg=180.0)
    opposite[:3, :3] = np.diag((1.0, -1.0, -1.0))
    aligned_opposite_sign = rigid_transform_from_matrix(opposite)
    tilted = np.eye(4, dtype=np.float64)
    angle = math.radians(20.0)
    tilted[:3, :3] = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(angle), -math.sin(angle)),
        (0.0, math.sin(angle), math.cos(angle)),
    )

    sign_only = task_plane_normal_hint_diagnostics(manual, aligned_opposite_sign)
    mismatch = task_plane_normal_hint_diagnostics(
        manual, rigid_transform_from_matrix(tilted)
    )

    assert sign_only["passed"] is True
    assert sign_only["undirected_normal_deviation_deg"] == pytest.approx(0.0)
    assert mismatch["passed"] is False
    assert mismatch["undirected_normal_deviation_deg"] == pytest.approx(20.0)
