"""Task-plane and fixed eye-in-hand camera transform helpers.

The functions in this module are deliberately local and deterministic.  They
never query TF or robot hardware; callers must supply checksum-bearing,
timestamped observations from an authorized adapter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.perception._geometry import (
    rotation_matrix_to_quaternion_xyzw,
)
from robot_skill_system.scene.models import Quaternion, Vector3
from robot_skill_system.scene.transforms import (
    RigidTransform,
    compose_transforms,
    rotate_vector,
)

Matrix44 = NDArray[np.float64]


def _rigid_matrix(value: Any, *, label: str) -> Matrix44:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be one finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{label} must use homogeneous rigid-transform convention")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError(f"{label} rotation must be proper")
    return matrix


def rigid_transform_from_matrix(value: Any, *, label: str = "transform") -> RigidTransform:
    """Convert a validated metre-unit homogeneous matrix into a typed transform."""

    matrix = _rigid_matrix(value, label=label)
    quaternion = rotation_matrix_to_quaternion_xyzw(matrix[:3, :3])
    return RigidTransform(
        translation_m=Vector3(
            x=float(matrix[0, 3]),
            y=float(matrix[1, 3]),
            z=float(matrix[2, 3]),
        ),
        rotation_xyzw=Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
        ),
    )


def rigid_transform_to_matrix(transform: RigidTransform) -> Matrix44:
    """Convert a typed transform into one metre-unit homogeneous matrix."""

    x, y, z, w = transform.rotation_xyzw.as_tuple()
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = transform.translation_m.as_tuple()
    return matrix


def camera_stationarity_diagnostics(
    start_base_to_flange: Any,
    end_base_to_flange: Any,
    *,
    maximum_translation_drift_m: float = 0.002,
    maximum_rotation_drift_deg: float = 0.25,
) -> dict[str, float | bool]:
    """Check whether a wrist-mounted camera stayed fixed during teaching."""

    if maximum_translation_drift_m <= 0.0 or maximum_rotation_drift_deg <= 0.0:
        raise ValueError("camera drift limits must be positive")
    start = _rigid_matrix(start_base_to_flange, label="start T_base_flange")
    end = _rigid_matrix(end_base_to_flange, label="end T_base_flange")
    translation_drift_m = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
    relative_rotation = start[:3, :3].T @ end[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_drift_deg = math.degrees(math.acos(cosine))
    passed = (
        translation_drift_m <= maximum_translation_drift_m
        and rotation_drift_deg <= maximum_rotation_drift_deg
    )
    return {
        "translation_drift_m": translation_drift_m,
        "rotation_drift_deg": rotation_drift_deg,
        "maximum_translation_drift_m": maximum_translation_drift_m,
        "maximum_rotation_drift_deg": maximum_rotation_drift_deg,
        "passed": passed,
    }


def compose_base_task_plane(
    *,
    base_to_flange: RigidTransform,
    flange_to_camera: RigidTransform,
    camera_to_task_plane: RigidTransform,
) -> RigidTransform:
    """Return ``T_base_task_plane = T_base_flange T_flange_camera T_camera_task``."""

    return compose_transforms(
        compose_transforms(base_to_flange, flange_to_camera),
        camera_to_task_plane,
    )


def task_plane_normal_hint_diagnostics(
    manual_camera_to_task_plane: RigidTransform,
    ransac_camera_to_task_plane: RigidTransform,
    *,
    maximum_undirected_deviation_deg: float = 15.0,
) -> dict[str, float | bool]:
    """Compare a manual plane normal with an advisory RANSAC normal.

    Plane direction is intentionally treated as undirected: the operator's +X/+Y
    convention owns the task-frame handedness, while RANSAC may choose the
    opposite normal sign.  The hint never replaces the manual transform.
    """

    if not 0.0 < maximum_undirected_deviation_deg <= 90.0:
        raise ValueError("maximum normal deviation must be in (0, 90] degrees")
    local_z = Vector3(x=0.0, y=0.0, z=1.0)
    manual_normal = rotate_vector(
        manual_camera_to_task_plane.rotation_xyzw, local_z
    )
    hint_normal = rotate_vector(ransac_camera_to_task_plane.rotation_xyzw, local_z)
    signed_cosine = float(
        np.clip(
            np.dot(manual_normal.as_tuple(), hint_normal.as_tuple()),
            -1.0,
            1.0,
        )
    )
    undirected_deviation_deg = math.degrees(math.acos(abs(signed_cosine)))
    return {
        "signed_normal_cosine": signed_cosine,
        "undirected_normal_deviation_deg": undirected_deviation_deg,
        "maximum_undirected_deviation_deg": maximum_undirected_deviation_deg,
        "passed": undirected_deviation_deg <= maximum_undirected_deviation_deg,
    }


__all__ = [
    "camera_stationarity_diagnostics",
    "compose_base_task_plane",
    "rigid_transform_from_matrix",
    "rigid_transform_to_matrix",
    "task_plane_normal_hint_diagnostics",
]
