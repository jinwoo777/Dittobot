#!/usr/bin/env python3
"""Build and enforce a fixed-plane TCP workspace from object width.

Only plane-Z changes per object.  The fixed ArUco plane frame, camera/TCP
calibration, reference TCP position, and safe XY polygon are copied unchanged
from ``fixed_workspace_reference.npz``.

Coordinate and unit contract
----------------------------
All internal translations, points, polygon vertices, distances, and Z bounds
are metres.  Input object width may be supplied in millimetres or centimetres.
The fixed plane has ``z=0`` on the measured table, ``+Z`` toward the reference
camera/away from the table, and decreasing Z toward the table.

The Z bounds constrain the *TCP origin*, with the gripper orientation and TCP
definition assumed unchanged from the validated reference geometry.  For the
default full-opening model, ``w`` is the full object width, so each symmetric
side of the gripper spans ``w/2``:

    theta                  = asin((w/2)/R)
    opening_offset         = R * (1 - cos(theta))
    geometric_allowed_down = 0.184 - opening_offset
    safe_allowed_down      = geometric_allowed_down - safety_margin
    z_max                  = frozen reference-camera origin plane-Z
    z_min                  = reference TCP plane-Z - safe_allowed_down

Thus ``0.184 - opening_offset`` is the requested geometric descent formula.
The pre-existing safety margin remains a separate conservative reduction of
that descent and is the predicted residual tip clearance at ``z_min``.

The optional legacy model uses ``theta = asin(2*w/R)``.  Invalid or unsafe
targets are rejected; this module never clamps or projects a point into range.
It does not issue robot, ROS, camera, CUDA, or network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPECTED_REFERENCE_JOINT_DEG = np.asarray(
    [0.0, 0.0, 90.0, 0.0, 90.0, -90.0], dtype=np.float64
)
EXPECTED_CLOSED_TIP_CLEARANCE_M = 0.184
EXPECTED_GRIPPER_RADIUS_M = 0.110
SUPPORTED_WIDTH_MODELS = frozenset({"full-opening", "legacy-half-factor"})
MINIMUM_WORKSPACE_WIDTH_M = 0.005
BOUNDARY_TOLERANCE_M = 1e-9
TRANSFORM_TOLERANCE = 1e-6
RUNTIME_SCHEMA_VERSION = "2.0"
UNSAFE_TARGET_EXIT_CODE = 2
CLI_ARGUMENT_EXIT_CODE = 2
DEFAULT_REFERENCE_NPZ = "~/Dittobot/aruco/fixed_workspace_reference.npz"
DEFAULT_RUNTIME_NPZ = "~/Dittobot/aruco/runtime/runtime_workspace.npz"


class CLIArgumentError(ValueError):
    """Raised instead of exiting when command-line arguments are invalid."""


class RuntimeArgumentParser(argparse.ArgumentParser):
    """Argument parser whose errors can invalidate a stale runtime artifact."""

    def error(self, message: str) -> None:
        raise CLIArgumentError(message)


class UnsafeWorkspaceTargetError(ValueError):
    """Raised when a target is invalid or outside the frozen runtime workspace."""

    def __init__(self, details: dict[str, object]) -> None:
        reasons = details.get("rejection_reasons", ["unsafe workspace target"])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        super().__init__("; ".join(str(reason) for reason in reasons))
        self.details = details


@dataclass(frozen=True)
class RuntimeWorkspace:
    """Per-object TCP Z interval expressed in the frozen ArUco plane frame."""

    object_width_mm: float
    theta_rad: float
    opening_offset_m: float
    allowed_down_from_reference_m: float
    z_min_plane_m: float
    z_max_plane_m: float
    z_reference_tcp_plane_m: float
    safety_margin_m: float
    closed_tip_clearance_m: float
    gripper_radius_m: float
    width_model: str

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.theta_rad)

    @property
    def object_half_width_mm(self) -> float:
        """One symmetric gripper-side span for the full object opening."""
        return self.object_width_mm / 2.0

    @property
    def geometric_allowed_down_from_reference_m(self) -> float:
        """Requested 184 mm minus opening-offset descent before safety margin."""
        return self.closed_tip_clearance_m - self.opening_offset_m

    @property
    def predicted_tip_clearance_at_z_min_m(self) -> float:
        return (
            self.closed_tip_clearance_m
            - self.opening_offset_m
            - self.allowed_down_from_reference_m
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_scalar(value: object, name: str) -> float:
    try:
        result = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _string_scalar(value: object, name: str) -> str:
    try:
        result = str(np.asarray(value).reshape(()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar string") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _integer_scalar(value: object, name: str) -> int:
    try:
        array = np.asarray(value)
        result = int(array.reshape(()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer scalar") from exc
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric with shape {shape}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array.copy()


def _validate_homogeneous(value: object, name: str) -> np.ndarray:
    transform = _finite_array(value, (4, 4), name)
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=TRANSFORM_TOLERANCE
    ):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=TRANSFORM_TOLERANCE,
    ):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=TRANSFORM_TOLERANCE):
        raise ValueError(f"{name} rotation determinant must be +1, got {determinant}")
    return transform


def _polygon_signed_area_m2(polygon_xy_m: np.ndarray) -> float:
    x_m = polygon_xy_m[:, 0]
    y_m = polygon_xy_m[:, 1]
    return 0.5 * float(
        np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m)
    )


def _validate_convex_polygon(
    polygon_xy_m: object,
    *,
    minimum_width_m: float = MINIMUM_WORKSPACE_WIDTH_M,
) -> tuple[np.ndarray, float, float]:
    try:
        polygon = np.asarray(polygon_xy_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace polygon must be a numeric Nx2 array") from exc
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
    ):
        raise ValueError("Workspace polygon must contain >=3 finite XY vertices")

    pairwise = polygon[:, None, :] - polygon[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    distances += np.eye(len(polygon), dtype=np.float64)
    if float(np.min(distances)) <= BOUNDARY_TOLERANCE_M:
        raise ValueError("Workspace polygon contains duplicate vertices")

    signed_area_m2 = _polygon_signed_area_m2(polygon)
    area_m2 = abs(signed_area_m2)
    if not math.isfinite(area_m2) or area_m2 <= BOUNDARY_TOLERANCE_M**2:
        raise ValueError("Workspace polygon has zero area")
    orientation = 1.0 if signed_area_m2 > 0.0 else -1.0

    widths_m: list[float] = []
    for index in range(len(polygon)):
        first = polygon[index]
        edge = polygon[(index + 1) % len(polygon)] - first
        edge_length_m = float(np.linalg.norm(edge))
        if edge_length_m <= BOUNDARY_TOLERANCE_M:
            raise ValueError("Workspace polygon contains a zero-length edge")
        relative = polygon - first
        signed_distances_m = orientation * (
            edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        ) / edge_length_m
        if float(np.min(signed_distances_m)) < -BOUNDARY_TOLERANCE_M:
            raise ValueError(
                "Workspace polygon must be convex, non-self-intersecting, and ordered"
            )
        widths_m.append(float(np.max(signed_distances_m)))

    minimum_polygon_width_m = min(widths_m)
    if minimum_polygon_width_m < minimum_width_m:
        raise ValueError(
            "Workspace polygon is nearly collinear: minimum width "
            f"{minimum_polygon_width_m * 1000.0:.3f} mm is below "
            f"{minimum_width_m * 1000.0:.3f} mm"
        )
    return polygon.copy(), area_m2, minimum_polygon_width_m


def _npz_value(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"Reference NPZ is missing key: {key}")
    return np.asarray(data[key])


def _read_aliased_polygon(data: np.lib.npyio.NpzFile) -> np.ndarray:
    keys = [
        key
        for key in (
            "workspace_safe_boundary_plane_xy_m",
            "workspace_safe_boundary_plane_xy",
            "workspace_xy_m",
        )
        if key in data.files
    ]
    if not keys:
        raise ValueError(
            "Reference NPZ has no workspace_safe_boundary_plane_xy(_m) polygon"
        )
    canonical = np.asarray(data[keys[0]], dtype=np.float64)
    for key in keys[1:]:
        alias = np.asarray(data[key], dtype=np.float64)
        if alias.shape != canonical.shape or not np.allclose(
            alias, canonical, rtol=0.0, atol=BOUNDARY_TOLERANCE_M
        ):
            raise ValueError(f"Reference polygon aliases disagree: {keys[0]} vs {key}")
    return canonical


def _optional_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    return _string_scalar(data[key], key)


def load_reference(path: Path | str) -> dict[str, object]:
    """Load and fully validate a frozen workspace reference.

    The returned arrays are copies, so the closed NPZ cannot be mutated through
    memory mapping.  Both the file checksum and all redundant transform/polygon
    invariants are checked fail-closed.
    """
    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if reference_path.suffix.lower() != ".npz":
        raise ValueError(f"Frozen workspace reference must be an .npz file: {reference_path}")

    checksum_before = _sha256_file(reference_path)
    try:
        with np.load(reference_path, allow_pickle=False) as data:
            schema_version = _string_scalar(
                _npz_value(data, "schema_version"), "schema_version"
            )
            if schema_version not in {"1.0", "2.0"}:
                raise ValueError(f"Unsupported frozen reference schema: {schema_version}")
            frame_definition = _string_scalar(
                _npz_value(data, "frame_definition"), "frame_definition"
            )
            if "+z=toward_camera" not in frame_definition.replace(" ", "").lower():
                raise ValueError("Frozen plane frame does not declare +Z toward camera")
            if schema_version == "2.0":
                artifact_kind = _string_scalar(
                    _npz_value(data, "artifact_kind"), "artifact_kind"
                )
                if artifact_kind != "fixed_aruco_workspace_reference":
                    raise ValueError(
                        f"Unexpected frozen reference artifact_kind: {artifact_kind}"
                    )
                artifact_status = _string_scalar(
                    _npz_value(data, "status"), "status"
                )
                if artifact_status != "fixed_geometry_reference_not_hardware_safety_approval":
                    raise ValueError(
                        f"Unexpected frozen reference status: {artifact_status}"
                    )

            reference_joint_deg = _finite_array(
                _npz_value(data, "reference_joint_deg"), (6,), "reference_joint_deg"
            )
            if not np.allclose(
                reference_joint_deg,
                EXPECTED_REFERENCE_JOINT_DEG,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "Reference joint must be exactly [0,0,90,0,90,-90] deg"
                )

            T_camera_plane = _validate_homogeneous(
                _npz_value(data, "T_camera_plane"), "T_camera_plane"
            )
            T_plane_camera = _validate_homogeneous(
                _npz_value(data, "T_plane_camera"), "T_plane_camera"
            )
            if not np.allclose(
                T_plane_camera @ T_camera_plane,
                np.eye(4),
                rtol=0.0,
                atol=TRANSFORM_TOLERANCE,
            ):
                raise ValueError("T_plane_camera is not the inverse of T_camera_plane")
            T_tcp_camera = _validate_homogeneous(
                _npz_value(data, "T_tcp_camera"), "T_tcp_camera"
            )

            if (
                schema_version == "2.0"
                and "reference_tcp_position_plane_xyz_m" not in data.files
            ):
                raise ValueError(
                    "Schema 2.0 reference lacks reference_tcp_position_plane_xyz_m"
                )
            tcp_position_key = (
                "reference_tcp_position_plane_xyz_m"
                if "reference_tcp_position_plane_xyz_m" in data.files
                else "tcp_reference_plane_xyz_m"
            )
            tcp_reference_plane_xyz_m = _finite_array(
                _npz_value(data, tcp_position_key),
                (3,),
                "reference TCP position in plane",
            )
            camera_reference_plane_xyz_m = T_plane_camera[:3, 3].copy()
            camera_position_keys = [
                key
                for key in (
                    "reference_camera_position_plane_xyz_m",
                    "camera_reference_plane_xyz_m",
                )
                if key in data.files
            ]
            if (
                schema_version == "2.0"
                and "reference_camera_position_plane_xyz_m" not in data.files
            ):
                raise ValueError(
                    "Schema 2.0 reference lacks reference_camera_position_plane_xyz_m"
                )
            for key in camera_position_keys:
                stored_camera_position = _finite_array(
                    data[key], (3,), key
                )
                if not np.allclose(
                    stored_camera_position,
                    camera_reference_plane_xyz_m,
                    rtol=0.0,
                    atol=TRANSFORM_TOLERANCE,
                ):
                    raise ValueError(
                        f"{key} disagrees with the frozen T_plane_camera"
                    )
            if float(camera_reference_plane_xyz_m[2]) <= 0.0:
                raise ValueError(
                    "Frozen reference-camera origin must be above plane z=0"
                )
            if float(camera_reference_plane_xyz_m[2]) + TRANSFORM_TOLERANCE < float(
                tcp_reference_plane_xyz_m[2]
            ):
                raise ValueError(
                    "Camera plane-Z must be >= reference TCP plane-Z"
                )

            T_tcp_plane = T_tcp_camera @ T_camera_plane
            if "T_tcp_plane" in data.files:
                stored_T_tcp_plane = _validate_homogeneous(
                    data["T_tcp_plane"], "T_tcp_plane"
                )
                if not np.allclose(
                    stored_T_tcp_plane,
                    T_tcp_plane,
                    rtol=0.0,
                    atol=TRANSFORM_TOLERANCE,
                ):
                    raise ValueError("T_tcp_plane is inconsistent with the transform chain")
            T_plane_tcp_reference = np.linalg.inv(T_tcp_plane)
            if "T_plane_tcp_reference" in data.files:
                stored_T_plane_tcp = _validate_homogeneous(
                    data["T_plane_tcp_reference"], "T_plane_tcp_reference"
                )
                if not np.allclose(
                    stored_T_plane_tcp,
                    T_plane_tcp_reference,
                    rtol=0.0,
                    atol=TRANSFORM_TOLERANCE,
                ):
                    raise ValueError("T_plane_tcp_reference is inconsistent")
            if not np.allclose(
                tcp_reference_plane_xyz_m,
                T_plane_tcp_reference[:3, 3],
                rtol=0.0,
                atol=TRANSFORM_TOLERANCE,
            ):
                raise ValueError(
                    "Stored reference TCP position disagrees with the transform chain"
                )

            workspace_xy_m, polygon_area_m2, polygon_minimum_width_m = (
                _validate_convex_polygon(_read_aliased_polygon(data))
            )
            if (
                schema_version == "2.0"
                and "workspace_safe_boundary_plane_xy" not in data.files
            ):
                raise ValueError(
                    "Schema 2.0 reference lacks workspace_safe_boundary_plane_xy"
                )
            closed_tip_clearance_m = _finite_scalar(
                _npz_value(data, "closed_tip_clearance_m"), "closed_tip_clearance_m"
            )
            if not math.isclose(
                closed_tip_clearance_m,
                EXPECTED_CLOSED_TIP_CLEARANCE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("closed_tip_clearance_m must be exactly 0.184 m")
            safety_margin_m = _finite_scalar(
                _npz_value(data, "safety_margin_m"), "safety_margin_m"
            )
            if not 0.0 <= safety_margin_m < closed_tip_clearance_m:
                raise ValueError(
                    "safety_margin_m must satisfy 0 <= margin < closed tip clearance"
                )
            zero_width_z_min_plane_m = float(tcp_reference_plane_xyz_m[2]) - (
                closed_tip_clearance_m - safety_margin_m
            )
            if "zero_width_z_min_plane_m" in data.files:
                stored_zero_width_z_min_m = _finite_scalar(
                    data["zero_width_z_min_plane_m"], "zero_width_z_min_plane_m"
                )
                if not math.isclose(
                    stored_zero_width_z_min_m,
                    zero_width_z_min_plane_m,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("zero_width_z_min_plane_m is inconsistent")
            gripper_radius_m = _finite_scalar(
                _npz_value(data, "gripper_radius_m"), "gripper_radius_m"
            )
            if not math.isclose(
                gripper_radius_m,
                EXPECTED_GRIPPER_RADIUS_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("gripper_radius_m must be exactly 0.110 m")
            default_width_model = _string_scalar(
                _npz_value(data, "default_width_model"), "default_width_model"
            )
            if default_width_model not in SUPPORTED_WIDTH_MODELS:
                raise ValueError(f"Unsupported default width model: {default_width_model}")

            stored_direction = _optional_string(data, "tcp_camera_direction_stored")
            if schema_version == "2.0" and stored_direction is None:
                raise ValueError("Schema 2.0 reference lacks transform direction metadata")
            if stored_direction is not None and stored_direction != "camera-to-tcp":
                raise ValueError("Frozen T_tcp_camera direction metadata is not canonical")
            stored_unit = _optional_string(data, "tcp_camera_translation_unit_stored")
            if schema_version == "2.0" and stored_unit is None:
                raise ValueError("Schema 2.0 reference lacks transform unit metadata")
            if stored_unit is not None and stored_unit != "m":
                raise ValueError("Frozen T_tcp_camera translation unit must be metres")
            translation_unit = _optional_string(data, "translation_unit")
            if schema_version == "2.0" and translation_unit is None:
                raise ValueError("Schema 2.0 reference lacks translation_unit")
            if translation_unit is not None and translation_unit != "m":
                raise ValueError("Frozen workspace translation_unit must be m")

            plane_coefficients = None
            if "plane_camera_coefficients" in data.files:
                plane_coefficients = _finite_array(
                    data["plane_camera_coefficients"],
                    (4,),
                    "plane_camera_coefficients",
                )
                plane_normal = plane_coefficients[:3]
                normal_norm = float(np.linalg.norm(plane_normal))
                if not math.isclose(normal_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError("Frozen plane normal must have unit length")
                if float(np.dot(plane_normal, T_camera_plane[:3, 2])) < 1.0 - 1e-6:
                    raise ValueError("Frozen plane normal disagrees with plane +Z")

            reference: dict[str, object] = {
                "schema_version": schema_version,
                "frame_name": (
                    _string_scalar(data["frame_name"], "frame_name")
                    if "frame_name" in data.files
                    else "aruco_plane_fixed"
                ),
                "frame_definition": frame_definition,
                "reference_joint_deg": reference_joint_deg,
                "T_camera_plane": T_camera_plane,
                "T_plane_camera": T_plane_camera,
                "T_tcp_camera": T_tcp_camera,
                "T_tcp_plane": _validate_homogeneous(T_tcp_plane, "derived T_tcp_plane"),
                "T_plane_tcp_reference": _validate_homogeneous(
                    T_plane_tcp_reference, "derived T_plane_tcp_reference"
                ),
                "tcp_reference_plane_xyz_m": tcp_reference_plane_xyz_m,
                "camera_reference_plane_xyz_m": camera_reference_plane_xyz_m,
                "workspace_xy_m": workspace_xy_m,
                "workspace_polygon_area_m2": polygon_area_m2,
                "workspace_polygon_minimum_width_m": polygon_minimum_width_m,
                "closed_tip_clearance_m": closed_tip_clearance_m,
                "safety_margin_m": safety_margin_m,
                "gripper_radius_m": gripper_radius_m,
                "zero_width_z_min_plane_m": zero_width_z_min_plane_m,
                "default_width_model": default_width_model,
                "plane_camera_coefficients": plane_coefficients,
                "reference_created_at_ns": (
                    _integer_scalar(data["created_at_ns"], "created_at_ns")
                    if "created_at_ns" in data.files
                    else None
                ),
                "source_plane_result_json": _optional_string(
                    data, "source_plane_result_json"
                ),
                "source_plane_result_sha256": _optional_string(
                    data, "source_plane_result_sha256"
                ),
                "source_tcp_camera_npy": _optional_string(
                    data, "source_tcp_camera_npy"
                ),
                "source_tcp_camera_sha256": _optional_string(
                    data, "source_tcp_camera_sha256"
                ),
                "source_plane_status": _optional_string(data, "source_plane_status"),
                "status": _optional_string(data, "status"),
            }
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Cannot load frozen workspace reference: {reference_path}") from exc

    checksum_after = _sha256_file(reference_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Frozen reference changed while it was being loaded")
    reference["_reference_path"] = str(reference_path)
    reference["_reference_sha256"] = checksum_before
    for value in reference.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return reference


def _validated_reference_snapshot(
    reference: dict[str, object],
) -> dict[str, object]:
    """Return a validated snapshot, reloading disk-backed frozen references.

    Reloading prevents accidental mutation of the dictionary returned by
    ``load_reference`` from changing a later safety decision.  Programmatic
    in-memory references are fully checked but have no external immutability
    guarantee and are intended for tests/offline construction only.
    """
    path_value = reference.get("_reference_path")
    checksum_value = reference.get("_reference_sha256")
    if path_value is not None or checksum_value is not None:
        if not isinstance(path_value, str) or not isinstance(checksum_value, str):
            raise ValueError("Disk-backed reference provenance is incomplete")
        trusted = load_reference(path_value)
        if trusted.get("_reference_sha256") != checksum_value:
            raise RuntimeError("Frozen reference checksum changed after initial load")
        return trusted

    joint_deg = _finite_array(
        reference["reference_joint_deg"], (6,), "reference_joint_deg"
    )
    if not np.allclose(
        joint_deg,
        EXPECTED_REFERENCE_JOINT_DEG,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Reference joint must be [0,0,90,0,90,-90] deg")
    T_camera_plane = _validate_homogeneous(
        reference["T_camera_plane"], "T_camera_plane"
    )
    T_plane_camera = _validate_homogeneous(
        reference["T_plane_camera"], "T_plane_camera"
    )
    if not np.allclose(
        T_plane_camera @ T_camera_plane,
        np.eye(4),
        rtol=0.0,
        atol=TRANSFORM_TOLERANCE,
    ):
        raise ValueError("T_plane_camera is not the inverse of T_camera_plane")
    T_tcp_camera = _validate_homogeneous(
        reference["T_tcp_camera"], "T_tcp_camera"
    )
    T_tcp_plane = _validate_homogeneous(
        T_tcp_camera @ T_camera_plane, "derived T_tcp_plane"
    )
    T_plane_tcp_reference = _validate_homogeneous(
        np.linalg.inv(T_tcp_plane), "derived T_plane_tcp_reference"
    )
    tcp_position = _finite_array(
        reference["tcp_reference_plane_xyz_m"],
        (3,),
        "reference TCP position in plane",
    )
    if not np.allclose(
        tcp_position,
        T_plane_tcp_reference[:3, 3],
        rtol=0.0,
        atol=TRANSFORM_TOLERANCE,
    ):
        raise ValueError("Reference TCP position disagrees with the transform chain")
    camera_position = _finite_array(
        reference["camera_reference_plane_xyz_m"],
        (3,),
        "reference camera position in plane",
    )
    if not np.allclose(
        camera_position,
        T_plane_camera[:3, 3],
        rtol=0.0,
        atol=TRANSFORM_TOLERANCE,
    ):
        raise ValueError("Reference camera position disagrees with T_plane_camera")
    if float(camera_position[2]) <= 0.0:
        raise ValueError("Reference camera plane-Z must be > 0 m")
    if float(camera_position[2]) + TRANSFORM_TOLERANCE < float(tcp_position[2]):
        raise ValueError("Reference camera Z must not be below reference TCP Z")

    polygon, polygon_area_m2, polygon_minimum_width_m = _validate_convex_polygon(
        reference["workspace_xy_m"]
    )
    closed_tip_clearance_m = _finite_scalar(
        reference["closed_tip_clearance_m"], "closed_tip_clearance_m"
    )
    if not math.isclose(
        closed_tip_clearance_m,
        EXPECTED_CLOSED_TIP_CLEARANCE_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("closed_tip_clearance_m must be exactly 0.184 m")
    safety_margin_m = _finite_scalar(reference["safety_margin_m"], "safety_margin_m")
    if not 0.0 <= safety_margin_m < closed_tip_clearance_m:
        raise ValueError("Invalid safety_margin_m")
    gripper_radius_m = _finite_scalar(
        reference["gripper_radius_m"], "gripper_radius_m"
    )
    if not math.isclose(
        gripper_radius_m,
        EXPECTED_GRIPPER_RADIUS_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("gripper_radius_m must be exactly 0.110 m")
    width_model = str(reference["default_width_model"])
    if width_model not in SUPPORTED_WIDTH_MODELS:
        raise ValueError(f"Unsupported default width model: {width_model}")
    frame_definition = str(reference["frame_definition"])
    if "+z=toward_camera" not in frame_definition.replace(" ", "").lower():
        raise ValueError("Reference frame does not declare +Z toward camera")

    normalized = dict(reference)
    normalized.update(
        {
            "reference_joint_deg": joint_deg,
            "T_camera_plane": T_camera_plane,
            "T_plane_camera": T_plane_camera,
            "T_tcp_camera": T_tcp_camera,
            "T_tcp_plane": T_tcp_plane,
            "T_plane_tcp_reference": T_plane_tcp_reference,
            "tcp_reference_plane_xyz_m": tcp_position,
            "camera_reference_plane_xyz_m": camera_position,
            "workspace_xy_m": polygon,
            "workspace_polygon_area_m2": polygon_area_m2,
            "workspace_polygon_minimum_width_m": polygon_minimum_width_m,
            "closed_tip_clearance_m": closed_tip_clearance_m,
            "safety_margin_m": safety_margin_m,
            "gripper_radius_m": gripper_radius_m,
            "default_width_model": width_model,
            "frame_definition": frame_definition,
        }
    )
    return normalized


def width_to_theta_rad(width_mm: float, radius_m: float, model: str) -> float:
    """Convert object width to theta on the principal physical branch [0, pi/2].

    ``full-opening`` treats ``width_mm`` as the complete symmetric object
    opening and therefore uses its half-width: ``sin(theta)=(width/2)/R``.
    The legacy model intentionally retains its historical ``2*width/R`` ratio
    only for old artifacts and remains explicitly selected by name.
    """
    width_mm = _finite_scalar(width_mm, "object width in mm")
    radius_m = _finite_scalar(radius_m, "gripper radius in m")
    if width_mm < 0.0:
        raise ValueError("Object width must be >= 0 mm")
    if radius_m <= 0.0:
        raise ValueError("Gripper radius must be > 0 m")
    if model not in SUPPORTED_WIDTH_MODELS:
        raise ValueError(f"Unknown width model: {model}")

    radius_mm = radius_m * 1000.0
    ratio = (
        (width_mm / 2.0) / radius_mm
        if model == "full-opening"
        else (2.0 * width_mm) / radius_mm
    )
    if ratio > 1.0:
        maximum_width_mm = (
            2.0 * radius_mm if model == "full-opening" else radius_mm / 2.0
        )
        raise ValueError(
            f"Width {width_mm:.6g} mm exceeds the {model} model maximum "
            f"{maximum_width_mm:.6g} mm"
        )
    return math.asin(ratio)


def build_runtime_workspace(
    reference: dict[str, object],
    object_width_mm: float,
    width_model: str | None = None,
) -> RuntimeWorkspace:
    """Calculate the per-object TCP Z bounds without changing XY or transforms."""
    reference = _validated_reference_snapshot(reference)
    model = width_model or str(reference["default_width_model"])
    radius_m = _finite_scalar(reference["gripper_radius_m"], "gripper radius")
    theta_rad = width_to_theta_rad(object_width_mm, radius_m, model)

    opening_offset_m = radius_m * (1.0 - math.cos(theta_rad))
    closed_tip_clearance_m = _finite_scalar(
        reference["closed_tip_clearance_m"], "closed tip clearance"
    )
    safety_margin_m = _finite_scalar(reference["safety_margin_m"], "safety margin")
    if not 0.0 <= safety_margin_m < closed_tip_clearance_m:
        raise ValueError("Safety margin must satisfy 0 <= margin < closed tip clearance")

    geometric_allowed_down_m = closed_tip_clearance_m - opening_offset_m
    if (
        not math.isfinite(geometric_allowed_down_m)
        or geometric_allowed_down_m <= 0.0
    ):
        raise ValueError("Computed geometric downward allowance must be finite and positive")
    allowed_down_m = geometric_allowed_down_m - safety_margin_m
    if not math.isfinite(allowed_down_m) or allowed_down_m <= 0.0:
        raise ValueError(
            "Computed safety-margined downward allowance must be finite and positive"
        )

    tcp_reference_plane_xyz_m = _finite_array(
        reference["tcp_reference_plane_xyz_m"],
        (3,),
        "reference TCP position in plane",
    )
    camera_reference_plane_xyz_m = _finite_array(
        reference.get(
            "camera_reference_plane_xyz_m",
            _validate_homogeneous(reference["T_plane_camera"], "T_plane_camera")[:3, 3],
        ),
        (3,),
        "reference camera position in plane",
    )
    z_reference_tcp_plane_m = float(tcp_reference_plane_xyz_m[2])
    z_max_plane_m = float(camera_reference_plane_xyz_m[2])
    if z_max_plane_m <= 0.0:
        raise ValueError("Reference camera plane-Z must be > 0 m")
    if z_max_plane_m + TRANSFORM_TOLERANCE < z_reference_tcp_plane_m:
        raise ValueError("Reference camera Z must not be below reference TCP Z")
    z_min_plane_m = z_reference_tcp_plane_m - allowed_down_m
    if not math.isfinite(z_min_plane_m):
        raise ValueError("Computed plane-Z lower bound is not finite")

    runtime = RuntimeWorkspace(
        object_width_mm=_finite_scalar(object_width_mm, "object width in mm"),
        theta_rad=theta_rad,
        opening_offset_m=opening_offset_m,
        allowed_down_from_reference_m=allowed_down_m,
        z_min_plane_m=z_min_plane_m,
        z_max_plane_m=z_max_plane_m,
        z_reference_tcp_plane_m=z_reference_tcp_plane_m,
        safety_margin_m=safety_margin_m,
        closed_tip_clearance_m=closed_tip_clearance_m,
        gripper_radius_m=radius_m,
        width_model=model,
    )
    if not math.isclose(
        runtime.predicted_tip_clearance_at_z_min_m,
        safety_margin_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Internal Z-bound clearance invariant failed")
    return runtime


def _validated_runtime_workspace(
    reference: dict[str, object], runtime: RuntimeWorkspace
) -> RuntimeWorkspace:
    """Recompute and verify a runtime object before checking or publishing it."""
    try:
        expected = build_runtime_workspace(
            reference, runtime.object_width_mm, runtime.width_model
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid runtime workspace: {exc}") from exc

    numeric_fields = (
        "object_width_mm",
        "theta_rad",
        "opening_offset_m",
        "allowed_down_from_reference_m",
        "z_min_plane_m",
        "z_max_plane_m",
        "z_reference_tcp_plane_m",
        "safety_margin_m",
        "closed_tip_clearance_m",
        "gripper_radius_m",
    )
    for field_name in numeric_fields:
        actual_value = _finite_scalar(getattr(runtime, field_name), field_name)
        expected_value = float(getattr(expected, field_name))
        if not math.isclose(
            actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Runtime field {field_name} is inconsistent with the frozen reference"
            )
    if runtime.width_model != expected.width_model:
        raise ValueError("Runtime width model is inconsistent with the frozen reference")
    return expected


def _point_xyz_m(value: object, name: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly three numeric metre values") from exc
    if not np.isfinite(point).all():
        raise ValueError(f"{name} must contain only finite metre values")
    return point


def _minimum_signed_boundary_distance_m(
    point_xy_m: np.ndarray, polygon_xy_m: np.ndarray
) -> float:
    signed_area_m2 = _polygon_signed_area_m2(polygon_xy_m)
    orientation = 1.0 if signed_area_m2 > 0.0 else -1.0
    distances_m: list[float] = []
    for index in range(len(polygon_xy_m)):
        first = polygon_xy_m[index]
        edge = polygon_xy_m[(index + 1) % len(polygon_xy_m)] - first
        edge_length_m = float(np.linalg.norm(edge))
        relative = point_xy_m - first
        distances_m.append(
            orientation
            * float(edge[0] * relative[1] - edge[1] * relative[0])
            / edge_length_m
        )
    return min(distances_m)


def point_in_convex_polygon(
    point_xy_m: np.ndarray,
    polygon_xy_m: np.ndarray,
    tolerance_m: float = BOUNDARY_TOLERANCE_M,
) -> bool:
    """Return whether a finite XY point is inside/on an ordered convex polygon.

    ``tolerance_m`` is a geometric distance in metres, not a cross-product
    tolerance in square metres.
    """
    tolerance_m = _finite_scalar(tolerance_m, "polygon tolerance in m")
    if tolerance_m < 0.0:
        raise ValueError("Polygon tolerance must be >= 0 m")
    try:
        point = np.asarray(point_xy_m, dtype=np.float64).reshape(2)
    except (TypeError, ValueError) as exc:
        raise ValueError("Point XY must contain exactly two numeric values") from exc
    if not np.isfinite(point).all():
        return False
    polygon, _, _ = _validate_convex_polygon(polygon_xy_m)
    return _minimum_signed_boundary_distance_m(point, polygon) >= -tolerance_m


def camera_ref_xyz_to_plane_xyz(
    reference: dict[str, object], point_camera_m: np.ndarray
) -> np.ndarray:
    """Transform frozen-reference-camera XYZ to the fixed ArUco plane frame.

    A live point from an eye-in-hand camera after the robot has moved is *not* in
    this frozen camera frame and must not be passed here without an external TF.
    """
    reference = _validated_reference_snapshot(reference)
    point = _point_xyz_m(point_camera_m, "frozen reference-camera point")
    T_plane_camera = _validate_homogeneous(
        reference["T_plane_camera"], "T_plane_camera"
    )
    result = T_plane_camera @ np.r_[point, 1.0]
    if not np.isfinite(result).all() or not math.isclose(
        float(result[3]), 1.0, rel_tol=0.0, abs_tol=TRANSFORM_TOLERANCE
    ):
        raise ValueError("Reference-camera to plane transform produced an invalid point")
    return result[:3]


def _invalid_point_details(frame: str, reason: str) -> dict[str, object]:
    return {
        "safe": False,
        "valid_input": False,
        "input_frame": frame,
        "rejection_reasons": [reason],
        "target_was_clamped": False,
    }


def check_workspace_point_plane(
    reference: dict[str, object], point_plane_m: np.ndarray
) -> tuple[bool, dict[str, object]]:
    """Check a generic geometric point in ``0 <= plane-Z <= camera-Z``.

    This predicate defines the requested plane-to-frozen-camera workspace
    volume.  It is not sufficient for a TCP target, whose gripper-dependent
    lower bound must be checked with ``check_tcp_point_plane``.
    """
    try:
        reference = _validated_reference_snapshot(reference)
        point = _point_xyz_m(point_plane_m, "plane-frame workspace point")
        polygon, _, _ = _validate_convex_polygon(reference["workspace_xy_m"])
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        return False, _invalid_point_details(
            "fixed_aruco_plane", f"invalid geometric workspace check: {exc}"
        )

    camera_z_m = float(
        _finite_array(
            reference["camera_reference_plane_xyz_m"],
            (3,),
            "reference camera position in plane",
        )[2]
    )
    signed_xy_distance_m = _minimum_signed_boundary_distance_m(point[:2], polygon)
    inside_xy = signed_xy_distance_m >= -BOUNDARY_TOLERANCE_M
    at_or_above_plane = float(point[2]) >= -BOUNDARY_TOLERANCE_M
    at_or_below_camera = float(point[2]) <= camera_z_m + BOUNDARY_TOLERANCE_M
    rejection_reasons: list[str] = []
    if not inside_xy:
        rejection_reasons.append("Workspace point XY is outside the frozen safe polygon")
    if not at_or_above_plane:
        rejection_reasons.append("Workspace point is below measured plane z=0")
    if not at_or_below_camera:
        rejection_reasons.append("Workspace point is above frozen reference-camera Z")
    safe = not rejection_reasons
    return safe, {
        "safe": safe,
        "valid_input": True,
        "input_frame": "fixed_aruco_plane",
        "target_kind": "generic_geometric_workspace_point_not_tcp",
        "point_plane_xyz_m": point.tolist(),
        "inside_xy": inside_xy,
        "at_or_above_plane_z_zero": at_or_above_plane,
        "at_or_below_frozen_camera_z": at_or_below_camera,
        "workspace_z_min_plane_m": 0.0,
        "workspace_z_max_plane_m": camera_z_m,
        "minimum_signed_xy_boundary_distance_m": signed_xy_distance_m,
        "rejection_reasons": rejection_reasons,
        "target_was_clamped": False,
        "hardware_safety_approval": False,
    }


def check_tcp_point_plane(
    reference: dict[str, object],
    runtime: RuntimeWorkspace,
    tcp_point_plane_m: np.ndarray,
    *,
    xy_tolerance_m: float = BOUNDARY_TOLERANCE_M,
) -> tuple[bool, dict[str, object]]:
    """Check one TCP origin already expressed in the fixed plane frame."""
    if not math.isfinite(xy_tolerance_m) or xy_tolerance_m < 0.0:
        return False, _invalid_point_details(
            "fixed_aruco_plane", "TCP target XY tolerance must be finite and >= 0"
        )
    try:
        reference = _validated_reference_snapshot(reference)
        runtime = _validated_runtime_workspace(reference, runtime)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        details = _invalid_point_details(
            "fixed_aruco_plane", f"invalid runtime workspace: {exc}"
        )
        return False, details
    try:
        point = _point_xyz_m(tcp_point_plane_m, "plane-frame TCP target")
    except ValueError as exc:
        details = _invalid_point_details("fixed_aruco_plane", str(exc))
        return False, details

    try:
        polygon, _, _ = _validate_convex_polygon(reference["workspace_xy_m"])
    except (KeyError, ValueError) as exc:
        details = _invalid_point_details(
            "fixed_aruco_plane", f"invalid frozen workspace reference: {exc}"
        )
        return False, details

    signed_xy_distance_m = _minimum_signed_boundary_distance_m(point[:2], polygon)
    inside_xy = signed_xy_distance_m >= -xy_tolerance_m
    distance_to_lower_z_m = float(point[2] - runtime.z_min_plane_m)
    distance_to_upper_z_m = float(runtime.z_max_plane_m - point[2])
    above_lower_z = distance_to_lower_z_m >= -BOUNDARY_TOLERANCE_M
    below_camera_z = distance_to_upper_z_m >= -BOUNDARY_TOLERANCE_M
    predicted_tip_clearance_m = (
        runtime.closed_tip_clearance_m
        - runtime.opening_offset_m
        + float(point[2] - runtime.z_reference_tcp_plane_m)
    )

    rejection_reasons: list[str] = []
    if not inside_xy:
        rejection_reasons.append("TCP target XY is outside the frozen safe polygon")
    if not above_lower_z:
        rejection_reasons.append("TCP target is below the object-width Z lower bound")
    if not below_camera_z:
        rejection_reasons.append("TCP target is above the frozen reference-camera Z upper bound")
    safe = not rejection_reasons

    details: dict[str, object] = {
        "safe": safe,
        "valid_input": True,
        "input_frame": "fixed_aruco_plane",
        "inside_xy": inside_xy,
        "above_dynamic_lower_z": above_lower_z,
        "below_reference_upper_z": below_camera_z,
        "below_frozen_camera_upper_z": below_camera_z,
        "point_plane_xyz_m": point.tolist(),
        "z_min_plane_m": runtime.z_min_plane_m,
        "z_max_plane_m": runtime.z_max_plane_m,
        "z_reference_tcp_plane_m": runtime.z_reference_tcp_plane_m,
        "minimum_signed_xy_boundary_distance_m": signed_xy_distance_m,
        "xy_tolerance_m": xy_tolerance_m,
        "distance_to_lower_z_m": distance_to_lower_z_m,
        "distance_to_upper_z_m": distance_to_upper_z_m,
        "predicted_tip_clearance_m": predicted_tip_clearance_m,
        "required_tip_clearance_m": runtime.safety_margin_m,
        "rejection_reasons": rejection_reasons,
        "target_was_clamped": False,
        "orientation_assumption": "same plane-normal gripper geometry as frozen reference",
    }
    return safe, details


def check_tcp_point_camera_ref(
    reference: dict[str, object],
    runtime: RuntimeWorkspace,
    tcp_point_camera_ref_m: np.ndarray,
) -> tuple[bool, dict[str, object]]:
    """Transform and check a TCP point in the frozen reference-camera frame."""
    try:
        camera_point = _point_xyz_m(
            tcp_point_camera_ref_m, "frozen reference-camera TCP target"
        )
        plane_point = camera_ref_xyz_to_plane_xyz(reference, camera_point)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        details = _invalid_point_details("frozen_reference_camera", str(exc))
        return False, details
    safe, details = check_tcp_point_plane(reference, runtime, plane_point)
    details["source_input_frame"] = "frozen_reference_camera"
    details["point_camera_reference_xyz_m"] = camera_point.tolist()
    return safe, details


def require_tcp_point_plane(
    reference: dict[str, object],
    runtime: RuntimeWorkspace,
    tcp_point_plane_m: np.ndarray,
) -> dict[str, object]:
    """Return check details or raise; never return an unsafe target as usable."""
    safe, details = check_tcp_point_plane(reference, runtime, tcp_point_plane_m)
    if not safe:
        raise UnsafeWorkspaceTargetError(details)
    return details


def require_tcp_point_camera_ref(
    reference: dict[str, object],
    runtime: RuntimeWorkspace,
    tcp_point_camera_ref_m: np.ndarray,
) -> dict[str, object]:
    """Transform/check a frozen-camera target and raise if it is unsafe."""
    safe, details = check_tcp_point_camera_ref(
        reference, runtime, tcp_point_camera_ref_m
    )
    if not safe:
        raise UnsafeWorkspaceTargetError(details)
    return details


def _paths_alias(first: Path, second: Path) -> bool:
    if first == second:
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def _forbidden_output_paths(reference: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    for key in (
        "_reference_path",
        "source_plane_result_json",
        "source_tcp_camera_npy",
    ):
        value = reference.get(key)
        if isinstance(value, str) and value:
            paths.append(Path(value).expanduser().resolve())
    return paths


def _reject_existing_fixed_output(output: Path) -> None:
    """Allow replacement only of a recognized runtime/rejection artifact.

    A corrupt or unknown existing file is never assumed to be disposable.  This
    also prevents a damaged frozen-reference NPZ from being overwritten merely
    because its identifying metadata can no longer be decoded.
    """
    if not output.is_file():
        return
    try:
        loaded = np.load(output, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Refusing to replace an unreadable or corrupt existing output: {output}"
        ) from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"Refusing to replace an existing non-NPZ artifact: {output}")
    with loaded as data:
        files = set(data.files)
        legacy_fixed_keys = {
            "reference_joint_deg",
            "T_camera_plane",
            "T_plane_camera",
            "T_tcp_camera",
            "closed_tip_clearance_m",
            "gripper_radius_m",
            "default_width_model",
        }
        looks_like_legacy_fixed = (
            legacy_fixed_keys.issubset(files) and "z_min_plane_m" not in files
        )
        if looks_like_legacy_fixed:
            raise ValueError(
                f"Refusing to replace an existing frozen reference: {output}"
            )
        try:
            artifact_kind = _string_scalar(data["artifact_kind"], "artifact_kind")
            schema_version = _string_scalar(data["schema_version"], "schema_version")
            status = _string_scalar(data["status"], "status")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Refusing to replace an unknown or malformed existing NPZ: {output}"
            ) from exc

    allowed_kind_status = {
        "object_width_runtime_workspace": {
            "runtime_geometry_constraint_not_hardware_safety_approval"
        },
        "rejected_object_width_runtime_workspace": {
            "unsafe_target_rejected",
            "runtime_generation_failed",
        },
    }
    if (
        schema_version != RUNTIME_SCHEMA_VERSION
        or artifact_kind not in allowed_kind_status
        or status not in allowed_kind_status[artifact_kind]
    ):
        raise ValueError(
            "Refusing to replace an existing NPZ that is not a recognized "
            f"runtime artifact: {output}"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_reference_unchanged(reference: dict[str, object]) -> None:
    path_value = reference.get("_reference_path")
    checksum = reference.get("_reference_sha256")
    if not isinstance(path_value, str) or not isinstance(checksum, str):
        return
    path = Path(path_value)
    if not path.is_file() or _sha256_file(path) != checksum:
        raise RuntimeError("Frozen reference changed after it was validated")


def save_runtime_npz(
    output_path: Path | str,
    reference: dict[str, object],
    runtime: RuntimeWorkspace,
) -> Path:
    """Atomically replace only the per-object runtime NPZ.

    The fixed reference, source JSON, and calibration NPY are explicit forbidden
    targets.  Readers therefore observe either the previous complete runtime file
    or the new complete runtime file, never a partially written ZIP archive.
    """
    reference = _validated_reference_snapshot(reference)
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError(f"Runtime workspace output must end in .npz: {output}")
    for forbidden in _forbidden_output_paths(reference):
        if _paths_alias(output, forbidden):
            raise ValueError(f"Runtime output must not replace frozen/source data: {forbidden}")
    _reject_existing_fixed_output(output)
    _verify_reference_unchanged(reference)
    runtime = _validated_runtime_workspace(reference, runtime)

    polygon, polygon_area_m2, polygon_minimum_width_m = _validate_convex_polygon(
        reference["workspace_xy_m"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.stem + ".",
        suffix=".npz",
        dir=str(output.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        values: dict[str, object] = {
            "schema_version": np.asarray(RUNTIME_SCHEMA_VERSION),
            "artifact_kind": np.asarray("object_width_runtime_workspace"),
            "status": np.asarray("runtime_geometry_constraint_not_hardware_safety_approval"),
            "usable_for_target_validation": np.asarray(True),
            "previous_runtime_invalidated": np.asarray(False),
            "runtime_generation_id": np.asarray(uuid.uuid4().hex),
            "created_at_ns": np.asarray(time.time_ns(), dtype=np.int64),
            "frame_name": np.asarray(str(reference["frame_name"])),
            "frame_definition": np.asarray(str(reference["frame_definition"])),
            "plane_z_direction": np.asarray("+Z toward camera / away from table"),
            "translation_unit": np.asarray("m"),
            "reference_joint_deg": np.asarray(
                reference["reference_joint_deg"], dtype=np.float64
            ),
            "T_camera_plane": np.asarray(reference["T_camera_plane"], dtype=np.float64),
            "T_plane_camera": np.asarray(reference["T_plane_camera"], dtype=np.float64),
            "T_tcp_camera": np.asarray(reference["T_tcp_camera"], dtype=np.float64),
            "T_tcp_plane": np.asarray(reference["T_tcp_plane"], dtype=np.float64),
            "T_plane_tcp_reference": np.asarray(
                reference["T_plane_tcp_reference"], dtype=np.float64
            ),
            "reference_tcp_position_plane_xyz_m": np.asarray(
                reference["tcp_reference_plane_xyz_m"], dtype=np.float64
            ),
            "tcp_reference_plane_xyz_m": np.asarray(
                reference["tcp_reference_plane_xyz_m"], dtype=np.float64
            ),
            "reference_camera_position_plane_xyz_m": np.asarray(
                reference["camera_reference_plane_xyz_m"], dtype=np.float64
            ),
            "camera_reference_plane_xyz_m": np.asarray(
                reference["camera_reference_plane_xyz_m"], dtype=np.float64
            ),
            "nominal_plane_to_camera_z_range_m": np.asarray(
                [0.0, runtime.z_max_plane_m], dtype=np.float64
            ),
            "workspace_point_z_bounds_plane_m": np.asarray(
                [0.0, runtime.z_max_plane_m], dtype=np.float64
            ),
            "workspace_z_min_plane_m": np.asarray(0.0),
            "workspace_z_max_plane_m": np.asarray(runtime.z_max_plane_m),
            "z_safe_upper_bound_source": np.asarray(
                "frozen_reference_camera_origin_plane_z"
            ),
            "workspace_safe_boundary_plane_xy": polygon,
            "workspace_safe_boundary_plane_xy_m": polygon,
            "workspace_xy_m": polygon,
            "workspace_polygon_area_m2": np.asarray(polygon_area_m2),
            "workspace_polygon_minimum_width_m": np.asarray(
                polygon_minimum_width_m
            ),
            "object_width_mm": np.asarray(runtime.object_width_mm),
            "object_half_width_mm": np.asarray(runtime.object_half_width_mm),
            "width_model": np.asarray(runtime.width_model),
            "width_to_theta_formula": np.asarray(
                "sin(theta)=(object_width_mm/2)/gripper_radius_mm"
                if runtime.width_model == "full-opening"
                else "legacy: sin(theta)=2*object_width_mm/gripper_radius_mm"
            ),
            "theta_rad": np.asarray(runtime.theta_rad),
            "theta_deg": np.asarray(runtime.theta_deg),
            "gripper_radius_m": np.asarray(runtime.gripper_radius_m),
            "opening_offset_m": np.asarray(runtime.opening_offset_m),
            "closed_tip_clearance_m": np.asarray(runtime.closed_tip_clearance_m),
            "safety_margin_m": np.asarray(runtime.safety_margin_m),
            "geometric_allowed_down_from_reference_m": np.asarray(
                runtime.geometric_allowed_down_from_reference_m
            ),
            "geometric_allowed_down_formula": np.asarray(
                "closed_tip_clearance_m - opening_offset_m"
            ),
            "safety_margin_application": np.asarray(
                "allowed_down_from_reference_m = "
                "geometric_allowed_down_from_reference_m - safety_margin_m"
            ),
            "allowed_down_from_reference_m": np.asarray(
                runtime.allowed_down_from_reference_m
            ),
            "z_min_plane_m": np.asarray(runtime.z_min_plane_m),
            "z_max_plane_m": np.asarray(runtime.z_max_plane_m),
            "z_reference_tcp_plane_m": np.asarray(
                runtime.z_reference_tcp_plane_m
            ),
            "z_bounds_plane_m": np.asarray(
                [runtime.z_min_plane_m, runtime.z_max_plane_m], dtype=np.float64
            ),
            "tcp_z_bounds_plane_m": np.asarray(
                [runtime.z_min_plane_m, runtime.z_max_plane_m], dtype=np.float64
            ),
            "predicted_tip_clearance_at_z_min_m": np.asarray(
                runtime.predicted_tip_clearance_at_z_min_m
            ),
            "unsafe_targets_are_clamped": np.asarray(False),
            "target_check_required_before_motion": np.asarray(True),
            "orientation_assumption": np.asarray(
                "same plane-normal gripper geometry as frozen reference"
            ),
            "reference_npz": np.asarray(str(reference.get("_reference_path", ""))),
            "reference_npz_sha256": np.asarray(
                str(reference.get("_reference_sha256", ""))
            ),
            "reference_created_at_ns": np.asarray(
                reference.get("reference_created_at_ns")
                if reference.get("reference_created_at_ns") is not None
                else -1,
                dtype=np.int64,
            ),
            "source_plane_result_sha256": np.asarray(
                str(reference.get("source_plane_result_sha256") or "")
            ),
            "source_tcp_camera_sha256": np.asarray(
                str(reference.get("source_tcp_camera_sha256") or "")
            ),
        }
        np.savez_compressed(temporary_path, **values)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _verify_reference_unchanged(reference)
        _reject_existing_fixed_output(output)
        os.replace(temporary_path, output)
        _fsync_directory(output.parent)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def save_rejected_runtime_npz(
    output_path: Path | str,
    reference: dict[str, object],
    runtime: RuntimeWorkspace,
    target_checks: dict[str, dict[str, object]],
) -> Path:
    """Atomically replace a stale runtime with a fail-closed rejection marker."""
    reference = _validated_reference_snapshot(reference)
    runtime = _validated_runtime_workspace(reference, runtime)
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError(f"Runtime rejection output must end in .npz: {output}")
    for forbidden in _forbidden_output_paths(reference):
        if _paths_alias(output, forbidden):
            raise ValueError(f"Runtime rejection must not replace source data: {forbidden}")
    _reject_existing_fixed_output(output)
    checks_json = json.dumps(
        target_checks,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.stem + ".rejected.",
        suffix=".npz",
        dir=str(output.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            schema_version=np.asarray(RUNTIME_SCHEMA_VERSION),
            artifact_kind=np.asarray("rejected_object_width_runtime_workspace"),
            status=np.asarray("unsafe_target_rejected"),
            usable_for_target_validation=np.asarray(False),
            previous_runtime_invalidated=np.asarray(True),
            runtime_generation_id=np.asarray(uuid.uuid4().hex),
            created_at_ns=np.asarray(time.time_ns(), dtype=np.int64),
            reference_npz=np.asarray(str(reference.get("_reference_path", ""))),
            reference_npz_sha256=np.asarray(
                str(reference.get("_reference_sha256", ""))
            ),
            object_width_mm=np.asarray(runtime.object_width_mm),
            width_model=np.asarray(runtime.width_model),
            target_checks_json=np.asarray(checks_json),
        )
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _verify_reference_unchanged(reference)
        _reject_existing_fixed_output(output)
        os.replace(temporary_path, output)
        _fsync_directory(output.parent)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def save_runtime_generation_failure_npz(
    output_path: Path | str,
    reference_path: Path | str,
    requested_width: object,
    failure: Exception,
) -> Path:
    """Invalidate a stale standard runtime path after configuration/load failure."""
    output = Path(output_path).expanduser().resolve()
    frozen_reference = Path(reference_path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError(f"Runtime failure marker must end in .npz: {output}")
    if _paths_alias(output, frozen_reference):
        raise ValueError("Failure marker must never replace the frozen reference")
    _reject_existing_fixed_output(output)

    reference_checksum = ""
    if frozen_reference.is_file():
        try:
            reference_checksum = _sha256_file(frozen_reference)
        except OSError:
            reference_checksum = ""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.stem + ".generation_failed.",
        suffix=".npz",
        dir=str(output.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            schema_version=np.asarray(RUNTIME_SCHEMA_VERSION),
            artifact_kind=np.asarray("rejected_object_width_runtime_workspace"),
            status=np.asarray("runtime_generation_failed"),
            usable_for_target_validation=np.asarray(False),
            previous_runtime_invalidated=np.asarray(True),
            runtime_generation_id=np.asarray(uuid.uuid4().hex),
            created_at_ns=np.asarray(time.time_ns(), dtype=np.int64),
            reference_npz=np.asarray(str(frozen_reference)),
            reference_npz_sha256=np.asarray(reference_checksum),
            requested_width=np.asarray(str(requested_width)),
            failure_type=np.asarray(type(failure).__name__),
            failure_message=np.asarray(str(failure)),
        )
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _reject_existing_fixed_output(output)
        os.replace(temporary_path, output)
        _fsync_directory(output.parent)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def _runtime_summary(
    reference: dict[str, object], runtime: RuntimeWorkspace
) -> dict[str, object]:
    return {
        "reference_joint_deg": np.asarray(reference["reference_joint_deg"]).tolist(),
        "frame": str(reference["frame_name"]),
        "object_width_mm": runtime.object_width_mm,
        "object_half_width_mm": runtime.object_half_width_mm,
        "width_model": runtime.width_model,
        "theta_deg": runtime.theta_deg,
        "gripper_radius_mm": runtime.gripper_radius_m * 1000.0,
        "opening_offset_mm": runtime.opening_offset_m * 1000.0,
        "closed_tip_clearance_mm": runtime.closed_tip_clearance_m * 1000.0,
        "safety_margin_mm": runtime.safety_margin_m * 1000.0,
        "geometric_allowed_down_from_reference_mm": (
            runtime.geometric_allowed_down_from_reference_m * 1000.0
        ),
        "allowed_down_from_reference_mm": (
            runtime.allowed_down_from_reference_m * 1000.0
        ),
        "tcp_reference_z_plane_mm": runtime.z_reference_tcp_plane_m * 1000.0,
        "camera_reference_z_plane_mm": runtime.z_max_plane_m * 1000.0,
        "tcp_min_safe_z_plane_mm": runtime.z_min_plane_m * 1000.0,
        "geometric_workspace_z_range_plane_mm": [0.0, runtime.z_max_plane_m * 1000.0],
        "tcp_z_range_plane_mm": [
            runtime.z_min_plane_m * 1000.0,
            runtime.z_max_plane_m * 1000.0,
        ],
        "tip_clearance_at_z_min_mm": (
            runtime.predicted_tip_clearance_at_z_min_m * 1000.0
        ),
        "workspace_rule": (
            "TCP XY stays inside the frozen polygon and "
            "z_min(width) <= TCP plane-Z <= frozen camera plane-Z; no clamping"
        ),
        "hardware_safety_approval": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = RuntimeArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-npz",
        default=DEFAULT_REFERENCE_NPZ,
    )
    width = parser.add_mutually_exclusive_group(required=True)
    width.add_argument("--width-mm", type=float)
    width.add_argument("--width-cm", type=float)
    parser.add_argument(
        "--width-model",
        choices=tuple(sorted(SUPPORTED_WIDTH_MODELS)),
        help="Omit to use the model frozen in the reference NPZ",
    )
    parser.add_argument(
        "--output-npz",
        default=DEFAULT_RUNTIME_NPZ,
        help="Per-object runtime file; atomically replaced only after validation",
    )
    parser.add_argument(
        "--check-plane-xyz",
        type=float,
        nargs=3,
        metavar=("X_M", "Y_M", "Z_M"),
        help="Reject unless this fixed-plane TCP target is inside the new workspace",
    )
    parser.add_argument(
        "--check-workspace-plane-xyz",
        type=float,
        nargs=3,
        metavar=("X_M", "Y_M", "Z_M"),
        help=(
            "Check a generic point against XY and plane z=0 through frozen camera Z; "
            "this is not a TCP/gripper clearance check"
        ),
    )
    parser.add_argument(
        "--check-camera-ref-xyz",
        type=float,
        nargs=3,
        metavar=("X_M", "Y_M", "Z_M"),
        help="Reject unless this frozen-reference-camera TCP target is safe",
    )
    return parser


def _raw_option_hint(
    arguments: list[str], option: str, default: str
) -> str:
    """Recover a path/string option even when full argparse validation fails."""
    result = default
    option_prefix = option + "="
    for index, token in enumerate(arguments):
        if token.startswith(option_prefix):
            candidate = token[len(option_prefix) :]
            if candidate:
                result = candidate
        elif token == option and index + 1 < len(arguments):
            candidate = arguments[index + 1]
            if candidate and not candidate.startswith("--"):
                result = candidate
    return result


def _raw_width_hint(arguments: list[str]) -> str:
    requested: list[str] = []
    for option, unit in (("--width-mm", "mm"), ("--width-cm", "cm")):
        sentinel = "<not supplied>"
        value = _raw_option_hint(arguments, option, sentinel)
        if value != sentinel:
            requested.append(f"{value!r} {unit}")
    return ", ".join(requested) if requested else "<missing>"


def _failure_payload_with_marker(
    *,
    output_path: Path,
    reference_path: Path,
    requested_width: object,
    failure: Exception,
) -> dict[str, object]:
    """Best-effort stale-runtime invalidation shared by all CLI failures."""
    marker_path: Path | None = None
    marker_error: str | None = None
    try:
        marker_path = save_runtime_generation_failure_npz(
            output_path,
            reference_path,
            requested_width,
            failure,
        )
    except (OSError, RuntimeError, ValueError) as marker_exc:
        marker_error = str(marker_exc)
    return {
        "runtime_workspace_published": False,
        "stale_runtime_invalidated": marker_path is not None,
        "failure_type": type(failure).__name__,
        "failure": str(failure),
        "rejection_marker_npz": str(marker_path) if marker_path else None,
        "rejection_marker_error": marker_error,
    }


def _run_cli(args: argparse.Namespace) -> int:
    reference_path = Path(args.reference_npz).expanduser().resolve()
    output_path = Path(args.output_npz).expanduser().resolve()
    width_mm = (
        _finite_scalar(args.width_mm, "object width in mm")
        if args.width_mm is not None
        else _finite_scalar(args.width_cm, "object width in cm") * 10.0
    )

    reference = load_reference(reference_path)
    runtime = build_runtime_workspace(reference, width_mm, args.width_model)
    summary = _runtime_summary(reference, runtime)

    checks: dict[str, dict[str, object]] = {}
    all_safe = True
    if args.check_plane_xyz is not None:
        safe, details = check_tcp_point_plane(
            reference, runtime, np.asarray(args.check_plane_xyz, dtype=np.float64)
        )
        checks["plane_target"] = details
        all_safe = all_safe and safe
    if args.check_workspace_plane_xyz is not None:
        safe, details = check_workspace_point_plane(
            reference,
            np.asarray(args.check_workspace_plane_xyz, dtype=np.float64),
        )
        checks["geometric_workspace_point"] = details
        all_safe = all_safe and safe
    if args.check_camera_ref_xyz is not None:
        safe, details = check_tcp_point_camera_ref(
            reference,
            runtime,
            np.asarray(args.check_camera_ref_xyz, dtype=np.float64),
        )
        checks["frozen_reference_camera_target"] = details
        all_safe = all_safe and safe
    if checks:
        summary["target_checks"] = checks

    if not all_safe:
        rejection_path = save_rejected_runtime_npz(
            output_path, reference, runtime, checks
        )
        summary["runtime_workspace_published"] = False
        summary["stale_runtime_invalidated"] = True
        summary["rejection_marker_npz"] = str(rejection_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        return UNSAFE_TARGET_EXIT_CODE

    saved_path = save_runtime_npz(output_path, reference, runtime)
    summary["runtime_workspace_published"] = True
    summary["saved_runtime_npz"] = str(saved_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(raw_arguments)
    except CLIArgumentError as exc:
        reference_path = Path(
            _raw_option_hint(
                raw_arguments, "--reference-npz", DEFAULT_REFERENCE_NPZ
            )
        ).expanduser().resolve()
        output_path = Path(
            _raw_option_hint(raw_arguments, "--output-npz", DEFAULT_RUNTIME_NPZ)
        ).expanduser().resolve()
        error_payload = _failure_payload_with_marker(
            output_path=output_path,
            reference_path=reference_path,
            requested_width=_raw_width_hint(raw_arguments),
            failure=exc,
        )
        print(
            json.dumps(error_payload, indent=2, ensure_ascii=False, allow_nan=False),
            file=sys.stderr,
        )
        return CLI_ARGUMENT_EXIT_CODE
    except SystemExit as exc:
        # argparse uses SystemExit(0) for --help.  Help is read-only and must not
        # invalidate a previously published runtime workspace.
        if exc.code == 0:
            return 0
        raise

    try:
        return _run_cli(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        reference_path = Path(args.reference_npz).expanduser().resolve()
        output_path = Path(args.output_npz).expanduser().resolve()
        requested_width = (
            f"{args.width_mm!r} mm"
            if args.width_mm is not None
            else f"{args.width_cm!r} cm"
        )
        error_payload = _failure_payload_with_marker(
            output_path=output_path,
            reference_path=reference_path,
            requested_width=requested_width,
            failure=exc,
        )
        print(
            json.dumps(error_payload, indent=2, ensure_ascii=False, allow_nan=False),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
