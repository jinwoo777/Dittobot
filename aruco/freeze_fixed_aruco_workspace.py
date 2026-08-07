#!/usr/bin/env python3
"""Freeze one measured ArUco plane as an immutable workspace reference.

This CPU-only utility does not connect to ROS, the camera, or the robot.  The
operator must capture ``plane_workspace_result.json`` at the fixed Doosan M0609
joint posture ``[0, 0, 90, 0, 90, -90]`` degrees.  That posture is recorded as
metadata; this script cannot independently observe or verify the robot joints.

All translations and XYZ/XY values stored in the output are metres.  Transform
names use ``T_A_B`` to mean "coordinates in frame B -> coordinates in frame A":

    p_camera = T_camera_plane @ p_plane
    p_tcp    = T_tcp_camera @ p_camera

The generated NPZ is create-only.  It must not be overwritten per object.  The
per-object Z range belongs in ``runtime/runtime_workspace.npz`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_JOINT_DEG = np.asarray([0.0, 0.0, 90.0, 0.0, 90.0, -90.0])
CLOSED_TIP_CLEARANCE_M = 0.184
DEFAULT_SAFETY_MARGIN_M = 0.005
GRIPPER_RADIUS_M = 0.110

# A nearly collinear marker hull is a scan failure, not a usable 2-D workspace.
# This is only a data-quality floor; it is not an XY collision/tool-footprint margin.
MINIMUM_WORKSPACE_WIDTH_M = 0.005
NUMERIC_TOLERANCE_M = 1e-9
TRANSFORM_TOLERANCE = 1e-6
SCHEMA_VERSION = "2.0"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _finite_scalar(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric with shape {shape}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _validate_homogeneous(transform: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a rigid, metre-valued 4x4 homogeneous transform."""
    result = _finite_array(transform, (4, 4), name).copy()
    if not np.allclose(
        result[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=TRANSFORM_TOLERANCE
    ):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = result[:3, :3]
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
    return result


def _load_homogeneous_npy(
    path: Path, translation_unit: str
) -> tuple[np.ndarray, str]:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Camera/TCP calibration must be a .npy file: {path}")

    checksum_before = _sha256_file(path)
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read camera/TCP calibration NPY: {path}") from exc
    checksum_after = _sha256_file(path)
    if checksum_before != checksum_after:
        raise RuntimeError("Camera/TCP calibration changed while it was being frozen")

    transform = _finite_array(raw, (4, 4), "camera/TCP calibration")
    result = transform.copy()
    if translation_unit == "mm":
        result[:3, 3] /= 1000.0
    elif translation_unit != "m":
        raise ValueError(f"Unsupported translation unit: {translation_unit}")
    return _validate_homogeneous(result, "T_tcp_camera input"), checksum_before


def _normalise_plane(normal: np.ndarray, d_m: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Plane normal has near-zero or non-finite length")
    return normal / norm, d_m / norm


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
    """Validate an ordered convex polygon and return it, area, and minimum width.

    ``minimum_width_m`` rejects marker centres that are technically non-collinear
    only because of depth noise.  No polygon is expanded or otherwise corrected.
    """
    try:
        polygon = np.asarray(polygon_xy_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace boundary must be a numeric Nx2 polygon") from exc
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
    ):
        raise ValueError("Workspace boundary must contain >=3 finite XY vertices")

    pairwise = polygon[:, None, :] - polygon[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    distances += np.eye(len(polygon), dtype=np.float64)
    if float(np.min(distances)) <= NUMERIC_TOLERANCE_M:
        raise ValueError("Workspace boundary contains duplicate vertices")

    signed_area_m2 = _polygon_signed_area_m2(polygon)
    area_m2 = abs(signed_area_m2)
    if not math.isfinite(area_m2) or area_m2 <= NUMERIC_TOLERANCE_M**2:
        raise ValueError("Workspace boundary has zero area")
    orientation = 1.0 if signed_area_m2 > 0.0 else -1.0

    edge_widths_m: list[float] = []
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        edge = second - first
        edge_length_m = float(np.linalg.norm(edge))
        if edge_length_m <= NUMERIC_TOLERANCE_M:
            raise ValueError("Workspace boundary contains a zero-length edge")
        relative = polygon - first
        signed_distances_m = orientation * (
            edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        ) / edge_length_m
        if float(np.min(signed_distances_m)) < -NUMERIC_TOLERANCE_M:
            raise ValueError(
                "Workspace boundary must be a non-self-intersecting convex polygon "
                "with ordered vertices"
            )
        edge_widths_m.append(float(np.max(signed_distances_m)))

    minimum_polygon_width_m = min(edge_widths_m)
    if minimum_polygon_width_m < minimum_width_m:
        raise ValueError(
            "Workspace boundary is nearly collinear: minimum width "
            f"{minimum_polygon_width_m * 1000.0:.3f} mm is below the "
            f"{minimum_width_m * 1000.0:.3f} mm data-quality floor. "
            "Do not expand it automatically; reacquire non-collinear marker/depth points."
        )
    return polygon.copy(), area_m2, minimum_polygon_width_m


def _paths_alias(first: Path, second: Path) -> bool:
    if first == second:
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_npz(output_path: Path, values: dict[str, object]) -> None:
    """Create an NPZ atomically without ever replacing an existing reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Frozen reference already exists and is immutable: {output_path}. "
            "Use a new path and promote it only after a separate review."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.stem + ".",
        suffix=".npz",
        dir=str(output_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    linked = False
    try:
        np.savez_compressed(temporary_path, **values)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Frozen reference appeared concurrently and was not replaced: {output_path}"
            ) from exc
        linked = True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    if linked:
        _fsync_directory(output_path.parent)


def freeze_reference(args: argparse.Namespace) -> Path:
    result_path = Path(args.plane_result_json).expanduser().resolve()
    tcp_camera_path = Path(args.tcp_camera_npy).expanduser().resolve()
    output_path = Path(args.output_npz).expanduser().resolve()

    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not tcp_camera_path.is_file():
        raise FileNotFoundError(tcp_camera_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"Frozen reference output must end in .npz: {output_path}")
    if _paths_alias(output_path, result_path) or _paths_alias(output_path, tcp_camera_path):
        raise ValueError("Frozen output must not overwrite either calibration input")

    safety_margin_m = _finite_scalar(args.safety_margin_mm, "safety margin") / 1000.0
    if not 0.0 <= safety_margin_m < CLOSED_TIP_CLEARANCE_M:
        raise ValueError("Safety margin must satisfy 0 <= margin < 184 mm")

    result_bytes = result_path.read_bytes()
    try:
        payload = json.loads(
            result_bytes.decode("utf-8"), parse_constant=_reject_nonfinite_json
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid plane workspace JSON: {result_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Plane workspace JSON root must be an object")

    plane_frame = payload.get("plane_frame")
    if not isinstance(plane_frame, dict):
        raise ValueError("Plane workspace JSON has no plane_frame object")
    frame_definition = str(plane_frame.get("definition", ""))
    if "+z=toward_camera" not in frame_definition.replace(" ", "").lower():
        raise ValueError(
            "The fixed safety frame must explicitly declare '+z=toward_camera'; "
            "an ambiguous or opposite Z direction cannot be overridden."
        )

    T_camera_plane = _validate_homogeneous(
        _finite_array(plane_frame.get("T_camera_plane"), (4, 4), "T_camera_plane"),
        "T_camera_plane",
    )
    T_plane_camera = np.linalg.inv(T_camera_plane)
    stored_inverse = plane_frame.get("T_plane_camera")
    if stored_inverse is not None:
        stored_T_plane_camera = _validate_homogeneous(
            np.asarray(stored_inverse, dtype=np.float64), "plane_frame.T_plane_camera"
        )
        if not np.allclose(
            stored_T_plane_camera,
            T_plane_camera,
            rtol=0.0,
            atol=TRANSFORM_TOLERANCE,
        ):
            raise ValueError("JSON T_plane_camera is not the inverse of T_camera_plane")

    plane_camera = payload.get("plane_camera")
    if not isinstance(plane_camera, dict):
        raise ValueError("Plane workspace JSON has no plane_camera object")
    normal_camera = _finite_array(
        plane_camera.get("normal_xyz"), (3,), "plane_camera.normal_xyz"
    )
    normal_camera, d_camera_m = _normalise_plane(
        normal_camera,
        _finite_scalar(plane_camera.get("d_m"), "plane_camera.d_m"),
    )
    if float(np.dot(normal_camera, T_camera_plane[:3, 2])) < 1.0 - TRANSFORM_TOLERANCE:
        raise ValueError("T_camera_plane +Z is not aligned with the fitted plane normal")
    plane_origin_camera_m = T_camera_plane[:3, 3]
    plane_residual_m = abs(float(np.dot(normal_camera, plane_origin_camera_m) + d_camera_m))
    if plane_residual_m > TRANSFORM_TOLERANCE:
        raise ValueError(
            f"T_camera_plane origin is {plane_residual_m:.6g} m away from the fitted plane"
        )
    camera_direction_m = float(
        np.dot(-plane_origin_camera_m, T_camera_plane[:3, 2])
    )
    if camera_direction_m <= 0.0:
        raise ValueError("T_camera_plane +Z does not point from the table toward the camera")

    workspace = payload.get("workspace_candidate")
    if not isinstance(workspace, dict):
        raise ValueError("Plane workspace JSON has no workspace_candidate object")
    if workspace.get("safe_boundary_xy_m") is None:
        raise ValueError(
            "workspace_candidate.safe_boundary_xy_m is required; the raw marker hull is not "
            "silently promoted to a safe boundary"
        )
    workspace_xy_m, polygon_area_m2, polygon_minimum_width_m = _validate_convex_polygon(
        workspace["safe_boundary_xy_m"]
    )
    workspace_reference_frame = str(workspace.get("reference_frame", ""))
    if workspace_reference_frame and workspace_reference_frame != "aruco_plane":
        raise ValueError(
            "workspace_candidate.safe_boundary_xy_m must be expressed in aruco_plane"
        )
    surface_z_plane_m = _finite_scalar(
        workspace.get("surface_z_plane_m", 0.0), "workspace surface_z_plane_m"
    )
    if abs(surface_z_plane_m) > TRANSFORM_TOLERANCE:
        raise ValueError("Workspace surface must be z=0 in the fixed plane frame")
    workspace_margin_m = _finite_scalar(
        workspace.get("workspace_margin_m", 0.0), "workspace margin"
    )
    if workspace_margin_m < 0.0:
        raise ValueError("Workspace margin must be >= 0 m")

    stored_boundary_camera = workspace.get("safe_boundary_camera_xyz_m")
    if stored_boundary_camera is not None:
        boundary_camera = np.asarray(stored_boundary_camera, dtype=np.float64)
        if boundary_camera.shape != (len(workspace_xy_m), 3) or not np.isfinite(
            boundary_camera
        ).all():
            raise ValueError("safe_boundary_camera_xyz_m must match the plane polygon")
        boundary_plane_h = np.column_stack(
            (workspace_xy_m, np.zeros(len(workspace_xy_m)), np.ones(len(workspace_xy_m)))
        )
        calculated_boundary_camera = (T_camera_plane @ boundary_plane_h.T).T[:, :3]
        if not np.allclose(
            boundary_camera,
            calculated_boundary_camera,
            rtol=0.0,
            atol=TRANSFORM_TOLERANCE,
        ):
            raise ValueError(
                "safe_boundary_camera_xyz_m is inconsistent with T_camera_plane and "
                "safe_boundary_xy_m"
            )

    input_transform, tcp_camera_source_sha256 = _load_homogeneous_npy(
        tcp_camera_path, args.tcp_camera_translation_unit
    )
    if args.tcp_camera_direction == "camera-to-tcp":
        T_tcp_camera = input_transform
    elif args.tcp_camera_direction == "tcp-to-camera":
        T_tcp_camera = np.linalg.inv(input_transform)
    else:
        raise ValueError(f"Unsupported camera/TCP direction: {args.tcp_camera_direction}")
    T_tcp_camera = _validate_homogeneous(T_tcp_camera, "canonical T_tcp_camera")

    # Fixed transform chain at the reference posture: plane -> camera -> TCP.
    T_tcp_plane = _validate_homogeneous(
        T_tcp_camera @ T_camera_plane, "T_tcp_plane"
    )
    T_plane_tcp = _validate_homogeneous(np.linalg.inv(T_tcp_plane), "T_plane_tcp")
    tcp_reference_plane_xyz_m = T_plane_tcp[:3, 3].copy()
    camera_reference_plane_xyz_m = T_plane_camera[:3, 3].copy()
    if float(camera_reference_plane_xyz_m[2]) <= 0.0:
        raise ValueError(
            "Frozen reference-camera origin must be above plane z=0 along plane +Z"
        )
    if float(camera_reference_plane_xyz_m[2]) + TRANSFORM_TOLERANCE < float(
        tcp_reference_plane_xyz_m[2]
    ):
        raise ValueError(
            "Reference camera plane-Z must be >= reference TCP plane-Z so the "
            "plane-to-camera +Z interval contains the reference TCP"
        )
    zero_width_z_min_plane_m = float(tcp_reference_plane_xyz_m[2]) - (
        CLOSED_TIP_CLEARANCE_M - safety_margin_m
    )

    # At the reference posture, closed fingertip plane-Z is exactly +0.184 m.
    # This fixed TCP-to-tip Z offset is diagnostic metadata; runtime uses the
    # explicitly required allowed_down formula.
    closed_tip_offset_from_tcp_z_m = (
        CLOSED_TIP_CLEARANCE_M - float(tcp_reference_plane_xyz_m[2])
    )

    created_at_ns = time.time_ns()
    source_status = str(payload.get("status", "unknown"))
    source_limitations = workspace.get("limitations", [])
    if not isinstance(source_limitations, list):
        source_limitations = [str(source_limitations)]

    values: dict[str, object] = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "artifact_kind": np.asarray("fixed_aruco_workspace_reference"),
        "status": np.asarray("fixed_geometry_reference_not_hardware_safety_approval"),
        "created_at_ns": np.asarray(created_at_ns, dtype=np.int64),
        "frame_name": np.asarray("aruco_plane_fixed_q_0_0_90_0_90_-90"),
        "frame_definition": np.asarray(frame_definition),
        "plane_z_direction": np.asarray("+Z toward camera / away from table"),
        "translation_unit": np.asarray("m"),
        "reference_joint_deg": REFERENCE_JOINT_DEG,
        "reference_joint_metadata_only": np.asarray(True),
        "plane_camera_coefficients": np.r_[normal_camera, d_camera_m],
        "T_camera_plane": T_camera_plane,
        "T_plane_camera": T_plane_camera,
        "T_tcp_camera": T_tcp_camera,
        "T_tcp_plane": T_tcp_plane,
        "T_plane_tcp_reference": T_plane_tcp,
        "reference_tcp_position_plane_xyz_m": tcp_reference_plane_xyz_m,
        # Compatibility alias retained for the first local draft.
        "tcp_reference_plane_xyz_m": tcp_reference_plane_xyz_m,
        "reference_camera_position_plane_xyz_m": camera_reference_plane_xyz_m,
        "camera_reference_plane_xyz_m": camera_reference_plane_xyz_m,
        "nominal_plane_to_camera_z_range_m": np.asarray(
            [0.0, camera_reference_plane_xyz_m[2]], dtype=np.float64
        ),
        "workspace_point_z_bounds_plane_m": np.asarray(
            [0.0, camera_reference_plane_xyz_m[2]], dtype=np.float64
        ),
        "zero_width_z_min_plane_m": np.asarray(zero_width_z_min_plane_m),
        "z_safe_upper_bound_source": np.asarray(
            "frozen_reference_camera_origin_plane_z"
        ),
        # Canonical requested name plus explicit-unit and compatibility aliases.
        "workspace_safe_boundary_plane_xy": workspace_xy_m,
        "workspace_safe_boundary_plane_xy_m": workspace_xy_m,
        "workspace_xy_m": workspace_xy_m,
        "workspace_boundary_source": np.asarray("safe_boundary_xy_m"),
        "workspace_source_margin_m": np.asarray(workspace_margin_m),
        "workspace_polygon_area_m2": np.asarray(polygon_area_m2),
        "workspace_polygon_minimum_width_m": np.asarray(polygon_minimum_width_m),
        "workspace_polygon_minimum_required_width_m": np.asarray(
            MINIMUM_WORKSPACE_WIDTH_M
        ),
        "closed_tip_clearance_m": np.asarray(CLOSED_TIP_CLEARANCE_M),
        "closed_tip_offset_from_tcp_z_m": np.asarray(closed_tip_offset_from_tcp_z_m),
        "safety_margin_m": np.asarray(safety_margin_m),
        "gripper_radius_m": np.asarray(GRIPPER_RADIUS_M),
        "default_width_model": np.asarray(args.width_model),
        "source_plane_result_json": np.asarray(str(result_path)),
        "source_plane_result_sha256": np.asarray(_sha256_bytes(result_bytes)),
        "source_plane_status": np.asarray(source_status),
        "source_workspace_limitations_json": np.asarray(
            json.dumps(source_limitations, ensure_ascii=False)
        ),
        "source_tcp_camera_npy": np.asarray(str(tcp_camera_path)),
        "source_tcp_camera_sha256": np.asarray(tcp_camera_source_sha256),
        "tcp_camera_input_direction": np.asarray(args.tcp_camera_direction),
        "tcp_camera_input_translation_unit": np.asarray(
            args.tcp_camera_translation_unit
        ),
        "tcp_camera_direction_stored": np.asarray("camera-to-tcp"),
        "tcp_camera_translation_unit_stored": np.asarray("m"),
    }
    if _sha256_file(result_path) != _sha256_bytes(result_bytes):
        raise RuntimeError("Plane workspace JSON changed before reference publication")
    if _sha256_file(tcp_camera_path) != tcp_camera_source_sha256:
        raise RuntimeError("Camera/TCP calibration changed before reference publication")
    _atomic_create_npz(output_path, values)

    print("=== Fixed ArUco Workspace Reference ===")
    print(f"saved                         : {output_path}")
    print(f"reference joint metadata [deg]: {REFERENCE_JOINT_DEG.tolist()}")
    print(f"TCP reference plane XYZ [m]   : {tcp_reference_plane_xyz_m.tolist()}")
    print(f"camera reference plane XYZ [m]: {camera_reference_plane_xyz_m.tolist()}")
    print(f"closed tip clearance [mm]     : {CLOSED_TIP_CLEARANCE_M * 1000.0:.1f}")
    print(f"safety margin [mm]            : {safety_margin_m * 1000.0:.1f}")
    print(f"XY polygon area [m^2]         : {polygon_area_m2:.9f}")
    print(f"XY polygon minimum width [mm] : {polygon_minimum_width_m * 1000.0:.3f}")
    print("This create-only NPZ is fixed. Per-object code must never replace it.")
    print("It is geometry metadata, not hardware/reachability/collision approval.")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plane-result-json",
        required=True,
        help="plane_workspace_result.json captured at the fixed reference posture",
    )
    parser.add_argument(
        "--tcp-camera-npy",
        required=True,
        help="Existing calibration NPY; this file is read only and never overwritten",
    )
    parser.add_argument(
        "--tcp-camera-direction",
        choices=("camera-to-tcp", "tcp-to-camera"),
        default="camera-to-tcp",
        help="Direction represented by the input matrix; output is always canonical T_tcp_camera",
    )
    parser.add_argument(
        "--tcp-camera-translation-unit",
        choices=("m", "mm"),
        default="mm",
        help="Translation unit in the input NPY; all output translations are metres",
    )
    parser.add_argument("--safety-margin-mm", type=float, default=5.0)
    parser.add_argument(
        "--width-model",
        choices=("full-opening", "legacy-half-factor"),
        default="full-opening",
        help=(
            "full-opening: w=R*sin(theta); legacy-half-factor: "
            "w=R*sin(theta)/2"
        ),
    )
    parser.add_argument(
        "--output-npz",
        default="~/Dittobot/aruco/fixed_workspace_reference.npz",
        help="Create-only frozen reference; an existing file is never replaced",
    )
    return parser


def main() -> int:
    freeze_reference(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
