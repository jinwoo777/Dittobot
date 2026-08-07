#!/usr/bin/env python3
"""Create a legacy DR_BASE visualization of an ArUco plane result.

This utility never issues motion commands.  It either consumes a supplied
Doosan TCP pose or calls the read-only ``motion/fkin`` service for a supplied
joint vector.  DSR ``posx`` values are interpreted as millimetres and ZYZ Euler
angles in degrees, matching the calibration tutorial used by this workspace.

This file does not consume ``fixed_workspace_reference.npz`` or the dynamic
object-width runtime NPZ and therefore is not a target safety gate.  It is
restricted to the fixed reference posture and generic plane-to-camera heights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REFERENCE_JOINT_DEG = np.asarray([0.0, 0.0, 90.0, 0.0, 90.0, -90.0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dsr_posx_to_matrix_m(posx: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(posx) != 6 or not np.isfinite(np.asarray(posx, dtype=np.float64)).all():
        raise ValueError("Doosan posx must contain six finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "ZYZ", np.asarray(posx[3:], dtype=np.float64), degrees=True
    ).as_matrix()
    transform[:3, 3] = np.asarray(posx[:3], dtype=np.float64) / 1000.0
    return transform


def load_tcp_camera_matrix(path: Path, translation_unit: str) -> np.ndarray:
    transform = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("TCP-camera NPY must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("TCP-camera matrix has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError("TCP-camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("TCP-camera rotation determinant must be +1")
    result = transform.copy()
    if translation_unit == "mm":
        result[:3, 3] /= 1000.0
    return result


def query_fkin(
    service_name: str, joint_deg: list[float], timeout_s: float
) -> list[float]:
    """Call only the Doosan read-only forward-kinematics service."""

    try:
        import rclpy
        from dsr_msgs2.srv import Fkin
    except ImportError as exc:
        raise RuntimeError(
            "Fkin mode requires sourced ROS 2 and Doosan workspaces"
        ) from exc
    started_here = not rclpy.ok()
    if started_here:
        rclpy.init(args=None)
    node = rclpy.create_node("dittobot_aruco_fkin_reader")
    try:
        client = node.create_client(Fkin, service_name)
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"Fkin service unavailable: {service_name}")
        request = Fkin.Request()
        request.pos = [float(value) for value in joint_deg]
        request.ref = 0  # DR_BASE
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise RuntimeError("Fkin request timed out")
        response = future.result()
        if not bool(response.success):
            raise RuntimeError("Doosan Fkin service rejected the joint vector")
        return [float(value) for value in response.conv_posx]
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()


def _transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    return (transform @ homogeneous.T).T[:, :3]


def _plane_boundary_xyz(plane_result: dict[str, Any]) -> np.ndarray:
    workspace = plane_result["workspace_candidate"]
    if "safe_boundary_xy_m" not in workspace:
        raise ValueError(
            "plane result has no safe_boundary_xy_m; raw marker hull is not a safe fallback"
        )
    boundary_xy = np.asarray(workspace["safe_boundary_xy_m"], dtype=np.float64)
    if (
        boundary_xy.ndim != 2
        or boundary_xy.shape[1] != 2
        or len(boundary_xy) < 3
        or not np.isfinite(boundary_xy).all()
    ):
        raise ValueError("plane result has no valid workspace boundary")

    x_m = boundary_xy[:, 0]
    y_m = boundary_xy[:, 1]
    signed_area_m2 = 0.5 * float(
        np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m)
    )
    if abs(signed_area_m2) <= 1e-18:
        raise ValueError("safe workspace boundary has zero area")
    orientation = 1.0 if signed_area_m2 > 0.0 else -1.0
    widths_m: list[float] = []
    for index in range(len(boundary_xy)):
        first = boundary_xy[index]
        edge = boundary_xy[(index + 1) % len(boundary_xy)] - first
        edge_length_m = float(np.linalg.norm(edge))
        if edge_length_m <= 1e-9:
            raise ValueError("safe workspace boundary contains duplicate vertices")
        relative = boundary_xy - first
        distances_m = orientation * (
            edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        ) / edge_length_m
        if float(np.min(distances_m)) < -1e-9:
            raise ValueError("safe workspace boundary must be ordered and convex")
        widths_m.append(float(np.max(distances_m)))
    if min(widths_m) < 0.005:
        raise ValueError(
            "safe workspace boundary is nearly collinear (<5 mm minimum width)"
        )
    return np.column_stack((boundary_xy, np.zeros(len(boundary_xy))))


def _height_prism(
    boundary_plane_xyz: np.ndarray,
    minimum_height_m: float,
    maximum_height_m: float,
) -> np.ndarray:
    if (
        not np.isfinite(minimum_height_m)
        or not np.isfinite(maximum_height_m)
        or minimum_height_m < 0.0
        or maximum_height_m <= minimum_height_m
    ):
        raise ValueError("workspace heights must satisfy 0 <= minimum < maximum")
    lower = boundary_plane_xyz.copy()
    upper = boundary_plane_xyz.copy()
    lower[:, 2] = minimum_height_m
    upper[:, 2] = maximum_height_m
    return np.vstack((lower, upper))


def transform_workspace(args: argparse.Namespace) -> int:
    plane_path = Path(args.plane_result).resolve()
    tcp_camera_path = Path(args.tcp_camera_npy).resolve()
    plane_result = json.loads(plane_path.read_text())
    if args.base_tcp_posx is not None:
        if not args.reference_pose_confirmed:
            raise ValueError(
                "--base-tcp-posx requires --reference-pose-confirmed because this "
                "legacy tool cannot read the associated joints"
            )
        posx = [float(value) for value in args.base_tcp_posx]
        pose_source = "operator_supplied_dsr_posx"
        joint_deg: list[float] | None = None
    else:
        joint_deg = [float(value) for value in args.joint_deg]
        if not np.allclose(joint_deg, REFERENCE_JOINT_DEG, rtol=0.0, atol=1e-9):
            raise ValueError(
                "The frozen ArUco plane may only be transformed at reference joint "
                "[0,0,90,0,90,-90] deg"
            )
        posx = query_fkin(args.fkin_service, joint_deg, args.service_timeout_s)
        pose_source = f"read_only_fkin:{args.fkin_service}"

    transform_base_tcp = dsr_posx_to_matrix_m(posx)
    input_tcp_camera = load_tcp_camera_matrix(
        tcp_camera_path, args.tcp_camera_translation_unit
    )
    transform_tcp_camera = (
        input_tcp_camera
        if args.tcp_camera_direction == "camera-to-tcp"
        else np.linalg.inv(input_tcp_camera)
    )
    transform_camera_plane = np.asarray(
        plane_result["plane_frame"]["T_camera_plane"], dtype=np.float64
    )
    if transform_camera_plane.shape != (4, 4) or not np.isfinite(
        transform_camera_plane
    ).all():
        raise ValueError("plane result T_camera_plane must be a finite 4x4 matrix")
    if not np.allclose(
        transform_camera_plane[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9
    ):
        raise ValueError("plane result T_camera_plane has an invalid homogeneous row")
    plane_rotation = transform_camera_plane[:3, :3]
    if not np.allclose(
        plane_rotation.T @ plane_rotation, np.eye(3), rtol=0.0, atol=1e-6
    ) or not np.isclose(np.linalg.det(plane_rotation), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("plane result T_camera_plane is not a rigid transform")
    transform_base_camera = transform_base_tcp @ transform_tcp_camera
    transform_base_plane = transform_base_camera @ transform_camera_plane

    camera_normal = np.asarray(
        plane_result["plane_camera"]["normal_xyz"], dtype=np.float64
    )
    camera_offset = float(plane_result["plane_camera"]["d_m"])
    if (
        camera_normal.shape != (3,)
        or not np.isfinite(camera_normal).all()
        or not np.isfinite(camera_offset)
        or float(np.linalg.norm(camera_normal)) <= 1e-12
    ):
        raise ValueError("plane camera coefficients must be finite and non-zero")
    base_normal = transform_base_camera[:3, :3] @ camera_normal
    base_normal /= np.linalg.norm(base_normal)
    base_offset = camera_offset - float(
        np.dot(base_normal, transform_base_camera[:3, 3])
    )

    boundary_plane_xyz = _plane_boundary_xyz(plane_result)
    boundary_base_xyz = _transform_points(transform_base_plane, boundary_plane_xyz)
    volume_payload: dict[str, Any] | None = None
    if args.minimum_height_m is not None or args.maximum_height_m is not None:
        if args.minimum_height_m is None or args.maximum_height_m is None:
            raise ValueError("both workspace height arguments are required together")
        camera_position_plane = np.linalg.inv(transform_camera_plane) @ np.asarray(
            [0.0, 0.0, 0.0, 1.0]
        )
        camera_z_plane_m = float(camera_position_plane[2])
        if args.maximum_height_m > camera_z_plane_m + 1e-9:
            raise ValueError(
                "maximum generic workspace height must not exceed frozen camera plane-Z "
                f"({camera_z_plane_m:.6f} m)"
            )
        prism_plane_xyz = _height_prism(
            boundary_plane_xyz, args.minimum_height_m, args.maximum_height_m
        )
        prism_base_xyz = _transform_points(transform_base_plane, prism_plane_xyz)
        volume_payload = {
            "minimum_height_along_plane_normal_m": args.minimum_height_m,
            "maximum_height_along_plane_normal_m": args.maximum_height_m,
            "vertices_base_m": prism_base_xyz.tolist(),
            "conservative_axis_aligned_bounds_base_m": {
                "minimum_xyz": np.min(prism_base_xyz, axis=0).tolist(),
                "maximum_xyz": np.max(prism_base_xyz, axis=0).tolist(),
            },
        }

    result = {
        "schema_version": "1.0",
        "status": "legacy_visualization_candidate_not_runtime_safety_gate",
        "reference_frame": "doosan_base_DR_BASE",
        "source": {
            "plane_result": str(plane_path),
            "plane_result_sha256": _sha256(plane_path),
            "tcp_camera_npy": str(tcp_camera_path),
            "tcp_camera_npy_sha256": _sha256(tcp_camera_path),
            "tcp_camera_translation_unit": args.tcp_camera_translation_unit,
            "base_tcp_pose_source": pose_source,
            "joint_positions_deg": joint_deg,
            "base_tcp_posx_mm_zyz_deg": posx,
            "transform_chain": "T_base_tcp @ T_tcp_camera @ T_camera_plane",
            "tcp_camera_input_direction": args.tcp_camera_direction,
        },
        "transforms": {
            "T_base_tcp": transform_base_tcp.tolist(),
            "T_tcp_camera": transform_tcp_camera.tolist(),
            "T_base_camera": transform_base_camera.tolist(),
            "T_base_plane": transform_base_plane.tolist(),
        },
        "plane_base": {
            "normal_xyz": base_normal.tolist(),
            "d_m": base_offset,
            "equation": (
                f"{base_normal[0]:+.12f}*x {base_normal[1]:+.12f}*y "
                f"{base_normal[2]:+.12f}*z {base_offset:+.12f} = 0"
            ),
        },
        "workspace_candidate": {
            "surface_boundary_base_m": boundary_base_xyz.tolist(),
            "surface_axis_aligned_bounds_base_m": {
                "minimum_xyz": np.min(boundary_base_xyz, axis=0).tolist(),
                "maximum_xyz": np.max(boundary_base_xyz, axis=0).tolist(),
            },
            "volume": volume_payload,
            "limitations": [
                "marker hull is an observed surface footprint, not an approved safe workspace",
                "this legacy tool does not consume fixed/runtime object-width NPZ artifacts",
                "height volume is generic plane-to-camera geometry, not TCP fingertip clearance",
                "no robot-link, tool, payload, fixture, or obstacle collision result is implied",
                "DR_BASE must be mapped explicitly before use in a base_link MoveIt scene",
            ],
        },
    }
    output_path = Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-result", required=True)
    parser.add_argument("--tcp-camera-npy", required=True)
    parser.add_argument(
        "--tcp-camera-translation-unit", choices=("m", "mm"), default="mm"
    )
    parser.add_argument(
        "--tcp-camera-direction",
        choices=("camera-to-tcp", "tcp-to-camera"),
        default="camera-to-tcp",
    )
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument(
        "--base-tcp-posx",
        type=float,
        nargs=6,
        metavar=("X_MM", "Y_MM", "Z_MM", "A_DEG", "B_DEG", "C_DEG"),
    )
    parser.add_argument(
        "--reference-pose-confirmed",
        action="store_true",
        help="Required with --base-tcp-posx; asserts fixed joint [0,0,90,0,90,-90]",
    )
    pose_group.add_argument(
        "--joint-deg",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--fkin-service", default="/dsr01/motion/fkin")
    parser.add_argument("--service-timeout-s", type=float, default=3.0)
    parser.add_argument("--minimum-height-m", type=float)
    parser.add_argument("--maximum-height-m", type=float)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> int:
    return transform_workspace(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
