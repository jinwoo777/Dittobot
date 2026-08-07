#!/usr/bin/env python3
"""Extract an ArUco-supported plane from a D435i RGB-D burst.

The metric result uses aligned depth and the active color intrinsics.  RealSense
camera coordinates are metres with +X right, +Y down, and +Z forward.  No robot
motion or robot connection is performed by this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}


def _detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    try:
        dictionary_id = ARUCO_DICTIONARIES[dictionary_name]
    except KeyError as exc:
        raise ValueError(f"unsupported ArUco dictionary: {dictionary_name}") from exc
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.01
    parameters.minDistanceToBorder = 0
    parameters.detectInvertedMarker = True
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id), parameters
    )


def detect_markers(
    image_bgr: np.ndarray, detector: cv2.aruco.ArucoDetector
) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(marker_corners[0], dtype=np.float64)
        for marker_corners, marker_id in zip(corners, ids.flatten())
    }


def _marker_payload(markers: dict[int, np.ndarray]) -> dict[str, Any]:
    return {
        str(marker_id): {
            "center_px": np.mean(corners, axis=0).tolist(),
            "corners_px": corners.tolist(),
        }
        for marker_id, corners in sorted(markers.items())
    }


def inspect_image(args: argparse.Namespace) -> int:
    image_path = Path(args.image).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    markers = detect_markers(image, _detector(args.dictionary))
    annotated = image.copy()
    if markers:
        ids = np.asarray(sorted(markers), dtype=np.int32).reshape(-1, 1)
        corners = [markers[int(marker_id)].astype(np.float32)[None, :, :] for marker_id in ids]
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    if args.annotated_output:
        output_path = Path(args.annotated_output).resolve()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"failed to save annotated image: {output_path}")
    payload = {
        "image": str(image_path),
        "image_size_px": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "dictionary": args.dictionary,
        "detected_ids": sorted(markers),
        "markers": _marker_payload(markers),
        "metric_plane_available": False,
        "metric_plane_blocker": (
            "a standalone JPEG has neither aligned depth nor camera intrinsics"
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _camera_matrix(fx_px: float, fy_px: float, ppx_px: float, ppy_px: float) -> np.ndarray:
    return np.asarray(
        [[fx_px, 0.0, ppx_px], [0.0, fy_px, ppy_px], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _single_marker_pose(
    corners_px: np.ndarray,
    marker_length_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    half = marker_length_m / 2.0
    object_points = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        corners_px,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or float(translation_vector[2, 0]) <= 0.0:
        raise RuntimeError("IPPE square pose estimation failed")
    projected, _jacobian = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    error = projected[:, 0, :] - corners_px
    reprojection_rmse_px = float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
    rotation, _jacobian = cv2.Rodrigues(rotation_vector)
    return rotation, translation_vector[:, 0], reprojection_rmse_px


def estimate_image_plane(args: argparse.Namespace) -> int:
    """Estimate a metric plane from known-size marker PnP in a stored RGB image."""

    image_path = Path(args.image).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    markers = detect_markers(image, _detector(args.dictionary))
    requested_ids = (
        {int(value) for value in args.marker_ids.split(",") if value.strip()}
        if args.marker_ids
        else set(markers)
    )
    missing = requested_ids - set(markers)
    if missing:
        raise RuntimeError(f"requested markers were not detected: {sorted(missing)}")
    camera_matrix = _camera_matrix(args.fx_px, args.fy_px, args.ppx_px, args.ppy_px)
    distortion = np.asarray(args.distortion, dtype=np.float64)
    centers: dict[int, np.ndarray] = {}
    pose_payload: dict[str, Any] = {}
    for marker_id in sorted(requested_ids):
        rotation, translation, error_px = _single_marker_pose(
            markers[marker_id], args.marker_length_m, camera_matrix, distortion
        )
        if error_px > args.maximum_reprojection_rmse_px:
            raise RuntimeError(
                f"marker {marker_id} reprojection RMSE {error_px:.3f}px exceeds "
                f"{args.maximum_reprojection_rmse_px:.3f}px"
            )
        centers[marker_id] = translation
        pose_payload[str(marker_id)] = {
            "center_camera_m": translation.tolist(),
            "marker_normal_camera_xyz": rotation[:, 2].tolist(),
            "reprojection_rmse_px": error_px,
            "center_px": np.mean(markers[marker_id], axis=0).tolist(),
            "corners_px": markers[marker_id].tolist(),
        }
    if len(centers) < 3:
        raise RuntimeError("image plane estimation requires at least three marker poses")
    center_points = np.asarray(list(centers.values()), dtype=np.float64)
    normal, offset, inlier_mask, residual_stats = fit_plane_ransac_svd(
        center_points, args.ransac_threshold_m
    )
    transform_camera_plane, frame_definition = make_plane_frame(
        normal, offset, centers, args.origin_id, args.x_axis_id
    )
    transform_plane_camera = np.linalg.inv(transform_camera_plane)
    marker_plane_xy = {
        marker_id: (transform_plane_camera @ np.append(center, 1.0))[:2]
        for marker_id, center in centers.items()
    }
    workspace_hull_xy = _convex_hull_xy(np.asarray(list(marker_plane_xy.values())))
    for marker_id in centers:
        pose_payload[str(marker_id)]["center_plane_xy_m"] = marker_plane_xy[
            marker_id
        ].tolist()
    annotated = image.copy()
    ids = np.asarray(sorted(requested_ids), dtype=np.int32).reshape(-1, 1)
    marker_corners = [
        markers[int(marker_id)].astype(np.float32)[None, :, :] for marker_id in ids
    ]
    cv2.aruco.drawDetectedMarkers(annotated, marker_corners, ids)
    result = {
        "schema_version": "1.0",
        "created_at_ns": time.time_ns(),
        "source": {
            "image": str(image_path),
            "method": "known_marker_size_ippe_square_then_center_plane_svd",
            "dictionary": args.dictionary,
            "marker_length_m": args.marker_length_m,
            "limitations": [
                "RGB PnP estimate; aligned depth was not available",
                "marker size and active camera intrinsics define metric scale",
                "observed marker hull is not a safety-approved robot workspace",
            ],
        },
        "camera": {
            "reference_frame": "camera_color_optical_frame",
            "coordinate_convention": "+X right, +Y down, +Z forward, metres",
            "intrinsics": {
                "width_px": int(image.shape[1]),
                "height_px": int(image.shape[0]),
                "fx_px": args.fx_px,
                "fy_px": args.fy_px,
                "ppx_px": args.ppx_px,
                "ppy_px": args.ppy_px,
                "distortion_coefficients": distortion.tolist(),
            },
        },
        "plane_camera": {
            "normal_xyz": normal.tolist(),
            "d_m": offset,
            "equation": (
                f"{normal[0]:+.12f}*x {normal[1]:+.12f}*y "
                f"{normal[2]:+.12f}*z {offset:+.12f} = 0"
            ),
            "normal_direction": "toward_camera_origin",
            "ransac_threshold_m": args.ransac_threshold_m,
            "point_count": len(center_points),
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "inlier_marker_ids": [
                marker_id
                for marker_id, keep in zip(sorted(centers), inlier_mask.tolist())
                if keep
            ],
            "residuals": residual_stats,
        },
        "plane_frame": {
            "definition": frame_definition,
            "T_camera_plane": transform_camera_plane.tolist(),
        },
        "markers": pose_payload,
        "workspace_candidate": {
            "status": "observed_marker_hull_not_safety_approved",
            "reference_frame": "aruco_plane",
            "boundary_xy_m": workspace_hull_xy.tolist(),
            "minimum_z_m": 0.0,
            "maximum_z_m": None,
            "safety_margin_applied_m": 0.0,
        },
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.annotated_output:
        annotated_path = Path(args.annotated_output).resolve()
        if annotated_path.exists():
            raise FileExistsError(f"refusing to overwrite: {annotated_path}")
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"failed to save annotated image: {annotated_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _median_depth_m(depth_frame: Any, u: float, v: float, radius_px: int) -> float | None:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    center_u = int(round(u))
    center_v = int(round(v))
    values: list[float] = []
    for y in range(max(0, center_v - radius_px), min(height, center_v + radius_px + 1)):
        for x in range(max(0, center_u - radius_px), min(width, center_u + radius_px + 1)):
            value = float(depth_frame.get_distance(x, y))
            if math.isfinite(value) and value > 0.0:
                values.append(value)
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    first, second, third = points
    normal = np.cross(second - first, third - first)
    magnitude = float(np.linalg.norm(normal))
    if magnitude < 1e-10:
        return None
    normal /= magnitude
    return normal, -float(np.dot(normal, first))


def fit_plane_ransac_svd(
    points_m: np.ndarray, threshold_m: float
) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
    if points_m.ndim != 2 or points_m.shape[1] != 3 or len(points_m) < 3:
        raise ValueError("plane fitting requires at least three 3-D points")
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    for indices in itertools.combinations(range(len(points_m)), 3):
        candidate = _plane_from_three(points_m[np.asarray(indices)])
        if candidate is None:
            continue
        normal, offset = candidate
        distances = np.abs(points_m @ normal + offset)
        mask = distances <= threshold_m
        if int(np.count_nonzero(mask)) < 3:
            continue
        score = (int(np.count_nonzero(mask)), -float(np.median(distances[mask])))
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None:
        raise RuntimeError("RANSAC could not find a non-degenerate plane")
    inliers = points_m[best_mask]
    centroid = np.mean(inliers, axis=0)
    _u, _s, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    # Point the plane normal toward the camera origin.
    if float(np.dot(normal, -centroid)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(points_m @ normal + offset)
    inlier_residuals = residuals[best_mask]
    stats = {
        "rmse_m": float(np.sqrt(np.mean(np.square(inlier_residuals)))),
        "median_m": float(np.median(inlier_residuals)),
        "maximum_m": float(np.max(inlier_residuals)),
        "p95_m": float(np.percentile(inlier_residuals, 95)),
    }
    return normal, offset, best_mask, stats


def _project_to_plane(point: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return point - (float(np.dot(normal, point)) + offset) * normal


def make_plane_frame(
    normal: np.ndarray,
    offset: float,
    marker_centers_m: dict[int, np.ndarray],
    origin_id: int | None,
    x_axis_id: int | None,
) -> tuple[np.ndarray, str]:
    if origin_id is not None:
        if origin_id not in marker_centers_m:
            raise ValueError(f"origin marker ID {origin_id} has no metric observation")
        origin = _project_to_plane(marker_centers_m[origin_id], normal, offset)
        origin_source = f"marker_{origin_id}_center"
    else:
        origin = _project_to_plane(
            np.mean(np.asarray(list(marker_centers_m.values())), axis=0), normal, offset
        )
        origin_source = "detected_marker_centroid"
    if x_axis_id is not None:
        if x_axis_id not in marker_centers_m:
            raise ValueError(f"X-axis marker ID {x_axis_id} has no metric observation")
        x_axis = _project_to_plane(marker_centers_m[x_axis_id], normal, offset) - origin
        x_source = f"origin_to_marker_{x_axis_id}"
    else:
        camera_x = np.asarray([1.0, 0.0, 0.0])
        x_axis = camera_x - float(np.dot(camera_x, normal)) * normal
        x_source = "camera_positive_x_projected_to_plane"
    if float(np.linalg.norm(x_axis)) < 1e-8:
        raise ValueError("plane X-axis is degenerate")
    x_axis /= np.linalg.norm(x_axis)
    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = origin
    return transform, f"origin={origin_source}; x_axis={x_source}; +z=toward_camera"


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 3:
        raise ValueError("workspace hull requires at least three marker centers")
    hull = cv2.convexHull(points_xy.astype(np.float32), clockwise=False, returnPoints=True)
    return hull[:, 0, :].astype(np.float64)


def _intrinsics_payload(intrinsics: Any) -> dict[str, Any]:
    return {
        "width_px": int(intrinsics.width),
        "height_px": int(intrinsics.height),
        "fx_px": float(intrinsics.fx),
        "fy_px": float(intrinsics.fy),
        "ppx_px": float(intrinsics.ppx),
        "ppy_px": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coefficients": [float(value) for value in intrinsics.coeffs],
    }


def capture_plane(args: argparse.Namespace) -> int:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("capture requires pyrealsense2") from exc

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    expected_ids = (
        {int(value) for value in args.expected_ids.split(",") if value.strip()}
        if args.expected_ids
        else set()
    )
    detector = _detector(args.dictionary)
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    observations: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    marker_frame_counts: dict[int, int] = defaultdict(int)
    last_image: np.ndarray | None = None
    last_depth_m: np.ndarray | None = None
    last_markers: dict[int, np.ndarray] = {}
    intrinsics: Any | None = None
    started_ns = time.time_ns()
    try:
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        firmware = device.get_info(rs.camera_info.firmware_version)
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
        for _ in range(args.frames):
            frames = pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            markers = detect_markers(image, detector)
            last_image = image.copy()
            last_depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(
                depth_frame.get_units()
            )
            last_markers = markers
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            for marker_id, corners in markers.items():
                if expected_ids and marker_id not in expected_ids:
                    continue
                marker_frame_counts[marker_id] += 1
                center = np.mean(corners, axis=0)
                for corner_index, corner in enumerate(corners):
                    sample = (
                        (1.0 - args.corner_inset_fraction) * corner
                        + args.corner_inset_fraction * center
                    )
                    u, v = float(sample[0]), float(sample[1])
                    if not (
                        args.edge_margin_px <= u < args.width - args.edge_margin_px
                        and args.edge_margin_px <= v < args.height - args.edge_margin_px
                    ):
                        continue
                    depth_m = _median_depth_m(depth_frame, u, v, args.depth_radius_px)
                    if depth_m is None or not (args.minimum_depth_m <= depth_m <= args.maximum_depth_m):
                        continue
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_m)
                    observations[(marker_id, corner_index)].append(
                        np.asarray(point, dtype=np.float64)
                    )
    finally:
        pipeline.stop()

    if last_image is None or last_depth_m is None or intrinsics is None:
        raise RuntimeError("D435i returned no aligned RGB-D frames")
    if expected_ids:
        never_seen = expected_ids - set(marker_frame_counts)
        if never_seen:
            raise RuntimeError(f"expected markers were never detected: {sorted(never_seen)}")

    aggregate_points: list[np.ndarray] = []
    point_labels: list[str] = []
    marker_corners_m: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    observation_counts: dict[str, int] = {}
    for key, values in sorted(observations.items()):
        marker_id, corner_index = key
        observation_counts[f"{marker_id}:{corner_index}"] = len(values)
        if len(values) < args.minimum_observations:
            continue
        median_point = np.median(np.asarray(values), axis=0)
        marker_corners_m[marker_id][corner_index] = median_point
        aggregate_points.append(median_point)
        point_labels.append(f"marker_{marker_id}_corner_{corner_index}")
    if len(aggregate_points) < 6:
        raise RuntimeError(
            "fewer than six temporally aggregated marker-corner points were available"
        )
    points_m = np.asarray(aggregate_points, dtype=np.float64)
    normal, offset, inlier_mask, residual_stats = fit_plane_ransac_svd(
        points_m, args.ransac_threshold_m
    )
    marker_centers_m = {
        marker_id: np.mean(np.asarray(list(corners.values())), axis=0)
        for marker_id, corners in marker_corners_m.items()
        if len(corners) >= 3
    }
    if len(marker_centers_m) < 3:
        raise RuntimeError("fewer than three markers have sufficient metric corners")
    transform_camera_plane, frame_definition = make_plane_frame(
        normal, offset, marker_centers_m, args.origin_id, args.x_axis_id
    )
    transform_plane_camera = np.linalg.inv(transform_camera_plane)
    marker_plane_xy = {
        marker_id: (transform_plane_camera @ np.append(center, 1.0))[:2]
        for marker_id, center in marker_centers_m.items()
    }
    workspace_hull_xy = _convex_hull_xy(np.asarray(list(marker_plane_xy.values())))
    annotated = last_image.copy()
    if last_markers:
        ids = np.asarray(sorted(last_markers), dtype=np.int32).reshape(-1, 1)
        corners = [
            last_markers[int(marker_id)].astype(np.float32)[None, :, :] for marker_id in ids
        ]
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    inlier_labels = [
        label for label, keep in zip(point_labels, inlier_mask.tolist()) if keep
    ]
    result = {
        "schema_version": "1.0",
        "created_at_ns": time.time_ns(),
        "capture_started_at_ns": started_ns,
        "camera": {
            "name": name,
            "serial": serial,
            "firmware": firmware,
            "reference_frame": "camera_color_optical_frame",
            "coordinate_convention": "+X right, +Y down, +Z forward, metres",
            "intrinsics": _intrinsics_payload(intrinsics),
        },
        "capture": {
            "width_px": args.width,
            "height_px": args.height,
            "frames_per_second": args.fps,
            "warmup_frames": args.warmup_frames,
            "sample_frames": args.frames,
            "dictionary": args.dictionary,
            "expected_ids": sorted(expected_ids),
            "marker_frame_counts": {
                str(key): value for key, value in sorted(marker_frame_counts.items())
            },
            "corner_observation_counts": observation_counts,
            "minimum_observations": args.minimum_observations,
            "depth_patch_radius_px": args.depth_radius_px,
            "corner_inset_fraction": args.corner_inset_fraction,
        },
        "plane_camera": {
            "normal_xyz": normal.tolist(),
            "d_m": offset,
            "equation": (
                f"{normal[0]:+.12f}*x {normal[1]:+.12f}*y "
                f"{normal[2]:+.12f}*z {offset:+.12f} = 0"
            ),
            "normal_direction": "toward_camera_origin",
            "ransac_threshold_m": args.ransac_threshold_m,
            "point_count": len(points_m),
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "inlier_labels": inlier_labels,
            "residuals": residual_stats,
        },
        "plane_frame": {
            "definition": frame_definition,
            "T_camera_plane": transform_camera_plane.tolist(),
        },
        "markers": {
            str(marker_id): {
                "center_camera_m": center.tolist(),
                "center_plane_xy_m": marker_plane_xy[marker_id].tolist(),
                "metric_corner_count": len(marker_corners_m[marker_id]),
            }
            for marker_id, center in sorted(marker_centers_m.items())
        },
        "workspace_candidate": {
            "status": "observed_marker_hull_not_safety_approved",
            "reference_frame": "aruco_plane",
            "boundary_xy_m": workspace_hull_xy.tolist(),
            "minimum_z_m": 0.0,
            "maximum_z_m": None,
            "safety_margin_applied_m": 0.0,
        },
    }
    result_path = output_dir / "plane_result.json"
    annotated_path = output_dir / "aruco_annotated.jpg"
    depth_path = output_dir / "last_aligned_depth_m.npz"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(f"failed to save annotated image: {annotated_path}")
    np.savez_compressed(depth_path, depth_m=last_depth_m)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {result_path}")
    print(f"saved: {annotated_path}")
    print(f"saved: {depth_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-image", help="detect ArUco IDs in an RGB image without claiming metric geometry"
    )
    inspect_parser.add_argument("--image", required=True)
    inspect_parser.add_argument("--annotated-output")
    inspect_parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTIONARIES), default="DICT_4X4_50")
    inspect_parser.set_defaults(handler=inspect_image)

    estimate_parser = subparsers.add_parser(
        "estimate-image-plane",
        help="estimate a metric camera-frame plane from known-size markers and intrinsics",
    )
    estimate_parser.add_argument("--image", required=True)
    estimate_parser.add_argument("--output-json")
    estimate_parser.add_argument("--annotated-output")
    estimate_parser.add_argument(
        "--dictionary", choices=sorted(ARUCO_DICTIONARIES), default="DICT_4X4_50"
    )
    estimate_parser.add_argument(
        "--marker-ids", help="comma-separated IDs; omitted means all detected IDs"
    )
    estimate_parser.add_argument("--marker-length-m", type=float, required=True)
    estimate_parser.add_argument("--fx-px", type=float, required=True)
    estimate_parser.add_argument("--fy-px", type=float, required=True)
    estimate_parser.add_argument("--ppx-px", type=float, required=True)
    estimate_parser.add_argument("--ppy-px", type=float, required=True)
    estimate_parser.add_argument(
        "--distortion", type=float, nargs=5, default=[0.0, 0.0, 0.0, 0.0, 0.0]
    )
    estimate_parser.add_argument("--origin-id", type=int)
    estimate_parser.add_argument("--x-axis-id", type=int)
    estimate_parser.add_argument("--ransac-threshold-m", type=float, default=0.008)
    estimate_parser.add_argument(
        "--maximum-reprojection-rmse-px", type=float, default=1.0
    )
    estimate_parser.set_defaults(handler=estimate_image_plane)

    capture_parser = subparsers.add_parser(
        "capture", help="capture aligned D435i RGB-D and fit a metric camera-frame plane"
    )
    capture_parser.add_argument("--output-dir", required=True)
    capture_parser.add_argument("--serial")
    capture_parser.add_argument("--width", type=int, default=640)
    capture_parser.add_argument("--height", type=int, default=480)
    capture_parser.add_argument("--fps", type=int, default=30)
    capture_parser.add_argument("--warmup-frames", type=int, default=60)
    capture_parser.add_argument("--frames", type=int, default=90)
    capture_parser.add_argument("--timeout-ms", type=int, default=5000)
    capture_parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTIONARIES), default="DICT_4X4_50")
    capture_parser.add_argument(
        "--expected-ids",
        help="comma-separated IDs; omitted means accept every detected ID",
    )
    capture_parser.add_argument("--origin-id", type=int)
    capture_parser.add_argument("--x-axis-id", type=int)
    capture_parser.add_argument("--minimum-observations", type=int, default=15)
    capture_parser.add_argument("--depth-radius-px", type=int, default=2)
    capture_parser.add_argument("--corner-inset-fraction", type=float, default=0.18)
    capture_parser.add_argument("--edge-margin-px", type=int, default=2)
    capture_parser.add_argument("--minimum-depth-m", type=float, default=0.1)
    capture_parser.add_argument("--maximum-depth-m", type=float, default=2.0)
    capture_parser.add_argument("--ransac-threshold-m", type=float, default=0.003)
    capture_parser.set_defaults(handler=capture_plane)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "corner_inset_fraction", 0.0) < 0.0 or getattr(
        args, "corner_inset_fraction", 0.0
    ) >= 0.5:
        parser.error("--corner-inset-fraction must be in [0, 0.5)")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
