#!/usr/bin/env python3
"""3-second D435i ArUco plane/workspace scanner (CPU only).

Pipeline
--------
1. Align D435i depth to the color image.
2. Detect every visible ArUco marker for ``--duration-s`` seconds.
3. Sample a grid inside each marker. For every sample:
   - use a local depth-patch median,
   - deproject color pixel + aligned depth to a 3-D camera point,
   - aggregate repeated observations with a temporal median.
4. Fit one common plane with RANSAC, then refine the inliers with SVD.
5. Build a local plane frame and a convex candidate workspace from marker centers.
6. Optionally inset that convex polygon by ``--workspace-margin-m``.

RealSense color optical frame convention:
    +X right, +Y down, +Z forward, metres.

The produced workspace is a geometric candidate only. It is not a robot collision/
reachability certificate until it is transformed into the robot base frame and checked
against robot/tool/fixture limits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": getattr(cv2.aruco, "DICT_4X4_50", None) if hasattr(cv2, "aruco") else None,
    "DICT_4X4_100": getattr(cv2.aruco, "DICT_4X4_100", None) if hasattr(cv2, "aruco") else None,
    "DICT_4X4_250": getattr(cv2.aruco, "DICT_4X4_250", None) if hasattr(cv2, "aruco") else None,
    "DICT_5X5_50": getattr(cv2.aruco, "DICT_5X5_50", None) if hasattr(cv2, "aruco") else None,
    "DICT_6X6_50": getattr(cv2.aruco, "DICT_6X6_50", None) if hasattr(cv2, "aruco") else None,
    "DICT_7X7_50": getattr(cv2.aruco, "DICT_7X7_50", None) if hasattr(cv2, "aruco") else None,
    "DICT_ARUCO_ORIGINAL": (
        getattr(cv2.aruco, "DICT_ARUCO_ORIGINAL", None) if hasattr(cv2, "aruco") else None
    ),
}
ARUCO_DICTIONARIES = {key: value for key, value in ARUCO_DICTIONARIES.items() if value is not None}


def _parse_ids(value: str | None) -> set[int]:
    if not value:
        return set()
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if token:
            result.add(int(token))
    return result


def _make_detector(dictionary_name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "현재 OpenCV에 cv2.aruco가 없습니다. "
            "opencv-contrib-python-headless 계열 설치가 필요합니다."
        )
    if dictionary_name not in ARUCO_DICTIONARIES:
        raise ValueError(f"지원하지 않는 ArUco dictionary: {dictionary_name}")
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.detectInvertedMarker = True
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name]),
        parameters,
    )


def _detect_markers(image_bgr: np.ndarray, detector: Any) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(marker_corners[0], dtype=np.float64)
        for marker_corners, marker_id in zip(corners, ids.flatten())
    }


def _bilinear_quad(corners: np.ndarray, s: float, t: float) -> np.ndarray:
    """Map (s,t) in [0,1]^2 to an ArUco quadrilateral.

    OpenCV ArUco corner order is clockwise: top-left, top-right,
    bottom-right, bottom-left for a front-facing marker image.
    """
    c0, c1, c2, c3 = corners
    return (
        (1.0 - s) * (1.0 - t) * c0
        + s * (1.0 - t) * c1
        + s * t * c2
        + (1.0 - s) * t * c3
    )


def _marker_sample_pixels(corners: np.ndarray, grid_size: int, inset_fraction: float) -> list[np.ndarray]:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if not (0.0 < inset_fraction < 0.5):
        raise ValueError("inset_fraction must be in (0, 0.5)")
    values = np.linspace(inset_fraction, 1.0 - inset_fraction, grid_size)
    return [_bilinear_quad(corners, float(s), float(t)) for t in values for s in values]


def _median_depth_patch_m(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius_px: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> float | None:
    height, width = depth_m.shape
    x = int(round(u))
    y = int(round(v))
    x0 = max(0, x - radius_px)
    x1 = min(width, x + radius_px + 1)
    y0 = max(0, y - radius_px)
    y1 = min(height, y + radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    valid = patch[
        np.isfinite(patch)
        & (patch >= minimum_depth_m)
        & (patch <= maximum_depth_m)
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    first, second, third = points
    normal = np.cross(second - first, third - first)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-10:
        return None
    normal /= norm
    offset = -float(np.dot(normal, first))
    return normal, offset


def _fit_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    _u, _s, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, -centroid)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def fit_plane_ransac_svd(
    points_m: np.ndarray,
    threshold_m: float,
    minimum_inlier_ratio: float,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
    """Deterministic RANSAC over triplets followed by two SVD refinements."""
    if points_m.ndim != 2 or points_m.shape[1] != 3 or len(points_m) < 3:
        raise ValueError("평면 피팅에는 3개 이상의 3-D 점이 필요합니다.")
    if threshold_m <= 0.0:
        raise ValueError("RANSAC threshold must be > 0")

    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    indices = range(len(points_m))
    for triplet in itertools.combinations(indices, 3):
        candidate = _plane_from_three(points_m[np.asarray(triplet)])
        if candidate is None:
            continue
        normal, offset = candidate
        distances = np.abs(points_m @ normal + offset)
        mask = distances <= threshold_m
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        score = (count, -float(np.median(distances[mask])))
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError("RANSAC이 유효한 평면을 찾지 못했습니다.")

    # SVD refinement + reclassification. Do it twice for a stable final mask.
    mask = best_mask
    normal, offset = _fit_svd(points_m[mask])
    for _ in range(2):
        residuals = np.abs(points_m @ normal + offset)
        refined_mask = residuals <= threshold_m
        if int(np.count_nonzero(refined_mask)) < 3:
            break
        mask = refined_mask
        normal, offset = _fit_svd(points_m[mask])

    residuals = np.abs(points_m @ normal + offset)
    mask = residuals <= threshold_m
    inlier_count = int(np.count_nonzero(mask))
    inlier_ratio = inlier_count / float(len(points_m))
    if inlier_count < 3 or inlier_ratio < minimum_inlier_ratio:
        raise RuntimeError(
            "RANSAC 평면 품질이 낮습니다: "
            f"inliers={inlier_count}/{len(points_m)} ({inlier_ratio:.1%}), "
            f"required>={minimum_inlier_ratio:.1%}"
        )

    inlier_residuals = residuals[mask]
    stats = {
        "rmse_m": float(np.sqrt(np.mean(np.square(inlier_residuals)))),
        "median_m": float(np.median(inlier_residuals)),
        "p95_m": float(np.percentile(inlier_residuals, 95)),
        "maximum_m": float(np.max(inlier_residuals)),
        "inlier_ratio": float(inlier_ratio),
    }
    return normal, offset, mask, stats


def _project_to_plane(point: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return point - (float(np.dot(normal, point)) + offset) * normal


def _make_plane_frame(
    normal: np.ndarray,
    offset: float,
    marker_centers_m: dict[int, np.ndarray],
    origin_id: int | None,
    x_axis_id: int | None,
) -> tuple[np.ndarray, str]:
    if origin_id is not None:
        if origin_id not in marker_centers_m:
            raise ValueError(f"--origin-id {origin_id}가 최종 유효 마커에 없습니다.")
        origin = _project_to_plane(marker_centers_m[origin_id], normal, offset)
        origin_source = f"marker_{origin_id}_center"
    else:
        center = np.mean(np.asarray(list(marker_centers_m.values())), axis=0)
        origin = _project_to_plane(center, normal, offset)
        origin_source = "valid_marker_centroid"

    if x_axis_id is not None:
        if x_axis_id not in marker_centers_m:
            raise ValueError(f"--x-axis-id {x_axis_id}가 최종 유효 마커에 없습니다.")
        target = _project_to_plane(marker_centers_m[x_axis_id], normal, offset)
        x_axis = target - origin
        x_source = f"origin_to_marker_{x_axis_id}"
    else:
        camera_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = camera_x - float(np.dot(camera_x, normal)) * normal
        if float(np.linalg.norm(x_axis)) < 1e-8:
            camera_y = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = camera_y - float(np.dot(camera_y, normal)) * normal
            x_source = "camera_positive_y_projected_to_plane"
        else:
            x_source = "camera_positive_x_projected_to_plane"

    if float(np.linalg.norm(x_axis)) < 1e-8:
        raise RuntimeError("평면 X축을 안정적으로 만들 수 없습니다.")
    x_axis /= np.linalg.norm(x_axis)
    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    transform_camera_plane = np.eye(4, dtype=np.float64)
    transform_camera_plane[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform_camera_plane[:3, 3] = origin
    definition = f"origin={origin_source}; x_axis={x_source}; +z=toward_camera"
    return transform_camera_plane, definition


def _minimum_convex_polygon_width_m(polygon_xy: np.ndarray) -> float:
    """Return minimum caliper width and reject a malformed/non-convex polygon."""
    signed_area = _polygon_signed_area(polygon_xy)
    if not math.isfinite(signed_area) or signed_area <= 1e-12:
        raise RuntimeError(
            "workspace polygon의 면적이 0이거나 꼭짓점 순서가 잘못됐습니다."
        )

    widths_m: list[float] = []
    for index in range(len(polygon_xy)):
        first = polygon_xy[index]
        edge = polygon_xy[(index + 1) % len(polygon_xy)] - first
        length_m = float(np.linalg.norm(edge))
        if length_m <= 1e-9:
            raise RuntimeError("workspace polygon에 중복 꼭짓점이 있습니다.")
        relative = polygon_xy - first
        signed_distances_m = (
            edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        ) / length_m
        if float(np.min(signed_distances_m)) < -1e-9:
            raise RuntimeError(
                "workspace polygon이 convex하지 않거나 서로 교차합니다."
            )
        widths_m.append(float(np.max(signed_distances_m)))
    return min(widths_m)


def _require_usable_workspace_width(
    polygon_xy: np.ndarray, minimum_width_m: float
) -> float:
    minimum_polygon_width_m = _minimum_convex_polygon_width_m(polygon_xy)
    if minimum_polygon_width_m < minimum_width_m:
        raise RuntimeError(
            "유효 마커들이 거의 일직선이라 workspace polygon을 만들 수 "
            "없습니다: "
            f"최소 폭 {minimum_polygon_width_m * 1000.0:.3f} mm < "
            f"요구값 {minimum_width_m * 1000.0:.3f} mm. "
            "누락된 비공선 마커의 depth를 복구한 뒤 다시 측정하세요."
        )
    return minimum_polygon_width_m


def _convex_hull_xy(points_xy: np.ndarray, minimum_width_m: float) -> np.ndarray:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or len(points_xy) < 3:
        raise ValueError("workspace hull에는 3개 이상의 2-D 점이 필요합니다.")
    hull = cv2.convexHull(points_xy.astype(np.float32), clockwise=False, returnPoints=True)
    hull_xy = hull[:, 0, :].astype(np.float64)
    if len(hull_xy) < 3 or abs(_polygon_signed_area(hull_xy)) < 1e-12:
        raise RuntimeError(
            "유효 마커들이 거의 일직선이라 workspace polygon을 만들 수 없습니다."
        )
    if _polygon_signed_area(hull_xy) < 0.0:
        hull_xy = hull_xy[::-1].copy()
    _require_usable_workspace_width(hull_xy, minimum_width_m)
    return hull_xy


def _polygon_signed_area(polygon_xy: np.ndarray) -> float:
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _line_intersection(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray) -> np.ndarray:
    denominator = _cross2(d, e)
    if abs(denominator) < 1e-12:
        raise RuntimeError("workspace inset 중 평행한 경계선을 처리할 수 없습니다.")
    t = _cross2(q - p, e) / denominator
    return p + t * d


def _inset_convex_polygon(polygon_xy: np.ndarray, margin_m: float) -> np.ndarray:
    """Inset a CCW convex polygon by a constant Euclidean margin."""
    if margin_m <= 0.0:
        return polygon_xy.copy()
    polygon = polygon_xy.copy()
    if _polygon_signed_area(polygon) < 0.0:
        polygon = polygon[::-1].copy()

    shifted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    count = len(polygon)
    for index in range(count):
        first = polygon[index]
        second = polygon[(index + 1) % count]
        direction = second - first
        length = float(np.linalg.norm(direction))
        if length < 1e-10:
            raise RuntimeError("workspace polygon에 중복 꼭짓점이 있습니다.")
        # CCW polygon interior is the left side of each directed edge.
        inward_normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
        shifted_lines.append((first + margin_m * inward_normal, direction))

    inset_vertices: list[np.ndarray] = []
    for index in range(count):
        prev_point, prev_dir = shifted_lines[(index - 1) % count]
        curr_point, curr_dir = shifted_lines[index]
        inset_vertices.append(
            _line_intersection(prev_point, prev_dir, curr_point, curr_dir)
        )
    inset = np.asarray(inset_vertices, dtype=np.float64)
    area = _polygon_signed_area(inset)
    if not np.isfinite(inset).all() or area <= 1e-10:
        raise RuntimeError(
            f"--workspace-margin-m={margin_m:.4f} m가 너무 커서 polygon이 사라집니다."
        )

    # Adjacent shifted-line intersections can form a new, flipped polygon after
    # the true inset has already become empty.  Every result vertex must satisfy
    # every inward-shifted half-plane; otherwise reject instead of expanding it.
    for vertex in inset:
        for shifted_point, direction in shifted_lines:
            length_m = float(np.linalg.norm(direction))
            signed_distance_m = _cross2(direction, vertex - shifted_point) / length_m
            if signed_distance_m < -1e-9:
                raise RuntimeError(
                    f"--workspace-margin-m={margin_m:.4f} m가 polygon 내부 반경보다 "
                    "커서 안전 경계가 사라집니다."
                )
    return inset


def _plane_xy_to_camera(transform_camera_plane: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    points_plane = np.column_stack(
        (polygon_xy, np.zeros(len(polygon_xy)), np.ones(len(polygon_xy)))
    )
    return (transform_camera_plane @ points_plane.T).T[:, :3]


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


def _marker_jitter_stats(frame_centers: dict[int, list[np.ndarray]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for marker_id, values in sorted(frame_centers.items()):
        if len(values) < 3:
            continue
        points = np.asarray(values, dtype=np.float64)
        center = np.median(points, axis=0)
        displacement = np.linalg.norm(points - center, axis=1)
        payload[str(marker_id)] = {
            "samples": int(len(points)),
            "median_center_camera_m": center.tolist(),
            "median_displacement_m": float(np.median(displacement)),
            "p95_displacement_m": float(np.percentile(displacement, 95)),
            "maximum_displacement_m": float(np.max(displacement)),
        }
    return payload


def _choose_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"scan_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"scan_{stamp}_{suffix:02d}"
        suffix += 1
    return candidate


def _save_annotated(
    path: Path,
    image: np.ndarray,
    markers: dict[int, np.ndarray],
    selected_ids: set[int],
) -> None:
    annotated = image.copy()
    filtered = {
        marker_id: corners
        for marker_id, corners in markers.items()
        if not selected_ids or marker_id in selected_ids
    }
    if filtered:
        ids = np.asarray(sorted(filtered), dtype=np.int32).reshape(-1, 1)
        corners = [filtered[int(marker_id)].astype(np.float32)[None, :, :] for marker_id in ids]
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"annotated image 저장 실패: {path}")


def _safe_pipeline_stop(pipeline: Any) -> None:
    """Stop a RealSense pipeline without masking the original camera error."""
    try:
        pipeline.stop()
    except RuntimeError as exc:
        message = str(exc).lower()
        # A USB disconnect can make librealsense mark the pipeline as already stopped.
        # In that case calling stop() from a finally-block raises a second exception.
        if "before start" in message or "device disconnected" in message:
            return
        raise


def _camera_stream_error(exc: RuntimeError) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if (
        "device disconnected" in lowered
        or "vidioc_s_fmt" in lowered
        or "input/output error" in lowered
        or "frame didn't arrive" in lowered
    ):
        return RuntimeError(
            "D435i RGB-D 스트림이 끊겼습니다. 이 오류는 ArUco/RANSAC 처리 전에 "
            "RealSense/V4L2 계층에서 발생했습니다. USB 3.x 연결, 다른 카메라 프로세스, "
            "udev/uvcvideo/librealsense 설치 상태를 먼저 확인하세요. 원본 오류: "
            + message
        )
    return exc


def scan(args: argparse.Namespace) -> int:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2가 없습니다. 프로젝트 루트에서 realsense extra를 설치하세요."
        ) from exc

    selected_ids = _parse_ids(args.marker_ids)
    detector = _make_detector(args.dictionary)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = _choose_output_dir(output_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    observations: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    marker_frame_counts: dict[int, int] = defaultdict(int)
    frame_centers: dict[int, list[np.ndarray]] = defaultdict(list)
    frames_processed = 0
    frames_with_markers = 0
    last_image: np.ndarray | None = None
    last_depth_m: np.ndarray | None = None
    last_markers: dict[int, np.ndarray] = {}
    intrinsics: Any | None = None
    camera_meta: dict[str, str] = {}

    pipeline_started = False
    align = rs.align(rs.stream.color)
    try:
        profile = pipeline.start(config)
        pipeline_started = True
        device = profile.get_device()
        camera_meta = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
        }

        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=args.timeout_ms)

        started_monotonic = time.monotonic()
        started_ns = time.time_ns()
        while time.monotonic() - started_monotonic < args.duration_s:
            frames = pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            raw_depth = np.asanyarray(depth_frame.get_data())
            depth_scale = float(depth_frame.get_units())
            depth_m = raw_depth.astype(np.float32) * depth_scale
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            markers = _detect_markers(image, detector)
            if selected_ids:
                markers = {
                    marker_id: corners
                    for marker_id, corners in markers.items()
                    if marker_id in selected_ids
                }

            frames_processed += 1
            if markers:
                frames_with_markers += 1
            last_image = image.copy()
            last_depth_m = depth_m.copy()
            last_markers = markers

            for marker_id, corners in markers.items():
                marker_frame_counts[marker_id] += 1
                samples_px = _marker_sample_pixels(
                    corners,
                    grid_size=args.grid_size,
                    inset_fraction=args.sample_inset_fraction,
                )
                frame_points: list[np.ndarray] = []
                for sample_index, pixel in enumerate(samples_px):
                    u, v = float(pixel[0]), float(pixel[1])
                    if not (
                        args.edge_margin_px <= u < args.width - args.edge_margin_px
                        and args.edge_margin_px <= v < args.height - args.edge_margin_px
                    ):
                        continue
                    depth_value_m = _median_depth_patch_m(
                        depth_m,
                        u,
                        v,
                        args.depth_radius_px,
                        args.minimum_depth_m,
                        args.maximum_depth_m,
                    )
                    if depth_value_m is None:
                        continue
                    point = np.asarray(
                        rs.rs2_deproject_pixel_to_point(
                            intrinsics, [float(u), float(v)], float(depth_value_m)
                        ),
                        dtype=np.float64,
                    )
                    if not np.isfinite(point).all():
                        continue
                    observations[(marker_id, sample_index)].append(point)
                    frame_points.append(point)

                if len(frame_points) >= max(3, args.grid_size):
                    frame_centers[marker_id].append(
                        np.median(np.asarray(frame_points, dtype=np.float64), axis=0)
                    )
    except RuntimeError as exc:
        raise _camera_stream_error(exc) from exc
    finally:
        if pipeline_started:
            _safe_pipeline_stop(pipeline)

    if last_image is None or last_depth_m is None or intrinsics is None:
        raise RuntimeError("D435i에서 정렬된 RGB-D 프레임을 얻지 못했습니다.")

    aggregated_points: list[np.ndarray] = []
    point_labels: list[str] = []
    marker_points: dict[int, list[np.ndarray]] = defaultdict(list)
    observation_counts: dict[str, int] = {}
    for (marker_id, sample_index), values in sorted(observations.items()):
        label = f"marker_{marker_id}_sample_{sample_index}"
        observation_counts[label] = len(values)
        if len(values) < args.minimum_observations:
            continue
        values_array = np.asarray(values, dtype=np.float64)
        point = np.median(values_array, axis=0)
        aggregated_points.append(point)
        point_labels.append(label)
        marker_points[marker_id].append(point)

    valid_marker_ids = sorted(
        marker_id
        for marker_id, values in marker_points.items()
        if len(values) >= args.minimum_samples_per_marker
    )
    marker_points = {marker_id: marker_points[marker_id] for marker_id in valid_marker_ids}
    if len(valid_marker_ids) < args.minimum_markers:
        seen = sorted(marker_frame_counts)
        raise RuntimeError(
            f"유효 마커가 {len(valid_marker_ids)}개뿐입니다. "
            f"최소 {args.minimum_markers}개가 필요합니다. 프레임에서 본 ID={seen}"
        )

    # Drop points belonging to markers that did not pass marker-level validation.
    valid_id_set = set(valid_marker_ids)
    filtered_points: list[np.ndarray] = []
    filtered_labels: list[str] = []
    for point, label in zip(aggregated_points, point_labels):
        marker_id = int(label.split("_")[1])
        if marker_id in valid_id_set:
            filtered_points.append(point)
            filtered_labels.append(label)
    points_m = np.asarray(filtered_points, dtype=np.float64)
    point_labels = filtered_labels

    normal, offset, inlier_mask, residual_stats = fit_plane_ransac_svd(
        points_m,
        threshold_m=args.ransac_threshold_m,
        minimum_inlier_ratio=args.minimum_inlier_ratio,
    )

    # Build marker centers from their inlier sample points only.
    inlier_by_marker: dict[int, list[np.ndarray]] = defaultdict(list)
    for point, label, keep in zip(points_m, point_labels, inlier_mask.tolist()):
        if not keep:
            continue
        marker_id = int(label.split("_")[1])
        inlier_by_marker[marker_id].append(point)

    marker_centers_m = {
        marker_id: np.median(np.asarray(values, dtype=np.float64), axis=0)
        for marker_id, values in sorted(inlier_by_marker.items())
        if len(values) >= args.minimum_inlier_samples_per_marker
    }
    if len(marker_centers_m) < args.minimum_markers:
        raise RuntimeError(
            "RANSAC 이후 평면 inlier를 충분히 가진 마커가 부족합니다: "
            f"{sorted(marker_centers_m)}"
        )

    transform_camera_plane, frame_definition = _make_plane_frame(
        normal,
        offset,
        marker_centers_m,
        args.origin_id,
        args.x_axis_id,
    )
    transform_plane_camera = np.linalg.inv(transform_camera_plane)

    marker_plane_xy = {
        marker_id: (transform_plane_camera @ np.append(center, 1.0))[:2]
        for marker_id, center in marker_centers_m.items()
    }
    workspace_boundary_xy = _convex_hull_xy(
        np.asarray(list(marker_plane_xy.values()), dtype=np.float64),
        args.minimum_workspace_width_m,
    )
    workspace_minimum_width_m = _minimum_convex_polygon_width_m(
        workspace_boundary_xy
    )
    safe_boundary_xy = _inset_convex_polygon(
        workspace_boundary_xy, args.workspace_margin_m
    )
    safe_boundary_minimum_width_m = _require_usable_workspace_width(
        safe_boundary_xy, args.minimum_workspace_width_m
    )
    workspace_boundary_camera = _plane_xy_to_camera(
        transform_camera_plane, workspace_boundary_xy
    )
    safe_boundary_camera = _plane_xy_to_camera(
        transform_camera_plane, safe_boundary_xy
    )

    jitter = _marker_jitter_stats(frame_centers)
    jitter_warnings = {
        marker_id: payload["p95_displacement_m"]
        for marker_id, payload in jitter.items()
        if float(payload["p95_displacement_m"]) > args.jitter_warning_m
    }

    inlier_labels = [
        label for label, keep in zip(point_labels, inlier_mask.tolist()) if keep
    ]
    missing_selected = sorted(selected_ids - set(marker_frame_counts)) if selected_ids else []
    plane_distance_m = abs(offset)
    camera_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    plane_tilt_from_front_deg = math.degrees(
        math.acos(float(np.clip(abs(np.dot(normal, camera_z)), 0.0, 1.0)))
    )

    result = {
        "schema_version": "1.0",
        "status": "geometric_candidate_not_robot_safety_approved",
        "created_at_ns": time.time_ns(),
        "capture_started_at_ns": started_ns,
        "camera": {
            **camera_meta,
            "reference_frame": "camera_color_optical_frame",
            "coordinate_convention": "+X right, +Y down, +Z forward, metres",
            "intrinsics": _intrinsics_payload(intrinsics),
        },
        "capture": {
            "duration_requested_s": args.duration_s,
            "frames_processed": frames_processed,
            "frames_with_at_least_one_selected_marker": frames_with_markers,
            "width_px": args.width,
            "height_px": args.height,
            "fps_requested": args.fps,
            "warmup_frames": args.warmup_frames,
            "dictionary": args.dictionary,
            "marker_filter_ids": sorted(selected_ids),
            "missing_filtered_ids": missing_selected,
            "marker_frame_counts": {
                str(key): int(value) for key, value in sorted(marker_frame_counts.items())
            },
            "grid_size": args.grid_size,
            "sample_inset_fraction": args.sample_inset_fraction,
            "depth_patch_radius_px": args.depth_radius_px,
            "minimum_observations_per_sample": args.minimum_observations,
            "observation_counts": observation_counts,
        },
        "stability": {
            "marker_center_temporal_jitter": jitter,
            "warning_threshold_m": args.jitter_warning_m,
            "markers_above_warning_threshold": jitter_warnings,
        },
        "plane_camera": {
            "normal_xyz": normal.tolist(),
            "d_m": float(offset),
            "equation": (
                f"{normal[0]:+.12f}*x {normal[1]:+.12f}*y "
                f"{normal[2]:+.12f}*z {offset:+.12f} = 0"
            ),
            "normal_direction": "toward_camera_origin",
            "distance_from_camera_origin_m": float(plane_distance_m),
            "tilt_from_camera_front_deg": float(plane_tilt_from_front_deg),
            "ransac_threshold_m": args.ransac_threshold_m,
            "point_count": int(len(points_m)),
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "inlier_labels": inlier_labels,
            "residuals": residual_stats,
        },
        "plane_frame": {
            "definition": frame_definition,
            "T_camera_plane": transform_camera_plane.tolist(),
            "T_plane_camera": transform_plane_camera.tolist(),
        },
        "markers": {
            str(marker_id): {
                "center_camera_m": marker_centers_m[marker_id].tolist(),
                "center_plane_xy_m": marker_plane_xy[marker_id].tolist(),
                "inlier_sample_count": int(len(inlier_by_marker[marker_id])),
                "frame_detection_count": int(marker_frame_counts.get(marker_id, 0)),
            }
            for marker_id in sorted(marker_centers_m)
        },
        "workspace_candidate": {
            "definition": "convex hull of valid marker centers on the fitted plane",
            "reference_frame": "aruco_plane",
            "boundary_xy_m": workspace_boundary_xy.tolist(),
            "boundary_camera_xyz_m": workspace_boundary_camera.tolist(),
            "workspace_margin_m": args.workspace_margin_m,
            "minimum_required_width_m": args.minimum_workspace_width_m,
            "boundary_minimum_width_m": workspace_minimum_width_m,
            "safe_boundary_xy_m": safe_boundary_xy.tolist(),
            "safe_boundary_camera_xyz_m": safe_boundary_camera.tolist(),
            "safe_boundary_minimum_width_m": safe_boundary_minimum_width_m,
            "axis_aligned_bounds_plane_m": {
                "minimum_xy": np.min(safe_boundary_xy, axis=0).tolist(),
                "maximum_xy": np.max(safe_boundary_xy, axis=0).tolist(),
            },
            "surface_z_plane_m": 0.0,
            "limitations": [
                "마커 중심들의 convex hull 안쪽만 기하학적 후보 영역으로 간주합니다.",
                "카메라에 보이지 않는 실제 테이블 영역까지 자동 확장하지 않습니다.",
                "로봇 reachability, self-collision, tool/gripper 크기, 장애물은 아직 반영하지 않습니다.",
                "실제 로봇 이동 전에는 T_base_camera 또는 hand-eye 변환으로 base frame에 옮겨 별도 검증해야 합니다.",
            ],
        },
    }

    result_path = output_dir / "plane_workspace_result.json"
    frame_path = output_dir / "T_camera_plane.npy"
    workspace_path = output_dir / "workspace_safe_boundary_plane_xy.npy"
    points_path = output_dir / "ransac_points_camera_xyz.npy"
    depth_path = output_dir / "last_aligned_depth_m.npz"
    annotated_path = output_dir / "aruco_annotated.jpg"

    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    np.save(frame_path, transform_camera_plane)
    np.save(workspace_path, safe_boundary_xy)
    np.save(points_path, points_m)
    np.savez_compressed(depth_path, depth_m=last_depth_m)
    _save_annotated(annotated_path, last_image, last_markers, selected_ids)

    print("\n=== D435i ArUco Plane Scan Result ===")
    print(f"output_dir: {output_dir}")
    print(f"frames: {frames_processed}, frames_with_marker: {frames_with_markers}")
    print(f"valid marker IDs: {sorted(marker_centers_m)}")
    if missing_selected:
        print(f"not seen (allowed): {missing_selected}")
    print("plane(camera):")
    print(f"  {result['plane_camera']['equation']}")
    print(f"  RANSAC inliers: {int(np.count_nonzero(inlier_mask))}/{len(points_m)}")
    print(f"  residual RMSE: {residual_stats['rmse_m'] * 1000.0:.3f} mm")
    print(f"  residual p95 : {residual_stats['p95_m'] * 1000.0:.3f} mm")
    print(f"workspace vertices: {len(safe_boundary_xy)}")
    print(f"workspace margin: {args.workspace_margin_m * 1000.0:.1f} mm")
    if jitter_warnings:
        readable = ", ".join(
            f"ID {marker_id}: p95={value * 1000.0:.1f}mm"
            for marker_id, value in jitter_warnings.items()
        )
        print(f"WARNING camera/scene jitter: {readable}")
    print(f"saved: {result_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="D435i serial number; omitted = first available RealSense")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--dictionary",
        choices=sorted(ARUCO_DICTIONARIES),
        default="DICT_4X4_50",
    )
    parser.add_argument(
        "--marker-ids",
        help="사용할 ID를 쉼표로 지정. 생략하면 화면에서 검출되는 모든 ID를 사용",
    )
    parser.add_argument("--origin-id", type=int, help="평면 좌표계 원점으로 쓸 마커 ID")
    parser.add_argument("--x-axis-id", type=int, help="원점 -> 이 ID 방향을 평면 +X로 사용")
    parser.add_argument("--grid-size", type=int, default=3, help="마커 내부 depth sample grid; 3 => 9 points")
    parser.add_argument("--sample-inset-fraction", type=float, default=0.20)
    parser.add_argument("--depth-radius-px", type=int, default=2)
    parser.add_argument("--edge-margin-px", type=int, default=2)
    parser.add_argument("--minimum-depth-m", type=float, default=0.15)
    parser.add_argument("--maximum-depth-m", type=float, default=2.0)
    parser.add_argument("--minimum-observations", type=int, default=10)
    parser.add_argument("--minimum-samples-per-marker", type=int, default=5)
    parser.add_argument("--minimum-inlier-samples-per-marker", type=int, default=4)
    parser.add_argument("--minimum-markers", type=int, default=3)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.004)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.65)
    parser.add_argument(
        "--workspace-margin-m",
        type=float,
        default=0.0,
        help="마커 hull에서 안쪽으로 줄일 거리. 예: 0.02 = 20 mm",
    )
    parser.add_argument(
        "--minimum-workspace-width-m",
        type=float,
        default=0.005,
        help=(
            "depth 잡음으로 생긴 거의 공선인 hull을 거부할 최소 caliper 폭. "
            "기본: 0.005 m"
        ),
    )
    parser.add_argument(
        "--jitter-warning-m",
        type=float,
        default=0.010,
        help="3초 동안 marker center p95 흔들림 경고 기준",
    )
    parser.add_argument(
        "--output-root",
        default="results/d435i_plane_scans",
        help="실행 위치 기준 결과 루트. 기본: ./results/d435i_plane_scans",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be > 0")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be >= 0")
    if args.grid_size < 2 or args.grid_size > 7:
        parser.error("--grid-size must be in [2, 7]")
    if not (0.0 < args.sample_inset_fraction < 0.5):
        parser.error("--sample-inset-fraction must be in (0, 0.5)")
    if args.depth_radius_px < 0:
        parser.error("--depth-radius-px must be >= 0")
    if not (0.0 < args.minimum_depth_m < args.maximum_depth_m):
        parser.error("depth range must satisfy 0 < minimum < maximum")
    if args.minimum_observations < 1:
        parser.error("--minimum-observations must be >= 1")
    if args.minimum_samples_per_marker < 3:
        parser.error("--minimum-samples-per-marker must be >= 3")
    if args.minimum_inlier_samples_per_marker < 3:
        parser.error("--minimum-inlier-samples-per-marker must be >= 3")
    if args.minimum_markers < 3:
        parser.error("--minimum-markers must be >= 3")
    if args.ransac_threshold_m <= 0.0:
        parser.error("--ransac-threshold-m must be > 0")
    if not (0.0 < args.minimum_inlier_ratio <= 1.0):
        parser.error("--minimum-inlier-ratio must be in (0,1]")
    if args.workspace_margin_m < 0.0:
        parser.error("--workspace-margin-m must be >= 0")
    if (
        not math.isfinite(args.minimum_workspace_width_m)
        or args.minimum_workspace_width_m <= 0.0
    ):
        parser.error("--minimum-workspace-width-m must be finite and > 0")
    if args.jitter_warning_m <= 0.0:
        parser.error("--jitter-warning-m must be > 0")
    if args.origin_id is not None and args.x_axis_id == args.origin_id:
        parser.error("--origin-id and --x-axis-id must be different")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    return scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
