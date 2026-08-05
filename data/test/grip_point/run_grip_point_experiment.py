#!/usr/bin/env python3
"""Offline train/validation experiment for object-relative grip points.

This script is deliberately diagnostic-only.  It never calls an LLM, writes to
the skill database, compiles a robot program, or contacts robot hardware.

Training labels are extracted from RGB-D demonstrations with MediaPipe thumb
and index fingertips.  For each object, the last recording is held out.  Its
prediction is made from frame 0 RGB and depth only; later frames are opened only
after that prediction has been serialized in memory, and are used exclusively
to construct a validation reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot_skill_system.capture.interfaces import (
    CameraIntrinsics,
    SynchronizedRGBDFrame,
)
from robot_skill_system.perception.hand_pose import MediaPipeHandPoseEstimator

SPLITS: dict[str, dict[str, tuple[str, ...]]] = {
    "hammer": {"train": ("hammer1", "hammer2"), "validation": ("hammer3",)},
    "jetty": {"train": ("jetty1", "jetty2"), "validation": ("jetty3",)},
    "knife": {"train": ("knife1",), "validation": ("knife2",)},
    "spanner": {"train": ("spanner1",), "validation": ("spanner2",)},
}

CONTACT_CLOSE_THRESHOLD_M = 0.03
CONTACT_STABLE_FRAMES = 3
MAXIMUM_TIMESTAMP_SKEW_NS = 20_000_000
DEPTH_PATCH_RADIUS_PX = 2


@dataclass(frozen=True, slots=True)
class LoadedFrame:
    index: int
    color_bgr: np.ndarray[Any, np.dtype[np.uint8]]
    depth_m: np.ndarray[Any, np.dtype[np.float32]]
    intrinsics: CameraIntrinsics
    synchronized: SynchronizedRGBDFrame


@dataclass(frozen=True, slots=True)
class ObjectFrame2D:
    center_xy: np.ndarray[Any, np.dtype[np.float64]]
    axis_xy: np.ndarray[Any, np.dtype[np.float64]]
    perpendicular_xy: np.ndarray[Any, np.dtype[np.float64]]
    half_length_px: float
    half_width_px: float
    mask: np.ndarray[Any, np.dtype[np.uint8]]
    pixel_count: int
    endpoint_width_low_px: float
    endpoint_width_high_px: float
    canonical_axis_policy: str

    def normalize_point(self, point_xy: np.ndarray[Any, Any]) -> tuple[float, float]:
        delta = np.asarray(point_xy, dtype=np.float64) - self.center_xy
        return (
            float(np.dot(delta, self.axis_xy) / self.half_length_px),
            float(np.dot(delta, self.perpendicular_xy) / self.half_width_px),
        )

    def denormalize_point(self, longitudinal: float, lateral: float) -> np.ndarray[Any, Any]:
        return (
            self.center_xy
            + longitudinal * self.half_length_px * self.axis_xy
            + lateral * self.half_width_px * self.perpendicular_xy
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_manifest(recording_root: Path) -> dict[str, Any]:
    manifest_path = recording_root / "rgbd_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{recording_root.name}: manifest contains no frames")
    if payload.get("depth_aligned_to_color") is not True:
        raise ValueError(f"{recording_root.name}: depth is not aligned to color")
    if [int(item["index"]) for item in frames] != list(range(len(frames))):
        raise ValueError(f"{recording_root.name}: frame indices are not contiguous")
    return payload


def _load_frame(
    recording_root: Path,
    manifest: dict[str, Any],
    index: int,
    *,
    verify_checksum: bool,
) -> LoadedFrame:
    entry = manifest["frames"][index]
    rgb_path = recording_root / "rgb" / f"{index:06d}.jpg"
    depth_path = recording_root / "depth" / f"{index:06d}.npz"
    if verify_checksum:
        if _sha256(rgb_path) != entry["rgb_checksum_sha256"]:
            raise ValueError(f"{recording_root.name} frame {index}: RGB checksum mismatch")
        if _sha256(depth_path) != entry["depth_checksum_sha256"]:
            raise ValueError(f"{recording_root.name} frame {index}: depth checksum mismatch")
    color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise ValueError(f"{recording_root.name} frame {index}: invalid RGB image")
    with np.load(depth_path, allow_pickle=False) as archive:
        if set(archive.files) != {"depth_m"}:
            raise ValueError(f"{recording_root.name} frame {index}: invalid depth archive")
        depth_m = np.asarray(archive["depth_m"], dtype=np.float32)
    intrinsics_payload = entry["color_intrinsics"]
    intrinsics = CameraIntrinsics(
        width_px=int(intrinsics_payload["width_px"]),
        height_px=int(intrinsics_payload["height_px"]),
        fx_px=float(intrinsics_payload["fx_px"]),
        fy_px=float(intrinsics_payload["fy_px"]),
        cx_px=float(intrinsics_payload["cx_px"]),
        cy_px=float(intrinsics_payload["cy_px"]),
        distortion_model=str(intrinsics_payload.get("distortion_model") or "none"),
        distortion_coefficients=tuple(
            float(value) for value in intrinsics_payload.get("distortion_coefficients") or ()
        ),
    )
    color_rgb = np.asarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    synchronized = SynchronizedRGBDFrame(
        color_image_rgb=color_rgb,
        depth_image_m=depth_m,
        color_timestamp_ns=int(entry["color_timestamp_ns"]),
        depth_timestamp_ns=int(entry["depth_timestamp_ns"]),
        color_intrinsics=intrinsics,
        frame_number=int(entry["frame_number"]),
        reference_frame=str(entry.get("reference_frame") or "camera_color_optical_frame"),
        depth_scale_m=float(entry.get("depth_scale_m") or 1.0),
        aligned_depth_to_color=True,
        timestamp_clock_domain=str(entry.get("timestamp_clock_domain") or "source_clock"),
        raw_color_timestamp_ns=(
            int(entry["raw_color_timestamp_ns"])
            if entry.get("raw_color_timestamp_ns") is not None
            else None
        ),
        raw_depth_timestamp_ns=(
            int(entry["raw_depth_timestamp_ns"])
            if entry.get("raw_depth_timestamp_ns") is not None
            else None
        ),
        raw_color_timestamp_clock_domain=entry.get("raw_color_timestamp_clock_domain"),
        raw_depth_timestamp_clock_domain=entry.get("raw_depth_timestamp_clock_domain"),
    )
    return LoadedFrame(index, color_bgr, depth_m, intrinsics, synchronized)


def _component_groups(
    mask: np.ndarray[Any, np.dtype[np.uint8]],
    *,
    maximum_centroid_distance_px: float,
    minimum_component_area_px: int = 40,
) -> list[tuple[list[int], int]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    valid = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_component_area_px
    ]
    remaining = set(valid)
    groups: list[tuple[list[int], int]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            current_xy = centroids[current]
            linked = {
                candidate
                for candidate in remaining
                if float(np.linalg.norm(centroids[candidate] - current_xy))
                <= maximum_centroid_distance_px
            }
            remaining.difference_update(linked)
            group.update(linked)
            frontier.extend(linked)
        area = sum(int(stats[index, cv2.CC_STAT_AREA]) for index in group)
        groups.append((sorted(group), area))
    groups.sort(key=lambda item: item[1], reverse=True)
    return groups


def _object_mask(color_bgr: np.ndarray[Any, Any], object_name: str) -> np.ndarray[Any, Any]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    if object_name == "jetty":
        raw = cv2.inRange(hsv, (145, 60, 45), (179, 255, 255))
        raw = cv2.morphologyEx(
            raw,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        groups = _component_groups(raw, maximum_centroid_distance_px=165.0)
        if not groups:
            raise ValueError("jetty: no magenta object components")
        component_count, labels, _, _ = cv2.connectedComponentsWithStats(raw)
        del component_count
        best_labels = groups[0][0]
        selected = np.zeros_like(raw)
        for label in best_labels:
            selected[labels == label] = 255
        points = np.column_stack(np.nonzero(selected))[:, ::-1]
        if len(points) >= 3:
            hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
            selected = np.zeros_like(raw)
            cv2.fillConvexPoly(selected, hull, 255)
        return selected

    raw = cv2.inRange(hsv, (85, 115, 8), (112, 255, 240))
    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    groups = _component_groups(
        raw,
        maximum_centroid_distance_px=190.0,
        minimum_component_area_px=100,
    )
    if not groups:
        raise ValueError(f"{object_name}: no blue/teal object component")
    _, labels, _, _ = cv2.connectedComponentsWithStats(raw)
    selected = np.zeros_like(raw)
    for label in groups[0][0]:
        selected[labels == label] = 255
    points = np.column_stack(np.nonzero(selected))[:, ::-1]
    if len(points) >= 3:
        hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
        selected = np.zeros_like(raw)
        cv2.fillConvexPoly(selected, hull, 255)
    return selected


def _endpoint_width(
    longitudinal: np.ndarray[Any, Any],
    lateral: np.ndarray[Any, Any],
    *,
    high: bool,
) -> float:
    threshold = float(np.percentile(longitudinal, 80.0 if high else 20.0))
    selected = lateral[longitudinal >= threshold] if high else lateral[longitudinal <= threshold]
    if len(selected) < 5:
        return 0.0
    return float(np.percentile(selected, 95.0) - np.percentile(selected, 5.0))


def _fit_object_frame(color_bgr: np.ndarray[Any, Any], object_name: str) -> ObjectFrame2D:
    mask = _object_mask(color_bgr, object_name)
    y_px, x_px = np.nonzero(mask)
    if len(x_px) < 300:
        raise ValueError(f"{object_name}: object mask has only {len(x_px)} pixels")
    points = np.column_stack((x_px, y_px)).astype(np.float64)
    mean = np.mean(points, axis=0)
    covariance = np.cov((points - mean).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    axis /= np.linalg.norm(axis)
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    longitudinal = (points - mean) @ axis
    lateral = (points - mean) @ perpendicular
    low_width = _endpoint_width(longitudinal, lateral, high=False)
    high_width = _endpoint_width(longitudinal, lateral, high=True)

    if object_name in {"hammer", "spanner"}:
        policy = "positive_axis_toward_wider_functional_end"
        flip = high_width < low_width
    elif object_name == "knife":
        policy = "positive_axis_toward_narrower_blade_end"
        flip = high_width > low_width
    else:
        policy = "positive_axis_toward_positive_image_x"
        flip = axis[0] < 0.0
    if flip:
        axis = -axis
        perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        longitudinal = (points - mean) @ axis
        lateral = (points - mean) @ perpendicular
        low_width, high_width = high_width, low_width

    long_low, long_high = np.percentile(longitudinal, (1.0, 99.0))
    lat_low, lat_high = np.percentile(lateral, (1.0, 99.0))
    center = mean + 0.5 * (long_low + long_high) * axis + 0.5 * (lat_low + lat_high) * perpendicular
    return ObjectFrame2D(
        center_xy=np.asarray(center, dtype=np.float64),
        axis_xy=np.asarray(axis, dtype=np.float64),
        perpendicular_xy=np.asarray(perpendicular, dtype=np.float64),
        half_length_px=max(0.5 * float(long_high - long_low), 1.0),
        half_width_px=max(0.5 * float(lat_high - lat_low), 1.0),
        mask=mask,
        pixel_count=len(points),
        endpoint_width_low_px=low_width,
        endpoint_width_high_px=high_width,
        canonical_axis_policy=policy,
    )


def _orient_contact_object_frame(
    color_bgr: np.ndarray[Any, Any],
    object_name: str,
    frame: ObjectFrame2D,
) -> ObjectFrame2D:
    """Resolve contact-frame endpoint ambiguity using visible semantic features."""
    if object_name != "knife":
        return frame
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white_blade = (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 100) & (frame.mask > 0)
    y_px, x_px = np.nonzero(white_blade)
    if len(x_px) < 100:
        return frame
    white_center = np.asarray((float(np.mean(x_px)), float(np.mean(y_px))))
    if float(np.dot(white_center - frame.center_xy, frame.axis_xy)) >= 0.0:
        return frame
    axis = -frame.axis_xy
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    return ObjectFrame2D(
        center_xy=frame.center_xy,
        axis_xy=axis,
        perpendicular_xy=perpendicular,
        half_length_px=frame.half_length_px,
        half_width_px=frame.half_width_px,
        mask=frame.mask,
        pixel_count=frame.pixel_count,
        endpoint_width_low_px=frame.endpoint_width_high_px,
        endpoint_width_high_px=frame.endpoint_width_low_px,
        canonical_axis_policy="positive_axis_toward_visible_white_blade_region",
    )


def _undirected_relative_angle(
    axis_xy: np.ndarray[Any, Any], jaw_xy: np.ndarray[Any, Any]
) -> float:
    jaw = np.asarray(jaw_xy, dtype=np.float64)
    jaw /= np.linalg.norm(jaw)
    dot = float(np.dot(axis_xy, jaw))
    cross = float(axis_xy[0] * jaw[1] - axis_xy[1] * jaw[0])
    angle = math.atan2(cross, dot)
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def _rotate(vector_xy: np.ndarray[Any, Any], angle_rad: float) -> np.ndarray[Any, Any]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(
        (
            cosine * vector_xy[0] - sine * vector_xy[1],
            sine * vector_xy[0] + cosine * vector_xy[1],
        ),
        dtype=np.float64,
    )


def _depth_at_pixel(depth_m: np.ndarray[Any, Any], point_xy: np.ndarray[Any, Any]) -> float | None:
    x_px, y_px = (int(round(float(value))) for value in point_xy)
    height, width = depth_m.shape
    if not (0 <= x_px < width and 0 <= y_px < height):
        return None
    x0 = max(0, x_px - DEPTH_PATCH_RADIUS_PX)
    x1 = min(width, x_px + DEPTH_PATCH_RADIUS_PX + 1)
    y0 = max(0, y_px - DEPTH_PATCH_RADIUS_PX)
    y1 = min(height, y_px + DEPTH_PATCH_RADIUS_PX + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0) & (patch <= 5.0)]
    return float(np.median(valid)) if len(valid) else None


def _deproject(
    point_xy: np.ndarray[Any, Any],
    depth_m: float | None,
    intrinsics: CameraIntrinsics,
) -> list[float] | None:
    if depth_m is None:
        return None
    return [
        (float(point_xy[0]) - intrinsics.cx_px) * depth_m / intrinsics.fx_px,
        (float(point_xy[1]) - intrinsics.cy_px) * depth_m / intrinsics.fy_px,
        depth_m,
    ]


def _jaw_axis_camera(jaw_xy: np.ndarray[Any, Any], intrinsics: CameraIntrinsics) -> list[float]:
    direction = np.asarray(
        (float(jaw_xy[0]) / intrinsics.fx_px, float(jaw_xy[1]) / intrinsics.fy_px, 0.0),
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    return direction.tolist()


def _track_recording(
    recording_root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    tracker = MediaPipeHandPoseEstimator(
        minimum_detection_confidence=0.6,
        minimum_tracking_confidence=0.6,
        close_threshold_m=CONTACT_CLOSE_THRESHOLD_M,
        stable_frames=CONTACT_STABLE_FRAMES,
        maximum_timestamp_skew_ns=MAXIMUM_TIMESTAMP_SKEW_NS,
    )
    try:
        for index in range(len(manifest["frames"])):
            frame = _load_frame(recording_root, manifest, index, verify_checksum=True)
            tracked = tracker.track(frame.synchronized)
            observation = tracked.observation.model_dump(mode="json")
            observation["frame_index"] = index
            observations.append(observation)
            if tracked.transition is not None:
                transition = tracked.transition.model_dump(mode="json")
                transition["frame_index"] = index
                transitions.append(transition)
    finally:
        tracker.close()
    return observations, transitions


def _select_contact_observation(
    observations: list[dict[str, Any]], transitions: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    valid = [item for item in observations if item["status"] == "valid"]
    if not valid:
        return None, "unavailable_no_valid_hand_landmarks"
    first_closed = next((item for item in transitions if item["state"] == "closed"), None)
    if first_closed is not None:
        frame_index = int(first_closed["frame_index"])
        return (
            next(item for item in valid if int(item["frame_index"]) == frame_index),
            "first_three_frame_stable_closed",
        )
    return min(valid, key=lambda item: float(item["distance_m"])), "minimum_distance_fallback"


def _extract_contact_case(
    demonstrations_root: Path,
    object_name: str,
    recording_name: str,
) -> tuple[dict[str, Any] | None, np.ndarray[Any, Any] | None]:
    recording_root = demonstrations_root / recording_name
    manifest = _load_manifest(recording_root)
    observations, transitions = _track_recording(recording_root, manifest)
    selected, selection_rule = _select_contact_observation(observations, transitions)
    if selected is None:
        return (
            {
                "recording": recording_name,
                "usable": False,
                "reason": selection_rule,
                "valid_hand_frames": 0,
                "frame_count": len(observations),
            },
            None,
        )
    frame_index = int(selected["frame_index"])
    frame = _load_frame(recording_root, manifest, frame_index, verify_checksum=False)
    object_frame = _fit_object_frame(frame.color_bgr, object_name)
    object_frame = _orient_contact_object_frame(
        frame.color_bgr,
        object_name,
        object_frame,
    )
    thumb = np.asarray(selected["thumb_pixel_xy"], dtype=np.float64)
    index = np.asarray(selected["index_pixel_xy"], dtype=np.float64)
    midpoint = 0.5 * (thumb + index)
    jaw = index - thumb
    jaw_norm = float(np.linalg.norm(jaw))
    if jaw_norm <= 1.0e-9:
        raise ValueError(f"{recording_name}: fingertip jaw axis is degenerate")
    jaw /= jaw_norm
    longitudinal, lateral = object_frame.normalize_point(midpoint)
    relative_angle = _undirected_relative_angle(object_frame.axis_xy, jaw)
    valid_count = sum(item["status"] == "valid" for item in observations)
    case = {
        "recording": recording_name,
        "usable": True,
        "contact_frame_index": frame_index,
        "contact_selection_rule": selection_rule,
        "operator_confirmed": False,
        "frame_count": len(observations),
        "valid_hand_frames": valid_count,
        "coverage_ratio": valid_count / len(observations),
        "thumb_pixel_xy": thumb.tolist(),
        "index_pixel_xy": index.tolist(),
        "grasp_midpoint_pixel_xy": midpoint.tolist(),
        "observed_fingertip_distance_m": float(selected["distance_m"]),
        "normalized_grasp_point": {
            "longitudinal": longitudinal,
            "lateral": lateral,
        },
        "jaw_relative_angle_deg": math.degrees(relative_angle),
        "object_frame": _object_frame_payload(object_frame),
        "warnings": (
            ["contact selected by minimum distance because no stable closed state existed"]
            if selection_rule == "minimum_distance_fallback"
            else []
        ),
    }
    return case, _draw_contact_overlay(frame.color_bgr, object_frame, case)


def _object_frame_payload(frame: ObjectFrame2D) -> dict[str, Any]:
    return {
        "center_pixel_xy": frame.center_xy.tolist(),
        "axis_pixel_xy": frame.axis_xy.tolist(),
        "perpendicular_pixel_xy": frame.perpendicular_xy.tolist(),
        "half_length_px": frame.half_length_px,
        "half_width_px": frame.half_width_px,
        "mask_pixel_count": frame.pixel_count,
        "endpoint_width_low_px": frame.endpoint_width_low_px,
        "endpoint_width_high_px": frame.endpoint_width_high_px,
        "canonical_axis_policy": frame.canonical_axis_policy,
    }


def _aggregate_model(object_name: str, train_cases: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [case for case in train_cases if case.get("usable")]
    if not usable:
        raise ValueError(f"{object_name}: no usable training grasp case")
    longitudinal = [float(case["normalized_grasp_point"]["longitudinal"]) for case in usable]
    lateral = [float(case["normalized_grasp_point"]["lateral"]) for case in usable]
    angles = np.radians([float(case["jaw_relative_angle_deg"]) for case in usable])
    width_samples = [
        {
            "recording": str(case["recording"]),
            "distance_m": float(case["observed_fingertip_distance_m"]),
            "contact_selection_rule": str(case["contact_selection_rule"]),
            "operator_confirmed": bool(case["operator_confirmed"]),
        }
        for case in usable
    ]
    widths_m = np.asarray(
        [float(sample["distance_m"]) for sample in width_samples], dtype=np.float64
    )
    doubled_mean = math.atan2(
        float(np.mean(np.sin(2.0 * angles))), float(np.mean(np.cos(2.0 * angles)))
    )
    relative_angle = 0.5 * doubled_mean
    return {
        "object": object_name,
        "source_recordings": [str(case["recording"]) for case in usable],
        "usable_case_count": len(usable),
        "normalized_grasp_point": {
            "longitudinal_median": float(np.median(longitudinal)),
            "lateral_median": float(np.median(lateral)),
            "longitudinal_range": [min(longitudinal), max(longitudinal)],
            "lateral_range": [min(lateral), max(lateral)],
        },
        "jaw_relative_angle_deg": math.degrees(relative_angle),
        "grip_width_statistics_m": {
            "estimator": "thumb_index_3d_euclidean_distance_at_selected_contact",
            "samples": width_samples,
            "sample_count": len(width_samples),
            "mean": float(np.mean(widths_m)),
            "median": float(np.median(widths_m)),
            "population_stddev": float(np.std(widths_m)),
            "minimum": float(np.min(widths_m)),
            "maximum": float(np.max(widths_m)),
            "example_gripper_width_m": float(np.mean(widths_m)),
            "usage": "advisory example width; requires RG2 fingertip/contact calibration",
        },
        "aggregation": "coordinate-wise median and doubled-angle circular mean",
        "warnings": ["fewer than three usable independent training demonstrations"]
        if len(usable) < 3
        else [],
    }


def _predict_from_first_frame(
    demonstrations_root: Path,
    object_name: str,
    recording_name: str,
    model: dict[str, Any],
) -> tuple[dict[str, Any], LoadedFrame, ObjectFrame2D]:
    recording_root = demonstrations_root / recording_name
    manifest = _load_manifest(recording_root)
    first = _load_frame(recording_root, manifest, 0, verify_checksum=True)
    object_frame = _fit_object_frame(first.color_bgr, object_name)
    normalized = model["normalized_grasp_point"]
    longitudinal = float(normalized["longitudinal_median"])
    lateral = float(normalized["lateral_median"])
    predicted_xy = object_frame.denormalize_point(longitudinal, lateral)
    relative_angle = math.radians(float(model["jaw_relative_angle_deg"]))
    predicted_jaw = _rotate(object_frame.axis_xy, relative_angle)
    predicted_depth_m = _depth_at_pixel(first.depth_m, predicted_xy)
    prediction = {
        "recording": recording_name,
        "prediction_locked_before_validation_labels": True,
        "input_policy": "rgb/000000.jpg plus depth/000000.npz only",
        "input_frame_indices": [0],
        "future_frames_used_for_prediction": False,
        "predicted_grasp_pixel_xy": predicted_xy.tolist(),
        "predicted_depth_m": predicted_depth_m,
        "predicted_grasp_point_camera_m": _deproject(
            predicted_xy, predicted_depth_m, first.intrinsics
        ),
        "predicted_jaw_axis_pixel_xy": predicted_jaw.tolist(),
        "predicted_jaw_axis_camera": _jaw_axis_camera(predicted_jaw, first.intrinsics),
        "normalized_grasp_point_from_train": {
            "longitudinal": longitudinal,
            "lateral": lateral,
        },
        "object_frame_from_first_image": _object_frame_payload(object_frame),
        "frame_zero_rgb_checksum_sha256": manifest["frames"][0]["rgb_checksum_sha256"],
        "frame_zero_depth_checksum_sha256": manifest["frames"][0]["depth_checksum_sha256"],
    }
    return prediction, first, object_frame


def _angle_error_deg(first: np.ndarray[Any, Any], second: np.ndarray[Any, Any]) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _validate_prediction(
    prediction: dict[str, Any],
    first: LoadedFrame,
    first_object_frame: ObjectFrame2D,
    validation_case: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    if not validation_case.get("usable"):
        result = {
            "status": "unscored",
            "reason": validation_case.get("reason") or "validation contact unavailable",
        }
        overlay = _draw_prediction_overlay(
            first.color_bgr, first_object_frame, prediction, None, result
        )
        return result, overlay
    actual_normalized = validation_case["normalized_grasp_point"]
    actual_xy_on_first = first_object_frame.denormalize_point(
        float(actual_normalized["longitudinal"]),
        float(actual_normalized["lateral"]),
    )
    actual_relative_angle = math.radians(float(validation_case["jaw_relative_angle_deg"]))
    actual_jaw_on_first = _rotate(first_object_frame.axis_xy, actual_relative_angle)
    actual_depth_m = _depth_at_pixel(first.depth_m, actual_xy_on_first)
    actual_camera = _deproject(actual_xy_on_first, actual_depth_m, first.intrinsics)
    predicted_camera = prediction["predicted_grasp_point_camera_m"]
    pixel_error = float(
        np.linalg.norm(
            np.asarray(prediction["predicted_grasp_pixel_xy"], dtype=np.float64)
            - actual_xy_on_first
        )
    )
    metric_error_m = (
        float(np.linalg.norm(np.asarray(predicted_camera) - np.asarray(actual_camera)))
        if predicted_camera is not None and actual_camera is not None
        else None
    )
    predicted_jaw = np.asarray(prediction["predicted_jaw_axis_pixel_xy"], dtype=np.float64)
    result = {
        "status": "scored",
        "reference_policy": (
            "later validation frames used only after prediction; contact is normalized "
            "in its own frame and projected onto validation frame 0"
        ),
        "validation_contact_frame_index": validation_case["contact_frame_index"],
        "validation_contact_selection_rule": validation_case["contact_selection_rule"],
        "reference_grasp_pixel_xy_on_first_frame": actual_xy_on_first.tolist(),
        "reference_grasp_depth_m_on_first_frame": actual_depth_m,
        "reference_grasp_point_camera_m_on_first_frame": actual_camera,
        "reference_jaw_axis_pixel_xy_on_first_frame": actual_jaw_on_first.tolist(),
        "grasp_point_pixel_error": pixel_error,
        "grasp_point_metric_error_m": metric_error_m,
        "jaw_axis_undirected_error_deg": _angle_error_deg(predicted_jaw, actual_jaw_on_first),
        "operator_confirmed": False,
    }
    overlay = _draw_prediction_overlay(
        first.color_bgr,
        first_object_frame,
        prediction,
        {"point": actual_xy_on_first, "jaw": actual_jaw_on_first},
        result,
    )
    return result, overlay


def _draw_mask(image: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> None:
    tint = np.zeros_like(image)
    tint[:, :] = (190, 130, 30)
    selected = mask > 0
    image[selected] = cv2.addWeighted(image[selected], 0.55, tint[selected], 0.45, 0.0)


def _draw_axis(
    image: np.ndarray[Any, Any],
    center_xy: np.ndarray[Any, Any],
    axis_xy: np.ndarray[Any, Any],
    half_length_px: float,
) -> None:
    start = np.rint(center_xy - half_length_px * axis_xy).astype(int)
    end = np.rint(center_xy + half_length_px * axis_xy).astype(int)
    cv2.arrowedLine(image, tuple(start), tuple(end), (255, 120, 20), 2, cv2.LINE_AA, tipLength=0.08)


def _draw_grasp(
    image: np.ndarray[Any, Any],
    point_xy: np.ndarray[Any, Any],
    jaw_xy: np.ndarray[Any, Any],
    color: tuple[int, int, int],
    label: str,
) -> None:
    point = np.rint(point_xy).astype(int)
    jaw = np.asarray(jaw_xy, dtype=np.float64)
    jaw /= np.linalg.norm(jaw)
    start = np.rint(point_xy - 35.0 * jaw).astype(int)
    end = np.rint(point_xy + 35.0 * jaw).astype(int)
    cv2.line(image, tuple(start), tuple(end), color, 4, cv2.LINE_AA)
    cv2.circle(image, tuple(point), 9, color, 3, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (int(point[0]) + 12, int(point[1]) - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def _banner(image: np.ndarray[Any, Any], lines: list[str]) -> np.ndarray[Any, Any]:
    banner_height = 30 + 23 * len(lines)
    canvas = np.zeros((image.shape[0] + banner_height, image.shape[1], 3), dtype=np.uint8)
    canvas[banner_height:] = image
    for index, line in enumerate(lines):
        font_scale = 0.52
        text_width = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )[0][0]
        if text_width > image.shape[1] - 20:
            font_scale *= (image.shape[1] - 20) / text_width
        cv2.putText(
            canvas,
            line,
            (10, 24 + 23 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_contact_overlay(
    color_bgr: np.ndarray[Any, Any],
    object_frame: ObjectFrame2D,
    case: dict[str, Any],
) -> np.ndarray[Any, Any]:
    image = color_bgr.copy()
    _draw_mask(image, object_frame.mask)
    _draw_axis(image, object_frame.center_xy, object_frame.axis_xy, object_frame.half_length_px)
    thumb = np.asarray(case["thumb_pixel_xy"], dtype=np.float64)
    index = np.asarray(case["index_pixel_xy"], dtype=np.float64)
    midpoint = np.asarray(case["grasp_midpoint_pixel_xy"], dtype=np.float64)
    _draw_grasp(image, midpoint, index - thumb, (50, 230, 80), "TRAIN CONTACT")
    normalized = case["normalized_grasp_point"]
    return _banner(
        image,
        [
            (
                f"{case['recording']} frame {case['contact_frame_index']} | "
                f"{case['contact_selection_rule']}"
            ),
            (
                f"normalized grip=({normalized['longitudinal']:+.3f}, "
                f"{normalized['lateral']:+.3f}) | jaw={case['jaw_relative_angle_deg']:+.1f} deg"
            ),
        ],
    )


def _draw_prediction_overlay(
    color_bgr: np.ndarray[Any, Any],
    object_frame: ObjectFrame2D,
    prediction: dict[str, Any],
    reference: dict[str, np.ndarray[Any, Any]] | None,
    metrics: dict[str, Any],
) -> np.ndarray[Any, Any]:
    image = color_bgr.copy()
    _draw_mask(image, object_frame.mask)
    _draw_axis(image, object_frame.center_xy, object_frame.axis_xy, object_frame.half_length_px)
    predicted_point = np.asarray(prediction["predicted_grasp_pixel_xy"], dtype=np.float64)
    predicted_jaw = np.asarray(prediction["predicted_jaw_axis_pixel_xy"], dtype=np.float64)
    _draw_grasp(image, predicted_point, predicted_jaw, (0, 225, 255), "PREDICTED")
    lines = [
        f"{prediction['recording']} frame 0 only | yellow=PREDICTED, red=HELD-OUT REFERENCE",
        "prediction input: rgb/000000.jpg + depth/000000.npz",
    ]
    if reference is not None:
        _draw_grasp(image, reference["point"], reference["jaw"], (30, 40, 245), "REFERENCE")
        cv2.line(
            image,
            tuple(np.rint(predicted_point).astype(int)),
            tuple(np.rint(reference["point"]).astype(int)),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        metric_error = metrics.get("grasp_point_metric_error_m")
        metric_text = (
            f"{float(metric_error) * 1000.0:.1f} mm" if metric_error is not None else "n/a"
        )
        lines.append(
            f"error: {metrics['grasp_point_pixel_error']:.1f} px / {metric_text}, "
            f"jaw {metrics['jaw_axis_undirected_error_deg']:.1f} deg"
        )
    else:
        lines.append(f"validation reference unavailable: {metrics['reason']}")
    return _banner(image, lines)


def _stack_images(images: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
    if not images:
        raise ValueError("cannot stack an empty image list")
    width = max(image.shape[1] for image in images)
    normalized: list[np.ndarray[Any, Any]] = []
    for image in images:
        if image.shape[1] == width:
            normalized.append(image)
            continue
        scale = width / image.shape[1]
        normalized.append(
            cv2.resize(
                image, (width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA
            )
        )
    return np.vstack(normalized)


def _write_image(path: Path, image: np.ndarray[Any, Any]) -> None:
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"could not write image: {path}")


def run(demonstrations_root: Path, output_root: Path) -> dict[str, Any]:
    demonstrations_root = demonstrations_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    objects: dict[str, Any] = {}
    overall_images: list[np.ndarray[Any, Any]] = []

    for object_name, split in SPLITS.items():
        object_output = output_root / object_name
        object_output.mkdir(parents=True, exist_ok=True)
        train_cases: list[dict[str, Any]] = []
        train_images: list[np.ndarray[Any, Any]] = []
        for recording_name in split["train"]:
            case, overlay = _extract_contact_case(demonstrations_root, object_name, recording_name)
            if case is None:
                raise AssertionError("contact extraction returned no result payload")
            train_cases.append(case)
            if overlay is not None:
                train_images.append(overlay)
                _write_image(object_output / f"train_{recording_name}_contact.jpg", overlay)

        model = _aggregate_model(object_name, train_cases)
        validation_results: list[dict[str, Any]] = []
        validation_images: list[np.ndarray[Any, Any]] = []
        for recording_name in split["validation"]:
            # Leakage boundary: prediction is completely formed from frame zero before
            # the tracker is permitted to open any later validation frame.
            prediction, first_frame, first_object_frame = _predict_from_first_frame(
                demonstrations_root, object_name, recording_name, model
            )
            locked_prediction_json = json.dumps(prediction, sort_keys=True)
            validation_case, _ = _extract_contact_case(
                demonstrations_root, object_name, recording_name
            )
            if validation_case is None:
                raise AssertionError("validation contact extraction returned no payload")
            if json.dumps(prediction, sort_keys=True) != locked_prediction_json:
                raise AssertionError("validation prediction mutated after opening labels")
            metrics, overlay = _validate_prediction(
                prediction,
                first_frame,
                first_object_frame,
                validation_case,
            )
            validation_payload = {
                "prediction": prediction,
                "held_out_reference": validation_case,
                "metrics": metrics,
            }
            validation_results.append(validation_payload)
            validation_images.append(overlay)
            _write_image(object_output / f"validation_{recording_name}_frame0.jpg", overlay)

        if train_images:
            _write_image(object_output / "train_examples.jpg", _stack_images(train_images))
        if validation_images:
            _write_image(
                object_output / "validation_first_frame.jpg",
                _stack_images(validation_images),
            )
        summary_images = train_images + validation_images
        if summary_images:
            summary = _stack_images(summary_images)
            _write_image(object_output / "summary.jpg", summary)
            resized = cv2.resize(
                summary, (640, int(round(summary.shape[0] * 640 / summary.shape[1])))
            )
            overall_images.append(resized)

        usable_count = sum(bool(case.get("usable")) for case in train_cases)
        object_warnings = list(model["warnings"])
        if usable_count != len(train_cases):
            object_warnings.append("one or more assigned training recordings were unusable")
        if any(
            result["held_out_reference"].get("contact_selection_rule")
            == "minimum_distance_fallback"
            for result in validation_results
        ):
            object_warnings.append(
                "validation contact uses minimum-distance fallback and needs operator confirmation"
            )
        objects[object_name] = {
            "split": {
                "train": list(split["train"]),
                "validation": list(split["validation"]),
                "split_unit": "independent recording",
            },
            "train_cases": train_cases,
            "learned_grip_model": model,
            "validation": validation_results,
            "warnings": object_warnings,
        }

    if overall_images:
        _write_image(output_root / "all_objects_summary.jpg", _stack_images(overall_images))

    scored = [
        item["metrics"]
        for object_payload in objects.values()
        for item in object_payload["validation"]
        if item["metrics"]["status"] == "scored"
    ]
    result = {
        "schema_version": "1.0",
        "experiment": "object_relative_grip_point_train_validation",
        "execution_mode": "offline_diagnostic_only",
        "llm_called": False,
        "robot_or_gripper_called": False,
        "database_modified": False,
        "validation_input_policy": (
            "prediction uses validation RGB frame 0 and aligned depth frame 0 only; "
            "later frames are held-out labels only"
        ),
        "grasp_direction_definition": "undirected thumb-to-index jaw closing axis",
        "approach_direction_estimated": False,
        "objects": objects,
        "aggregate": {
            "scored_validation_recordings": len(scored),
            "mean_pixel_error": (
                float(np.mean([item["grasp_point_pixel_error"] for item in scored]))
                if scored
                else None
            ),
            "mean_metric_error_m": (
                float(
                    np.mean(
                        [
                            item["grasp_point_metric_error_m"]
                            for item in scored
                            if item["grasp_point_metric_error_m"] is not None
                        ]
                    )
                )
                if any(item["grasp_point_metric_error_m"] is not None for item in scored)
                else None
            ),
            "mean_jaw_axis_error_deg": (
                float(np.mean([item["jaw_axis_undirected_error_deg"] for item in scored]))
                if scored
                else None
            ),
        },
        "limitations": [
            "contact labels are inferred from fingertips and are not operator-confirmed",
            "no post-lift object-following signal is available to prove grasp success",
            "object masks use dataset-specific color heuristics rather than an LLM detector",
            "camera-relative 3D points are diagnostics and are not executable skill targets",
            "fewer than three usable training demonstrations exist for every object",
        ],
        "artifacts": {
            "report": "report.md",
            "result": "result.json",
            "all_objects_summary": "all_objects_summary.jpg",
            "per_object": "<object>/train_examples.jpg, validation_first_frame.jpg, summary.jpg",
        },
    }
    _write_json(output_root / "result.json", result)
    _write_report(output_root / "report.md", result)
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    rows: list[str] = []
    detail_sections: list[str] = []
    for object_name, payload in result["objects"].items():
        validation = payload["validation"][0]
        metrics = validation["metrics"]
        train_names = ", ".join(payload["split"]["train"])
        validation_names = ", ".join(payload["split"]["validation"])
        usable = payload["learned_grip_model"]["usable_case_count"]
        if metrics["status"] == "scored":
            metric_error = metrics["grasp_point_metric_error_m"]
            metric_text = (
                f"{float(metric_error) * 1000.0:.1f} mm" if metric_error is not None else "n/a"
            )
            result_text = (
                f"{metrics['grasp_point_pixel_error']:.1f} px / {metric_text} / "
                f"{metrics['jaw_axis_undirected_error_deg']:.1f}°"
            )
        else:
            result_text = "unscored"
        rows.append(
            f"| {object_name} | {train_names} ({usable} usable) | "
            f"{validation_names} | {result_text} |"
        )
        model = payload["learned_grip_model"]
        point = model["normalized_grasp_point"]
        width = model["grip_width_statistics_m"]
        width_samples = ", ".join(
            f"{sample['recording']}={float(sample['distance_m']) * 1000.0:.1f} mm"
            for sample in width["samples"]
        )
        warnings = "\n".join(f"- {warning}" for warning in payload["warnings"]) or "- 없음"
        detail_sections.append(
            f"""## {object_name}

- train: `{train_names}`
- validation: `{validation_names}` — 예측 입력은 첫 RGB-D 프레임 한 쌍뿐
- 학습 파지점: longitudinal `{point["longitudinal_median"]:+.3f}`,
  lateral `{point["lateral_median"]:+.3f}`
- 학습 jaw 상대각: `{model["jaw_relative_angle_deg"]:+.1f}°`
- 손가락 3D 거리 표본: `{width_samples}`
- 예시 파지 폭(산술평균): `{float(width["example_gripper_width_m"]) * 1000.0:.1f} mm`
- 이미지: `{object_name}/summary.jpg`

경고:

{warnings}
"""
        )
    aggregate = result["aggregate"]
    mean_metric = aggregate["mean_metric_error_m"]
    mean_metric_text = f"{float(mean_metric) * 1000.0:.1f} mm" if mean_metric is not None else "n/a"
    report = f"""# Grip point train/validation 오프라인 실험

## 결론

각 물체의 독립 녹화 마지막 1개를 validation으로 분리했습니다. Validation 파지점은
`rgb/000000.jpg`와 `depth/000000.npz`만으로 먼저 확정했고, 이후 프레임의 MediaPipe 접촉점은
예측에 사용하지 않고 held-out reference로만 사용했습니다. 이 실험은 로봇 실행이나 DB 승격을
수행하지 않은 camera-relative 진단입니다.

## 결과

| 물체 | Train | Validation | point px / point metric / jaw angle |
|---|---|---|---|
{chr(10).join(rows)}

- 평균 pixel error: `{aggregate["mean_pixel_error"]:.1f} px`
- 평균 metric error: `{mean_metric_text}`
- 평균 jaw-axis error: `{aggregate["mean_jaw_axis_error_deg"]:.1f}°`
- 전체 이미지: `all_objects_summary.jpg`

노란색은 train에서 전이한 예측이며, 빨간색은 validation 접촉을 물체 상대좌표로 정규화한 뒤
첫 프레임에 다시 투영한 reference입니다. 파란 화살표는 물체의 canonical 장축입니다.

{chr(10).join(detail_sections)}
## 해석 제한

- 파지 성공을 증명하는 object-following/post-lift label이 없어 접촉 후보는 운영자 미확인입니다.
- knife는 3 cm closed 상태가 없어서 최소 fingertip 거리 프레임을 사용했습니다.
- 물체 검출은 이번 데이터에 맞춘 색상 기반 local CV이며 LLM을 호출하지 않았습니다.
- 생성된 3D 점은 camera frame 진단값이며 task-plane/hand-eye TF가 없어 실행할 수 없습니다.
- 손가락 거리 평균은 사람의 접촉 폭이며 RG2 명령 폭과 동일하다고 보장되지 않습니다.
- 모든 물체의 usable train 시연이 3개 미만이므로 일반화 성능 수치로 해석하면 안 됩니다.
"""
    path.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demonstrations",
        type=Path,
        default=Path("data/demonstrations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test/grip_point"),
    )
    args = parser.parse_args()
    try:
        result = run(args.demonstrations, args.output)
    except Exception as exc:
        print(f"grip point experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
