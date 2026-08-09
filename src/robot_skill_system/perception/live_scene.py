"""Build one fail-closed, current object/grasp anchor from aligned RGB-D.

The preview image is not a robot target.  This module turns a fresh aligned
frame, a local detector, and the fixed ArUco transform into a typed scene
object.  Its object-frame origin is the predicted contact point; its axes are
the live ArUco plane axes.  Downstream motion remains object-relative.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.scene.models import ObjectInstance, Pose, Quaternion, Vector3

from ._geometry import rotation_matrix_to_quaternion_xyzw
from .interfaces import ObjectDetection2D, ObjectDetector


@dataclass(frozen=True, slots=True)
class LearnedGripPoint:
    """Locally stored object-relative grasp parameters for one semantic class."""

    class_name: str
    longitudinal: float
    lateral: float
    jaw_relative_angle_rad: float
    target_gripper_width_m: float
    source_path: Path


def load_learned_grip_point(path: Path, *, class_name: str = "hammer") -> LearnedGripPoint:
    """Load only the deterministic numeric profile needed at runtime."""

    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "local-live-grasp/1.0":
            if payload.get("class_name") != class_name:
                raise ValueError("profile class does not match requested class")
            normalized = payload["normalized_grasp_point"]
            longitudinal = float(normalized["longitudinal"])
            lateral = float(normalized["lateral"])
            jaw_relative_angle_rad = math.radians(float(payload["jaw_relative_angle_deg"]))
            target_gripper_width_m = float(payload["target_gripper_width_m"])
        else:
            model = payload["objects"][class_name]["learned_grip_model"]
            normalized = model["normalized_grasp_point"]
            width = model["grip_width_statistics_m"]["example_gripper_width_m"]
            longitudinal = float(normalized["longitudinal_median"])
            lateral = float(normalized["lateral_median"])
            jaw_relative_angle_rad = math.radians(float(model["jaw_relative_angle_deg"]))
            target_gripper_width_m = float(width)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid local grasp profile for {class_name!r}: {resolved}") from exc
    if not all(
        math.isfinite(value)
        for value in (longitudinal, lateral, jaw_relative_angle_rad, target_gripper_width_m)
    ):
        raise ValueError("local grasp profile contains non-finite values")
    if not -1.5 <= longitudinal <= 1.5 or not -1.5 <= lateral <= 1.5:
        raise ValueError("local grasp profile normalized point is outside its safe envelope")
    if not 0.0 < target_gripper_width_m <= 0.110:
        raise ValueError("local grasp profile width is outside the RG2 envelope")
    return LearnedGripPoint(
        class_name=class_name,
        longitudinal=longitudinal,
        lateral=lateral,
        jaw_relative_angle_rad=jaw_relative_angle_rad,
        target_gripper_width_m=target_gripper_width_m,
        source_path=resolved,
    )


def _matrix44(value: Any, *, label: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{label} rotation is not orthonormal")
    return matrix


def _foreground_pixels(
    frame: SynchronizedRGBDFrame, detection: ObjectDetection2D
) -> NDArray[np.float64]:
    """Select a conservative coloured/dark foreground inside a YOLO box."""

    height, width = frame.depth_image_m.shape
    box = detection.bounding_box
    x0 = max(0, min(width - 1, int(math.floor(box.x_min_px))))
    x1 = max(x0 + 1, min(width, int(math.ceil(box.x_max_px))))
    y0 = max(0, min(height - 1, int(math.floor(box.y_min_px))))
    y1 = max(y0 + 1, min(height, int(math.ceil(box.y_max_px))))
    rgb = frame.color_image_rgb[y0:y1, x0:x1].astype(np.int16, copy=False)
    channel_span = rgb.max(axis=2) - rgb.min(axis=2)
    value = rgb.max(axis=2)
    # The supplied tool model was trained on coloured/dark tools on a pale
    # table.  This only extracts a current image-space frame; depth remains the
    # metric authority and insufficient evidence rejects the target.
    mask = ((channel_span >= 28) & (value <= 245)) | (value <= 105)
    y, x = np.nonzero(mask)
    if len(x) < 80:
        raise ValueError("live object foreground has fewer than 80 pixels")
    return np.column_stack((x + x0, y + y0)).astype(np.float64)


def _object_frame(
    points_xy: NDArray[np.float64], *, class_name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Return canonical object center, long axis, and half extents in pixels."""

    if points_xy.shape[0] < 3:
        raise ValueError("object foreground is too small for a 2-D frame")
    center = np.mean(points_xy, axis=0)
    covariance = np.cov((points_xy - center).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    axis /= np.linalg.norm(axis)
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    longitudinal = (points_xy - center) @ axis
    lateral = (points_xy - center) @ perpendicular
    low_width = np.percentile(lateral[longitudinal <= np.percentile(longitudinal, 20)], [5, 95])
    high_width = np.percentile(lateral[longitudinal >= np.percentile(longitudinal, 80)], [5, 95])
    if class_name == "hammer" and float(high_width[1] - high_width[0]) < float(
        low_width[1] - low_width[0]
    ):
        axis = -axis
        perpendicular = -perpendicular
        longitudinal = -longitudinal
        lateral = -lateral
    long_low, long_high = np.percentile(longitudinal, [1, 99])
    lat_low, lat_high = np.percentile(lateral, [1, 99])
    center = (
        center
        + 0.5 * (long_low + long_high) * axis
        + 0.5 * (lat_low + lat_high) * perpendicular
    )
    return (
        center,
        axis,
        max(0.5 * float(long_high - long_low), 1.0),
        max(0.5 * float(lat_high - lat_low), 1.0),
    )


def _grip_region_depth(
    frame: SynchronizedRGBDFrame,
    foreground_pixels_xy: NDArray[np.float64],
    *,
    grasp_pixel_xy: NDArray[np.float64],
    object_axis_xy: NDArray[np.float64],
    object_half_length_px: float,
    object_half_width_px: float,
) -> tuple[float, float, int, float]:
    """Average aligned depth only over the foreground under the gripper jaws."""

    perpendicular = np.asarray(
        (-object_axis_xy[1], object_axis_xy[0]), dtype=np.float64
    )
    relative = foreground_pixels_xy - grasp_pixel_xy
    longitudinal = relative @ object_axis_xy
    lateral = relative @ perpendicular
    longitudinal_limit_px = max(5.0, min(0.20 * object_half_length_px, 24.0))
    lateral_limit_px = max(4.0, min(1.10 * object_half_width_px, 32.0))
    region = foreground_pixels_xy[
        (np.abs(longitudinal) <= longitudinal_limit_px)
        & (np.abs(lateral) <= lateral_limit_px)
    ]
    if len(region) < 30:
        raise ValueError("predicted grip region has fewer than 30 foreground pixels")
    height, width = frame.depth_image_m.shape
    x = np.clip(np.rint(region[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(region[:, 1]).astype(np.int64), 0, height - 1)
    samples = frame.depth_image_m[y, x].astype(np.float64, copy=False)
    valid = samples[np.isfinite(samples) & (samples > 0.0) & (samples <= 2.0)]
    valid_fraction = float(len(valid) / len(samples))
    if valid_fraction < 0.60:
        raise ValueError("predicted grip region has insufficient aligned depth")
    if len(valid) < 30:
        raise ValueError("predicted grip region has fewer than 30 valid depth samples")
    mean_depth_m = float(np.mean(valid))
    depth_std_m = float(np.std(valid))
    if depth_std_m > 0.020:
        raise ValueError("predicted grip region depth spread exceeds 20 mm")
    return mean_depth_m, valid_fraction, len(valid), depth_std_m


class LiveSceneBuilder:
    """Create exactly one current hammer grasp anchor or reject the capture."""

    def __init__(
        self,
        detector: ObjectDetector,
        grip_profile: LearnedGripPoint,
        *,
        minimum_confidence: float = 0.60,
        grip_profile_version_id: str | None = None,
        grip_profile_checksum_sha256: str | None = None,
    ) -> None:
        if not 0.0 < minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in (0, 1]")
        self.detector = detector
        self.grip_profile = grip_profile
        self.minimum_confidence = minimum_confidence
        self.grip_profile_version_id = grip_profile_version_id
        self.grip_profile_checksum_sha256 = grip_profile_checksum_sha256

    def build_object(
        self,
        frame: SynchronizedRGBDFrame,
        *,
        base_to_camera: Any,
        base_to_plane: Any,
    ) -> ObjectInstance:
        """Return one base-framed contact anchor from one fresh RGB-D frame."""

        base_to_camera_matrix = _matrix44(base_to_camera, label="T_base_camera")
        base_to_plane_matrix = _matrix44(base_to_plane, label="T_base_plane")
        matches = [
            item
            for item in self.detector.detect(frame)
            if item.class_name == self.grip_profile.class_name
            and item.confidence >= self.minimum_confidence
        ]
        if not matches:
            raise ValueError(f"no confident live {self.grip_profile.class_name!r} detection")
        matches.sort(key=lambda item: item.confidence, reverse=True)
        if len(matches) > 1 and matches[0].confidence - matches[1].confidence < 0.10:
            raise ValueError("live object detection is ambiguous")
        detection = matches[0]
        pixels = _foreground_pixels(frame, detection)
        center, axis, half_length, half_width = _object_frame(
            pixels, class_name=detection.class_name
        )
        perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        grasp_pixel = (
            center
            + self.grip_profile.longitudinal * half_length * axis
            + self.grip_profile.lateral * half_width * perpendicular
        )
        depth_m, depth_fraction, depth_sample_count, depth_std_m = _grip_region_depth(
            frame,
            pixels,
            grasp_pixel_xy=grasp_pixel,
            object_axis_xy=axis,
            object_half_length_px=half_length,
            object_half_width_px=half_width,
        )
        intrinsics = frame.color_intrinsics
        grasp_camera = np.asarray(
            (
                (float(grasp_pixel[0]) - intrinsics.cx_px) * depth_m / intrinsics.fx_px,
                (float(grasp_pixel[1]) - intrinsics.cy_px) * depth_m / intrinsics.fy_px,
                depth_m,
            ),
            dtype=np.float64,
        )
        observed_grasp_base = (base_to_camera_matrix @ np.r_[grasp_camera, 1.0])[:3]
        observed_grasp_plane = (
            np.linalg.inv(base_to_plane_matrix) @ np.r_[observed_grasp_base, 1.0]
        )[:3]
        grasp_plane = np.asarray(
            (observed_grasp_plane[0], observed_grasp_plane[1], 0.0),
            dtype=np.float64,
        )
        grasp_base = (base_to_plane_matrix @ np.r_[grasp_plane, 1.0])[:3]
        jaw_axis_image = np.asarray(
            (
                math.cos(self.grip_profile.jaw_relative_angle_rad) * axis[0]
                - math.sin(self.grip_profile.jaw_relative_angle_rad) * axis[1],
                math.sin(self.grip_profile.jaw_relative_angle_rad) * axis[0]
                + math.cos(self.grip_profile.jaw_relative_angle_rad) * axis[1],
                0.0,
            ),
            dtype=np.float64,
        )
        jaw_camera = np.asarray(
            (jaw_axis_image[0] / intrinsics.fx_px, jaw_axis_image[1] / intrinsics.fy_px, 0.0),
            dtype=np.float64,
        )
        jaw_camera /= np.linalg.norm(jaw_camera)
        jaw_plane = base_to_plane_matrix[:3, :3].T @ base_to_camera_matrix[:3, :3] @ jaw_camera
        jaw_plane[2] = 0.0
        if float(np.linalg.norm(jaw_plane)) < 1e-6:
            raise ValueError("live grasp jaw axis is degenerate in the ArUco plane")
        jaw_plane /= np.linalg.norm(jaw_plane)
        observed_cross_section_width_m = max(
            0.005,
            min(
                0.110,
                (2.0 * half_width) * depth_m / max(intrinsics.fx_px, intrinsics.fy_px),
            ),
        )
        confidence = min(1.0, detection.confidence * depth_fraction)
        plane_quaternion = rotation_matrix_to_quaternion_xyzw(base_to_plane_matrix[:3, :3])
        return ObjectInstance(
            instance_id=f"live_{detection.class_name}_grasp",
            class_name=detection.class_name,
            attributes={
                "anchor_semantics": "predicted_grasp_xy_with_rgbd_mean_grip_depth",
                "grasp_point_plane_m": [float(value) for value in grasp_plane],
                "observed_grasp_point_plane_m": [
                    float(value) for value in observed_grasp_plane
                ],
                "observed_height_above_plane_m": float(observed_grasp_plane[2]),
                "grip_region_mean_depth_camera_m": depth_m,
                "grip_region_depth_std_m": depth_std_m,
                "grip_region_depth_sample_count": depth_sample_count,
                "grip_region_depth_valid_fraction": depth_fraction,
                "grasp_pixel_xy": [float(value) for value in grasp_pixel],
                "jaw_axis_plane_xy": [float(jaw_plane[0]), float(jaw_plane[1])],
                "jaw_yaw_plane_rad": float(math.atan2(jaw_plane[1], jaw_plane[0])),
                "target_gripper_width_m": self.grip_profile.target_gripper_width_m,
                # Width controls jaw opening and the safety envelope, not target Z.
                "object_width_mm": self.grip_profile.target_gripper_width_m * 1000.0,
                "observed_cross_section_width_mm": observed_cross_section_width_m * 1000.0,
                "grip_profile_uri": str(self.grip_profile.source_path),
                "grip_profile_version_id": self.grip_profile_version_id,
                "grip_profile_checksum_sha256": self.grip_profile_checksum_sha256,
                "depth_valid_fraction": depth_fraction,
                "rgbd_pairing_policy": "aligned_frameset_pair",
                "raw_rgbd_timestamp_skew_ms": (
                    abs(frame.raw_color_timestamp_ns - frame.raw_depth_timestamp_ns)
                    / 1_000_000.0
                    if frame.raw_color_timestamp_ns is not None
                    and frame.raw_depth_timestamp_ns is not None
                    and frame.raw_color_timestamp_clock_domain
                    == frame.raw_depth_timestamp_clock_domain
                    else None
                ),
            },
            pose=Pose(
                frame_id="base",
                position_m=Vector3(
                    x=float(grasp_base[0]),
                    y=float(grasp_base[1]),
                    z=float(grasp_base[2]),
                ),
                orientation_xyzw=Quaternion(
                    x=plane_quaternion[0],
                    y=plane_quaternion[1],
                    z=plane_quaternion[2],
                    w=plane_quaternion[3],
                ),
                timestamp_ns=frame.timestamp_ns,
                source="live_yolo_rgbd_grasp_anchor",
                confidence=confidence,
                uncertainty={"depth_valid_fraction": 1.0 - depth_fraction},
            ),
            bounding_box_2d=detection.bounding_box,
            confidence=confidence,
            visible_fraction=depth_fraction,
            pose_source="live_yolo_rgbd_grasp_anchor",
        )


__all__ = ["LearnedGripPoint", "LiveSceneBuilder", "load_learned_grip_point"]
