"""Local metric geometry for audited RGB-D teaching evidence.

The helpers in this module never create robot-base targets.  They deproject
operator-confirmed image points with aligned depth and express the resulting
TCP path relative to a surface anchor selected in the same recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.perception._geometry import rotation_matrix_to_quaternion_xyzw
from robot_skill_system.perception.deprojection import (
    deproject_color_pixel,
    deproject_color_pixels,
)
from robot_skill_system.scene.models import Quaternion, Vector3
from robot_skill_system.scene.transforms import (
    RigidTransform,
    compose_transforms,
    invert_transform,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PixelPoint:
    """One finite pixel coordinate in the recorded colour image."""

    x_px: float
    y_px: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_px) or not math.isfinite(self.y_px):
            raise ValueError("pixel coordinates must be finite")


@dataclass(frozen=True, slots=True)
class ManualTCPPathSample:
    """One two-fingertip midpoint expressed in the calibrated surface frame."""

    frame_index: int
    timestamp_ns: int
    position_surface_m: tuple[float, float, float]
    orientation_surface_xyzw: tuple[float, float, float, float]
    gripper_width_m: float
    confidence: float


def deproject_pixel(
    depth_image_m: NDArray[np.float32],
    intrinsics: CameraIntrinsics,
    point: PixelPoint,
    *,
    patch_radius_px: int = 2,
) -> FloatArray:
    """Return a camera-optical 3D point using robust local median depth."""

    if patch_radius_px < 0 or patch_radius_px > 8:
        raise ValueError("patch_radius_px must be in [0, 8]")
    if depth_image_m.shape != (intrinsics.height_px, intrinsics.width_px):
        raise ValueError("depth dimensions do not match colour intrinsics")
    pixel_x = int(round(point.x_px))
    pixel_y = int(round(point.y_px))
    if not 0 <= pixel_x < intrinsics.width_px or not 0 <= pixel_y < intrinsics.height_px:
        raise ValueError("teaching point is outside the RGB-D frame")
    lower_x = max(0, pixel_x - patch_radius_px)
    upper_x = min(intrinsics.width_px, pixel_x + patch_radius_px + 1)
    lower_y = max(0, pixel_y - patch_radius_px)
    upper_y = min(intrinsics.height_px, pixel_y + patch_radius_px + 1)
    patch = depth_image_m[lower_y:upper_y, lower_x:upper_x]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)]
    minimum_valid = max(1, (patch.size + 3) // 4)
    if valid.size < minimum_valid:
        raise ValueError("selected point has insufficient valid aligned depth")
    depth_m = float(np.median(valid))
    return deproject_color_pixel(intrinsics, point.x_px, point.y_px, depth_m)


def calibrate_surface_from_three_points(
    frame: SynchronizedRGBDFrame,
    *,
    origin_px: PixelPoint,
    positive_x_px: PixelPoint,
    positive_y_px: PixelPoint,
) -> tuple[RigidTransform, dict[str, float]]:
    """Construct ``T_camera_surface`` from operator-selected surface axes."""

    origin = deproject_pixel(frame.depth_image_m, frame.color_intrinsics, origin_px)
    positive_x = deproject_pixel(
        frame.depth_image_m, frame.color_intrinsics, positive_x_px
    )
    positive_y = deproject_pixel(
        frame.depth_image_m, frame.color_intrinsics, positive_y_px
    )
    x_vector = positive_x - origin
    y_vector = positive_y - origin
    x_span_m = float(np.linalg.norm(x_vector))
    y_span_m = float(np.linalg.norm(y_vector))
    if not 0.03 <= x_span_m <= 1.5 or not 0.03 <= y_span_m <= 1.5:
        raise ValueError("surface axis points must be 3 cm to 1.5 m from the origin")
    x_axis = x_vector / x_span_m
    y_orthogonal = y_vector - float(np.dot(y_vector, x_axis)) * x_axis
    y_orthogonal_norm = float(np.linalg.norm(y_orthogonal))
    if y_orthogonal_norm < 0.25 * y_span_m:
        raise ValueError("surface +X and +Y points are too close to one line")
    y_axis = y_orthogonal / y_orthogonal_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    input_angle_deg = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(x_vector, y_vector) / (x_span_m * y_span_m),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    return (
        RigidTransform(
            translation_m=Vector3(
                x=float(origin[0]), y=float(origin[1]), z=float(origin[2])
            ),
            rotation_xyzw=Quaternion(
                x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
            ),
        ),
        {
            "x_axis_span_m": x_span_m,
            "y_axis_span_m": y_span_m,
            "input_axis_angle_deg": input_angle_deg,
        },
    )


def segment_dominant_depth_plane(
    frame: SynchronizedRGBDFrame,
    *,
    region_normalized: tuple[float, float, float, float] | None = None,
    distance_threshold_m: float = 0.008,
    maximum_iterations: int = 256,
    minimum_inlier_ratio: float = 0.35,
) -> tuple[RigidTransform, dict[str, Any]]:
    """Estimate ``T_camera_surface`` from raw aligned depth with deterministic RANSAC.

    A model-supplied normalized work-surface region may limit the search, but the
    metric plane is always fitted locally from the recorded depth and intrinsics.
    The resulting surface +Z normal is oriented toward the camera.
    """

    if not 0.002 <= distance_threshold_m <= 0.03:
        raise ValueError("plane distance threshold must be between 2 mm and 3 cm")
    if not 32 <= maximum_iterations <= 4096:
        raise ValueError("plane RANSAC iterations must be in [32, 4096]")
    if not 0.2 <= minimum_inlier_ratio <= 0.95:
        raise ValueError("minimum plane inlier ratio must be in [0.2, 0.95]")
    intrinsics = frame.color_intrinsics
    depth = frame.depth_image_m
    if depth.shape != (intrinsics.height_px, intrinsics.width_px):
        raise ValueError("depth dimensions do not match colour intrinsics")

    if region_normalized is None:
        region_normalized = (0.05, 0.15, 0.95, 0.95)
    x_min, y_min, x_max, y_max = region_normalized
    if not (
        0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0
    ):
        raise ValueError("normalized surface region must be ordered inside [0, 1]")
    lower_x = max(0, int(math.floor(x_min * intrinsics.width_px)))
    upper_x = min(
        intrinsics.width_px, int(math.ceil(x_max * intrinsics.width_px))
    )
    lower_y = max(0, int(math.floor(y_min * intrinsics.height_px)))
    upper_y = min(
        intrinsics.height_px, int(math.ceil(y_max * intrinsics.height_px))
    )
    roi_depth = depth[lower_y:upper_y, lower_x:upper_x]
    valid_mask = (
        np.isfinite(roi_depth) & (roi_depth > 0.10) & (roi_depth < 3.0)
    )
    valid_y, valid_x = np.nonzero(valid_mask)
    if valid_x.size < 100:
        raise ValueError("surface ROI has fewer than 100 valid aligned-depth pixels")

    # Bound the RANSAC input without changing its deterministic spatial coverage.
    stride = max(1, int(math.ceil(math.sqrt(valid_x.size / 20_000))))
    valid_x = valid_x[::stride] + lower_x
    valid_y = valid_y[::stride] + lower_y
    z = depth[valid_y, valid_x].astype(np.float64)
    points = deproject_color_pixels(
        intrinsics,
        np.column_stack(
            (valid_x.astype(np.float64), valid_y.astype(np.float64))
        ),
        z,
    )
    if len(points) < 100:
        raise ValueError("surface ROI has insufficient sampled depth points")

    random = np.random.default_rng(0)
    best_mask: NDArray[np.bool_] | None = None
    best_count = 0
    for _ in range(maximum_iterations):
        sample = points[random.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-9:
            continue
        normal /= normal_norm
        offset = -float(np.dot(normal, sample[0]))
        mask = np.abs(points @ normal + offset) <= distance_threshold_m
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None:
        raise ValueError("no stable depth plane hypothesis was found")
    inlier_ratio = best_count / len(points)
    if inlier_ratio < minimum_inlier_ratio:
        raise ValueError(
            f"dominant depth plane inlier ratio {inlier_ratio:.3f} is below "
            f"{minimum_inlier_ratio:.3f}"
        )

    inliers = points[best_mask]
    centroid = np.mean(inliers, axis=0)
    _u, _singular, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, centroid)) > 0.0:
        normal = -normal
    residuals = np.abs((points - centroid) @ normal)
    refined_mask = residuals <= distance_threshold_m
    inliers = points[refined_mask]
    centroid = np.mean(inliers, axis=0)
    rms_residual_m = float(
        np.sqrt(np.mean(np.square((inliers - centroid) @ normal)))
    )
    if rms_residual_m > 0.01:
        raise ValueError("dominant depth plane residual exceeds 1 cm")

    camera_x = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    x_axis = camera_x - float(np.dot(camera_x, normal)) * normal
    if float(np.linalg.norm(x_axis)) < 0.1:
        camera_y = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        x_axis = camera_y - float(np.dot(camera_y, normal)) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, normal))
    coordinates = (inliers - centroid) @ rotation
    extent_x_m = float(np.ptp(coordinates[:, 0]))
    extent_y_m = float(np.ptp(coordinates[:, 1]))
    if min(extent_x_m, extent_y_m) < 0.05:
        raise ValueError("dominant depth plane spans less than 5 cm on one surface axis")
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    transform = RigidTransform(
        translation_m=Vector3(
            x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])
        ),
        rotation_xyzw=Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        ),
    )
    return transform, {
        "algorithm": "deterministic_numpy_ransac_svd",
        "region_normalized": list(region_normalized),
        "sample_count": len(points),
        "inlier_count": len(inliers),
        "inlier_ratio": len(inliers) / len(points),
        "rms_residual_m": rms_residual_m,
        "distance_threshold_m": distance_threshold_m,
        "plane_normal_camera": [float(value) for value in normal],
        "plane_offset_m": -float(np.dot(normal, centroid)),
        "extent_x_m": extent_x_m,
        "extent_y_m": extent_y_m,
    }


def transform_camera_pose_to_surface(
    camera_to_surface: RigidTransform,
    camera_to_tcp: RigidTransform,
) -> RigidTransform:
    """Return ``T_surface_tcp = inverse(T_camera_surface) * T_camera_tcp``."""

    return compose_transforms(invert_transform(camera_to_surface), camera_to_tcp)


def manual_two_finger_sample(
    frame: SynchronizedRGBDFrame,
    *,
    frame_index: int,
    jaw_tip_a_px: PixelPoint,
    jaw_tip_b_px: PixelPoint,
    camera_to_surface: RigidTransform,
) -> ManualTCPPathSample:
    """Build a surface-relative virtual TCP from two clicked fingertip ends."""

    tip_a_camera = deproject_pixel(
        frame.depth_image_m, frame.color_intrinsics, jaw_tip_a_px
    )
    tip_b_camera = deproject_pixel(
        frame.depth_image_m, frame.color_intrinsics, jaw_tip_b_px
    )
    midpoint_camera = (tip_a_camera + tip_b_camera) / 2.0
    gripper_width_m = float(np.linalg.norm(tip_b_camera - tip_a_camera))
    if not 0.0005 <= gripper_width_m <= 0.20:
        raise ValueError("two fingertip distance must be between 0.5 mm and 20 cm")

    surface_to_camera = invert_transform(camera_to_surface)
    tip_a_surface = compose_transforms(
        surface_to_camera,
        RigidTransform(
            translation_m=Vector3(
                x=float(tip_a_camera[0]),
                y=float(tip_a_camera[1]),
                z=float(tip_a_camera[2]),
            ),
            rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    ).translation_m
    tip_b_surface = compose_transforms(
        surface_to_camera,
        RigidTransform(
            translation_m=Vector3(
                x=float(tip_b_camera[0]),
                y=float(tip_b_camera[1]),
                z=float(tip_b_camera[2]),
            ),
            rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    ).translation_m
    jaw_axis = np.asarray(
        (
            tip_b_surface.x - tip_a_surface.x,
            tip_b_surface.y - tip_a_surface.y,
            tip_b_surface.z - tip_a_surface.z,
        ),
        dtype=np.float64,
    )
    # The taught gripper approaches along surface +Z; project the jaw direction
    # into the surface plane to remove noisy per-tip depth differences.
    jaw_axis[2] = 0.0
    jaw_axis_norm = float(np.linalg.norm(jaw_axis))
    # Very small separations are a valid closed-gripper demonstration.  In that
    # state the tip-to-tip axis is ill-conditioned, so use the task-plane +X as
    # the deterministic manual fallback.  MediaPipe replay prefers its last
    # valid palm orientation before reaching this fallback.
    x_axis = (
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        if jaw_axis_norm < 0.003
        else jaw_axis / jaw_axis_norm
    )
    z_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    midpoint_surface = compose_transforms(
        surface_to_camera,
        RigidTransform(
            translation_m=Vector3(
                x=float(midpoint_camera[0]),
                y=float(midpoint_camera[1]),
                z=float(midpoint_camera[2]),
            ),
            rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    ).translation_m
    return ManualTCPPathSample(
        frame_index=frame_index,
        timestamp_ns=frame.timestamp_ns,
        position_surface_m=midpoint_surface.as_tuple(),
        orientation_surface_xyzw=quaternion,
        gripper_width_m=gripper_width_m,
        confidence=0.9,
    )


def validate_surface_relative_path(
    samples: list[ManualTCPPathSample],
) -> dict[str, float | int]:
    """Fail closed on sparse, static, discontinuous, or cell-scale paths."""

    if not 2 <= len(samples) <= 6000:
        raise ValueError("TCP trajectory requires 2 to 6000 valid samples")
    if any(
        current.frame_index <= previous.frame_index
        for previous, current in zip(samples, samples[1:], strict=False)
    ):
        raise ValueError("TCP trajectory samples must have increasing frame indices")
    positions = np.asarray([sample.position_surface_m for sample in samples], dtype=np.float64)
    extents = np.ptp(positions, axis=0)
    if float(np.max(extents)) > 1.0:
        raise ValueError("TCP trajectory exceeds the 1 m candidate geometry envelope")
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if len(deltas) and float(np.max(deltas)) > 0.5:
        raise ValueError("TCP trajectory contains a jump larger than 0.5 m")
    path_length_m = float(np.sum(deltas))
    if path_length_m < 0.005:
        raise ValueError("TCP trajectory is too short to materialize a motion skill")
    return {
        "sample_count": len(samples),
        "path_length_m": path_length_m,
        "maximum_step_m": float(np.max(deltas)) if len(deltas) else 0.0,
        "mean_confidence": float(np.mean([sample.confidence for sample in samples])),
    }


__all__ = [
    "ManualTCPPathSample",
    "PixelPoint",
    "calibrate_surface_from_three_points",
    "deproject_pixel",
    "manual_two_finger_sample",
    "segment_dominant_depth_plane",
    "transform_camera_pose_to_surface",
    "validate_surface_relative_path",
]
