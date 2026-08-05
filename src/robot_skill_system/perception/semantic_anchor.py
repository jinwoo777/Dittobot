"""Recover auditable camera-frame anchor points from model-supplied image ROIs.

The model-provided rectangle is only a semantic hint.  Metric depth,
deprojection, validity filtering, and confidence are computed locally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.perception.deprojection import deproject_color_pixels


@dataclass(frozen=True, slots=True)
class SemanticAnchorObservation:
    """A locally reconstructed point inside one normalized semantic ROI."""

    position_camera_m: tuple[float, float, float]
    median_depth_m: float
    valid_depth_count: int
    sampled_pixel_count: int
    valid_depth_fraction: float
    region_normalized: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "position_camera_m": list(self.position_camera_m),
            "median_depth_m": self.median_depth_m,
            "valid_depth_count": self.valid_depth_count,
            "sampled_pixel_count": self.sampled_pixel_count,
            "valid_depth_fraction": self.valid_depth_fraction,
            "region_normalized": list(self.region_normalized),
            "metric_geometry_source": "local_aligned_depth_and_color_intrinsics",
        }


def reconstruct_semantic_roi_anchor(
    frame: SynchronizedRGBDFrame,
    region_normalized: tuple[float, float, float, float],
    *,
    maximum_depth_deviation_m: float = 0.05,
    minimum_valid_depth_count: int = 9,
    maximum_sample_count: int = 20_000,
    central_seed_fraction: float = 0.4,
) -> SemanticAnchorObservation:
    """Return a robust camera-frame point for one semantic ROI.

    Foreground depth is seeded from the ROI's central core and every matching
    full-ROI pixel is deprojected with the locally recorded colour intrinsics.
    This prevents a model from supplying metric coordinates or selecting a
    robot-base target while keeping a centered minority object from being
    replaced by the surrounding background median.
    """

    if not frame.aligned_depth_to_color:
        raise ValueError("semantic ROI reconstruction requires depth aligned to color")
    if maximum_depth_deviation_m <= 0.0:
        raise ValueError("maximum_depth_deviation_m must be positive")
    if minimum_valid_depth_count < 1 or maximum_sample_count < minimum_valid_depth_count:
        raise ValueError("semantic ROI depth sample limits are invalid")
    if not 0.1 <= central_seed_fraction <= 0.75:
        raise ValueError("central_seed_fraction must be in [0.1, 0.75]")
    x_min, y_min, x_max, y_max = (float(value) for value in region_normalized)
    if not (
        0.0 <= x_min < x_max <= 1.0
        and 0.0 <= y_min < y_max <= 1.0
    ):
        raise ValueError("semantic ROI must contain ordered normalized bounds")

    intrinsics = frame.color_intrinsics
    left = max(0, min(intrinsics.width_px - 1, int(np.floor(x_min * intrinsics.width_px))))
    right = max(left + 1, min(intrinsics.width_px, int(np.ceil(x_max * intrinsics.width_px))))
    top = max(0, min(intrinsics.height_px - 1, int(np.floor(y_min * intrinsics.height_px))))
    bottom = max(top + 1, min(intrinsics.height_px, int(np.ceil(y_max * intrinsics.height_px))))
    patch = frame.depth_image_m[top:bottom, left:right]
    valid_mask = np.isfinite(patch) & (patch > 0.0)
    sampled_pixel_count = int(patch.size)
    valid_values = np.asarray(patch[valid_mask], dtype=np.float64)
    if len(valid_values) < minimum_valid_depth_count:
        raise ValueError("semantic ROI has insufficient valid aligned depth")

    patch_height, patch_width = patch.shape
    seed_height = max(1, int(np.floor(patch_height * central_seed_fraction)))
    seed_width = max(1, int(np.floor(patch_width * central_seed_fraction)))
    seed_top = (patch_height - seed_height) // 2
    seed_left = (patch_width - seed_width) // 2
    seed_patch = patch[
        seed_top : seed_top + seed_height,
        seed_left : seed_left + seed_width,
    ]
    seed_valid = seed_patch[np.isfinite(seed_patch) & (seed_patch > 0.0)]
    minimum_seed_depth_count = min(
        minimum_valid_depth_count,
        max(1, int(np.ceil(seed_patch.size * 0.25))),
    )
    if len(seed_valid) < minimum_seed_depth_count:
        raise ValueError("semantic ROI central core has insufficient valid depth")
    median_depth_m = float(np.median(seed_valid))
    foreground_mask = valid_mask & (
        np.abs(patch.astype(np.float64) - median_depth_m) <= maximum_depth_deviation_m
    )
    rows, columns = np.nonzero(foreground_mask)
    if len(rows) < minimum_valid_depth_count:
        raise ValueError("semantic ROI has no stable local depth foreground")
    foreground_inlier_count = int(len(rows))
    if len(rows) > maximum_sample_count:
        selected = np.linspace(0, len(rows) - 1, maximum_sample_count, dtype=np.int64)
        rows = rows[selected]
        columns = columns[selected]
    pixels_x = columns.astype(np.float64) + float(left)
    pixels_y = rows.astype(np.float64) + float(top)
    depths_m = patch[rows, columns].astype(np.float64)
    points = deproject_color_pixels(
        intrinsics,
        np.column_stack((pixels_x, pixels_y)),
        depths_m,
    )
    position = (
        float(np.median(points[:, 0])),
        float(np.median(points[:, 1])),
        float(np.median(points[:, 2])),
    )
    return SemanticAnchorObservation(
        position_camera_m=position,
        median_depth_m=median_depth_m,
        valid_depth_count=foreground_inlier_count,
        sampled_pixel_count=sampled_pixel_count,
        valid_depth_fraction=float(foreground_inlier_count / sampled_pixel_count),
        region_normalized=(x_min, y_min, x_max, y_max),
    )


__all__ = ["SemanticAnchorObservation", "reconstruct_semantic_roi_anchor"]
