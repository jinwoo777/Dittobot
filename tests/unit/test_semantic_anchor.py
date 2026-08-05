from __future__ import annotations

import numpy as np
import pytest

from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.perception.semantic_anchor import reconstruct_semantic_roi_anchor


def _frame(
    *,
    aligned: bool = True,
    depth_image_m: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
) -> SynchronizedRGBDFrame:
    if depth_image_m is None:
        depth = np.ones((20, 20), dtype=np.float32)
        depth[5:15, 5:15] = 0.75
        depth[9, 9] = 4.0
    else:
        depth = np.asarray(depth_image_m, dtype=np.float32)
    return SynchronizedRGBDFrame(
        color_image_rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth_image_m=depth,
        color_timestamp_ns=1,
        depth_timestamp_ns=1,
        color_intrinsics=intrinsics
        or CameraIntrinsics(
            width_px=20,
            height_px=20,
            fx_px=20.0,
            fy_px=20.0,
            cx_px=9.5,
            cy_px=9.5,
        ),
        frame_number=0,
        aligned_depth_to_color=aligned,
    )


def test_semantic_roi_is_locally_deprojected_from_aligned_depth() -> None:
    observation = reconstruct_semantic_roi_anchor(
        _frame(), (0.25, 0.25, 0.75, 0.75)
    )

    assert observation.median_depth_m == pytest.approx(0.75)
    assert observation.position_camera_m == pytest.approx((0.01875, 0.01875, 0.75))
    assert observation.valid_depth_count == 99
    assert observation.valid_depth_fraction == pytest.approx(0.99)
    assert observation.as_dict()["metric_geometry_source"] == (
        "local_aligned_depth_and_color_intrinsics"
    )


def test_semantic_roi_rejects_unaligned_depth_and_invalid_region() -> None:
    with pytest.raises(ValueError, match="aligned"):
        reconstruct_semantic_roi_anchor(_frame(aligned=False), (0.2, 0.2, 0.8, 0.8))
    with pytest.raises(ValueError, match="ordered"):
        reconstruct_semantic_roi_anchor(_frame(), (0.8, 0.2, 0.2, 0.8))


def test_central_minority_foreground_seeds_full_roi_depth_selection() -> None:
    depth = np.ones((20, 20), dtype=np.float32)
    depth[8:12, 8:12] = 0.5

    observation = reconstruct_semantic_roi_anchor(
        _frame(depth_image_m=depth),
        (0.25, 0.25, 0.75, 0.75),
    )

    assert observation.median_depth_m == pytest.approx(0.5)
    assert observation.position_camera_m == pytest.approx((0.0, 0.0, 0.5))
    assert observation.valid_depth_count == 16
    assert observation.valid_depth_fraction == pytest.approx(0.16)
