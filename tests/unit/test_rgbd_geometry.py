from __future__ import annotations

from dataclasses import replace

import pytest

from robot_skill_system.capture import MockCapture, MockCaptureConfig
from robot_skill_system.capture.interfaces import CameraIntrinsics
from robot_skill_system.demonstrations.rgbd_geometry import (
    PixelPoint,
    calibrate_surface_from_three_points,
    manual_two_finger_sample,
    segment_dominant_depth_plane,
    validate_surface_relative_path,
)
from robot_skill_system.perception import deprojection


def test_three_point_surface_tf_and_manual_two_finger_path_are_metric() -> None:
    capture = MockCapture(MockCaptureConfig(width_px=64, height_px=48))
    capture.start()
    frames = [next(capture.stream()) for _ in range(4)]
    camera_to_surface, diagnostics = calibrate_surface_from_three_points(
        frames[0],
        origin_px=PixelPoint(16.0, 12.0),
        positive_x_px=PixelPoint(24.0, 12.0),
        positive_y_px=PixelPoint(16.0, 20.0),
    )

    assert diagnostics["x_axis_span_m"] == pytest.approx(0.09375)
    assert diagnostics["input_axis_angle_deg"] == pytest.approx(90.0)
    assert camera_to_surface.translation_m.z == pytest.approx(0.75)

    samples = [
        manual_two_finger_sample(
            frame,
            frame_index=index,
            jaw_tip_a_px=PixelPoint(20.0 + index, 16.0),
            jaw_tip_b_px=PixelPoint(24.0 + index, 16.0),
            camera_to_surface=camera_to_surface,
        )
        for index, frame in enumerate(frames)
    ]
    quality = validate_surface_relative_path(samples)

    assert quality["sample_count"] == 4
    assert quality["path_length_m"] > 0.03
    assert samples[0].gripper_width_m == pytest.approx(0.046875)
    assert samples[0].orientation_surface_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_surface_calibration_rejects_collinear_points() -> None:
    capture = MockCapture(MockCaptureConfig(width_px=64, height_px=48))
    frame = next(capture.stream())

    with pytest.raises(ValueError, match="one line"):
        calibrate_surface_from_three_points(
            frame,
            origin_px=PixelPoint(10.0, 10.0),
            positive_x_px=PixelPoint(20.0, 10.0),
            positive_y_px=PixelPoint(30.0, 10.0),
        )


def test_dominant_depth_plane_is_fitted_from_raw_aligned_depth() -> None:
    capture = MockCapture(MockCaptureConfig(width_px=64, height_px=48))
    frame = next(capture.stream())

    camera_to_surface, diagnostics = segment_dominant_depth_plane(
        frame,
        region_normalized=(0.05, 0.30, 0.95, 0.95),
    )

    assert camera_to_surface.translation_m.z == pytest.approx(0.75)
    assert diagnostics["algorithm"] == "deterministic_numpy_ransac_svd"
    assert diagnostics["inlier_ratio"] == pytest.approx(1.0)
    assert diagnostics["rms_residual_m"] < 1e-8
    assert diagnostics["plane_normal_camera"][2] == pytest.approx(-1.0)


def test_manual_surface_geometry_fails_closed_when_distortion_cannot_be_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> object:
        raise ValueError(
            "distortion-aware metric deprojection requires optional pyrealsense2"
        )

    monkeypatch.setattr(deprojection, "_load_pyrealsense2", unavailable)
    capture = MockCapture(MockCaptureConfig(width_px=64, height_px=48))
    frame = next(capture.stream())
    distorted = replace(
        frame,
        color_intrinsics=CameraIntrinsics(
            width_px=64,
            height_px=48,
            fx_px=64.0,
            fy_px=64.0,
            cx_px=31.5,
            cy_px=23.5,
            distortion_model="brown_conrady",
            distortion_coefficients=(0.1, 0.01, 0.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="requires optional pyrealsense2"):
        calibrate_surface_from_three_points(
            distorted,
            origin_px=PixelPoint(16.0, 12.0),
            positive_x_px=PixelPoint(24.0, 12.0),
            positive_y_px=PixelPoint(16.0, 20.0),
        )
