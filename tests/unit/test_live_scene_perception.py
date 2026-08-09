from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.perception.interfaces import ObjectDetection2D
from robot_skill_system.perception.live_scene import (
    LearnedGripPoint,
    LiveSceneBuilder,
    load_learned_grip_point,
)
from robot_skill_system.scene.models import BoundingBox2D


class _Detector:
    def __init__(self, detections: tuple[ObjectDetection2D, ...]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, frame: SynchronizedRGBDFrame) -> tuple[ObjectDetection2D, ...]:
        del frame
        self.calls += 1
        return self.detections


def _frame() -> SynchronizedRGBDFrame:
    color = np.full((80, 120, 3), 235, dtype=np.uint8)
    color[20:60, 20:100] = (10, 130, 190)
    depth = np.full((80, 120), 0.50, dtype=np.float32)
    intrinsics = CameraIntrinsics(
        width_px=120, height_px=80, fx_px=100.0, fy_px=100.0, cx_px=60.0, cy_px=40.0
    )
    return SynchronizedRGBDFrame(
        color_image_rgb=color,
        depth_image_m=depth,
        color_timestamp_ns=10,
        depth_timestamp_ns=10,
        color_intrinsics=intrinsics,
        frame_number=1,
        timestamp_clock_domain="unix_epoch_ns",
    )


def _detection(confidence: float = 0.9) -> ObjectDetection2D:
    return ObjectDetection2D(
        detection_id="hammer_0",
        class_name="hammer",
        bounding_box=BoundingBox2D(x_min_px=15, y_min_px=15, x_max_px=105, y_max_px=65),
        confidence=confidence,
    )


def _builder(*detections: ObjectDetection2D) -> LiveSceneBuilder:
    return LiveSceneBuilder(
        _Detector(tuple(detections)),
        LearnedGripPoint(
            class_name="hammer",
            longitudinal=0.0,
            lateral=0.0,
            jaw_relative_angle_rad=0.0,
            target_gripper_width_m=0.025,
            source_path=__file__,
        ),
    )


def test_live_scene_builder_creates_current_plane_grasp_anchor() -> None:
    builder = _builder(_detection())
    anchor = builder.build_object(
        _frame(), base_to_camera=np.eye(4), base_to_plane=np.eye(4)
    )

    assert anchor.instance_id == "live_hammer_grasp"
    assert anchor.pose.frame_id == "base"
    assert anchor.pose_source == "live_yolo_rgbd_grasp_anchor"
    assert anchor.pose.position_m.z == pytest.approx(0.0)
    assert anchor.attributes["grasp_point_plane_m"][2] == pytest.approx(0.0)
    assert anchor.attributes["observed_grasp_point_plane_m"][2] == pytest.approx(0.50)
    assert anchor.attributes["grip_region_mean_depth_camera_m"] == pytest.approx(0.50)
    assert anchor.attributes["grip_region_depth_sample_count"] >= 30
    assert anchor.attributes["grip_region_depth_std_m"] == pytest.approx(0.0)
    assert anchor.attributes["target_gripper_width_m"] == pytest.approx(0.025)
    assert anchor.attributes["object_width_mm"] > 0.0
    assert anchor.attributes["jaw_yaw_plane_rad"] == pytest.approx(0.0, abs=0.15)
    assert anchor.confidence == pytest.approx(0.9)
    assert builder.detector.calls == 1


def test_live_scene_builder_rejects_missing_or_ambiguous_hammer() -> None:
    with pytest.raises(ValueError, match="no confident"):
        _builder().build_object(_frame(), base_to_camera=np.eye(4), base_to_plane=np.eye(4))

    with pytest.raises(ValueError, match="ambiguous"):
        _builder(_detection(0.90), _detection(0.85)).build_object(
            _frame(), base_to_camera=np.eye(4), base_to_plane=np.eye(4)
        )


def test_live_scene_builder_rejects_insufficient_depth_at_grasp() -> None:
    frame = _frame()
    depth = frame.depth_image_m.copy()
    depth[20:60, 20:100] = 0.0
    missing_depth = SynchronizedRGBDFrame(
        color_image_rgb=frame.color_image_rgb,
        depth_image_m=depth,
        color_timestamp_ns=frame.color_timestamp_ns,
        depth_timestamp_ns=frame.depth_timestamp_ns,
        color_intrinsics=frame.color_intrinsics,
        frame_number=frame.frame_number,
        timestamp_clock_domain=frame.timestamp_clock_domain,
    )
    with pytest.raises(ValueError, match="foreground|depth"):
        _builder(_detection()).build_object(
            missing_depth, base_to_camera=np.eye(4), base_to_plane=np.eye(4)
        )


def test_live_scene_builder_uses_mean_depth_across_grip_region() -> None:
    frame = _frame()
    depth = frame.depth_image_m.copy()
    depth[:, ::4] = 0.51
    averaged = SynchronizedRGBDFrame(
        color_image_rgb=frame.color_image_rgb,
        depth_image_m=depth,
        color_timestamp_ns=frame.color_timestamp_ns,
        depth_timestamp_ns=frame.depth_timestamp_ns,
        color_intrinsics=frame.color_intrinsics,
        frame_number=frame.frame_number,
        timestamp_clock_domain=frame.timestamp_clock_domain,
    )

    anchor = _builder(_detection()).build_object(
        averaged, base_to_camera=np.eye(4), base_to_plane=np.eye(4)
    )

    assert anchor.attributes["grip_region_mean_depth_camera_m"] == pytest.approx(
        0.5025, abs=0.001
    )
    assert anchor.attributes["observed_grasp_point_plane_m"][2] == pytest.approx(
        anchor.attributes["grip_region_mean_depth_camera_m"]
    )


def test_checked_in_live_grasp_profile_is_loadable() -> None:
    profile = load_learned_grip_point(
        Path("configs/grasp_profiles/hammer_live_grasp.json")
    )

    assert profile.class_name == "hammer"
    assert profile.longitudinal == pytest.approx(-0.7177469962321966)
    assert profile.target_gripper_width_m == pytest.approx(0.021576227217825573)
