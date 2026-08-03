"""Unit tests for strict Scene schemas and local rigid transforms."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from robot_skill_system.scene.models import (
    AccessPolicy,
    ConfidenceSummary,
    Pose,
    Quaternion,
    SceneSnapshot,
    Vector3,
    WorkspaceRegion,
    WorkspaceRole,
)
from robot_skill_system.scene.transforms import AnchorType, RelativePose, bind_relative_pose


def _pose(*, frame_id: str = "base", timestamp_ns: int = 1_000_000_000) -> Pose:
    return Pose(
        frame_id=frame_id,
        position_m=Vector3(x=1.0, y=2.0, z=0.0),
        orientation_xyzw=Quaternion(
            x=0.0,
            y=0.0,
            z=math.sqrt(0.5),
            w=math.sqrt(0.5),
        ),
        timestamp_ns=timestamp_ns,
        source="robot_tf",
        confidence=0.99,
    )


@pytest.mark.parametrize(
    "quaternion",
    [
        {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
        {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0},
    ],
)
def test_invalid_quaternion_is_rejected(quaternion: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="quaternion"):
        Quaternion.model_validate(quaternion)


@pytest.mark.parametrize("frame_id", ["", "/base", "bad frame", "base//tool", "a/../b"])
def test_invalid_frame_id_is_rejected(frame_id: str) -> None:
    payload = _pose().model_dump()
    payload["frame_id"] = frame_id
    with pytest.raises(ValidationError, match="frame_id"):
        Pose.model_validate(payload)


def test_anchor_relative_pose_binds_with_quaternion_composition() -> None:
    relative = RelativePose(
        anchor_id="surface_01",
        anchor_type=AnchorType.SURFACE,
        position_m=Vector3(x=1.0, y=0.0, z=0.2),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    bound = bind_relative_pose(relative, _pose(), target_frame_id="base")

    assert bound.frame_id == "base"
    assert bound.position_m.x == pytest.approx(1.0)
    assert bound.position_m.y == pytest.approx(3.0)
    assert bound.position_m.z == pytest.approx(0.2)
    assert bound.orientation_xyzw.z == pytest.approx(math.sqrt(0.5))


def test_base_absolute_relative_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absolute anchors"):
        RelativePose(
            anchor_id="base_link",
            position_m=Vector3(x=0.1, y=0.2, z=0.3),
            orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )


def test_unknown_workspace_must_fail_closed() -> None:
    with pytest.raises(ValidationError, match="occupied or forbidden"):
        WorkspaceRegion(
            region_id="unknown_01",
            role=WorkspaceRole.UNKNOWN_REGION,
            geometry={"type": "box", "size_m": [1.0, 1.0, 1.0]},
            frame_id="base",
            minimum_clearance_m=0.03,
            access_policy=AccessPolicy.ALLOWED,
            confidence=0.5,
        )


def test_scene_freshness_uses_nanoseconds_and_validity_window() -> None:
    scene = SceneSnapshot(
        scene_id="scene_01",
        timestamp_ns=1_000_000_000,
        reference_frame="base",
        valid_for_ms=500,
        calibration_id="cal_01",
        confidence_summary=ConfidenceSummary(overall=0.9),
    )

    assert scene.is_fresh(now_ns=1_400_000_000)
    assert not scene.is_fresh(now_ns=1_600_000_000)
    assert not scene.is_fresh(now_ns=900_000_000)
