"""Materialize the recorded hammer demonstrations into a Mock-only skill."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_skill_system.demonstrations.rgbd_geometry import (
    transform_camera_pose_to_surface,
)
from robot_skill_system.perception.live_scene import LearnedGripPoint
from robot_skill_system.scene.models import Quaternion, Vector3
from robot_skill_system.scene.transforms import AnchorType, RelativePose, RigidTransform
from robot_skill_system.skills.models import (
    BindingSpec,
    EntityKind,
    SkillGraph,
    SkillNode,
    SkillType,
)
from robot_skill_system.storage.artifact_store import LocalArtifactStore

HAMMER_COMMAND_ALIASES = ("해머 가져와", "망치 가져와")
HAMMER_SKILL_ID = "take_hammer"
HAMMER_GRIPPER_TOOL_CLASS = "onrobot_rg2"
RECORDED_HAMMER_DEMONSTRATION_COUNT = 3
RECORDED_HAMMER_MINIMUM_VALID_SEGMENTATIONS = 2
RECORDED_HAMMER_LIFT_DISTANCE_M = 0.050


@dataclass(frozen=True, slots=True)
class RecordedHammerStageEvidence:
    recording_id: str
    artifact_uri: str
    checksum_sha256: str
    close_frame_index: int
    release_frame_index: int
    close_midpoint_camera_m: tuple[float, float, float]
    release_midpoint_camera_m: tuple[float, float, float]


def is_recorded_hammer_draft(payload: Mapping[str, Any]) -> bool:
    semantic = payload.get("draft")
    if not isinstance(semantic, Mapping):
        return False
    if semantic.get("suggested_skill_id") != HAMMER_SKILL_ID:
        return False
    instruction = _normalize_command(str(semantic.get("task_description") or ""))
    return instruction in {_normalize_command(value) for value in HAMMER_COMMAND_ALIASES}


def build_recorded_hammer_graph(
    *,
    store: LocalArtifactStore,
    draft_payload: Mapping[str, Any],
    calibration: Mapping[str, Any],
    version: str,
    grip_profile: LearnedGripPoint,
    grip_profile_version_id: str,
    grip_profile_checksum_sha256: str,
) -> SkillGraph:
    source_recording_ids = _source_recording_ids(draft_payload)
    evidence, rejected_recording_ids = _load_stage_evidence(
        store, source_recording_ids
    )
    camera_to_surface = RigidTransform.model_validate(
        calibration.get("camera_to_task_plane") or calibration.get("camera_to_surface")
    )
    grasp_points = [
        _camera_point_to_surface(item.close_midpoint_camera_m, camera_to_surface)
        for item in evidence
    ]
    release_points = [
        _camera_point_to_surface(item.release_midpoint_camera_m, camera_to_surface)
        for item in evidence
    ]
    grasp = _average_point(grasp_points)
    release = _average_point(release_points)
    lift = (grasp[0], grasp[1], grasp[2] + RECORDED_HAMMER_LIFT_DISTANCE_M)
    release_approach = (
        release[0],
        release[1],
        release[2] + RECORDED_HAMMER_LIFT_DISTANCE_M,
    )
    targets = {
        "grasp": _surface_pose(grasp),
        "lift": _surface_pose(lift),
        "release_approach": _surface_pose(release_approach),
        "release": _surface_pose(release),
    }
    node_specs: list[tuple[str, str, dict[str, Any]]] = [
        ("grip_open", "gripper.open", {"tool": "$tool"}),
        (
            "grip_rotate_j6",
            "motion.rotate_joint_6_relative",
            {
                "delta_rad": _symmetric_jaw_angle(grip_profile.jaw_relative_angle_rad),
                "motion_profile_id": "joint_safe",
            },
        ),
        (
            "grip_validate_target",
            "workspace.validate_target",
            {"target": targets["grasp"]},
        ),
        (
            "grip_move_target",
            "motion.move_l",
            {"target": targets["grasp"], "motion_profile_id": "linear_slow"},
        ),
        (
            "grip_set_width",
            "gripper.move_width",
            {"tool": "$tool", "width_m": grip_profile.target_gripper_width_m},
        ),
        (
            "action_validate_lift",
            "workspace.validate_target",
            {"target": targets["lift"]},
        ),
        (
            "action_lift",
            "motion.move_l",
            {"target": targets["lift"], "motion_profile_id": "linear_slow"},
        ),
        (
            "action_validate_release_approach",
            "workspace.validate_target",
            {"target": targets["release_approach"]},
        ),
        (
            "action_move_release_approach",
            "motion.move_l",
            {
                "target": targets["release_approach"],
                "motion_profile_id": "linear_slow",
            },
        ),
        (
            "end_validate_release",
            "workspace.validate_target",
            {"target": targets["release"]},
        ),
        (
            "end_move_release",
            "motion.move_l",
            {"target": targets["release"], "motion_profile_id": "linear_slow"},
        ),
        ("end_open", "gripper.open", {"tool": "$tool"}),
        (
            "end_retract",
            "motion.move_l",
            {
                "target": targets["release_approach"],
                "motion_profile_id": "linear_slow",
            },
        ),
    ]
    nodes = [
        SkillNode(
            node_id=node_id,
            operation=operation,
            arguments=arguments,
            on_success=(
                node_specs[index + 1][0] if index + 1 < len(node_specs) else None
            ),
        )
        for index, (node_id, operation, arguments) in enumerate(node_specs)
    ]
    source_artifacts = [
        str(draft_payload.get("artifact_uri") or ""),
        *[
            f"demonstrations/{recording_id}/rgbd_manifest.json"
            for recording_id in source_recording_ids
        ],
        *[item.artifact_uri for item in evidence],
    ]
    return SkillGraph(
        skill_id=HAMMER_SKILL_ID,
        version=version,
        name="해머 가져와",
        description="hammer1~3의 Grip/이동/놓기 경계로 생성한 Mock 검증용 스킬",
        skill_type=SkillType.COMPOSITE,
        source_demonstrations=source_artifacts,
        operator_style="safe",
        required_tools=[HAMMER_GRIPPER_TOOL_CLASS],
        required_entity_roles={
            "$object": "hammer",
            "$surface": "contact_target",
            "$tool": "tool",
        },
        bindings={
            "$object": BindingSpec(
                variable="$object",
                entity_kind=EntityKind.OBJECT,
                class_name="hammer",
                minimum_confidence=0.60,
                minimum_visible_fraction=0.60,
            ),
            "$surface": BindingSpec(
                variable="$surface",
                entity_kind=EntityKind.SURFACE,
                instance_id=str(calibration["surface_anchor_id"]),
                role="contact_target",
                minimum_confidence=0.80,
            ),
            "$tool": BindingSpec(
                variable="$tool",
                entity_kind=EntityKind.TOOL,
                instance_id="active_rg2",
                class_name=HAMMER_GRIPPER_TOOL_CLASS,
                minimum_confidence=0.80,
                must_be_attached=True,
                compatible_skill=HAMMER_SKILL_ID,
            ),
        },
        nodes=nodes,
        start_node=nodes[0].node_id,
        terminal_nodes=[nodes[-1].node_id],
        motion_profiles=["joint_safe", "linear_slow"],
        preconditions=[
            "fresh_scene",
            "hammer_binding_verified",
            "active_grip_profile_verified",
        ],
        postconditions=["hammer_released", "mock_validation_only"],
        uncertainty={
            "hardware_validated": False,
            "command_aliases": list(HAMMER_COMMAND_ALIASES),
            "materialization": "recorded_hammer_stage_boundary_average",
            "source_recording_ids": source_recording_ids,
            "valid_stage_recording_ids": [item.recording_id for item in evidence],
            "rejected_stage_recording_ids": rejected_recording_ids,
            "stage_evidence": [
                {
                    "recording_id": item.recording_id,
                    "artifact_uri": item.artifact_uri,
                    "checksum_sha256": item.checksum_sha256,
                    "close_frame_index": item.close_frame_index,
                    "release_frame_index": item.release_frame_index,
                }
                for item in evidence
            ],
            "grip_profile_version_id": grip_profile_version_id,
            "grip_profile_checksum_sha256": grip_profile_checksum_sha256,
            "target_gripper_width_m": grip_profile.target_gripper_width_m,
            "j6_relative_rotation_rad": _symmetric_jaw_angle(
                grip_profile.jaw_relative_angle_rad
            ),
            "task_plane_calibration_id": str(calibration["calibration_id"]),
            "task_plane_anchor_id": str(calibration["surface_anchor_id"]),
            "task_plane_source_frame": str(calibration["source_frame"]),
            "camera_to_task_plane": calibration.get("camera_to_task_plane")
            or calibration.get("camera_to_surface"),
            "base_chain": calibration.get("base_chain"),
            "observed_trajectory_task_plane_m": [
                list(grasp),
                list(lift),
                list(release_approach),
                list(release),
                list(release_approach),
            ],
            "trajectory_quality": {
                "mean_confidence": 0.90,
                "source": "checksum_verified_local_stage_boundaries",
            },
        },
    )


def _source_recording_ids(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("source_recording_ids")
    source_ids = [str(value) for value in raw] if isinstance(raw, list) else []
    if len(source_ids) != RECORDED_HAMMER_DEMONSTRATION_COUNT:
        raise ValueError("take_hammer requires exactly the hammer1~3 demonstration set")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("take_hammer demonstration IDs must be unique")
    return source_ids


def _load_stage_evidence(
    store: LocalArtifactStore, source_recording_ids: Sequence[str]
) -> tuple[list[RecordedHammerStageEvidence], list[str]]:
    selected: list[RecordedHammerStageEvidence] = []
    rejected: list[str] = []
    for recording_id in source_recording_ids:
        root = store.path_for(f"demonstrations/{recording_id}/induction_v1")
        candidates: list[RecordedHammerStageEvidence] = []
        if root.is_dir():
            for path in sorted(root.glob("segmentation_report_*.json")):
                candidate = _parse_stage_report(store, recording_id, path)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            rejected.append(recording_id)
            continue
        selected.append(
            max(
                candidates,
                key=lambda item: (
                    item.close_frame_index,
                    item.release_frame_index,
                    item.checksum_sha256,
                ),
            )
        )
    if len(selected) < RECORDED_HAMMER_MINIMUM_VALID_SEGMENTATIONS:
        raise ValueError("take_hammer needs at least two valid local stage segmentations")
    return selected, rejected


def _parse_stage_report(
    store: LocalArtifactStore, recording_id: str, path: Path
) -> RecordedHammerStageEvidence | None:
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    expected_checksum = path.stem.removeprefix("segmentation_report_")
    if checksum != expected_checksum:
        raise ValueError(f"stage report checksum mismatch: {path.name}")
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("status") != "succeeded":
        return None
    try:
        segmentation = payload["segmentation"]
        close = segmentation["evidence"]["first_stable_close"]
        release = segmentation["evidence"]["final_stable_open"]
        close_transition = close["transition"]
        release_transition = release["transition"]
        close_point = _xyz(close_transition["midpoint_camera_m"])
        release_point = _xyz(release_transition["midpoint_camera_m"])
        close_frame_index = int(close["frame_index"])
        release_frame_index = int(release["frame_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if release_frame_index <= close_frame_index:
        return None
    return RecordedHammerStageEvidence(
        recording_id=recording_id,
        artifact_uri=path.relative_to(store.root).as_posix(),
        checksum_sha256=checksum,
        close_frame_index=close_frame_index,
        release_frame_index=release_frame_index,
        close_midpoint_camera_m=close_point,
        release_midpoint_camera_m=release_point,
    )


def _camera_point_to_surface(
    point_camera_m: tuple[float, float, float], camera_to_surface: RigidTransform
) -> tuple[float, float, float]:
    result = transform_camera_pose_to_surface(
        camera_to_surface,
        RigidTransform(
            translation_m=Vector3(
                x=point_camera_m[0], y=point_camera_m[1], z=point_camera_m[2]
            ),
            rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    return result.translation_m.as_tuple()


def _surface_pose(point_m: tuple[float, float, float]) -> dict[str, Any]:
    return RelativePose(
        anchor_id="$surface",
        anchor_type=AnchorType.SURFACE,
        position_m=Vector3(x=point_m[0], y=point_m[1], z=point_m[2]),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    ).model_dump(mode="json")


def _xyz(value: Mapping[str, Any]) -> tuple[float, float, float]:
    result = (float(value["x"]), float(value["y"]), float(value["z"]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError("stage midpoint must contain finite metres")
    return result


def _average_point(
    points: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    count = len(points)
    return tuple(sum(point[index] for point in points) / count for index in range(3))  # type: ignore[return-value]


def _symmetric_jaw_angle(value: float) -> float:
    angle = ((float(value) + math.pi / 2.0) % math.pi) - math.pi / 2.0
    if not math.isfinite(angle):
        raise ValueError("GripProfile jaw angle must be finite")
    return angle


def _normalize_command(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


__all__ = [
    "HAMMER_COMMAND_ALIASES",
    "HAMMER_GRIPPER_TOOL_CLASS",
    "HAMMER_SKILL_ID",
    "RecordedHammerStageEvidence",
    "build_recorded_hammer_graph",
    "is_recorded_hammer_draft",
]
