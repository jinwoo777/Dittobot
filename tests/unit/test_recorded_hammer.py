from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robot_skill_system.application import MVPApplication
from robot_skill_system.grasping.recorded_hammer import build_recorded_hammer_graph
from robot_skill_system.openai_integration.mock_client import MockOpenAIClient
from robot_skill_system.openai_integration.schemas import RecordingSkillDraftInput
from robot_skill_system.perception.live_scene import LearnedGripPoint
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.scene.transforms import RigidTransform
from robot_skill_system.settings import Settings
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.vertical_slice import capture_mock_scene

SOURCE_IDS = ["rgbd_hammer1", "rgbd_hammer2", "rgbd_hammer3"]


def _put_stage_report(
    store: LocalArtifactStore,
    recording_id: str,
    *,
    close_frame: int,
    release_frame: int,
    close_xyz: tuple[float, float, float],
    release_xyz: tuple[float, float, float],
) -> str:
    payload = {
        "status": "succeeded",
        "segmentation": {
            "evidence": {
                "first_stable_close": {
                    "frame_index": close_frame,
                    "transition": {
                        "midpoint_camera_m": dict(zip("xyz", close_xyz, strict=True))
                    },
                },
                "final_stable_open": {
                    "frame_index": release_frame,
                    "transition": {
                        "midpoint_camera_m": dict(zip("xyz", release_xyz, strict=True))
                    },
                },
            }
        },
    }
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    checksum = hashlib.sha256(content).hexdigest()
    uri = (
        f"demonstrations/{recording_id}/induction_v1/"
        f"segmentation_report_{checksum}.json"
    )
    store.put_bytes(uri, content, media_type="application/json")
    return uri


def test_recorded_hammer_uses_two_valid_stage_boundaries_and_all_three_sources(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first_uri = _put_stage_report(
        store,
        SOURCE_IDS[0],
        close_frame=16,
        release_frame=34,
        close_xyz=(0.10, 0.20, 0.30),
        release_xyz=(0.30, 0.20, 0.10),
    )
    second_uri = _put_stage_report(
        store,
        SOURCE_IDS[1],
        close_frame=27,
        release_frame=63,
        close_xyz=(0.20, 0.20, 0.30),
        release_xyz=(0.40, 0.20, 0.10),
    )
    profile = LearnedGripPoint(
        class_name="hammer",
        longitudinal=-0.7,
        lateral=-0.02,
        jaw_relative_angle_rad=1.1,
        target_gripper_width_m=0.022,
        source_path=tmp_path / "profile.json",
    )
    graph = build_recorded_hammer_graph(
        store=store,
        draft_payload={
            "artifact_uri": "demonstrations/rgbd_hammer1/skill_drafts/draft.json",
            "source_recording_ids": SOURCE_IDS,
            "draft": {
                "suggested_skill_id": "take_hammer",
                "task_description": "해머 가져와",
            },
        },
        calibration={
            "calibration_id": "cal_test",
            "surface_anchor_id": "surface_test",
            "source_frame": "camera_color_optical_frame",
            "camera_to_task_plane": RigidTransform.identity().model_dump(mode="json"),
            "base_chain": None,
        },
        version="0.1.0-candidate",
        grip_profile=profile,
        grip_profile_version_id="grip_test",
        grip_profile_checksum_sha256="0" * 64,
    )

    assert SkillGraphValidator().inspect(graph).valid is True
    assert SkillCompiler().compile(graph).validation_report.valid is True
    assert graph.uncertainty["source_recording_ids"] == SOURCE_IDS
    assert graph.uncertainty["valid_stage_recording_ids"] == SOURCE_IDS[:2]
    assert graph.uncertainty["rejected_stage_recording_ids"] == [SOURCE_IDS[2]]
    assert [item["artifact_uri"] for item in graph.uncertainty["stage_evidence"]] == [
        first_uri,
        second_uri,
    ]
    assert [node.operation for node in graph.nodes] == [
        "gripper.open",
        "motion.rotate_joint_6_relative",
        "workspace.validate_target",
        "motion.move_l",
        "gripper.move_width",
        "workspace.validate_target",
        "motion.move_l",
        "workspace.validate_target",
        "motion.move_l",
        "workspace.validate_target",
        "motion.move_l",
        "gripper.open",
        "motion.move_l",
    ]
    assert graph.nodes[4].arguments["width_m"] == pytest.approx(0.022)
    assert graph.required_tools == ["onrobot_rg2"]
    assert graph.bindings["$tool"].class_name == "onrobot_rg2"
    assert graph.uncertainty["observed_trajectory_task_plane_m"][0] == pytest.approx(
        [0.15, 0.20, 0.30]
    )

    mock_scene = MVPApplication._scene_with_graph_task_plane(
        capture_mock_scene(), graph
    )
    mock_scene = MVPApplication._scene_with_mock_binding_fixtures(
        mock_scene, graph
    )
    bindings = EntityBinder().bind_entities(
        mock_scene,
        [graph.bindings["$object"], graph.bindings["$tool"]],
    )
    assert bindings["$object"].entity_id == "mock_hammer_01"
    assert bindings["$tool"].entity_id == "active_rg2"
    assert bindings["$tool"].entity.tool_class == "onrobot_rg2"


def test_mock_semantic_analyzer_identifies_hammer_grip_action_end() -> None:
    request = RecordingSkillDraftInput(
        recording_id="rgbd_hammer1",
        name_hint="take_hammer",
        operator_instruction="해머 가져와",
        recording_summary={"frame_count": 5},
        primitive_catalog=[
            "gripper.open",
            "motion.rotate_joint_6_relative",
            "motion.move_l",
            "gripper.move_width",
        ],
        entity_role_catalog=["tool", "target_object", "target_surface"],
        keyframe_indices=[0, 1, 2, 3, 4],
        limitations=["Mock semantic draft only."],
    )

    draft, _metadata = MockOpenAIClient().analyze_recording_skill_draft(
        request, "trace"
    )

    assert draft.required_entity_roles == ["tool", "target_object", "target_surface"]
    assert draft.scene_observation.target_object is not None
    assert draft.scene_observation.target_object.class_name == "hammer"
    assert [item.operation for item in draft.primitive_sequence] == [
        "gripper.open",
        "motion.rotate_joint_6_relative",
        "motion.move_l",
        "gripper.move_width",
        "motion.move_l",
        "gripper.open",
    ]


def test_archive_recording_draft_preserves_the_source_demonstration(
    tmp_path: Path,
) -> None:
    service = MVPApplication(
        Settings.from_env(
            {
                "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
                "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
                "OPENAI_MODE": "mock",
                "ROBOT_EXECUTION_MODE": "mock",
                "DRY_RUN": "true",
            },
            root=Path(__file__).resolve().parents[2],
        )
    )
    draft_id = "draft_0123456789abcdef0123456789abcdef"
    manifest = service.store.put_json(
        "demonstrations/rgbd_hammer1/rgbd_manifest.json", {"recording_id": "rgbd_hammer1"}
    )
    service.store.put_json(
        f"demonstrations/rgbd_hammer1/skill_drafts/{draft_id}.json",
        {
            "draft_id": draft_id,
            "source_recording_id": "rgbd_hammer1",
            "status": "semantic_draft",
        },
    )
    service.store.put_json(
        f"demonstrations/rgbd_hammer1/skill_drafts/{draft_id}_evidence/local.json",
        {"draft_id": draft_id},
    )
    try:
        result = service.archive_recording_skill_draft(draft_id)
    finally:
        service.close()

    assert result == {
        "archived": True,
        "draft_id": draft_id,
        "archive_uri": f"archive/recording_skill_drafts/{draft_id}",
        "archived_draft_files": 1,
        "archived_evidence_files": 1,
    }
    assert service.store.path_for(manifest.uri).is_file()
    assert service.store.path_for(
        f"archive/recording_skill_drafts/{draft_id}/draft.json"
    ).is_file()
