"""Storage-backed task-flow materialization trust-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from robot_skill_system.runtime.task_flow_materializer import (
    ExecutableGripProfileArtifact,
    TaskFlowMaterializationError,
    TaskFlowMaterializer,
)
from robot_skill_system.skills.models import SkillEdge, SkillGraph
from robot_skill_system.skills.task_flow import GripProfile
from robot_skill_system.storage import Database, LocalArtifactStore, StorageRepository
from robot_skill_system.storage.orm import (
    ActionEndMappingRecord,
    SkillVersionRecord,
    StageDefinitionRecord,
)


def _pose(z_m: float) -> dict[str, Any]:
    return {
        "anchor_id": "$object",
        "anchor_type": "object",
        "position_m": {"x": 0.0, "y": 0.0, "z": z_m},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _grip_profile() -> GripProfile:
    return GripProfile(
        profile_id="hammer_default",
        version="1.0.0",
        object_class_id="hammer",
        object_frame_policy="object_relative_6d",
        object_frame_revision="object-frame-v1",
        pregrasp_pose=_pose(0.08),
        grasp_pose=_pose(0.01),
        jaw_axis={"x": 1.0, "y": 0.0, "z": 0.0},
        approach_axis={"x": 0.0, "y": 0.0, "z": -1.0},
        required_tool_class="rg2",
        motion_profile_id="linear_slow",
        gripper_calibration_profile_id="rg2_default",
        verification_profile_id="grasp_default",
        lifecycle_status="active",
        validation_status="passed",
        hardware_compatible=False,
    )


def _action_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="bring_action",
        version="1.0.0",
        name="bring action",
        description="Move a held object toward a bounded destination role.",
        skill_type="motion",
        bindings={
            "$object": {"entity_kind": "object", "class_name": "hammer"},
            "$destination": {
                "entity_kind": "workspace",
                "role": "destination",
            },
        },
        required_entity_roles={"$destination": "destination"},
        nodes=[
            {
                "node_id": "perform",
                "operation": "motion.wait",
                "arguments": {"duration_s": 0.01},
            }
        ],
        start_node="perform",
        terminal_nodes=["perform"],
    )


def _end_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="release_safe_end",
        version="1.0.0",
        name="safe release end motion",
        description="Open the gripper and verify that the object was released.",
        skill_type="manipulation",
        bindings={
            "$object": {"entity_kind": "object", "class_name": "hammer"},
            "$gripper": {
                "entity_kind": "tool",
                "class_name": "rg2",
                "must_be_attached": True,
            },
        },
        nodes=[
            {
                "node_id": "open",
                "operation": "gripper.open",
                "arguments": {"tool": "$gripper"},
            },
            {
                "node_id": "verify",
                "operation": "grasp.verify_released",
                "arguments": {
                    "object": "$object",
                    "tool": "$gripper",
                    "verification_profile_id": "grasp_default",
                },
            },
        ],
        edges=[SkillEdge(source_node="open", target_node="verify")],
        start_node="open",
        terminal_nodes=["verify"],
    )


@dataclass(frozen=True)
class RegistryFixture:
    database: Database
    repository: StorageRepository
    store: LocalArtifactStore
    materializer: TaskFlowMaterializer
    grip_artifact_uri: str
    action_skill_version_id: str
    action_stage_id: str
    mapping_id: str


def _registry(
    tmp_path: Path, *, grip_payload: dict[str, Any] | None = None
) -> RegistryFixture:
    database = Database.from_path(tmp_path / "registry.sqlite3")
    database.create_schema()
    repository = StorageRepository(database)
    store = LocalArtifactStore(tmp_path / "artifacts")

    object_entry = repository.create_catalog_entry(
        kind="object", canonical_id="hammer", aliases=("망치",)
    )
    action_entry = repository.create_catalog_entry(
        kind="action", canonical_id="bring", aliases=("가져와",)
    )
    end_entry = repository.create_catalog_entry(
        kind="end_motion", canonical_id="release_safe"
    )

    artifact_payload = grip_payload or ExecutableGripProfileArtifact(
        grip_profile=_grip_profile(),
        provenance={"source": "unit-test-rgbd-demonstration"},
    ).model_dump(mode="json")
    artifact = store.put_json("grips/hammer/1.0.0.json", artifact_payload)
    grip = repository.register_grip_profile_version(
        object_class_id="hammer",
        semantic_version="1.0.0",
        artifact_uri=artifact.uri,
        artifact_checksum_sha256=artifact.checksum_sha256,
        object_frame_policy="object_relative_6d",
        object_frame_revision="object-frame-v1",
        status="validated",
        validation_status="passed",
        hardware_compatible=False,
        gripper_calibration_profile_id="rg2_default",
    )
    repository.activate_grip_profile_version(grip.id)
    repository.activate_catalog_entry(object_entry.id)

    action_graph = _action_graph()
    action_skill = repository.register_skill_version(
        name="action:bring",
        intent="bring",
        semantic_version=action_graph.version,
        graph=action_graph,
        status="validated",
        validation_status="passed",
    )
    action_stage = repository.register_stage_definition(
        kind="action",
        canonical_id="bring",
        skill_version_id=action_skill.id,
        semantic_version="1.0.0",
        status="validated",
        validation_status="passed",
        required_roles=("$destination",),
        input_contract={"attachment": "holding", "force_mode": "disabled"},
        output_contract={"attachment": "holding", "force_mode": "disabled"},
        anchor_policy={"allowed_anchor_types": ["destination", "workspace_region"]},
    )
    repository.activate_stage_definition(action_stage.id)

    end_graph = _end_graph()
    end_skill = repository.register_skill_version(
        name="end_motion:release_safe",
        intent="release_safe",
        semantic_version=end_graph.version,
        graph=end_graph,
        status="validated",
        validation_status="passed",
    )
    end_stage = repository.register_stage_definition(
        kind="end_motion",
        canonical_id="release_safe",
        skill_version_id=end_skill.id,
        semantic_version="1.0.0",
        status="validated",
        validation_status="passed",
        input_contract={"attachment": "holding", "force_mode": "disabled"},
        output_contract={"attachment": "released", "force_mode": "disabled"},
        anchor_policy={"allowed_anchor_types": ["object", "workspace_region"]},
    )
    repository.activate_stage_definition(end_stage.id)
    repository.activate_catalog_entry(end_entry.id)

    mapping = repository.map_action_to_end_motion(
        action_id="bring", end_motion_id="release_safe"
    )
    repository.activate_catalog_entry(action_entry.id)

    return RegistryFixture(
        database=database,
        repository=repository,
        store=store,
        materializer=TaskFlowMaterializer(repository, store),
        grip_artifact_uri=artifact.uri,
        action_skill_version_id=action_skill.id,
        action_stage_id=action_stage.id,
        mapping_id=mapping.id,
    )


def test_materializer_loads_exact_records_and_composes_deterministically(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    try:
        first = registry.materializer.materialize(
            object_class_id="hammer",
            action_id="bring",
            resolved_role_bindings={"$destination": "destination_01"},
        )
        second = registry.materializer.materialize(
            object_class_id="망치",
            action_id="가져와",
            resolved_role_bindings={"$destination": "destination_01"},
        )

        assert first.manifest == second.manifest
        assert first.manifest.mapping_id == registry.mapping_id
        assert first.selection.action_end_mapping_record_id == registry.mapping_id
        assert first.selection.action_end_mapping_revision == 1
        assert first.resolved_role_bindings == {"$destination": "destination_01"}
        assert [node.node_id.split("__", 1)[0] for node in first.graph.nodes] == [
            "grip",
            "grip",
            "grip",
            "grip",
            "grip",
            "grip",
            "grip",
            "action",
            "end",
            "end",
        ]
    finally:
        registry.database.close()


def test_materializer_rejects_tampered_grip_artifact_checksum(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    try:
        registry.store.path_for(registry.grip_artifact_uri).write_text(
            '{"schema_version":"grip-profile/1.0"}', encoding="utf-8"
        )
        with pytest.raises(TaskFlowMaterializationError, match="checksum mismatch"):
            registry.materializer.materialize(
                object_class_id="hammer", action_id="bring"
            )
    finally:
        registry.database.close()


def test_materializer_rejects_legacy_2d_grip_diagnostic_payload(
    tmp_path: Path,
) -> None:
    registry = _registry(
        tmp_path,
        grip_payload={
            "schema_version": "grip-point-diagnostic-profile/1.0",
            "object_class_id": "hammer",
            "execution_mode": "offline_diagnostic_only",
            "hardware_compatible": False,
            "evidence": {"normalized_grasp_point": [0.1, 0.2]},
        },
    )
    try:
        with pytest.raises(TaskFlowMaterializationError, match="2D grip-point diagnostic"):
            registry.materializer.materialize(
                object_class_id="hammer", action_id="bring"
            )
    finally:
        registry.database.close()


def test_materializer_rechecks_graph_checksum_and_stored_role_contract(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    try:
        with registry.database.session() as session:
            skill = session.get(SkillVersionRecord, registry.action_skill_version_id)
            assert skill is not None
            skill.graph_json = {**skill.graph_json, "description": "tampered graph"}
        with pytest.raises(TaskFlowMaterializationError, match="SkillGraph checksum mismatch"):
            registry.materializer.materialize(
                object_class_id="hammer", action_id="bring"
            )

        with registry.database.session() as session:
            skill = session.get(SkillVersionRecord, registry.action_skill_version_id)
            stage = session.get(StageDefinitionRecord, registry.action_stage_id)
            assert skill is not None and stage is not None
            skill.graph_json = _action_graph().model_dump(mode="json")
            stage.required_roles_json = []
        with pytest.raises(TaskFlowMaterializationError, match="required-role snapshot"):
            registry.materializer.materialize(
                object_class_id="hammer", action_id="bring"
            )
    finally:
        registry.database.close()


def test_materializer_rechecks_exact_mapping_revision_checksum(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    try:
        with registry.database.session() as session:
            mapping = session.get(ActionEndMappingRecord, registry.mapping_id)
            assert mapping is not None
            mapping.mapping_checksum_sha256 = "0" * 64

        with pytest.raises(TaskFlowMaterializationError, match="mapping checksum"):
            registry.materializer.materialize(
                object_class_id="hammer", action_id="bring"
            )
    finally:
        registry.database.close()


def test_materializer_rejects_absolute_policy_and_incomplete_scene_roles(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    try:
        with pytest.raises(TaskFlowMaterializationError, match="exactly cover"):
            registry.materializer.materialize(
                object_class_id="hammer",
                action_id="bring",
                resolved_role_bindings={},
            )

        with registry.database.session() as session:
            stage = session.get(StageDefinitionRecord, registry.action_stage_id)
            assert stage is not None
            stage.anchor_policy_json = {
                "anchor_type": "robot_base_absolute",
                "frame_id": "base_link",
            }
        with pytest.raises(TaskFlowMaterializationError, match="absolute target"):
            registry.materializer.materialize(
                object_class_id="hammer",
                action_id="bring",
                resolved_role_bindings={"$destination": "destination_01"},
            )
    finally:
        registry.database.close()
