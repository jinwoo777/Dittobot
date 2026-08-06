from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import inspect

from robot_skill_system.storage import Database, LocalArtifactStore, StorageRepository
from robot_skill_system.storage.grip_point_importer import GripPointResultImporter
from robot_skill_system.storage.orm import ActionEndMappingRecord

COMPONENT_TABLES = {
    "semantic_catalog",
    "grip_profiles",
    "grip_profile_versions",
    "stage_definitions",
    "action_end_mappings",
    "task_flow_plans",
}


def _repository(tmp_path: Path) -> tuple[Database, StorageRepository]:
    database = Database.from_path(tmp_path / "registry.sqlite3")
    database.create_schema()
    return database, StorageRepository(database)


def _register_active_stage(
    repository: StorageRepository,
    *,
    kind: str,
    canonical_id: str,
    required_roles: tuple[str, ...] = (),
    hardware_compatible: bool = False,
):
    catalog = repository.create_catalog_entry(kind=kind, canonical_id=canonical_id)
    skill = repository.register_skill_version(
        name=f"{kind}:{canonical_id}",
        intent=canonical_id,
        semantic_version="1.0.0",
        graph={"stage": kind, "id": canonical_id},
        status="validated",
        validation_status="passed",
        hardware_compatible=hardware_compatible,
    )
    stage = repository.register_stage_definition(
        kind=kind,
        canonical_id=canonical_id,
        skill_version_id=skill.id,
        status="validated",
        validation_status="passed",
        required_roles=required_roles,
        input_contract={"attachment": "held"},
        output_contract={"attachment": "held" if kind == "action" else "released"},
        anchor_policy={"anchor_type": "object"},
    )
    repository.activate_stage_definition(stage.id)
    return catalog, stage


def test_component_schema_is_additive_and_contains_no_blob_columns(tmp_path: Path) -> None:
    database, _ = _repository(tmp_path)
    inspector = inspect(database.engine)

    assert set(inspector.get_table_names()) >= COMPONENT_TABLES
    for table_name in COMPONENT_TABLES:
        assert all(
            "BLOB" not in str(column["type"]).upper()
            for column in inspector.get_columns(table_name)
        )
    database.close()


def test_semantic_catalog_is_draft_first_alias_aware_and_conflict_safe(
    tmp_path: Path,
) -> None:
    database, repository = _repository(tmp_path)

    action = repository.create_catalog_entry(
        kind="action",
        canonical_id="bring",
        display_name="Bring",
        aliases=("가져와", "bring it"),
    )

    assert action.status == "draft"
    assert repository.get_catalog_entry(kind="action", identifier="가져와") is not None
    assert repository.create_catalog_entry(kind="action", canonical_id="bring").id == action.id
    assert [item.canonical_id for item in repository.list_catalog_entries(kind="action")] == [
        "bring"
    ]
    with pytest.raises(ValueError, match="different content"):
        repository.create_catalog_entry(
            kind="action", canonical_id="bring", display_name="Different"
        )
    with pytest.raises(ValueError, match="already owned"):
        repository.create_catalog_entry(
            kind="action", canonical_id="fetch", aliases=("가져와",)
        )
    with pytest.raises(ValueError, match="active, passed stage"):
        repository.activate_catalog_entry(action.id)
    database.close()


def test_mapping_reuses_end_motion_pins_versions_and_has_one_active_revision(
    tmp_path: Path,
) -> None:
    database, repository = _repository(tmp_path)
    object_entry = repository.create_catalog_entry(kind="object", canonical_id="hammer")
    grip = repository.register_grip_profile_version(
        object_class_id="hammer",
        semantic_version="1.0.0",
        artifact_uri="grips/hammer/1.0.0.json",
        artifact_checksum_sha256="a" * 64,
        object_frame_policy="object_relative_6d",
        object_frame_revision="object-frame-v1",
        status="validated",
        validation_status="passed",
    )
    repository.activate_grip_profile_version(grip.id)
    repository.activate_catalog_entry(object_entry.id)

    end_catalog, end_stage = _register_active_stage(
        repository, kind="end_motion", canonical_id="release_safe"
    )
    repository.activate_catalog_entry(end_catalog.id)
    other_end_catalog, _ = _register_active_stage(
        repository, kind="end_motion", canonical_id="release_bin"
    )
    repository.activate_catalog_entry(other_end_catalog.id)
    bring_catalog, bring_stage = _register_active_stage(
        repository,
        kind="action",
        canonical_id="bring",
        required_roles=("$destination",),
    )
    wipe_catalog, _ = _register_active_stage(
        repository,
        kind="action",
        canonical_id="wipe",
        required_roles=("$surface",),
    )

    bring_mapping = repository.map_action_to_end_motion(
        action_id="bring", end_motion_id="release_safe"
    )
    wipe_mapping = repository.map_action_to_end_motion(
        action_id="wipe", end_motion_id="release_safe"
    )
    repository.activate_catalog_entry(bring_catalog.id)
    repository.activate_catalog_entry(wipe_catalog.id)

    assert bring_mapping.action_stage_definition_id == bring_stage.id
    assert bring_mapping.end_motion_stage_definition_id == end_stage.id
    assert wipe_mapping.end_motion_stage_definition_id == end_stage.id
    assert (
        repository.map_action_to_end_motion(
            action_id="bring", end_motion_id="release_safe"
        ).id
        == bring_mapping.id
    )

    composite = repository.register_skill_version(
        name="flow:hammer:bring",
        intent="bring",
        semantic_version="1.0.0",
        graph={"order": ["grip", "action", "end_motion"]},
        status="validated",
        validation_status="passed",
    )
    plan = repository.record_task_flow_plan(
        grip_profile_version_id=grip.id,
        action_end_mapping_id=bring_mapping.id,
        composer_version="1.0.0",
        composite_skill_version_id=composite.id,
    )
    duplicate_plan = repository.record_task_flow_plan(
        grip_profile_version_id=grip.id,
        action_end_mapping_id=bring_mapping.id,
        composer_version="1.0.0",
        composite_skill_version_id=composite.id,
    )
    assert duplicate_plan.id == plan.id
    assert plan.action_checksum_sha256 == bring_stage.graph_checksum_sha256
    assert plan.end_motion_checksum_sha256 == end_stage.graph_checksum_sha256

    original_mapping_checksum = bring_mapping.mapping_checksum_sha256
    with database.session() as session:
        tampered = session.get(ActionEndMappingRecord, bring_mapping.id)
        assert tampered is not None
        tampered.mapping_checksum_sha256 = "f" * 64
    with pytest.raises(ValueError, match="checksum verification failed"):
        repository.get_action_end_mapping(action_id="bring")
    with database.session() as session:
        tampered = session.get(ActionEndMappingRecord, bring_mapping.id)
        assert tampered is not None
        tampered.mapping_checksum_sha256 = original_mapping_checksum

    replacement = repository.map_action_to_end_motion(
        action_id="bring", end_motion_id="release_bin"
    )
    assert replacement.revision == 2
    active = repository.get_action_end_mapping(action_id="bring")
    assert active is not None and active.id == replacement.id
    assert len(repository.list_action_end_mappings(action_id="bring", active_only=True)) == 1
    database.close()


def test_legacy_grip_point_import_is_idempotent_candidate_only_and_external(
    tmp_path: Path,
) -> None:
    database, repository = _repository(tmp_path)
    store = LocalArtifactStore(tmp_path / "artifacts")
    importer = GripPointResultImporter(repository, store)
    source = Path(__file__).parents[2] / "data/test/grip_point/result.json"

    first = importer.import_file(source)
    second = importer.import_file(source)

    assert [item.object_class_id for item in first.profiles] == [
        "hammer",
        "jetty",
        "knife",
        "spanner",
    ]
    assert [item.grip_profile_version_id for item in second.profiles] == [
        item.grip_profile_version_id for item in first.profiles
    ]
    assert {item.status for item in repository.list_catalog_entries(kind="object")} == {"draft"}
    for imported in first.profiles:
        version = repository.get_grip_profile_version(imported.grip_profile_version_id)
        assert version is not None
        assert version.status == "candidate"
        assert version.validation_status == "pending"
        assert version.hardware_compatible is False
        assert version.auto_activation_allowed is False
        assert version.gripper_calibration_profile_id is None
        assert "normalized_grasp_point" not in json.dumps(version.metadata_json)
        artifact = json.loads(
            store.read_bytes(
                imported.artifact_uri,
                expected_checksum_sha256=imported.artifact_checksum_sha256,
            )
        )
        assert "normalized_grasp_point" in artifact["evidence"]["learned_grip_model"]
        with pytest.raises(ValueError, match="passed"):
            repository.activate_grip_profile_version(version.id)
    database.close()


def test_legacy_import_rejects_unsafe_object_identifier(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path)
    source = tmp_path / "unsafe.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "execution_mode": "offline_diagnostic_only",
                "robot_or_gripper_called": False,
                "approach_direction_estimated": False,
                "limitations": [],
                "objects": {"../escape": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe object class ID"):
        GripPointResultImporter(
            repository, LocalArtifactStore(tmp_path / "artifacts")
        ).import_file(source)
    assert repository.list_catalog_entries() == []
    database.close()
