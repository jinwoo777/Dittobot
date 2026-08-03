from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, select

from robot_skill_system.scene.models import ConfidenceSummary, SceneSnapshot
from robot_skill_system.storage import (
    Database,
    ExecutionEventRecord,
    LocalArtifactStore,
    StorageRepository,
)

REQUIRED_TABLES = {
    "operators",
    "demonstrations",
    "scenes",
    "objects",
    "tools",
    "workspaces",
    "skills",
    "skill_versions",
    "skill_embeddings",
    "validation_runs",
    "execution_runs",
    "execution_events",
}


def _scene(timestamp_ns: int = 1_000_000_000) -> SceneSnapshot:
    return SceneSnapshot(
        schema_version="1.0",
        scene_id="scene_storage",
        timestamp_ns=timestamp_ns,
        reference_frame="base",
        valid_for_ms=5_000,
        calibration_id="calibration_01",
        confidence_summary=ConfidenceSummary(overall=0.99),
    )


def test_sqlite_schema_contains_required_metadata_tables(tmp_path: Path) -> None:
    database = Database.from_path(tmp_path / "registry.sqlite3")
    database.create_schema()

    table_names = set(inspect(database.engine).get_table_names())

    assert table_names >= REQUIRED_TABLES
    for table_name in REQUIRED_TABLES:
        column_types = {
            column["name"]: str(column["type"]).upper()
            for column in inspect(database.engine).get_columns(table_name)
        }
        assert all("BLOB" not in column_type for column_type in column_types.values())
    database.close()


def test_artifact_store_is_atomic_checksum_verified_and_traversal_safe(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    metadata = store.put_text("skills/wipe/compiled_skill.py", "async def run(runtime): pass\n")

    assert len(metadata.checksum_sha256) == 64
    assert store.read_bytes(
        metadata.uri, expected_checksum_sha256=metadata.checksum_sha256
    ).startswith(b"async def")
    assert store.put_text(metadata.uri, "async def run(runtime): pass\n") == metadata
    with pytest.raises(FileExistsError):
        store.put_text(metadata.uri, "different")
    with pytest.raises(ValueError):
        store.put_text("../escape.txt", "unsafe")
    with pytest.raises(ValueError):
        store.put_text("skills\\escape.txt", "unsafe")


def test_skill_registry_and_execution_events_are_transactional(tmp_path: Path) -> None:
    database = Database.from_path(tmp_path / "registry.sqlite3")
    database.create_schema()
    repository = StorageRepository(database)
    scene = repository.record_scene(_scene())
    version = repository.register_skill_version(
        name="wipe surface",
        intent="wipe_surface",
        semantic_version="1.0.0",
        graph={"skill_id": "wipe", "nodes": [{"operation": "motion.wait"}]},
        status="active",
        validation_status="passed",
    )

    run = repository.start_execution(
        started_at_ns=1_000,
        execution_mode="mock",
        skill_version_id=version.id,
        scene_id=scene.id,
    )
    repository.append_execution_event(
        run.id, timestamp_ns=1_010, event_type="primitive_started"
    )
    repository.append_execution_event(
        run.id, timestamp_ns=1_020, event_type="primitive_completed"
    )
    repository.finish_execution(run.id, status="succeeded", ended_at_ns=1_030)

    with database.session() as session:
        events = list(
            session.scalars(
                select(ExecutionEventRecord)
                .where(ExecutionEventRecord.execution_run_id == run.id)
                .order_by(ExecutionEventRecord.sequence)
            )
        )
    assert [event.sequence for event in events] == [1, 2]
    assert repository.active_skill_version(name="wipe surface") is not None
    assert [item.id for item in repository.search_active_skills_keyword("wipe")] == [version.id]
    database.close()


def test_active_skill_must_be_validated(tmp_path: Path) -> None:
    database = Database.from_path(tmp_path / "registry.sqlite3")
    database.create_schema()
    repository = StorageRepository(database)

    with pytest.raises(ValueError, match="validated"):
        repository.register_skill_version(
            name="unsafe",
            intent="unsafe",
            semantic_version="1.0.0",
            graph={"nodes": []},
            status="active",
            validation_status="pending",
        )

    assert repository.active_skill_version(name="unsafe") is None
    database.close()


def test_teaching_session_and_skill_lifecycle_helpers(tmp_path: Path) -> None:
    database = Database.from_path(tmp_path / "registry.sqlite3")
    database.create_schema()
    repository = StorageRepository(database)
    teaching = repository.create_teaching_session(started_at_ns=100, metadata={"mode": "mock"})
    finalized = repository.finalize_teaching_session(
        teaching.id,
        ended_at_ns=200,
        status="completed",
        artifact_uri="demonstrations/session/metadata.json",
        artifact_checksum_sha256="a" * 64,
    )
    assert finalized.status == "completed"
    assert repository.get_teaching_session(teaching.id) is not None

    active = repository.register_skill_version(
        name="versioned wipe",
        intent="wipe_surface",
        semantic_version="1.0.0",
        graph={"version": "1.0.0"},
        status="active",
        validation_status="passed",
    )
    candidate = repository.register_skill_version(
        name="versioned wipe",
        intent="wipe_surface",
        semantic_version="1.1.0-candidate",
        graph={"version": "1.1.0-candidate"},
        status="candidate",
        validation_status="passed",
        parent_version_id=active.id,
    )
    with pytest.raises(ValueError, match="stabilized"):
        repository.activate_skill_version(candidate.id)
    promoted = repository.register_skill_version(
        name="versioned wipe",
        intent="wipe_surface",
        semantic_version="1.1.0",
        graph={"version": "1.1.0"},
        status="validated",
        validation_status="passed",
        parent_version_id=candidate.id,
    )
    repository.activate_skill_version(promoted.id)
    current = repository.active_skill_version(name="versioned wipe")
    assert current is not None and current.id == promoted.id
    repository.rollback_skill(skill_id=active.skill_id, target_version_id=active.id)
    current = repository.active_skill_version(name="versioned wipe")
    assert current is not None and current.id == active.id
    assert len(repository.list_skill_versions(skill_id=active.skill_id)) == 3
    database.close()
