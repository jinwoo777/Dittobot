"""Database lifecycle and small transactional repository helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, delete, event, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from robot_skill_system.storage.orm import (
    Base,
    ExecutionEventRecord,
    ExecutionRunRecord,
    ObjectRecord,
    SceneRecord,
    SkillEmbeddingRecord,
    SkillRecord,
    SkillVariantRecord,
    SkillVersionRecord,
    TeachingSessionRecord,
    ToolRecord,
    ValidationRunRecord,
    WorkspaceRecord,
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_checksum(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Database:
    """Own an SQLAlchemy engine and short-lived transaction-scoped sessions."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(url, echo=echo, connect_args=connect_args)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine, class_=Session, expire_on_commit=False, autoflush=False
        )

    @classmethod
    def from_path(cls, path: Path, *, echo: bool = False) -> Database:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{resolved.as_posix()}", echo=echo)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()

    @staticmethod
    def _enable_sqlite_foreign_keys(connection: Any, connection_record: Any) -> None:
        del connection_record
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class StorageRepository:
    """Transactional operations shared by teaching, registry, and runtime flows."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def record_scene(self, scene: Any) -> SceneRecord:
        data = _jsonable(scene)
        scene_id = str(_get(scene, "scene_id", _get(scene, "id")))
        confidence_summary = _get(scene, "confidence_summary", {})
        confidence = _get(confidence_summary, "overall", _get(scene, "confidence", 1.0))
        occupancy_uri = _get(scene, "occupancy_map_uri")
        record = SceneRecord(
            id=scene_id,
            schema_version=str(_get(scene, "schema_version", "1.0")),
            timestamp_ns=int(_get(scene, "timestamp_ns")),
            reference_frame=str(_get(scene, "reference_frame")),
            valid_for_ms=int(_get(scene, "valid_for_ms")),
            calibration_id=str(_get(scene, "calibration_id")),
            confidence=float(confidence),
            scene_json=data,
            occupancy_map_uri=occupancy_uri,
            occupancy_map_checksum_sha256=_get(scene, "occupancy_map_checksum_sha256"),
        )
        for item in _get(scene, "objects", ()):
            record.objects.append(
                ObjectRecord(
                    instance_id=str(_get(item, "instance_id")),
                    class_name=str(_get(item, "class_name")),
                    confidence=float(_get(item, "confidence")),
                    visible_fraction=_get(item, "visible_fraction"),
                    object_definition_id=_get(item, "object_definition_id"),
                    data_json=_jsonable(item),
                )
            )
        for item in _get(scene, "tools", ()):
            record.tools.append(
                ToolRecord(
                    instance_id=str(_get(item, "instance_id")),
                    tool_class=str(_get(item, "tool_class")),
                    attached=bool(_get(item, "attached", False)),
                    verification_confidence=float(_get(item, "verification_confidence", 0.0)),
                    data_json=_jsonable(item),
                )
            )
        regions: Sequence[Any] = _get(scene, "workspace_regions", ())
        for item in regions:
            record.workspaces.append(
                WorkspaceRecord(
                    region_id=str(_get(item, "region_id")),
                    role=str(_get(item, "role")),
                    frame_id=str(_get(item, "frame_id")),
                    minimum_clearance_m=float(_get(item, "minimum_clearance_m", 0.0)),
                    confidence=float(_get(item, "confidence", 0.0)),
                    data_json=_jsonable(item),
                )
            )
        with self.database.session() as session:
            session.add(record)
        return record

    def create_teaching_session(
        self,
        *,
        started_at_ns: int,
        operator_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TeachingSessionRecord:
        record = TeachingSessionRecord(
            operator_id=operator_id,
            status="recording",
            started_at_ns=started_at_ns,
            metadata_json=_jsonable(dict(metadata or {})),
        )
        with self.database.session() as session:
            session.add(record)
        return record

    def get_teaching_session(self, session_id: str) -> TeachingSessionRecord | None:
        with self.database.session() as session:
            return session.get(TeachingSessionRecord, session_id)

    def update_teaching_session_metadata(
        self,
        session_id: str,
        *,
        metadata: Mapping[str, Any],
        status: str | None = None,
    ) -> TeachingSessionRecord:
        """Replace the small session metadata document transactionally."""

        with self.database.session() as session:
            record = session.get(TeachingSessionRecord, session_id)
            if record is None:
                raise KeyError(session_id)
            if record.ended_at_ns is not None:
                raise ValueError("teaching session is already finalized")
            record.metadata_json = _jsonable(dict(metadata))
            if status is not None:
                record.status = status
            result = record
        return result

    def finalize_teaching_session(
        self,
        session_id: str,
        *,
        ended_at_ns: int,
        status: str,
        artifact_uri: str | None = None,
        artifact_checksum_sha256: str | None = None,
    ) -> TeachingSessionRecord:
        if (artifact_uri is None) is not (artifact_checksum_sha256 is None):
            raise ValueError("teaching artifact URI and checksum must be provided together")
        with self.database.session() as session:
            record = session.get(TeachingSessionRecord, session_id)
            if record is None:
                raise KeyError(session_id)
            if record.ended_at_ns is not None:
                raise ValueError("teaching session is already finalized")
            record.status = status
            record.ended_at_ns = ended_at_ns
            record.artifact_uri = artifact_uri
            record.artifact_checksum_sha256 = artifact_checksum_sha256
            result = record
        return result

    def register_skill_version(
        self,
        *,
        name: str,
        intent: str,
        semantic_version: str,
        graph: Any,
        status: str = "draft",
        variant: str = "default",
        description: str = "",
        parent_version_id: str | None = None,
        generated_code_uri: str | None = None,
        generated_code_checksum_sha256: str | None = None,
        validation_status: str = "pending",
        hardware_compatible: bool = False,
    ) -> SkillVersionRecord:
        graph_json = _jsonable(graph)
        with self.database.session() as session:
            skill = session.scalar(
                select(SkillRecord).where(SkillRecord.name == name, SkillRecord.variant == variant)
            )
            if skill is None:
                skill = SkillRecord(
                    name=name, intent=intent, variant=variant, description=description
                )
                session.add(skill)
                session.flush()
            version = SkillVersionRecord(
                skill_id=skill.id,
                semantic_version=semantic_version,
                parent_version_id=parent_version_id,
                status=status,
                graph_json=graph_json,
                graph_checksum_sha256=_canonical_checksum(graph_json),
                generated_code_uri=generated_code_uri,
                generated_code_checksum_sha256=generated_code_checksum_sha256,
                validation_status=validation_status,
                hardware_compatible=hardware_compatible,
            )
            session.add(version)
            session.flush()
            if status == "active":
                if validation_status != "passed":
                    raise ValueError("only validated skill versions may become active")
                if semantic_version.endswith("-candidate"):
                    raise ValueError("candidate prereleases cannot become active directly")
                if skill.active_version_id is not None:
                    previous = session.get(SkillVersionRecord, skill.active_version_id)
                    if previous is not None and previous.id != version.id:
                        previous.status = "retired"
                skill.active_version_id = version.id
            result = version
        return result

    def update_skill_version_graph(
        self,
        *,
        version_id: str,
        graph: Any,
    ) -> SkillVersionRecord:
        """Update the graph of an editable skill version."""

        graph_json = _jsonable(graph)

        with self.database.session() as session:
            version = session.get(SkillVersionRecord, version_id)

            if version is None:
                raise KeyError(version_id)

            # 활성화된 버전은 직접 수정하지 않는다.
            if version.status == "active":
                raise ValueError(
                    "active skill versions cannot be edited directly"
                )

            version.graph_json = graph_json
            version.graph_checksum_sha256 = _canonical_checksum(graph_json)

            result = version

        return result

    def record_validation_run(
        self,
        *,
        skill_version_id: str,
        status: str,
        result: Mapping[str, Any],
        validator: str = "mock_runtime_regression",
        is_mock: bool = True,
        artifact_uri: str | None = None,
        artifact_checksum_sha256: str | None = None,
    ) -> ValidationRunRecord:
        """Persist immutable validation evidence for a concrete skill version."""

        if (artifact_uri is None) is not (artifact_checksum_sha256 is None):
            raise ValueError("validation artifact URI and checksum must be provided together")
        record = ValidationRunRecord(
            skill_version_id=skill_version_id,
            validator=validator,
            status=status,
            is_mock=is_mock,
            result_json=_jsonable(result),
            artifact_uri=artifact_uri,
            artifact_checksum_sha256=artifact_checksum_sha256,
        )
        with self.database.session() as session:
            if session.get(SkillVersionRecord, skill_version_id) is None:
                raise KeyError(skill_version_id)
            session.add(record)
        return record

    def active_skill_version(
        self, *, name: str, variant: str = "default"
    ) -> SkillVersionRecord | None:
        with self.database.session() as session:
            skill = session.scalar(
                select(SkillRecord).where(SkillRecord.name == name, SkillRecord.variant == variant)
            )
            if skill is None or skill.active_version_id is None:
                return None
            return session.get(SkillVersionRecord, skill.active_version_id)

    def get_skill_version(self, version_id: str) -> SkillVersionRecord | None:
        with self.database.session() as session:
            return session.get(SkillVersionRecord, version_id)

    def put_skill_embedding(
        self,
        *,
        skill_version_id: str,
        model: str,
        vector: Sequence[float],
        source_checksum_sha256: str,
        purpose: str = "retrieval",
    ) -> SkillEmbeddingRecord:
        """Persist one immutable embedding associated with versioned source text."""

        materialized = [float(value) for value in vector]
        if not materialized:
            raise ValueError("embedding vector cannot be empty")
        with self.database.session() as session:
            if session.get(SkillVersionRecord, skill_version_id) is None:
                raise KeyError(skill_version_id)
            existing = session.scalar(
                select(SkillEmbeddingRecord).where(
                    SkillEmbeddingRecord.skill_version_id == skill_version_id,
                    SkillEmbeddingRecord.model == model,
                    SkillEmbeddingRecord.purpose == purpose,
                )
            )
            if existing is not None:
                if existing.source_checksum_sha256 != source_checksum_sha256:
                    raise ValueError("embedding source checksum changed for immutable version")
                return existing
            record = SkillEmbeddingRecord(
                skill_version_id=skill_version_id,
                model=model,
                purpose=purpose,
                dimensions=len(materialized),
                embedding_json=materialized,
                source_checksum_sha256=source_checksum_sha256,
            )
            session.add(record)
            result = record
        return result

    def get_skill_embedding(
        self,
        *,
        skill_version_id: str,
        model: str,
        purpose: str = "retrieval",
    ) -> SkillEmbeddingRecord | None:
        with self.database.session() as session:
            return session.scalar(
                select(SkillEmbeddingRecord).where(
                    SkillEmbeddingRecord.skill_version_id == skill_version_id,
                    SkillEmbeddingRecord.model == model,
                    SkillEmbeddingRecord.purpose == purpose,
                )
            )

    def list_skill_versions(
        self, *, skill_id: str | None = None, name: str | None = None
    ) -> list[SkillVersionRecord]:
        if skill_id is None and name is None:
            raise ValueError("skill_id or name is required")
        statement = select(SkillVersionRecord).join(SkillRecord)
        if skill_id is not None:
            statement = statement.where(SkillVersionRecord.skill_id == skill_id)
        if name is not None:
            statement = statement.where(SkillRecord.name == name)
        statement = statement.order_by(SkillVersionRecord.created_at, SkillVersionRecord.id)
        with self.database.session() as session:
            return list(session.scalars(statement))

    def activate_skill_version(self, version_id: str) -> SkillVersionRecord:
        """Promote one validated version without overwriting graph history."""

        with self.database.session() as session:
            version = session.get(SkillVersionRecord, version_id)
            if version is None:
                raise KeyError(version_id)
            if version.validation_status != "passed":
                raise ValueError("only a passed skill version may become active")
            if version.semantic_version.endswith("-candidate"):
                raise ValueError("candidate prereleases must be stabilized before activation")
            if version.status not in {"validated", "retired", "active"}:
                raise ValueError("only validated or previously active stable versions can activate")
            skill = session.get(SkillRecord, version.skill_id)
            if skill is None:
                raise KeyError(version.skill_id)
            if skill.active_version_id is not None and skill.active_version_id != version.id:
                previous = session.get(SkillVersionRecord, skill.active_version_id)
                if previous is not None:
                    previous.status = "retired"
            version.status = "active"
            skill.active_version_id = version.id
            result = version
        return result

    def rollback_skill(self, *, skill_id: str, target_version_id: str) -> SkillVersionRecord:
        with self.database.session() as session:
            target = session.get(SkillVersionRecord, target_version_id)
            if target is None or target.skill_id != skill_id:
                raise KeyError(target_version_id)
            if target.status not in {"retired", "active"} or target.semantic_version.endswith(
                "-candidate"
            ):
                raise ValueError("rollback target must be a previously active stable version")
        return self.activate_skill_version(target_version_id)

    def delete_skill(self, skill_id: str) -> dict[str, Any]:
        """Delete a skill and all related registry data."""

        with self.database.session() as session:
            skill = session.get(SkillRecord, skill_id)

            # skill_id가 DB UUID가 아니라 skill name으로 전달된 경우
            if skill is None:
                skill = session.scalar(
                    select(SkillRecord).where(
                        SkillRecord.name == skill_id
                    )
                )

            if skill is None:
                raise KeyError(skill_id)

            # 활성 스킬은 삭제하지 않는다.
            if skill.active_version_id is not None:
                raise ValueError(
                    "active skill cannot be deleted. "
                    "Retire or deactivate the skill first."
                )

            versions = list(
                session.scalars(
                    select(SkillVersionRecord).where(
                        SkillVersionRecord.skill_id == skill.id
                    )
                )
            )

            version_ids = [version.id for version in versions]

            if version_ids:
                session.execute(
                    delete(SkillEmbeddingRecord).where(
                        SkillEmbeddingRecord.skill_version_id.in_(version_ids)
                    )
                )

                session.execute(
                    delete(ValidationRunRecord).where(
                        ValidationRunRecord.skill_version_id.in_(version_ids)
                    )
                )

            # self-reference 제거
            for version in versions:
                version.parent_version_id = None

            # 버전 삭제
            for version in versions:
                session.delete(version)

            # variant 삭제
            session.execute(
                delete(SkillVariantRecord).where(
                    SkillVariantRecord.skill_id == skill.id
                )
            )

            # skill 삭제
            session.delete(skill)

            return {
                "skill_id": skill_id,
                "deleted_versions": len(versions),
            }

    def search_active_skills_keyword(
        self, query: str, *, limit: int = 10
    ) -> list[SkillVersionRecord]:
        """Deterministic offline fallback when embeddings/OpenAI are unavailable."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        tokens = [token.casefold() for token in query.split() if token.strip()]
        if not tokens:
            return []
        predicates: list[Any] = []
        for token in tokens:
            escaped = token.replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            predicates.extend(
                (
                    SkillRecord.name.ilike(pattern, escape="\\"),
                    SkillRecord.intent.ilike(pattern, escape="\\"),
                    SkillRecord.description.ilike(pattern, escape="\\"),
                )
            )
        statement = (
            select(SkillVersionRecord)
            .join(SkillRecord, SkillVersionRecord.skill_id == SkillRecord.id)
            .where(
                SkillRecord.active_version_id == SkillVersionRecord.id,
                SkillVersionRecord.status == "active",
                SkillVersionRecord.validation_status == "passed",
                or_(*predicates),
            )
            .order_by(SkillRecord.name, SkillVersionRecord.semantic_version.desc())
            .limit(limit)
        )
        with self.database.session() as session:
            return list(session.scalars(statement))

    def start_execution(
        self,
        *,
        started_at_ns: int,
        execution_mode: str,
        execution_run_id: str | None = None,
        skill_version_id: str | None = None,
        scene_id: str | None = None,
        command_text: str | None = None,
        preflight: Mapping[str, Any] | None = None,
        bindings: Mapping[str, Any] | None = None,
    ) -> ExecutionRunRecord:
        values: dict[str, Any] = {
            "skill_version_id": skill_version_id,
            "scene_id": scene_id,
            "command_text": command_text,
            "execution_mode": execution_mode,
            "status": "running",
            "started_at_ns": started_at_ns,
            "preflight_json": dict(preflight or {}),
            "bindings_json": dict(bindings or {}),
        }
        if execution_run_id is not None:
            values["id"] = execution_run_id
        record = ExecutionRunRecord(**values)
        with self.database.session() as session:
            session.add(record)
        return record

    def append_execution_event(
        self,
        execution_run_id: str,
        *,
        timestamp_ns: int,
        event_type: str,
        details: Mapping[str, Any] | None = None,
        severity: str = "info",
    ) -> ExecutionEventRecord:
        with self.database.session() as session:
            maximum = session.scalar(
                select(func.max(ExecutionEventRecord.sequence)).where(
                    ExecutionEventRecord.execution_run_id == execution_run_id
                )
            )
            record = ExecutionEventRecord(
                execution_run_id=execution_run_id,
                sequence=int(maximum or 0) + 1,
                timestamp_ns=timestamp_ns,
                event_type=event_type,
                severity=severity,
                details_json=_jsonable(dict(details or {})),
            )
            session.add(record)
        return record

    def finish_execution(
        self,
        execution_run_id: str,
        *,
        status: str,
        ended_at_ns: int,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionRunRecord:
        with self.database.session() as session:
            record = session.get(ExecutionRunRecord, execution_run_id)
            if record is None:
                raise KeyError(execution_run_id)
            record.status = status
            record.ended_at_ns = ended_at_ns
            record.error_code = error_code
            record.error_message = error_message
            result = record
        return result
