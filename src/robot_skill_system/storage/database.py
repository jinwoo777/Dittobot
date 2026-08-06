"""Database lifecycle and small transactional repository helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from robot_skill_system.storage.orm import (
    ActionEndMappingRecord,
    Base,
    ExecutionEventRecord,
    ExecutionRunRecord,
    GripProfileRecord,
    GripProfileVersionRecord,
    ObjectRecord,
    SceneRecord,
    SemanticCatalogRecord,
    SkillEmbeddingRecord,
    SkillRecord,
    SkillVersionRecord,
    StageDefinitionRecord,
    TaskFlowPlanRecord,
    TeachingSessionRecord,
    ToolRecord,
    ValidationRunRecord,
    WorkspaceRecord,
)

_CATALOG_KINDS = frozenset({"object", "action", "end_motion"})
_CATALOG_STATUSES = frozenset({"draft", "active", "retired"})
_COMPONENT_STATUSES = frozenset({"candidate", "validated", "active", "retired"})
_VALIDATION_STATUSES = frozenset({"pending", "passed", "failed"})
_CANONICAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def _validate_catalog_kind(kind: str) -> str:
    normalized = kind.strip().casefold()
    if normalized not in _CATALOG_KINDS:
        raise ValueError(f"unsupported semantic catalog kind: {kind!r}")
    return normalized


def _validate_canonical_id(canonical_id: str) -> str:
    normalized = canonical_id.strip()
    if not _CANONICAL_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "canonical semantic ID must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-'"
        )
    return normalized


def _validate_checksum(checksum_sha256: str, *, field: str = "checksum") -> str:
    normalized = checksum_sha256.strip().casefold()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return normalized


def _normalize_aliases(aliases: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = alias.strip()
        if not normalized:
            raise ValueError("semantic aliases cannot be empty")
        if len(normalized) > 200:
            raise ValueError("semantic aliases cannot exceed 200 characters")
        folded = normalized.casefold()
        if folded not in seen:
            result.append(normalized)
            seen.add(folded)
    return result


def _require_status(value: str, allowed: frozenset[str], *, field: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in allowed:
        raise ValueError(f"unsupported {field}: {value!r}")
    return normalized


def _is_candidate_version(semantic_version: str) -> bool:
    return semantic_version.casefold().endswith("-candidate")


def _contains_base_absolute_anchor(value: Any) -> bool:
    """Reject policies that would persist a robot-base absolute target."""

    serialized = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False).casefold()
    forbidden = ("base_absolute", "robot_base_absolute", '"anchor_type": "robot_base"')
    return any(token in serialized for token in forbidden)


def _action_end_mapping_checksum(
    *,
    action_catalog_entry_id: str,
    action_stage: StageDefinitionRecord,
    end_stage: StageDefinitionRecord,
    revision: int,
) -> str:
    return _canonical_checksum(
        {
            "action_catalog_entry_id": action_catalog_entry_id,
            "action_stage_definition_id": action_stage.id,
            "action_checksum_sha256": action_stage.graph_checksum_sha256,
            "end_motion_stage_definition_id": end_stage.id,
            "end_motion_checksum_sha256": end_stage.graph_checksum_sha256,
            "revision": revision,
        }
    )


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

    @staticmethod
    def _catalog_entry_in_session(
        session: Session,
        *,
        kind: str,
        identifier: str,
        active_only: bool = False,
    ) -> SemanticCatalogRecord | None:
        normalized_kind = _validate_catalog_kind(kind)
        normalized_identifier = identifier.strip().casefold()
        if not normalized_identifier:
            raise ValueError("semantic catalog identifier cannot be empty")
        direct = session.get(SemanticCatalogRecord, identifier)
        if direct is not None and direct.kind == normalized_kind:
            if not active_only or direct.status == "active":
                return direct
            return None
        entries = session.scalars(
            select(SemanticCatalogRecord).where(SemanticCatalogRecord.kind == normalized_kind)
        )
        for entry in entries:
            if active_only and entry.status != "active":
                continue
            identifiers = [entry.canonical_id, *entry.aliases_json]
            if normalized_identifier in {value.casefold() for value in identifiers}:
                return entry
        return None

    @staticmethod
    def _stage_is_active_and_passed(
        session: Session, stage: StageDefinitionRecord | None
    ) -> bool:
        if stage is None or stage.status != "active" or stage.validation_status != "passed":
            return False
        linked = session.get(SkillVersionRecord, stage.skill_version_id)
        return bool(
            linked is not None
            and linked.validation_status == "passed"
            and linked.graph_checksum_sha256 == stage.graph_checksum_sha256
        )

    def create_catalog_entry(
        self,
        *,
        kind: str,
        canonical_id: str,
        display_name: str | None = None,
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SemanticCatalogRecord:
        """Create one draft semantic ID, returning an identical existing row idempotently."""

        normalized_kind = _validate_catalog_kind(kind)
        normalized_id = _validate_canonical_id(canonical_id)
        normalized_aliases = _normalize_aliases(aliases or ())
        resolved_display_name = (display_name or normalized_id).strip()
        if not resolved_display_name:
            raise ValueError("catalog display name cannot be empty")
        if len(resolved_display_name) > 200:
            raise ValueError("catalog display name cannot exceed 200 characters")
        metadata_json = _jsonable(dict(metadata or {}))
        with self.database.session() as session:
            existing = session.scalar(
                select(SemanticCatalogRecord).where(
                    SemanticCatalogRecord.kind == normalized_kind,
                    SemanticCatalogRecord.canonical_id == normalized_id,
                )
            )
            if existing is not None:
                conflicts = (
                    (display_name is not None and existing.display_name != resolved_display_name)
                    or (
                        aliases is not None
                        and {item.casefold() for item in existing.aliases_json}
                        != {item.casefold() for item in normalized_aliases}
                    )
                    or (metadata is not None and existing.metadata_json != metadata_json)
                )
                if conflicts:
                    raise ValueError(
                        f"semantic catalog entry {normalized_kind}:{normalized_id} already exists "
                        "with different content"
                    )
                return existing

            requested_identifiers = {
                normalized_id.casefold(),
                *(alias.casefold() for alias in normalized_aliases),
            }
            for other in session.scalars(
                select(SemanticCatalogRecord).where(
                    SemanticCatalogRecord.kind == normalized_kind
                )
            ):
                existing_identifiers = {
                    other.canonical_id.casefold(),
                    *(alias.casefold() for alias in other.aliases_json),
                }
                if requested_identifiers & existing_identifiers:
                    raise ValueError(
                        f"semantic identifier or alias is already owned by "
                        f"{normalized_kind}:{other.canonical_id}"
                    )
            record = SemanticCatalogRecord(
                kind=normalized_kind,
                canonical_id=normalized_id,
                display_name=resolved_display_name,
                aliases_json=normalized_aliases,
                status="draft",
                metadata_json=metadata_json,
            )
            session.add(record)
        return record

    def get_catalog_entry(
        self, *, kind: str, identifier: str, active_only: bool = False
    ) -> SemanticCatalogRecord | None:
        with self.database.session() as session:
            return self._catalog_entry_in_session(
                session, kind=kind, identifier=identifier, active_only=active_only
            )

    def list_catalog_entries(
        self, *, kind: str | None = None, status: str | None = None
    ) -> list[SemanticCatalogRecord]:
        statement = select(SemanticCatalogRecord)
        if kind is not None:
            statement = statement.where(SemanticCatalogRecord.kind == _validate_catalog_kind(kind))
        if status is not None:
            normalized_status = _require_status(
                status, _CATALOG_STATUSES, field="catalog status"
            )
            statement = statement.where(SemanticCatalogRecord.status == normalized_status)
        statement = statement.order_by(
            SemanticCatalogRecord.kind, SemanticCatalogRecord.canonical_id
        )
        with self.database.session() as session:
            return list(session.scalars(statement))

    def update_catalog_entry(
        self,
        entry_id: str,
        *,
        display_name: str | None = None,
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SemanticCatalogRecord:
        """Update mutable labels only; canonical IDs and component history remain immutable."""

        with self.database.session() as session:
            record = session.get(SemanticCatalogRecord, entry_id)
            if record is None:
                raise KeyError(entry_id)
            if display_name is not None:
                normalized_name = display_name.strip()
                if not normalized_name or len(normalized_name) > 200:
                    raise ValueError("catalog display name must contain 1 to 200 characters")
                record.display_name = normalized_name
            if aliases is not None:
                normalized_aliases = _normalize_aliases(aliases)
                requested = {
                    record.canonical_id.casefold(),
                    *(alias.casefold() for alias in normalized_aliases),
                }
                for other in session.scalars(
                    select(SemanticCatalogRecord).where(
                        SemanticCatalogRecord.kind == record.kind,
                        SemanticCatalogRecord.id != record.id,
                    )
                ):
                    owned = {
                        other.canonical_id.casefold(),
                        *(alias.casefold() for alias in other.aliases_json),
                    }
                    if requested & owned:
                        raise ValueError(
                            f"semantic identifier or alias is already owned by "
                            f"{record.kind}:{other.canonical_id}"
                        )
                record.aliases_json = normalized_aliases
            if metadata is not None:
                record.metadata_json = _jsonable(dict(metadata))
            result = record
        return result

    def set_catalog_entry_status(
        self, entry_id: str, *, status: str
    ) -> SemanticCatalogRecord:
        normalized_status = _require_status(status, _CATALOG_STATUSES, field="catalog status")
        with self.database.session() as session:
            record = session.get(SemanticCatalogRecord, entry_id)
            if record is None:
                raise KeyError(entry_id)
            if record.status == "active" and normalized_status == "draft":
                raise ValueError("an active catalog entry may only be retired")
            if normalized_status == "active":
                if record.kind == "object":
                    profile = session.scalar(
                        select(GripProfileRecord).where(
                            GripProfileRecord.object_catalog_entry_id == record.id
                        )
                    )
                    version = (
                        session.get(GripProfileVersionRecord, profile.active_version_id)
                        if profile is not None and profile.active_version_id is not None
                        else None
                    )
                    if (
                        version is None
                        or version.status != "active"
                        or version.validation_status != "passed"
                    ):
                        raise ValueError(
                            "an object catalog entry requires an active, passed grip profile"
                        )
                else:
                    stage = (
                        session.get(StageDefinitionRecord, record.active_stage_definition_id)
                        if record.active_stage_definition_id is not None
                        else None
                    )
                    if not self._stage_is_active_and_passed(session, stage):
                        raise ValueError(
                            f"a {record.kind} catalog entry requires an active, passed stage"
                        )
                    if record.kind == "action":
                        mappings = list(
                            session.scalars(
                                select(ActionEndMappingRecord).where(
                                    ActionEndMappingRecord.action_catalog_entry_id == record.id,
                                    ActionEndMappingRecord.status == "active",
                                )
                            )
                        )
                        if (
                            len(mappings) != 1
                            or stage is None
                            or mappings[0].action_stage_definition_id != stage.id
                        ):
                            raise ValueError(
                                "an action catalog entry requires exactly one active end mapping "
                                "for its active stage"
                            )
                record.status = "active"
            else:
                record.status = normalized_status
                if record.kind == "end_motion" and normalized_status == "retired":
                    stage_ids = list(
                        session.scalars(
                            select(StageDefinitionRecord.id).where(
                                StageDefinitionRecord.catalog_entry_id == record.id
                            )
                        )
                    )
                    if stage_ids:
                        for mapping in session.scalars(
                            select(ActionEndMappingRecord).where(
                                ActionEndMappingRecord.end_motion_stage_definition_id.in_(
                                    stage_ids
                                ),
                                ActionEndMappingRecord.status == "active",
                            )
                        ):
                            mapping.status = "retired"
                            action = session.get(
                                SemanticCatalogRecord, mapping.action_catalog_entry_id
                            )
                            if action is not None and action.status == "active":
                                action.status = "draft"
            result = record
        return result

    def activate_catalog_entry(self, entry_id: str) -> SemanticCatalogRecord:
        return self.set_catalog_entry_status(entry_id, status="active")

    def retire_catalog_entry(self, entry_id: str) -> SemanticCatalogRecord:
        return self.set_catalog_entry_status(entry_id, status="retired")

    def ensure_grip_profile(
        self,
        *,
        object_class_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> GripProfileRecord:
        """Return/create the single default profile identity for an object class."""

        with self.database.session() as session:
            object_entry = self._catalog_entry_in_session(
                session, kind="object", identifier=object_class_id
            )
            if object_entry is None:
                raise KeyError(object_class_id)
            profile = session.scalar(
                select(GripProfileRecord).where(
                    GripProfileRecord.object_catalog_entry_id == object_entry.id
                )
            )
            if profile is not None:
                if metadata is not None and profile.metadata_json != _jsonable(dict(metadata)):
                    raise ValueError(
                        f"grip profile for {object_entry.canonical_id!r} already exists with "
                        "different metadata"
                    )
                return profile
            profile = GripProfileRecord(
                object_catalog_entry_id=object_entry.id,
                metadata_json=_jsonable(dict(metadata or {})),
            )
            session.add(profile)
        return profile

    def register_grip_profile_version(
        self,
        *,
        object_class_id: str,
        semantic_version: str,
        artifact_uri: str,
        artifact_checksum_sha256: str,
        object_frame_policy: str,
        object_frame_revision: str,
        status: str = "candidate",
        validation_status: str = "pending",
        hardware_compatible: bool = False,
        auto_activation_allowed: bool = False,
        gripper_calibration_profile_id: str | None = None,
        parent_version_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GripProfileVersionRecord:
        """Register an immutable grip version that references an external artifact."""

        normalized_status = _require_status(status, _COMPONENT_STATUSES, field="grip status")
        normalized_validation = _require_status(
            validation_status, _VALIDATION_STATUSES, field="grip validation status"
        )
        normalized_checksum = _validate_checksum(
            artifact_checksum_sha256, field="grip artifact checksum"
        )
        semantic_version = semantic_version.strip()
        if not semantic_version or len(semantic_version) > 64:
            raise ValueError("grip semantic version must contain 1 to 64 characters")
        artifact_uri = artifact_uri.strip()
        if not artifact_uri:
            raise ValueError("grip artifact URI cannot be empty")
        object_frame_policy = object_frame_policy.strip()
        object_frame_revision = object_frame_revision.strip()
        if not object_frame_policy or not object_frame_revision:
            raise ValueError("object frame policy and revision are required")
        if _contains_base_absolute_anchor({"object_frame_policy": object_frame_policy}):
            raise ValueError("robot-base absolute grip targets cannot be persisted")
        if hardware_compatible and object_frame_policy != "object_relative_6d":
            raise ValueError("hardware-compatible grips require object_relative_6d frame policy")
        if hardware_compatible and not gripper_calibration_profile_id:
            raise ValueError("hardware-compatible grips require a gripper calibration profile")

        metadata_json = _jsonable(dict(metadata or {}))
        with self.database.session() as session:
            object_entry = self._catalog_entry_in_session(
                session, kind="object", identifier=object_class_id
            )
            if object_entry is None:
                raise KeyError(object_class_id)
            profile = session.scalar(
                select(GripProfileRecord).where(
                    GripProfileRecord.object_catalog_entry_id == object_entry.id
                )
            )
            if profile is None:
                profile = GripProfileRecord(object_catalog_entry_id=object_entry.id)
                session.add(profile)
                session.flush()
            if parent_version_id is not None:
                parent = session.get(GripProfileVersionRecord, parent_version_id)
                if parent is None or parent.grip_profile_id != profile.id:
                    raise ValueError("grip parent version must belong to the same object profile")
            existing = session.scalar(
                select(GripProfileVersionRecord).where(
                    GripProfileVersionRecord.grip_profile_id == profile.id,
                    GripProfileVersionRecord.semantic_version == semantic_version,
                )
            )
            if existing is not None:
                immutable_fields_match = (
                    existing.artifact_uri == artifact_uri
                    and existing.artifact_checksum_sha256 == normalized_checksum
                    and existing.object_frame_policy == object_frame_policy
                    and existing.object_frame_revision == object_frame_revision
                    and existing.hardware_compatible is hardware_compatible
                    and existing.auto_activation_allowed is auto_activation_allowed
                    and existing.gripper_calibration_profile_id
                    == gripper_calibration_profile_id
                    and existing.parent_version_id == parent_version_id
                    and existing.metadata_json == metadata_json
                )
                if not immutable_fields_match:
                    raise ValueError(
                        f"grip version {semantic_version!r} already exists with different content"
                    )
                return existing
            record = GripProfileVersionRecord(
                grip_profile_id=profile.id,
                semantic_version=semantic_version,
                parent_version_id=parent_version_id,
                status=normalized_status,
                validation_status=normalized_validation,
                hardware_compatible=hardware_compatible,
                auto_activation_allowed=auto_activation_allowed,
                object_frame_policy=object_frame_policy,
                object_frame_revision=object_frame_revision,
                gripper_calibration_profile_id=gripper_calibration_profile_id,
                artifact_uri=artifact_uri,
                artifact_checksum_sha256=normalized_checksum,
                metadata_json=metadata_json,
            )
            session.add(record)
            session.flush()
            if normalized_status == "active":
                self._activate_grip_version_in_session(session, record, automatic=False)
            result = record
        return result

    @staticmethod
    def _activate_grip_version_in_session(
        session: Session,
        version: GripProfileVersionRecord,
        *,
        automatic: bool,
    ) -> None:
        if version.validation_status != "passed":
            raise ValueError("only a passed grip profile version may become active")
        if _is_candidate_version(version.semantic_version):
            raise ValueError("candidate grip versions must be stabilized before activation")
        if version.status not in {"validated", "retired", "active"}:
            raise ValueError("only validated or previously active grip versions can activate")
        if automatic and not version.auto_activation_allowed:
            raise ValueError("this grip profile version forbids automatic activation")
        profile = session.get(GripProfileRecord, version.grip_profile_id)
        if profile is None:
            raise KeyError(version.grip_profile_id)
        if profile.active_version_id is not None and profile.active_version_id != version.id:
            previous = session.get(GripProfileVersionRecord, profile.active_version_id)
            if previous is not None:
                previous.status = "retired"
        version.status = "active"
        profile.active_version_id = version.id

    def set_grip_profile_validation(
        self,
        version_id: str,
        *,
        validation_status: str,
    ) -> GripProfileVersionRecord:
        normalized_validation = _require_status(
            validation_status, _VALIDATION_STATUSES, field="grip validation status"
        )
        with self.database.session() as session:
            version = session.get(GripProfileVersionRecord, version_id)
            if version is None:
                raise KeyError(version_id)
            if version.status in {"active", "retired"} and (
                version.validation_status != normalized_validation
            ):
                raise ValueError("validation evidence for an activated grip version is immutable")
            version.validation_status = normalized_validation
            version.status = "validated" if normalized_validation == "passed" else "candidate"
            result = version
        return result

    def activate_grip_profile_version(
        self, version_id: str, *, automatic: bool = False
    ) -> GripProfileVersionRecord:
        with self.database.session() as session:
            version = session.get(GripProfileVersionRecord, version_id)
            if version is None:
                raise KeyError(version_id)
            self._activate_grip_version_in_session(session, version, automatic=automatic)
            result = version
        return result

    def get_grip_profile_version(self, version_id: str) -> GripProfileVersionRecord | None:
        with self.database.session() as session:
            return session.get(GripProfileVersionRecord, version_id)

    def list_grip_profile_versions(
        self, *, object_class_id: str
    ) -> list[GripProfileVersionRecord]:
        with self.database.session() as session:
            entry = self._catalog_entry_in_session(
                session, kind="object", identifier=object_class_id
            )
            if entry is None:
                return []
            statement = (
                select(GripProfileVersionRecord)
                .join(GripProfileRecord)
                .where(GripProfileRecord.object_catalog_entry_id == entry.id)
                .order_by(
                    GripProfileVersionRecord.created_at, GripProfileVersionRecord.id
                )
            )
            return list(session.scalars(statement))

    def active_grip_profile_version(
        self,
        *,
        object_class_id: str,
        require_hardware_compatible: bool = False,
    ) -> GripProfileVersionRecord | None:
        with self.database.session() as session:
            entry = self._catalog_entry_in_session(
                session, kind="object", identifier=object_class_id, active_only=True
            )
            if entry is None:
                return None
            profile = session.scalar(
                select(GripProfileRecord).where(
                    GripProfileRecord.object_catalog_entry_id == entry.id
                )
            )
            version = (
                session.get(GripProfileVersionRecord, profile.active_version_id)
                if profile is not None and profile.active_version_id is not None
                else None
            )
            if (
                version is None
                or version.status != "active"
                or version.validation_status != "passed"
                or (require_hardware_compatible and not version.hardware_compatible)
            ):
                return None
            return version

    def register_stage_definition(
        self,
        *,
        kind: str,
        canonical_id: str,
        skill_version_id: str,
        semantic_version: str | None = None,
        status: str = "candidate",
        validation_status: str | None = None,
        required_roles: Sequence[str] = (),
        input_contract: Mapping[str, Any] | None = None,
        output_contract: Mapping[str, Any] | None = None,
        anchor_policy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageDefinitionRecord:
        """Pin one semantic Action/EndMotion definition to an exact SkillGraph version."""

        normalized_kind = _validate_catalog_kind(kind)
        if normalized_kind not in {"action", "end_motion"}:
            raise ValueError("stage definitions may only be action or end_motion entries")
        normalized_status = _require_status(status, _COMPONENT_STATUSES, field="stage status")
        normalized_roles = _normalize_aliases(required_roles)
        if any(not role.startswith("$") for role in normalized_roles):
            raise ValueError("required Scene roles must start with '$'")
        input_json = _jsonable(dict(input_contract or {}))
        output_json = _jsonable(dict(output_contract or {}))
        anchor_json = _jsonable(dict(anchor_policy or {}))
        if _contains_base_absolute_anchor(anchor_json):
            raise ValueError("robot-base absolute stage anchors cannot be persisted")
        metadata_json = _jsonable(dict(metadata or {}))

        with self.database.session() as session:
            entry = self._catalog_entry_in_session(
                session, kind=normalized_kind, identifier=canonical_id
            )
            if entry is None:
                raise KeyError(canonical_id)
            skill_version = session.get(SkillVersionRecord, skill_version_id)
            if skill_version is None:
                raise KeyError(skill_version_id)
            resolved_version = (semantic_version or skill_version.semantic_version).strip()
            if not resolved_version or len(resolved_version) > 64:
                raise ValueError("stage semantic version must contain 1 to 64 characters")
            resolved_validation = _require_status(
                validation_status or skill_version.validation_status,
                _VALIDATION_STATUSES,
                field="stage validation status",
            )
            if resolved_validation == "passed" and skill_version.validation_status != "passed":
                raise ValueError("a stage cannot pass before its linked SkillGraph version passes")

            existing_for_skill = session.scalar(
                select(StageDefinitionRecord).where(
                    StageDefinitionRecord.skill_version_id == skill_version_id
                )
            )
            existing_for_version = session.scalar(
                select(StageDefinitionRecord).where(
                    StageDefinitionRecord.catalog_entry_id == entry.id,
                    StageDefinitionRecord.semantic_version == resolved_version,
                )
            )
            existing = existing_for_skill or existing_for_version
            if existing is not None:
                immutable_fields_match = (
                    existing.catalog_entry_id == entry.id
                    and existing.stage_type == normalized_kind
                    and existing.semantic_version == resolved_version
                    and existing.skill_version_id == skill_version_id
                    and existing.graph_checksum_sha256
                    == skill_version.graph_checksum_sha256
                    and existing.required_roles_json == normalized_roles
                    and existing.input_contract_json == input_json
                    and existing.output_contract_json == output_json
                    and existing.anchor_policy_json == anchor_json
                    and existing.metadata_json == metadata_json
                )
                if not immutable_fields_match:
                    raise ValueError(
                        "SkillGraph or semantic stage version is already linked with different "
                        "immutable content"
                    )
                return existing

            record = StageDefinitionRecord(
                catalog_entry_id=entry.id,
                stage_type=normalized_kind,
                semantic_version=resolved_version,
                skill_version_id=skill_version.id,
                status=normalized_status,
                validation_status=resolved_validation,
                hardware_compatible=skill_version.hardware_compatible,
                graph_checksum_sha256=skill_version.graph_checksum_sha256,
                required_roles_json=normalized_roles,
                input_contract_json=input_json,
                output_contract_json=output_json,
                anchor_policy_json=anchor_json,
                metadata_json=metadata_json,
            )
            session.add(record)
            session.flush()
            if normalized_status == "active":
                self._activate_stage_in_session(session, record)
            result = record
        return result

    @classmethod
    def _activate_stage_in_session(
        cls, session: Session, stage: StageDefinitionRecord
    ) -> None:
        linked = session.get(SkillVersionRecord, stage.skill_version_id)
        if stage.validation_status != "passed" or (
            linked is None or linked.validation_status != "passed"
        ):
            raise ValueError("only a passed stage and SkillGraph version may become active")
        if linked.graph_checksum_sha256 != stage.graph_checksum_sha256:
            raise ValueError("linked SkillGraph checksum no longer matches the stage snapshot")
        if _is_candidate_version(stage.semantic_version):
            raise ValueError("candidate stage versions must be stabilized before activation")
        if stage.status not in {"validated", "retired", "active"}:
            raise ValueError("only validated or previously active stages can activate")
        entry = session.get(SemanticCatalogRecord, stage.catalog_entry_id)
        if entry is None:
            raise KeyError(stage.catalog_entry_id)
        if entry.kind != stage.stage_type:
            raise ValueError("stage type does not match its semantic catalog entry")
        if entry.active_stage_definition_id is not None and (
            entry.active_stage_definition_id != stage.id
        ):
            previous = session.get(StageDefinitionRecord, entry.active_stage_definition_id)
            if previous is not None:
                previous.status = "retired"
            if entry.kind == "action":
                for mapping in session.scalars(
                    select(ActionEndMappingRecord).where(
                        ActionEndMappingRecord.action_catalog_entry_id == entry.id,
                        ActionEndMappingRecord.status == "active",
                    )
                ):
                    mapping.status = "retired"
                if entry.status == "active":
                    entry.status = "draft"
            elif entry.kind == "end_motion" and previous is not None:
                for mapping in session.scalars(
                    select(ActionEndMappingRecord).where(
                        ActionEndMappingRecord.end_motion_stage_definition_id
                        == previous.id,
                        ActionEndMappingRecord.status == "active",
                    )
                ):
                    mapping.status = "retired"
                    mapped_action = session.get(
                        SemanticCatalogRecord, mapping.action_catalog_entry_id
                    )
                    if mapped_action is not None and mapped_action.status == "active":
                        mapped_action.status = "draft"
        stage.status = "active"
        entry.active_stage_definition_id = stage.id

    def set_stage_validation(
        self,
        stage_id: str,
        *,
        validation_status: str,
    ) -> StageDefinitionRecord:
        normalized_validation = _require_status(
            validation_status, _VALIDATION_STATUSES, field="stage validation status"
        )
        with self.database.session() as session:
            stage = session.get(StageDefinitionRecord, stage_id)
            if stage is None:
                raise KeyError(stage_id)
            linked = session.get(SkillVersionRecord, stage.skill_version_id)
            if normalized_validation == "passed" and (
                linked is None or linked.validation_status != "passed"
            ):
                raise ValueError("a stage cannot pass before its linked SkillGraph version passes")
            if stage.status in {"active", "retired"} and (
                stage.validation_status != normalized_validation
            ):
                raise ValueError("validation evidence for an activated stage is immutable")
            stage.validation_status = normalized_validation
            stage.status = "validated" if normalized_validation == "passed" else "candidate"
            result = stage
        return result

    def activate_stage_definition(self, stage_id: str) -> StageDefinitionRecord:
        with self.database.session() as session:
            stage = session.get(StageDefinitionRecord, stage_id)
            if stage is None:
                raise KeyError(stage_id)
            self._activate_stage_in_session(session, stage)
            result = stage
        return result

    def get_stage_definition(self, stage_id: str) -> StageDefinitionRecord | None:
        with self.database.session() as session:
            return session.get(StageDefinitionRecord, stage_id)

    def list_stage_definitions(
        self,
        *,
        kind: str | None = None,
        canonical_id: str | None = None,
        status: str | None = None,
    ) -> list[StageDefinitionRecord]:
        statement = select(StageDefinitionRecord).join(SemanticCatalogRecord)
        if kind is not None:
            normalized_kind = _validate_catalog_kind(kind)
            if normalized_kind not in {"action", "end_motion"}:
                return []
            statement = statement.where(StageDefinitionRecord.stage_type == normalized_kind)
        if canonical_id is not None:
            normalized_id = _validate_canonical_id(canonical_id)
            statement = statement.where(SemanticCatalogRecord.canonical_id == normalized_id)
        if status is not None:
            normalized_status = _require_status(
                status, _COMPONENT_STATUSES, field="stage status"
            )
            statement = statement.where(StageDefinitionRecord.status == normalized_status)
        statement = statement.order_by(
            SemanticCatalogRecord.canonical_id,
            StageDefinitionRecord.created_at,
            StageDefinitionRecord.id,
        )
        with self.database.session() as session:
            return list(session.scalars(statement))

    def active_stage_definition(
        self, *, kind: str, canonical_id: str
    ) -> StageDefinitionRecord | None:
        normalized_kind = _validate_catalog_kind(kind)
        if normalized_kind not in {"action", "end_motion"}:
            raise ValueError("active stage lookup requires action or end_motion kind")
        with self.database.session() as session:
            entry = self._catalog_entry_in_session(
                session,
                kind=normalized_kind,
                identifier=canonical_id,
                active_only=True,
            )
            if entry is None or entry.active_stage_definition_id is None:
                return None
            stage = session.get(StageDefinitionRecord, entry.active_stage_definition_id)
            return stage if self._stage_is_active_and_passed(session, stage) else None

    def map_action_to_end_motion(
        self,
        *,
        action_id: str,
        end_motion_id: str,
        revision: int | None = None,
        activate: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionEndMappingRecord:
        """Pin an Action's current stage to one exact EndMotion stage.

        Activating a new revision retires the prior mapping for that action. No uniqueness is
        imposed on the EndMotion side, so the same safe terminal stage can be reused.
        """

        metadata_json = _jsonable(dict(metadata or {}))
        with self.database.session() as session:
            action = self._catalog_entry_in_session(
                session, kind="action", identifier=action_id
            )
            end_motion = self._catalog_entry_in_session(
                session, kind="end_motion", identifier=end_motion_id, active_only=True
            )
            if action is None:
                raise KeyError(action_id)
            if end_motion is None:
                raise ValueError("end-motion catalog entry must be active before mapping")
            action_stage = (
                session.get(StageDefinitionRecord, action.active_stage_definition_id)
                if action.active_stage_definition_id is not None
                else None
            )
            end_stage = (
                session.get(StageDefinitionRecord, end_motion.active_stage_definition_id)
                if end_motion.active_stage_definition_id is not None
                else None
            )
            if not self._stage_is_active_and_passed(session, action_stage):
                raise ValueError("action requires an active, passed stage before mapping")
            if not self._stage_is_active_and_passed(session, end_stage):
                raise ValueError("end motion requires an active, passed stage before mapping")
            assert action_stage is not None
            assert end_stage is not None
            current_mappings = list(
                session.scalars(
                    select(ActionEndMappingRecord).where(
                        ActionEndMappingRecord.action_catalog_entry_id == action.id,
                        ActionEndMappingRecord.status == "active",
                    )
                )
            )
            if len(current_mappings) > 1:
                raise ValueError("action catalog integrity error: multiple active end mappings")
            current = current_mappings[0] if current_mappings else None
            if (
                activate
                and revision is None
                and current is not None
                and current.action_stage_definition_id == action_stage.id
                and current.end_motion_stage_definition_id == end_stage.id
            ):
                expected_current_checksum = _action_end_mapping_checksum(
                    action_catalog_entry_id=action.id,
                    action_stage=action_stage,
                    end_stage=end_stage,
                    revision=current.revision,
                )
                if current.mapping_checksum_sha256 != expected_current_checksum:
                    raise ValueError("action/end mapping checksum verification failed")
                if metadata is not None and current.metadata_json != metadata_json:
                    raise ValueError("active action/end mapping already exists with other metadata")
                return current
            if revision is None:
                maximum_revision = session.scalar(
                    select(func.max(ActionEndMappingRecord.revision)).where(
                        ActionEndMappingRecord.action_catalog_entry_id == action.id
                    )
                )
                resolved_revision = int(maximum_revision or 0) + 1
            else:
                if revision <= 0:
                    raise ValueError("mapping revision must be positive")
                resolved_revision = revision
            mapping_checksum = _action_end_mapping_checksum(
                action_catalog_entry_id=action.id,
                action_stage=action_stage,
                end_stage=end_stage,
                revision=resolved_revision,
            )
            existing = session.scalar(
                select(ActionEndMappingRecord).where(
                    ActionEndMappingRecord.action_catalog_entry_id == action.id,
                    ActionEndMappingRecord.revision == resolved_revision,
                )
            )
            if existing is not None:
                if (
                    existing.mapping_checksum_sha256 != mapping_checksum
                    or existing.metadata_json != metadata_json
                ):
                    raise ValueError(
                        f"action/end mapping revision {resolved_revision} already exists with "
                        "different content"
                    )
                if activate and existing.status != "active":
                    self._activate_action_end_mapping_in_session(session, existing)
                return existing
            record = ActionEndMappingRecord(
                action_catalog_entry_id=action.id,
                action_stage_definition_id=action_stage.id,
                end_motion_stage_definition_id=end_stage.id,
                revision=resolved_revision,
                status="active" if activate else "draft",
                mapping_checksum_sha256=mapping_checksum,
                metadata_json=metadata_json,
            )
            session.add(record)
            session.flush()
            if activate:
                self._activate_action_end_mapping_in_session(session, record)
            result = record
        return result

    @classmethod
    def _activate_action_end_mapping_in_session(
        cls, session: Session, mapping: ActionEndMappingRecord
    ) -> None:
        action = session.get(SemanticCatalogRecord, mapping.action_catalog_entry_id)
        action_stage = session.get(
            StageDefinitionRecord, mapping.action_stage_definition_id
        )
        end_stage = session.get(
            StageDefinitionRecord, mapping.end_motion_stage_definition_id
        )
        if action is None or action.kind != "action":
            raise ValueError("mapping action catalog entry is invalid")
        if action.active_stage_definition_id != mapping.action_stage_definition_id:
            raise ValueError("mapping must pin the action's current active stage")
        if not cls._stage_is_active_and_passed(session, action_stage):
            raise ValueError("mapping action stage is not active and passed")
        if not cls._stage_is_active_and_passed(session, end_stage):
            raise ValueError("mapping end-motion stage is not active and passed")
        assert action_stage is not None
        assert end_stage is not None
        expected_checksum = _action_end_mapping_checksum(
            action_catalog_entry_id=action.id,
            action_stage=action_stage,
            end_stage=end_stage,
            revision=mapping.revision,
        )
        if mapping.mapping_checksum_sha256 != expected_checksum:
            raise ValueError("action/end mapping checksum verification failed")
        end_entry = session.get(SemanticCatalogRecord, end_stage.catalog_entry_id)
        if end_entry is None or end_entry.kind != "end_motion" or end_entry.status != "active":
            raise ValueError("mapping end-motion catalog entry must be active")
        for previous in session.scalars(
            select(ActionEndMappingRecord).where(
                ActionEndMappingRecord.action_catalog_entry_id
                == mapping.action_catalog_entry_id,
                ActionEndMappingRecord.status == "active",
                ActionEndMappingRecord.id != mapping.id,
            )
        ):
            previous.status = "retired"
        mapping.status = "active"

    def activate_action_end_mapping(self, mapping_id: str) -> ActionEndMappingRecord:
        with self.database.session() as session:
            mapping = session.get(ActionEndMappingRecord, mapping_id)
            if mapping is None:
                raise KeyError(mapping_id)
            self._activate_action_end_mapping_in_session(session, mapping)
            result = mapping
        return result

    def get_action_end_mapping(
        self, *, action_id: str, active_only: bool = True
    ) -> ActionEndMappingRecord | None:
        with self.database.session() as session:
            action = self._catalog_entry_in_session(
                session, kind="action", identifier=action_id, active_only=active_only
            )
            if action is None:
                return None
            statement = select(ActionEndMappingRecord).where(
                ActionEndMappingRecord.action_catalog_entry_id == action.id
            )
            if active_only:
                statement = statement.where(ActionEndMappingRecord.status == "active")
            statement = statement.order_by(ActionEndMappingRecord.revision.desc())
            mappings = list(session.scalars(statement))
            if active_only and len(mappings) > 1:
                raise ValueError("action catalog integrity error: multiple active end mappings")
            mapping = mappings[0] if mappings else None
            if not active_only or mapping is None:
                return mapping
            action_stage = session.get(
                StageDefinitionRecord, mapping.action_stage_definition_id
            )
            end_stage = session.get(
                StageDefinitionRecord, mapping.end_motion_stage_definition_id
            )
            if (
                action.active_stage_definition_id != mapping.action_stage_definition_id
                or not self._stage_is_active_and_passed(session, action_stage)
                or not self._stage_is_active_and_passed(session, end_stage)
            ):
                return None
            assert action_stage is not None
            assert end_stage is not None
            expected_checksum = _action_end_mapping_checksum(
                action_catalog_entry_id=action.id,
                action_stage=action_stage,
                end_stage=end_stage,
                revision=mapping.revision,
            )
            if mapping.mapping_checksum_sha256 != expected_checksum:
                raise ValueError("action/end mapping checksum verification failed")
            end_entry = session.get(SemanticCatalogRecord, end_stage.catalog_entry_id)
            return mapping if end_entry is not None and end_entry.status == "active" else None

    def list_action_end_mappings(
        self, *, action_id: str | None = None, active_only: bool = False
    ) -> list[ActionEndMappingRecord]:
        statement = select(ActionEndMappingRecord).join(
            SemanticCatalogRecord,
            ActionEndMappingRecord.action_catalog_entry_id == SemanticCatalogRecord.id,
        )
        if action_id is not None:
            statement = statement.where(
                SemanticCatalogRecord.canonical_id == _validate_canonical_id(action_id)
            )
        if active_only:
            statement = statement.where(ActionEndMappingRecord.status == "active")
        statement = statement.order_by(
            SemanticCatalogRecord.canonical_id, ActionEndMappingRecord.revision
        )
        with self.database.session() as session:
            return list(session.scalars(statement))

    def record_task_flow_plan(
        self,
        *,
        grip_profile_version_id: str,
        action_end_mapping_id: str,
        composer_version: str,
        composite_skill_version_id: str | None = None,
        composite_graph_checksum_sha256: str | None = None,
        validation_status: str = "pending",
        hardware_compatible: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskFlowPlanRecord:
        """Snapshot a deterministic, fully active Grip -> Action -> End composition."""

        normalized_validation = _require_status(
            validation_status, _VALIDATION_STATUSES, field="task-flow validation status"
        )
        composer_version = composer_version.strip()
        if not composer_version or len(composer_version) > 64:
            raise ValueError("composer version must contain 1 to 64 characters")
        with self.database.session() as session:
            grip = session.get(GripProfileVersionRecord, grip_profile_version_id)
            mapping = session.get(ActionEndMappingRecord, action_end_mapping_id)
            if grip is None:
                raise KeyError(grip_profile_version_id)
            if mapping is None:
                raise KeyError(action_end_mapping_id)
            if grip.status != "active" or grip.validation_status != "passed":
                raise ValueError("task-flow plans require an active, passed grip version")
            profile = session.get(GripProfileRecord, grip.grip_profile_id)
            object_entry = (
                session.get(SemanticCatalogRecord, profile.object_catalog_entry_id)
                if profile is not None
                else None
            )
            if (
                profile is None
                or profile.active_version_id != grip.id
                or object_entry is None
                or object_entry.status != "active"
            ):
                raise ValueError("task-flow grip is not active for an active object catalog entry")
            if mapping.status != "active":
                raise ValueError("task-flow plans require an active action/end mapping")
            action_stage = session.get(
                StageDefinitionRecord, mapping.action_stage_definition_id
            )
            end_stage = session.get(
                StageDefinitionRecord, mapping.end_motion_stage_definition_id
            )
            if not self._stage_is_active_and_passed(session, action_stage):
                raise ValueError("task-flow action stage is not active and passed")
            if not self._stage_is_active_and_passed(session, end_stage):
                raise ValueError("task-flow end-motion stage is not active and passed")
            assert action_stage is not None
            assert end_stage is not None
            expected_mapping_checksum = _action_end_mapping_checksum(
                action_catalog_entry_id=mapping.action_catalog_entry_id,
                action_stage=action_stage,
                end_stage=end_stage,
                revision=mapping.revision,
            )
            if mapping.mapping_checksum_sha256 != expected_mapping_checksum:
                raise ValueError("action/end mapping checksum verification failed")
            action_entry = session.get(SemanticCatalogRecord, action_stage.catalog_entry_id)
            end_entry = session.get(SemanticCatalogRecord, end_stage.catalog_entry_id)
            if action_entry is None or action_entry.status != "active":
                raise ValueError("task-flow action catalog entry is not active")
            if end_entry is None or end_entry.status != "active":
                raise ValueError("task-flow end-motion catalog entry is not active")

            composite = (
                session.get(SkillVersionRecord, composite_skill_version_id)
                if composite_skill_version_id is not None
                else None
            )
            if composite_skill_version_id is not None and composite is None:
                raise KeyError(composite_skill_version_id)
            if composite is not None:
                expected_composite_checksum = composite.graph_checksum_sha256
                if (
                    composite_graph_checksum_sha256 is not None
                    and composite_graph_checksum_sha256 != expected_composite_checksum
                ):
                    raise ValueError("composite SkillGraph checksum does not match its version")
            elif composite_graph_checksum_sha256 is not None:
                expected_composite_checksum = _validate_checksum(
                    composite_graph_checksum_sha256,
                    field="composite graph checksum",
                )
            else:
                raise ValueError(
                    "composite skill version or composite graph checksum must be provided"
                )

            derived_hardware_compatible = bool(
                grip.hardware_compatible
                and action_stage.hardware_compatible
                and end_stage.hardware_compatible
                and (composite is None or composite.hardware_compatible)
            )
            if hardware_compatible is True and not derived_hardware_compatible:
                raise ValueError("task-flow cannot elevate component hardware compatibility")
            resolved_hardware = (
                derived_hardware_compatible
                if hardware_compatible is None
                else hardware_compatible
            )
            pinned = {
                "composer_version": composer_version,
                "mapping": {
                    "id": mapping.id,
                    "revision": mapping.revision,
                    "checksum_sha256": mapping.mapping_checksum_sha256,
                },
                "grip": {
                    "version_id": grip.id,
                    "checksum_sha256": grip.artifact_checksum_sha256,
                },
                "action": {
                    "stage_definition_id": action_stage.id,
                    "checksum_sha256": action_stage.graph_checksum_sha256,
                },
                "end_motion": {
                    "stage_definition_id": end_stage.id,
                    "checksum_sha256": end_stage.graph_checksum_sha256,
                },
                "composite": {
                    "skill_version_id": composite_skill_version_id,
                    "checksum_sha256": expected_composite_checksum,
                },
            }
            plan_checksum = _canonical_checksum(pinned)
            existing = session.scalar(
                select(TaskFlowPlanRecord).where(
                    TaskFlowPlanRecord.plan_checksum_sha256 == plan_checksum
                )
            )
            if existing is not None:
                return existing
            manifest = {
                "schema_version": "1.0",
                **pinned,
                "metadata": _jsonable(dict(metadata or {})),
            }
            record = TaskFlowPlanRecord(
                plan_checksum_sha256=plan_checksum,
                composer_version=composer_version,
                action_end_mapping_id=mapping.id,
                mapping_revision=mapping.revision,
                mapping_checksum_sha256=mapping.mapping_checksum_sha256,
                grip_profile_version_id=grip.id,
                grip_checksum_sha256=grip.artifact_checksum_sha256,
                action_stage_definition_id=action_stage.id,
                action_checksum_sha256=action_stage.graph_checksum_sha256,
                end_motion_stage_definition_id=end_stage.id,
                end_motion_checksum_sha256=end_stage.graph_checksum_sha256,
                composite_skill_version_id=composite_skill_version_id,
                composite_graph_checksum_sha256=expected_composite_checksum,
                validation_status=normalized_validation,
                hardware_compatible=resolved_hardware,
                manifest_json=manifest,
            )
            session.add(record)
        return record

    def get_task_flow_plan(self, identifier: str) -> TaskFlowPlanRecord | None:
        with self.database.session() as session:
            record = session.get(TaskFlowPlanRecord, identifier)
            if record is not None:
                return record
            if _SHA256_PATTERN.fullmatch(identifier.casefold()):
                return session.scalar(
                    select(TaskFlowPlanRecord).where(
                        TaskFlowPlanRecord.plan_checksum_sha256 == identifier.casefold()
                    )
                )
            return None

    def list_task_flow_plans(self) -> list[TaskFlowPlanRecord]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(TaskFlowPlanRecord).order_by(
                        TaskFlowPlanRecord.created_at, TaskFlowPlanRecord.id
                    )
                )
            )

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
