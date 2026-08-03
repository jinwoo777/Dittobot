"""SQLAlchemy schema for durable metadata.

Large payloads are intentionally represented by URI/checksum pairs. The database never owns
camera frames, point clouds, generated source files, or execution-log files.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative metadata root."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow, nullable=False)


class OperatorRecord(TimestampMixin, Base):
    __tablename__ = "operators"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="operator")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    demonstrations: Mapped[list[DemonstrationRecord]] = relationship(back_populates="operator")


class TeachingSessionRecord(TimestampMixin, Base):
    __tablename__ = "teaching_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    operator_id: Mapped[str | None] = mapped_column(ForeignKey("operators.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    started_at_ns: Mapped[int] = mapped_column(Integer, nullable=False)
    ended_at_ns: Mapped[int | None] = mapped_column(Integer)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ObjectDefinitionRecord(TimestampMixin, Base):
    __tablename__ = "object_definitions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    class_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    revision: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    definition_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    source_checksum_sha256: Mapped[str | None] = mapped_column(String(64))


class ToolDefinitionRecord(TimestampMixin, Base):
    __tablename__ = "tool_definitions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tool_class: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    revision: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    definition_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    source_checksum_sha256: Mapped[str | None] = mapped_column(String(64))


class WorkspaceDefinitionRecord(TimestampMixin, Base):
    __tablename__ = "workspace_definitions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    role: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    revision: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    definition_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    source_checksum_sha256: Mapped[str | None] = mapped_column(String(64))


class SceneRecord(TimestampMixin, Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp_ns: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    reference_frame: Mapped[str] = mapped_column(String(128), nullable=False)
    valid_for_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    calibration_id: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    scene_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    occupancy_map_uri: Mapped[str | None] = mapped_column(Text)
    occupancy_map_checksum_sha256: Mapped[str | None] = mapped_column(String(64))

    objects: Mapped[list[ObjectRecord]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )
    tools: Mapped[list[ToolRecord]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )
    workspaces: Mapped[list[WorkspaceRecord]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )


class ObjectRecord(TimestampMixin, Base):
    __tablename__ = "objects"
    __table_args__ = (UniqueConstraint("scene_id", "instance_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), nullable=False, index=True)
    instance_id: Mapped[str] = mapped_column(String(128), nullable=False)
    class_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    visible_fraction: Mapped[float | None] = mapped_column(Float)
    object_definition_id: Mapped[str | None] = mapped_column(String(128))
    data_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    scene: Mapped[SceneRecord] = relationship(back_populates="objects")


class ToolRecord(TimestampMixin, Base):
    __tablename__ = "tools"
    __table_args__ = (UniqueConstraint("scene_id", "instance_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), nullable=False, index=True)
    instance_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_class: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    attached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    data_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    scene: Mapped[SceneRecord] = relationship(back_populates="tools")


class WorkspaceRecord(TimestampMixin, Base):
    __tablename__ = "workspaces"
    __table_args__ = (UniqueConstraint("scene_id", "region_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), nullable=False, index=True)
    region_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    frame_id: Mapped[str] = mapped_column(String(128), nullable=False)
    minimum_clearance_m: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    data_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    scene: Mapped[SceneRecord] = relationship(back_populates="workspaces")


class DemonstrationRecord(TimestampMixin, Base):
    __tablename__ = "demonstrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    operator_id: Mapped[str | None] = mapped_column(ForeignKey("operators.id"), index=True)
    initial_scene_id: Mapped[str | None] = mapped_column(ForeignKey("scenes.id"))
    final_scene_id: Mapped[str | None] = mapped_column(ForeignKey("scenes.id"))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="recorded")
    success: Mapped[bool | None] = mapped_column(Boolean)
    notes: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    operator: Mapped[OperatorRecord | None] = relationship(back_populates="demonstrations")


class SkillRecord(TimestampMixin, Base):
    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("name", "variant"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    intent: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    variant: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    active_version_id: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    versions: Mapped[list[SkillVersionRecord]] = relationship(
        back_populates="skill",
        cascade="all, delete-orphan",
        foreign_keys="SkillVersionRecord.skill_id",
    )


class SkillVariantRecord(TimestampMixin, Base):
    __tablename__ = "skill_variants"
    __table_args__ = (UniqueConstraint("skill_id", "name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    skill_id: Mapped[str] = mapped_column(ForeignKey("skills.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    route_signature: Mapped[str | None] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class SkillVersionRecord(TimestampMixin, Base):
    __tablename__ = "skill_versions"
    __table_args__ = (UniqueConstraint("skill_id", "semantic_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    skill_id: Mapped[str] = mapped_column(ForeignKey("skills.id"), nullable=False, index=True)
    semantic_version: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_version_id: Mapped[str | None] = mapped_column(ForeignKey("skill_versions.id"))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft", index=True)
    graph_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    graph_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_code_uri: Mapped[str | None] = mapped_column(Text)
    generated_code_checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    hardware_compatible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    skill: Mapped[SkillRecord] = relationship(back_populates="versions", foreign_keys=[skill_id])
    parent: Mapped[SkillVersionRecord | None] = relationship(
        remote_side="SkillVersionRecord.id", foreign_keys=[parent_version_id]
    )


class SkillEmbeddingRecord(TimestampMixin, Base):
    __tablename__ = "skill_embeddings"
    __table_args__ = (UniqueConstraint("skill_version_id", "model", "purpose"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    skill_version_id: Mapped[str] = mapped_column(
        ForeignKey("skill_versions.id"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False, default="retrieval")
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_json: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    source_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ValidationRunRecord(TimestampMixin, Base):
    __tablename__ = "validation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    skill_version_id: Mapped[str] = mapped_column(
        ForeignKey("skill_versions.id"), nullable=False, index=True
    )
    validator: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    is_mock: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_checksum_sha256: Mapped[str | None] = mapped_column(String(64))


class MotionProfileRecord(TimestampMixin, Base):
    __tablename__ = "motion_profiles"

    profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    revision: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ForceProfileRecord(TimestampMixin, Base):
    __tablename__ = "force_profiles"

    profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    revision: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ExecutionRunRecord(TimestampMixin, Base):
    __tablename__ = "execution_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    skill_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("skill_versions.id"), index=True
    )
    scene_id: Mapped[str | None] = mapped_column(ForeignKey("scenes.id"), index=True)
    command_text: Mapped[str | None] = mapped_column(Text)
    execution_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created", index=True)
    started_at_ns: Mapped[int] = mapped_column(Integer, nullable=False)
    ended_at_ns: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    preflight_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    bindings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    log_artifact_uri: Mapped[str | None] = mapped_column(Text)
    log_artifact_checksum_sha256: Mapped[str | None] = mapped_column(String(64))

    events: Mapped[list[ExecutionEventRecord]] = relationship(
        back_populates="execution_run",
        cascade="all, delete-orphan",
        order_by="ExecutionEventRecord.sequence",
    )


class ExecutionEventRecord(Base):
    __tablename__ = "execution_events"
    __table_args__ = (
        UniqueConstraint("execution_run_id", "sequence"),
        Index("ix_execution_events_run_time", "execution_run_id", "timestamp_ns"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    execution_run_id: Mapped[str] = mapped_column(
        ForeignKey("execution_runs.id"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ns: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    execution_run: Mapped[ExecutionRunRecord] = relationship(back_populates="events")
