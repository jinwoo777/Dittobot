"""Pydantic schemas for deterministic, versioned SkillGraphs."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import (
    AliasChoices,
    Field,
    field_validator,
    model_validator,
)

from robot_skill_system.scene.models import StrictModel

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_NODE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_PLACEHOLDER_PATTERN = re.compile(r"^\$[A-Za-z][A-Za-z0-9_]*$")


class SkillType(str, Enum):
    """High-level behavior category used to constrain primitive selection."""

    MOTION = "motion"
    MANIPULATION = "manipulation"
    CONTACT = "contact"
    INSPECTION = "inspection"
    RECOVERY = "recovery"
    COMPOSITE = "composite"


class SkillLifecycleStatus(str, Enum):
    """Version lifecycle; an active version is never overwritten in place."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"


class ValidationStatus(str, Enum):
    """Local compiler/simulation validation state."""

    UNVALIDATED = "unvalidated"
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class TransitionCondition(str, Enum):
    """Conditions emitted by a primitive invocation."""

    SUCCESS = "success"
    FAILURE = "failure"
    ALWAYS = "always"


class EntityKind(str, Enum):
    """Scene collection from which a binding is resolved."""

    OBJECT = "object"
    TOOL = "tool"
    SURFACE = "surface"
    WORKSPACE = "workspace"


class BindingSpec(StrictModel):
    """Deterministic constraints for resolving one ``$placeholder``."""

    variable: str = Field(validation_alias=AliasChoices("variable", "placeholder"))
    entity_kind: EntityKind = Field(validation_alias=AliasChoices("entity_kind", "kind"))
    instance_id: str | None = None
    class_name: str | None = None
    role: str | None = None
    minimum_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    minimum_visible_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    must_be_attached: bool | None = None
    compatible_skill: str | None = None

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, value: str) -> str:
        """Require an explicit binding placeholder rather than free text."""

        if not _PLACEHOLDER_PATTERN.fullmatch(value):
            raise ValueError("binding variable must use $name syntax")
        return value

    @property
    def placeholder(self) -> str:
        """Runtime-compatible alias."""

        return self.variable


class SkillNode(StrictModel):
    """One typed invocation in a SkillGraph."""

    node_id: str
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("timeout_s", "timeout"),
    )
    checkpoint: str | bool | None = None
    on_success: str | None = None
    on_failure: str | None = None
    required_scene_freshness_ms: int | None = Field(default=None, gt=0)

    @field_validator("node_id", "on_success", "on_failure")
    @classmethod
    def validate_node_reference(cls, value: str | None) -> str | None:
        """Validate node ids without treating them as filesystem paths."""

        if value is not None and not _NODE_ID_PATTERN.fullmatch(value):
            raise ValueError("node identifiers contain unsupported characters")
        return value

    @field_validator("operation")
    @classmethod
    def validate_operation_shape(cls, value: str) -> str:
        """Validate syntax; whitelist membership is checked by the graph validator."""

        if not _OPERATION_PATTERN.fullmatch(value):
            raise ValueError("operation must use lower-case namespace.operation syntax")
        return value

    @property
    def checkpoint_id(self) -> str | None:
        """Normalize boolean checkpoint shorthand to a stable string id."""

        if self.checkpoint is True:
            return self.node_id
        if self.checkpoint is False or self.checkpoint is None:
            return None
        return self.checkpoint

    @property
    def timeout(self) -> float | None:
        """Compatibility alias for schemas that call the field ``timeout``."""

        return self.timeout_s


class SkillEdge(StrictModel):
    """A directed result-conditioned graph edge."""

    source_node: str = Field(
        validation_alias=AliasChoices("source_node", "from_node", "source", "from")
    )
    target_node: str = Field(
        validation_alias=AliasChoices("target_node", "to_node", "target", "to")
    )
    condition: TransitionCondition = TransitionCondition.SUCCESS

    @field_validator("source_node", "target_node")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        """Validate node references."""

        if not _NODE_ID_PATTERN.fullmatch(value):
            raise ValueError("edge node identifiers contain unsupported characters")
        return value

    @property
    def from_node(self) -> str:
        """Readable compatibility alias."""

        return self.source_node

    @property
    def to_node(self) -> str:
        """Readable compatibility alias."""

        return self.target_node


class SkillGraph(StrictModel):
    """Versioned, anchor-relative, locally compilable skill representation."""

    schema_version: str = "1.0"
    skill_id: str
    version: str
    parent_version: str | None = None
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=2000)
    skill_type: SkillType
    source_demonstrations: list[str] = Field(default_factory=list)
    operator_style: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    required_entity_roles: dict[str, str] = Field(default_factory=dict)
    bindings: dict[str, BindingSpec] = Field(default_factory=dict)
    nodes: list[SkillNode] = Field(min_length=1)
    edges: list[SkillEdge] = Field(default_factory=list)
    start_node: str
    terminal_nodes: list[str] = Field(min_length=1)
    motion_profiles: list[str] = Field(default_factory=list)
    force_profiles: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    recovery_policy: str = "global_safe_stop"
    global_policy_requirements: list[str] = Field(
        default_factory=lambda: [
            "global_safety_supervisor",
            "workspace_monitor",
            "force_supervisor",
            "emergency_stop_monitor",
        ]
    )
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    lifecycle_status: SkillLifecycleStatus = SkillLifecycleStatus.DRAFT

    @model_validator(mode="before")
    @classmethod
    def normalize_bindings(cls, value: Any) -> Any:
        """Accept keyed dictionaries or lists while preserving explicit variables."""

        if not isinstance(value, dict) or "bindings" not in value:
            return value
        normalized = dict(value)
        raw_bindings = normalized.get("bindings")
        if isinstance(raw_bindings, list):
            keyed: dict[str, Any] = {}
            for item in raw_bindings:
                if isinstance(item, BindingSpec):
                    keyed[item.variable] = item
                elif isinstance(item, dict):
                    variable = item.get("variable", item.get("placeholder"))
                    if not isinstance(variable, str):
                        raise ValueError("each binding list item requires a variable")
                    keyed[variable] = item
                else:
                    raise ValueError("bindings must contain objects")
            normalized["bindings"] = keyed
        elif isinstance(raw_bindings, dict):
            keyed = {}
            for variable, item in raw_bindings.items():
                if isinstance(item, BindingSpec):
                    if item.variable != variable:
                        raise ValueError("binding key must match its variable")
                    keyed[variable] = item
                elif isinstance(item, dict):
                    binding_data = dict(item)
                    binding_data.setdefault("variable", variable)
                    keyed[variable] = binding_data
                elif isinstance(item, str):
                    keyed[variable] = {
                        "variable": variable,
                        "entity_kind": item,
                    }
                else:
                    raise ValueError("binding values must be objects or entity-kind strings")
            normalized["bindings"] = keyed
        return normalized

    @field_validator("skill_id")
    @classmethod
    def validate_skill_id(cls, value: str) -> str:
        """Validate stable skill identifiers."""

        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("skill_id contains unsupported characters")
        return value

    @field_validator("version", "parent_version")
    @classmethod
    def validate_version(cls, value: str | None) -> str | None:
        """Require semantic versions, including candidate prerelease labels."""

        if value is not None and not _SEMVER_PATTERN.fullmatch(value):
            raise ValueError("version must be a semantic version such as 1.0.0")
        return value

    @field_validator("start_node")
    @classmethod
    def validate_start_node(cls, value: str) -> str:
        """Validate the start node identifier."""

        if not _NODE_ID_PATTERN.fullmatch(value):
            raise ValueError("start_node contains unsupported characters")
        return value

    @field_validator("terminal_nodes")
    @classmethod
    def validate_terminal_nodes(cls, value: list[str]) -> list[str]:
        """Reject duplicate or malformed terminal ids."""

        if len(value) != len(set(value)):
            raise ValueError("terminal_nodes must be unique")
        if any(not _NODE_ID_PATTERN.fullmatch(node_id) for node_id in value):
            raise ValueError("terminal node identifiers contain unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_local_references(self) -> SkillGraph:
        """Perform inexpensive reference checks before registry-aware validation."""

        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node_id values must be unique")
        known = set(node_ids)
        if self.start_node not in known:
            raise ValueError("start_node does not reference a graph node")
        if not set(self.terminal_nodes).issubset(known):
            raise ValueError("terminal_nodes contain an unknown node")
        if set(self.bindings) != {binding.variable for binding in self.bindings.values()}:
            raise ValueError("binding dictionary keys must match binding variables")
        if len(self.motion_profiles) != len(set(self.motion_profiles)):
            raise ValueError("motion_profiles must be unique")
        if len(self.force_profiles) != len(set(self.force_profiles)):
            raise ValueError("force_profiles must be unique")
        return self

    def node_by_id(self, node_id: str) -> SkillNode:
        """Return a node or raise ``KeyError``."""

        return next(node for node in self.nodes if node.node_id == node_id)

    def ordered_nodes(self) -> tuple[SkillNode, ...]:
        """Return registry-independent topological order after full validation."""

        from robot_skill_system.skills.graph import topological_order

        return tuple(self.node_by_id(node_id) for node_id in topological_order(self))

    def binding_requirements(self) -> tuple[BindingSpec, ...]:
        """Return binding requirements in stable placeholder order."""

        return tuple(self.bindings[key] for key in sorted(self.bindings))


# Concise aliases used by earlier design notes.
Node = SkillNode
Edge = SkillEdge
SkillStatus = SkillLifecycleStatus


class ValidationSeverity(str, Enum):
    """Severity of a structured validation finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(StrictModel):
    """One schema, graph, compiler, simulation, or safety finding."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: ValidationSeverity
    node_id: str | None = None


class ValidationReport(StrictModel):
    """Persistable validation summary associated with generated artifacts."""

    schema_version: str = "1.0"
    skill_id: str
    version: str
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    graph_checksum_sha256: str | None = None
    generated_code_checksum_sha256: str | None = None
    mock_validation: bool = True
    hardware_validated: bool = False
    timestamp_ns: int = Field(default=0, ge=0)


class SkillManifest(StrictModel):
    """URI/checksum-only manifest for artifacts stored outside SQLite."""

    schema_version: str = "1.0"
    skill_id: str
    version: str
    skill_graph_uri: str
    skill_graph_checksum_sha256: str
    compiled_skill_uri: str
    compiled_skill_checksum_sha256: str
    validation_report_uri: str | None = None
    validation_report_checksum_sha256: str | None = None
    source_demonstration_uris: list[str] = Field(default_factory=list)

    @field_validator(
        "skill_graph_checksum_sha256",
        "compiled_skill_checksum_sha256",
        "validation_report_checksum_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        """Require lower-case SHA-256 hex digests."""

        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("artifact checksum must be a lower-case SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_optional_report_pair(self) -> SkillManifest:
        """Require validation-report URI and checksum metadata together."""

        if (self.validation_report_uri is None) != (
            self.validation_report_checksum_sha256 is None
        ):
            raise ValueError("validation report URI and checksum must be supplied together")
        return self
