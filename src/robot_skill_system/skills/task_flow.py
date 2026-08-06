"""Deterministic Grip -> Action -> End motion composition.

This module is intentionally local and schema-driven.  It never asks a model to
produce geometry, profile numbers, graph topology, or an end-motion choice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from robot_skill_system.exceptions import SkillGraphValidationError
from robot_skill_system.primitives.models import validate_profile_id
from robot_skill_system.primitives.registry import PrimitiveRegistry, get_default_registry
from robot_skill_system.scene.models import StrictModel, Vector3
from robot_skill_system.scene.transforms import AnchorType, RelativePose
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.graph import NodeTransitions, SkillGraphValidator
from robot_skill_system.skills.models import (
    BindingSpec,
    EntityKind,
    SkillEdge,
    SkillGraph,
    SkillLifecycleStatus,
    SkillNode,
    SkillType,
    TransitionCondition,
    ValidationStatus,
)

COMPOSER_VERSION = "1.0.0"

_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_CATALOG_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_MAPPING_ID_PATTERN = re.compile(
    r"^(?:[a-z][a-z0-9_.-]*|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_PLACEHOLDER_PATTERN = re.compile(r"^\$[A-Za-z][A-Za-z0-9_]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_KEY_PATTERN = re.compile(
    r"^(?:motion|force|gripper|verification|policy):[a-z][a-z0-9_.-]*$"
)


class TaskFlowCompositionError(SkillGraphValidationError):
    """Raised when independently valid components cannot be safely composed."""


def _canonical_checksum(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_catalog_id(value: str) -> str:
    if not _CATALOG_ID_PATTERN.fullmatch(value):
        raise ValueError("catalog ids must be stable lower-case identifiers")
    return value


def _validate_mapping_id(value: str) -> str:
    """Accept stable catalog IDs and database-issued RFC 4122 UUIDs only."""

    if not _MAPPING_ID_PATTERN.fullmatch(value):
        raise ValueError("mapping_id must be a stable catalog ID or lower-case UUID")
    return value


def _validate_semver(value: str) -> str:
    if not _SEMVER_PATTERN.fullmatch(value):
        raise ValueError("version must be semantic version syntax")
    return value


def _validate_sha256(value: str | None) -> str | None:
    if value is not None and not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("checksum must be a lower-case SHA-256 digest")
    return value


def _validate_placeholder(value: str) -> str:
    if not _PLACEHOLDER_PATTERN.fullmatch(value):
        raise ValueError("binding references must use $name syntax")
    return value


def _validate_axis(value: Vector3) -> Vector3:
    if not math.isclose(value.norm(), 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("grip axes must be normalized")
    return value


def _validate_revisions(value: dict[str, str]) -> dict[str, str]:
    for key, revision in value.items():
        if not _REVISION_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"unsupported profile/policy revision key {key!r}")
        if not revision or len(revision) > 128:
            raise ValueError("profile/policy revisions must be stable non-empty labels")
    return value


class AttachmentState(str, Enum):
    """Object attachment state at a component boundary."""

    ANY = "any"
    HOLDING = "holding"
    RELEASED = "released"


class ForceModeState(str, Enum):
    """Force-supervisor state at a component boundary."""

    ANY = "any"
    DISABLED = "disabled"
    ENABLED = "enabled"


class StageStateContract(StrictModel):
    """Small, explicit state contract between independently learned stages."""

    attachment: AttachmentState
    force_mode: ForceModeState


class GripProfile(StrictModel):
    """Executable, object-relative grasp geometry selected by object class."""

    profile_id: str
    version: str
    object_class_id: str
    object_frame_policy: str = Field(min_length=1, max_length=120)
    object_frame_revision: str = Field(min_length=1, max_length=120)
    object_binding: str = "$object"
    tool_binding: str = "$gripper"
    required_tool_class: str | None = Field(default=None, min_length=1, max_length=120)
    pregrasp_pose: RelativePose
    grasp_pose: RelativePose
    jaw_axis: Vector3
    approach_axis: Vector3
    motion_profile_id: str
    gripper_calibration_profile_id: str
    verification_profile_id: str
    lifecycle_status: SkillLifecycleStatus = SkillLifecycleStatus.CANDIDATE
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    hardware_compatible: bool = False
    source_artifact_uri: str | None = None
    source_artifact_checksum_sha256: str | None = None
    profile_checksum_sha256: str | None = None
    profile_revisions: dict[str, str] = Field(default_factory=dict)
    policy_revisions: dict[str, str] = Field(default_factory=dict)

    _validate_profile_id = field_validator("profile_id", "object_class_id")(
        _validate_catalog_id
    )
    _validate_version = field_validator("version")(_validate_semver)
    _validate_object_binding = field_validator("object_binding", "tool_binding")(
        _validate_placeholder
    )
    _validate_profiles = field_validator(
        "motion_profile_id",
        "gripper_calibration_profile_id",
        "verification_profile_id",
    )(validate_profile_id)
    _validate_axes = field_validator("jaw_axis", "approach_axis")(_validate_axis)
    _validate_checksums = field_validator(
        "source_artifact_checksum_sha256", "profile_checksum_sha256"
    )(_validate_sha256)
    _validate_profile_revisions = field_validator("profile_revisions", "policy_revisions")(
        _validate_revisions
    )

    @model_validator(mode="after")
    def validate_executable_geometry(self) -> GripProfile:
        if self.object_binding == self.tool_binding:
            raise ValueError("object and gripper bindings must be distinct")
        for label, pose in (
            ("pregrasp_pose", self.pregrasp_pose),
            ("grasp_pose", self.grasp_pose),
        ):
            if pose.anchor_id != self.object_binding or pose.anchor_type is not AnchorType.OBJECT:
                raise ValueError(f"{label} must be relative to the object binding")
        dot = sum(
            left * right
            for left, right in zip(
                self.jaw_axis.as_tuple(), self.approach_axis.as_tuple(), strict=True
            )
        )
        if abs(dot) > 0.25:
            raise ValueError("jaw_axis and approach_axis must be materially independent")
        approach_distance_m = math.sqrt(
            sum(
                (grasp - pregrasp) ** 2
                for grasp, pregrasp in zip(
                    self.grasp_pose.position_m.as_tuple(),
                    self.pregrasp_pose.position_m.as_tuple(),
                    strict=True,
                )
            )
        )
        if approach_distance_m <= 1e-5:
            raise ValueError("pregrasp_pose and grasp_pose must define a non-zero approach")
        if (self.source_artifact_uri is None) != (
            self.source_artifact_checksum_sha256 is None
        ):
            raise ValueError("source artifact URI and checksum must be supplied together")
        return self

    def computed_checksum_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"profile_checksum_sha256"})
        return _canonical_checksum(payload)


def _validate_required_roles(
    required_roles: dict[str, str], graph: SkillGraph
) -> None:
    for placeholder, role in required_roles.items():
        _validate_placeholder(placeholder)
        if not role:
            raise ValueError("required role values cannot be empty")
        binding = graph.bindings.get(placeholder)
        if binding is None:
            raise ValueError(f"required role {placeholder!r} has no graph binding")
        if binding.role is not None and binding.role != role:
            raise ValueError(f"required role {placeholder!r} conflicts with its binding")


class ActionDefinition(StrictModel):
    """Canonical action and its exact versioned SkillGraph."""

    action_id: str
    version: str
    aliases: list[str] = Field(default_factory=list)
    required_roles: dict[str, str] = Field(default_factory=dict)
    input_contract: StageStateContract
    output_contract: StageStateContract
    skill_graph: SkillGraph
    graph_checksum_sha256: str | None = None
    lifecycle_status: SkillLifecycleStatus = SkillLifecycleStatus.CANDIDATE
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    hardware_compatible: bool = False
    profile_revisions: dict[str, str] = Field(default_factory=dict)
    policy_revisions: dict[str, str] = Field(default_factory=dict)

    _validate_id = field_validator("action_id")(_validate_catalog_id)
    _validate_version = field_validator("version")(_validate_semver)
    _validate_checksum = field_validator("graph_checksum_sha256")(_validate_sha256)
    _validate_revisions = field_validator("profile_revisions", "policy_revisions")(
        _validate_revisions
    )

    @model_validator(mode="after")
    def validate_definition(self) -> ActionDefinition:
        if self.skill_graph.version != self.version:
            raise ValueError("action version must equal its SkillGraph version")
        _validate_required_roles(self.required_roles, self.skill_graph)
        return self


class EndMotionDefinition(StrictModel):
    """Action-mapped release, retreat, and safe terminal graph."""

    end_motion_id: str
    version: str
    required_roles: dict[str, str] = Field(default_factory=dict)
    input_contract: StageStateContract
    output_contract: StageStateContract
    skill_graph: SkillGraph
    graph_checksum_sha256: str | None = None
    lifecycle_status: SkillLifecycleStatus = SkillLifecycleStatus.CANDIDATE
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    hardware_compatible: bool = False
    profile_revisions: dict[str, str] = Field(default_factory=dict)
    policy_revisions: dict[str, str] = Field(default_factory=dict)

    _validate_id = field_validator("end_motion_id")(_validate_catalog_id)
    _validate_version = field_validator("version")(_validate_semver)
    _validate_checksum = field_validator("graph_checksum_sha256")(_validate_sha256)
    _validate_revisions = field_validator("profile_revisions", "policy_revisions")(
        _validate_revisions
    )

    @model_validator(mode="after")
    def validate_definition(self) -> EndMotionDefinition:
        if self.skill_graph.version != self.version:
            raise ValueError("end-motion version must equal its SkillGraph version")
        _validate_required_roles(self.required_roles, self.skill_graph)
        return self


class ActionEndMotionMapping(StrictModel):
    """One active, revisioned action-to-end-motion selection."""

    mapping_id: str
    revision: int = Field(ge=1)
    action_id: str
    action_version: str
    end_motion_id: str
    end_motion_version: str
    lifecycle_status: SkillLifecycleStatus = SkillLifecycleStatus.ACTIVE

    _validate_mapping_id = field_validator("mapping_id")(_validate_mapping_id)
    _validate_ids = field_validator("action_id", "end_motion_id")(_validate_catalog_id)
    _validate_versions = field_validator("action_version", "end_motion_version")(
        _validate_semver
    )


class TaskFlowComponentReference(StrictModel):
    component_type: str
    component_id: str
    version: str
    checksum_sha256: str

    _validate_checksum = field_validator("checksum_sha256")(_validate_sha256)


class TaskFlowManifest(StrictModel):
    """Exact immutable selection and materialized graph evidence for one flow."""

    schema_version: str = "1.0"
    plan_checksum_sha256: str
    composer_version: str
    mapping_id: str
    mapping_revision: int = Field(ge=1)
    grip: TaskFlowComponentReference
    action: TaskFlowComponentReference
    end_motion: TaskFlowComponentReference
    composite_skill_id: str
    composite_skill_version: str
    composite_graph_checksum_sha256: str
    hardware_compatible: bool

    _validate_checksums = field_validator(
        "plan_checksum_sha256", "composite_graph_checksum_sha256"
    )(_validate_sha256)
    _validate_versions = field_validator("composer_version", "composite_skill_version")(
        _validate_semver
    )


class ComposedTaskFlow(StrictModel):
    graph: SkillGraph
    manifest: TaskFlowManifest


def _component_graph_checksum(
    label: str, graph: SkillGraph, declared_checksum: str | None
) -> str:
    actual = SkillCompiler.graph_checksum(graph)
    if declared_checksum is not None and declared_checksum != actual:
        raise TaskFlowCompositionError(
            f"{label} graph checksum mismatch: expected={declared_checksum}, actual={actual}"
        )
    return actual


def _assert_runtime_component(label: str, component: Any) -> None:
    lifecycle = component.lifecycle_status
    validation = component.validation_status
    if lifecycle is not SkillLifecycleStatus.ACTIVE:
        raise TaskFlowCompositionError(f"{label} component is not active")
    if validation is not ValidationStatus.PASSED:
        raise TaskFlowCompositionError(f"{label} component has not passed validation")


def _merge_revision_maps(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for values in maps:
        for key, revision in values.items():
            existing = merged.get(key)
            if existing is not None and existing != revision:
                raise TaskFlowCompositionError(
                    f"component profile/policy revision conflict for {key!r}: "
                    f"{existing!r} != {revision!r}"
                )
            merged[key] = revision
    return merged


def _merge_binding(left: BindingSpec, right: BindingSpec) -> BindingSpec:
    if left.entity_kind is not right.entity_kind:
        raise TaskFlowCompositionError(
            f"binding {left.variable!r} has conflicting entity kinds"
        )

    def compatible_value(name: str) -> Any:
        first = getattr(left, name)
        second = getattr(right, name)
        if first is not None and second is not None and first != second:
            raise TaskFlowCompositionError(
                f"binding {left.variable!r} has conflicting {name} constraints"
            )
        return first if first is not None else second

    return BindingSpec(
        variable=left.variable,
        entity_kind=left.entity_kind,
        instance_id=compatible_value("instance_id"),
        class_name=compatible_value("class_name"),
        role=compatible_value("role"),
        minimum_confidence=max(left.minimum_confidence, right.minimum_confidence),
        minimum_visible_fraction=max(
            left.minimum_visible_fraction, right.minimum_visible_fraction
        ),
        must_be_attached=compatible_value("must_be_attached"),
        compatible_skill=compatible_value("compatible_skill"),
    )


def _merge_bindings(*binding_maps: dict[str, BindingSpec]) -> dict[str, BindingSpec]:
    merged: dict[str, BindingSpec] = {}
    for values in binding_maps:
        for placeholder, binding in values.items():
            existing = merged.get(placeholder)
            merged[placeholder] = (
                binding if existing is None else _merge_binding(existing, binding)
            )
    return {key: merged[key] for key in sorted(merged)}


def _merge_roles(*role_maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for roles in role_maps:
        for placeholder, role in roles.items():
            existing = merged.get(placeholder)
            if existing is not None and existing != role:
                raise TaskFlowCompositionError(
                    f"required role conflict for {placeholder!r}: {existing!r} != {role!r}"
                )
            merged[placeholder] = role
    return {key: merged[key] for key in sorted(merged)}


def _contract_accepts(
    produced: StageStateContract, required: StageStateContract, *, boundary: str
) -> None:
    for field_name in ("attachment", "force_mode"):
        actual = getattr(produced, field_name)
        expected = getattr(required, field_name)
        if expected.value == "any":
            continue
        if actual.value == "any" or actual != expected:
            raise TaskFlowCompositionError(
                f"{boundary} contract conflict for {field_name}: "
                f"produced={actual.value}, required={expected.value}"
            )


def _namespace_graph(graph: SkillGraph, prefix: str) -> tuple[list[SkillNode], list[SkillEdge]]:
    def namespace(node_id: str | None) -> str | None:
        return None if node_id is None else f"{prefix}__{node_id}"

    nodes: list[SkillNode] = []
    for node in graph.nodes:
        checkpoint: str | bool | None = node.checkpoint
        if isinstance(checkpoint, str):
            checkpoint = f"{prefix}__{checkpoint}"
        arguments = dict(node.arguments)
        if node.operation == "recovery.return_checkpoint":
            checkpoint_id = arguments.get("checkpoint_id")
            if isinstance(checkpoint_id, str):
                arguments["checkpoint_id"] = f"{prefix}__{checkpoint_id}"
        nodes.append(
            node.model_copy(
                update={
                    "node_id": namespace(node.node_id),
                    "arguments": arguments,
                    "checkpoint": checkpoint,
                    "on_success": namespace(node.on_success),
                    "on_failure": namespace(node.on_failure),
                }
            )
        )
    edges = [
        edge.model_copy(
            update={
                "source_node": namespace(edge.source_node),
                "target_node": namespace(edge.target_node),
            }
        )
        for edge in graph.edges
    ]
    return nodes, edges


class TaskFlowComposer:
    """Compose validated components into one deterministic executable SkillGraph."""

    def __init__(
        self,
        registry: PrimitiveRegistry | None = None,
        *,
        composer_version: str = COMPOSER_VERSION,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.validator = SkillGraphValidator(self.registry)
        self.compiler = SkillCompiler(self.registry)
        self.composer_version = _validate_semver(composer_version)
        self._cache: dict[str, ComposedTaskFlow] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def compose(
        self,
        *,
        grip_profile: GripProfile,
        action: ActionDefinition,
        end_motion: EndMotionDefinition,
        mapping: ActionEndMotionMapping,
    ) -> ComposedTaskFlow:
        """Validate, pin, namespace, connect, and checksum one task flow."""

        self._validate_selection(grip_profile, action, end_motion, mapping)
        action_report = self.validator.validate(action.skill_graph)
        end_report = self.validator.validate(end_motion.skill_graph)
        del action_report
        action_checksum = _component_graph_checksum(
            "action", action.skill_graph, action.graph_checksum_sha256
        )
        end_checksum = _component_graph_checksum(
            "end motion", end_motion.skill_graph, end_motion.graph_checksum_sha256
        )
        grip_checksum = grip_profile.computed_checksum_sha256()
        if (
            grip_profile.profile_checksum_sha256 is not None
            and grip_profile.profile_checksum_sha256 != grip_checksum
        ):
            raise TaskFlowCompositionError(
                "grip profile checksum mismatch: "
                f"expected={grip_profile.profile_checksum_sha256}, actual={grip_checksum}"
            )

        self._validate_end_motion(end_motion, end_report.transitions)
        _merge_revision_maps(
            grip_profile.profile_revisions,
            action.profile_revisions,
            end_motion.profile_revisions,
            grip_profile.policy_revisions,
            action.policy_revisions,
            end_motion.policy_revisions,
        )
        seed = {
            "composer_version": self.composer_version,
            "mapping_id": mapping.mapping_id,
            "mapping_revision": mapping.revision,
            "grip": {
                "id": grip_profile.profile_id,
                "version": grip_profile.version,
                "checksum": grip_checksum,
            },
            "action": {
                "id": action.action_id,
                "version": action.version,
                "checksum": action_checksum,
                "compatibility": self._stage_compatibility_fingerprint(action),
            },
            "end_motion": {
                "id": end_motion.end_motion_id,
                "version": end_motion.version,
                "checksum": end_checksum,
                "compatibility": self._stage_compatibility_fingerprint(end_motion),
            },
        }
        plan_checksum = _canonical_checksum(seed)
        cached = self._cache.get(plan_checksum)
        if cached is not None:
            return cached.model_copy(deep=True)

        graph = self._build_graph(
            plan_checksum=plan_checksum,
            grip_profile=grip_profile,
            action=action,
            end_motion=end_motion,
        )
        self.validator.validate(graph)
        # Compile in memory as part of composition so the cached plan has passed
        # the same AST-only compiler and code validator used by persisted skills.
        self.compiler.compile_source(graph)
        graph_checksum = SkillCompiler.graph_checksum(graph)
        manifest = TaskFlowManifest(
            plan_checksum_sha256=plan_checksum,
            composer_version=self.composer_version,
            mapping_id=mapping.mapping_id,
            mapping_revision=mapping.revision,
            grip=TaskFlowComponentReference(
                component_type="grip",
                component_id=grip_profile.profile_id,
                version=grip_profile.version,
                checksum_sha256=grip_checksum,
            ),
            action=TaskFlowComponentReference(
                component_type="action",
                component_id=action.action_id,
                version=action.version,
                checksum_sha256=action_checksum,
            ),
            end_motion=TaskFlowComponentReference(
                component_type="end_motion",
                component_id=end_motion.end_motion_id,
                version=end_motion.version,
                checksum_sha256=end_checksum,
            ),
            composite_skill_id=graph.skill_id,
            composite_skill_version=graph.version,
            composite_graph_checksum_sha256=graph_checksum,
            hardware_compatible=(
                grip_profile.hardware_compatible
                and action.hardware_compatible
                and end_motion.hardware_compatible
            ),
        )
        result = ComposedTaskFlow(graph=graph, manifest=manifest)
        self._cache[plan_checksum] = result.model_copy(deep=True)
        return result

    def _validate_selection(
        self,
        grip: GripProfile,
        action: ActionDefinition,
        end: EndMotionDefinition,
        mapping: ActionEndMotionMapping,
    ) -> None:
        _assert_runtime_component("grip", grip)
        _assert_runtime_component("action", action)
        _assert_runtime_component("end motion", end)
        if mapping.lifecycle_status is not SkillLifecycleStatus.ACTIVE:
            raise TaskFlowCompositionError("action/end-motion mapping is not active")
        if (mapping.action_id, mapping.action_version) != (action.action_id, action.version):
            raise TaskFlowCompositionError("mapping does not pin the selected action version")
        if (mapping.end_motion_id, mapping.end_motion_version) != (
            end.end_motion_id,
            end.version,
        ):
            raise TaskFlowCompositionError("mapping does not pin the selected end-motion version")

        grip_output = StageStateContract(
            attachment=AttachmentState.HOLDING,
            force_mode=ForceModeState.DISABLED,
        )
        _contract_accepts(grip_output, action.input_contract, boundary="grip -> action")
        _contract_accepts(action.output_contract, end.input_contract, boundary="action -> end")
        if action.output_contract.force_mode is not ForceModeState.DISABLED:
            raise TaskFlowCompositionError(
                "action output contract must guarantee disabled force mode before end motion"
            )
        if end.output_contract.attachment is not AttachmentState.RELEASED:
            raise TaskFlowCompositionError("end motion must guarantee object release")
        if end.output_contract.force_mode is not ForceModeState.DISABLED:
            raise TaskFlowCompositionError("end motion must terminate with force mode disabled")

        recovery_policies = {
            action.skill_graph.recovery_policy,
            end.skill_graph.recovery_policy,
            "global_safe_stop",
        }
        if len(recovery_policies) != 1:
            raise TaskFlowCompositionError("component recovery policies conflict")

    @staticmethod
    def _validate_end_motion(
        end_motion: EndMotionDefinition, transitions: Mapping[str, NodeTransitions]
    ) -> None:
        node_by_id = {node.node_id: node for node in end_motion.skill_graph.nodes}
        current: str | None = end_motion.skill_graph.start_node
        opened = False
        release_verified = False
        while current is not None:
            operation = node_by_id[current].operation
            if operation == "gripper.open":
                opened = True
            elif operation == "grasp.verify_released":
                if not opened:
                    raise TaskFlowCompositionError(
                        "end motion must verify release after opening the gripper"
                    )
                release_verified = True
            current = transitions[current].success
        if not opened or not release_verified:
            raise TaskFlowCompositionError(
                "end motion success path requires gripper.open followed by "
                "grasp.verify_released"
            )

    @staticmethod
    def _stage_compatibility_fingerprint(
        stage: ActionDefinition | EndMotionDefinition,
    ) -> str:
        return _canonical_checksum(
            {
                "required_roles": stage.required_roles,
                "input_contract": stage.input_contract.model_dump(mode="json"),
                "output_contract": stage.output_contract.model_dump(mode="json"),
                "profile_revisions": stage.profile_revisions,
                "policy_revisions": stage.policy_revisions,
            }
        )

    def _build_graph(
        self,
        *,
        plan_checksum: str,
        grip_profile: GripProfile,
        action: ActionDefinition,
        end_motion: EndMotionDefinition,
    ) -> SkillGraph:
        grip_nodes = self._materialize_grip(grip_profile)
        grip_edges = [
            SkillEdge(source_node=left.node_id, target_node=right.node_id)
            for left, right in zip(grip_nodes, grip_nodes[1:], strict=False)
        ]
        action_nodes, action_edges = _namespace_graph(action.skill_graph, "action")
        end_nodes, end_edges = _namespace_graph(end_motion.skill_graph, "end")

        action_start = f"action__{action.skill_graph.start_node}"
        end_start = f"end__{end_motion.skill_graph.start_node}"
        connector_edges = [
            SkillEdge(
                source_node=grip_nodes[-1].node_id,
                target_node=action_start,
                condition=TransitionCondition.SUCCESS,
            )
        ]
        connector_edges.extend(
            SkillEdge(
                source_node=f"action__{terminal}",
                target_node=end_start,
                condition=TransitionCondition.SUCCESS,
            )
            for terminal in action.skill_graph.terminal_nodes
        )

        grip_bindings = {
            grip_profile.object_binding: BindingSpec(
                variable=grip_profile.object_binding,
                entity_kind=EntityKind.OBJECT,
                class_name=grip_profile.object_class_id,
            ),
            grip_profile.tool_binding: BindingSpec(
                variable=grip_profile.tool_binding,
                entity_kind=EntityKind.TOOL,
                class_name=grip_profile.required_tool_class,
                must_be_attached=True,
            ),
        }
        bindings = _merge_bindings(
            grip_bindings, action.skill_graph.bindings, end_motion.skill_graph.bindings
        )
        required_roles = _merge_roles(
            action.skill_graph.required_entity_roles,
            action.required_roles,
            end_motion.skill_graph.required_entity_roles,
            end_motion.required_roles,
        )
        for placeholder, role in required_roles.items():
            binding = bindings.get(placeholder)
            if binding is None:
                raise TaskFlowCompositionError(
                    f"required role {placeholder!r} is absent from merged bindings"
                )
            if binding.role is not None and binding.role != role:
                raise TaskFlowCompositionError(
                    f"required role {placeholder!r} conflicts with merged binding"
                )

        global_requirements = sorted(
            set(action.skill_graph.global_policy_requirements)
            | set(end_motion.skill_graph.global_policy_requirements)
            | {
                "global_safety_supervisor",
                "workspace_monitor",
                "force_supervisor",
                "emergency_stop_monitor",
            }
        )
        sources = list(
            dict.fromkeys(
                action.skill_graph.source_demonstrations
                + end_motion.skill_graph.source_demonstrations
            )
        )
        return SkillGraph(
            skill_id=f"task_flow:{plan_checksum[:24]}",
            version="1.0.0",
            name=f"{grip_profile.object_class_id} / {action.action_id} task flow",
            description="Deterministic Grip -> Action -> End motion composite.",
            skill_type=SkillType.COMPOSITE,
            source_demonstrations=sources,
            required_tools=list(
                dict.fromkeys(
                    ([grip_profile.required_tool_class] if grip_profile.required_tool_class else [])
                    + action.skill_graph.required_tools
                    + end_motion.skill_graph.required_tools
                )
            ),
            required_entity_roles=required_roles,
            bindings=bindings,
            nodes=grip_nodes + action_nodes + end_nodes,
            edges=grip_edges + action_edges + end_edges + connector_edges,
            start_node=grip_nodes[0].node_id,
            terminal_nodes=[f"end__{node_id}" for node_id in end_motion.skill_graph.terminal_nodes],
            motion_profiles=list(
                dict.fromkeys(
                    [grip_profile.motion_profile_id]
                    + action.skill_graph.motion_profiles
                    + end_motion.skill_graph.motion_profiles
                )
            ),
            force_profiles=list(
                dict.fromkeys(
                    action.skill_graph.force_profiles + end_motion.skill_graph.force_profiles
                )
            ),
            preconditions=list(
                dict.fromkeys(
                    ["grip_profile_active"]
                    + action.skill_graph.preconditions
                    + end_motion.skill_graph.preconditions
                )
            ),
            postconditions=list(
                dict.fromkeys(end_motion.skill_graph.postconditions + ["object_released"])
            ),
            recovery_policy="global_safe_stop",
            global_policy_requirements=global_requirements,
            uncertainty={
                "task_flow_plan_checksum_sha256": plan_checksum,
                "grip_hardware_compatible": grip_profile.hardware_compatible,
            },
            validation_status=ValidationStatus.PASSED,
            lifecycle_status=SkillLifecycleStatus.ACTIVE,
        )

    @staticmethod
    def _materialize_grip(profile: GripProfile) -> list[SkillNode]:
        tool = profile.tool_binding
        object_binding = profile.object_binding
        motion_profile = profile.motion_profile_id
        verification_profile = profile.verification_profile_id
        return [
            SkillNode(
                node_id="grip__open",
                operation="gripper.open",
                arguments={"tool": tool},
            ),
            SkillNode(
                node_id="grip__validate_pregrasp",
                operation="workspace.validate_target",
                arguments={"target": profile.pregrasp_pose.model_dump(mode="json")},
            ),
            SkillNode(
                node_id="grip__move_pregrasp",
                operation="motion.move_l",
                arguments={
                    "target": profile.pregrasp_pose.model_dump(mode="json"),
                    "motion_profile_id": motion_profile,
                },
            ),
            SkillNode(
                node_id="grip__validate_grasp",
                operation="workspace.validate_target",
                arguments={"target": profile.grasp_pose.model_dump(mode="json")},
            ),
            SkillNode(
                node_id="grip__move_grasp",
                operation="motion.move_l",
                arguments={
                    "target": profile.grasp_pose.model_dump(mode="json"),
                    "motion_profile_id": motion_profile,
                },
            ),
            SkillNode(
                node_id="grip__close",
                operation="gripper.close",
                arguments={"tool": tool},
            ),
            SkillNode(
                node_id="grip__verify_holding",
                operation="grasp.verify_holding",
                arguments={
                    "object": object_binding,
                    "tool": tool,
                    "verification_profile_id": verification_profile,
                },
            ),
        ]


def compose_task_flow(
    *,
    grip_profile: GripProfile,
    action: ActionDefinition,
    end_motion: EndMotionDefinition,
    mapping: ActionEndMotionMapping,
    registry: PrimitiveRegistry | None = None,
) -> ComposedTaskFlow:
    """Stateless convenience wrapper for one deterministic composition."""

    return TaskFlowComposer(registry).compose(
        grip_profile=grip_profile,
        action=action,
        end_motion=end_motion,
        mapping=mapping,
    )


__all__ = [
    "COMPOSER_VERSION",
    "ActionDefinition",
    "ActionEndMotionMapping",
    "AttachmentState",
    "ComposedTaskFlow",
    "EndMotionDefinition",
    "ForceModeState",
    "GripProfile",
    "StageStateContract",
    "TaskFlowComponentReference",
    "TaskFlowComposer",
    "TaskFlowCompositionError",
    "TaskFlowManifest",
    "compose_task_flow",
]
