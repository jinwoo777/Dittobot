"""Load and compose one exact, storage-backed Grip -> Action -> End flow.

The materializer is the trust boundary between mutable catalog lookups and the
pure :mod:`robot_skill_system.skills.task_flow` composer.  It only accepts
active, validation-passed component records, re-checks every persisted
checksum, and parses the external Grip artifact through a strict data schema.
It never invents geometry or selects an End motion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import Field

from robot_skill_system.exceptions import SkillGraphValidationError
from robot_skill_system.scene.models import StrictModel
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.models import SkillGraph, SkillLifecycleStatus
from robot_skill_system.skills.task_flow import (
    ActionDefinition,
    ActionEndMotionMapping,
    ComposedTaskFlow,
    EndMotionDefinition,
    GripProfile,
    StageStateContract,
    TaskFlowComposer,
    TaskFlowManifest,
)
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import StorageRepository
from robot_skill_system.storage.orm import (
    ActionEndMappingRecord,
    GripProfileVersionRecord,
    SemanticCatalogRecord,
    SkillVersionRecord,
    StageDefinitionRecord,
)

_ROLE_PATTERN = re.compile(r"^\$[A-Za-z][A-Za-z0-9_]*$")
_ENTITY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_FORBIDDEN_FRAME_IDS = frozenset(
    {"base", "base_link", "robot_base", "robot_base_link", "world", "map"}
)
_FORBIDDEN_KEY_PARTS = (
    "absolute_pose",
    "base_pose",
    "base_target",
    "target_in_base",
    "robot_base",
)


class TaskFlowMaterializationError(SkillGraphValidationError):
    """Raised when persisted component evidence cannot safely be composed."""


class ExecutableGripProfileArtifact(StrictModel):
    """The only external Grip artifact shape accepted for execution."""

    schema_version: Literal["grip-profile/1.0"] = "grip-profile/1.0"
    grip_profile: GripProfile
    provenance: dict[str, Any] = Field(default_factory=dict)


class TaskFlowStorageSelection(StrictModel):
    """Exact database and artifact identities used by one composition."""

    object_catalog_entry_id: str
    grip_profile_version_id: str
    grip_artifact_uri: str
    grip_artifact_checksum_sha256: str
    action_catalog_entry_id: str
    action_stage_definition_id: str
    action_skill_version_id: str
    action_graph_checksum_sha256: str
    end_motion_catalog_entry_id: str
    end_motion_stage_definition_id: str
    end_motion_skill_version_id: str
    end_motion_graph_checksum_sha256: str
    action_end_mapping_record_id: str
    action_end_mapping_revision: int = Field(ge=1)
    action_end_mapping_checksum_sha256: str


class MaterializedTaskFlow(StrictModel):
    """Composed graph plus the exact storage snapshot used to create it."""

    composed: ComposedTaskFlow
    selection: TaskFlowStorageSelection
    resolved_role_bindings: dict[str, str] = Field(default_factory=dict)

    @property
    def graph(self) -> SkillGraph:
        return self.composed.graph

    @property
    def manifest(self) -> TaskFlowManifest:
        return self.composed.manifest


def _canonical_checksum(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(value: str) -> Any:
    raise TaskFlowMaterializationError(
        f"Grip artifact contains a non-finite JSON number: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskFlowMaterializationError(
                f"Grip artifact contains a duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _absolute_reference_path(value: Any, path: str = "value") -> str | None:
    """Return the first persisted base/world reference in arbitrary JSON data."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            item_path = f"{path}.{raw_key}"
            if any(fragment in key for fragment in _FORBIDDEN_KEY_PARTS):
                return item_path
            if (
                key in {"frame_id", "anchor_id", "anchor_frame_id"}
                and isinstance(item, str)
                and item.casefold() in _FORBIDDEN_FRAME_IDS
            ):
                return item_path
            if key == "anchor_type" and isinstance(item, str) and item.casefold() in {
                "base",
                "robot_base",
                "world",
                "base_absolute",
                "robot_base_absolute",
            }:
                return item_path
            nested = _absolute_reference_path(item, item_path)
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            nested = _absolute_reference_path(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _require_relative_policy(stage: StageDefinitionRecord) -> None:
    path = _absolute_reference_path(
        stage.anchor_policy_json, f"{stage.stage_type}.anchor_policy"
    )
    if path is not None:
        raise TaskFlowMaterializationError(
            f"{stage.stage_type} persists a robot-base/world absolute target at {path}"
        )


def _mapping_checksum(
    mapping: ActionEndMappingRecord,
    action: StageDefinitionRecord,
    end_motion: StageDefinitionRecord,
) -> str:
    return _canonical_checksum(
        {
            "action_catalog_entry_id": mapping.action_catalog_entry_id,
            "action_stage_definition_id": action.id,
            "action_checksum_sha256": action.graph_checksum_sha256,
            "end_motion_stage_definition_id": end_motion.id,
            "end_motion_checksum_sha256": end_motion.graph_checksum_sha256,
            "revision": mapping.revision,
        }
    )


class TaskFlowMaterializer:
    """Resolve active records and invoke the deterministic local composer."""

    def __init__(
        self,
        repository: StorageRepository,
        artifact_store: LocalArtifactStore,
        *,
        composer: TaskFlowComposer | None = None,
        maximum_grip_artifact_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if maximum_grip_artifact_bytes <= 0:
            raise ValueError("maximum_grip_artifact_bytes must be positive")
        self.repository = repository
        self.artifact_store = artifact_store
        self.composer = composer or TaskFlowComposer()
        self.maximum_grip_artifact_bytes = maximum_grip_artifact_bytes

    def materialize(
        self,
        *,
        object_class_id: str,
        action_id: str,
        resolved_role_bindings: Mapping[str, str] | None = None,
        require_hardware_compatible: bool = False,
    ) -> MaterializedTaskFlow:
        """Materialize an exact active flow or fail closed before composition.

        ``resolved_role_bindings`` contains the current Scene entity IDs chosen
        for stage role placeholders (for example ``{"$destination": "bin_01"}``).
        Geometry binding remains the responsibility of the runtime binder.
        """

        object_entry = self._active_catalog("object", object_class_id)
        action_entry = self._active_catalog("action", action_id)
        grip_record = self.repository.active_grip_profile_version(
            object_class_id=object_entry.canonical_id,
            require_hardware_compatible=require_hardware_compatible,
        )
        if grip_record is None:
            suffix = " hardware-compatible" if require_hardware_compatible else ""
            raise TaskFlowMaterializationError(
                f"object {object_entry.canonical_id!r} has no active, passed{suffix} GripProfile"
            )

        try:
            mapping_record = self.repository.get_action_end_mapping(
                action_id=action_entry.canonical_id, active_only=True
            )
        except (KeyError, ValueError) as exc:
            raise TaskFlowMaterializationError(str(exc)) from exc
        if mapping_record is None:
            raise TaskFlowMaterializationError(
                f"action {action_entry.canonical_id!r} has no exact active End-motion mapping"
            )
        if mapping_record.action_catalog_entry_id != action_entry.id:
            raise TaskFlowMaterializationError(
                "active action/end mapping references another action catalog entry"
            )

        action_stage = self._required_stage(
            mapping_record.action_stage_definition_id, "action"
        )
        end_stage = self._required_stage(
            mapping_record.end_motion_stage_definition_id, "end_motion"
        )
        if action_stage.catalog_entry_id != action_entry.id:
            raise TaskFlowMaterializationError(
                "mapped Action definition does not belong to the selected action"
            )
        end_entry = self._active_catalog("end_motion", end_stage.catalog_entry_id)

        active_action_stage = self.repository.active_stage_definition(
            kind="action", canonical_id=action_entry.canonical_id
        )
        active_end_stage = self.repository.active_stage_definition(
            kind="end_motion", canonical_id=end_entry.canonical_id
        )
        if active_action_stage is None or active_action_stage.id != action_stage.id:
            raise TaskFlowMaterializationError(
                "mapping does not pin the selected action's exact active definition"
            )
        if active_end_stage is None or active_end_stage.id != end_stage.id:
            raise TaskFlowMaterializationError(
                "mapping does not pin the selected End motion's exact active definition"
            )

        expected_mapping_checksum = _mapping_checksum(
            mapping_record, action_stage, end_stage
        )
        if expected_mapping_checksum != mapping_record.mapping_checksum_sha256:
            raise TaskFlowMaterializationError(
                "action/end mapping checksum does not match its exact stage versions/revision"
            )

        grip_profile = self._load_grip_profile(
            record=grip_record,
            object_class_id=object_entry.canonical_id,
            require_hardware_compatible=require_hardware_compatible,
        )
        loaded_action, action_skill = self._load_stage_definition(
            stage=action_stage,
            catalog=action_entry,
            expected_kind="action",
        )
        action_definition = cast(ActionDefinition, loaded_action)
        loaded_end, end_skill = self._load_stage_definition(
            stage=end_stage,
            catalog=end_entry,
            expected_kind="end_motion",
        )
        end_definition = cast(EndMotionDefinition, loaded_end)

        required_placeholders = set(action_definition.required_roles) | set(
            end_definition.required_roles
        )
        role_bindings = self._validate_resolved_roles(
            resolved_role_bindings, required_placeholders
        )
        mapping = ActionEndMotionMapping(
            mapping_id=mapping_record.id,
            revision=mapping_record.revision,
            action_id=action_entry.canonical_id,
            action_version=action_stage.semantic_version,
            end_motion_id=end_entry.canonical_id,
            end_motion_version=end_stage.semantic_version,
            lifecycle_status=SkillLifecycleStatus.ACTIVE,
        )
        composed = self.composer.compose(
            grip_profile=grip_profile,
            action=action_definition,
            end_motion=end_definition,
            mapping=mapping,
        )
        selection = TaskFlowStorageSelection(
            object_catalog_entry_id=object_entry.id,
            grip_profile_version_id=grip_record.id,
            grip_artifact_uri=grip_record.artifact_uri,
            grip_artifact_checksum_sha256=grip_record.artifact_checksum_sha256,
            action_catalog_entry_id=action_entry.id,
            action_stage_definition_id=action_stage.id,
            action_skill_version_id=action_skill.id,
            action_graph_checksum_sha256=action_stage.graph_checksum_sha256,
            end_motion_catalog_entry_id=end_entry.id,
            end_motion_stage_definition_id=end_stage.id,
            end_motion_skill_version_id=end_skill.id,
            end_motion_graph_checksum_sha256=end_stage.graph_checksum_sha256,
            action_end_mapping_record_id=mapping_record.id,
            action_end_mapping_revision=mapping_record.revision,
            action_end_mapping_checksum_sha256=mapping_record.mapping_checksum_sha256,
        )
        return MaterializedTaskFlow(
            composed=composed,
            selection=selection,
            resolved_role_bindings=role_bindings,
        )

    def _active_catalog(self, kind: str, identifier: str) -> SemanticCatalogRecord:
        entry = self.repository.get_catalog_entry(
            kind=kind, identifier=identifier, active_only=True
        )
        if entry is None:
            raise TaskFlowMaterializationError(
                f"{kind} catalog entry {identifier!r} is not active"
            )
        return entry

    def _required_stage(
        self, stage_id: str, expected_kind: Literal["action", "end_motion"]
    ) -> StageDefinitionRecord:
        stage = self.repository.get_stage_definition(stage_id)
        if stage is None:
            raise TaskFlowMaterializationError(
                f"mapping references missing {expected_kind} definition {stage_id!r}"
            )
        if stage.stage_type != expected_kind:
            raise TaskFlowMaterializationError(
                f"mapped stage {stage.id!r} is not an {expected_kind} definition"
            )
        if stage.status != "active" or stage.validation_status != "passed":
            raise TaskFlowMaterializationError(
                f"mapped {expected_kind} definition is not active and validation-passed"
            )
        _require_relative_policy(stage)
        return stage

    def _load_grip_profile(
        self,
        *,
        record: GripProfileVersionRecord,
        object_class_id: str,
        require_hardware_compatible: bool,
    ) -> GripProfile:
        if record.object_frame_policy != "object_relative_6d":
            raise TaskFlowMaterializationError(
                "GripProfile artifact is diagnostic/non-executable: "
                "object_relative_6d geometry is required"
            )
        if record.metadata_json.get("diagnostic_only") is True:
            raise TaskFlowMaterializationError(
                "2D grip-point diagnostic artifacts cannot be materialized for execution"
            )
        if record.gripper_calibration_profile_id is None:
            raise TaskFlowMaterializationError(
                "executable GripProfile requires an approved gripper calibration profile"
            )
        if require_hardware_compatible and not record.hardware_compatible:
            raise TaskFlowMaterializationError(
                "GripProfile is not hardware-compatible"
            )

        path = self.artifact_store.path_for(record.artifact_uri)
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise TaskFlowMaterializationError(
                f"GripProfile artifact is unavailable: {record.artifact_uri!r}"
            ) from exc
        if size_bytes > self.maximum_grip_artifact_bytes:
            raise TaskFlowMaterializationError(
                "GripProfile artifact exceeds the configured materialization size limit"
            )
        try:
            raw = self.artifact_store.read_bytes(
                record.artifact_uri,
                expected_checksum_sha256=record.artifact_checksum_sha256,
            )
        except (OSError, ValueError) as exc:
            raise TaskFlowMaterializationError(str(exc)) from exc
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TaskFlowMaterializationError(
                "GripProfile artifact must be strict UTF-8 JSON"
            ) from exc
        if isinstance(payload, Mapping) and str(payload.get("schema_version", "")).startswith(
            "grip-point-diagnostic-profile/"
        ):
            raise TaskFlowMaterializationError(
                "2D grip-point diagnostic artifacts cannot be materialized for execution"
            )
        try:
            artifact = ExecutableGripProfileArtifact.model_validate(payload)
        except Exception as exc:
            raise TaskFlowMaterializationError(
                "GripProfile artifact does not satisfy grip-profile/1.0"
            ) from exc

        profile = artifact.grip_profile
        mismatches: list[str] = []
        if profile.object_class_id != object_class_id:
            mismatches.append("object_class_id")
        if profile.version != record.semantic_version:
            mismatches.append("semantic version")
        if profile.object_frame_policy != record.object_frame_policy:
            mismatches.append("object-frame policy")
        if profile.object_frame_revision != record.object_frame_revision:
            mismatches.append("object-frame revision")
        if (
            profile.gripper_calibration_profile_id
            != record.gripper_calibration_profile_id
        ):
            mismatches.append("gripper calibration profile")
        if profile.lifecycle_status.value != record.status:
            mismatches.append("lifecycle status")
        if profile.validation_status.value != record.validation_status:
            mismatches.append("validation status")
        if profile.hardware_compatible is not record.hardware_compatible:
            mismatches.append("hardware compatibility")
        if mismatches:
            raise TaskFlowMaterializationError(
                "GripProfile artifact disagrees with its immutable registry record: "
                + ", ".join(mismatches)
            )
        return profile

    def _load_stage_definition(
        self,
        *,
        stage: StageDefinitionRecord,
        catalog: SemanticCatalogRecord,
        expected_kind: Literal["action", "end_motion"],
    ) -> tuple[ActionDefinition | EndMotionDefinition, SkillVersionRecord]:
        skill = self.repository.get_skill_version(stage.skill_version_id)
        if skill is None:
            raise TaskFlowMaterializationError(
                f"{expected_kind} definition references a missing SkillVersion"
            )
        if skill.validation_status != "passed":
            raise TaskFlowMaterializationError(
                f"{expected_kind} SkillVersion has not passed validation"
            )
        if stage.graph_checksum_sha256 != skill.graph_checksum_sha256:
            raise TaskFlowMaterializationError(
                f"{expected_kind} stage checksum no longer matches its SkillVersion"
            )
        try:
            graph = SkillGraph.model_validate(skill.graph_json)
        except Exception as exc:
            raise TaskFlowMaterializationError(
                f"{expected_kind} SkillVersion does not contain a valid SkillGraph"
            ) from exc
        actual_checksum = SkillCompiler.graph_checksum(graph)
        if actual_checksum != stage.graph_checksum_sha256:
            raise TaskFlowMaterializationError(
                f"{expected_kind} SkillGraph checksum mismatch"
            )
        if graph.version != stage.semantic_version or graph.version != skill.semantic_version:
            raise TaskFlowMaterializationError(
                f"{expected_kind} semantic version does not exactly pin its SkillGraph"
            )
        absolute_path = _absolute_reference_path(
            graph, f"{expected_kind}.skill_graph"
        )
        if absolute_path is not None:
            raise TaskFlowMaterializationError(
                f"{expected_kind} contains a robot-base/world absolute target at "
                f"{absolute_path}"
            )
        required_roles = self._required_role_map(stage, graph)
        try:
            input_contract = StageStateContract.model_validate(
                stage.input_contract_json
            )
            output_contract = StageStateContract.model_validate(
                stage.output_contract_json
            )
        except Exception as exc:
            raise TaskFlowMaterializationError(
                f"{expected_kind} has an invalid stored state contract"
            ) from exc

        common: dict[str, Any] = {
            "version": stage.semantic_version,
            "required_roles": required_roles,
            "input_contract": input_contract,
            "output_contract": output_contract,
            "skill_graph": graph,
            "graph_checksum_sha256": stage.graph_checksum_sha256,
            "lifecycle_status": stage.status,
            "validation_status": stage.validation_status,
            "hardware_compatible": stage.hardware_compatible,
            "profile_revisions": self._revision_map(
                stage.metadata_json, "profile_revisions", expected_kind
            ),
            "policy_revisions": self._revision_map(
                stage.metadata_json, "policy_revisions", expected_kind
            ),
        }
        try:
            if expected_kind == "action":
                definition: ActionDefinition | EndMotionDefinition = ActionDefinition(
                    action_id=catalog.canonical_id,
                    aliases=list(catalog.aliases_json),
                    **common,
                )
            else:
                definition = EndMotionDefinition(
                    end_motion_id=catalog.canonical_id,
                    **common,
                )
        except Exception as exc:
            raise TaskFlowMaterializationError(
                f"stored {expected_kind} definition is inconsistent with its SkillGraph"
            ) from exc
        return definition, skill

    @staticmethod
    def _required_role_map(
        stage: StageDefinitionRecord, graph: SkillGraph
    ) -> dict[str, str]:
        stored = set(stage.required_roles_json)
        role_map: dict[str, str] = dict(graph.required_entity_roles)
        for placeholder, binding in graph.bindings.items():
            if binding.role is None:
                continue
            previous = role_map.get(placeholder)
            if previous is not None and previous != binding.role:
                raise TaskFlowMaterializationError(
                    f"binding role for {placeholder!r} conflicts with required_entity_roles"
                )
            role_map[placeholder] = binding.role
        for placeholder in stored:
            if not _ROLE_PATTERN.fullmatch(placeholder):
                raise TaskFlowMaterializationError(
                    f"stored required role {placeholder!r} is malformed"
                )
            if placeholder not in graph.bindings:
                raise TaskFlowMaterializationError(
                    f"stored required role {placeholder!r} has no SkillGraph binding"
                )
            if placeholder not in role_map:
                raise TaskFlowMaterializationError(
                    f"stored required role {placeholder!r} has no declared semantic role"
                )
        if stored != set(role_map):
            raise TaskFlowMaterializationError(
                "stored required-role snapshot does not exactly match SkillGraph binding roles"
            )
        return {placeholder: role_map[placeholder] for placeholder in sorted(role_map)}

    @staticmethod
    def _revision_map(
        metadata: Mapping[str, Any], key: str, stage_type: str
    ) -> dict[str, str]:
        value = metadata.get(key, {})
        if not isinstance(value, Mapping) or any(
            not isinstance(item_key, str) or not isinstance(item, str)
            for item_key, item in value.items()
        ):
            raise TaskFlowMaterializationError(
                f"{stage_type} {key} must be a string-to-string mapping"
            )
        return dict(value)

    @staticmethod
    def _validate_resolved_roles(
        provided: Mapping[str, str] | None,
        required: set[str],
    ) -> dict[str, str]:
        if provided is None:
            if required:
                raise TaskFlowMaterializationError(
                    "resolved Scene role bindings are required for "
                    f"{sorted(required)}"
                )
            return {}
        normalized = dict(provided)
        if set(normalized) != required:
            missing = sorted(required - set(normalized))
            unexpected = sorted(set(normalized) - required)
            raise TaskFlowMaterializationError(
                "resolved Scene role bindings do not exactly cover required roles; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for placeholder, entity_id in normalized.items():
            if not _ROLE_PATTERN.fullmatch(placeholder):
                raise TaskFlowMaterializationError(
                    f"resolved role placeholder {placeholder!r} is malformed"
                )
            if not isinstance(entity_id, str) or not _ENTITY_ID_PATTERN.fullmatch(
                entity_id
            ):
                raise TaskFlowMaterializationError(
                    f"resolved Scene entity ID for {placeholder!r} is malformed"
                )
        return {key: normalized[key] for key in sorted(normalized)}


def materialize_task_flow(
    *,
    repository: StorageRepository,
    artifact_store: LocalArtifactStore,
    object_class_id: str,
    action_id: str,
    resolved_role_bindings: Mapping[str, str] | None = None,
    require_hardware_compatible: bool = False,
) -> MaterializedTaskFlow:
    """Stateless convenience wrapper for one storage-backed composition."""

    return TaskFlowMaterializer(repository, artifact_store).materialize(
        object_class_id=object_class_id,
        action_id=action_id,
        resolved_role_bindings=resolved_role_bindings,
        require_hardware_compatible=require_hardware_compatible,
    )


__all__ = [
    "ExecutableGripProfileArtifact",
    "MaterializedTaskFlow",
    "TaskFlowMaterializationError",
    "TaskFlowMaterializer",
    "TaskFlowStorageSelection",
    "materialize_task_flow",
]
