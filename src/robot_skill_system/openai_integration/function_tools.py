"""Read-only local function dispatcher for bounded OpenAI tool calls.

The dispatcher has no robot, shell, file-write, dynamic import, or safety-control capability.
It accepts only six fixed names, validates arguments before selecting an explicit handler, and
validates every provider result before returning JSON to a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from robot_skill_system.exceptions import OpenAIIntegrationError
from robot_skill_system.primitives.registry import PrimitiveRegistry, get_default_registry
from robot_skill_system.scene.models import SceneSnapshot
from robot_skill_system.skills.models import SkillGraph, SkillManifest

from .schemas import SafeFunctionName
from .tool_schemas import (
    CompareSkillVersionsArguments,
    GetPrimitiveCatalogArguments,
    GetSceneSummaryArguments,
    GetSkillManifestArguments,
    QuerySceneEntitiesArguments,
    SearchSkillRegistryArguments,
    ToolModel,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"

FORBIDDEN_TOOL_NAMES = frozenset(
    {
        "arbitrary_file_write",
        "arbitrary_python",
        "disable_safety",
        "eval",
        "exec",
        "execute_python",
        "execute_robot_directly",
        "execute_robot_motion",
        "execute_shell",
        "overwrite_active_skill",
        "raw_force_control",
        "raw_move_c",
        "raw_move_j",
        "raw_move_l",
        "raw_movec",
        "raw_movej",
        "raw_movel",
        "shell",
    }
)


class FunctionToolError(OpenAIIntegrationError):
    """Base class for bounded local function-call failures."""


class ToolNotAllowedError(FunctionToolError):
    """A known execution, file, shell, or safety-bypass tool was requested."""


class UnknownToolError(FunctionToolError):
    """A name outside both the read-only allowlist and explicit denylist was requested."""


class InvalidToolArgumentsError(FunctionToolError):
    """A read-only tool received malformed or schema-invalid arguments."""


class ToolOutputValidationError(FunctionToolError):
    """A local provider returned data outside the declared output schema."""


class ToolResourceNotFoundError(FunctionToolError):
    """A requested local immutable record was not found."""


class FunctionSkillVersion(ToolModel):
    """Small searchable view of one immutable skill version."""

    skill_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    intent: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    lifecycle_status: str = Field(min_length=1, max_length=32)
    validation_status: str = Field(min_length=1, max_length=32)
    variant: str = Field(min_length=1, max_length=128)
    description: str = Field(max_length=2000)
    graph_checksum_sha256: str = Field(pattern=_SHA256_PATTERN)
    generated_code_checksum_sha256: str | None
    hardware_compatible: bool

    @classmethod
    def from_graph(
        cls,
        graph: SkillGraph,
        *,
        graph_checksum_sha256: str,
        generated_code_checksum_sha256: str | None,
        hardware_compatible: bool = False,
        variant: str = "default",
    ) -> FunctionSkillVersion:
        """Build a tool-safe index record from a validated local graph."""

        return cls(
            skill_id=graph.skill_id,
            name=graph.name,
            intent=graph.skill_type.value,
            version=graph.version,
            lifecycle_status=graph.lifecycle_status.value,
            validation_status=graph.validation_status.value,
            variant=variant,
            description=graph.description,
            graph_checksum_sha256=graph_checksum_sha256,
            generated_code_checksum_sha256=generated_code_checksum_sha256,
            hardware_compatible=hardware_compatible,
        )


class SkillRegistryMatch(ToolModel):
    skill_id: str
    name: str
    intent: str
    version: str
    lifecycle_status: Literal["active", "validated"]
    variant: str
    description: str


class SearchSkillRegistryOutput(ToolModel):
    matches: list[SkillRegistryMatch]


class FunctionSkillManifest(ToolModel):
    skill_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    skill_graph_uri: str = Field(min_length=1, max_length=1000)
    skill_graph_checksum_sha256: str = Field(pattern=_SHA256_PATTERN)
    compiled_skill_uri: str = Field(min_length=1, max_length=1000)
    compiled_skill_checksum_sha256: str = Field(pattern=_SHA256_PATTERN)
    validation_report_uri: str | None
    validation_report_checksum_sha256: str | None

    @classmethod
    def from_manifest(cls, manifest: SkillManifest) -> FunctionSkillManifest:
        """Drop source-demonstration paths while retaining immutable artifact integrity data."""

        return cls(
            skill_id=manifest.skill_id,
            version=manifest.version,
            skill_graph_uri=manifest.skill_graph_uri,
            skill_graph_checksum_sha256=manifest.skill_graph_checksum_sha256,
            compiled_skill_uri=manifest.compiled_skill_uri,
            compiled_skill_checksum_sha256=manifest.compiled_skill_checksum_sha256,
            validation_report_uri=manifest.validation_report_uri,
            validation_report_checksum_sha256=manifest.validation_report_checksum_sha256,
        )


class GetSkillManifestOutput(ToolModel):
    manifest: FunctionSkillManifest


class PrimitiveCatalogEntry(ToolModel):
    operation_name: str
    description: str
    argument_fields: list[str]
    allowed_skill_types: list[str]
    required_preconditions: list[str]
    side_effects: list[str]
    maximum_timeout_s: float = Field(gt=0.0)
    recovery_operation: str
    mock_support: bool


class GetPrimitiveCatalogOutput(ToolModel):
    primitives: list[PrimitiveCatalogEntry]


SceneEntityKind = Literal["object", "tool", "surface", "workspace_region", "obstacle"]


class FunctionSceneEntity(ToolModel):
    entity_id: str = Field(min_length=1, max_length=128)
    entity_kind: SceneEntityKind
    class_or_role: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    attached: bool | None


class FunctionScene(ToolModel):
    scene_id: str = Field(min_length=1, max_length=128)
    timestamp_ns: int = Field(ge=0)
    reference_frame: str = Field(min_length=1, max_length=128)
    valid_for_ms: int = Field(gt=0)
    calibration_id: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[FunctionSceneEntity]

    @classmethod
    def from_scene(cls, scene: SceneSnapshot) -> FunctionScene:
        """Build a geometry-free function view from a validated scene snapshot."""

        entities = [
            FunctionSceneEntity(
                entity_id=item.instance_id,
                entity_kind="object",
                class_or_role=item.class_name,
                confidence=item.confidence,
                attached=None,
            )
            for item in scene.objects
        ]
        entities.extend(
            FunctionSceneEntity(
                entity_id=item.instance_id,
                entity_kind="tool",
                class_or_role=item.tool_class,
                confidence=item.verification_confidence,
                attached=item.attached,
            )
            for item in scene.tools
        )
        entities.extend(
            FunctionSceneEntity(
                entity_id=item.instance_id,
                entity_kind="surface",
                class_or_role=item.role.value,
                confidence=item.confidence,
                attached=None,
            )
            for item in scene.surfaces
        )
        entities.extend(
            FunctionSceneEntity(
                entity_id=item.region_id,
                entity_kind="workspace_region",
                class_or_role=item.role.value,
                confidence=item.confidence,
                attached=None,
            )
            for item in scene.workspace_regions
        )
        entities.extend(
            FunctionSceneEntity(
                entity_id=item.instance_id,
                entity_kind="obstacle",
                class_or_role="static_obstacle",
                confidence=item.confidence,
                attached=None,
            )
            for item in scene.static_obstacles
        )
        entities.extend(
            FunctionSceneEntity(
                entity_id=item.instance_id,
                entity_kind="obstacle",
                class_or_role="dynamic_obstacle",
                confidence=item.confidence,
                attached=None,
            )
            for item in scene.dynamic_obstacles
        )
        return cls(
            scene_id=scene.scene_id,
            timestamp_ns=scene.timestamp_ns,
            reference_frame=scene.reference_frame,
            valid_for_ms=scene.valid_for_ms,
            calibration_id=scene.calibration_id,
            confidence=scene.confidence_summary.overall,
            entities=entities,
        )


class QuerySceneEntitiesOutput(ToolModel):
    scene_id: str
    entities: list[FunctionSceneEntity]


class SceneEntityCounts(ToolModel):
    objects: int = Field(ge=0)
    tools: int = Field(ge=0)
    surfaces: int = Field(ge=0)
    workspace_regions: int = Field(ge=0)
    obstacles: int = Field(ge=0)


class GetSceneSummaryOutput(ToolModel):
    scene_id: str
    timestamp_ns: int
    reference_frame: str
    valid_for_ms: int
    calibration_id: str
    confidence: float
    entity_counts: SceneEntityCounts
    entity_ids: list[str]


class ComparedSkillVersion(ToolModel):
    version: str
    lifecycle_status: str
    validation_status: str
    graph_checksum_sha256: str
    generated_code_checksum_sha256: str | None
    hardware_compatible: bool


VersionDifference = Literal[
    "lifecycle_status",
    "validation_status",
    "graph_checksum_sha256",
    "generated_code_checksum_sha256",
    "hardware_compatible",
]


class CompareSkillVersionsOutput(ToolModel):
    skill_id: str
    left: ComparedSkillVersion
    right: ComparedSkillVersion
    differing_fields: list[VersionDifference]


ToolOutput = (
    SearchSkillRegistryOutput
    | GetSkillManifestOutput
    | GetPrimitiveCatalogOutput
    | QuerySceneEntitiesOutput
    | GetSceneSummaryOutput
    | CompareSkillVersionsOutput
)


class ReadOnlyToolProvider(Protocol):
    """Provider contract intentionally contains no mutating operation."""

    def search_skill_registry(
        self, arguments: SearchSkillRegistryArguments
    ) -> SearchSkillRegistryOutput: ...

    def get_skill_manifest(
        self, arguments: GetSkillManifestArguments
    ) -> GetSkillManifestOutput: ...

    def get_primitive_catalog(
        self, arguments: GetPrimitiveCatalogArguments
    ) -> GetPrimitiveCatalogOutput: ...

    def query_scene_entities(
        self, arguments: QuerySceneEntitiesArguments
    ) -> QuerySceneEntitiesOutput: ...

    def get_scene_summary(
        self, arguments: GetSceneSummaryArguments
    ) -> GetSceneSummaryOutput: ...

    def compare_skill_versions(
        self, arguments: CompareSkillVersionsArguments
    ) -> CompareSkillVersionsOutput: ...


class LocalReadOnlyToolProvider:
    """Read-only implementation over caller-supplied validated local snapshots."""

    def __init__(
        self,
        *,
        skill_versions: Sequence[FunctionSkillVersion] = (),
        manifests: Sequence[FunctionSkillManifest] = (),
        scenes: Sequence[FunctionScene] = (),
        primitive_registry: PrimitiveRegistry | None = None,
    ) -> None:
        self._skill_versions = tuple(skill_versions)
        self._manifests = {(item.skill_id, item.version): item for item in manifests}
        self._scenes = {item.scene_id: item for item in scenes}
        self._primitive_registry = primitive_registry or get_default_registry()
        if len(self._manifests) != len(manifests):
            raise ValueError("duplicate function-tool skill manifests")
        if len(self._scenes) != len(scenes):
            raise ValueError("duplicate function-tool scene identifiers")
        version_keys = {(item.skill_id, item.version) for item in skill_versions}
        if len(version_keys) != len(skill_versions):
            raise ValueError("duplicate function-tool skill versions")

    def search_skill_registry(
        self, arguments: SearchSkillRegistryArguments
    ) -> SearchSkillRegistryOutput:
        query = arguments.query.casefold()
        allowed_statuses = (
            {"active"} if arguments.lifecycle == "active" else {"active", "validated"}
        )
        matches = [
            item
            for item in self._skill_versions
            if item.validation_status == "passed"
            and item.lifecycle_status in allowed_statuses
            and query
            in " ".join(
                (
                    item.skill_id,
                    item.name,
                    item.intent,
                    item.variant,
                    item.description,
                )
            ).casefold()
        ]
        matches.sort(
            key=lambda item: (
                item.lifecycle_status != "active",
                item.name.casefold(),
                item.version,
            )
        )
        return SearchSkillRegistryOutput(
            matches=[
                SkillRegistryMatch(
                    skill_id=item.skill_id,
                    name=item.name,
                    intent=item.intent,
                    version=item.version,
                    lifecycle_status=(
                        "active" if item.lifecycle_status == "active" else "validated"
                    ),
                    variant=item.variant,
                    description=item.description,
                )
                for item in matches[: arguments.limit]
            ]
        )

    def get_skill_manifest(
        self, arguments: GetSkillManifestArguments
    ) -> GetSkillManifestOutput:
        try:
            manifest = self._manifests[(arguments.skill_id, arguments.version)]
        except KeyError as exc:
            raise ToolResourceNotFoundError("skill manifest was not found") from exc
        return GetSkillManifestOutput(manifest=manifest)

    def get_primitive_catalog(
        self, arguments: GetPrimitiveCatalogArguments
    ) -> GetPrimitiveCatalogOutput:
        primitives: list[PrimitiveCatalogEntry] = []
        for metadata in self._primitive_registry.catalog():
            category = metadata.operation_name.split(".", maxsplit=1)[0]
            if arguments.category != "all" and category != arguments.category:
                continue
            raw_properties = metadata.typed_parameter_schema.get("properties", {})
            argument_fields = (
                sorted(str(item) for item in raw_properties)
                if isinstance(raw_properties, Mapping)
                else []
            )
            primitives.append(
                PrimitiveCatalogEntry(
                    operation_name=metadata.operation_name,
                    description=metadata.description,
                    argument_fields=argument_fields,
                    allowed_skill_types=list(metadata.allowed_skill_types),
                    required_preconditions=list(metadata.required_preconditions),
                    side_effects=list(metadata.side_effects),
                    maximum_timeout_s=metadata.maximum_timeout_s,
                    recovery_operation=metadata.recovery_operation,
                    mock_support=metadata.mock_support,
                )
            )
        return GetPrimitiveCatalogOutput(primitives=primitives[: arguments.limit])

    def query_scene_entities(
        self, arguments: QuerySceneEntitiesArguments
    ) -> QuerySceneEntitiesOutput:
        scene = self._scene(arguments.scene_id)
        query = arguments.query.casefold()
        entities = [
            entity
            for entity in scene.entities
            if (arguments.entity_kind == "all" or entity.entity_kind == arguments.entity_kind)
            and (
                not query
                or query in entity.entity_id.casefold()
                or query in entity.class_or_role.casefold()
            )
        ]
        entities.sort(key=lambda item: (item.entity_kind, item.entity_id))
        return QuerySceneEntitiesOutput(
            scene_id=scene.scene_id, entities=entities[: arguments.limit]
        )

    def get_scene_summary(
        self, arguments: GetSceneSummaryArguments
    ) -> GetSceneSummaryOutput:
        scene = self._scene(arguments.scene_id)
        return GetSceneSummaryOutput(
            scene_id=scene.scene_id,
            timestamp_ns=scene.timestamp_ns,
            reference_frame=scene.reference_frame,
            valid_for_ms=scene.valid_for_ms,
            calibration_id=scene.calibration_id,
            confidence=scene.confidence,
            entity_counts=SceneEntityCounts(
                objects=sum(item.entity_kind == "object" for item in scene.entities),
                tools=sum(item.entity_kind == "tool" for item in scene.entities),
                surfaces=sum(item.entity_kind == "surface" for item in scene.entities),
                workspace_regions=sum(
                    item.entity_kind == "workspace_region" for item in scene.entities
                ),
                obstacles=sum(item.entity_kind == "obstacle" for item in scene.entities),
            ),
            entity_ids=sorted(item.entity_id for item in scene.entities),
        )

    def compare_skill_versions(
        self, arguments: CompareSkillVersionsArguments
    ) -> CompareSkillVersionsOutput:
        left = self._version(arguments.skill_id, arguments.left_version)
        right = self._version(arguments.skill_id, arguments.right_version)
        left_output = _compared_version(left)
        right_output = _compared_version(right)
        fields: tuple[VersionDifference, ...] = (
            "lifecycle_status",
            "validation_status",
            "graph_checksum_sha256",
            "generated_code_checksum_sha256",
            "hardware_compatible",
        )
        return CompareSkillVersionsOutput(
            skill_id=arguments.skill_id,
            left=left_output,
            right=right_output,
            differing_fields=[
                field
                for field in fields
                if getattr(left_output, field) != getattr(right_output, field)
            ],
        )

    def _scene(self, scene_id: str) -> FunctionScene:
        try:
            return self._scenes[scene_id]
        except KeyError as exc:
            raise ToolResourceNotFoundError("scene was not found") from exc

    def _version(self, skill_id: str, version: str) -> FunctionSkillVersion:
        for item in self._skill_versions:
            if item.skill_id == skill_id and item.version == version:
                return item
        raise ToolResourceNotFoundError("skill version was not found")


OutputT = TypeVar("OutputT", bound=BaseModel)


def _validated_output(output_type: type[OutputT], value: object) -> OutputT:
    try:
        return output_type.model_validate(value)
    except ValidationError as exc:
        raise ToolOutputValidationError("local function output failed validation") from exc


def _validated_arguments(
    name: str, argument_type: type[OutputT], arguments: str | Mapping[str, object]
) -> OutputT:
    try:
        raw: object = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        return argument_type.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidToolArgumentsError(
            f"invalid arguments for read-only tool {name!r}"
        ) from exc


def _compared_version(value: FunctionSkillVersion) -> ComparedSkillVersion:
    return ComparedSkillVersion(
        version=value.version,
        lifecycle_status=value.lifecycle_status,
        validation_status=value.validation_status,
        graph_checksum_sha256=value.graph_checksum_sha256,
        generated_code_checksum_sha256=value.generated_code_checksum_sha256,
        hardware_compatible=value.hardware_compatible,
    )


class SafeFunctionDispatcher:
    """Validate and dispatch one allowlisted read-only local function call."""

    def __init__(self, provider: ReadOnlyToolProvider) -> None:
        self._provider = provider

    @property
    def allowed_names(self) -> frozenset[str]:
        """Return the immutable model-visible function allowlist."""

        return frozenset(item.value for item in SafeFunctionName)

    def dispatch(
        self, name: str, arguments: str | Mapping[str, object]
    ) -> ToolOutput:
        """Dispatch one call without dynamic lookup or any mutating capability."""

        if name in FORBIDDEN_TOOL_NAMES:
            raise ToolNotAllowedError(f"tool {name!r} is explicitly forbidden")
        try:
            function_name = SafeFunctionName(name)
        except ValueError as exc:
            raise UnknownToolError(f"unknown function tool {name!r}") from exc

        if function_name is SafeFunctionName.SEARCH_SKILL_REGISTRY:
            search_arguments = _validated_arguments(
                name, SearchSkillRegistryArguments, arguments
            )
            return _validated_output(
                SearchSkillRegistryOutput,
                self._provider.search_skill_registry(search_arguments),
            )
        if function_name is SafeFunctionName.GET_SKILL_MANIFEST:
            manifest_arguments = _validated_arguments(
                name, GetSkillManifestArguments, arguments
            )
            return _validated_output(
                GetSkillManifestOutput,
                self._provider.get_skill_manifest(manifest_arguments),
            )
        if function_name is SafeFunctionName.GET_PRIMITIVE_CATALOG:
            catalog_arguments = _validated_arguments(
                name, GetPrimitiveCatalogArguments, arguments
            )
            return _validated_output(
                GetPrimitiveCatalogOutput,
                self._provider.get_primitive_catalog(catalog_arguments),
            )
        if function_name is SafeFunctionName.QUERY_SCENE_ENTITIES:
            entity_arguments = _validated_arguments(
                name, QuerySceneEntitiesArguments, arguments
            )
            return _validated_output(
                QuerySceneEntitiesOutput,
                self._provider.query_scene_entities(entity_arguments),
            )
        if function_name is SafeFunctionName.GET_SCENE_SUMMARY:
            scene_arguments = _validated_arguments(
                name, GetSceneSummaryArguments, arguments
            )
            return _validated_output(
                GetSceneSummaryOutput,
                self._provider.get_scene_summary(scene_arguments),
            )
        comparison_arguments = _validated_arguments(
            name, CompareSkillVersionsArguments, arguments
        )
        return _validated_output(
            CompareSkillVersionsOutput,
            self._provider.compare_skill_versions(comparison_arguments),
        )

    def dispatch_json(
        self, name: str, arguments: str | Mapping[str, object]
    ) -> str:
        """Return a validated, compact JSON function-call output."""

        return self.dispatch(name, arguments).model_dump_json(exclude_none=False)


__all__ = [
    "CompareSkillVersionsOutput",
    "FORBIDDEN_TOOL_NAMES",
    "FunctionScene",
    "FunctionSceneEntity",
    "FunctionSkillManifest",
    "FunctionSkillVersion",
    "FunctionToolError",
    "GetPrimitiveCatalogOutput",
    "GetSceneSummaryOutput",
    "GetSkillManifestOutput",
    "InvalidToolArgumentsError",
    "LocalReadOnlyToolProvider",
    "QuerySceneEntitiesOutput",
    "ReadOnlyToolProvider",
    "SafeFunctionDispatcher",
    "SearchSkillRegistryOutput",
    "ToolNotAllowedError",
    "ToolOutputValidationError",
    "ToolResourceNotFoundError",
    "UnknownToolError",
]
