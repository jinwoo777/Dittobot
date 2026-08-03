"""Strict schemas for the only local functions exposed to OpenAI models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import SafeFunctionName


class ToolModel(BaseModel):
    """Reject coercion and undeclared fields at the function-call boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class SearchSkillRegistryArguments(ToolModel):
    query: str = Field(min_length=1, max_length=200)
    lifecycle: Literal["active", "validated"]
    limit: int = Field(ge=1, le=20)


class GetSkillManifestArguments(ToolModel):
    skill_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    version: str = Field(min_length=1, max_length=64, pattern=r"^[0-9A-Za-z.+-]+$")


class GetPrimitiveCatalogArguments(ToolModel):
    category: Literal["all", "motion", "gripper", "contact", "workspace", "recovery"]
    limit: int = Field(ge=1, le=100)


class QuerySceneEntitiesArguments(ToolModel):
    scene_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    query: str = Field(max_length=200)
    entity_kind: Literal[
        "all", "object", "tool", "surface", "workspace_region", "obstacle"
    ]
    limit: int = Field(ge=1, le=100)


class GetSceneSummaryArguments(ToolModel):
    scene_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class CompareSkillVersionsArguments(ToolModel):
    skill_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    left_version: str = Field(min_length=1, max_length=64, pattern=r"^[0-9A-Za-z.+-]+$")
    right_version: str = Field(min_length=1, max_length=64, pattern=r"^[0-9A-Za-z.+-]+$")


ARGUMENT_MODELS: dict[SafeFunctionName, type[ToolModel]] = {
    SafeFunctionName.SEARCH_SKILL_REGISTRY: SearchSkillRegistryArguments,
    SafeFunctionName.GET_SKILL_MANIFEST: GetSkillManifestArguments,
    SafeFunctionName.GET_PRIMITIVE_CATALOG: GetPrimitiveCatalogArguments,
    SafeFunctionName.QUERY_SCENE_ENTITIES: QuerySceneEntitiesArguments,
    SafeFunctionName.GET_SCENE_SUMMARY: GetSceneSummaryArguments,
    SafeFunctionName.COMPARE_SKILL_VERSIONS: CompareSkillVersionsArguments,
}

_DESCRIPTIONS: dict[SafeFunctionName, str] = {
    SafeFunctionName.SEARCH_SKILL_REGISTRY: "Search active or locally validated skill metadata.",
    SafeFunctionName.GET_SKILL_MANIFEST: (
        "Read immutable artifact URIs and checksums for one skill."
    ),
    SafeFunctionName.GET_PRIMITIVE_CATALOG: "Read operation metadata from the local whitelist.",
    SafeFunctionName.QUERY_SCENE_ENTITIES: "Search sanitized entities in one validated scene.",
    SafeFunctionName.GET_SCENE_SUMMARY: "Read a geometry-free summary of one validated scene.",
    SafeFunctionName.COMPARE_SKILL_VERSIONS: "Compare immutable metadata for two skill versions.",
}


def _strict_function_tool(
    name: SafeFunctionName, argument_model: type[ToolModel]
) -> dict[str, object]:
    schema = argument_model.model_json_schema()
    schema.pop("title", None)
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or set(required or ()) != set(properties):
        raise RuntimeError(f"strict function schema {name.value!r} has optional properties")
    if schema.get("additionalProperties") is not False:
        raise RuntimeError(f"strict function schema {name.value!r} permits extra properties")
    return {
        "type": "function",
        "name": name.value,
        "description": _DESCRIPTIONS[name],
        "parameters": schema,
        "strict": True,
    }


SAFE_FUNCTION_TOOLS: tuple[dict[str, object], ...] = tuple(
    _strict_function_tool(name, argument_model)
    for name, argument_model in ARGUMENT_MODELS.items()
)


__all__ = [
    "ARGUMENT_MODELS",
    "CompareSkillVersionsArguments",
    "GetPrimitiveCatalogArguments",
    "GetSceneSummaryArguments",
    "GetSkillManifestArguments",
    "QuerySceneEntitiesArguments",
    "SAFE_FUNCTION_TOOLS",
    "SearchSkillRegistryArguments",
    "ToolModel",
]
