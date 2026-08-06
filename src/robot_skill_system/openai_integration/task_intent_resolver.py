"""Resolve text into catalog-bounded Grip -> Action task intent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI
from pydantic import Field, model_validator

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .schemas import APICallMetadata, StrictModel, TaskIntent
from .semantic_validation import validate_task_intent

TASK_INTENT_INSTRUCTIONS = """Classify the command using only the supplied object and action
catalog identifiers. Select current Scene entity IDs only for object_instance_id and role_bindings.
Role keys must come from required_roles. Do not return an end-motion ID: local code owns the
Action-to-End mapping. Never return coordinates, poses, transforms, paths, force, speed,
acceleration, code, robot calls, or execution permission. Report ambiguity instead of inventing an
identifier. When exactly one supplied Scene entity is compatible with a required role, bind it and
do not mark the intent ambiguous merely because the command omitted that role's destination name;
zero or multiple compatible entities remain ambiguous. Return role_bindings as an array of objects
with exactly role and entity_id fields."""


class _TaskRoleBinding(StrictModel):
    """Responses-API-safe serialization of one dynamic semantic role binding."""

    role: str = Field(pattern=r"^\$[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    entity_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class _TaskIntentResponse(StrictModel):
    """Strict API wire schema converted into the stable internal ``TaskIntent``."""

    object_class_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    action_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    object_instance_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
    )
    role_bindings: list[_TaskRoleBinding] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: bool = False
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=32)
    confidence_rationale: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_unique_roles(self) -> _TaskIntentResponse:
        roles = [item.role for item in self.role_bindings]
        if len(roles) != len(set(roles)):
            raise ValueError("role_bindings cannot contain duplicate roles")
        return self

    def to_task_intent(self) -> TaskIntent:
        return TaskIntent(
            object_class_id=self.object_class_id,
            action_id=self.action_id,
            object_instance_id=self.object_instance_id,
            role_bindings={item.role: item.entity_id for item in self.role_bindings},
            confidence=self.confidence,
            ambiguity=self.ambiguity,
            unresolved_ambiguities=self.unresolved_ambiguities,
            confidence_rationale=self.confidence_rationale,
        )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _catalog_matches(
    command: str, catalog: Mapping[str, Sequence[str]]
) -> list[tuple[int, str]]:
    normalized_command = _normalized(command)
    matches: list[tuple[int, str]] = []
    for canonical_id, aliases in catalog.items():
        terms = {canonical_id, *aliases}
        matched_lengths = [
            len(term_normalized)
            for term in terms
            if (term_normalized := _normalized(str(term)))
            and term_normalized in normalized_command
        ]
        if matched_lengths:
            matches.append((max(matched_lengths), canonical_id))
    return sorted(matches, key=lambda item: (-item[0], item[1]))


def _select_catalog_id(
    command: str,
    catalog: Mapping[str, Sequence[str]],
    *,
    label: str,
) -> tuple[str, list[str], float]:
    matches = _catalog_matches(command, catalog)
    if matches:
        best_score = matches[0][0]
        best = [canonical_id for score, canonical_id in matches if score == best_score]
        return (
            best[0],
            [f"multiple {label} matches: {best!r}"] if len(best) > 1 else [],
            0.95 if len(best) == 1 else 0.55,
        )
    if len(catalog) == 1:
        return next(iter(catalog)), [f"{label} inferred from the only active catalog entry"], 0.6
    raise ValueError(f"command does not identify an active {label}")


def _mock_task_intent(
    command: str,
    *,
    object_catalog: Mapping[str, Sequence[str]],
    action_catalog: Mapping[str, Sequence[str]],
    scene_entities: Mapping[str, Mapping[str, str]],
    required_roles: Mapping[str, Sequence[str]],
) -> TaskIntent:
    object_class_id, object_ambiguities, object_confidence = _select_catalog_id(
        command, object_catalog, label="object class"
    )
    action_id, action_ambiguities, action_confidence = _select_catalog_id(
        command, action_catalog, label="action"
    )
    ambiguities = [*object_ambiguities, *action_ambiguities]
    object_candidates = sorted(
        entity_id
        for entity_id, payload in scene_entities.items()
        if payload.get("kind") == "object"
        and payload.get("class_or_role") == object_class_id
    )
    object_instance_id: str | None = None
    if len(object_candidates) == 1:
        object_instance_id = object_candidates[0]
    elif len(object_candidates) > 1:
        explicit = [item for item in object_candidates if _normalized(item) in _normalized(command)]
        if len(explicit) == 1:
            object_instance_id = explicit[0]
        else:
            ambiguities.append(
                f"multiple Scene object instances match {object_class_id!r}: {object_candidates!r}"
            )

    role_bindings: dict[str, str] = {}
    for role in required_roles.get(action_id, ()):
        role_name = role.removeprefix("$").casefold()
        expected_kinds = (
            {"surface"}
            if "surface" in role_name
            else {"tool"}
            if "tool" in role_name or "gripper" in role_name
            else {"object"}
            if "object" in role_name
            else {"workspace_region", "surface"}
            if "destination" in role_name or "region" in role_name
            else {"workspace_region"}
        )
        candidates = sorted(
            entity_id
            for entity_id, payload in scene_entities.items()
            if payload.get("kind") in expected_kinds
        )
        explicit = [item for item in candidates if _normalized(item) in _normalized(command)]
        if len(explicit) == 1:
            role_bindings[role] = explicit[0]
        elif len(candidates) == 1:
            role_bindings[role] = candidates[0]
        elif candidates:
            ambiguities.append(f"role {role!r} has multiple Scene candidates: {candidates!r}")
        else:
            ambiguities.append(f"role {role!r} has no current Scene candidate")

    return TaskIntent(
        object_class_id=object_class_id,
        action_id=action_id,
        object_instance_id=object_instance_id,
        role_bindings=role_bindings,
        confidence=min(object_confidence, action_confidence),
        ambiguity=bool(ambiguities),
        unresolved_ambiguities=ambiguities,
        confidence_rationale="Deterministic mock alias matching over active local catalogs.",
    )


class TaskIntentResolver:
    """Classify object/action semantics without selecting geometry or End Motion."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
        retry: RetryExecutor | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._retry = retry or RetryExecutor(settings.openai_api_max_retries)

    def resolve(
        self,
        command: str,
        *,
        object_catalog: Mapping[str, Sequence[str]],
        action_catalog: Mapping[str, Sequence[str]],
        scene_entities: Mapping[str, Mapping[str, str]],
        required_roles: Mapping[str, Sequence[str]],
    ) -> tuple[TaskIntent, APICallMetadata]:
        """Return a validated TaskIntent; never compose or execute a flow."""

        if not object_catalog:
            raise ValueError("active object catalog is empty")
        if not action_catalog:
            raise ValueError("active action catalog is empty")
        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            intent = _mock_task_intent(
                command,
                object_catalog=object_catalog,
                action_catalog=action_catalog,
                scene_entities=scene_entities,
                required_roles=required_roles,
            )
            metadata = APICallMetadata(trace_id=trace_id)
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            payload: dict[str, Any] = {
                "command": command,
                "object_catalog": {
                    key: list(value) for key, value in object_catalog.items()
                },
                "action_catalog": {
                    key: list(value) for key, value in action_catalog.items()
                },
                "scene_entities": dict(scene_entities),
                "required_roles": {
                    key: list(value) for key, value in required_roles.items()
                },
            }
            response_intent, metadata = parse_structured_response(
                client=client,
                retry=self._retry,
                model=self.settings.openai_reasoning_model,
                instructions=TASK_INTENT_INSTRUCTIONS,
                payload=payload,
                output_type=_TaskIntentResponse,
                trace_id=trace_id,
            )
            intent = response_intent.to_task_intent()
        allowed_roles = set(required_roles.get(intent.action_id, ()))
        validate_task_intent(
            intent,
            object_catalog=object_catalog,
            action_catalog=action_catalog,
            entity_catalog=scene_entities,
            allowed_roles=allowed_roles,
        )
        return intent, metadata


__all__ = ["TASK_INTENT_INSTRUCTIONS", "TaskIntentResolver"]
