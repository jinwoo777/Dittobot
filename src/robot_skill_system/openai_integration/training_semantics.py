"""Semantic-only classification for RGB-D component induction.

This resolver deliberately has no scene geometry, motion, profile, end-motion,
or execution fields in its output contract.  The mock implementation is
conservative: it accepts unambiguous catalog aliases or explicit canonical-ID
annotations in the transcript and otherwise reports ambiguity instead of
inventing a label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI
from pydantic import ConfigDict, Field, model_validator

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .schemas import APICallMetadata, StrictModel

_CANONICAL_ID_PATTERN = r"^[a-z][a-z0-9_.-]{0,127}$"
_ROLE_PATTERN = r"^\$[A-Za-z][A-Za-z0-9_.-]{0,63}$"
_OBJECT_TAG = re.compile(
    r"(?<![A-Za-z0-9_.-])object_class_id\s*[:=]\s*"
    r"([a-z][a-z0-9_.-]{0,127})(?![A-Za-z0-9_.-])"
)
_ACTION_TAG = re.compile(
    r"(?<![A-Za-z0-9_.-])action_id\s*[:=]\s*"
    r"([a-z][a-z0-9_.-]{0,127})(?![A-Za-z0-9_.-])"
)
_ROLES_TAG = re.compile(
    r"(?<![A-Za-z0-9_.-])required_roles\s*[:=]\s*"
    r"(\$[A-Za-z][A-Za-z0-9_.-]{0,63}"
    r"(?:\s*,\s*\$[A-Za-z][A-Za-z0-9_.-]{0,63})*)"
)

TRAINING_SEMANTIC_INSTRUCTIONS = """Classify only the semantic object class, action, and
required Scene role names described by the transcript. Return a known supplied canonical ID when
one fits. Otherwise propose a conservative lowercase canonical ID only when the transcript names
the concept clearly. Set ambiguity=true and leave an ID null instead of guessing. Never output an
end-motion ID, scene entity ID, coordinate, pose, transform, path, force, speed, acceleration,
profile, primitive, code, robot call, or execution decision. Required role names use $name syntax.
"""


class TrainingTaskSemantics(StrictModel):
    """The complete model-visible contract for component-training semantics."""

    model_config = ConfigDict(extra="forbid")

    object_class_id: str | None = Field(default=None, pattern=_CANONICAL_ID_PATTERN)
    action_id: str | None = Field(default=None, pattern=_CANONICAL_ID_PATTERN)
    required_roles: list[str] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: bool
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=16)
    confidence_rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _coherent_resolution(self) -> TrainingTaskSemantics:
        if len(set(self.required_roles)) != len(self.required_roles):
            raise ValueError("required_roles must be unique")
        if any(re.fullmatch(_ROLE_PATTERN, role) is None for role in self.required_roles):
            raise ValueError("required_roles must use $name syntax")
        incomplete = self.object_class_id is None or self.action_id is None
        if incomplete and not self.ambiguity:
            raise ValueError("missing semantic IDs require ambiguity=true")
        if self.ambiguity and not self.unresolved_ambiguities:
            raise ValueError("ambiguous semantics require an explanation")
        if not self.ambiguity and self.unresolved_ambiguities:
            raise ValueError("resolved semantics cannot retain unresolved ambiguity")
        return self


def _normalized(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _tag_value(pattern: re.Pattern[str], transcript: str) -> tuple[str | None, str | None]:
    values = sorted(set(pattern.findall(transcript)))
    if len(values) == 1:
        return values[0], None
    if len(values) > 1:
        return None, f"conflicting explicit semantic annotations: {values!r}"
    return None, None


def _alias_value(
    transcript: str,
    catalog: Mapping[str, Sequence[str]],
    *,
    label: str,
) -> tuple[str | None, str | None]:
    normalized_transcript = _normalized(transcript)
    matches: list[tuple[int, str]] = []
    for canonical_id, aliases in catalog.items():
        terms = {canonical_id, *aliases}
        lengths = [
            len(normalized_term)
            for term in terms
            if (normalized_term := _normalized(str(term)))
            and normalized_term in normalized_transcript
        ]
        if lengths:
            matches.append((max(lengths), canonical_id))
    if not matches:
        return None, f"transcript does not identify a known {label}"
    best_length = max(length for length, _ in matches)
    best = sorted(identifier for length, identifier in matches if length == best_length)
    if len(best) != 1:
        return None, f"transcript has multiple {label} matches: {best!r}"
    return best[0], None


def _mock_semantics(
    transcript: str,
    *,
    object_catalog: Mapping[str, Sequence[str]],
    action_catalog: Mapping[str, Sequence[str]],
    action_required_roles: Mapping[str, Sequence[str]],
) -> TrainingTaskSemantics:
    object_id, object_tag_error = _tag_value(_OBJECT_TAG, transcript)
    action_id, action_tag_error = _tag_value(_ACTION_TAG, transcript)
    errors = [error for error in (object_tag_error, action_tag_error) if error]
    used_annotations = object_id is not None or action_id is not None

    if object_id is None and object_tag_error is None:
        object_id, error = _alias_value(
            transcript, object_catalog, label="object class"
        )
        if error:
            errors.append(error)
    if action_id is None and action_tag_error is None:
        action_id, error = _alias_value(transcript, action_catalog, label="action")
        if error:
            errors.append(error)

    roles_match = _ROLES_TAG.search(transcript)
    annotated_roles = (
        [part.strip() for part in roles_match.group(1).split(",")]
        if roles_match is not None
        else None
    )
    known_roles = list(action_required_roles.get(action_id or "", ()))
    if annotated_roles is not None and action_id in action_required_roles:
        if annotated_roles != known_roles:
            errors.append(
                "required_roles annotation conflicts with the registered ActionDefinition"
            )
        roles = known_roles
    elif annotated_roles is not None:
        roles = annotated_roles
        used_annotations = True
    else:
        roles = known_roles

    ambiguity = bool(errors) or object_id is None or action_id is None
    return TrainingTaskSemantics(
        object_class_id=object_id,
        action_id=action_id,
        required_roles=roles,
        confidence=0.0 if ambiguity else (1.0 if used_annotations else 0.95),
        ambiguity=ambiguity,
        unresolved_ambiguities=errors,
        confidence_rationale=(
            "Deterministic explicit transcript annotations and catalog alias matching."
            if used_annotations
            else "Deterministic unambiguous alias matching over the local catalog."
        ),
    )


class TrainingSemanticResolver:
    """Resolve learning text without giving the model any motion authority."""

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
        transcript: str,
        *,
        object_catalog: Mapping[str, Sequence[str]],
        action_catalog: Mapping[str, Sequence[str]],
        action_required_roles: Mapping[str, Sequence[str]],
    ) -> tuple[TrainingTaskSemantics, APICallMetadata]:
        """Return semantic labels; unknown canonical IDs are permitted but never activated."""

        transcript = transcript.strip()
        if not transcript:
            raise ValueError("training transcript cannot be empty")
        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            result = _mock_semantics(
                transcript,
                object_catalog=object_catalog,
                action_catalog=action_catalog,
                action_required_roles=action_required_roles,
            )
            return result, APICallMetadata(trace_id=trace_id)

        client = self._client or OpenAIClientFactory(self.settings).create()
        payload: dict[str, Any] = {
            "transcript_text": transcript,
            "known_object_catalog": {
                key: list(value) for key, value in object_catalog.items()
            },
            "known_action_catalog": {
                key: list(value) for key, value in action_catalog.items()
            },
            "known_action_required_roles": {
                key: list(value) for key, value in action_required_roles.items()
            },
        }
        result, metadata = parse_structured_response(
            client=client,
            retry=self._retry,
            model=self.settings.openai_reasoning_model,
            instructions=TRAINING_SEMANTIC_INSTRUCTIONS,
            payload=payload,
            output_type=TrainingTaskSemantics,
            trace_id=trace_id,
        )
        known_roles = action_required_roles.get(result.action_id or "")
        if known_roles is not None and list(known_roles) != result.required_roles:
            result = result.model_copy(
                update={
                    "ambiguity": True,
                    "unresolved_ambiguities": [
                        *result.unresolved_ambiguities,
                        "model required_roles conflict with the registered ActionDefinition",
                    ],
                    "required_roles": list(known_roles),
                }
            )
        return result, metadata


__all__ = ["TrainingSemanticResolver", "TrainingTaskSemantics"]
