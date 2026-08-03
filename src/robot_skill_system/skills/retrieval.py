"""Deterministic ranking over active, validated skill metadata and embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from robot_skill_system.openai_integration.embeddings import cosine_similarity


@dataclass(frozen=True, slots=True)
class SkillCandidate:
    """Searchable immutable view of one registered skill version."""

    skill_id: str
    version: str
    name: str
    description: str
    embedding: tuple[float, ...]
    intent: str
    tool_classes: tuple[str, ...]
    target_types: tuple[str, ...]
    contact: bool
    operator_style: str
    status: str
    validation_status: str
    hardware_compatible: bool


@dataclass(frozen=True, slots=True)
class SkillSearchQuery:
    embedding: tuple[float, ...]
    intent: str | None = None
    tool_class: str | None = None
    target_type: str | None = None
    contact: bool | None = None
    operator_style: str | None = None
    require_hardware_compatibility: bool = False


@dataclass(frozen=True, slots=True)
class RankedSkill:
    candidate: SkillCandidate
    score: float
    factors: dict[str, float]


def _exact(requested: str | None, available: tuple[str, ...] | str) -> float:
    if requested is None:
        return 0.5
    normalized = requested.strip().lower()
    values = (available,) if isinstance(available, str) else available
    return 1.0 if normalized in {value.lower() for value in values} else 0.0


def rank_skills(
    query: SkillSearchQuery,
    candidates: list[SkillCandidate],
    *,
    limit: int = 5,
) -> list[RankedSkill]:
    """Rank only active and validated versions using explicit, auditable factors."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked: list[RankedSkill] = []
    for candidate in candidates:
        if candidate.status != "active" or candidate.validation_status != "passed":
            continue
        if query.require_hardware_compatibility and not candidate.hardware_compatible:
            continue
        if len(query.embedding) != len(candidate.embedding):
            continue
        embedding = (
            cosine_similarity(list(query.embedding), list(candidate.embedding)) + 1.0
        ) / 2.0
        factors = {
            "embedding": embedding,
            "intent": _exact(query.intent, candidate.intent),
            "tool": _exact(query.tool_class, candidate.tool_classes),
            "target": _exact(query.target_type, candidate.target_types),
            "contact": (
                0.5 if query.contact is None else float(query.contact == candidate.contact)
            ),
            "style": _exact(query.operator_style, candidate.operator_style),
            "hardware": float(candidate.hardware_compatible),
        }
        score = (
            0.55 * factors["embedding"]
            + 0.15 * factors["intent"]
            + 0.10 * factors["tool"]
            + 0.05 * factors["target"]
            + 0.05 * factors["contact"]
            + 0.05 * factors["style"]
            + 0.05 * factors["hardware"]
        )
        ranked.append(RankedSkill(candidate=candidate, score=score, factors=factors))
    ranked.sort(key=lambda item: (-item.score, item.candidate.skill_id, item.candidate.version))
    return ranked[:limit]
