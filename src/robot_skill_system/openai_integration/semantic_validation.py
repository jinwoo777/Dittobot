"""Post-parse validation against caller-owned semantic catalogs.

Structured output validation proves that a response has the expected shape.  It does not prove
that identifier strings came from the current scene or approved configuration.  These checks are
therefore deliberately performed after both live and mock responses are produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from robot_skill_system.exceptions import SemanticCatalogViolationError

from .schemas import DemonstrationAnalysis, RuntimeIntent, SkillGraphProposal


def _unknown(values: Iterable[str], catalog: Iterable[str]) -> set[str]:
    return set(values).difference(catalog)


def _raise_if_violations(context: str, violations: Mapping[str, set[str]]) -> None:
    details = [
        f"{field}={sorted(values)!r}"
        for field, values in sorted(violations.items())
        if values
    ]
    if details:
        joined = "; ".join(details)
        raise SemanticCatalogViolationError(
            f"{context} referenced identifiers outside authoritative catalogs: {joined}"
        )


def validate_demonstration_analysis(
    analysis: DemonstrationAnalysis,
    *,
    entity_catalog: Iterable[str],
    primitive_catalog: Iterable[str],
    approved_motion_profiles: Iterable[str],
    approved_force_profiles: Iterable[str],
) -> None:
    """Reject invented entity, operation, and profile identifiers."""

    entity_ids = set(analysis.involved_entity_ids)
    entity_ids.update(analysis.target_ids)
    if analysis.tool_id is not None:
        entity_ids.add(analysis.tool_id)
    for phase in analysis.phase_labels:
        entity_ids.update(phase.involved_entity_ids)
    entity_ids.update(interval.surface_id for interval in analysis.contact_intervals)

    operations = {item.operation for item in analysis.primitive_recommendations}
    motion_profiles = {
        item.motion_profile_id
        for item in analysis.primitive_recommendations
        if item.motion_profile_id is not None
    }
    force_profiles = {
        item.force_profile_id
        for item in analysis.primitive_recommendations
        if item.force_profile_id is not None
    }
    _raise_if_violations(
        "DemonstrationAnalysis",
        {
            "entity_ids": _unknown(entity_ids, entity_catalog),
            "operations": _unknown(operations, primitive_catalog),
            "motion_profile_ids": _unknown(motion_profiles, approved_motion_profiles),
            "force_profile_ids": _unknown(force_profiles, approved_force_profiles),
        },
    )


def validate_runtime_intent(
    intent: RuntimeIntent,
    *,
    entity_catalog: Iterable[str],
    approved_motion_profiles: Iterable[str],
    approved_force_profiles: Iterable[str],
) -> None:
    """Reject entity and profile selections outside the current runtime catalogs."""

    motion_profiles = (
        {intent.speed_profile_request} if intent.speed_profile_request is not None else set()
    )
    force_profiles = (
        {intent.force_profile_request} if intent.force_profile_request is not None else set()
    )
    _raise_if_violations(
        "RuntimeIntent",
        {
            "candidate_entity_ids": _unknown(intent.candidate_entity_ids, entity_catalog),
            "motion_profile_ids": _unknown(motion_profiles, approved_motion_profiles),
            "force_profile_ids": _unknown(force_profiles, approved_force_profiles),
        },
    )


def validate_skill_graph_proposal(
    proposal: SkillGraphProposal,
    *,
    primitive_catalog: Iterable[str],
    binding_ref_catalog: Iterable[str],
    approved_motion_profiles: Iterable[str],
    approved_force_profiles: Iterable[str],
) -> None:
    """Reject proposal identifiers not supplied by deterministic local code."""

    operations = {node.operation for node in proposal.nodes}
    binding_refs = set(proposal.required_entity_roles)
    binding_refs.update(ref for node in proposal.nodes for ref in node.binding_refs)
    motion_profiles = {
        node.motion_profile_id for node in proposal.nodes if node.motion_profile_id is not None
    }
    force_profiles = {
        node.force_profile_id for node in proposal.nodes if node.force_profile_id is not None
    }
    _raise_if_violations(
        "SkillGraphProposal",
        {
            "operations": _unknown(operations, primitive_catalog),
            "binding_refs": _unknown(binding_refs, binding_ref_catalog),
            "motion_profile_ids": _unknown(motion_profiles, approved_motion_profiles),
            "force_profile_ids": _unknown(force_profiles, approved_force_profiles),
        },
    )
