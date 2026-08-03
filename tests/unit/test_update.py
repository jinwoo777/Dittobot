from __future__ import annotations

import pytest

from robot_skill_system.skills.models import (
    BindingSpec,
    EntityKind,
    SkillGraph,
    SkillLifecycleStatus,
    SkillNode,
    SkillType,
    ValidationStatus,
)
from robot_skill_system.skills.updater import (
    SkillUpdater,
    UpdateComponent,
    UpdateEvidence,
)
from robot_skill_system.skills.versioning import ensure_transition, next_candidate_version


def relative(x: float, y: float, z: float) -> dict[str, object]:
    return {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": x, "y": y, "z": z},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def active_contact_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="wipe_surface",
        version="1.0.0",
        name="wipe surface",
        description="anchor-relative contact wipe",
        skill_type=SkillType.CONTACT,
        source_demonstrations=["novice_demo"],
        operator_style="safe",
        required_tools=["wiper"],
        required_entity_roles={"$surface": "contact_target"},
        bindings={
            "$surface": BindingSpec(
                variable="$surface", entity_kind=EntityKind.SURFACE, role="contact_target"
            )
        },
        nodes=[
            SkillNode(
                node_id="search",
                operation="contact.search_surface",
                arguments={"surface": "$surface", "force_profile_id": "wipe_standard"},
                on_success="enable",
            ),
            SkillNode(
                node_id="enable",
                operation="contact.enable_force",
                arguments={"surface": "$surface", "force_profile_id": "wipe_standard"},
                on_success="stroke",
            ),
            SkillNode(
                node_id="stroke",
                operation="motion.move_l",
                arguments={"target": relative(0.1, 0.0, 0.0), "motion_profile_id": "linear_slow"},
                on_success="disable",
            ),
            SkillNode(node_id="disable", operation="contact.disable_force", arguments={}),
        ],
        start_node="search",
        terminal_nodes=["disable"],
        motion_profiles=["linear_slow"],
        force_profiles=["wipe_standard"],
        validation_status=ValidationStatus.PASSED,
        lifecycle_status=SkillLifecycleStatus.ACTIVE,
    )


def test_expert_timing_creates_candidate_without_mutating_parent() -> None:
    parent = active_contact_graph()
    proposal = SkillUpdater().propose(
        parent,
        UpdateEvidence(
            source_demonstration_id="expert_demo",
            operator_role="expert",
            operator_style="expert_fast",
            timing_motion_profile_id="linear_expert",
            quality_score=0.9,
        ),
    )
    assert proposal.candidate_graph.lifecycle_status is SkillLifecycleStatus.CANDIDATE
    assert proposal.candidate_graph.validation_status is ValidationStatus.UNVALIDATED
    assert proposal.candidate_graph.parent_version == "1.0.0"
    assert (
        proposal.candidate_graph.node_by_id("stroke").arguments["motion_profile_id"]
        == "linear_expert"
    )
    assert parent.node_by_id("stroke").arguments["motion_profile_id"] == "linear_slow"
    assert UpdateComponent.TIMING_PROFILE in proposal.components


def test_force_profile_is_preserved_without_force_evidence() -> None:
    parent = active_contact_graph()
    proposal = SkillUpdater().propose(
        parent,
        UpdateEvidence(
            source_demonstration_id="expert_rgbd_only",
            operator_role="expert",
            requested_force_profile_id="wipe_aggressive",
        ),
    )
    assert proposal.force_profile_preserved is True
    assert proposal.candidate_graph.force_profiles == ["wipe_standard"]
    assert (
        proposal.candidate_graph.node_by_id("enable").arguments["force_profile_id"]
        == "wipe_standard"
    )
    assert UpdateComponent.FORCE_PROFILE not in proposal.components


def test_materially_different_route_becomes_variant() -> None:
    parent = active_contact_graph()
    proposal = SkillUpdater(route_variant_threshold_m=0.05).propose(
        parent,
        UpdateEvidence(
            source_demonstration_id="right_route_demo",
            operator_style="right_route",
            normalized_path_m=[(0.0, 0.2, 0.0), (0.2, 0.2, 0.0)],
        ),
        parent_normalized_path_m=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0)],
    )
    assert proposal.variant == "right_route"
    assert proposal.route_distance_m == pytest.approx(0.2)
    assert any("style variant" in reason for reason in proposal.rationale)


def test_version_transitions_cannot_skip_validation() -> None:
    assert next_candidate_version("1.0.0") == "1.1.0-candidate"
    with pytest.raises(ValueError, match="invalid skill lifecycle transition"):
        ensure_transition(SkillLifecycleStatus.CANDIDATE, SkillLifecycleStatus.ACTIVE)
    ensure_transition(SkillLifecycleStatus.CANDIDATE, SkillLifecycleStatus.VALIDATED)
