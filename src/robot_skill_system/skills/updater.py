"""Candidate-only expert update engine with independent component decisions."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from robot_skill_system.skills.models import (
    SkillGraph,
    SkillLifecycleStatus,
    ValidationStatus,
)
from robot_skill_system.skills.versioning import next_candidate_version


class UpdateComponent(str, Enum):
    SPATIAL_PATH = "spatial_path"
    ORIENTATION_PATH = "orientation_path"
    TIMING_PROFILE = "timing_profile"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    BLEND = "blend"
    GRIPPER_PROFILE = "gripper_profile"
    FORCE_PROFILE = "force_profile"
    RECOVERY_POLICY = "recovery_policy"
    WORKSPACE_CONDITIONS = "workspace_conditions"


class UpdateEvidence(BaseModel):
    """Locally measured and operator-approved evidence for one new demonstration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_demonstration_id: str
    operator_role: str = "operator"
    operator_style: str = "normal"
    successful: bool = True
    normalized_path_m: list[tuple[float, float, float]] = Field(default_factory=list)
    route_label: str | None = None
    timing_motion_profile_id: str | None = None
    gripper_profile_id: str | None = None
    requested_force_profile_id: str | None = None
    has_robot_force_log: bool = False
    has_ft_sensor_log: bool = False
    has_force_instrumented_tool: bool = False
    operator_approved_force_profile: bool = False
    node_argument_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def has_force_authority(self) -> bool:
        """Return whether force-profile updates have valid evidence/approval."""

        return any(
            (
                self.has_robot_force_log,
                self.has_ft_sensor_log,
                self.has_force_instrumented_tool,
                self.operator_approved_force_profile,
            )
        )


class SkillUpdateProposal(BaseModel):
    """Auditable result of comparing a new demonstration to an immutable parent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    skill_id: str
    parent_version: str
    candidate_version: str
    variant: str
    components: list[UpdateComponent]
    force_profile_preserved: bool
    route_distance_m: float | None = None
    rationale: list[str] = Field(default_factory=list)
    candidate_graph: SkillGraph


def _resample_path(
    path: list[tuple[float, float, float]], sample_count: int
) -> list[tuple[float, float, float]]:
    if not path or sample_count <= 0:
        return []
    if len(path) == 1:
        return [path[0]] * sample_count
    result: list[tuple[float, float, float]] = []
    for index in range(sample_count):
        position = index * (len(path) - 1) / max(1, sample_count - 1)
        left = int(math.floor(position))
        right = min(left + 1, len(path) - 1)
        weight = position - left
        result.append(
            tuple(
                path[left][axis] * (1.0 - weight) + path[right][axis] * weight
                for axis in range(3)
            )  # type: ignore[arg-type]
        )
    return result


def path_distance_m(
    left: list[tuple[float, float, float]], right: list[tuple[float, float, float]]
) -> float | None:
    """Return mean normalized point distance for route-variant detection."""

    if not left or not right:
        return None
    count = max(len(left), len(right), 2)
    left_sampled = _resample_path(left, count)
    right_sampled = _resample_path(right, count)
    distances = [
        math.dist(left_sampled[index], right_sampled[index]) for index in range(count)
    ]
    return sum(distances) / len(distances)


class SkillUpdater:
    """Build a child candidate; never mutate or activate the parent graph."""

    def __init__(self, *, route_variant_threshold_m: float = 0.08) -> None:
        if route_variant_threshold_m <= 0:
            raise ValueError("route_variant_threshold_m must be positive")
        self.route_variant_threshold_m = route_variant_threshold_m

    def propose(
        self,
        parent: SkillGraph,
        evidence: UpdateEvidence,
        *,
        parent_normalized_path_m: list[tuple[float, float, float]] | None = None,
    ) -> SkillUpdateProposal:
        """Create an unvalidated candidate using only explicitly authorized updates."""

        if not evidence.successful:
            raise ValueError("failed demonstrations cannot create candidates")
        distance = path_distance_m(parent_normalized_path_m or [], evidence.normalized_path_m)
        route_changed = bool(
            (distance is not None and distance > self.route_variant_threshold_m)
            or (
                evidence.route_label is not None
                and parent.operator_style is not None
                and evidence.route_label != parent.operator_style
            )
        )
        variant = evidence.operator_style if route_changed else (parent.operator_style or "default")
        components: list[UpdateComponent] = []
        rationale: list[str] = []
        updated_nodes = []
        allowed_node_updates = evidence.node_argument_updates
        for node in parent.nodes:
            arguments = dict(node.arguments)
            node_update = allowed_node_updates.get(node.node_id, {})
            if node_update:
                arguments.update(node_update)
                components.append(UpdateComponent.SPATIAL_PATH)
            if (
                evidence.timing_motion_profile_id
                and node.operation
                in {"motion.move_l", "motion.move_spline", "motion.move_sx"}
            ):
                arguments["motion_profile_id"] = evidence.timing_motion_profile_id
                components.extend(
                    [
                        UpdateComponent.TIMING_PROFILE,
                        UpdateComponent.VELOCITY,
                        UpdateComponent.ACCELERATION,
                    ]
                )
            if (
                evidence.requested_force_profile_id
                and evidence.has_force_authority
                and node.operation
                in {"contact.search_surface", "contact.enable_force", "contact.verify_force"}
            ):
                arguments["force_profile_id"] = evidence.requested_force_profile_id
                components.append(UpdateComponent.FORCE_PROFILE)
            updated_nodes.append(node.model_copy(update={"arguments": arguments}, deep=True))

        parent_uses_force = any(
            node.operation == "contact.enable_force" for node in parent.nodes
        )
        force_preserved = parent_uses_force and not evidence.has_force_authority
        if force_preserved:
            rationale.append(
                "force profile preserved because no force evidence or operator approval exists"
            )
        if route_changed:
            rationale.append(
                "material route difference retained as a style variant instead of averaging"
            )
        if evidence.operator_role == "expert":
            rationale.append("expert role is metadata; measured quality remains authoritative")
        if evidence.gripper_profile_id:
            components.append(UpdateComponent.GRIPPER_PROFILE)
        motion_profiles = list(parent.motion_profiles)
        if (
            evidence.timing_motion_profile_id
            and evidence.timing_motion_profile_id not in motion_profiles
        ):
            motion_profiles.append(evidence.timing_motion_profile_id)
        force_profiles = list(parent.force_profiles)
        if (
            evidence.requested_force_profile_id
            and evidence.has_force_authority
            and evidence.requested_force_profile_id not in force_profiles
        ):
            force_profiles.append(evidence.requested_force_profile_id)
        candidate = parent.model_copy(
            update={
                "version": next_candidate_version(parent.version),
                "parent_version": parent.version,
                "source_demonstrations": [
                    *parent.source_demonstrations,
                    evidence.source_demonstration_id,
                ],
                "operator_style": variant,
                "nodes": updated_nodes,
                "motion_profiles": motion_profiles,
                "force_profiles": force_profiles,
                "uncertainty": {
                    **parent.uncertainty,
                    **(
                        {"candidate_gripper_profile_id": evidence.gripper_profile_id}
                        if evidence.gripper_profile_id
                        else {}
                    ),
                    "candidate_quality_score": evidence.quality_score,
                },
                "validation_status": ValidationStatus.UNVALIDATED,
                "lifecycle_status": SkillLifecycleStatus.CANDIDATE,
            },
            deep=True,
        )
        # Deduplicate while retaining a stable component order.
        unique_components = list(dict.fromkeys(components))
        return SkillUpdateProposal(
            skill_id=parent.skill_id,
            parent_version=parent.version,
            candidate_version=candidate.version,
            variant=variant,
            components=unique_components,
            force_profile_preserved=force_preserved,
            route_distance_m=distance,
            rationale=rationale,
            candidate_graph=candidate,
        )
