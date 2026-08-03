"""Registry-aware topology and safety validation for SkillGraphs."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from robot_skill_system.exceptions import (
    PrimitiveValidationError,
    SkillGraphValidationError,
)
from robot_skill_system.primitives.registry import PrimitiveRegistry, get_default_registry
from robot_skill_system.skills.models import (
    SkillGraph,
    SkillType,
    TransitionCondition,
)

_FORBIDDEN_ABSOLUTE_FRAMES = frozenset(
    {"base", "base_link", "robot_base", "robot_base_link", "world", "map"}
)
_ABSOLUTE_KEY_PARTS = (
    "absolute_pose",
    "base_pose",
    "base_target",
    "target_in_base",
    "robot_base",
)


@dataclass(frozen=True)
class NodeTransitions:
    """Resolved result targets for one node."""

    success: str | None
    failure: str | None

    def targets(self) -> tuple[str, ...]:
        """Return unique concrete targets."""

        return tuple(dict.fromkeys(target for target in (self.success, self.failure) if target))


@dataclass(frozen=True)
class SkillGraphValidationReport:
    """Complete local validation outcome and compiler-normalized values."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    topological_node_ids: tuple[str, ...] = ()
    transitions: Mapping[str, NodeTransitions] = field(default_factory=dict)
    normalized_arguments: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    resolved_timeouts_s: Mapping[str, float] = field(default_factory=dict)


def _edge_transitions(
    graph: SkillGraph,
) -> tuple[dict[str, NodeTransitions], list[str]]:
    errors: list[str] = []
    node_ids = {node.node_id for node in graph.nodes}
    per_node: dict[str, dict[TransitionCondition, str]] = {
        node_id: {} for node_id in node_ids
    }
    for edge in graph.edges:
        if edge.source_node not in node_ids or edge.target_node not in node_ids:
            errors.append(
                f"edge {edge.source_node!r}->{edge.target_node!r} references an unknown node"
            )
            continue
        existing = per_node[edge.source_node].get(edge.condition)
        if existing is not None and existing != edge.target_node:
            errors.append(
                f"node {edge.source_node!r} has multiple {edge.condition.value} edges"
            )
        per_node[edge.source_node][edge.condition] = edge.target_node

    transitions: dict[str, NodeTransitions] = {}
    terminals = set(graph.terminal_nodes)
    for node in graph.nodes:
        configured = per_node[node.node_id]
        always = configured.get(TransitionCondition.ALWAYS)
        edge_success = always or configured.get(TransitionCondition.SUCCESS)
        edge_failure = always or configured.get(TransitionCondition.FAILURE)
        if node.on_success is not None and edge_success not in (None, node.on_success):
            errors.append(f"node {node.node_id!r} has conflicting success transitions")
        if node.on_failure is not None and edge_failure not in (None, node.on_failure):
            errors.append(f"node {node.node_id!r} has conflicting failure transitions")
        success = node.on_success or edge_success
        failure = node.on_failure or edge_failure
        for label, target in (("success", success), ("failure", failure)):
            if target is not None and target not in node_ids:
                errors.append(
                    f"node {node.node_id!r} {label} transition references unknown node {target!r}"
                )
        if node.node_id in terminals:
            if success is not None or failure is not None:
                errors.append(f"terminal node {node.node_id!r} cannot have outgoing transitions")
        elif success is None:
            errors.append(f"non-terminal node {node.node_id!r} needs a success transition")
        transitions[node.node_id] = NodeTransitions(success=success, failure=failure)
    return transitions, errors


def _topological_order_from_transitions(
    graph: SkillGraph, transitions: Mapping[str, NodeTransitions]
) -> tuple[tuple[str, ...], list[str]]:
    node_ids = {node.node_id for node in graph.nodes}
    adjacency = {node_id: transitions[node_id].targets() for node_id in node_ids}
    indegree = dict.fromkeys(node_ids, 0)
    for targets in adjacency.values():
        for target in targets:
            if target in indegree:
                indegree[target] += 1
    ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    ordered: list[str] = []
    while ready:
        node_id = ready.popleft()
        ordered.append(node_id)
        for target in sorted(adjacency[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    errors: list[str] = []
    if len(ordered) != len(node_ids):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        errors.append(f"SkillGraph contains a cycle involving {cyclic}")

    reachable: set[str] = set()
    pending = [graph.start_node]
    while pending:
        node_id = pending.pop()
        if node_id in reachable or node_id not in adjacency:
            continue
        reachable.add(node_id)
        pending.extend(adjacency[node_id])
    unreachable = sorted(node_ids - reachable)
    if unreachable:
        errors.append(f"SkillGraph contains unreachable nodes {unreachable}")
    unreachable_terminals = sorted(set(graph.terminal_nodes) - reachable)
    if unreachable_terminals:
        errors.append(f"terminal nodes are unreachable: {unreachable_terminals}")
    return tuple(ordered), errors


def topological_order(graph: SkillGraph) -> tuple[str, ...]:
    """Return a stable topological order or raise on invalid topology."""

    transitions, transition_errors = _edge_transitions(graph)
    order, order_errors = _topological_order_from_transitions(graph, transitions)
    errors = transition_errors + order_errors
    if errors:
        raise SkillGraphValidationError("; ".join(errors))
    return order


def _find_absolute_target(value: Any, path: str = "arguments") -> str | None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            item_path = f"{path}.{raw_key}"
            if any(fragment in key for fragment in _ABSOLUTE_KEY_PARTS):
                return item_path
            if (
                key in {"frame_id", "anchor_id", "anchor_frame_id"}
                and isinstance(item, str)
                and item.casefold() in _FORBIDDEN_ABSOLUTE_FRAMES
            ):
                return item_path
            nested = _find_absolute_target(item, item_path)
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            nested = _find_absolute_target(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _validate_force_paths(
    graph: SkillGraph,
    transitions: Mapping[str, NodeTransitions],
) -> list[str]:
    errors: list[str] = []
    node_by_id = {node.node_id: node for node in graph.nodes}
    enable_nodes = {
        node.node_id for node in graph.nodes if node.operation == "contact.enable_force"
    }
    disable_nodes = {
        node.node_id for node in graph.nodes if node.operation == "contact.disable_force"
    }
    if graph.skill_type is SkillType.CONTACT:
        if not enable_nodes:
            errors.append("contact skills require contact.enable_force")
        if not disable_nodes:
            errors.append("contact skills require contact.disable_force")
    if not enable_nodes and not disable_nodes:
        return errors

    terminal_nodes = set(graph.terminal_nodes)
    visited: set[tuple[str, bool, bool]] = set()
    pending: list[tuple[str, bool, bool]] = [(graph.start_node, False, False)]
    while pending:
        node_id, force_active, contact_found = pending.pop()
        state = (node_id, force_active, contact_found)
        if state in visited:
            continue
        visited.add(state)
        node = node_by_id[node_id]
        next_force_active = force_active
        next_contact_found = contact_found
        if node.operation == "contact.search_surface":
            next_contact_found = True
        elif node.operation == "contact.enable_force":
            if force_active:
                errors.append(f"node {node_id!r} enables force while force mode is already active")
            if not contact_found:
                errors.append(
                    f"node {node_id!r} enables force without a preceding successful contact search"
                )
            next_force_active = True
        elif node.operation == "contact.disable_force":
            # Disable is intentionally idempotent so an explicit failure cleanup
            # path remains safe when enable_force itself reports failure.
            next_force_active = False
        elif node.operation in {
            "contact.follow_path",
            "contact.follow_surface_path",
            "contact.verify_force",
        } and not force_active:
            errors.append(f"node {node_id!r} requires active force mode")

        if node_id in terminal_nodes:
            if next_force_active:
                errors.append(f"terminal node {node_id!r} is reachable with force mode active")
            continue
        transition = transitions[node_id]
        if transition.success is not None:
            pending.append((transition.success, next_force_active, next_contact_found))
        # A declared failure route is part of the graph and must also clean up.
        # An absent failure route is handled by the fixed runtime supervisor.
        if transition.failure is not None:
            pending.append((transition.failure, next_force_active, contact_found))
    return sorted(set(errors))


class SkillGraphValidator:
    """Validate topology, whitelist arguments, profiles, and contact invariants."""

    def __init__(self, registry: PrimitiveRegistry | None = None) -> None:
        self.registry = registry or get_default_registry()

    def inspect(self, graph: SkillGraph | Mapping[str, Any]) -> SkillGraphValidationReport:
        """Return all deterministic validation findings without raising."""

        if not isinstance(graph, SkillGraph):
            try:
                graph = SkillGraph.model_validate(graph)
            except Exception as error:  # Pydantic supplies a complete structured message.
                return SkillGraphValidationReport(False, (str(error),), ())

        errors: list[str] = []
        warnings: list[str] = []
        transitions, transition_errors = _edge_transitions(graph)
        errors.extend(transition_errors)
        order, order_errors = _topological_order_from_transitions(graph, transitions)
        errors.extend(order_errors)
        normalized_arguments: dict[str, Mapping[str, Any]] = {}
        resolved_timeouts: dict[str, float] = {}
        for node in graph.nodes:
            absolute_path = _find_absolute_target(node.arguments)
            if absolute_path is not None:
                errors.append(
                    f"node {node.node_id!r} persists a robot-base/world absolute target at "
                    f"{absolute_path}"
                )
            try:
                metadata = self.registry.metadata(node.operation)
                if graph.skill_type.value not in metadata.allowed_skill_types:
                    errors.append(
                        f"operation {node.operation!r} is not allowed for skill type "
                        f"{graph.skill_type.value!r}"
                    )
                validated = self.registry.validate_arguments(node.operation, node.arguments)
                normalized = validated.model_dump(mode="json", exclude_none=True)
                normalized_arguments[node.node_id] = normalized
                resolved_timeouts[node.node_id] = self.registry.validate_timeout(
                    node.operation, node.timeout_s
                )
                motion_profile = normalized.get("motion_profile_id")
                if (
                    isinstance(motion_profile, str)
                    and motion_profile not in graph.motion_profiles
                ):
                    errors.append(
                        f"node {node.node_id!r} uses undeclared motion profile "
                        f"{motion_profile!r}"
                    )
                force_profile = normalized.get("force_profile_id")
                if isinstance(force_profile, str) and force_profile not in graph.force_profiles:
                    errors.append(
                        f"node {node.node_id!r} uses undeclared force profile "
                        f"{force_profile!r}"
                    )
            except PrimitiveValidationError as error:
                errors.append(f"node {node.node_id!r}: {error}")

        if not transition_errors and not order_errors:
            errors.extend(_validate_force_paths(graph, transitions))
        uses_force = any(
            node.operation.startswith("contact.")
            and node.operation
            in {
                "contact.enable_force",
                "contact.follow_path",
                "contact.follow_surface_path",
                "contact.verify_force",
                "contact.disable_force",
            }
            for node in graph.nodes
        )
        if uses_force and not any(
            requirement in {"force_supervisor", "global_force_supervisor"}
            for requirement in graph.global_policy_requirements
        ):
            errors.append("force graphs must require the global force supervisor")
        if not graph.source_demonstrations:
            warnings.append("skill has no source demonstration metadata")
        unique_errors = tuple(dict.fromkeys(errors))
        return SkillGraphValidationReport(
            valid=not unique_errors,
            errors=unique_errors,
            warnings=tuple(warnings),
            topological_node_ids=order if not order_errors else (),
            transitions=transitions,
            normalized_arguments=normalized_arguments,
            resolved_timeouts_s=resolved_timeouts,
        )

    def validate(self, graph: SkillGraph | Mapping[str, Any]) -> SkillGraphValidationReport:
        """Return the validated report or raise ``SkillGraphValidationError``."""

        report = self.inspect(graph)
        if not report.valid:
            raise SkillGraphValidationError("; ".join(report.errors))
        return report


def validate_skill_graph(
    graph: SkillGraph | Mapping[str, Any],
    registry: PrimitiveRegistry | None = None,
) -> SkillGraphValidationReport:
    """Convenience function for registry-aware graph validation."""

    return SkillGraphValidator(registry).validate(graph)


def transitions_for(graph: SkillGraph) -> Mapping[str, NodeTransitions]:
    """Return transitions after validating graph topology."""

    transitions, errors = _edge_transitions(graph)
    if errors:
        raise SkillGraphValidationError("; ".join(errors))
    return transitions
