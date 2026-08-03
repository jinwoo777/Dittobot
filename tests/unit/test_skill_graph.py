"""SkillGraph topology, profile, anchor, and contact safety tests."""

from __future__ import annotations

import pytest

from robot_skill_system.exceptions import SkillGraphValidationError
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.skills.models import SkillEdge, SkillGraph, SkillNode, SkillType


def relative_target(anchor_id: str = "$surface") -> dict[str, object]:
    return {
        "anchor_id": anchor_id,
        "anchor_type": "surface",
        "position_m": {"x": 0.0, "y": 0.0, "z": 0.02},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def motion_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="move_demo",
        version="1.0.0",
        name="move demo",
        description="A simple anchored linear move.",
        skill_type=SkillType.MOTION,
        source_demonstrations=["demo_01"],
        bindings={"$surface": {"entity_kind": "surface"}},
        nodes=[
            SkillNode(
                node_id="validate",
                operation="workspace.validate_target",
                arguments={"target": relative_target()},
            ),
            SkillNode(
                node_id="move",
                operation="motion.move_l",
                arguments={
                    "target": relative_target(),
                    "motion_profile_id": "linear_normal",
                },
            ),
        ],
        edges=[SkillEdge(source_node="validate", target_node="move")],
        start_node="validate",
        terminal_nodes=["move"],
        motion_profiles=["linear_normal"],
    )


def contact_graph(*, close_force: bool = True, declare_force_profile: bool = True) -> SkillGraph:
    nodes = [
        SkillNode(
            node_id="search",
            operation="contact.search_surface",
            arguments={"surface": "$surface", "force_profile_id": "wipe_light"},
        ),
        SkillNode(
            node_id="enable",
            operation="contact.enable_force",
            arguments={"surface": "$surface", "force_profile_id": "wipe_light"},
        ),
        SkillNode(
            node_id="wipe",
            operation="contact.follow_path",
            arguments={
                "path": [
                    relative_target(),
                    {
                        **relative_target(),
                        "position_m": {"x": 0.2, "y": 0.0, "z": 0.02},
                    },
                ],
                "motion_profile_id": "linear_slow",
            },
        ),
    ]
    if close_force:
        nodes.append(
            SkillNode(node_id="disable", operation="contact.disable_force", arguments={})
        )
    node_ids = [node.node_id for node in nodes]
    return SkillGraph(
        skill_id="wipe_demo",
        version="1.0.0",
        name="wipe demo",
        description="A supervised anchored contact path.",
        skill_type=SkillType.CONTACT,
        source_demonstrations=["demo_contact_01"],
        bindings={"$surface": {"entity_kind": "surface", "role": "contact_target"}},
        nodes=nodes,
        edges=[
            SkillEdge(source_node=source, target_node=target)
            for source, target in zip(node_ids, node_ids[1:], strict=False)
        ],
        start_node="search",
        terminal_nodes=[node_ids[-1]],
        motion_profiles=["linear_slow"],
        force_profiles=["wipe_light"] if declare_force_profile else [],
    )


def test_valid_graph_returns_normalized_arguments_and_order() -> None:
    report = SkillGraphValidator().validate(motion_graph())

    assert report.valid
    assert report.topological_node_ids == ("validate", "move")
    assert report.normalized_arguments["move"]["motion_profile_id"] == "linear_normal"


def test_force_graph_requires_enable_and_disable_on_terminal_paths() -> None:
    assert SkillGraphValidator().validate(contact_graph()).valid

    with pytest.raises(SkillGraphValidationError, match="disable_force|force mode active"):
        SkillGraphValidator().validate(contact_graph(close_force=False))


def test_missing_force_profile_is_rejected() -> None:
    with pytest.raises(SkillGraphValidationError, match="undeclared force profile"):
        SkillGraphValidator().validate(contact_graph(declare_force_profile=False))


def test_base_absolute_target_is_rejected() -> None:
    graph = motion_graph()
    graph.nodes[1].arguments["target"] = relative_target("base_link")

    with pytest.raises(SkillGraphValidationError, match="absolute target|absolute anchors"):
        SkillGraphValidator().validate(graph)


def test_unknown_operation_is_rejected() -> None:
    graph = motion_graph()
    graph.nodes[1].operation = "motion.unknown"

    with pytest.raises(SkillGraphValidationError, match="whitelist"):
        SkillGraphValidator().validate(graph)


def test_cycle_is_rejected() -> None:
    graph = motion_graph()
    graph.nodes.append(
        SkillNode(node_id="finish", operation="motion.wait", arguments={"duration_s": 0.1})
    )
    graph.edges = [
        SkillEdge(source_node="validate", target_node="move"),
        SkillEdge(source_node="move", target_node="validate"),
    ]
    graph.terminal_nodes = ["finish"]

    with pytest.raises(SkillGraphValidationError, match="cycle"):
        SkillGraphValidator().validate(graph)
