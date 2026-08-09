from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from robot_skill_system.adapters import (
    DoosanM0609Adapter,
    HardwareExecutionDisabledError,
    MockRobotAdapter,
    NotConfiguredError,
)
from robot_skill_system.primitives.profiles import load_force_profiles, load_motion_profiles
from robot_skill_system.primitives.registry import PrimitiveRegistry
from robot_skill_system.runtime import (
    EntityBinder,
    EntityKind,
    EntityRequirement,
    GlobalWorkspaceSupervisor,
    MockObstacleMonitor,
    ObstacleDetectedError,
    ObstacleObservation,
    RuntimeOrchestrator,
    SceneStaleError,
    SkillHashMismatchError,
    canonical_skill_checksum_sha256,
)
from robot_skill_system.scene.models import (
    ConfidenceSummary,
    Pose,
    Quaternion,
    SceneSnapshot,
    SurfaceInstance,
    ToolInstance,
    Vector3,
)
from robot_skill_system.skills.models import SkillGraph

NOW_NS = 1_000_000_000_000
ROOT = Path(__file__).resolve().parents[2]


def _pose(x: float = 0.5, y: float = 0.0, z: float = 0.7) -> Pose:
    return Pose(
        frame_id="base",
        position_m=Vector3(x=x, y=y, z=z),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        timestamp_ns=NOW_NS,
        source="mock_fixture",
        confidence=0.99,
    )


def _scene(
    *, timestamp_ns: int = NOW_NS, tool_confidence: float = 0.99, tool_class: str = "wiper"
) -> SceneSnapshot:
    return SceneSnapshot(
        schema_version="1.0",
        scene_id="scene_runtime",
        timestamp_ns=timestamp_ns,
        reference_frame="base",
        valid_for_ms=5_000,
        tools=[
            ToolInstance(
                instance_id="wiper_01",
                tool_class=tool_class,
                attached=True,
                tcp_frame="wiper_01_tcp",
                pose=_pose(),
                compatible_skills=["wipe_surface"],
                verification_confidence=tool_confidence,
            )
        ],
        surfaces=[
            SurfaceInstance(
                instance_id="table_01",
                role="contact_target",
                center_m=Vector3(x=0.5, y=0.0, z=0.7),
                normal=Vector3(x=0.0, y=0.0, z=1.0),
                confidence=0.99,
                allowed_contact_links=["wiper_01_tcp"],
            )
        ],
        calibration_id="calibration_01",
        confidence_summary=ConfidenceSummary(overall=0.99),
    )


def _relative(x: float, z: float = 0.0) -> dict[str, Any]:
    return {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": x, "y": 0.0, "z": z},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _motion_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="wipe_motion",
        version="1.0.0",
        name="move over surface",
        description="One anchor-relative mock linear motion.",
        skill_type="motion",
        bindings={
            "$surface": {
                "entity_kind": "surface",
                "instance_id": "table_01",
                "role": "contact_target",
            }
        },
        nodes=[
            {
                "node_id": "move",
                "operation": "motion.move_l",
                "arguments": {
                    "target": _relative(0.1, 0.1),
                    "motion_profile_id": "linear_slow",
                },
            }
        ],
        start_node="move",
        terminal_nodes=["move"],
        motion_profiles=["linear_slow"],
        validation_status="passed",
        lifecycle_status="active",
    )


def _contact_graph() -> SkillGraph:
    nodes = [
        {
            "node_id": "search",
            "operation": "contact.search_surface",
            "arguments": {"surface": "$surface", "force_profile_id": "wipe_light"},
        },
        {
            "node_id": "enable",
            "operation": "contact.enable_force",
            "arguments": {"surface": "$surface", "force_profile_id": "wipe_light"},
        },
        {
            "node_id": "follow",
            "operation": "contact.follow_path",
            "arguments": {
                "path": [_relative(0.05), _relative(0.1)],
                "motion_profile_id": "linear_slow",
            },
        },
        {"node_id": "disable", "operation": "contact.disable_force", "arguments": {}},
    ]
    edges = [
        {"source_node": "search", "target_node": "enable"},
        {"source_node": "enable", "target_node": "follow"},
        {"source_node": "follow", "target_node": "disable"},
    ]
    return SkillGraph(
        skill_id="wipe_contact",
        version="1.0.0",
        name="wipe contact",
        description="Contact wipe with approved profiles.",
        skill_type="contact",
        required_tools=["wiper"],
        bindings={
            "$tool": {
                "entity_kind": "tool",
                "instance_id": "wiper_01",
                "class_name": "wiper",
                "must_be_attached": True,
            },
            "$surface": {
                "entity_kind": "surface",
                "instance_id": "table_01",
                "role": "contact_target",
            },
        },
        nodes=nodes,
        edges=edges,
        start_node="search",
        terminal_nodes=["disable"],
        motion_profiles=["linear_slow"],
        force_profiles=["wipe_light"],
        validation_status="passed",
        lifecycle_status="active",
    )


def _branched_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="branched_motion",
        version="1.0.0",
        name="branched motion",
        description="Run exactly one result-conditioned branch.",
        skill_type="composite",
        bindings={
            "$surface": {
                "entity_kind": "surface",
                "instance_id": "table_01",
                "role": "contact_target",
            }
        },
        nodes=[
            {
                "node_id": "choose",
                "operation": "motion.wait",
                "arguments": {"duration_s": 0.001},
                "on_success": "success_move",
                "on_failure": "failure_abort",
            },
            {
                "node_id": "success_move",
                "operation": "motion.move_l",
                "arguments": {
                    "target": _relative(0.1, 0.1),
                    "motion_profile_id": "linear_slow",
                },
            },
            {
                "node_id": "failure_abort",
                "operation": "recovery.abort",
                "arguments": {},
            },
        ],
        start_node="choose",
        terminal_nodes=["success_move", "failure_abort"],
        motion_profiles=["linear_slow"],
        validation_status="passed",
        lifecycle_status="active",
    )


@pytest.fixture
def profiles() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_motion_profiles(ROOT / "configs/motion_profiles/default.json"),
        load_force_profiles(ROOT / "configs/force_profiles/default.json"),
    )


def test_binder_rejects_stale_low_confidence_and_ambiguous_entities() -> None:
    binder = EntityBinder(clock_ns=lambda: NOW_NS)
    requirement = EntityRequirement(
        "$tool", EntityKind.TOOL, class_name="wiper", minimum_confidence=0.8
    )

    with pytest.raises(SceneStaleError):
        binder.bind_entities(_scene(timestamp_ns=NOW_NS - 6_000_000_000), [requirement])
    with pytest.raises(Exception, match="no scene entity"):
        binder.bind_entities(_scene(tool_confidence=0.4), [requirement])


def test_relative_target_is_rebound_to_current_surface() -> None:
    scene = _scene()
    binder = EntityBinder(clock_ns=lambda: NOW_NS)
    bindings = binder.bind_entities(
        scene,
        [EntityRequirement("$surface", EntityKind.SURFACE, instance_id="table_01")],
    )

    target = binder.bind_pose(_relative(0.2, 0.1), scene=scene, bindings=bindings)

    assert target.anchor_entity_id == "table_01"
    assert target.position_m == pytest.approx((0.7, 0.0, 0.8))


def test_relative_target_rotates_with_calibrated_surface_frame() -> None:
    scene = _scene()
    rotated_surface = scene.surfaces[0].model_copy(
        update={
            "pose": Pose(
                frame_id="base",
                position_m=Vector3(x=0.5, y=0.0, z=0.7),
                orientation_xyzw=Quaternion(
                    x=0.0,
                    y=0.0,
                    z=2**-0.5,
                    w=2**-0.5,
                ),
                timestamp_ns=NOW_NS,
                source="calibrated_surface_fixture",
                confidence=0.99,
            )
        }
    )
    scene = scene.model_copy(update={"surfaces": [rotated_surface]})
    binder = EntityBinder(clock_ns=lambda: NOW_NS)
    bindings = binder.bind_entities(
        scene,
        [EntityRequirement("$surface", EntityKind.SURFACE, instance_id="table_01")],
    )

    target = binder.bind_pose(_relative(0.2, 0.1), scene=scene, bindings=bindings)

    assert target.position_m == pytest.approx((0.5, 0.2, 0.8))
    assert target.orientation_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))


def test_tool_mismatch_and_skill_checksum_are_rejected(
    profiles: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    motion_profiles, force_profiles = profiles
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    mismatch = asyncio.run(
        runtime.run(command_text="wipe", skill=_contact_graph(), scene=_scene(tool_class="brush"))
    )
    assert not mismatch.success
    runtime.state_machine.reset()
    checksum_result = asyncio.run(
        runtime.run(
            command_text="move",
            skill=_motion_graph(),
            scene=_scene(),
            expected_skill_checksum_sha256="0" * 64,
        )
    )
    assert not checksum_result.success
    assert checksum_result.error_code == SkillHashMismatchError.code


def test_hardware_requires_all_flags_and_verified_vendor_binding(
    profiles: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    motion_profiles, force_profiles = profiles
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="hardware",
        enable_hardware_execution=False,
        robot_backend="mock",
        enable_real_robot=False,
        dry_run=True,
        clock_ns=lambda: NOW_NS,
    )
    result = asyncio.run(runtime.run(command_text="move", skill=_motion_graph(), scene=_scene()))
    assert not result.success
    assert robot.commands == []

    fully_flagged_robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    fully_flagged_runtime = RuntimeOrchestrator(
        robot=fully_flagged_robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="hardware",
        enable_hardware_execution=True,
        robot_backend="doosan",
        enable_real_robot=True,
        dry_run=False,
        clock_ns=lambda: NOW_NS,
    )
    monitor_result = asyncio.run(
        fully_flagged_runtime.run(
            command_text="move", skill=_motion_graph(), scene=_scene()
        )
    )
    assert not monitor_result.success
    assert "verified dynamic obstacle" in (monitor_result.error_message or "")
    assert fully_flagged_robot.commands == []

    with pytest.raises(HardwareExecutionDisabledError):
        DoosanM0609Adapter().connect()
    with pytest.raises(NotConfiguredError):
        DoosanM0609Adapter(
            execution_mode="hardware", hardware_enabled=True
        ).connect()


class _DelayedObstacleMonitor(MockObstacleMonitor):
    def __init__(self, observation: ObstacleObservation, trigger_poll: int) -> None:
        super().__init__()
        self.observation = observation
        self.trigger_poll = trigger_poll
        self.poll_count = 0

    def poll(self) -> ObstacleObservation | None:
        self.poll_count += 1
        return self.observation if self.poll_count == self.trigger_poll else None


def test_force_obstacle_stops_releases_compliance_then_retracts(
    profiles: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    motion_profiles, force_profiles = profiles
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    monitor = _DelayedObstacleMonitor(
        ObstacleObservation("person_01", NOW_NS, distance_m=0.1), trigger_poll=5
    )
    workspace = GlobalWorkspaceSupervisor(monitor)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        workspace_supervisor=workspace,
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(runtime.run(command_text="wipe", skill=_contact_graph(), scene=_scene()))

    assert not result.success
    assert result.error_code == ObstacleDetectedError.code
    operations = [command.operation for command in robot.commands]
    cleanup = ["stop", "release_force", "release_compliance", "safe_retract"]
    start = operations.index("stop")
    assert operations[start : start + len(cleanup)] == cleanup
    assert not robot.force_active
    assert not robot.compliance_active


def test_mock_runtime_executes_profile_scaled_motion_offline(
    profiles: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    motion_profiles, force_profiles = profiles
    graph = _motion_graph()
    checksum = canonical_skill_checksum_sha256(graph)
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(
        runtime.run(
            command_text="move",
            skill=graph,
            scene=_scene(),
            expected_skill_checksum_sha256=checksum,
        )
    )

    assert result.success
    move = next(command for command in robot.commands if command.operation == "move_l")
    assert move.arguments["velocity_m_s"] == pytest.approx(0.03 * 0.45)
    assert move.arguments["target_pose"].position_m == pytest.approx((0.6, 0.0, 0.8))


def test_relative_j6_uses_current_feedback_and_preserves_other_joints(
    profiles: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    motion_profiles, force_profiles = profiles
    graph = SkillGraph(
        skill_id="rotate_grip_wrist",
        version="1.0.0",
        name="rotate grip wrist",
        description="Rotate J6 from live feedback.",
        skill_type="motion",
        nodes=[
            {
                "node_id": "rotate",
                "operation": "motion.rotate_joint_6_relative",
                "arguments": {
                    "delta_rad": 0.4,
                    "motion_profile_id": "joint_safe",
                },
            }
        ],
        start_node="rotate",
        terminal_nodes=["rotate"],
        motion_profiles=["joint_safe"],
        validation_status="passed",
        lifecycle_status="active",
    )
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    robot.connect()
    robot.move_j(
        (0.1, -0.2, 0.3, -0.4, 0.5, -0.6),
        velocity_rad_s=0.1,
        acceleration_rad_s2=0.1,
    )
    robot.disconnect()
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(runtime.run(command_text="rotate", skill=graph, scene=_scene()))

    assert result.success
    rotate = [command for command in robot.commands if command.operation == "move_j"][-1]
    assert rotate.arguments["target"] == pytest.approx(
        (0.1, -0.2, 0.3, -0.4, 0.5, -0.2)
    )


def test_graph_runtime_executes_only_the_selected_success_branch(
    profiles: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    motion_profiles, force_profiles = profiles
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(
        runtime.run(command_text="move", skill=_branched_graph(), scene=_scene())
    )

    assert result.success
    assert "move_l" in [command.operation for command in robot.commands]
    assert not any(event.event_type == "primitive_failed" for event in result.events)
