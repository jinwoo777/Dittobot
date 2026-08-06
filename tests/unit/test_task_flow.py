"""Focused tests for deterministic Grip -> Action -> End composition."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from robot_skill_system.adapters import MockGripperAdapter, MockRobotAdapter
from robot_skill_system.primitives import GraspVerificationProfile, PrimitiveRegistry
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import ProfileNotApprovedError
from robot_skill_system.runtime.executor import RuntimeExecutor
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.models import (
    EntityBinding,
    EntityKind,
    ExecutionMode,
    RuntimeContext,
)
from robot_skill_system.runtime.preflight import PreflightValidator
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.skills.models import SkillEdge, SkillGraph
from robot_skill_system.skills.task_flow import (
    ActionDefinition,
    ActionEndMotionMapping,
    AttachmentState,
    EndMotionDefinition,
    ForceModeState,
    GripProfile,
    StageStateContract,
    TaskFlowComposer,
    TaskFlowCompositionError,
)


def _pose(z_m: float) -> dict[str, Any]:
    return {
        "anchor_id": "$object",
        "anchor_type": "object",
        "position_m": {"x": 0.0, "y": 0.0, "z": z_m},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _contract(
    attachment: AttachmentState, force_mode: ForceModeState = ForceModeState.DISABLED
) -> StageStateContract:
    return StageStateContract(attachment=attachment, force_mode=force_mode)


def _grip(**updates: Any) -> GripProfile:
    values: dict[str, Any] = {
        "profile_id": "hammer_default",
        "version": "1.0.0",
        "object_class_id": "hammer",
        "object_frame_policy": "principal_axis",
        "object_frame_revision": "calibration-1",
        "object_binding": "$object",
        "tool_binding": "$gripper",
        "required_tool_class": "rg2",
        "pregrasp_pose": _pose(0.08),
        "grasp_pose": _pose(0.01),
        "jaw_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "approach_axis": {"x": 0.0, "y": 0.0, "z": -1.0},
        "motion_profile_id": "linear_slow",
        "gripper_calibration_profile_id": "rg2_default",
        "verification_profile_id": "grasp_default",
        "lifecycle_status": "active",
        "validation_status": "passed",
    }
    values.update(updates)
    return GripProfile.model_validate(values)


def _action_graph(*, object_class: str = "hammer") -> SkillGraph:
    return SkillGraph(
        skill_id="bring_action_graph",
        version="1.0.0",
        name="bring action",
        description="A minimal locally validated action graph.",
        skill_type="motion",
        source_demonstrations=["bring_demo"],
        bindings={"$object": {"entity_kind": "object", "class_name": object_class}},
        nodes=[
            {"node_id": "perform", "operation": "motion.wait", "arguments": {"duration_s": 0.01}}
        ],
        start_node="perform",
        terminal_nodes=["perform"],
    )


def _action(**updates: Any) -> ActionDefinition:
    values: dict[str, Any] = {
        "action_id": "bring",
        "version": "1.0.0",
        "input_contract": _contract(AttachmentState.HOLDING),
        "output_contract": _contract(AttachmentState.HOLDING),
        "skill_graph": _action_graph(),
        "lifecycle_status": "active",
        "validation_status": "passed",
    }
    values.update(updates)
    return ActionDefinition.model_validate(values)


def _end_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="release_end_graph",
        version="1.0.0",
        name="release end motion",
        description="Open and verify release.",
        skill_type="manipulation",
        source_demonstrations=["bring_demo"],
        bindings={
            "$object": {"entity_kind": "object", "class_name": "hammer"},
            "$gripper": {
                "entity_kind": "tool",
                "class_name": "rg2",
                "must_be_attached": True,
            },
        },
        nodes=[
            {
                "node_id": "open",
                "operation": "gripper.open",
                "arguments": {"tool": "$gripper"},
            },
            {
                "node_id": "verify_release",
                "operation": "grasp.verify_released",
                "arguments": {
                    "object": "$object",
                    "tool": "$gripper",
                    "verification_profile_id": "grasp_default",
                },
            },
        ],
        edges=[SkillEdge(source_node="open", target_node="verify_release")],
        start_node="open",
        terminal_nodes=["verify_release"],
    )


def _end(**updates: Any) -> EndMotionDefinition:
    values: dict[str, Any] = {
        "end_motion_id": "release_safe",
        "version": "1.0.0",
        "input_contract": _contract(AttachmentState.HOLDING),
        "output_contract": _contract(AttachmentState.RELEASED),
        "skill_graph": _end_graph(),
        "lifecycle_status": "active",
        "validation_status": "passed",
    }
    values.update(updates)
    return EndMotionDefinition.model_validate(values)


def _mapping(**updates: Any) -> ActionEndMotionMapping:
    values: dict[str, Any] = {
        "mapping_id": "bring_release",
        "revision": 3,
        "action_id": "bring",
        "action_version": "1.0.0",
        "end_motion_id": "release_safe",
        "end_motion_version": "1.0.0",
    }
    values.update(updates)
    return ActionEndMotionMapping.model_validate(values)


def test_composer_materializes_namespaces_connects_success_only_and_caches() -> None:
    composer = TaskFlowComposer()

    first = composer.compose(
        grip_profile=_grip(), action=_action(), end_motion=_end(), mapping=_mapping()
    )
    second = composer.compose(
        grip_profile=_grip(), action=_action(), end_motion=_end(), mapping=_mapping()
    )

    node_ids = [node.node_id for node in first.graph.nodes]
    assert node_ids[:2] == ["grip__open", "grip__validate_pregrasp"]
    assert "grip__verify_holding" in node_ids
    assert "action__perform" in node_ids
    assert "end__open" in node_ids
    assert first.graph.start_node == "grip__open"
    assert first.graph.terminal_nodes == ["end__verify_release"]
    transitions = SkillGraphValidator().validate(first.graph).transitions
    assert transitions["grip__verify_holding"].success == "action__perform"
    assert transitions["grip__verify_holding"].failure is None
    assert transitions["action__perform"].success == "end__open"
    assert transitions["action__perform"].failure is None
    assert first.manifest == second.manifest
    assert composer.cache_size == 1
    assert len(first.manifest.plan_checksum_sha256) == 64
    assert len(first.manifest.composite_graph_checksum_sha256) == 64


def test_composer_rejects_mapping_contract_binding_profile_and_checksum_conflicts() -> None:
    composer = TaskFlowComposer()
    with pytest.raises(TaskFlowCompositionError, match="mapping does not pin"):
        composer.compose(
            grip_profile=_grip(),
            action=_action(),
            end_motion=_end(),
            mapping=_mapping(action_version="2.0.0"),
        )

    incompatible_action = _action(
        input_contract=_contract(AttachmentState.RELEASED)
    )
    with pytest.raises(TaskFlowCompositionError, match="grip -> action contract conflict"):
        composer.compose(
            grip_profile=_grip(),
            action=incompatible_action,
            end_motion=_end(),
            mapping=_mapping(),
        )

    with pytest.raises(TaskFlowCompositionError, match="conflicting class_name"):
        composer.compose(
            grip_profile=_grip(),
            action=_action(skill_graph=_action_graph(object_class="sponge")),
            end_motion=_end(),
            mapping=_mapping(),
        )

    with pytest.raises(TaskFlowCompositionError, match="revision conflict"):
        composer.compose(
            grip_profile=_grip(
                profile_revisions={"motion:linear_slow": "motion-revision-1"}
            ),
            action=_action(
                profile_revisions={"motion:linear_slow": "motion-revision-2"}
            ),
            end_motion=_end(),
            mapping=_mapping(),
        )

    with pytest.raises(TaskFlowCompositionError, match="checksum mismatch"):
        composer.compose(
            grip_profile=_grip(profile_checksum_sha256="0" * 64),
            action=_action(),
            end_motion=_end(),
            mapping=_mapping(),
        )


def test_composer_rejects_force_enabled_action_boundary_and_non_release_end() -> None:
    composer = TaskFlowComposer()
    with pytest.raises(TaskFlowCompositionError, match="disabled force mode"):
        composer.compose(
            grip_profile=_grip(),
            action=_action(
                output_contract=_contract(
                    AttachmentState.HOLDING, ForceModeState.ENABLED
                )
            ),
            end_motion=_end(
                input_contract=_contract(
                    AttachmentState.HOLDING, ForceModeState.ENABLED
                )
            ),
            mapping=_mapping(),
        )

    with pytest.raises(TaskFlowCompositionError, match="guarantee object release"):
        composer.compose(
            grip_profile=_grip(),
            action=_action(),
            end_motion=_end(output_contract=_contract(AttachmentState.HOLDING)),
            mapping=_mapping(),
        )


def _executor(
    *, approved: bool = True
) -> tuple[RuntimeExecutor, RuntimeContext, MockGripperAdapter]:
    object_entity = {"instance_id": "hammer_01", "class_name": "hammer"}
    tool_entity = {"instance_id": "rg2_01", "tool_class": "rg2", "attached": True}
    bindings = {
        "$object": EntityBinding(
            placeholder="$object",
            entity_id="hammer_01",
            entity_kind=EntityKind.OBJECT,
            confidence=1.0,
            entity=object_entity,
        ),
        "$gripper": EntityBinding(
            placeholder="$gripper",
            entity_id="rg2_01",
            entity_kind=EntityKind.TOOL,
            confidence=1.0,
            entity=tool_entity,
        ),
    }
    profiles = (
        {
            "grasp_default": GraspVerificationProfile(
                profile_id="grasp_default",
                require_holding_signal=True,
                minimum_released_width_m=0.05,
            )
        }
        if approved
        else {}
    )
    context = RuntimeContext(
        scene={"timestamp_ns": 1, "valid_for_ms": 1000},
        bindings=bindings,
        motion_profiles={},
        force_profiles={},
        verification_profiles=profiles,
        execution_mode=ExecutionMode.MOCK,
    )
    robot = MockRobotAdapter(clock_ns=lambda: 1)
    robot.connect()
    gripper = MockGripperAdapter()
    gripper.connect()
    executor = RuntimeExecutor(
        context=context,
        robot=robot,
        gripper=gripper,
        binder=EntityBinder(clock_ns=lambda: 1),
        workspace_supervisor=GlobalWorkspaceSupervisor(),
        force_supervisor=GlobalForceSupervisor(robot, {}),
        safety_supervisor=GlobalSafetySupervisor(),
        primitive_registry=PrimitiveRegistry.default(),
    )
    return executor, context, gripper


def test_mock_grasp_verification_tracks_holding_and_release() -> None:
    executor, context, gripper = _executor()

    async def run() -> None:
        gripper.close()
        await executor.execute_primitive(
            "grasp.verify_holding",
            {
                "object": "$object",
                "tool": "$gripper",
                "verification_profile_id": "grasp_default",
            },
            None,
            None,
        )
        gripper.open()
        await executor.execute_primitive(
            "grasp.verify_released",
            {
                "object": "$object",
                "tool": "$gripper",
                "verification_profile_id": "grasp_default",
            },
            None,
            None,
        )

    asyncio.run(run())
    assert context.attachment_states == {}
    assert [event.event_type for event in executor.event_sink.events] == [
        "primitive_started",
        "grasp_holding_verified",
        "primitive_completed",
        "primitive_started",
        "grasp_release_verified",
        "primitive_completed",
    ]


def test_grasp_verification_fails_closed_for_unapproved_profile() -> None:
    executor, _, gripper = _executor(approved=False)
    gripper.close()

    with pytest.raises(ProfileNotApprovedError, match="not approved"):
        asyncio.run(
            executor.execute_primitive(
                "grasp.verify_holding",
                {
                    "object": "$object",
                    "tool": "$gripper",
                    "verification_profile_id": "grasp_default",
                },
                None,
                None,
            )
        )


def test_preflight_rejects_missing_grasp_verification_profile_before_execution() -> None:
    executor, context, _ = _executor(approved=False)
    report = PreflightValidator(clock_ns=lambda: 1).validate(
        scene={"timestamp_ns": 1, "valid_for_ms": 1000, "calibration_id": "calibration_1"},
        skill=_end_graph(),
        bindings=context.bindings,
        robot=executor.robot,
        execution_mode=ExecutionMode.MOCK,
        skill_validation_status="passed",
        verification_profiles={},
        raise_on_failure=False,
    )

    check = next(
        item
        for item in report.checks
        if item.name == "grasp_verification_profiles_loaded"
    )
    assert not check.passed
    assert "grasp_default" in check.detail
