"""End-to-end hierarchical task-flow resolution and Mock execution tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_skill_system.application import MVPApplication
from robot_skill_system.runtime.errors import RuntimeSafetyError, SkillHashMismatchError
from robot_skill_system.runtime.task_flow_materializer import (
    ExecutableGripProfileArtifact,
)
from robot_skill_system.settings import Settings
from robot_skill_system.skills.models import SkillEdge, SkillGraph
from robot_skill_system.skills.task_flow import GripProfile
from robot_skill_system.storage.orm import SkillVersionRecord


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )


def _relative_object_pose(x_m: float) -> dict[str, Any]:
    return {
        "anchor_id": "$object",
        "anchor_type": "object",
        "position_m": {"x": x_m, "y": 0.0, "z": 0.0},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _grip_profile() -> GripProfile:
    return GripProfile(
        profile_id="cloth_default",
        version="1.0.0",
        object_class_id="cloth",
        object_frame_policy="object_relative_6d",
        object_frame_revision="mock-object-frame-v1",
        pregrasp_pose=_relative_object_pose(-0.06),
        grasp_pose=_relative_object_pose(-0.01),
        jaw_axis={"x": 0.0, "y": 1.0, "z": 0.0},
        approach_axis={"x": 1.0, "y": 0.0, "z": 0.0},
        required_tool_class="wiper",
        motion_profile_id="linear_slow",
        gripper_calibration_profile_id="rg2_default",
        verification_profile_id="grasp_default",
        lifecycle_status="active",
        validation_status="passed",
        hardware_compatible=False,
    )


def _action_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="bring_action",
        version="1.0.0",
        name="bring action",
        description="Perform the locally learned action while holding the object.",
        skill_type="motion",
        bindings={
            "$object": {"entity_kind": "object", "class_name": "cloth"},
            "$surface": {
                "entity_kind": "surface",
                "role": "contact_target",
            },
        },
        required_entity_roles={"$surface": "contact_target"},
        nodes=[
            {
                "node_id": "perform",
                "operation": "motion.wait",
                "arguments": {"duration_s": 0.001},
            }
        ],
        start_node="perform",
        terminal_nodes=["perform"],
    )


def _end_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="release_safe_end",
        version="1.0.0",
        name="safe release end motion",
        description="Open and verify release after the action succeeds.",
        skill_type="manipulation",
        bindings={
            "$object": {"entity_kind": "object", "class_name": "cloth"},
            "$gripper": {
                "entity_kind": "tool",
                "class_name": "wiper",
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
                "node_id": "verify_released",
                "operation": "grasp.verify_released",
                "arguments": {
                    "object": "$object",
                    "tool": "$gripper",
                    "verification_profile_id": "grasp_default",
                },
            },
        ],
        edges=[SkillEdge(source_node="open", target_node="verify_released")],
        start_node="open",
        terminal_nodes=["verify_released"],
    )


def _seed_active_task_flow(
    service: MVPApplication,
    *,
    action_graph: SkillGraph | None = None,
) -> str:
    repository = service.repository
    object_entry = repository.create_catalog_entry(
        kind="object",
        canonical_id="cloth",
        display_name="천",
        aliases=("천", "cloth"),
    )
    action_entry = repository.create_catalog_entry(
        kind="action",
        canonical_id="bring",
        display_name="가져오기",
        aliases=("가져와", "bring"),
    )
    end_entry = repository.create_catalog_entry(
        kind="end_motion",
        canonical_id="release_safe",
        display_name="안전 해제",
    )

    grip_artifact = service.store.put_json(
        "grips/cloth/1.0.0.json",
        ExecutableGripProfileArtifact(
            grip_profile=_grip_profile(),
            provenance={"source": "mock-rgbd-integration-demonstration"},
        ).model_dump(mode="json"),
    )
    grip_version = repository.register_grip_profile_version(
        object_class_id="cloth",
        semantic_version="1.0.0",
        artifact_uri=grip_artifact.uri,
        artifact_checksum_sha256=grip_artifact.checksum_sha256,
        object_frame_policy="object_relative_6d",
        object_frame_revision="mock-object-frame-v1",
        status="validated",
        validation_status="passed",
        hardware_compatible=False,
        gripper_calibration_profile_id="rg2_default",
    )
    repository.activate_grip_profile_version(grip_version.id)
    repository.activate_catalog_entry(object_entry.id)

    action_graph = action_graph or _action_graph()
    action_skill = repository.register_skill_version(
        name="action:bring",
        intent="bring",
        semantic_version=action_graph.version,
        graph=action_graph,
        status="validated",
        validation_status="passed",
    )
    action_stage = repository.register_stage_definition(
        kind="action",
        canonical_id="bring",
        skill_version_id=action_skill.id,
        semantic_version=action_graph.version,
        status="validated",
        validation_status="passed",
        required_roles=("$surface",),
        input_contract={"attachment": "holding", "force_mode": "disabled"},
        output_contract={"attachment": "holding", "force_mode": "disabled"},
        anchor_policy={"allowed_anchor_types": ["surface", "object"]},
    )
    repository.activate_stage_definition(action_stage.id)

    end_graph = _end_graph()
    end_skill = repository.register_skill_version(
        name="end_motion:release_safe",
        intent="release_safe",
        semantic_version=end_graph.version,
        graph=end_graph,
        status="validated",
        validation_status="passed",
    )
    end_stage = repository.register_stage_definition(
        kind="end_motion",
        canonical_id="release_safe",
        skill_version_id=end_skill.id,
        semantic_version=end_graph.version,
        status="validated",
        validation_status="passed",
        input_contract={"attachment": "holding", "force_mode": "disabled"},
        output_contract={"attachment": "released", "force_mode": "disabled"},
        anchor_policy={"allowed_anchor_types": ["object", "workspace_region"]},
    )
    repository.activate_stage_definition(end_stage.id)
    repository.activate_catalog_entry(end_entry.id)

    mapping = repository.map_action_to_end_motion(
        action_id="bring", end_motion_id="release_safe"
    )
    repository.activate_catalog_entry(action_entry.id)
    return mapping.id


def _failing_action_graph() -> SkillGraph:
    """A valid Action that deterministically fails after Grip has closed."""

    return SkillGraph(
        skill_id="bring_action",
        version="1.0.0",
        name="failing bring action",
        description="Verify an impossible open state to exercise failure routing.",
        skill_type="manipulation",
        bindings={
            "$object": {"entity_kind": "object", "class_name": "cloth"},
            "$gripper": {
                "entity_kind": "tool",
                "class_name": "wiper",
                "must_be_attached": True,
            },
            "$surface": {
                "entity_kind": "surface",
                "role": "contact_target",
            },
        },
        required_entity_roles={"$surface": "contact_target"},
        nodes=[
            {
                "node_id": "fail_after_grip",
                "operation": "gripper.verify_state",
                "arguments": {
                    "tool": "$gripper",
                    "expected_state": "open",
                },
            }
        ],
        start_node="fail_after_grip",
        terminal_nodes=["fail_after_grip"],
    )


def test_active_catalog_resolves_materializes_and_executes_mock_flow(
    tmp_path: Path,
) -> None:
    service = MVPApplication(_settings(tmp_path))
    try:
        mapping_id = _seed_active_task_flow(service)
        scene = service.capture_scene({"mode": "mock"})

        resolved = service.resolve_runtime(
            {"text": "천을 가져와", "scene_id": scene["scene_id"]}
        )

        assert resolved["resolver_mode"] == "grip_action_end_hierarchical"
        assert resolved["intent"]["object_class_id"] == "cloth"
        assert resolved["intent"]["action_id"] == "bring"
        assert resolved["intent"]["role_bindings"] == {
            "$surface": "table_surface_01"
        }
        assert resolved["skill_candidates"]["blockers"] == []
        candidate = resolved["skill_candidates"]["results"][0]
        assert candidate["executable"] is True
        manifest = resolved["task_flow_manifest"]
        assert manifest is not None
        assert manifest["mapping_id"] == mapping_id
        assert manifest["mapping_revision"] == 1
        assert manifest["grip"]["component_id"] == "cloth_default"
        assert manifest["action"]["component_id"] == "bring"
        assert manifest["end_motion"]["component_id"] == "release_safe"
        assert manifest["preflight"]["passed"] is True

        row = service._find_version(candidate["skill_id"], candidate["version"])
        graph = SkillGraph.model_validate(row.graph_json)
        assert [node.node_id.split("__", 1)[0] for node in graph.nodes] == [
            "grip",
            "grip",
            "grip",
            "grip",
            "grip",
            "grip",
            "grip",
            "action",
            "end",
            "end",
        ]

        execution = service.execute_runtime(
            {
                "run_id": "task_flow_mock_e2e",
                "skill_id": candidate["skill_id"],
                "version": candidate["version"],
                "scene_id": scene["scene_id"],
                "bindings": candidate["bindings"],
                "mode": "mock",
                "text": "천을 가져와",
            }
        )

        assert execution["status"] == "succeeded"
        assert execution["preflight"]["passed"] is True
        assert execution["bindings"] == {
            "$gripper": "wiper_01",
            "$object": "cloth_01",
            "$surface": "table_surface_01",
        }
        run = service.get_runtime_run(execution["run_id"])
        primitive_order = [
            event["details"]["operation"]
            for event in run["events"]
            if event["event_type"] == "primitive_started"
        ]
        assert primitive_order == [
            "gripper.open",
            "workspace.validate_target",
            "motion.move_l",
            "workspace.validate_target",
            "motion.move_l",
            "gripper.close",
            "grasp.verify_holding",
            "motion.wait",
            "gripper.open",
            "grasp.verify_released",
        ]
        assert "grasp_holding_verified" in execution["events"]
        assert "grasp_release_verified" in execution["events"]
    finally:
        service.close()


def test_action_failure_never_enters_the_normal_end_motion(tmp_path: Path) -> None:
    service = MVPApplication(_settings(tmp_path))
    try:
        _seed_active_task_flow(service, action_graph=_failing_action_graph())
        scene = service.capture_scene({"mode": "mock"})
        resolved = service.resolve_runtime(
            {"text": "천을 가져와", "scene_id": scene["scene_id"]}
        )
        candidate = resolved["skill_candidates"]["results"][0]

        with pytest.raises(RuntimeSafetyError, match="gripper state mismatch"):
            service.execute_runtime(
                {
                    "run_id": "task_flow_action_failure",
                    "skill_id": candidate["skill_id"],
                    "version": candidate["version"],
                    "scene_id": scene["scene_id"],
                    "bindings": candidate["bindings"],
                    "mode": "mock",
                }
            )

        run = service.get_runtime_run("task_flow_action_failure")
        operations = [
            event["details"]["operation"]
            for event in run["events"]
            if event["event_type"] == "primitive_started"
        ]
        assert operations[-1] == "gripper.verify_state"
        # The first open belongs to Grip.  A second open would mean the normal
        # End motion ran despite Action failure.
        assert operations.count("gripper.open") == 1
        assert run["status"] == "failed"
        assert any(
            event["event_type"] == "compiled_skill_execution_cleanup"
            for event in run["events"]
        )
    finally:
        service.close()


def _candidate_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="mock_override_candidate",
        version="1.0.0",
        name="Mock override candidate",
        description="Structurally valid candidate used for per-run override tests.",
        skill_type="motion",
        nodes=[
            {
                "node_id": "wait",
                "operation": "motion.wait",
                "arguments": {"duration_s": 0.001},
            }
        ],
        start_node="wait",
        terminal_nodes=["wait"],
        lifecycle_status="candidate",
        validation_status="pending",
    )


def _override() -> dict[str, Any]:
    return {
        "override_all_overridable": True,
        "operator_id": "operator_mock_01",
        "reason": "offline candidate behavior inspection",
        "acknowledge_mock_only": True,
    }


def test_mock_override_is_run_scoped_audited_and_cannot_bypass_hard_gates(
    tmp_path: Path,
) -> None:
    service = MVPApplication(_settings(tmp_path))
    try:
        scene = service.capture_scene({"mode": "mock"})
        graph = _candidate_graph()
        row = service._persist_graph(
            graph,
            status="candidate",
            validation_status="pending",
            variant="mock_override_test",
            index_for_legacy_retrieval=False,
        )
        request = {
            "run_id": "mock_override_run_01",
            "skill_id": graph.skill_id,
            "version": graph.version,
            "scene_id": scene["scene_id"],
            "mode": "mock",
            "mock_override": _override(),
        }

        execution = service.execute_runtime(request)

        assert execution["status"] == "succeeded"
        assert execution["events"][0] == "mock_validation_override"
        persisted = service.get_runtime_run(execution["run_id"])
        audit = persisted["events"][0]
        assert audit["event_type"] == "mock_validation_override"
        assert audit["severity"] == "warning"
        assert audit["details"]["operator_id"] == "operator_mock_01"
        assert audit["details"]["bypassed_check_ids"] == [
            "component_lifecycle_active",
            "skill_validation",
        ]
        assert audit["details"]["component_activation_changed"] is False
        assert audit["details"]["hardware_compatibility_changed"] is False
        unchanged = service.repository.get_skill_version(row.id)
        assert unchanged is not None
        assert unchanged.status == "candidate"
        assert unchanged.validation_status == "pending"

        with pytest.raises(ValueError, match="active, validated"):
            service.execute_runtime(
                {
                    "skill_id": graph.skill_id,
                    "version": graph.version,
                    "scene_id": scene["scene_id"],
                    "mode": "mock",
                }
            )

        for forbidden_mode in ("dry_run", "simulation", "hardware"):
            with pytest.raises(ValueError, match="only in mode=mock"):
                service.execute_runtime(
                    {
                        "skill_id": graph.skill_id,
                        "version": graph.version,
                        "scene_id": scene["scene_id"],
                        "mode": forbidden_mode,
                        "mock_override": _override(),
                    }
                )

        with service.database.session() as session:
            stored = session.get(SkillVersionRecord, row.id)
            assert stored is not None
            stored.graph_json = {
                **stored.graph_json,
                "description": "tampered after compilation",
            }
        with pytest.raises(SkillHashMismatchError, match="checksum mismatch"):
            service.execute_runtime(
                {
                    "skill_id": graph.skill_id,
                    "version": graph.version,
                    "scene_id": scene["scene_id"],
                    "mode": "mock",
                    "mock_override": _override(),
                }
            )
    finally:
        service.close()
