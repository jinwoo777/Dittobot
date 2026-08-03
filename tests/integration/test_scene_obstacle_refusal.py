from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from robot_skill_system.adapters import MockRobotAdapter
from robot_skill_system.primitives.profiles import load_force_profiles, load_motion_profiles
from robot_skill_system.primitives.registry import PrimitiveRegistry
from robot_skill_system.runtime import RuntimeOrchestrator
from robot_skill_system.scene.models import (
    ConfidenceSummary,
    ObstacleInstance,
    Pose,
    Quaternion,
    SceneSnapshot,
    SurfaceInstance,
    SurfaceRole,
    Vector3,
)
from robot_skill_system.skills.models import SkillGraph

NOW_NS = 9_000_000_000_000
ROOT = Path(__file__).resolve().parents[2]


def _pose(x: float, y: float, z: float) -> Pose:
    return Pose(
        frame_id="base",
        position_m=Vector3(x=x, y=y, z=z),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        timestamp_ns=NOW_NS,
        source="obstacle_integration_fixture",
        confidence=0.99,
    )


def _relative(x: float) -> dict[str, object]:
    return {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": x, "y": 0.0, "z": 0.1},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _path_graph() -> SkillGraph:
    return SkillGraph.model_validate(
        {
            "skill_id": "obstacle_path",
            "version": "1.0.0",
            "name": "obstacle path refusal",
            "description": (
                "An explicit surface-relative path used by the offline collision gate."
            ),
            "skill_type": "motion",
            "bindings": {
                "$surface": {
                    "entity_kind": "surface",
                    "instance_id": "surface_01",
                    "role": "contact_target",
                }
            },
            "nodes": [
                {
                    "node_id": "validate",
                    "operation": "workspace.validate_path",
                    "arguments": {"path": [_relative(0.0), _relative(0.2)]},
                }
            ],
            "start_node": "validate",
            "terminal_nodes": ["validate"],
            "validation_status": "passed",
            "lifecycle_status": "active",
        }
    )


def _consecutive_linear_graph() -> SkillGraph:
    return SkillGraph.model_validate(
        {
            "skill_id": "linear_segment_obstacle",
            "version": "1.0.0",
            "name": "linear segment obstacle refusal",
            "description": "Two safe endpoints whose connecting MoveL crosses an obstacle.",
            "skill_type": "motion",
            "bindings": {
                "$surface": {
                    "entity_kind": "surface",
                    "instance_id": "surface_01",
                    "role": "contact_target",
                }
            },
            "nodes": [
                {
                    "node_id": "left",
                    "operation": "motion.move_l",
                    "arguments": {
                        "target": _relative(-0.1),
                        "motion_profile_id": "linear_slow",
                    },
                },
                {
                    "node_id": "right",
                    "operation": "motion.move_l",
                    "arguments": {
                        "target": _relative(0.3),
                        "motion_profile_id": "linear_slow",
                    },
                },
            ],
            "edges": [{"source_node": "left", "target_node": "right"}],
            "start_node": "left",
            "terminal_nodes": ["right"],
            "motion_profiles": ["linear_slow"],
            "validation_status": "passed",
            "lifecycle_status": "active",
        }
    )


def _scene(*, dynamic: bool, geometry: dict[str, object] | None = None) -> SceneSnapshot:
    obstacle = ObstacleInstance(
        instance_id="new_obstacle",
        pose=_pose(0.6, 0.0, 0.8),
        geometry=geometry or {"type": "sphere", "radius_m": 0.01},
        confidence=0.99,
        velocity_m_s=Vector3(x=0.0, y=0.0, z=0.0) if dynamic else None,
    )
    return SceneSnapshot(
        scene_id=f"obstacle_scene_{'dynamic' if dynamic else 'static'}",
        timestamp_ns=NOW_NS,
        reference_frame="base",
        valid_for_ms=5_000,
        surfaces=[
            SurfaceInstance(
                instance_id="surface_01",
                role=SurfaceRole.CONTACT_TARGET,
                center_m=Vector3(x=0.5, y=0.0, z=0.7),
                normal=Vector3(x=0.0, y=0.0, z=1.0),
                confidence=0.99,
            )
        ],
        static_obstacles=[] if dynamic else [obstacle],
        dynamic_obstacles=[obstacle] if dynamic else [],
        calibration_id="obstacle_calibration",
        confidence_summary=ConfidenceSummary(overall=0.99),
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_scene_obstacle_on_bound_relative_path_refuses_execution(dynamic: bool) -> None:
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=load_motion_profiles(
            ROOT / "configs/motion_profiles/default.json"
        ),
        force_profiles=load_force_profiles(ROOT / "configs/force_profiles/default.json"),
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(
        runtime.run(
            command_text="validate path",
            skill=_path_graph(),
            scene=_scene(dynamic=dynamic),
        )
    )

    assert not result.success
    assert result.error_code == "preflight_failed"
    assert "new_obstacle" in (result.error_message or "")
    assert not any(command.operation.startswith("move_") for command in robot.commands)


def test_unsupported_relevant_obstacle_geometry_fails_closed() -> None:
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=load_motion_profiles(
            ROOT / "configs/motion_profiles/default.json"
        ),
        force_profiles=load_force_profiles(ROOT / "configs/force_profiles/default.json"),
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(
        runtime.run(
            command_text="validate path",
            skill=_path_graph(),
            scene=_scene(dynamic=False, geometry={"type": "mesh", "uri": "mesh.stl"}),
        )
    )

    assert not result.success
    assert "unsupported geometry type" in (result.error_message or "")


def test_consecutive_movel_segment_is_checked_without_explicit_workspace_node() -> None:
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=load_motion_profiles(
            ROOT / "configs/motion_profiles/default.json"
        ),
        force_profiles=load_force_profiles(ROOT / "configs/force_profiles/default.json"),
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(
        runtime.run(
            command_text="cross the obstacle",
            skill=_consecutive_linear_graph(),
            scene=_scene(dynamic=False),
        )
    )

    assert not result.success
    assert result.error_code == "preflight_failed"
    assert "new_obstacle" in (result.error_message or "")
    assert not any(command.operation.startswith("move_") for command in robot.commands)
