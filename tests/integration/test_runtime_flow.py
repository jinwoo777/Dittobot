from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import select

from robot_skill_system.adapters import MockRobotAdapter
from robot_skill_system.primitives.profiles import load_force_profiles, load_motion_profiles
from robot_skill_system.primitives.registry import PrimitiveRegistry
from robot_skill_system.runtime import RuntimeOrchestrator
from robot_skill_system.scene.models import (
    ConfidenceSummary,
    Pose,
    Quaternion,
    SceneSnapshot,
    SurfaceInstance,
    Vector3,
)
from robot_skill_system.skills.models import SkillGraph
from robot_skill_system.storage import Database, ExecutionRunRecord, StorageRepository

NOW_NS = 5_000_000_000_000
ROOT = Path(__file__).resolve().parents[2]


def test_registered_skill_binds_executes_and_persists_events(tmp_path: Path) -> None:
    pose = Pose(
        frame_id="base",
        position_m=Vector3(x=0.4, y=0.2, z=0.6),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        timestamp_ns=NOW_NS,
        source="integration_fixture",
        confidence=0.99,
    )
    scene = SceneSnapshot(
        schema_version="1.0",
        scene_id="scene_integration",
        timestamp_ns=NOW_NS,
        reference_frame="base",
        valid_for_ms=5_000,
        surfaces=[
            SurfaceInstance(
                instance_id="surface_integration",
                role="contact_target",
                center_m=pose.position_m,
                normal=Vector3(x=0.0, y=0.0, z=1.0),
                confidence=0.99,
                allowed_contact_links=[],
            )
        ],
        calibration_id="calibration_integration",
        confidence_summary=ConfidenceSummary(overall=0.99),
    )
    graph = SkillGraph(
        skill_id="move_integration",
        version="1.0.0",
        name="integration move",
        description="Persist and execute one relative linear target.",
        skill_type="motion",
        bindings={
            "$surface": {
                "entity_kind": "surface",
                "instance_id": "surface_integration",
            }
        },
        nodes=[
            {
                "node_id": "move",
                "operation": "motion.move_l",
                "arguments": {
                    "target": {
                        "anchor_id": "$surface",
                        "anchor_type": "surface",
                        "position_m": {"x": 0.1, "y": 0.0, "z": 0.1},
                        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
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
    database = Database.from_path(tmp_path / "runtime.sqlite3")
    database.create_schema()
    repository = StorageRepository(database)
    scene_record = repository.record_scene(scene)
    version = repository.register_skill_version(
        name=graph.name,
        intent="move_surface",
        semantic_version=graph.version,
        graph=graph,
        status="active",
        validation_status="passed",
    )
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    runtime = RuntimeOrchestrator(
        robot=robot,
        motion_profiles=load_motion_profiles(ROOT / "configs/motion_profiles/default.json"),
        force_profiles=load_force_profiles(ROOT / "configs/force_profiles/default.json"),
        execution_mode="mock",
        primitive_registry=PrimitiveRegistry.default(),
        storage_repository=repository,
        clock_ns=lambda: NOW_NS,
    )

    result = asyncio.run(
        runtime.run(
            command_text="move the tool over the surface",
            skill=graph,
            scene=scene,
            expected_skill_checksum_sha256=version.graph_checksum_sha256,
            skill_version_id=version.id,
            persisted_scene_id=scene_record.id,
        )
    )

    assert result.success
    assert result.execution_run_id is not None
    assert any(command.operation == "move_l" for command in robot.commands)
    with database.session() as session:
        run = session.scalar(
            select(ExecutionRunRecord).where(ExecutionRunRecord.id == result.execution_run_id)
        )
        assert run is not None
        assert run.status == "succeeded"
        assert [event.event_type for event in run.events] == [
            "execution_started",
            "skill_execution_started",
            "primitive_started",
            "primitive_completed",
            "skill_execution_completed",
            "execution_succeeded",
        ]
    database.close()
