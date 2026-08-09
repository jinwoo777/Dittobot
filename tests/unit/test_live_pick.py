from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from aruco.object_width_workspace import build_runtime_workspace, load_reference
from robot_skill_system.application import MVPApplication
from robot_skill_system.grasping.live_pick import build_live_pick_graph
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import SceneStaleError
from robot_skill_system.runtime.hardware_verification import (
    HARDWARE_FIXED_REFERENCE_SCENE_VALIDITY_MS,
    FixedReferenceSceneMonitor,
)
from robot_skill_system.scene.models import ObjectInstance, Pose, Quaternion, Vector3
from robot_skill_system.settings import Settings
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.skills.models import SkillGraph
from robot_skill_system.vertical_slice import capture_mock_scene


def _source_graph() -> SkillGraph:
    return SkillGraph.model_validate(
        {
            "skill_id": "take_hammer",
            "version": "0.6.0",
            "name": "take hammer",
            "description": "replayed source that must not run in hardware",
            "skill_type": "manipulation",
            "source_demonstrations": ["demo_1"],
            "nodes": [
                {
                    "node_id": "source_open",
                    "operation": "gripper.open",
                    "arguments": {"tool": "$tool"},
                }
            ],
            "start_node": "source_open",
            "terminal_nodes": ["source_open"],
        }
    )


def _anchor() -> ObjectInstance:
    return ObjectInstance(
        instance_id="live_hammer_grasp",
        class_name="hammer",
        pose=Pose(
            frame_id="base",
            position_m=Vector3(x=0.4, y=0.0, z=0.2),
            orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            timestamp_ns=1,
            source="live_yolo_rgbd_grasp_anchor",
            confidence=0.9,
        ),
        attributes={
            "anchor_semantics": "predicted_grasp_xy_with_rgbd_mean_grip_depth",
            "grasp_point_plane_m": [0.0, 0.0, 0.0],
            "observed_grasp_point_plane_m": [0.0, 0.0, 0.196],
            "grip_region_mean_depth_camera_m": 0.41,
            "grip_region_depth_std_m": 0.002,
            "grip_region_depth_sample_count": 120,
            "grip_region_depth_valid_fraction": 0.95,
            "jaw_yaw_plane_rad": 0.25,
            "target_gripper_width_m": 0.025,
            "object_width_mm": 25.0,
            "observed_cross_section_width_mm": 85.0,
        },
        confidence=0.9,
        visible_fraction=0.9,
        pose_source="live_yolo_rgbd_grasp_anchor",
    )


def test_live_pick_graph_uses_rgbd_mean_depth_object_relative_targets() -> None:
    reference = load_reference(Path("aruco/fixed_workspace_reference.npz"))
    runtime = build_runtime_workspace(reference, 80.0)
    graph = build_live_pick_graph(
        _source_graph(), object_anchor=_anchor(), reference=reference, runtime=runtime
    )

    SkillGraphValidator().validate(graph)
    SkillCompiler().compile(graph)
    assert set(graph.bindings) == {"$object", "$tool"}
    assert [node.operation for node in graph.nodes] == [
        "gripper.open",
        "motion.rotate_joint_6_relative",
        "workspace.validate_target",
        "motion.move_l",
        "workspace.validate_target",
        "motion.move_l",
        "gripper.move_width",
    ]
    assert graph.nodes[1].arguments["delta_rad"] == pytest.approx(0.25)
    grasp = graph.nodes[5].arguments["target"]
    pregrasp = graph.nodes[3].arguments["target"]
    assert grasp["anchor_id"] == "$object"
    assert grasp["position_m"]["z"] == pytest.approx(0.191)
    assert pregrasp["position_m"]["z"] == pytest.approx(
        0.191 + 0.050
    )
    assert graph.nodes[6].arguments["width_m"] == pytest.approx(0.025)
    assert graph.uncertainty["live_pick"]["safety_margin_m"] == pytest.approx(0.005)
    assert graph.uncertainty["live_pick"]["z_descent_m"] == pytest.approx(0.050)
    assert graph.uncertainty["live_pick"]["z_target_source"] == (
        "rgbd_grip_region_mean_depth_plus_local_offset"
    )
    assert graph.uncertainty["live_pick"]["observed_grasp_tcp_z_plane_m"] == (
        pytest.approx(0.196)
    )
    assert graph.uncertainty["live_pick"]["grasp_depth_offset_m"] == pytest.approx(
        -0.005
    )
    assert graph.uncertainty["live_pick"]["workspace_xy_tolerance_m"] == pytest.approx(
        0.001
    )
    assert runtime.allowed_down_from_reference_m == pytest.approx(
        0.184 - 0.110 * (1.0 - math.cos(math.asin((80.0 / 2.0) / 110.0))) - 0.005
    )


def test_runtime_materializes_and_executes_the_live_graph(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )
    service = MVPApplication(settings)
    try:
        source = _source_graph()
        row = service._persist_graph(
            source,
            status="active",
            validation_status="passed",
            variant="test_live_pick",
        )
        reference = load_reference(Path("aruco/fixed_workspace_reference.npz"))
        runtime = build_runtime_workspace(reference, 80.0)
        live_graph = build_live_pick_graph(
            source,
            object_anchor=_anchor(),
            reference=reference,
            runtime=runtime,
        )
        run, artifact = service._load_execution_run(row, live_graph)

        class RecordingRuntime:
            def __init__(self) -> None:
                self.operations: list[str] = []

            async def execute_primitive(
                self, operation: str, arguments: object, timeout_s: float, checkpoint: str | None
            ) -> bool:
                del arguments, timeout_s, checkpoint
                self.operations.append(operation)
                return True

        recorder = RecordingRuntime()
        assert asyncio.run(run(recorder)) is True
        stored_graph = json.loads(service.store.read_bytes(artifact["skill_graph_uri"]))

        assert artifact["kind"] == "runtime_materialized"
        assert set(stored_graph["bindings"]) == {"$object", "$tool"}
        assert recorder.operations == [node.operation for node in live_graph.nodes]
        assert recorder.operations != [node.operation for node in source.nodes]
    finally:
        service.close()


def test_live_hammer_z_workspace_uses_grip_profile_width() -> None:
    reference = load_reference(Path("aruco/fixed_workspace_reference.npz"))
    manually_entered_workspace = build_runtime_workspace(reference, 80.0)

    runtime = MVPApplication._runtime_workspace_for_live_hammer(
        reference,
        manually_entered_workspace,
        _anchor(),
    )

    assert runtime.object_width_mm == pytest.approx(25.0)
    assert runtime.z_min_plane_m < manually_entered_workspace.z_min_plane_m


def test_live_pick_rejects_legacy_point_depth_anchor() -> None:
    reference = load_reference(Path("aruco/fixed_workspace_reference.npz"))
    runtime = build_runtime_workspace(reference, 25.0)
    attributes = dict(_anchor().attributes)
    attributes.pop("grip_region_mean_depth_camera_m")
    legacy_anchor = _anchor().model_copy(update={"attributes": attributes}, deep=True)

    with pytest.raises(ValueError, match="grip_region_mean_depth_camera_m"):
        build_live_pick_graph(
            _source_graph(),
            object_anchor=legacy_anchor,
            reference=reference,
            runtime=runtime,
        )


def test_live_pick_rejects_depth_offset_outside_local_envelope() -> None:
    reference = load_reference(Path("aruco/fixed_workspace_reference.npz"))
    runtime = build_runtime_workspace(reference, 25.0)

    with pytest.raises(ValueError, match="grasp depth offset"):
        build_live_pick_graph(
            _source_graph(),
            object_anchor=_anchor(),
            reference=reference,
            runtime=runtime,
            grasp_depth_offset_m=-0.011,
        )


def test_live_pick_accepts_only_the_configured_near_boundary_xy_tolerance() -> None:
    reference = load_reference(Path("aruco/fixed_workspace_reference.npz"))
    runtime = build_runtime_workspace(reference, 25.0)
    attributes = dict(_anchor().attributes)
    attributes.update(
        {
            "grasp_point_plane_m": [0.14541417191065614, 0.02771710291965085, 0.0],
            "observed_grasp_point_plane_m": [
                0.14541417191065614,
                0.02771710291965085,
                0.20200635899311703,
            ],
        }
    )
    near_boundary_anchor = _anchor().model_copy(
        update={"attributes": attributes}, deep=True
    )

    with pytest.raises(ValueError, match="outside the frozen safe polygon"):
        build_live_pick_graph(
            _source_graph(),
            object_anchor=near_boundary_anchor,
            reference=reference,
            runtime=runtime,
            workspace_xy_tolerance_m=0.0,
        )

    graph = build_live_pick_graph(
        _source_graph(),
        object_anchor=near_boundary_anchor,
        reference=reference,
        runtime=runtime,
        workspace_xy_tolerance_m=0.001,
    )
    assert graph.uncertainty["live_pick"]["workspace_xy_tolerance_m"] == pytest.approx(
        0.001
    )


def test_fixed_reference_scene_supports_execution_without_bypassing_fresh_binding() -> None:
    now_ns = 30_000_000_000
    scene = capture_mock_scene(now_ns=now_ns - 11_000_000_000).model_copy(
        update={"valid_for_ms": HARDWARE_FIXED_REFERENCE_SCENE_VALIDITY_MS}, deep=True
    )
    binder = EntityBinder(clock_ns=lambda: now_ns)

    with pytest.raises(SceneStaleError):
        binder.ensure_scene_fresh(scene, maximum_age_ms=5_000)

    FixedReferenceSceneMonitor(
        maximum_scene_age_ms=HARDWARE_FIXED_REFERENCE_SCENE_VALIDITY_MS,
        clock_ns=lambda: now_ns,
    ).assert_fresh(scene)
