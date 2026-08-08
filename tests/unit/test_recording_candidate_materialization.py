from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from robot_skill_system.application import MVPApplication
from robot_skill_system.demonstrations.synthetic import generate_periodic_trajectory
from robot_skill_system.openai_integration.motion_policy import (
    RECORDING_BLOCK_MOTION_OPERATIONS,
)
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.geometry import DeterministicSceneGeometry
from robot_skill_system.scene.transforms import RigidTransform
from robot_skill_system.settings import Settings
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.models import SkillGraph
from robot_skill_system.vertical_slice import capture_mock_scene


@pytest.fixture
def service(tmp_path: Path) -> MVPApplication:
    application = MVPApplication(
        Settings.from_env(
            {
                "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
                "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
                "OPENAI_MODE": "mock",
                "ROBOT_EXECUTION_MODE": "mock",
                "DRY_RUN": "true",
            },
            root=Path(__file__).resolve().parents[2],
        )
    )
    try:
        yield application
    finally:
        application.close()


def _draft(*, contact: bool = False) -> dict[str, Any]:
    return {
        "artifact_uri": "demonstrations/rgbd_test/skill_drafts/draft.json",
        "draft": {
            "suggested_skill_id": "finger_sequence",
            "display_name": "finger sequence",
            "task_description": "wipe a surface" if contact else "move over a surface",
            "observed_task_summary": "chronological two-finger teaching",
            "primitive_sequence": (
                [{"operation": "contact.follow_path"}] if contact else []
            ),
            "confidence": 0.9,
        },
    }


def _calibration() -> dict[str, Any]:
    return {
        "calibration_id": "task_plane_test",
        "surface_anchor_id": "surface_test",
        "source_frame": "camera_color_optical_frame",
        "artifact_uri": "demonstrations/rgbd_test/task_plane.json",
        "camera_to_task_plane": {
            "translation_m": {"x": 0.0, "y": 0.0, "z": 0.75},
            "rotation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "base_chain": {"available": False},
    }


def _sample(
    frame_index: int,
    position_m: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_ns": frame_index * 100_000_000,
        "position_surface_m": position_m,
        "orientation_surface_xyzw": (0.0, 0.0, 0.0, 1.0),
        "confidence": 0.95,
    }


def _transition(frame_index: int, state: str, previous_state: str | None) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_ns": frame_index * 100_000_000,
        "previous_state": previous_state,
        "state": state,
        "distance_m": 0.04 if state == "open" else 0.02,
    }


def _trajectory(
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_uri": "demonstrations/rgbd_test/tcp_trajectory.json",
        "trajectory_id": "trajectory_test",
        "quality": {"sample_count": len(samples), "path_length_m": 0.1},
        "samples": samples,
        "state_transitions": transitions or [],
        "semantic_conflicts": [],
    }


def _materialize(
    service: MVPApplication,
    *,
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]] | None = None,
    contact: bool = False,
) -> SkillGraph:
    graph = service._recording_candidate_graph(
        _draft(contact=contact),
        calibration=_calibration(),
        trajectory=_trajectory(samples, transitions),
        handeye_transform=None,
    )
    SkillCompiler().compile(graph)
    return graph


def test_measured_trajectory_segments_extend_only_mock_observed_space(
    service: MVPApplication,
) -> None:
    samples = [
        _sample(0, (-0.10, 0.00, -0.55)),
        _sample(1, (-0.05, 0.01, -0.50)),
        _sample(2, (0.00, 0.02, -0.45)),
    ]
    graph = _materialize(service, samples=samples)
    assert graph.uncertainty["observed_trajectory_task_plane_m"] == [
        [-0.10, 0.00, -0.55],
        [-0.05, 0.01, -0.50],
        [0.00, 0.02, -0.45],
    ]

    base_scene = capture_mock_scene(frame_count=3)
    scene = service._scene_with_graph_task_plane(base_scene, graph)
    observed_regions = [
        region
        for region in scene.workspace_regions
        if region.region_id.startswith("surface_test_observed_segment_")
    ]
    assert len(observed_regions) == 2
    assert all(region.access_policy.value == "allowed" for region in observed_regions)

    bindings = EntityBinder().bind_entities(scene, graph.binding_requirements())
    validation_node = next(
        node for node in graph.nodes if node.operation == "workspace.validate_path"
    )
    arguments = EntityBinder().bind_arguments(
        validation_node.arguments, scene=scene, bindings=bindings
    )
    assessment = DeterministicSceneGeometry(
        minimum_clearance_m=0.03
    ).validate_path(path=arguments["path"], scene=scene)
    assert assessment.passed is True


def test_open_closed_open_is_inserted_once_in_chronological_motion_order(
    service: MVPApplication,
) -> None:
    samples = [_sample(index, (index * 0.01, 0.0, 0.0)) for index in range(7)]
    # Deliberately unordered, with one repeated stable-open report. The local
    # materializer must sort and de-duplicate without losing the later reopen.
    transitions = [
        _transition(5, "open", "closed"),
        _transition(1, "open", "open"),
        _transition(3, "closed", "open"),
        _transition(0, "open", None),
    ]

    graph = _materialize(service, samples=samples, transitions=transitions)
    behavior_nodes = [
        node for node in graph.nodes if node.operation != "workspace.validate_path"
    ]

    assert [node.operation for node in behavior_nodes] == [
        "gripper.open",
        "motion.move_l",
        "gripper.close",
        "motion.move_l",
        "gripper.open",
        "motion.move_l",
    ]
    move_targets_x = [
        node.arguments["target"]["position_m"]["x"]
        for node in behavior_nodes
        if node.operation == "motion.move_l"
    ]
    assert move_targets_x == pytest.approx([0.03, 0.05, 0.06])
    provenance = graph.uncertainty["motion_simplification"]
    assert [
        (item["start_frame_index"], item["end_frame_index"])
        for item in provenance
    ] == [(0, 3), (3, 5), (5, 6)]
    assert all(item["chosen_primitive_id"] == "motion.move_l" for item in provenance)
    assert all(item["maximum_error_m"] <= 0.004 for item in provenance)
    assert [item["grip_action_end_phase"] for item in provenance] == [
        "grip",
        "action",
        "end_motion",
    ]
    phase_policy = graph.uncertainty["grip_action_end_motion_policy"]
    assert phase_policy["passed"] is True
    assert phase_policy["maximum_motion_blocks"] == {
        "action": 3,
        "end_motion": 3,
    }
    assert phase_policy["allowed_motion_operations"] == sorted(
        RECORDING_BLOCK_MOTION_OPERATIONS
    )
    assert phase_policy["phases"] == {
        "grip": {
            "motion_node_ids": ["move_l_000"],
            "motion_block_count": 1,
        },
        "action": {
            "motion_node_ids": ["move_l_001"],
            "motion_block_count": 1,
        },
        "end_motion": {
            "motion_node_ids": ["move_l_002"],
            "motion_block_count": 1,
        },
    }

    service._persist_graph(
        graph,
        status="candidate",
        validation_status="pending",
        variant="recording_materialization_test",
    )
    validation = service.validate_skill(
        graph.skill_id, {"version": graph.version, "mode": "mock"}
    )
    assert validation["passed"] is True, validation
    assert validation["runtime"]["gripper_commands"] == [
        "connect",
        "open",
        "close",
        "open",
    ]
    assert validation["runtime"]["final_gripper_state"] == {
        "connected": True,
        "width_m": pytest.approx(0.11),
        "is_holding": False,
        "fault_code": None,
    }


def test_action_motion_budget_rejects_more_than_three_boundary_preserving_blocks(
    service: MVPApplication,
) -> None:
    samples = [_sample(index, (index * 0.01, 0.0, 0.0)) for index in range(11)]
    transitions = [
        _transition(0, "open", None),
        *[
            _transition(
                index,
                "closed" if index % 2 else "open",
                "open" if index % 2 else "closed",
            )
            for index in range(1, 11)
        ],
    ]

    with pytest.raises(
        ValueError,
        match=r"action has 9 motion blocks \(maximum 3\)",
    ):
        _materialize(service, samples=samples, transitions=transitions)


def test_stationary_segment_at_gripper_boundary_is_skipped_not_rejected(
    service: MVPApplication,
) -> None:
    samples = [
        _sample(0, (0.0, 0.0, 0.0)),
        _sample(1, (0.0, 0.0, 0.0)),
        _sample(2, (0.0, 0.0, 0.0)),
        _sample(3, (0.01, 0.0, 0.0)),
        _sample(4, (0.02, 0.0, 0.0)),
    ]

    graph = _materialize(
        service,
        samples=samples,
        transitions=[
            _transition(0, "open", None),
            _transition(2, "closed", "open"),
        ],
    )

    behavior_operations = [
        node.operation
        for node in graph.nodes
        if node.operation != "workspace.validate_path"
    ]
    assert behavior_operations == ["gripper.open", "gripper.close", "motion.move_l"]
    [stationary, moving] = graph.uncertainty["motion_simplification"]
    assert stationary["chosen_primitive_id"] == "none"
    assert stationary["skipped_reason"] == "stationary_segment"
    assert stationary["start_frame_index"] == 0
    assert stationary["end_frame_index"] == 2
    assert moving["start_frame_index"] == 2


def test_verified_arc_uses_move_c_and_reports_arc_representation(
    service: MVPApplication,
) -> None:
    radius_m = 0.05
    samples = [
        _sample(
            index,
            (
                radius_m * math.cos(angle),
                radius_m * math.sin(angle),
                0.0,
            ),
        )
        for index, angle in enumerate(
            index * (math.pi / 2.0) / 8.0 for index in range(9)
        )
    ]

    graph = _materialize(service, samples=samples)

    behavior_operations = [
        node.operation
        for node in graph.nodes
        if node.operation != "workspace.validate_path"
    ]
    assert behavior_operations == ["motion.move_c"]
    [provenance] = graph.uncertainty["motion_simplification"]
    assert provenance["chosen_primitive_id"] == "motion.move_c"
    assert provenance["emitted_operation"] == "motion.move_c"
    assert provenance["original_sample_count"] == 9
    assert provenance["simplified_sample_count"] == 3
    assert provenance["maximum_error_m"] == pytest.approx(
        provenance["arc_fit_maximum_error_m"]
    )
    assert provenance["maximum_error_m"] <= 0.004


def test_verified_contact_arc_runs_move_c_inside_force_scope(
    service: MVPApplication,
) -> None:
    radius_m = 0.05
    samples = [
        _sample(
            index,
            (
                radius_m * math.cos(angle),
                radius_m * math.sin(angle),
                0.002,
            ),
        )
        for index, angle in enumerate(
            index * (math.pi / 2.0) / 8.0 for index in range(9)
        )
    ]

    graph = _materialize(service, samples=samples, contact=True)

    behavior_nodes = [
        node for node in graph.nodes if node.operation != "workspace.validate_path"
    ]
    assert [node.operation for node in behavior_nodes] == [
        "contact.search_surface",
        "contact.enable_force",
        "motion.move_c",
        "contact.disable_force",
    ]
    [provenance] = graph.uncertainty["motion_simplification"]
    assert provenance["chosen_primitive_id"] == "motion.move_c"
    assert provenance["emitted_operation"] == "motion.move_c"
    assert provenance["maximum_error_m"] <= 0.004
    assert "circular_normal" in graph.motion_profiles

    service._persist_graph(
        graph,
        status="candidate",
        validation_status="pending",
        variant="contact_arc_materialization_test",
    )
    validation = service.validate_skill(
        graph.skill_id, {"version": graph.version, "mode": "mock"}
    )
    assert validation["passed"] is True, validation
    assert "move_c" in validation["runtime"]["robot_commands"]


def test_non_arc_path_uses_bounded_spline_fallback(
    service: MVPApplication,
) -> None:
    positions = [
        (0.000, 0.000, 0.0),
        (0.010, 0.020, 0.0),
        (0.025, -0.015, 0.0),
        (0.040, 0.025, 0.0),
        (0.055, -0.010, 0.0),
        (0.070, 0.020, 0.0),
        (0.090, 0.000, 0.0),
    ]

    graph = _materialize(
        service,
        samples=[_sample(index, position) for index, position in enumerate(positions)],
    )

    behavior_nodes = [
        node for node in graph.nodes if node.operation != "workspace.validate_path"
    ]
    assert [node.operation for node in behavior_nodes] == ["motion.move_spline"]
    [provenance] = graph.uncertainty["motion_simplification"]
    assert provenance["chosen_primitive_id"] == "motion.move_spline"
    assert provenance["emitted_operation"] == "motion.move_spline"
    assert provenance["maximum_error_m"] <= 0.004
    assert provenance["simplified_sample_count"] == len(
        behavior_nodes[0].arguments["waypoints"]
    )
    assert behavior_nodes[0].arguments["waypoints"][0]["position_m"]["x"] == 0.0
    assert behavior_nodes[0].arguments["waypoints"][-1]["position_m"]["x"] == 0.09


def test_locally_verified_periodic_contact_path_is_preserved(
    service: MVPApplication,
) -> None:
    demonstration = generate_periodic_trajectory(
        cycles=3.0,
        drift_m=0.0,
        noise_std_m=0.0,
    )
    samples = [
        _sample(index, sample.position_m)
        for index, sample in enumerate(demonstration.samples)
    ]

    graph = _materialize(service, samples=samples, contact=True)

    behavior_nodes = [
        node for node in graph.nodes if node.operation != "workspace.validate_path"
    ]
    assert [node.operation for node in behavior_nodes] == [
        "contact.search_surface",
        "contact.enable_force",
        "motion.move_periodic",
        "contact.disable_force",
    ]
    periodic = behavior_nodes[2]
    assert periodic.arguments["center"]["position_m"] == pytest.approx(
        {"x": 0.2, "y": -0.1, "z": 0.002}, abs=1.0e-6
    )
    assert periodic.arguments["amplitude_m"] == pytest.approx(
        {"x": 0.045, "y": 0.0, "z": 0.0}, abs=1.0e-6
    )
    assert periodic.arguments["repetitions"] == 3
    assert periodic.arguments["motion_profile_id"] == "periodic_safe"
    assert "periodic_safe" in graph.motion_profiles
    [provenance] = graph.uncertainty["motion_simplification"]
    assert provenance["chosen_primitive_id"] == "motion.move_periodic"
    assert provenance["emitted_operation"] == "motion.move_periodic"
    assert provenance["maximum_error_m"] <= 0.004
    assert provenance["periodic_repetitions"] == 3
    assert provenance["periodic_timing_policy"] == (
        "execution speed remains profile-owned"
    )
    service._persist_graph(
        graph,
        status="candidate",
        validation_status="pending",
        variant="periodic_materialization_test",
    )
    validation = service.validate_skill(
        graph.skill_id, {"version": graph.version, "mode": "mock"}
    )
    assert validation["passed"] is True, validation
    assert "move_periodic" in validation["runtime"]["robot_commands"]


def test_periodic_fit_with_unrepresentable_drift_falls_back_to_spline(
    service: MVPApplication,
) -> None:
    demonstration = generate_periodic_trajectory(
        cycles=3.0,
        drift_m=0.025,
        noise_std_m=0.0,
    )
    samples = [
        _sample(index, sample.position_m)
        for index, sample in enumerate(demonstration.samples)
    ]

    graph = _materialize(service, samples=samples)

    behavior_nodes = [
        node for node in graph.nodes if node.operation != "workspace.validate_path"
    ]
    assert behavior_nodes
    assert all(node.operation == "motion.move_spline" for node in behavior_nodes)
    [provenance] = graph.uncertainty["motion_simplification"]
    assert provenance["chosen_primitive_id"] == "motion.move_spline"
    assert provenance["maximum_error_m"] <= 0.004
    assert "fixed-centre periodic" in provenance["periodic_fallback_reason"]


def test_contact_segments_and_validation_chunks_preserve_boundaries(
    service: MVPApplication,
) -> None:
    samples = [_sample(index, (index * 0.001, 0.0, 0.0)) for index in range(300)]

    graph = _materialize(
        service,
        samples=samples,
        transitions=[
            _transition(0, "open", None),
            _transition(150, "closed", "open"),
        ],
        contact=True,
    )

    validation_nodes = [
        node for node in graph.nodes if node.operation == "workspace.validate_path"
    ]
    assert len(validation_nodes) == 2
    assert validation_nodes[0].arguments["path"][-1] == validation_nodes[1].arguments[
        "path"
    ][0]

    follow_nodes = [node for node in graph.nodes if node.operation == "contact.follow_path"]
    assert len(follow_nodes) == 2
    assert follow_nodes[0].arguments["path"][-1]["position_m"]["x"] == pytest.approx(
        0.15
    )
    assert follow_nodes[1].arguments["path"][0]["position_m"]["x"] == pytest.approx(
        0.15
    )
    first_follow_index = graph.nodes.index(follow_nodes[0])
    second_follow_index = graph.nodes.index(follow_nodes[1])
    close_index = next(
        index for index, node in enumerate(graph.nodes) if node.operation == "gripper.close"
    )
    assert first_follow_index < close_index < second_follow_index
    for follow_node in follow_nodes:
        follow_index = graph.nodes.index(follow_node)
        assert graph.nodes[follow_index - 1].operation == "contact.enable_force"
        assert graph.nodes[follow_index + 1].operation == "contact.disable_force"


@pytest.mark.parametrize("missing_frame, timestamp_offset_ns", [(True, 0), (False, 1)])
def test_transition_requires_an_exact_metric_pose_timestamp(
    service: MVPApplication,
    missing_frame: bool,
    timestamp_offset_ns: int,
) -> None:
    samples = [_sample(index, (index * 0.01, 0.0, 0.0)) for index in range(4)]
    frame_index = 9 if missing_frame else 2
    transition = _transition(frame_index, "closed", None)
    transition["timestamp_ns"] += timestamp_offset_ns

    with pytest.raises(ValueError, match="metric pose|timestamp"):
        _materialize(service, samples=samples, transitions=[transition])


def test_first_frame_semantic_roi_anchor_becomes_advisory_task_plane_input(
    service: MVPApplication,
) -> None:
    draft_id = "draft_" + "a" * 32
    artifact = service.store.put_json(
        "demonstrations/rgbd_test/skill_drafts/anchor_evidence.json",
        {
            "draft_id": draft_id,
            "recording_id": "rgbd_test",
            "frame_index": 0,
            "frame_id": "camera_color_optical_frame",
            "anchors": [
                {
                    "anchor_id": "initial_target_object",
                    "entity_role": "target_object",
                    "semantic_class": "cup",
                    "semantic_confidence": 0.91,
                    "position_camera_m": [0.10, 0.20, 0.75],
                    "valid_depth_fraction": 0.88,
                }
            ],
        },
    )
    draft = {
        **_draft(),
        "draft_id": draft_id,
        "source_recording_id": "rgbd_test",
        "keyframe_indices": [0],
        "initial_scene_anchors": {
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        },
    }
    calibration = {
        **_calibration(),
        "camera_to_task_plane": {
            "translation_m": {"x": 0.0, "y": 0.0, "z": 0.75},
            "rotation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }

    graph = service._recording_candidate_graph(
        draft,
        calibration=calibration,
        trajectory=_trajectory(
            [_sample(0, (0.0, 0.0, 0.0)), _sample(1, (0.01, 0.0, 0.0))]
        ),
        handeye_transform=None,
    )

    [anchor_input] = graph.uncertainty["initial_scene_anchor_inputs"]
    assert anchor_input["position_task_plane_m"] == pytest.approx(
        {"x": 0.10, "y": 0.20, "z": 0.0}
    )
    assert anchor_input["operator_confirmed"] is False
    assert anchor_input["usable_for_runtime_binding"] is False


def test_task_plane_base_chain_rejects_active_tcp_mismatch(
    service: MVPApplication,
) -> None:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    service.store.put_json(
        "demonstrations/rgbd_test/camera_stationarity.json",
        {
            "diagnostics": {"passed": True},
            "start": {
                "active_tcp_name": "teaching_tcp",
                "base_to_flange": identity,
            },
        },
    )
    service.store.put_json(
        "calibrations/legacy_import_test/result.json",
        {
            "import_id": "legacy_import_test",
            "passed": True,
            "active_tcp_name": "different_tcp",
            "flange_to_camera": identity,
        },
    )

    result = service._task_plane_base_chain("rgbd_test", RigidTransform.identity())

    assert result["available"] is False
    assert result["verified"] is False
    assert "active TCP" in result["reason"]
