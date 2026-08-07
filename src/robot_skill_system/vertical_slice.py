"""Executable offline teaching-to-runtime vertical slice."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from robot_skill_system.adapters.mock_robot import MockGripperAdapter, MockRobotAdapter
from robot_skill_system.capture.interfaces import CaptureMode, CaptureRequest
from robot_skill_system.capture.mock_capture import MockCapture, MockCaptureConfig
from robot_skill_system.demonstrations.preprocessing import preprocess_trajectory
from robot_skill_system.demonstrations.primitive_fitter import (
    fit_arc,
    fit_line,
    fit_periodic,
)
from robot_skill_system.demonstrations.segmentation import segment_trajectory
from robot_skill_system.demonstrations.synthetic import (
    generate_arc_trajectory,
    generate_line_trajectory,
    generate_novice_wipe_trajectory,
    generate_periodic_trajectory,
)
from robot_skill_system.openai_integration.demonstration_analyzer import DemonstrationAnalyzer
from robot_skill_system.openai_integration.schemas import DemonstrationAnalysisInput
from robot_skill_system.perception.interfaces import PerceptionResult
from robot_skill_system.perception.mock_perception import MockPerception
from robot_skill_system.primitives.profiles import (
    load_force_profiles,
    load_motion_profiles,
    load_safety_policies,
)
from robot_skill_system.primitives.registry import get_default_registry
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.event_log import InMemoryEventSink
from robot_skill_system.runtime.executor import RuntimeExecutor
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.models import ExecutionMode, RuntimeContext
from robot_skill_system.runtime.preflight import PreflightPolicy, PreflightValidator
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor
from robot_skill_system.scene.models import CameraMetadata, SceneSnapshot
from robot_skill_system.settings import Settings
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.skills.loader import load_compiled_run
from robot_skill_system.skills.models import (
    BindingSpec,
    EntityKind,
    SkillGraph,
    SkillLifecycleStatus,
    SkillManifest,
    SkillNode,
    SkillType,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationStatus,
)
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import Database, StorageRepository


class OfflineDemoResult(BaseModel):
    """Small JSON-safe summary emitted by the CLI and examples."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    success: bool
    skill_id: str
    version: str
    scene_id: str
    fitted_operations: list[str]
    compiled_skill_uri: str
    compiled_checksum_sha256: str
    validation_passed: bool
    preflight_mock: bool
    robot_commands: list[str]
    execution_events: list[str]
    database_url: str


def _relative(position_m: tuple[float, float, float]) -> dict[str, object]:
    return {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": position_m[0], "y": position_m[1], "z": position_m[2]},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def capture_mock_scene(*, now_ns: int | None = None, frame_count: int = 5) -> SceneSnapshot:
    """Perform one user-visible mock burst and construct one local SceneSnapshot."""

    timestamp_ns = time.time_ns() if now_ns is None else now_ns
    capture_backend = MockCapture(
        MockCaptureConfig(
            default_mode=CaptureMode.BURST,
            default_burst_frame_count=frame_count,
            start_timestamp_ns=timestamp_ns - (frame_count // 2) * 33_333_333,
        )
    )
    capture = capture_backend.capture(
        CaptureRequest(mode=CaptureMode.BURST, frame_count=frame_count)
    )
    perception: PerceptionResult = MockPerception().analyze(capture)
    frame = capture.representative_frame
    camera = CameraMetadata(
        camera_id="mock_d435i",
        color_frame_id=frame.reference_frame,
        depth_frame_id="camera_depth_optical_frame",
        width_px=frame.color_intrinsics.width_px,
        height_px=frame.color_intrinsics.height_px,
        depth_scale_m=frame.depth_scale_m,
        capture_mode="burst",
        frame_count=len(capture.frames),
    )
    return perception.to_scene_snapshot(
        scene_id=f"scene_{timestamp_ns}",
        valid_for_ms=5_000,
        calibration_id="mock_calibration_v1",
        camera_metadata=camera,
    )


def build_wipe_skill_graph() -> tuple[SkillGraph, list[str]]:
    """Fit synthetic line/arc/line evidence and materialize an anchor-relative graph."""

    line_forward = fit_line(generate_line_trajectory().samples)
    arc = fit_arc(generate_arc_trajectory().samples)
    line_backward = fit_line(
        generate_line_trajectory(
            start_m=(0.24, 0.06, 0.05), end_m=(0.0, 0.0, 0.05)
        ).samples
    )
    periodic = fit_periodic(generate_periodic_trajectory().samples)
    novice_processed = preprocess_trajectory(generate_novice_wipe_trajectory())
    segmentation = segment_trajectory(novice_processed)
    if segmentation.reteach_required:
        raise ValueError(
            "synthetic teaching fixture unexpectedly requires retake: "
            f"{segmentation.reason}"
        )
    fitted_operations = [
        line_forward.primitive_id,
        arc.primitive_id,
        line_backward.primitive_id,
        periodic.primitive_id,
    ]
    pre_approach = _relative((line_forward.start_m[0], line_forward.start_m[1], 0.10))
    contact_start = _relative(line_forward.start_m)
    stroke_end = _relative(line_forward.end_m)
    arc_via = _relative(arc.via_m or arc.start_m)
    arc_end = _relative(arc.end_m)
    return (
        SkillGraph(
            skill_id="wipe_surface",
            version="1.0.0",
            name="평면 걸레질",
            description="표면 프레임에 상대적인 직선-원호-직선 접촉 경로",
            skill_type=SkillType.CONTACT,
            source_demonstrations=["synthetic_novice_wipe"],
            operator_style="safe",
            required_tools=["wiper"],
            required_entity_roles={"$surface": "contact_target"},
            bindings={
                "$tool": BindingSpec(
                    variable="$tool",
                    entity_kind=EntityKind.TOOL,
                    class_name="wiper",
                    minimum_confidence=0.8,
                    must_be_attached=True,
                    compatible_skill="wipe_surface",
                ),
                "$surface": BindingSpec(
                    variable="$surface",
                    entity_kind=EntityKind.SURFACE,
                    role="contact_target",
                    minimum_confidence=0.8,
                ),
            },
            nodes=[
                SkillNode(
                    node_id="validate_path",
                    operation="workspace.validate_path",
                    arguments={"path": [pre_approach, contact_start]},
                    on_success="approach_joint",
                ),
                SkillNode(
                    node_id="approach_joint",
                    operation="motion.move_l",
                    arguments={"target": pre_approach, "motion_profile_id": "linear_slow"},
                    on_success="approach_linear",
                    checkpoint="pre_approach",
                ),
                SkillNode(
                    node_id="approach_linear",
                    operation="motion.move_l",
                    arguments={"target": contact_start, "motion_profile_id": "linear_slow"},
                    on_success="contact_search",
                ),
                SkillNode(
                    node_id="contact_search",
                    operation="contact.search_surface",
                    arguments={"surface": "$surface", "force_profile_id": "wipe_standard"},
                    on_success="force_enable",
                ),
                SkillNode(
                    node_id="force_enable",
                    operation="contact.enable_force",
                    arguments={"surface": "$surface", "force_profile_id": "wipe_standard"},
                    on_success="stroke_forward",
                ),
                SkillNode(
                    node_id="stroke_forward",
                    operation="motion.move_l",
                    arguments={"target": stroke_end, "motion_profile_id": "linear_normal"},
                    on_success="turn_arc",
                ),
                SkillNode(
                    node_id="turn_arc",
                    operation="motion.move_c",
                    arguments={
                        "via": arc_via,
                        "target": arc_end,
                        "motion_profile_id": "circular_normal",
                    },
                    on_success="stroke_backward",
                ),
                SkillNode(
                    node_id="stroke_backward",
                    operation="motion.move_l",
                    arguments={
                        "target": _relative(line_backward.end_m),
                        "motion_profile_id": "linear_normal",
                    },
                    on_success="force_disable",
                ),
                SkillNode(
                    node_id="force_disable",
                    operation="contact.disable_force",
                    arguments={},
                    on_success="retract",
                ),
                SkillNode(
                    node_id="retract",
                    operation="recovery.safe_retract",
                    arguments={"recovery_profile_id": "safe_retract_default"},
                ),
            ],
            start_node="validate_path",
            terminal_nodes=["retract"],
            motion_profiles=["joint_safe", "linear_slow", "linear_normal", "circular_normal"],
            force_profiles=["wipe_standard"],
            preconditions=["fresh_scene", "attached_wiper", "contact_target"],
            postconditions=["force_released", "surface_retracted"],
            validation_status=ValidationStatus.PENDING,
            lifecycle_status=SkillLifecycleStatus.CANDIDATE,
        ),
        fitted_operations,
    )


def _database_from_settings(settings: Settings) -> Database:
    if not settings.database_url.startswith("sqlite"):
        return Database(settings.database_url)
    return Database(settings.database_url)


def run_offline_demo(settings: Settings | None = None) -> OfflineDemoResult:
    """Run the complete offline MVP and persist artifacts, registry rows, and events."""

    configured = settings or Settings.from_env()
    configured.artifact_root.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(configured.artifact_root)
    database = _database_from_settings(configured)
    database.create_schema()
    repository = StorageRepository(database)
    scene = capture_mock_scene(frame_count=configured.scene_burst_frame_count)
    try:
        repository.record_scene(scene)
    except Exception as exc:
        # Re-running the deterministic CLI may encounter the extremely unlikely same scene ID;
        # all other database errors must remain visible.
        if "UNIQUE constraint failed: scenes.id" not in str(exc):
            raise

    graph, fitted_operations = build_wipe_skill_graph()
    local_fits = [{"operation": operation} for operation in fitted_operations]
    semantic_analysis, _metadata = DemonstrationAnalyzer(configured).analyze(
        DemonstrationAnalysisInput(
            transcript_text="파란 걸레로 오른쪽 테이블을 닦는다",
            scene_summary={"tool_id": "wiper_01", "target_ids": ["table_surface_01"]},
            pose_summary={"anchor_frame": "$surface", "sampled_fps": 10},
            entity_catalog=["wiper_01", "table_surface_01"],
            primitive_catalog=[
                item.operation_name for item in get_default_registry().catalog()
            ],
            approved_motion_profiles=[
                "joint_safe",
                "linear_slow",
                "linear_normal",
                "circular_normal",
            ],
            approved_force_profiles=["wipe_standard"],
            motion_fitting_candidates=local_fits,
            confidence_summary={"pose": 0.98, "scene": 0.98},
        )
    )
    if semantic_analysis.reteach_required:
        raise ValueError("mock semantic analysis rejected the teaching fixture")
    graph_report = SkillGraphValidator().validate(graph)
    compiler = SkillCompiler()
    compilation = compiler.compile(graph)

    base_uri = f"skills/{graph.skill_id}/{graph.version}"
    graph_artifact = store.put_json(
        f"{base_uri}/skill_graph.json", graph.model_dump(mode="json")
    )
    code_artifact = store.put_text(
        f"{base_uri}/compiled_skill.py",
        compilation.source,
        media_type="text/x-python; charset=utf-8",
    )
    validation = ValidationReport(
        skill_id=graph.skill_id,
        version=graph.version,
        passed=graph_report.valid and compilation.validation_report.valid,
        issues=[
            ValidationIssue(
                code="graph_warning",
                message=warning,
                severity=ValidationSeverity.WARNING,
            )
            for warning in graph_report.warnings
        ],
        checks={
            "schema": True,
            "primitive_allowlist": graph_report.valid,
            "ast": compilation.validation_report.valid,
            "py_compile": compilation.validation_report.py_compile_passed,
        },
        graph_checksum_sha256=graph_artifact.checksum_sha256,
        generated_code_checksum_sha256=code_artifact.checksum_sha256,
        mock_validation=False,
        hardware_validated=False,
        timestamp_ns=0,
    )
    validation_artifact = store.put_json(
        f"{base_uri}/validation_report.json", validation.model_dump(mode="json")
    )
    manifest = SkillManifest(
        skill_id=graph.skill_id,
        version=graph.version,
        skill_graph_uri=graph_artifact.uri,
        skill_graph_checksum_sha256=graph_artifact.checksum_sha256,
        compiled_skill_uri=code_artifact.uri,
        compiled_skill_checksum_sha256=code_artifact.checksum_sha256,
        validation_report_uri=validation_artifact.uri,
        validation_report_checksum_sha256=validation_artifact.checksum_sha256,
        source_demonstration_uris=["demonstrations/synthetic_novice_wipe"],
    )
    store.put_json(f"{base_uri}/manifest.json", manifest.model_dump(mode="json"))
    if not compiler.verify_manifest(manifest, configured.artifact_root):
        raise ValueError("freshly written skill artifacts failed checksum verification")
    version_record = repository.active_skill_version(
        name=graph.name, variant=graph.operator_style or "default"
    )
    if version_record is None:
        version_record = repository.register_skill_version(
            name=graph.name,
            intent="wipe_surface",
            semantic_version=graph.version,
            graph=graph,
            status="candidate",
            variant=graph.operator_style or "default",
            description=graph.description,
            generated_code_uri=code_artifact.uri,
            generated_code_checksum_sha256=code_artifact.checksum_sha256,
            validation_status="pending",
            hardware_compatible=False,
        )

    binder = EntityBinder()
    bindings = binder.bind_entities(
        scene,
        graph.binding_requirements(),
        maximum_scene_age_ms=configured.scene_freshness_ms,
    )
    robot = MockRobotAdapter()
    gripper = MockGripperAdapter()
    robot.connect()
    gripper.connect()
    motion_profiles = load_motion_profiles(
        configured.repo_root / "configs/motion_profiles/default.json"
    )
    force_profiles = load_force_profiles(
        configured.repo_root / "configs/force_profiles/default.json"
    )
    safety_policy = load_safety_policies(
        configured.repo_root / "configs/safety_policies/default.json"
    )["global_default"]
    preflight = PreflightValidator(
        policy=PreflightPolicy(
            minimum_clearance_m=safety_policy.minimum_clearance_m
        )
    ).validate(
        scene=scene,
        skill=graph,
        bindings=bindings,
        robot=robot,
        execution_mode=ExecutionMode.MOCK,
        enable_hardware_execution=False,
        skill_validation_status="passed",
        expected_calibration_id="mock_calibration_v1",
        expected_tool_id="wiper_01",
        expected_tool_class="wiper",
        global_safety_active=True,
        workspace_monitor_active=True,
        force_supervisor_active=True,
        skill_uses_force=True,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        safety_policy=safety_policy,
        robot_backend="mock",
        enable_real_robot=False,
        dry_run=True,
    )
    context = RuntimeContext(
        scene=scene,
        bindings=bindings,
        motion_profiles=motion_profiles,
        force_profiles=force_profiles,
        execution_mode=ExecutionMode.MOCK,
        skill=graph,
        preflight_report=preflight,
        safety_policy=safety_policy,
        command_text="파란 걸레로 오른쪽 테이블을 닦아줘",
    )
    event_sink = InMemoryEventSink()
    contact_links = tuple(
        link for surface in scene.surfaces for link in surface.allowed_contact_links
    )
    force_supervisor = GlobalForceSupervisor(
        robot,
        force_profiles,
        allowed_contact_links=contact_links,
        maximum_force_n=safety_policy.maximum_force_n,
    )
    workspace_supervisor = GlobalWorkspaceSupervisor(
        minimum_clearance_m=safety_policy.minimum_clearance_m
    )
    executor = RuntimeExecutor(
        context=context,
        robot=robot,
        gripper=gripper,
        binder=binder,
        workspace_supervisor=workspace_supervisor,
        force_supervisor=force_supervisor,
        safety_supervisor=GlobalSafetySupervisor(),
        primitive_registry=get_default_registry(),
        event_sink=event_sink,
    )
    compiled_run = load_compiled_run(
        Path(code_artifact.uri),
        artifact_root=configured.artifact_root,
        expected_checksum_sha256=code_artifact.checksum_sha256,
    )
    asyncio.run(executor.execute_compiled(compiled_run))
    force_supervisor.assert_released()

    repository.record_validation_run(
        skill_version_id=version_record.id,
        status="passed",
        result={
            "passed": True,
            "mock_validation": True,
            "preflight": preflight.as_dict(),
            "robot_commands": [command.operation for command in robot.commands],
            "events": [event.event_type for event in event_sink.events],
        },
    )
    with database.session() as session:
        attached_version = session.get(type(version_record), version_record.id)
        if attached_version is None:
            raise KeyError(version_record.id)
        attached_version.validation_status = "passed"
        if attached_version.status == "candidate":
            attached_version.status = "validated"
    version_record = repository.activate_skill_version(version_record.id)

    execution = repository.start_execution(
        started_at_ns=time.time_ns(),
        execution_mode="mock",
        skill_version_id=version_record.id,
        scene_id=scene.scene_id,
        command_text=context.command_text,
        preflight=preflight.as_dict(),
        bindings={name: binding.entity_id for name, binding in bindings.items()},
    )
    for event in event_sink.events:
        repository.append_execution_event(
            execution.id,
            timestamp_ns=event.timestamp_ns,
            event_type=event.event_type,
            severity=event.severity,
            details=event.details,
        )
    repository.finish_execution(
        execution.id, status="succeeded", ended_at_ns=time.time_ns()
    )
    result = OfflineDemoResult(
        success=True,
        skill_id=graph.skill_id,
        version=graph.version,
        scene_id=scene.scene_id,
        fitted_operations=fitted_operations,
        compiled_skill_uri=code_artifact.uri,
        compiled_checksum_sha256=code_artifact.checksum_sha256,
        validation_passed=validation.passed,
        preflight_mock=all(
            check.is_mock
            for check in preflight.checks
            if check.source == "mock_geometry"
        ),
        robot_commands=[command.operation for command in robot.commands],
        execution_events=[event.event_type for event in event_sink.events],
        database_url=configured.database_url,
    )
    database.close()
    return result
