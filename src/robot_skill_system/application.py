"""Synchronous application service shared by FastAPI and the CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from sqlalchemy import select

from robot_skill_system.adapters.doosan_m0609 import DoosanM0609Adapter
from robot_skill_system.adapters.mock_robot import MockGripperAdapter, MockRobotAdapter
from robot_skill_system.adapters.onrobot_rg2 import OnRobotRG2Adapter
from robot_skill_system.aruco_experiment.controller import (
    ArucoExperimentController,
    DoosanArucoExperimentRobot,
    MockArucoExperimentRobot,
)
from robot_skill_system.calibration.controller import HandEyeCalibrationController
from robot_skill_system.calibration.robot import DoosanHandEyeCalibrationRobot
from robot_skill_system.calibration.task_plane import (
    camera_stationarity_diagnostics,
    compose_base_task_plane,
    rigid_transform_from_matrix,
    task_plane_normal_hint_diagnostics,
)
from robot_skill_system.capture.realsense_capture import (
    RealSenseCapture,
    RealSenseCaptureConfig,
)
from robot_skill_system.capture.rgbd_recording import RGBDCameraController
from robot_skill_system.demonstrations.models import (
    DemonstrationTrajectory,
    PoseSample,
    ProcessedTrajectory,
)
from robot_skill_system.demonstrations.models import (
    PrimitiveRecommendation as LocalPrimitiveRecommendation,
)
from robot_skill_system.demonstrations.path_simplification import (
    DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M,
    simplify_anchor_relative_path,
)
from robot_skill_system.demonstrations.preprocessing import (
    PreprocessingConfig,
    TrajectoryPreprocessingError,
    preprocess_trajectory,
)
from robot_skill_system.demonstrations.primitive_fitter import (
    PrimitiveFittingError,
    fit_periodic_primitive_geometry,
    recommend_primitive,
)
from robot_skill_system.demonstrations.quality import (
    QualityAssessment,
    assess_trajectory_quality,
)
from robot_skill_system.demonstrations.recorder import load_demonstration
from robot_skill_system.demonstrations.rgbd_dataset import (
    RGBDDatasetSegmentation,
    load_rgbd_dataset,
    segment_rgbd_dataset,
)
from robot_skill_system.demonstrations.rgbd_geometry import (
    ManualTCPPathSample,
    PixelPoint,
    calibrate_surface_from_three_points,
    manual_two_finger_sample,
    segment_dominant_depth_plane,
    transform_camera_pose_to_surface,
    validate_surface_relative_path,
)
from robot_skill_system.demonstrations.segmentation import segment_trajectory
from robot_skill_system.demonstrations.stage_segmentation import (
    GripActionEndSegmentationConfig,
    GripActionEndSegmentationError,
)
from robot_skill_system.demonstrations.synthetic import (
    generate_expert_wipe_trajectory,
    generate_novice_wipe_trajectory,
    generate_periodic_trajectory,
)
from robot_skill_system.demonstrations.trajectory import (
    TrajectorySummary,
    summarize_trajectory,
)
from robot_skill_system.exceptions import NotConfiguredError, RobotSkillError
from robot_skill_system.jog.controller import DoosanJogRobot, JogController, MockJogRobot
from robot_skill_system.openai_integration.demonstration_analyzer import DemonstrationAnalyzer
from robot_skill_system.openai_integration.embeddings import (
    SkillEmbeddingService,
    SkillSearchDocument,
)
from robot_skill_system.openai_integration.function_tools import (
    FunctionScene,
    FunctionSkillManifest,
    FunctionSkillVersion,
    LocalReadOnlyToolProvider,
    SafeFunctionDispatcher,
)
from robot_skill_system.openai_integration.intent_resolver import RuntimeIntentResolver
from robot_skill_system.openai_integration.motion_policy import (
    MAXIMUM_ACTION_MOTION_BLOCKS,
    MAXIMUM_END_MOTION_BLOCKS,
    RECORDING_BLOCK_MOTION_OPERATIONS,
)
from robot_skill_system.openai_integration.recording_skill_analyzer import (
    ImageInputRejectedError,
    RecordingSkillDraftAnalyzer,
)
from robot_skill_system.openai_integration.schemas import (
    MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES,
    DemonstrationAnalysisInput,
    RecordingSkillDraftInput,
)
from robot_skill_system.openai_integration.task_intent_resolver import TaskIntentResolver
from robot_skill_system.openai_integration.training_semantics import (
    TrainingSemanticResolver,
    TrainingTaskSemantics,
)
from robot_skill_system.perception._geometry import rotation_matrix_to_quaternion_xyzw
from robot_skill_system.perception.hand_pose import (
    FingerObservation,
    MediaPipeHandPoseEstimator,
)
from robot_skill_system.perception.semantic_anchor import (
    reconstruct_semantic_roi_anchor,
)
from robot_skill_system.primitives.models import SafetyPolicy
from robot_skill_system.primitives.profiles import (
    load_force_profiles,
    load_grasp_verification_profiles,
    load_motion_profiles,
    load_safety_policies,
)
from robot_skill_system.primitives.registry import get_default_registry
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import ExecutionAbortedError, RuntimeErrorBase
from robot_skill_system.runtime.event_log import InMemoryEventSink
from robot_skill_system.runtime.executor import RuntimeExecutor
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.hardware_verification import (
    DoosanFixedPlaneGeometryValidator,
    DoosanStateMonitor,
    FixedReferenceSceneMonitor,
)
from robot_skill_system.runtime.integrity import verify_skill_checksum
from robot_skill_system.runtime.models import BoundTargetPose, ExecutionMode, RuntimeContext
from robot_skill_system.runtime.preflight import PreflightPolicy, PreflightValidator
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.task_flow_materializer import (
    TaskFlowMaterializer,
)
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor
from robot_skill_system.scene.models import (
    AccessPolicy,
    Pose,
    Quaternion,
    SceneSnapshot,
    SurfaceInstance,
    SurfaceRole,
    Vector3,
    WorkspaceRegion,
    WorkspaceRole,
)
from robot_skill_system.scene.transforms import RigidTransform, rotate_vector
from robot_skill_system.settings import ExecutionMode as SettingsExecutionMode
from robot_skill_system.settings import Settings
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.skills.loader import CompiledRun, load_compiled_run
from robot_skill_system.skills.models import (
    BindingSpec,
    EntityKind,
    SkillGraph,
    SkillLifecycleStatus,
    SkillManifest,
    SkillNode,
    SkillType,
    ValidationReport,
    ValidationStatus,
)
from robot_skill_system.skills.promotion import PromotionPolicy
from robot_skill_system.skills.retrieval import (
    SkillCandidate,
    SkillSearchQuery,
    rank_skills,
)
from robot_skill_system.skills.updater import SkillUpdater, UpdateEvidence
from robot_skill_system.skills.versioning import (
    SemanticVersion,
    next_candidate_version,
    stable_version,
)
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import Database, StorageRepository
from robot_skill_system.storage.grip_point_importer import GripPointResultImporter
from robot_skill_system.storage.orm import (
    ActionEndMappingRecord,
    ExecutionRunRecord,
    GripProfileVersionRecord,
    SceneRecord,
    SemanticCatalogRecord,
    SkillRecord,
    SkillVersionRecord,
    StageDefinitionRecord,
    TeachingSessionRecord,
)
from robot_skill_system.vertical_slice import build_wipe_skill_graph, capture_mock_scene

_RECORDING_DRAFT_ID_PATTERN = re.compile(r"^draft_[a-f0-9]{32}$")


@dataclass(frozen=True, slots=True)
class DemonstrationEvidence:
    """Locally validated evidence extracted from one approved demonstration artifact."""

    source: Path
    trajectory: DemonstrationTrajectory
    processed: ProcessedTrajectory
    quality: QualityAssessment
    summary: TrajectorySummary
    recommendation: LocalPrimitiveRecommendation
    promotion_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActiveExecution:
    """Abort handles for a currently running mock/dry-run execution."""

    robot: Any
    force_supervisor: GlobalForceSupervisor
    safety_supervisor: GlobalSafetySupervisor


class MVPApplication:
    """Application workflow with fail-closed mock defaults and durable metadata."""

    def __init__(
        self,
        settings: Settings,
        *,
        camera_controller: RGBDCameraController | None = None,
        calibration_controller: HandEyeCalibrationController | None = None,
        jog_controller: JogController | None = None,
        aruco_experiment_controller: ArucoExperimentController | None = None,
    ) -> None:
        self.settings = settings
        self.store = LocalArtifactStore(settings.artifact_root)
        self.database = Database(settings.database_url)
        self.database.create_schema()
        self.repository = StorageRepository(self.database)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._scenes: dict[str, SceneSnapshot] = {}
        self._active_executions: dict[str, ActiveExecution] = {}
        self._active_execution_lock = threading.RLock()
        self._robot_motion_transition_lock = threading.RLock()
        self._fixed_hardware_session_lock = threading.RLock()
        self._fixed_hardware_robot: DoosanArucoExperimentRobot | None = None
        self._fixed_base_plane_cache: tuple[Any, str] | None = None
        self._recording_flange_starts: dict[str, dict[str, Any]] = {}
        real_sense_config = RealSenseCaptureConfig(
            width_px=settings.realsense_width_px,
            height_px=settings.realsense_height_px,
            frames_per_second=settings.realsense_frames_per_second,
            device_serial=settings.realsense_device_serial,
            maximum_timestamp_skew_ms=settings.rgbd_max_timestamp_delta_ms,
        )
        self.camera_controller = camera_controller or RGBDCameraController(
            lambda: RealSenseCapture(real_sense_config),
            self.store,
            frames_per_second=settings.realsense_frames_per_second,
            recording_frames_per_second=(
                settings.realsense_recording_frames_per_second
            ),
            maximum_recording_duration_s=(
                settings.realsense_maximum_recording_duration_s
            ),
        )
        calibration_gates = {
            "ROBOT_EXECUTION_MODE=hardware": (
                settings.robot_execution_mode is SettingsExecutionMode.HARDWARE
            ),
            "ENABLE_HARDWARE_EXECUTION=true": settings.enable_hardware_execution,
            "ROBOT_BACKEND=doosan": settings.robot_backend == "doosan",
            "ENABLE_REAL_ROBOT=true": settings.enable_real_robot,
            "DRY_RUN=false": not settings.dry_run,
            "ENABLE_HANDEYE_CALIBRATION=true": settings.enable_handeye_calibration,
            "CALIBRATION_POSE_PLAN_APPROVED=true": (
                settings.calibration_pose_plan_approved
            ),
            "CALIBRATION_CELL_SAFETY_VERIFIED=true": (
                settings.calibration_cell_safety_verified
            ),
        }
        self.calibration_controller = calibration_controller or HandEyeCalibrationController(
            store=self.store,
            camera=self.camera_controller,
            robot_factory=lambda: DoosanHandEyeCalibrationRobot(
                robot_id=settings.doosan_robot_id,
                robot_model=settings.doosan_robot_model,
                execution_mode=settings.robot_execution_mode.value,
                hardware_enabled=settings.handeye_hardware_enabled,
            ),
            hardware_authorized=settings.handeye_hardware_enabled,
            gate_summary=calibration_gates,
            legacy_npy_path=settings.handeye_legacy_npy_path,
            legacy_expected_tcp_name=settings.handeye_legacy_expected_tcp,
        )
        jog_gates = {
            "ROBOT_EXECUTION_MODE=hardware": (
                settings.robot_execution_mode is SettingsExecutionMode.HARDWARE
            ),
            "ENABLE_HARDWARE_EXECUTION=true": settings.enable_hardware_execution,
            "ROBOT_BACKEND=doosan": settings.robot_backend == "doosan",
            "ENABLE_REAL_ROBOT=true": settings.enable_real_robot,
            "DRY_RUN=false": not settings.dry_run,
            "ENABLE_WEB_JOG=true": settings.enable_web_jog,
            "JOG_CELL_SAFETY_VERIFIED=true": settings.jog_cell_safety_verified,
        }
        jog_profiles = load_motion_profiles(
            settings.repo_root / "configs/motion_profiles/default.json"
        )
        jog_profile = jog_profiles["joint_safe"]
        jog_linear_profile = jog_profiles["linear_slow"]
        if (
            jog_profile.joint_velocity_rad_s is None
            or jog_profile.joint_acceleration_rad_s2 is None
        ):
            raise ValueError("joint_safe must define joint velocity and acceleration")
        if (
            jog_linear_profile.linear_velocity_m_s is None
            or jog_linear_profile.linear_acceleration_m_s2 is None
            or jog_linear_profile.angular_velocity_rad_s is None
            or jog_linear_profile.angular_acceleration_rad_s2 is None
        ):
            raise ValueError("linear_slow must define linear and angular limits")
        use_hardware_jog = settings.jog_hardware_enabled
        self.jog_controller = jog_controller or JogController(
            robot_factory=(
                lambda: DoosanJogRobot(
                    robot_id=settings.doosan_robot_id,
                    robot_model=settings.doosan_robot_model,
                    execution_mode=settings.robot_execution_mode.value,
                    hardware_enabled=settings.jog_hardware_enabled,
                )
                if use_hardware_jog
                else MockJogRobot()
            ),
            mode="hardware" if use_hardware_jog else "mock",
            hardware_authorized=use_hardware_jog,
            gate_summary=jog_gates,
            joint_velocity_rad_s=(
                jog_profile.joint_velocity_rad_s * jog_profile.safety_scale
            ),
            joint_acceleration_rad_s2=(
                jog_profile.joint_acceleration_rad_s2 * jog_profile.safety_scale
            ),
            linear_velocity_m_s=(
                jog_linear_profile.linear_velocity_m_s
                * jog_linear_profile.safety_scale
            ),
            linear_acceleration_m_s2=(
                jog_linear_profile.linear_acceleration_m_s2
                * jog_linear_profile.safety_scale
            ),
            angular_velocity_rad_s=(
                jog_linear_profile.angular_velocity_rad_s
                * jog_linear_profile.safety_scale
            ),
            angular_acceleration_rad_s2=(
                jog_linear_profile.angular_acceleration_rad_s2
                * jog_linear_profile.safety_scale
            ),
        )
        aruco_gates = {
            "ROBOT_EXECUTION_MODE=hardware": (
                settings.robot_execution_mode is SettingsExecutionMode.HARDWARE
            ),
            "ENABLE_HARDWARE_EXECUTION=true": settings.enable_hardware_execution,
            "ROBOT_BACKEND=doosan": settings.robot_backend == "doosan",
            "ENABLE_REAL_ROBOT=true": settings.enable_real_robot,
            "DRY_RUN=false": not settings.dry_run,
            "ENABLE_ARUCO_EXPERIMENT=true": settings.enable_aruco_experiment,
            "ARUCO_EXPERIMENT_CELL_SAFETY_VERIFIED=true": (
                settings.aruco_experiment_cell_safety_verified
            ),
        }
        linear_profile = load_motion_profiles(
            settings.repo_root / "configs/motion_profiles/default.json"
        )["linear_slow"]
        if (
            linear_profile.linear_velocity_m_s is None
            or linear_profile.linear_acceleration_m_s2 is None
            or linear_profile.angular_velocity_rad_s is None
            or linear_profile.angular_acceleration_rad_s2 is None
        ):
            raise ValueError("linear_slow must define linear and angular limits")
        use_hardware_aruco = settings.aruco_experiment_hardware_enabled
        self.aruco_experiment_controller = (
            aruco_experiment_controller
            or ArucoExperimentController(
                robot_factory=(
                    lambda: DoosanArucoExperimentRobot(
                        robot_id=settings.doosan_robot_id,
                        robot_model=settings.doosan_robot_model,
                        execution_mode=settings.robot_execution_mode.value,
                        hardware_enabled=settings.aruco_experiment_hardware_enabled,
                        expected_tcp_name=settings.aruco_experiment_expected_tcp,
                    )
                    if use_hardware_aruco
                    else MockArucoExperimentRobot(
                        active_tcp_name=settings.aruco_experiment_expected_tcp
                    )
                ),
                mode="hardware" if use_hardware_aruco else "mock",
                hardware_authorized=use_hardware_aruco,
                gate_summary=aruco_gates,
                reference_npz=settings.aruco_fixed_reference_npz,
                runtime_npz=settings.aruco_runtime_workspace_npz,
                expected_tcp_name=settings.aruco_experiment_expected_tcp,
                joint_velocity_rad_s=(
                    jog_profile.joint_velocity_rad_s * jog_profile.safety_scale
                ),
                joint_acceleration_rad_s2=(
                    jog_profile.joint_acceleration_rad_s2 * jog_profile.safety_scale
                ),
                linear_velocity_m_s=(
                    linear_profile.linear_velocity_m_s * linear_profile.safety_scale
                ),
                linear_acceleration_m_s2=(
                    linear_profile.linear_acceleration_m_s2
                    * linear_profile.safety_scale
                ),
                angular_velocity_rad_s=(
                    linear_profile.angular_velocity_rad_s * linear_profile.safety_scale
                ),
                angular_acceleration_rad_s2=(
                    linear_profile.angular_acceleration_rad_s2
                    * linear_profile.safety_scale
                ),
            )
        )

    def close(self) -> None:
        """Release database resources."""

        with self._fixed_hardware_session_lock:
            fixed_robot, self._fixed_hardware_robot = self._fixed_hardware_robot, None
        if fixed_robot is not None:
            fixed_robot.disconnect()
        self.aruco_experiment_controller.close()
        self.jog_controller.close()
        self.calibration_controller.close()
        self.camera_controller.close()
        self.database.close()

    def create_teaching_session(self, request: dict[str, Any]) -> dict[str, Any]:
        started_at_ns = time.time_ns()
        record = self.repository.create_teaching_session(
            started_at_ns=started_at_ns,
            metadata={
                "operator_id": request.get("operator_id", "operator_mock"),
                "operator_role": request.get("operator_role", "operator"),
                "notes": request.get("notes"),
            },
        )
        session_id = record.id
        session = {
            "schema_version": "1.0",
            "session_id": session_id,
            "status": "created",
            "operator_id": request.get("operator_id", "operator_mock"),
            "operator_role": request.get("operator_role", "operator"),
            "notes": request.get("notes"),
            "created_at_ns": started_at_ns,
            "scene_ids": [],
        }
        self._sessions[session_id] = session
        self.repository.update_teaching_session_metadata(
            session_id, metadata=session, status="created"
        )
        self.store.put_json(
            f"demonstrations/{session_id}/metadata_created.json", session
        )
        return dict(session)

    def capture_teaching_session(
        self, session_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        scene_result = self.capture_scene({"mode": request.get("mode", "mock")})
        scene_ids = list(session["scene_ids"])
        scene_ids.append(scene_result["scene_id"])
        session.update({"status": "capturing", "scene_ids": scene_ids})
        self.repository.update_teaching_session_metadata(
            session_id, metadata=session, status="capturing"
        )
        self.store.put_json(
            f"demonstrations/{session_id}/captures/{len(scene_ids):03d}.json",
            scene_result,
        )
        return dict(session)

    def finish_teaching_session(
        self, session_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        session.update(
            {
                "status": "finished",
                "success": bool(request.get("success", True)),
                "transcript_text": str(request.get("transcript_text", "")),
                "finished_at_ns": time.time_ns(),
            }
        )
        artifact = self.store.put_json(
            f"demonstrations/{session_id}/final.json", session
        )
        self.repository.update_teaching_session_metadata(
            session_id, metadata=session, status="finishing"
        )
        self.repository.finalize_teaching_session(
            session_id,
            ended_at_ns=int(session["finished_at_ns"]),
            status="finished",
            artifact_uri=artifact.uri,
            artifact_checksum_sha256=artifact.checksum_sha256,
        )
        return dict(session)

    def get_teaching_session(self, session_id: str) -> dict[str, Any]:
        return dict(self._require_session(session_id))

    def _require_session(self, session_id: str) -> dict[str, Any]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        record = self.repository.get_teaching_session(session_id)
        if record is None:
            raise KeyError(f"unknown teaching session {session_id!r}")
        session = dict(record.metadata_json)
        session.setdefault("session_id", record.id)
        session["status"] = record.status
        session.setdefault("created_at_ns", record.started_at_ns)
        if record.ended_at_ns is not None:
            session.setdefault("finished_at_ns", record.ended_at_ns)
        self._sessions[session_id] = session
        return session

    def capture_scene(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = str(request.get("mode", "mock"))
        if mode not in {"mock", "single", "burst", "hardware"}:
            raise ValueError("scene capture supports mock/single/burst/hardware")
        frame_count = 1 if mode == "single" else self.settings.scene_burst_frame_count
        scene = capture_mock_scene(frame_count=frame_count)
        if mode == "hardware":
            aruco_robot, base_to_plane = self._acquire_fixed_hardware_session()
            tcp_transform = rigid_transform_from_matrix(
                aruco_robot.get_base_to_tcp_matrix(), label="hardware T_base_tcp"
            )
            active_tool = scene.tools[0].model_copy(
                update={
                    "instance_id": "active_rg2",
                    "tool_class": "onrobot_rg2",
                    "tcp_frame": self.settings.aruco_experiment_expected_tcp,
                    "pose": Pose(
                        frame_id="base",
                        position_m=tcp_transform.translation_m,
                        orientation_xyzw=tcp_transform.rotation_xyzw,
                        timestamp_ns=time.time_ns(),
                        source="live_doosan_active_tcp_feedback",
                        confidence=1.0,
                    ),
                    "compatible_skills": ["take_hammer"],
                    "verification_confidence": 1.0,
                },
                deep=True,
            )
            scene = scene.model_copy(
                update={
                    "scene_id": f"hardware_{uuid.uuid4().hex}",
                    "timestamp_ns": time.time_ns(),
                    "reference_frame": "base",
                    "valid_for_ms": 1_800_000,
                    "calibration_id": "aruco_fixed_plane_operator_session",
                    "objects": [],
                    "tools": [active_tool],
                    "surfaces": [],
                    "workspace_regions": [],
                    "static_obstacles": [],
                    "dynamic_obstacles": [],
                },
                deep=True,
            )
            self._scenes[scene.scene_id] = scene
            self.store.put_json(
                f"scenes/{scene.scene_id}_base_plane.json",
                {
                    "scene_id": scene.scene_id,
                    "T_base_plane": base_to_plane.tolist(),
                    "source": "enabled_aruco_reference_session",
                },
            )
        self.repository.record_scene(scene)
        self.store.put_json(
            f"scenes/{scene.scene_id}.json", scene.model_dump(mode="json")
        )
        self._scenes[scene.scene_id] = scene
        return scene.model_dump(mode="json")

    def get_scene(self, scene_id: str) -> dict[str, Any]:
        return self._scene(scene_id).model_dump(mode="json")

    def get_camera_status(self) -> dict[str, Any]:
        status = self.camera_controller.status()
        status["maximum_timestamp_skew_ms"] = (
            self.settings.rgbd_max_timestamp_delta_ms
        )
        return status

    def get_handeye_calibration_status(self) -> dict[str, Any]:
        return self.calibration_controller.status()

    def get_jog_status(self) -> dict[str, Any]:
        return self.jog_controller.status()

    def get_aruco_experiment_status(self) -> dict[str, Any]:
        status = self.aruco_experiment_controller.status()
        try:
            transform, source = self._load_fixed_base_plane()
            status["fixed_workspace_ready"] = True
            status["fixed_workspace_source"] = source
            status["fixed_T_base_plane"] = transform.tolist()
            status["skill_execution_requires_aruco_enable"] = False
        except Exception as exc:
            status["fixed_workspace_ready"] = False
            status["fixed_workspace_error"] = str(exc)
        return status

    def enable_jog(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._robot_motion_transition_lock:
            self._ensure_no_other_robot_motion("enable jog")
            return self.jog_controller.enable(
                operator_id=str(request.get("operator_id") or ""),
                workspace_cleared=request.get("workspace_cleared") is True,
                estop_ready=request.get("estop_ready") is True,
                acknowledge_direct_motion=(
                    request.get("acknowledge_direct_motion") is True
                ),
            )

    def move_jog_joint(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_no_other_robot_motion("jog")
        return self.jog_controller.move_joint(
            joint_index=int(request["joint_index"]),
            delta_deg=float(request["delta_deg"]),
        )

    def move_jog_joints(self, request: dict[str, Any]) -> dict[str, Any]:
        target = request["target_joint_positions_deg"]
        if not isinstance(target, (list, tuple)):
            raise ValueError("movej target must be a six-angle sequence")
        with self._robot_motion_transition_lock:
            self._ensure_no_other_robot_motion("movej")
            return self.jog_controller.move_to_joint_positions(
                target_joint_positions_deg=tuple(float(value) for value in target)
            )

    def move_jog_linear(self, request: dict[str, Any]) -> dict[str, Any]:
        target = request["target_tcp_pose_base_mm_zyz_deg"]
        if not isinstance(target, (list, tuple)):
            raise ValueError("movel target must be a six-value pose")
        with self._robot_motion_transition_lock:
            self._ensure_no_other_robot_motion("movel")
            return self.jog_controller.move_to_cartesian_pose(
                target_tcp_pose_base_mm_zyz_deg=tuple(float(value) for value in target)
            )

    def stop_jog(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.jog_controller.stop(
            reason=str(request.get("reason") or "operator_request")
        )

    def enable_aruco_experiment(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._robot_motion_transition_lock:
                self._ensure_aruco_motion_available("enable ArUco experiment")
                return self.aruco_experiment_controller.enable(
                    operator_id=str(request.get("operator_id") or ""),
                    workspace_cleared=request.get("workspace_cleared") is True,
                    estop_ready=request.get("estop_ready") is True,
                    acknowledge_direct_motion=(
                        request.get("acknowledge_direct_motion") is True
                    ),
                    object_width_mm=float(request["object_width_mm"]),
                    width_model=str(request.get("width_model") or "full-opening"),
                )
        except Exception as exc:
            self.aruco_experiment_controller.record_failure("enable", exc)
            raise

    def move_aruco_reference(self) -> dict[str, Any]:
        try:
            self._ensure_aruco_motion_available("move ArUco reference")
            return self.aruco_experiment_controller.move_to_reference()
        except Exception as exc:
            self.aruco_experiment_controller.record_failure("move_to_reference", exc)
            raise

    def move_aruco_plane_z_test(self) -> dict[str, Any]:
        try:
            self._ensure_aruco_motion_available("run ArUco +Z test")
            return self.aruco_experiment_controller.move_plane_z_test()
        except Exception as exc:
            self.aruco_experiment_controller.record_failure("move_plane_z_test", exc)
            raise

    def stop_aruco_experiment(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.aruco_experiment_controller.stop(
            reason=str(request.get("reason") or "operator_request")
        )

    def _ensure_aruco_motion_available(self, action: str) -> None:
        with self._active_execution_lock:
            if self._active_executions:
                raise ValueError(f"cannot {action} while a skill execution is active")
        if self.jog_controller.enabled:
            raise ValueError(f"cannot {action} while web jog is enabled")
        calibration = self.calibration_controller.status().get("session")
        if isinstance(calibration, dict) and calibration.get("status") in {
            "starting",
            "moving_to_reference",
            "running",
            "aborting",
        }:
            raise ValueError(f"cannot {action} while hand-eye calibration is active")

    def _ensure_no_other_robot_motion(self, action: str) -> None:
        with self._active_execution_lock:
            if self._active_executions:
                raise ValueError(f"cannot {action} while a skill execution is active")
        calibration = self.calibration_controller.status().get("session")
        if isinstance(calibration, dict) and calibration.get("status") in {
            "starting",
            "moving_to_reference",
            "running",
            "aborting",
        }:
            raise ValueError(f"cannot {action} while hand-eye calibration is active")
        if self.aruco_experiment_controller.enabled:
            raise ValueError(f"cannot {action} while the ArUco experiment is enabled")

    def list_task_planes(self) -> dict[str, Any]:
        """List operator-confirmed task-plane revisions and their base-chain status."""

        root = self.store.root / "demonstrations"
        task_planes: list[dict[str, Any]] = []
        if root.is_dir():
            for path in root.glob(
                "rgbd_*/skill_drafts/draft_*_evidence/surface_calibration_*.json"
            ):
                try:
                    evidence = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    not isinstance(evidence, dict)
                    or evidence.get("transform_convention")
                    != "T_camera_task_plane"
                    or evidence.get("operator_confirmed") is not True
                ):
                    continue
                task_planes.append(
                    {
                        **evidence,
                        "artifact_uri": path.relative_to(self.store.root).as_posix(),
                    }
                )
        task_planes.sort(
            key=lambda item: int(item.get("created_at_ns") or 0), reverse=True
        )
        return {
            "task_planes": task_planes,
            "count": len(task_planes),
            "ros_tf_published": False,
        }

    def start_handeye_calibration(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._robot_motion_transition_lock:
            if self.jog_controller.enabled:
                raise ValueError("stop and disable jog before starting hand-eye calibration")
            if self.aruco_experiment_controller.enabled:
                raise ValueError(
                    "stop and disable the ArUco experiment before hand-eye calibration"
                )
            required = (
                "operator_confirmed",
                "board_secured",
                "workspace_cleared",
                "estop_ready",
            )
            if not all(request.get(key) is True for key in required):
                raise ValueError(
                    "all hand-eye calibration safety acknowledgements are required"
                )
            return self.calibration_controller.start(
                operator_id=str(request.get("operator_id") or "operator")
            )

    def abort_handeye_calibration(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.calibration_controller.abort(
            reason=str(request.get("reason") or "operator_request")
        )

    def import_legacy_handeye_npy(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("operator_confirmed") is not True:
            raise ValueError("legacy NPY import requires operator confirmation")
        if request.get("acknowledge_candidate_only") is not True:
            raise ValueError("legacy NPY import remains candidate-only")
        return self.calibration_controller.import_legacy_npy(
            operator_id=str(request.get("operator_id") or "operator")
        )

    def start_camera_preview(self) -> dict[str, Any]:
        return self.camera_controller.start_preview()

    def stop_camera_preview(self) -> dict[str, Any]:
        return self.camera_controller.stop_preview()

    def stream_camera_preview(
        self, kind: Literal["rgb", "depth"]
    ) -> Iterator[bytes]:
        return self.camera_controller.iter_mjpeg(kind)

    def start_camera_recording(self, request: dict[str, Any]) -> dict[str, Any]:
        maximum_duration_s = float(request.get("maximum_duration_s", 30.0))
        maximum_trace_frames = math.ceil(
            maximum_duration_s
            * self.settings.realsense_recording_frames_per_second
        )
        if maximum_trace_frames > MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES:
            raise ValueError(
                "recording would exceed the full-frame fingertip trace limit: "
                f"{maximum_trace_frames} > "
                f"{MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES}; reduce duration or "
                "REALSENSE_RECORDING_FRAMES_PER_SECOND"
            )
        summary = self.camera_controller.start_recording(
            maximum_duration_s=maximum_duration_s
        )
        recording_id = str(summary["recording_id"])
        start_snapshot = self._teaching_flange_snapshot()
        self._recording_flange_starts[recording_id] = start_snapshot
        return {**summary, "teaching_camera_pose_start": start_snapshot}

    def stop_camera_recording(self, recording_id: str) -> dict[str, Any]:
        summary = self.camera_controller.stop_recording(recording_id)
        start_snapshot = self._recording_flange_starts.pop(
            recording_id,
            {
                "available": False,
                "reason": "recording_start_pose_unavailable_after_restart",
            },
        )
        end_snapshot = self._teaching_flange_snapshot()
        diagnostics: dict[str, Any] = {
            "available": False,
            "passed": False,
            "reason": "base_to_flange_chain_unavailable",
        }
        if start_snapshot.get("available") is True and end_snapshot.get("available") is True:
            diagnostics = {
                "available": True,
                **camera_stationarity_diagnostics(
                    start_snapshot["base_to_flange"],
                    end_snapshot["base_to_flange"],
                ),
                "active_tcp_matches": (
                    start_snapshot.get("active_tcp_name")
                    == end_snapshot.get("active_tcp_name")
                ),
            }
            diagnostics["passed"] = bool(
                diagnostics["passed"] and diagnostics["active_tcp_matches"]
            )
        evidence = {
            "schema_version": "1.0",
            "evidence_type": "teaching_camera_stationarity",
            "recording_id": recording_id,
            "start": start_snapshot,
            "end": end_snapshot,
            "diagnostics": diagnostics,
            "created_at_ns": time.time_ns(),
        }
        artifact = self.store.put_json(
            f"demonstrations/{recording_id}/camera_stationarity.json",
            evidence,
        )
        return {
            **summary,
            "camera_stationarity": {
                **evidence,
                "artifact_uri": artifact.uri,
                "artifact_checksum_sha256": artifact.checksum_sha256,
            },
        }

    def _teaching_flange_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(
            self.calibration_controller,
            "capture_base_to_flange_snapshot",
            None,
        )
        if not callable(snapshot):
            return {
                "available": False,
                "reason": "robot_pose_provider_not_configured",
                "captured_at_ns": time.time_ns(),
            }
        try:
            value = snapshot()
            return dict(value) if isinstance(value, dict) else {
                "available": False,
                "reason": "robot_pose_provider_returned_invalid_data",
                "captured_at_ns": time.time_ns(),
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: robot pose snapshot failed",
                "captured_at_ns": time.time_ns(),
            }

    def get_camera_recording(self, recording_id: str) -> dict[str, Any]:
        return self.camera_controller.get_recording(recording_id)

    def list_camera_recordings(self) -> dict[str, Any]:
        return self.camera_controller.list_recordings()

    def get_camera_recording_frame(
        self,
        recording_id: str,
        frame_index: int,
        kind: Literal["rgb", "depth"],
    ) -> bytes:
        return self.camera_controller.get_recording_frame_jpeg(
            recording_id, frame_index, kind
        )

    def get_recording_skill_draft_capabilities(self) -> dict[str, Any]:
        return {
            "openai_mode": self.settings.openai_mode.value,
            "api_key_configured": self.settings.openai_api_key is not None,
            "model": self.settings.openai_reasoning_model,
            "maximum_keyframes": self.settings.openai_max_keyframes,
            "maximum_demonstration_recordings": 8,
            "multiple_demonstration_recordings_supported": True,
            "image_detail": self.settings.openai_image_detail,
            "uploads_rgb_keyframes_only": True,
            "uploads_rgb_and_aligned_depth_pairs": False,
            "uploaded_rgb_frame_count": "operator_selected_1_to_maximum_keyframes",
            "maximum_compact_trace_frames": MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES,
            "uploaded_frame_policy": "evenly_spaced_manifest_rgb_keyframes",
            "request_keyframe_count_is_deprecated_and_ignored": False,
            "depth_stays_local": True,
            "tcp_proxy_mode": "full_recording_local_thumb_index_trace",
            "provider_video_input_supported": False,
            "direct_image_transport": True,
            "fallback_analysis_transport": "rgb_keyframe_input_files",
            "creates_zip_archive_on_fallback": False,
            "returns_frame_complete_tcp_audit": True,
            "llm_receives_metric_finger_distance": False,
            "local_finger_state_authoritative": True,
            "semantic_anchor_local_depth_verified_per_representative_frame": True,
            "returns_semantic_scene_regions": True,
            "automatic_surface_plane_backend": "local_numpy_ransac_raw_depth",
            "npy_validation_blocks_mock_candidate": False,
            "stores_openai_response": False,
            "creates_executable_skill": False,
        }

    def create_recording_skill_draft(self, request: dict[str, Any]) -> dict[str, Any]:
        """Analyze selected RGB frames plus all-frame local fingertip coordinates."""

        recording_id = str(request["recording_id"])
        requested_recording_ids = request.get("recording_ids") or []
        source_recording_ids = list(
            dict.fromkeys([recording_id, *(str(item) for item in requested_recording_ids)])
        )
        if len(source_recording_ids) > 8:
            raise ValueError("skill drafting supports at most 8 demonstration recordings")

        cases: list[dict[str, Any]] = []
        total_frame_count = 0
        for source_recording_id in source_recording_ids:
            case_manifest = self.camera_controller.get_recording_manifest(
                source_recording_id, include_frames=True
            )
            case_frames = case_manifest.get("frames") or []
            if not isinstance(case_frames, list) or not case_frames:
                raise ValueError(
                    f"RGB-D recording {source_recording_id!r} contains no manifest frames"
                )
            if case_manifest.get("depth_aligned_to_color") is not True:
                raise ValueError(
                    f"RGB-D recording {source_recording_id!r} requires aligned depth"
                )
            total_frame_count += len(case_frames)
            cases.append(
                {
                    "recording_id": source_recording_id,
                    "manifest": case_manifest,
                    "frames": case_frames,
                    "tracking": self._track_recording_fingertips(
                        source_recording_id, case_manifest
                    ),
                }
            )
        if total_frame_count > MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES:
            raise ValueError(
                "skill drafting supports at most "
                f"{MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES} combined trace frames"
            )

        manifest = cases[0]["manifest"]
        tracking = cases[0]["tracking"]
        requested_keyframe_count = min(
            int(request.get("keyframe_count") or 1),
            self.settings.openai_max_keyframes,
        )
        if requested_keyframe_count < len(cases):
            raise ValueError(
                "RGB image count must be at least the number of demonstration recordings"
            )
        image_budget = min(requested_keyframe_count, total_frame_count)
        allocations = [1] * len(cases)
        remaining = image_budget - len(cases)
        while remaining:
            changed = False
            for case_index, case in enumerate(cases):
                if allocations[case_index] >= len(case["frames"]):
                    continue
                allocations[case_index] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
            if not changed:
                break

        keyframe_indices: list[int] = []
        rgb_paths: list[Path] = []
        analysis_trace: list[Any] = []
        demonstration_cases: list[dict[str, Any]] = []
        global_frame_mapping: dict[int, tuple[str, int, dict[str, Any]]] = {}
        global_offset = 0
        timestamp_cursor: int | None = None
        for case, allocation in zip(cases, allocations, strict=True):
            source_id = str(case["recording_id"])
            selected = self.camera_controller.select_recording_keyframes(
                source_id, maximum_count=allocation
            )
            local_keyframes = [index for index, _path in selected]
            global_keyframes = [global_offset + index for index in local_keyframes]
            keyframe_indices.extend(global_keyframes)
            rgb_paths.extend(path for _index, path in selected)
            compact_trace = case["tracking"]["compact_fingertip_trace"]
            first_timestamp = int(compact_trace[0]["timestamp_ns"])
            case_timestamp_origin = (
                first_timestamp if timestamp_cursor is None else timestamp_cursor + 1
            )
            for trace_entry in compact_trace:
                analysis_trace.append(
                    {
                        **trace_entry,
                        "frame_index": global_offset + int(trace_entry["frame_index"]),
                        "timestamp_ns": case_timestamp_origin
                        + int(trace_entry["timestamp_ns"])
                        - first_timestamp,
                    }
                )
            timestamp_cursor = int(analysis_trace[-1]["timestamp_ns"])
            for local_index, entry in enumerate(case["frames"]):
                global_frame_mapping[global_offset + local_index] = (
                    source_id,
                    local_index,
                    entry,
                )
            demonstration_cases.append(
                {
                    "recording_id": source_id,
                    "global_frame_offset": global_offset,
                    "frame_count": len(case["frames"]),
                    "local_keyframe_indices": local_keyframes,
                    "global_keyframe_indices": global_keyframes,
                }
            )
            global_offset += len(case["frames"])
        first_frame_index = keyframe_indices[0]
        primitive_catalog = list(get_default_registry().operation_names())
        entity_role_catalog = [
            "tool",
            "target_object",
            "target_surface",
            "fixture",
            "workspace_region",
        ]
        analysis_input = RecordingSkillDraftInput(
            recording_id=recording_id,
            demonstration_cases=demonstration_cases,
            name_hint=str(request.get("name_hint", "recorded_skill")),
            operator_instruction=str(request["operator_instruction"]),
            recording_summary={
                "duration_s": sum(
                    float(case["manifest"].get("duration_s") or 0.0) for case in cases
                ),
                "frame_count": total_frame_count,
                "demonstration_count": len(cases),
                "recording_fps": manifest.get("recording_fps"),
                "depth_aligned_to_color": manifest.get("depth_aligned_to_color"),
                "timestamps_preserved": manifest.get("timestamps_preserved"),
            },
            primitive_catalog=primitive_catalog,
            entity_role_catalog=entity_role_catalog,
            keyframe_indices=keyframe_indices,
            first_frame_index=first_frame_index,
            fingertip_trace=analysis_trace,
            visual_input_policy="rgb_keyframes_plus_local_fingertip_trace",
            limitations=[
                "Selected RGB keyframes and compact thumb/index traces from every named "
                "demonstration case reach OpenAI.",
                "Depth, metric fingertip distance, and local gripper state remain local.",
                "The recording contains no trusted robot-base trajectory unless a separate "
                "stationarity/base-chain artifact passes validation.",
                "Normalized OpenAI regions are hints; local raw depth and intrinsics "
                "own metric geometry.",
                "Local geometry fitting owns MoveL, verified MoveC, and spline selection.",
                "Force, velocity, acceleration, and execution permission remain local-only.",
            ],
        )
        draft_id = f"draft_{uuid.uuid4().hex}"
        analyzer = RecordingSkillDraftAnalyzer(self.settings)
        transport: dict[str, Any] = {
            "mode": "rgb_keyframes_plus_compact_fingertip_trace",
            "first_frame_index": first_frame_index,
            "keyframe_indices": keyframe_indices,
            "image_count": len(rgb_paths),
            "demonstration_count": len(cases),
            "demonstration_cases": demonstration_cases,
            "depth_image_count": 0,
            "trace_frame_count": len(analysis_trace),
            "requested_keyframe_count": requested_keyframe_count,
            "fallback_used": False,
        }
        try:
            draft, metadata = analyzer.analyze_rgb_keyframe_trace(
                analysis_input,
                rgb_paths=rgb_paths,
            )
        except ImageInputRejectedError:
            draft, metadata = analyzer.analyze_rgb_keyframe_trace(
                analysis_input,
                rgb_paths=rgb_paths,
                as_file_fallback=True,
            )
            transport = {
                "mode": "rgb_keyframe_files_plus_compact_fingertip_trace",
                "first_frame_index": first_frame_index,
                "keyframe_indices": keyframe_indices,
                "image_count": len(rgb_paths),
                "demonstration_count": len(cases),
                "demonstration_cases": demonstration_cases,
                "depth_image_count": 0,
                "trace_frame_count": len(analysis_trace),
                "requested_keyframe_count": requested_keyframe_count,
                "fallback_used": True,
                "reason": "direct_image_input_rejected",
            }
        tracking["semantic_conflicts"] = self._finger_semantic_conflicts(
            tracking, draft.model_dump(mode="json")
        )
        tracking["demonstration_cases"] = [
            {
                **case_summary,
                "tracking": {**case["tracking"]},
            }
            for case_summary, case in zip(demonstration_cases, cases, strict=True)
        ]
        tracking["draft_id"] = draft_id
        evidence_root = (
            f"demonstrations/{recording_id}/skill_drafts/{draft_id}_evidence"
        )
        tracking_artifact = self.store.put_json(
            f"{evidence_root}/finger_tracking_{tracking['tracking_id']}.json",
            tracking,
        )
        initial_anchors = self._reconstruct_initial_scene_anchors(
            recording_id=recording_id,
            frame_index=first_frame_index,
            allowed_frame_indices=set(keyframe_indices),
            global_frame_mapping=global_frame_mapping,
            semantic_draft=draft.model_dump(mode="json"),
            manifest=manifest,
        )
        initial_anchors["draft_id"] = draft_id
        anchor_artifact = self.store.put_json(
            f"{evidence_root}/initial_scene_anchors.json", initial_anchors
        )
        artifact_payload = {
            "schema_version": "1.0",
            "draft_id": draft_id,
            "status": "semantic_draft",
            "created_at_ns": time.time_ns(),
            "source_recording_id": recording_id,
            "source_recording_ids": source_recording_ids,
            "keyframe_indices": keyframe_indices,
            "full_recording_frame_count": total_frame_count,
            "openai_mode": self.settings.openai_mode.value,
            "openai_model": self.settings.openai_reasoning_model,
            "openai_trace_id": metadata.trace_id,
            "transport": transport,
            "local_fingertip_tracking": {
                "tracking_id": tracking["tracking_id"],
                "artifact_uri": tracking_artifact.uri,
                "artifact_checksum_sha256": tracking_artifact.checksum_sha256,
                "valid_frame_count": sum(
                    int(case["tracking"]["valid_frame_count"]) for case in cases
                ),
                "transition_count": sum(
                    len(case["tracking"]["state_transitions"]) for case in cases
                ),
            },
            "initial_scene_anchors": {
                "artifact_uri": anchor_artifact.uri,
                "artifact_checksum_sha256": anchor_artifact.checksum_sha256,
                "anchor_count": len(initial_anchors["anchors"]),
            },
            "draft": draft.model_dump(mode="json"),
        }
        artifact = self.store.put_json(
            f"demonstrations/{recording_id}/skill_drafts/{draft_id}.json",
            artifact_payload,
        )
        return {
            **artifact_payload,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
            "openai": {
                "mode": self.settings.openai_mode.value,
                "model": self.settings.openai_reasoning_model,
                "trace_id": metadata.trace_id,
                "response_id": metadata.response_id,
                "input_tokens": metadata.input_tokens,
                "output_tokens": metadata.output_tokens,
                "total_tokens": metadata.total_tokens,
                "attempts": metadata.attempts,
            },
            "executable": False,
            "requires_pose_trajectory": True,
        }

    def _track_recording_fingertips(
        self,
        recording_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one persistent tracker over every chronological manifest frame."""

        frames = manifest.get("frames") or []
        if not isinstance(frames, list) or not frames:
            raise ValueError("RGB-D manifest contains no frames")
        if manifest.get("depth_aligned_to_color") is not True:
            raise ValueError("full-recording fingertip tracking requires aligned depth")
        if not all(isinstance(entry, dict) for entry in frames):
            raise ValueError("RGB-D manifest frame entries must be objects")
        frame_indices = [
            int(entry.get("index", sequence_index))
            for sequence_index, entry in enumerate(frames)
        ]
        if len(set(frame_indices)) != len(frame_indices) or frame_indices != sorted(
            frame_indices
        ):
            raise ValueError("RGB-D manifest frames must have unique chronological indices")
        tracker = MediaPipeHandPoseEstimator(
            minimum_detection_confidence=(
                self.settings.mediapipe_minimum_detection_confidence
            ),
            minimum_tracking_confidence=(
                self.settings.mediapipe_minimum_tracking_confidence
            ),
            close_threshold_m=self.settings.finger_close_threshold_m,
            stable_frames=self.settings.finger_state_stable_frames,
            maximum_timestamp_skew_ns=int(
                self.settings.rgbd_max_timestamp_delta_ms * 1_000_000
            ),
        )
        observations: list[dict[str, Any]] = []
        compact_trace: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        invalid_frames: list[dict[str, Any]] = []
        optional_error: str | None = None
        try:
            for sequence_index, entry in enumerate(frames):
                frame_index = int(entry.get("index", sequence_index))
                result = None
                observation = None
                frame_error: str | None = None
                frame = None
                try:
                    frame = self.camera_controller.load_recording_rgbd_frame(
                        recording_id, frame_index
                    )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    frame_error = f"{type(exc).__name__}: {exc}"
                if frame is not None and optional_error is None:
                    try:
                        result = tracker.track(frame)
                        observation = result.observation
                    except NotConfiguredError as exc:
                        optional_error = str(exc)
                        frame_error = "mediapipe_not_configured"
                elif frame is not None and optional_error is not None:
                    frame_error = "mediapipe_not_configured"

                if observation is None:
                    if frame is not None:
                        timestamp_ns = frame.timestamp_ns
                        frame_number = frame.frame_number
                        reference_frame = frame.reference_frame
                    else:
                        color_timestamp_ns = int(entry.get("color_timestamp_ns") or 0)
                        depth_timestamp_ns = int(entry.get("depth_timestamp_ns") or 0)
                        timestamp_ns = (color_timestamp_ns + depth_timestamp_ns) // 2
                        raw_frame_number = entry.get("frame_number")
                        frame_number = (
                            int(raw_frame_number)
                            if raw_frame_number is not None
                            else frame_index
                        )
                        reference_frame = str(
                            entry.get("reference_frame")
                            or "camera_color_optical_frame"
                        )
                    decision = tracker.state_stabilizer.update(
                        None,
                        frame_number=frame_number,
                        timestamp_ns=timestamp_ns,
                    )
                    observation = FingerObservation(
                        frame_number=frame_number,
                        timestamp_ns=timestamp_ns,
                        reference_frame=reference_frame,
                        status="uncertain",
                        stabilization_progress_frames=0,
                        required_stable_frames=(
                            self.settings.finger_state_stable_frames
                        ),
                        invalid_reason=(
                            frame_error
                            or (
                                "mediapipe_not_configured"
                                if optional_error is not None
                                else "frame_geometry_unavailable"
                            )
                        ),
                        stable_state=decision.stable_state,
                        confidence=0.0,
                    )
                observation_payload = observation.model_dump(mode="json")
                observation_payload["frame_index"] = frame_index
                observation_payload["hand_pose"] = (
                    result.hand_pose.model_dump(mode="json")
                    if result is not None and result.hand_pose is not None
                    else None
                )
                observations.append(observation_payload)
                trace_item = observation.to_compact_landmark_trace()
                trace_item["frame_index"] = frame_index
                compact_trace.append(trace_item)
                if observation.status == "uncertain":
                    invalid_frames.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ns": observation.timestamp_ns,
                            "reason": observation.invalid_reason,
                        }
                    )
                if result is not None and result.transition is not None:
                    transition = result.transition.model_dump(mode="json")
                    transition["frame_index"] = frame_index
                    transitions.append(transition)
        finally:
            tracker.close()
        return {
            "schema_version": "1.0",
            "evidence_type": "mediapipe_rgbd_full_recording_fingertip_tracking",
            "tracking_id": f"finger_{uuid.uuid4().hex}",
            "recording_id": recording_id,
            "processed_frame_count": len(frames),
            "valid_frame_count": sum(
                item.get("status") == "valid" for item in observations
            ),
            "finger_observations": observations,
            "compact_fingertip_trace": compact_trace,
            "state_transitions": transitions,
            "invalid_frames": invalid_frames,
            "optional_dependency_error": optional_error,
            "semantic_conflicts": [],
            "settings": {
                "thumb_tip_landmark_index": 4,
                "index_tip_landmark_index": 8,
                "depth_patch_size_px": 5,
                "close_threshold_m": self.settings.finger_close_threshold_m,
                "closed_comparison": "distance_m <= close_threshold_m",
                "open_comparison": "distance_m > close_threshold_m",
                "stable_frames": self.settings.finger_state_stable_frames,
                "minimum_detection_confidence": (
                    self.settings.mediapipe_minimum_detection_confidence
                ),
                "minimum_tracking_confidence": (
                    self.settings.mediapipe_minimum_tracking_confidence
                ),
                "static_image_mode": False,
                "max_num_hands": 1,
            },
            "metric_state_authority": "local_mediapipe_landmarks_plus_aligned_depth",
            "llm_trace_excludes_depth_distance_and_state": True,
            "created_at_ns": time.time_ns(),
        }

    @staticmethod
    def _finger_semantic_conflicts(
        tracking: dict[str, Any], semantic_draft: dict[str, Any]
    ) -> list[dict[str, Any]]:
        observations = {
            int(item["frame_index"]): item
            for item in tracking.get("finger_observations") or []
            if isinstance(item, dict) and item.get("frame_index") is not None
        }
        tcp = semantic_draft.get("tcp_proxy_observation") or {}
        conflicts: list[dict[str, Any]] = []
        for state in tcp.get("observed_states") or []:
            if not isinstance(state, dict):
                continue
            frame_index = int(state.get("frame_index") or 0)
            local = observations.get(frame_index)
            semantic_state = state.get("gripper_state")
            local_state = local.get("candidate_state") if local else None
            if (
                semantic_state in {"open", "closed"}
                and local_state in {"open", "closed"}
                and semantic_state != local_state
            ):
                conflicts.append(
                    {
                        "frame_index": frame_index,
                        "semantic_state": semantic_state,
                        "local_metric_state": local_state,
                        "resolution": "local_metric_state_wins",
                    }
                )
        return conflicts

    def _reconstruct_initial_scene_anchors(
        self,
        *,
        recording_id: str,
        frame_index: int,
        allowed_frame_indices: set[int] | None = None,
        global_frame_mapping: dict[int, tuple[str, int, dict[str, Any]]] | None = None,
        semantic_draft: dict[str, Any],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Turn keyframe semantic ROIs into local, unconfirmed metric evidence."""

        manifest_frames = manifest.get("frames") or []
        entries_by_index = {
            int(item.get("index", sequence_index)): item
            for sequence_index, item in enumerate(manifest_frames)
            if isinstance(item, dict)
        }
        if global_frame_mapping is None:
            global_frame_mapping = {
                index: (recording_id, index, entry)
                for index, entry in entries_by_index.items()
            }
        first_source_id, first_local_index, frame_entry = global_frame_mapping.get(
            frame_index, (recording_id, frame_index, None)
        )
        if not isinstance(frame_entry, dict):
            raise ValueError("first semantic frame is outside the recording manifest")
        first_frame = self.camera_controller.load_recording_rgbd_frame(
            first_source_id, first_local_index
        )
        permitted = allowed_frame_indices or {frame_index}
        loaded_frames = {frame_index: first_frame}
        scene = semantic_draft.get("scene_observation") or {}
        anchors: list[dict[str, Any]] = []
        warnings: list[str] = []
        for role, field in (
            ("tool", "tool"),
            ("target_object", "target_object"),
            ("target_surface", "work_surface"),
        ):
            observation = scene.get(field)
            if not isinstance(observation, dict) or observation.get("detected") is not True:
                continue
            region = observation.get("region_normalized")
            if not isinstance(region, dict):
                warnings.append(f"{role}: semantic ROI missing")
                continue
            observation_frame_index = int(
                observation.get("representative_frame_index", frame_index)
            )
            if observation_frame_index not in permitted:
                warnings.append(
                    f"{role}: representative frame is not a supplied RGB keyframe"
                )
                continue
            mapped_observation = global_frame_mapping.get(observation_frame_index)
            if mapped_observation is None:
                warnings.append(f"{role}: representative frame is outside the manifest")
                continue
            observation_source_id, observation_local_index, observation_entry = (
                mapped_observation
            )
            observation_frame = loaded_frames.get(observation_frame_index)
            if observation_frame is None:
                observation_frame = self.camera_controller.load_recording_rgbd_frame(
                    observation_source_id, observation_local_index
                )
                loaded_frames[observation_frame_index] = observation_frame
            bounds = (
                float(region["x_min"]),
                float(region["y_min"]),
                float(region["x_max"]),
                float(region["y_max"]),
            )
            try:
                metric = reconstruct_semantic_roi_anchor(observation_frame, bounds)
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(f"{role}: {exc}")
                continue
            anchors.append(
                {
                    "anchor_id": f"initial_{role}",
                    "entity_role": role,
                    "semantic_class": observation.get("class_name")
                    or observation.get("shape"),
                    "semantic_confidence": observation.get("confidence"),
                    "semantic_region_normalized": region,
                    **metric.as_dict(),
                    "frame_index": observation_frame_index,
                    "source_recording_id": observation_source_id,
                    "source_frame_index": observation_local_index,
                    "frame_id": observation_frame.reference_frame,
                    "timestamp_ns": observation_frame.timestamp_ns,
                    "rgb_uri": observation_entry.get("rgb_uri"),
                    "rgb_checksum_sha256": observation_entry.get(
                        "rgb_checksum_sha256"
                    ),
                    "depth_uri": observation_entry.get("depth_uri"),
                    "depth_checksum_sha256": observation_entry.get(
                        "depth_checksum_sha256"
                    ),
                    "operator_confirmed": False,
                    "usable_for_skill_binding": False,
                }
            )
        return {
            "schema_version": "1.0",
            "evidence_type": "keyframe_semantic_anchor_candidates",
            "recording_id": recording_id,
            "frame_index": frame_index,
            "keyframe_indices": sorted(permitted),
            "frame_id": first_frame.reference_frame,
            "timestamp_ns": first_frame.timestamp_ns,
            "rgb_uri": frame_entry.get("rgb_uri"),
            "rgb_checksum_sha256": frame_entry.get("rgb_checksum_sha256"),
            "depth_uri": frame_entry.get("depth_uri"),
            "depth_checksum_sha256": frame_entry.get("depth_checksum_sha256"),
            "intrinsics": {
                "width_px": first_frame.color_intrinsics.width_px,
                "height_px": first_frame.color_intrinsics.height_px,
                "fx_px": first_frame.color_intrinsics.fx_px,
                "fy_px": first_frame.color_intrinsics.fy_px,
                "cx_px": first_frame.color_intrinsics.cx_px,
                "cy_px": first_frame.color_intrinsics.cy_px,
                "distortion_model": first_frame.color_intrinsics.distortion_model,
                "distortion_coefficients": list(
                    first_frame.color_intrinsics.distortion_coefficients
                ),
            },
            "anchors": anchors,
            "warnings": warnings,
            "coordinate_policy": (
                "semantic_roi_hint_then_local_depth; operator_confirmation_required"
            ),
            "created_at_ns": time.time_ns(),
        }

    def list_recording_skill_drafts(self) -> dict[str, Any]:
        """List persisted semantic drafts separately from registered SkillGraph versions."""

        root = self.store.root / "demonstrations"
        drafts: list[dict[str, Any]] = []
        invalid_draft_count = 0
        if root.is_dir():
            for path in root.glob("rgbd_*/skill_drafts/draft_*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    artifact_uri = path.relative_to(self.store.root).as_posix()
                    drafts.append(
                        self._recording_skill_draft_view(
                            payload,
                            artifact_uri=artifact_uri,
                            fallback_created_at_ns=path.stat().st_mtime_ns,
                        )
                    )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    invalid_draft_count += 1
        drafts.sort(key=lambda item: int(item["created_at_ns"]), reverse=True)
        return {"drafts": drafts, "invalid_draft_count": invalid_draft_count}

    def get_recording_skill_draft(self, draft_id: str) -> dict[str, Any]:
        """Return one persisted draft and its fail-closed promotion readiness."""

        path = self._recording_skill_draft_path(draft_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._recording_skill_draft_view(
            payload,
            artifact_uri=path.relative_to(self.store.root).as_posix(),
            fallback_created_at_ns=path.stat().st_mtime_ns,
        )

    def calibrate_recording_draft_surface(
        self, draft_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Create operator-confirmed ``T_camera_surface`` from three RGB-D points."""

        if request.get("operator_confirmed") is not True:
            raise ValueError("surface calibration requires explicit operator confirmation")
        draft_path = self._recording_skill_draft_path(draft_id)
        draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
        draft_payload["artifact_uri"] = draft_path.relative_to(self.store.root).as_posix()
        recording_id = str(draft_payload["source_recording_id"])
        frame_index = int(request["frame_index"])
        frame = self.camera_controller.load_recording_rgbd_frame(
            recording_id, frame_index
        )

        def pixel(name: str) -> PixelPoint:
            value = request[name]
            return PixelPoint(x_px=float(value["x_px"]), y_px=float(value["y_px"]))

        diagnostics: dict[str, Any]
        transform, diagnostics = calibrate_surface_from_three_points(
            frame,
            origin_px=pixel("origin_px"),
            positive_x_px=pixel("positive_x_px"),
            positive_y_px=pixel("positive_y_px"),
        )
        calibration_id = f"cal_{uuid.uuid4().hex}"
        manifest = self.camera_controller.get_recording_manifest(
            recording_id, include_frames=True
        )
        manifest_frames = manifest.get("frames") or []
        frame_evidence = next(
            (
                item
                for sequence_index, item in enumerate(manifest_frames)
                if isinstance(item, dict)
                and int(item.get("index", sequence_index)) == frame_index
            ),
            None,
        )
        if frame_evidence is None:
            raise ValueError("surface calibration frame is outside the recording manifest")
        selected_pixels = {
            name: {
                "x_px": float(request[name]["x_px"]),
                "y_px": float(request[name]["y_px"]),
            }
            for name in ("origin_px", "positive_x_px", "positive_y_px")
        }
        surface_hint = self._latest_draft_evidence(
            draft_payload, "surface_hint_*.json"
        )
        hint_validation: dict[str, Any] = {
            "available": False,
            "passed": False,
            "reason": "no same-frame RANSAC plane hint was available",
        }
        if (
            isinstance(surface_hint, dict)
            and surface_hint.get("frame_index") == frame_index
            and surface_hint.get("source_frame") == frame.reference_frame
            and isinstance(surface_hint.get("camera_to_task_plane_hint"), dict)
        ):
            hint_validation = {
                "available": True,
                **task_plane_normal_hint_diagnostics(
                    transform,
                    RigidTransform.model_validate(
                        surface_hint["camera_to_task_plane_hint"]
                    ),
                ),
                "hint_artifact_uri": surface_hint.get("artifact_uri"),
                "advisory_only": True,
            }
            if hint_validation["passed"] is not True:
                hint_validation["warning"] = (
                    "manual task-plane normal differs from the advisory RANSAC plane"
                )
        diagnostics = {
            **diagnostics,
            "ransac_normal_validation": hint_validation,
        }
        base_chain = self._task_plane_base_chain(recording_id, transform)
        evidence = {
            "schema_version": "1.0",
            "evidence_type": "surface_frame_calibration",
            "calibration_id": calibration_id,
            "draft_id": draft_id,
            "recording_id": recording_id,
            "frame_index": frame_index,
            "source_frame": frame.reference_frame,
            "parent_frame_id": frame.reference_frame,
            "child_frame_id": str(request["surface_anchor_id"]),
            "measurement_timestamp_ns": frame.timestamp_ns,
            "surface_anchor_id": str(request["surface_anchor_id"]),
            "method": "operator_three_point_aligned_depth",
            "transform_convention": "T_camera_task_plane",
            "task_plane_revision": calibration_id,
            "camera_to_task_plane": transform.model_dump(mode="json"),
            "camera_to_surface": transform.model_dump(mode="json"),
            "selected_pixels": selected_pixels,
            "source_artifacts": {
                "rgb_uri": frame_evidence.get("rgb_uri"),
                "rgb_checksum_sha256": frame_evidence.get("rgb_checksum_sha256"),
                "depth_uri": frame_evidence.get("depth_uri"),
                "depth_checksum_sha256": frame_evidence.get("depth_checksum_sha256"),
                "manifest_uri": manifest.get("manifest_uri"),
                "manifest_checksum_sha256": manifest.get(
                    "manifest_checksum_sha256"
                ),
            },
            "color_intrinsics": {
                "width_px": frame.color_intrinsics.width_px,
                "height_px": frame.color_intrinsics.height_px,
                "fx_px": frame.color_intrinsics.fx_px,
                "fy_px": frame.color_intrinsics.fy_px,
                "cx_px": frame.color_intrinsics.cx_px,
                "cy_px": frame.color_intrinsics.cy_px,
                "distortion_model": frame.color_intrinsics.distortion_model,
                "distortion_coefficients": list(
                    frame.color_intrinsics.distortion_coefficients
                ),
            },
            "diagnostics": diagnostics,
            "base_chain": base_chain,
            "operator_confirmed": True,
            "hardware_validated": False,
            "created_at_ns": time.time_ns(),
        }
        artifact = self.store.put_json(
            f"{self._draft_evidence_root(draft_payload)}/"
            f"surface_calibration_{calibration_id}.json",
            evidence,
        )
        return {
            **evidence,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }

    def auto_calibrate_recording_draft_surface(
        self, draft_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Fit a local metric surface plane inside GPT's optional semantic ROI."""

        if request.get("operator_confirmed") is not True:
            raise ValueError("automatic surface calibration requires operator confirmation")
        draft_path = self._recording_skill_draft_path(draft_id)
        draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
        draft_payload["artifact_uri"] = draft_path.relative_to(self.store.root).as_posix()
        recording_id = str(draft_payload["source_recording_id"])
        keyframe_indices = [
            int(index) for index in draft_payload.get("keyframe_indices") or []
        ]
        if not keyframe_indices:
            raise ValueError("semantic draft has no RGB-D keyframes")
        scene = draft_payload.get("draft", {}).get("scene_observation") or {}
        surface = scene.get("work_surface") if isinstance(scene, dict) else None
        surface = surface if isinstance(surface, dict) else {}
        requested_frame_index = request.get("frame_index")
        representative_frame_index = surface.get("representative_frame_index")
        frame_index = (
            int(requested_frame_index)
            if requested_frame_index is not None
            else representative_frame_index
            if isinstance(representative_frame_index, int)
            and representative_frame_index in keyframe_indices
            else keyframe_indices[len(keyframe_indices) // 2]
        )
        if frame_index not in keyframe_indices:
            raise ValueError("automatic surface frame must be one of the analyzed keyframes")
        region_value = surface.get("region_normalized")
        region: tuple[float, float, float, float] | None = None
        if isinstance(region_value, dict):
            region = (
                float(region_value["x_min"]),
                float(region_value["y_min"]),
                float(region_value["x_max"]),
                float(region_value["y_max"]),
            )
        frame = self.camera_controller.load_recording_rgbd_frame(
            recording_id, frame_index
        )
        transform, diagnostics = segment_dominant_depth_plane(
            frame, region_normalized=region
        )
        effective_region = region or (0.05, 0.15, 0.95, 0.95)
        x_min, y_min, x_max, y_max = effective_region
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        axis_fraction = 0.20
        width_scale = frame.color_intrinsics.width_px - 1
        height_scale = frame.color_intrinsics.height_px - 1
        suggested_pixels = {
            "origin_px": {
                "x_px": center_x * width_scale,
                "y_px": center_y * height_scale,
            },
            "positive_x_px": {
                "x_px": min(x_max, center_x + axis_fraction * (x_max - x_min))
                * width_scale,
                "y_px": center_y * height_scale,
            },
            "positive_y_px": {
                "x_px": center_x * width_scale,
                "y_px": max(y_min, center_y - axis_fraction * (y_max - y_min))
                * height_scale,
            },
            "advisory_only": True,
            "operator_must_confirm_or_adjust": True,
        }
        calibration_id = f"cal_{uuid.uuid4().hex}"
        evidence = {
            "schema_version": "1.0",
            "evidence_type": "surface_frame_calibration",
            "calibration_id": calibration_id,
            "draft_id": draft_id,
            "recording_id": recording_id,
            "frame_index": frame_index,
            "source_frame": frame.reference_frame,
            "surface_anchor_id": str(request["surface_anchor_id"]),
            "method": "local_depth_ransac_plane_hint",
            "semantic_surface_hint": surface or None,
            "transform_convention": "T_camera_task_plane_hint",
            "camera_to_task_plane_hint": transform.model_dump(mode="json"),
            "camera_to_surface": transform.model_dump(mode="json"),
            "diagnostics": diagnostics,
            "manual_click_assist": suggested_pixels,
            "operator_confirmed": True,
            "hardware_validated": False,
            "created_at_ns": time.time_ns(),
        }
        artifact = self.store.put_json(
            f"{self._draft_evidence_root(draft_payload)}/"
            f"surface_hint_{calibration_id}.json",
            evidence,
        )
        return {
            **evidence,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }

    def create_recording_draft_tcp_trajectory(
        self, draft_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Create an audited surface-relative TCP path from local RGB-D evidence."""

        if request.get("operator_confirmed") is not True:
            raise ValueError("TCP trajectory requires explicit operator confirmation")
        draft_path = self._recording_skill_draft_path(draft_id)
        draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
        recording_id = str(draft_payload["source_recording_id"])
        calibration = self._latest_draft_evidence(
            draft_payload, "surface_calibration_*.json"
        )
        if calibration is None:
            raise ValueError("create a camera-to-surface calibration first")
        camera_to_surface = RigidTransform.model_validate(
            calibration.get("camera_to_task_plane")
            or calibration["camera_to_surface"]
        )
        source_frame = str(calibration["source_frame"])
        method = str(request["method"])
        samples: list[ManualTCPPathSample] = []
        tracking: dict[str, Any] | None = None
        attempted_count = 0
        extraction_failures: list[str] = []
        if method == "manual_two_fingertip":
            annotations = sorted(
                request["annotations"], key=lambda item: int(item["frame_index"])
            )
            if len({int(item["frame_index"]) for item in annotations}) != len(
                annotations
            ):
                raise ValueError("manual TCP annotations require unique frame indices")
            attempted_count = len(annotations)
            for annotation in annotations:
                frame_index = int(annotation["frame_index"])
                frame = self.camera_controller.load_recording_rgbd_frame(
                    recording_id, frame_index
                )
                if frame.reference_frame != source_frame:
                    raise ValueError("recording frame does not match calibration source frame")
                samples.append(
                    manual_two_finger_sample(
                        frame,
                        frame_index=frame_index,
                        jaw_tip_a_px=PixelPoint(
                            x_px=float(annotation["jaw_tip_a_px"]["x_px"]),
                            y_px=float(annotation["jaw_tip_a_px"]["y_px"]),
                        ),
                        jaw_tip_b_px=PixelPoint(
                            x_px=float(annotation["jaw_tip_b_px"]["x_px"]),
                            y_px=float(annotation["jaw_tip_b_px"]["y_px"]),
                        ),
                        camera_to_surface=camera_to_surface,
                    )
                )
        elif method == "openai_rgbd":
            tcp_observation = draft_payload.get("draft", {}).get(
                "tcp_proxy_observation"
            )
            if not isinstance(tcp_observation, dict) or not isinstance(
                tcp_observation.get("observed_states"), list
            ):
                raise ValueError(
                    "semantic draft predates GPT landmark auditing; analyze the recording again"
                )
            if tcp_observation.get("usable_for_local_depth_path") is not True:
                reason = tcp_observation.get("failure_reason") or (
                    "fewer than four keyframes contain both fingertip landmarks"
                )
                raise ValueError(f"GPT TCP trajectory is unavailable: {reason}")
            states = sorted(
                tcp_observation["observed_states"],
                key=lambda item: int(item["frame_index"]),
            )
            attempted_count = len(states)
            for state in states:
                if state.get("landmarks_detected") is not True:
                    continue
                frame_index = int(state["frame_index"])
                frame = self.camera_controller.load_recording_rgbd_frame(
                    recording_id, frame_index
                )
                if frame.reference_frame != source_frame:
                    raise ValueError("recording frame does not match calibration source frame")
                tip_a = state.get("jaw_tip_a_normalized")
                tip_b = state.get("jaw_tip_b_normalized")
                if not isinstance(tip_a, dict) or not isinstance(tip_b, dict):
                    extraction_failures.append(f"frame {frame_index}: missing normalized tips")
                    continue
                width_scale = frame.color_intrinsics.width_px - 1
                height_scale = frame.color_intrinsics.height_px - 1
                try:
                    sample = manual_two_finger_sample(
                        frame,
                        frame_index=frame_index,
                        jaw_tip_a_px=PixelPoint(
                            x_px=float(tip_a["x"]) * width_scale,
                            y_px=float(tip_a["y"]) * height_scale,
                        ),
                        jaw_tip_b_px=PixelPoint(
                            x_px=float(tip_b["x"]) * width_scale,
                            y_px=float(tip_b["y"]) * height_scale,
                        ),
                        camera_to_surface=camera_to_surface,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    extraction_failures.append(f"frame {frame_index}: {exc}")
                    continue
                samples.append(
                    replace(
                        sample,
                        confidence=min(sample.confidence, float(state["confidence"])),
                    )
                )
            if len(samples) < 2:
                detail = "; ".join(extraction_failures[:4]) or "no valid local depth"
                raise ValueError(
                    "GPT returned a TCP landmark trajectory, but local aligned depth "
                    f"reconstructed only {len(samples)} valid samples: {detail}"
                )
        elif method == "mediapipe_rgbd":
            tracking = self._latest_draft_evidence(
                draft_payload, "finger_tracking_*.json"
            )
            if tracking is None:
                manifest = self.camera_controller.get_recording_manifest(
                    recording_id, include_frames=True
                )
                tracking = self._track_recording_fingertips(recording_id, manifest)
            observations = tracking.get("finger_observations") or []
            attempted_count = len(observations)
            last_orientation_surface_xyzw: tuple[float, float, float, float] | None = None
            for observation in observations:
                if not isinstance(observation, dict) or observation.get("status") != "valid":
                    continue
                frame_index = int(observation["frame_index"])
                if str(observation.get("reference_frame")) != source_frame:
                    extraction_failures.append(
                        f"frame {frame_index}: recording frame differs from task plane frame"
                    )
                    continue
                midpoint = observation.get("midpoint_camera_m")
                if not isinstance(midpoint, dict):
                    extraction_failures.append(
                        f"frame {frame_index}: fingertip midpoint unavailable"
                    )
                    continue
                hand_pose = observation.get("hand_pose")
                pose_payload = (
                    hand_pose.get("pose") if isinstance(hand_pose, dict) else None
                )
                if isinstance(pose_payload, dict):
                    camera_orientation = Quaternion.model_validate(
                        pose_payload["orientation_xyzw"]
                    )
                elif last_orientation_surface_xyzw is None:
                    # Make T_surface_tcp orientation identity for the first
                    # closed/partially occluded frame with no palm orientation.
                    camera_orientation = camera_to_surface.rotation_xyzw
                else:
                    camera_orientation = camera_to_surface.rotation_xyzw
                surface_to_tcp = transform_camera_pose_to_surface(
                    camera_to_surface,
                    RigidTransform(
                        translation_m=Vector3.model_validate(midpoint),
                        rotation_xyzw=camera_orientation,
                    ),
                )
                orientation_surface_xyzw = (
                    surface_to_tcp.rotation_xyzw.as_tuple()
                    if isinstance(pose_payload, dict)
                    or last_orientation_surface_xyzw is None
                    else last_orientation_surface_xyzw
                )
                if isinstance(pose_payload, dict):
                    last_orientation_surface_xyzw = orientation_surface_xyzw
                samples.append(
                    ManualTCPPathSample(
                        frame_index=frame_index,
                        timestamp_ns=int(observation["timestamp_ns"]),
                        position_surface_m=surface_to_tcp.translation_m.as_tuple(),
                        orientation_surface_xyzw=orientation_surface_xyzw,
                        gripper_width_m=float(observation["distance_m"]),
                        confidence=float(observation.get("confidence") or 0.0),
                    )
                )
        else:
            raise ValueError("unsupported TCP trajectory extraction method")
        quality = validate_surface_relative_path(samples)
        coverage_ratio = len(samples) / max(1, attempted_count)
        tracking_evidence = (
            tracking if method == "mediapipe_rgbd" else None
        )
        trajectory_id = f"trajectory_{uuid.uuid4().hex}"
        evidence = {
            "schema_version": "1.0",
            "evidence_type": "surface_relative_tcp_trajectory",
            "trajectory_id": trajectory_id,
            "draft_id": draft_id,
            "recording_id": recording_id,
            "calibration_id": calibration["calibration_id"],
            "surface_anchor_id": calibration["surface_anchor_id"],
            "frame_id": calibration["surface_anchor_id"],
            "method": method,
            "tcp_definition": "midpoint_between_two_fingertips",
            "orientation_definition": (
                "mediapipe_palm_then_last_valid_then_task_plane"
                if method == "mediapipe_rgbd"
                else "jaw_axis_x_surface_normal_z"
            ),
            "semantic_draft_artifact_uri": (
                draft_path.relative_to(self.store.root).as_posix()
                if method == "openai_rgbd"
                else None
            ),
            "extraction_failures": extraction_failures,
            "finger_observations": (
                tracking_evidence.get("finger_observations", [])
                if isinstance(tracking_evidence, dict)
                else []
            ),
            "state_transitions": (
                tracking_evidence.get("state_transitions", [])
                if isinstance(tracking_evidence, dict)
                else []
            ),
            "invalid_frames": (
                tracking_evidence.get("invalid_frames", [])
                if isinstance(tracking_evidence, dict)
                else []
            ),
            "semantic_conflicts": (
                tracking_evidence.get("semantic_conflicts", [])
                if isinstance(tracking_evidence, dict)
                else []
            ),
            "finger_tracking_settings": (
                tracking_evidence.get("settings")
                if isinstance(tracking_evidence, dict)
                else None
            ),
            "samples": [
                {
                    "frame_index": sample.frame_index,
                    "timestamp_ns": sample.timestamp_ns,
                    "position_surface_m": sample.position_surface_m,
                    "orientation_surface_xyzw": sample.orientation_surface_xyzw,
                    "gripper_width_m": sample.gripper_width_m,
                    "confidence": sample.confidence,
                }
                for sample in samples
            ],
            "quality": {**quality, "coverage_ratio": coverage_ratio},
            "operator_confirmed": True,
            "hardware_validated": False,
            "created_at_ns": time.time_ns(),
        }
        artifact = self.store.put_json(
            f"{self._draft_evidence_root(draft_payload)}/"
            f"tcp_trajectory_{trajectory_id}.json",
            evidence,
        )
        return {
            **evidence,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }

    def register_recording_draft_candidate(
        self, draft_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Materialize, compile, and Mock-validate a surface-relative candidate."""

        if request.get("acknowledge_mock_only") is not True:
            raise ValueError("candidate registration requires Mock-only acknowledgement")
        draft_path = self._recording_skill_draft_path(draft_id)
        draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
        draft_payload["artifact_uri"] = draft_path.relative_to(self.store.root).as_posix()
        calibration: dict[str, Any] | None
        trajectory: dict[str, Any] | None
        if self._fixed_workspace_reuse_enabled():
            calibration = self._ensure_fixed_workspace_calibration(draft_payload)
            trajectory = self._ensure_fixed_workspace_trajectory(
                draft_payload, calibration
            )
        else:
            calibration = self._latest_draft_evidence(
                draft_payload, "surface_calibration_*.json"
            )
            trajectory = self._latest_draft_evidence(
                draft_payload, "tcp_trajectory_*.json"
            )
        if calibration is None or trajectory is None:
            raise ValueError("surface calibration and TCP trajectory are required")
        if trajectory.get("calibration_id") != calibration.get("calibration_id"):
            raise ValueError("TCP trajectory was not generated from the latest calibration")
        latest_handeye_transform = self._latest_legacy_handeye_transform()
        handeye_transform = (
            latest_handeye_transform
            if latest_handeye_transform
            and latest_handeye_transform.get("passed") is True
            else None
        )
        tcp_observation = (draft_payload.get("draft") or {}).get(
            "tcp_proxy_observation"
        )
        promotion = PromotionPolicy().evaluate_recording(
            calibration=calibration,
            trajectory=trajectory,
            has_rgbd_evidence=self._recording_has_complete_rgbd_evidence(
                draft_payload
            ),
            has_semantic_schema=isinstance(draft_payload.get("draft"), dict),
            gpt_fingertips_detected=bool(
                isinstance(tcp_observation, dict)
                and tcp_observation.get("detected") is True
            ),
            handeye_verified=handeye_transform is not None,
            semantic_confidence=(
                float((draft_payload.get("draft") or {})["confidence"])
                if isinstance((draft_payload.get("draft") or {}).get("confidence"), (int, float))
                else None
            ),
        )
        if not promotion.eligible:
            raise ValueError(
                "candidate promotion blocked: " + "; ".join(promotion.blockers)
            )
        graph = self._recording_candidate_graph(
            draft_payload,
            calibration=calibration,
            trajectory=trajectory,
            handeye_transform=handeye_transform,
        )
        graph = graph.model_copy(
            update={
                "uncertainty": {
                    **graph.uncertainty,
                    "promotion_policy": promotion.as_dict(),
                    "promotion_warnings": list(promotion.warnings),
                }
            },
            deep=True,
        )
        row = self._persist_graph(
            graph,
            status="candidate",
            validation_status="pending",
            variant=f"rgbd_{graph.skill_id}",
        )
        validation = self.validate_skill(
            graph.skill_id, {"version": graph.version, "mode": "mock"}
        )
        registration_id = f"candidate_{uuid.uuid4().hex}"
        evidence = {
            "schema_version": "1.0",
            "evidence_type": "candidate_registration",
            "registration_id": registration_id,
            "draft_id": draft_id,
            "skill_id": graph.skill_id,
            "version": row.semantic_version,
            "status": "validated" if validation["passed"] else "rejected",
            "mock_validation_passed": bool(validation["passed"]),
            "promotion_policy": promotion.as_dict(),
            "warnings": list(promotion.warnings),
            "handeye_transform_candidate": (
                {
                    key: latest_handeye_transform[key]
                    for key in (
                        "import_id",
                        "passed",
                        "hardware_validated",
                        "runtime_authorized",
                        "artifact_uri",
                        "artifact_checksum_sha256",
                    )
                    if key in latest_handeye_transform
                }
                | {"attached_to_candidate": handeye_transform is not None}
                if latest_handeye_transform
                else None
            ),
            "hardware_validated": False,
            "created_at_ns": time.time_ns(),
        }
        artifact = self.store.put_json(
            f"{self._draft_evidence_root(draft_payload)}/"
            f"candidate_registration_{registration_id}.json",
            evidence,
        )
        return {
            **evidence,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
            "validation": validation,
        }

    def _recording_skill_draft_path(self, draft_id: str) -> Path:
        if not _RECORDING_DRAFT_ID_PATTERN.fullmatch(draft_id):
            raise ValueError("draft_id has an invalid format")
        root = self.store.root / "demonstrations"
        matches = list(root.glob(f"rgbd_*/skill_drafts/{draft_id}.json"))
        if len(matches) != 1:
            raise KeyError(f"unknown recording skill draft {draft_id!r}")
        return matches[0]

    @staticmethod
    def _draft_evidence_root(payload: dict[str, Any]) -> str:
        return (
            f"demonstrations/{payload['source_recording_id']}/skill_drafts/"
            f"{payload['draft_id']}_evidence"
        )

    def _fixed_workspace_reuse_enabled(self) -> bool:
        """Return whether this server is running the acknowledged fixed cell setup.

        The fixed workspace is intentionally a hardware-session capability.  Mock
        tests and ordinary recording review keep the manual teaching tools, while
        the M0609 station reuses its already selected [0, 0, 90, 0, 90, -90]
        workspace for every draft.
        """

        return self.settings.hardware_enabled

    def _fixed_workspace_calibration_payload(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Represent the frozen NPZ/URDF workspace as a draft task-plane TF.

        This is deliberately deterministic: it reads the same reference camera
        plane used to produce the stored base-plane candidate instead of asking
        for three pixels or running ArUco again.
        """

        source = (
            self.settings.repo_root
            / "aruco/results/d435i_plane_scans/scan_20260807_123803/"
            "base_workspace_urdf_candidate.json"
        ).resolve()
        fixed = json.loads(source.read_text(encoding="utf-8"))
        transforms = fixed.get("transforms")
        if not isinstance(transforms, dict):
            raise ValueError("fixed workspace artifact has no transform payload")
        camera_to_plane = rigid_transform_from_matrix(
            transforms.get("reference_T_camera_plane"),
            label="fixed reference T_camera_plane",
        )
        base_to_plane = rigid_transform_from_matrix(
            transforms.get(
                "T_base_plane_from_urdf_camera_and_reference_T_camera_plane"
            ),
            label="fixed reference T_base_plane",
        )
        frame_id = "camera_color_optical_frame"
        anchors = payload.get("initial_scene_anchors") or {}
        if isinstance(anchors, dict):
            artifact_uri = anchors.get("artifact_uri")
            checksum = anchors.get("artifact_checksum_sha256")
            if isinstance(artifact_uri, str) and isinstance(checksum, str):
                try:
                    anchor_evidence = json.loads(
                        self.store.read_bytes(
                            artifact_uri,
                            expected_checksum_sha256=checksum,
                        ).decode("utf-8")
                    )
                    if isinstance(anchor_evidence, dict) and isinstance(
                        anchor_evidence.get("frame_id"), str
                    ):
                        frame_id = str(anchor_evidence["frame_id"])
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    # The recorded frame metadata is advisory here.  The frozen
                    # artifact itself is explicitly for the optical camera frame.
                    pass
        if frame_id != "camera_color_optical_frame":
            raise ValueError(
                "fixed workspace is defined for camera_color_optical_frame, "
                f"not {frame_id!r}"
            )
        calibration_id = f"cal_fixed_workspace_{str(payload['draft_id']).removeprefix('draft_')}"
        return {
            "schema_version": "1.0",
            "evidence_type": "fixed_workspace_task_plane_reuse",
            "calibration_id": calibration_id,
            "draft_id": str(payload["draft_id"]),
            "recording_id": str(payload["source_recording_id"]),
            "source_frame": frame_id,
            "parent_frame_id": frame_id,
            "child_frame_id": "fixed_workspace_surface",
            "surface_anchor_id": "fixed_workspace_surface",
            "method": "fixed_workspace_npz_urdf_reuse",
            "transform_convention": "T_camera_task_plane",
            "task_plane_revision": calibration_id,
            "camera_to_task_plane": camera_to_plane.model_dump(mode="json"),
            "camera_to_surface": camera_to_plane.model_dump(mode="json"),
            "base_chain": {
                "available": True,
                "verified": False,
                "operator_fixed_workspace_reuse": True,
                "parent_frame_id": "base_link",
                "child_frame_id": "fixed_workspace_surface",
                "transform_convention": "T_base_task_plane",
                "base_to_task_plane": base_to_plane.model_dump(mode="json"),
                "source_artifact": source.as_posix(),
            },
            "fixed_workspace_reuse": True,
            "operator_confirmed": True,
            "hardware_validated": False,
            "created_at_ns": time.time_ns(),
        }

    def _ensure_fixed_workspace_calibration(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist the fixed station's task plane once per semantic draft."""

        existing = self._latest_draft_evidence(payload, "surface_calibration_*.json")
        if existing is not None and existing.get("fixed_workspace_reuse") is True:
            return existing
        evidence = self._fixed_workspace_calibration_payload(payload)
        artifact = self.store.put_json(
            f"{self._draft_evidence_root(payload)}/"
            f"surface_calibration_{evidence['calibration_id']}.json",
            evidence,
        )
        return {
            **evidence,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }

    def _fixed_workspace_semantic_trajectory(
        self,
        payload: dict[str, Any],
        calibration: dict[str, Any],
    ) -> dict[str, Any]:
        """Project the recorded semantic fingertip trace onto the fixed plane.

        This fallback keeps older recordings usable even when their raw RGB-D
        frame files have been archived.  It only uses the stored image intrinsics,
        the frozen camera-to-plane transform, and the already persisted trace.
        """

        anchor_metadata = payload.get("initial_scene_anchors") or {}
        if not isinstance(anchor_metadata, dict):
            raise ValueError("fixed workspace projection needs recorded intrinsics")
        anchor_uri = anchor_metadata.get("artifact_uri")
        anchor_checksum = anchor_metadata.get("artifact_checksum_sha256")
        if not isinstance(anchor_uri, str) or not isinstance(anchor_checksum, str):
            raise ValueError("fixed workspace projection needs recorded intrinsics")
        anchor_evidence = json.loads(
            self.store.read_bytes(
                anchor_uri, expected_checksum_sha256=anchor_checksum
            ).decode("utf-8")
        )
        intrinsics = (
            anchor_evidence.get("intrinsics")
            if isinstance(anchor_evidence, dict)
            else None
        )
        if not isinstance(intrinsics, dict):
            raise ValueError("fixed workspace projection needs recorded intrinsics")
        try:
            width_px = int(intrinsics["width_px"])
            height_px = int(intrinsics["height_px"])
            fx_px = float(intrinsics["fx_px"])
            fy_px = float(intrinsics["fy_px"])
            cx_px = float(intrinsics["cx_px"])
            cy_px = float(intrinsics["cy_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("fixed workspace projection has invalid intrinsics") from exc
        if min(width_px, height_px, fx_px, fy_px) <= 0:
            raise ValueError("fixed workspace projection has invalid intrinsics")
        semantic = payload.get("draft") or {}
        tcp = semantic.get("tcp_proxy_observation") if isinstance(semantic, dict) else None
        states = tcp.get("observed_states") if isinstance(tcp, dict) else None
        if not isinstance(states, list):
            raise ValueError("semantic draft has no fingertip trace to project")
        camera_to_surface = RigidTransform.model_validate(
            calibration["camera_to_task_plane"]
        )
        plane_normal_camera = rotate_vector(
            camera_to_surface.rotation_xyzw, Vector3(x=0.0, y=0.0, z=1.0)
        )
        plane_point_camera = camera_to_surface.translation_m
        denominator_offset = (
            plane_normal_camera.x * plane_point_camera.x
            + plane_normal_camera.y * plane_point_camera.y
            + plane_normal_camera.z * plane_point_camera.z
        )
        samples: list[ManualTCPPathSample] = []
        for sequence_index, state in enumerate(states):
            if not isinstance(state, dict) or state.get("landmarks_detected") is not True:
                continue
            midpoint = state.get("midpoint_normalized")
            if not isinstance(midpoint, dict):
                continue
            try:
                pixel_x = float(midpoint["x"]) * (width_px - 1)
                pixel_y = float(midpoint["y"]) * (height_px - 1)
            except (KeyError, TypeError, ValueError):
                continue
            ray_x = (pixel_x - cx_px) / fx_px
            ray_y = (pixel_y - cy_px) / fy_px
            denominator = (
                plane_normal_camera.x * ray_x
                + plane_normal_camera.y * ray_y
                + plane_normal_camera.z
            )
            if abs(denominator) <= 1e-9:
                continue
            scale = denominator_offset / denominator
            if not math.isfinite(scale) or scale <= 0.0:
                continue
            surface_pose = transform_camera_pose_to_surface(
                camera_to_surface,
                RigidTransform(
                    translation_m=Vector3(
                        x=scale * ray_x, y=scale * ray_y, z=scale
                    ),
                    rotation_xyzw=camera_to_surface.rotation_xyzw,
                ),
            )
            samples.append(
                ManualTCPPathSample(
                    frame_index=int(state.get("frame_index", sequence_index)),
                    timestamp_ns=int(payload.get("created_at_ns") or 0) + sequence_index,
                    position_surface_m=surface_pose.translation_m.as_tuple(),
                    orientation_surface_xyzw=surface_pose.rotation_xyzw.as_tuple(),
                    gripper_width_m=0.03,
                    confidence=float(state.get("confidence") or 0.5),
                )
            )
        quality = validate_surface_relative_path(samples)
        trajectory_id = f"trajectory_{uuid.uuid4().hex}"
        evidence = {
            "schema_version": "1.0",
            "evidence_type": "fixed_workspace_projected_tcp_trajectory",
            "trajectory_id": trajectory_id,
            "draft_id": str(payload["draft_id"]),
            "recording_id": str(payload["source_recording_id"]),
            "calibration_id": calibration["calibration_id"],
            "surface_anchor_id": calibration["surface_anchor_id"],
            "frame_id": calibration["surface_anchor_id"],
            "method": "fixed_workspace_semantic_trace_projection",
            "tcp_definition": "semantic_fingertip_midpoint_projected_to_fixed_plane",
            "orientation_definition": "fixed_task_plane",
            "samples": [
                {
                    "frame_index": sample.frame_index,
                    "timestamp_ns": sample.timestamp_ns,
                    "position_surface_m": sample.position_surface_m,
                    "orientation_surface_xyzw": sample.orientation_surface_xyzw,
                    "gripper_width_m": sample.gripper_width_m,
                    "confidence": sample.confidence,
                }
                for sample in samples
            ],
            "quality": {
                **quality,
                "coverage_ratio": len(samples) / max(1, len(states)),
            },
            "state_transitions": [],
            "finger_observations": [],
            "invalid_frames": [],
            "semantic_conflicts": [],
            "fixed_workspace_reuse": True,
            "operator_confirmed": True,
            "hardware_validated": False,
            "created_at_ns": time.time_ns(),
        }
        artifact = self.store.put_json(
            f"{self._draft_evidence_root(payload)}/"
            f"tcp_trajectory_{trajectory_id}.json",
            evidence,
        )
        return {
            **evidence,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }

    def _ensure_fixed_workspace_trajectory(
        self,
        payload: dict[str, Any],
        calibration: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the fixed-workspace path automatically when Candidate is clicked."""

        existing = self._latest_draft_evidence(payload, "tcp_trajectory_*.json")
        if existing is not None and existing.get("calibration_id") == calibration.get(
            "calibration_id"
        ):
            return existing
        try:
            return self.create_recording_draft_tcp_trajectory(
                str(payload["draft_id"]),
                {"method": "mediapipe_rgbd", "operator_confirmed": True},
            )
        except (KeyError, OSError, TypeError, ValueError):
            try:
                return self.create_recording_draft_tcp_trajectory(
                    str(payload["draft_id"]),
                    {"method": "openai_rgbd", "operator_confirmed": True},
                )
            except (KeyError, OSError, TypeError, ValueError):
                return self._fixed_workspace_semantic_trajectory(payload, calibration)

    @staticmethod
    def _recording_has_complete_rgbd_evidence(payload: dict[str, Any]) -> bool:
        """Return the single readiness/registration definition of RGB-D coverage."""

        transport = payload.get("transport") or {}
        full_frame_count = int(payload.get("full_recording_frame_count") or 0)
        trace_frame_count = (
            int(transport.get("trace_frame_count") or 0)
            if isinstance(transport, dict)
            else 0
        )
        return bool(
            full_frame_count > 0
            and trace_frame_count == full_frame_count
            and isinstance(payload.get("local_fingertip_tracking"), dict)
        )

    def _latest_draft_evidence(
        self, payload: dict[str, Any], pattern: str
    ) -> dict[str, Any] | None:
        root = self.store.path_for(self._draft_evidence_root(payload))
        if not root.is_dir():
            return None
        candidates = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime_ns)
        if not candidates:
            return None
        raw_evidence = json.loads(candidates[-1].read_text(encoding="utf-8"))
        if not isinstance(raw_evidence, dict):
            raise ValueError("draft evidence must be a JSON object")
        evidence: dict[str, Any] = raw_evidence
        if evidence.get("draft_id") != payload.get("draft_id"):
            raise ValueError("draft evidence does not match its semantic draft")
        evidence["artifact_uri"] = candidates[-1].relative_to(self.store.root).as_posix()
        return evidence

    def _latest_legacy_handeye_transform(self) -> dict[str, Any] | None:
        root = self.store.root / "calibrations"
        if not root.is_dir():
            return None
        candidates = sorted(
            root.glob("legacy_import_*/result.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            return None
        path = candidates[-1]
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("legacy hand-eye evidence must be a JSON object")
        return {
            **raw,
            "artifact_uri": path.relative_to(self.store.root).as_posix(),
        }

    def _initial_scene_anchor_inputs(
        self,
        draft_payload: dict[str, Any],
        calibration: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Express first-frame local semantic anchors in the final task plane.

        The LLM contributes only normalized ROIs.  The source artifact contains
        locally deprojected camera-frame points and is checksum verified here;
        these advisory inputs never become runtime bindings without a separate
        operator confirmation workflow.
        """

        metadata = draft_payload.get("initial_scene_anchors")
        if not isinstance(metadata, dict):
            return []
        artifact_uri = metadata.get("artifact_uri")
        checksum = metadata.get("artifact_checksum_sha256")
        if not isinstance(artifact_uri, str) or not isinstance(checksum, str):
            return []
        raw = json.loads(
            self.store.read_bytes(
                artifact_uri,
                expected_checksum_sha256=checksum,
            ).decode("utf-8")
        )
        if not isinstance(raw, dict) or raw.get("draft_id") != draft_payload.get(
            "draft_id"
        ):
            raise ValueError("initial scene-anchor evidence does not match its draft")
        if raw.get("recording_id") != draft_payload.get("source_recording_id"):
            raise ValueError("initial scene-anchor evidence recording lineage mismatch")
        if raw.get("frame_id") != calibration.get("source_frame"):
            raise ValueError("initial scene-anchor evidence frame lineage mismatch")
        keyframe_indices = draft_payload.get("keyframe_indices") or []
        if keyframe_indices and raw.get("frame_index") != keyframe_indices[0]:
            raise ValueError("initial scene-anchor evidence first-frame lineage mismatch")
        camera_to_task_plane = RigidTransform.model_validate(
            calibration.get("camera_to_task_plane")
            or calibration.get("camera_to_surface")
        )
        inputs: list[dict[str, Any]] = []
        for anchor in raw.get("anchors") or []:
            if not isinstance(anchor, dict):
                continue
            position = anchor.get("position_camera_m")
            if not isinstance(position, list) or len(position) != 3:
                continue
            task_plane_pose = transform_camera_pose_to_surface(
                camera_to_task_plane,
                RigidTransform(
                    translation_m=Vector3(
                        x=float(position[0]),
                        y=float(position[1]),
                        z=float(position[2]),
                    ),
                    rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            inputs.append(
                {
                    "anchor_id": anchor.get("anchor_id"),
                    "entity_role": anchor.get("entity_role"),
                    "semantic_class": anchor.get("semantic_class"),
                    "task_plane_anchor_id": calibration["surface_anchor_id"],
                    "position_task_plane_m": (
                        task_plane_pose.translation_m.model_dump(mode="json")
                    ),
                    "semantic_confidence": anchor.get("semantic_confidence"),
                    "metric_depth_confidence": anchor.get("valid_depth_fraction"),
                    "source_artifact_uri": artifact_uri,
                    "source_artifact_checksum_sha256": checksum,
                    "operator_confirmed": False,
                    "usable_for_runtime_binding": False,
                }
            )
        return inputs

    def _recording_stationarity_evidence(
        self, recording_id: str
    ) -> dict[str, Any] | None:
        path = self.store.path_for(
            f"demonstrations/{recording_id}/camera_stationarity.json"
        )
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    def _task_plane_base_chain(
        self,
        recording_id: str,
        camera_to_task_plane: RigidTransform,
    ) -> dict[str, Any]:
        """Compose a base task plane only from verified, stationary evidence."""

        stationarity = self._recording_stationarity_evidence(recording_id)
        handeye = self._latest_legacy_handeye_transform()
        if not stationarity or (
            stationarity.get("diagnostics") or {}
        ).get("passed") is not True:
            return {
                "available": False,
                "verified": False,
                "hardware_compatible": False,
                "reason": "camera stationarity evidence is unavailable or failed",
            }
        if not handeye or handeye.get("passed") is not True:
            return {
                "available": False,
                "verified": False,
                "hardware_compatible": False,
                "reason": "validated T_flange_camera evidence is unavailable",
            }
        start = stationarity.get("start") or {}
        if start.get("active_tcp_name") != handeye.get("active_tcp_name"):
            return {
                "available": False,
                "verified": False,
                "hardware_compatible": False,
                "reason": "teaching active TCP does not match hand-eye calibration",
            }
        try:
            base_to_flange = rigid_transform_from_matrix(
                start["base_to_flange"], label="T_base_flange"
            )
            flange_to_camera = rigid_transform_from_matrix(
                handeye["flange_to_camera"], label="T_flange_camera"
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "available": False,
                "verified": False,
                "hardware_compatible": False,
                "reason": f"invalid base/camera chain evidence: {exc}",
            }
        base_to_task_plane = compose_base_task_plane(
            base_to_flange=base_to_flange,
            flange_to_camera=flange_to_camera,
            camera_to_task_plane=camera_to_task_plane,
        )
        return {
            "available": True,
            "verified": True,
            "hardware_compatible": False,
            "parent_frame_id": "base",
            "child_frame_id": "task_plane",
            "transform_convention": "T_base_task_plane",
            "base_to_task_plane": base_to_task_plane.model_dump(mode="json"),
            "handeye_calibration_revision": handeye.get("import_id"),
            "camera_stationarity_artifact_uri": (
                f"demonstrations/{recording_id}/camera_stationarity.json"
            ),
            "active_tcp_name": start.get("active_tcp_name"),
            "hardware_validation_pending": True,
        }

    def _recording_candidate_graph(
        self,
        draft_payload: dict[str, Any],
        *,
        calibration: dict[str, Any],
        trajectory: dict[str, Any],
        handeye_transform: dict[str, Any] | None,
    ) -> SkillGraph:
        """Materialize only locally measured, surface-relative geometry."""

        semantic = draft_payload["draft"]
        initial_scene_anchor_inputs = self._initial_scene_anchor_inputs(
            draft_payload, calibration
        )
        skill_id = str(semantic["suggested_skill_id"])
        primitive_operations = {
            str(item.get("operation"))
            for item in semantic.get("primitive_sequence") or []
            if isinstance(item, dict)
        }
        requested_contact = any(
            operation.startswith("contact.") for operation in primitive_operations
        )
        task_text = " ".join(
            (
                str(semantic.get("task_description") or ""),
                str(semantic.get("observed_task_summary") or ""),
            )
        ).casefold()
        contact_tool_class = (
            "wiper"
            if any(token in task_text for token in ("닦", "걸레", "wipe"))
            else "polisher"
            if any(token in task_text for token in ("연마", "polish"))
            else None
        )
        # Contact/force materialization is allowed only when the task maps to a
        # locally approved force-profile tool class. Unknown tools keep only the
        # measured motion route and retain the deferred contact uncertainty.
        has_contact = requested_contact and contact_tool_class is not None
        samples = trajectory.get("samples") or []
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError("TCP trajectory has insufficient materialization samples")
        samples = sorted(
            samples,
            key=lambda item: (
                int(item["frame_index"]),
                int(item["timestamp_ns"]),
            ),
        )
        sample_index_by_frame: dict[int, int] = {}
        for sample_index, sample in enumerate(samples):
            frame_index = int(sample["frame_index"])
            if frame_index in sample_index_by_frame:
                raise ValueError("TCP trajectory contains duplicate frame indices")
            sample_index_by_frame[frame_index] = sample_index

        def relative_pose(sample: dict[str, Any]) -> dict[str, Any]:
            return {
                "anchor_id": "$surface",
                "anchor_type": "surface",
                "position_m": {
                    "x": float(sample["position_surface_m"][0]),
                    "y": float(sample["position_surface_m"][1]),
                    "z": float(sample["position_surface_m"][2]),
                },
                "orientation_xyzw": {
                    "x": float(sample["orientation_surface_xyzw"][0]),
                    "y": float(sample["orientation_surface_xyzw"][1]),
                    "z": float(sample["orientation_surface_xyzw"][2]),
                    "w": float(sample["orientation_surface_xyzw"][3]),
                },
            }

        path = [relative_pose(sample) for sample in samples]
        node_specs: list[tuple[str, str, dict[str, Any]]] = []
        for chunk_index, path_chunk in enumerate(
            self._bounded_path_chunks(path, 256, overlap=True)
        ):
            node_specs.append(
                (
                    f"validate_path_{chunk_index:03d}",
                    "workspace.validate_path",
                    {"path": path_chunk},
                )
            )

        transitions = sorted(
            (
                item
                for item in trajectory.get("state_transitions") or []
                if isinstance(item, dict) and item.get("state") in {"open", "closed"}
            ),
            key=lambda item: (int(item["frame_index"]), int(item["timestamp_ns"])),
        )
        gripper_events: list[tuple[int, dict[str, Any]]] = []
        previous_state: str | None = None
        for transition in transitions:
            state = str(transition["state"])
            if state == previous_state:
                continue
            previous_state = state
            transition_frame_index = int(transition["frame_index"])
            boundary = sample_index_by_frame.get(transition_frame_index)
            if boundary is None:
                raise ValueError(
                    "gripper state transition has no metric pose at frame "
                    f"{transition_frame_index}"
                )
            if int(samples[boundary]["timestamp_ns"]) != int(
                transition["timestamp_ns"]
            ):
                raise ValueError(
                    "gripper state transition timestamp does not match its metric pose"
                )
            gripper_events.append((boundary, transition))

        simplification_provenance: list[dict[str, Any]] = []

        def append_motion_segment(
            segment_samples: list[dict[str, Any]], segment_index: int
        ) -> None:
            if len(segment_samples) < 2:
                return
            segment_positions = [
                tuple(float(value) for value in sample["position_surface_m"])
                for sample in segment_samples
            ]
            path_length_m = sum(
                math.dist(start, end)
                for start, end in zip(
                    segment_positions, segment_positions[1:], strict=False
                )
            )
            if path_length_m <= 1.0e-12:
                simplification_provenance.append(
                    {
                        "segment_index": segment_index,
                        "start_frame_index": int(segment_samples[0]["frame_index"]),
                        "end_frame_index": int(segment_samples[-1]["frame_index"]),
                        "start_timestamp_ns": int(segment_samples[0]["timestamp_ns"]),
                        "end_timestamp_ns": int(segment_samples[-1]["timestamp_ns"]),
                        "original_sample_count": len(segment_samples),
                        "simplified_sample_count": 0,
                        "maximum_error_m": 0.0,
                        "tolerance_m": DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M,
                        "chosen_primitive_id": "none",
                        "emitted_operation": None,
                        "skipped_reason": "stationary_segment",
                    }
                )
                return
            simplified = simplify_anchor_relative_path(segment_positions)
            selected_samples = [
                segment_samples[index]
                for index in simplified.retained_sample_indices
            ]
            provenance: dict[str, Any] = {
                "segment_index": segment_index,
                "start_frame_index": int(segment_samples[0]["frame_index"]),
                "end_frame_index": int(segment_samples[-1]["frame_index"]),
                "start_timestamp_ns": int(segment_samples[0]["timestamp_ns"]),
                "end_timestamp_ns": int(segment_samples[-1]["timestamp_ns"]),
                **simplified.provenance.as_dict(),
            }
            verified_arc = None
            verified_periodic = None
            if (
                simplified.provenance.chosen_primitive_id
                == "motion.move_spline"
                and len(segment_samples) >= 3
            ):
                local_pose_samples = tuple(
                    PoseSample(
                        timestamp_ns=int(sample["timestamp_ns"]),
                        position_m=(
                            float(sample["position_surface_m"][0]),
                            float(sample["position_surface_m"][1]),
                            float(sample["position_surface_m"][2]),
                        ),
                        orientation_xyzw=(
                            float(sample["orientation_surface_xyzw"][0]),
                            float(sample["orientation_surface_xyzw"][1]),
                            float(sample["orientation_surface_xyzw"][2]),
                            float(sample["orientation_surface_xyzw"][3]),
                        ),
                        frame_id=str(calibration["surface_anchor_id"]),
                        source="mediapipe_rgbd_task_plane",
                        confidence=float(sample.get("confidence") or 0.0),
                    )
                    for sample in segment_samples
                )
                try:
                    recommendation = recommend_primitive(local_pose_samples)
                except (PrimitiveFittingError, ValueError):
                    recommendation = None
                if (
                    recommendation is not None
                    and recommendation.recommended_primitive_id
                    == "motion.move_periodic"
                ):
                    try:
                        verified_periodic = fit_periodic_primitive_geometry(
                            local_pose_samples,
                            recommendation.selected_fit,
                            maximum_error_m=(
                                DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M
                            ),
                        )
                    except PrimitiveFittingError as exc:
                        provenance["periodic_fallback_reason"] = str(exc)
                    else:
                        provenance.update(
                            {
                                "chosen_primitive_id": "motion.move_periodic",
                                "simplified_sample_count": 1,
                                "maximum_error_m": (
                                    verified_periodic.residuals.maximum_m
                                ),
                                "periodic_fit_rms_m": (
                                    verified_periodic.residuals.root_mean_square_m
                                ),
                                "periodic_fit_maximum_error_m": (
                                    verified_periodic.residuals.maximum_m
                                ),
                                "periodic_observed_period_s": (
                                    verified_periodic.observed_period_s
                                ),
                                "periodic_observed_cycle_count": (
                                    verified_periodic.observed_cycle_count
                                ),
                                "periodic_repetitions": (
                                    verified_periodic.repetitions
                                ),
                                "periodic_amplitude_vector_m": (
                                    verified_periodic.amplitude_vector_m
                                ),
                                "periodic_timing_policy": (
                                    "execution speed remains profile-owned"
                                ),
                            }
                        )
                elif (
                    recommendation is not None
                    and recommendation.recommended_primitive_id == "motion.move_c"
                ):
                    arc_fit = (
                        recommendation.selected_fit
                    )
                    if (
                        arc_fit.residuals.root_mean_square_m
                        <= DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M
                        and arc_fit.residuals.maximum_m
                        <= DEFAULT_PATH_SIMPLIFICATION_TOLERANCE_M
                        and arc_fit.confidence >= 0.5
                        and arc_fit.via_m is not None
                        and arc_fit.radius_m is not None
                    ):
                        verified_arc = arc_fit
                        provenance.update(
                            {
                                "chosen_primitive_id": "motion.move_c",
                                "simplified_sample_count": 3,
                                "maximum_error_m": arc_fit.residuals.maximum_m,
                                "arc_fit_rms_m": arc_fit.residuals.root_mean_square_m,
                                "arc_fit_maximum_error_m": arc_fit.residuals.maximum_m,
                                "arc_fit_confidence": arc_fit.confidence,
                                "arc_radius_m": arc_fit.radius_m,
                            }
                        )
            simplified_path = [relative_pose(sample) for sample in selected_samples]
            periodic_node_spec: tuple[str, str, dict[str, Any]] | None = None
            if verified_periodic is not None:
                center_sample = dict(segment_samples[len(segment_samples) // 2])
                center_sample["position_surface_m"] = verified_periodic.center_m
                amplitude_x, amplitude_y, amplitude_z = (
                    verified_periodic.amplitude_vector_m
                )
                periodic_node_spec = (
                    f"move_periodic_{segment_index:03d}",
                    "motion.move_periodic",
                    {
                        "center": relative_pose(center_sample),
                        "amplitude_m": {
                            "x": amplitude_x,
                            "y": amplitude_y,
                            "z": amplitude_z,
                        },
                        "repetitions": verified_periodic.repetitions,
                        "motion_profile_id": "periodic_safe",
                    },
                )
                provenance["emitted_operation"] = "motion.move_periodic"
            arc_node_spec: tuple[str, str, dict[str, Any]] | None = None
            if verified_arc is not None:
                via_sample = dict(segment_samples[len(segment_samples) // 2])
                target_sample = dict(segment_samples[-1])
                via_sample["position_surface_m"] = verified_arc.via_m
                target_sample["position_surface_m"] = verified_arc.end_m
                arc_node_spec = (
                    f"move_c_{segment_index:03d}",
                    "motion.move_c",
                    {
                        "via": relative_pose(via_sample),
                        "target": relative_pose(target_sample),
                        "motion_profile_id": "circular_normal",
                    },
                )
                provenance["emitted_operation"] = "motion.move_c"
            if has_contact:
                node_specs.extend(
                    [
                        (
                            f"contact_search_{segment_index:03d}",
                            "contact.search_surface",
                            {
                                "surface": "$surface",
                                "force_profile_id": "contact_search_soft",
                            },
                        ),
                        (
                            f"force_enable_{segment_index:03d}",
                            "contact.enable_force",
                            {
                                "surface": "$surface",
                                "force_profile_id": "contact_search_soft",
                            },
                        ),
                    ]
                )
                if periodic_node_spec is not None:
                    node_specs.append(periodic_node_spec)
                elif arc_node_spec is not None:
                    node_specs.append(arc_node_spec)
                else:
                    provenance["emitted_operation"] = "contact.follow_path"
                    for chunk_index, path_chunk in enumerate(
                        self._bounded_path_chunks(
                            simplified_path, 256, overlap=True
                        )
                    ):
                        node_specs.append(
                            (
                                f"follow_path_{segment_index:03d}_{chunk_index:03d}",
                                "contact.follow_path",
                                {
                                    "path": path_chunk,
                                    "motion_profile_id": "linear_slow",
                                },
                            )
                        )
                node_specs.append(
                    (
                        f"force_disable_{segment_index:03d}",
                        "contact.disable_force",
                        {},
                    )
                )
            elif periodic_node_spec is not None:
                node_specs.append(periodic_node_spec)
            elif arc_node_spec is not None:
                node_specs.append(arc_node_spec)
            elif simplified.provenance.chosen_primitive_id == "motion.move_l":
                provenance["emitted_operation"] = "motion.move_l"
                node_specs.append(
                    (
                        f"move_l_{segment_index:03d}",
                        "motion.move_l",
                        {
                            "target": simplified_path[-1],
                            "motion_profile_id": "linear_slow",
                        },
                    )
                )
            else:
                provenance["emitted_operation"] = "motion.move_spline"
                for chunk_index, path_chunk in enumerate(
                    self._bounded_path_chunks(simplified_path, 128, overlap=True)
                ):
                    node_specs.append(
                        (
                            f"move_spline_{segment_index:03d}_{chunk_index:03d}",
                            "motion.move_spline",
                            {
                                "waypoints": path_chunk,
                                "motion_profile_id": "linear_slow",
                            },
                        )
                    )
            simplification_provenance.append(provenance)

        cursor = 0
        segment_index = 0
        for event_index, (boundary, transition) in enumerate(gripper_events):
            if boundary > cursor:
                append_motion_segment(samples[cursor : boundary + 1], segment_index)
                segment_index += 1
            state = str(transition["state"])
            node_specs.append(
                (
                    f"gripper_{state}_{event_index:03d}",
                    f"gripper.{state if state == 'open' else 'close'}",
                    {"tool": "$tool"},
                )
            )
            cursor = boundary
        if cursor < len(samples) - 1:
            append_motion_segment(samples[cursor:], segment_index)
        elif not gripper_events:
            append_motion_segment(samples, segment_index)
        has_gripper = bool(gripper_events)
        motion_phase_policy = self._recording_motion_phase_policy(node_specs)
        close_frame_index = next(
            (
                int(item["frame_index"])
                for _boundary, item in gripper_events
                if item["state"] == "closed"
            ),
            None,
        )
        final_open_frame_index = next(
            (
                int(item["frame_index"])
                for _boundary, item in reversed(gripper_events)
                if item["state"] == "open"
                and (
                    close_frame_index is None
                    or int(item["frame_index"]) > close_frame_index
                )
            ),
            None,
        )
        for item in simplification_provenance:
            start_frame_index = int(item["start_frame_index"])
            end_frame_index = int(item["end_frame_index"])
            if close_frame_index is None:
                phase = "action"
            elif end_frame_index <= close_frame_index:
                phase = "grip"
            elif (
                final_open_frame_index is not None
                and start_frame_index >= final_open_frame_index
            ):
                phase = "end_motion"
            else:
                phase = "action"
            item["grip_action_end_phase"] = phase
        nodes = [
            SkillNode(
                node_id=node_id,
                operation=operation,
                arguments=arguments,
                on_success=(
                    node_specs[index + 1][0]
                    if index + 1 < len(node_specs)
                    else None
                ),
            )
            for index, (node_id, operation, arguments) in enumerate(node_specs)
        ]
        bindings = {
            "$surface": BindingSpec(
                variable="$surface",
                entity_kind=EntityKind.SURFACE,
                instance_id=str(calibration["surface_anchor_id"]),
                role="contact_target",
                minimum_confidence=0.8,
            )
        }
        if has_gripper or has_contact:
            bindings["$tool"] = BindingSpec(
                variable="$tool",
                entity_kind=EntityKind.TOOL,
                class_name=contact_tool_class,
                minimum_confidence=0.8,
                must_be_attached=True,
            )
        skill_type = (
            SkillType.COMPOSITE
            if has_gripper and has_contact
            else SkillType.CONTACT
            if has_contact
            else SkillType.MANIPULATION
            if has_gripper
            else SkillType.MOTION
        )
        motion_profiles = ["linear_slow"]
        if any(
            item.get("chosen_primitive_id") == "motion.move_c"
            for item in simplification_provenance
        ):
            motion_profiles.append("circular_normal")
        if any(
            item.get("chosen_primitive_id") == "motion.move_periodic"
            for item in simplification_provenance
        ):
            motion_profiles.append("periodic_safe")
        return SkillGraph(
            skill_id=skill_id,
            version=self._next_recording_candidate_version(skill_id),
            name=str(semantic["display_name"]),
            description=(
                f"RGB-D two-fingertip teaching candidate: "
                f"{semantic['task_description']}"
            ),
            skill_type=skill_type,
            source_demonstrations=(
                [
                    str(draft_payload.get("artifact_uri") or ""),
                    str(calibration["artifact_uri"]),
                    str(trajectory["artifact_uri"]),
                ]
                + (
                    [str(handeye_transform["artifact_uri"])]
                    if handeye_transform
                    else []
                )
            ),
            operator_style="safe",
            required_tools=[contact_tool_class] if contact_tool_class else [],
            required_entity_roles={"$surface": "contact_target"},
            bindings=bindings,
            nodes=nodes,
            start_node=nodes[0].node_id,
            terminal_nodes=[nodes[-1].node_id],
            motion_profiles=motion_profiles,
            force_profiles=["contact_search_soft"] if has_contact else [],
            preconditions=[
                "fresh_scene",
                "surface_binding_verified",
                "operator_review_required",
            ],
            postconditions=["mock_validation_only"],
            uncertainty={
                "hardware_validated": False,
                "task_plane_calibration_id": calibration["calibration_id"],
                "task_plane_anchor_id": calibration["surface_anchor_id"],
                "task_plane_source_frame": calibration["source_frame"],
                "camera_to_task_plane": calibration.get("camera_to_task_plane")
                or calibration.get("camera_to_surface"),
                "base_chain": calibration.get("base_chain"),
                "tcp_trajectory_id": trajectory["trajectory_id"],
                "trajectory_quality": trajectory["quality"],
                "observed_trajectory_task_plane_m": [
                    [
                        float(sample["position_surface_m"][0]),
                        float(sample["position_surface_m"][1]),
                        float(sample["position_surface_m"][2]),
                    ]
                    for sample in samples
                ],
                "gripper_state_transitions": [
                    {
                        "frame_index": int(item["frame_index"]),
                        "timestamp_ns": int(item["timestamp_ns"]),
                        "previous_state": item.get("previous_state"),
                        "state": item["state"],
                        "distance_m": item.get("distance_m"),
                    }
                    for _boundary, item in gripper_events
                ],
                "gripper_behavior_unresolved": not has_gripper,
                "motion_simplification": simplification_provenance,
                "grip_action_end_motion_policy": motion_phase_policy,
                "semantic_conflicts": trajectory.get("semantic_conflicts") or [],
                "initial_scene_anchor_evidence": draft_payload.get(
                    "initial_scene_anchors"
                ),
                "initial_scene_anchor_inputs": initial_scene_anchor_inputs,
                "semantic_confidence": semantic.get("confidence"),
                "contact_semantics_deferred": requested_contact and not has_contact,
                "handeye_transform_candidate": (
                    {
                        "import_id": handeye_transform.get("import_id"),
                        "passed": handeye_transform.get("passed") is True,
                        "hardware_validated": False,
                        "runtime_authorized": False,
                    }
                    if handeye_transform
                    else None
                ),
            },
            validation_status=ValidationStatus.PENDING,
            lifecycle_status=SkillLifecycleStatus.CANDIDATE,
        )

    @staticmethod
    def _recording_motion_phase_policy(
        node_specs: list[tuple[str, str, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Classify and enforce the generated Grip -> Action -> End block budget.

        Geometry has already passed the bounded local simplifier when this is
        called.  A larger result is rejected rather than silently crossing a
        stable gripper boundary or dropping an observed motion interval.
        """

        first_close_index = next(
            (
                index
                for index, (_node_id, operation, _arguments) in enumerate(node_specs)
                if operation == "gripper.close"
            ),
            None,
        )
        final_open_index = next(
            (
                index
                for index in range(len(node_specs) - 1, -1, -1)
                if node_specs[index][1] == "gripper.open"
                and (first_close_index is None or index > first_close_index)
            ),
            None,
        )
        phase_node_ids: dict[str, list[str]] = {
            "grip": [],
            "action": [],
            "end_motion": [],
        }
        for index, (node_id, operation, _arguments) in enumerate(node_specs):
            if not operation.startswith("motion."):
                continue
            if operation not in RECORDING_BLOCK_MOTION_OPERATIONS:
                raise ValueError(
                    "recording materialization emitted a path operation that is not "
                    f"available as an approved Blockly motion block: {operation}"
                )
            if first_close_index is None:
                phase = "action"
            elif index < first_close_index:
                phase = "grip"
            elif final_open_index is not None and index > final_open_index:
                phase = "end_motion"
            else:
                phase = "action"
            phase_node_ids[phase].append(node_id)

        limits = {
            "action": MAXIMUM_ACTION_MOTION_BLOCKS,
            "end_motion": MAXIMUM_END_MOTION_BLOCKS,
        }
        violations = [
            f"{phase} has {len(phase_node_ids[phase])} motion blocks (maximum {limit})"
            for phase, limit in limits.items()
            if len(phase_node_ids[phase]) > limit
        ]
        if violations:
            raise ValueError(
                "Grip -> Action -> End path cannot be represented within the Blockly "
                "motion budget after bounded local simplification without crossing a "
                "gripper boundary: "
                + "; ".join(violations)
            )
        return {
            "schema_version": "1.0",
            "source": "deterministic_local_trajectory_simplifier",
            "allowed_motion_operations": sorted(RECORDING_BLOCK_MOTION_OPERATIONS),
            "maximum_motion_blocks": limits,
            "phases": {
                phase: {
                    "motion_node_ids": node_ids,
                    "motion_block_count": len(node_ids),
                }
                for phase, node_ids in phase_node_ids.items()
            },
            "passed": True,
        }

    @staticmethod
    def _bounded_path_chunks(
        path: list[dict[str, Any]],
        maximum_count: int,
        *,
        overlap: bool = False,
    ) -> list[list[dict[str, Any]]]:
        if maximum_count < 2:
            raise ValueError("path chunk maximum_count must be at least two")
        if not path:
            raise ValueError("path chunks require at least one pose")
        if len(path) <= maximum_count:
            return [path]
        chunks: list[list[dict[str, Any]]] = []
        start = 0
        while start < len(path):
            end = min(len(path), start + maximum_count)
            chunk = path[start:end]
            chunks.append(chunk)
            if end == len(path):
                break
            start = end - 1 if overlap else end
        return chunks

    def _next_recording_candidate_version(self, skill_id: str) -> str:
        for minor in range(1, 1000):
            candidate = f"0.{minor}.0-candidate"
            if self._find_version_optional(skill_id, candidate) is None:
                return candidate
        raise ValueError("no candidate semantic version slot remains")

    def _recording_skill_draft_view(
        self,
        payload: dict[str, Any],
        *,
        artifact_uri: str,
        fallback_created_at_ns: int,
    ) -> dict[str, Any]:
        if payload.get("status") != "semantic_draft":
            raise ValueError("recording draft artifact has an invalid status")
        draft_id = str(payload["draft_id"])
        if not _RECORDING_DRAFT_ID_PATTERN.fullmatch(draft_id):
            raise ValueError("recording draft artifact has an invalid draft_id")
        draft = payload["draft"]
        transport = payload.get("transport") or {}
        if not isinstance(draft, dict) or not isinstance(transport, dict):
            raise ValueError("recording draft artifact has an invalid payload")
        tcp_observation = draft.get("tcp_proxy_observation")
        tcp_detected = (
            bool(tcp_observation.get("detected"))
            if isinstance(tcp_observation, dict)
            else False
        )
        has_rgbd_evidence = self._recording_has_complete_rgbd_evidence(payload)
        calibration = self._latest_draft_evidence(
            payload, "surface_calibration_*.json"
        )
        trajectory = self._latest_draft_evidence(payload, "tcp_trajectory_*.json")
        if self._fixed_workspace_reuse_enabled() and (
            calibration is None or calibration.get("fixed_workspace_reuse") is not True
        ):
            # The button can be enabled immediately: the exact evidence is
            # persisted only when Candidate registration begins.
            calibration = self._fixed_workspace_calibration_payload(payload)
        registration = self._latest_draft_evidence(
            payload, "candidate_registration_*.json"
        )
        handeye_transform = self._latest_legacy_handeye_transform()
        promotion = PromotionPolicy().evaluate_recording(
            calibration=calibration,
            trajectory=trajectory,
            has_rgbd_evidence=has_rgbd_evidence,
            has_semantic_schema=True,
            gpt_fingertips_detected=tcp_detected,
            handeye_verified=bool(
                handeye_transform and handeye_transform.get("passed") is True
            ),
            semantic_confidence=(
                float(draft["confidence"])
                if isinstance(draft.get("confidence"), (int, float))
                else None
            ),
        )
        mock_validation_passed = bool(
            registration and registration.get("mock_validation_passed") is True
        )
        registration_finalized = registration is not None
        mock_validation_check = {
            "id": "mock_validation",
            "label": "컴파일 및 Mock 회귀 검증",
            "passed": mock_validation_passed,
            "required": registration_finalized,
            "blocking": registration_finalized,
            "pending": not registration_finalized,
            "detail": (
                f"{registration['skill_id']}@{registration['version']} Mock 검증 통과"
                if mock_validation_passed and isinstance(registration, dict)
                else "Candidate SkillGraph의 Mock 회귀 검증이 실패했습니다."
                if registration_finalized
                else "Candidate 등록 시 자동 실행되는 후속 검증입니다."
            ),
        }
        checks = [
            *promotion.as_dict()["checks"],
            mock_validation_check,
        ]
        failed_blocking_ids = {
            item["id"]
            for item in promotion.as_dict()["checks"]
            if item["blocking"] and not item["passed"]
        }
        if "operator_task_plane" in failed_blocking_ids:
            readiness_status = "needs_calibration"
        elif failed_blocking_ids:
            readiness_status = "needs_pose_evidence"
        elif mock_validation_passed:
            readiness_status = "candidate_registered"
        elif registration is not None:
            readiness_status = "candidate_validation_failed"
        else:
            readiness_status = "ready_for_candidate"
        can_register_candidate = promotion.eligible and not mock_validation_passed
        readiness_blockers = list(promotion.blockers)
        if registration_finalized and not mock_validation_passed:
            readiness_blockers.append(str(mock_validation_check["detail"]))
        return {
            **payload,
            "artifact_uri": artifact_uri,
            "created_at_ns": int(payload.get("created_at_ns") or fallback_created_at_ns),
            "keyframe_count": len(payload.get("keyframe_indices") or []),
            "promotion_evidence": {
                "calibration": calibration,
                "trajectory": (
                    {
                        key: trajectory[key]
                        for key in (
                            "trajectory_id",
                            "calibration_id",
                            "surface_anchor_id",
                            "method",
                            "quality",
                            "artifact_uri",
                        )
                        if key in trajectory
                    }
                    if trajectory
                    else None
                ),
                "candidate_registration": registration,
                "handeye_transform_candidate": handeye_transform,
            },
            "promotion_readiness": {
                "status": readiness_status,
                "can_register_candidate": can_register_candidate,
                "checks": checks,
                "blockers": readiness_blockers,
                "warnings": list(promotion.warnings),
                "next_action": (
                    "RGB에서 원점·+X·+Y를 지정해 최종 task-plane TF를 확정하세요."
                    if readiness_status == "needs_calibration"
                    else "같은 task-plane revision에서 metric TCP 경로를 생성하세요."
                    if readiness_status == "needs_pose_evidence"
                    else "Candidate 등록을 눌러 컴파일과 Mock 검증을 실행하세요."
                    if readiness_status == "ready_for_candidate"
                    else "Candidate 검증 실패 원인을 확인한 뒤 다시 등록하세요."
                    if readiness_status == "candidate_validation_failed"
                    else "검증된 후보가 스킬 목록에 등록되었습니다. 활성화 전 검토하세요."
                ),
            },
        }

    def _scene(self, scene_id: str) -> SceneSnapshot:
        if scene_id in self._scenes:
            return self._scenes[scene_id]
        with self.database.session() as session:
            record = session.get(SceneRecord, scene_id)
            if record is None:
                raise KeyError(f"unknown scene {scene_id!r}")
            scene = SceneSnapshot.model_validate(record.scene_json)
        self._scenes[scene_id] = scene
        return scene

    def induce_skill(self, request: dict[str, Any]) -> dict[str, Any]:
        source = self._safe_demo_path(str(request["demo_path"]))
        if source.is_dir() and (source / "rgbd_manifest.json").is_file():
            return self._induce_rgbd_components(source, request)
        evidence = self._demonstration_evidence(source)
        graph, fitted_operations = build_wipe_skill_graph()
        graph = self._apply_node_argument_updates(
            graph,
            self._wipe_path_updates(
                graph, self._normalized_path(evidence.processed)
            ),
        )
        requested_name = str(request.get("name", graph.skill_id))
        if requested_name != graph.skill_id:
            graph = graph.model_copy(update={"skill_id": requested_name}, deep=True)
        graph = graph.model_copy(
            update={
                "source_demonstrations": [self._portable_artifact_reference(source)],
                "operator_style": (
                    "expert_precise"
                    if evidence.trajectory.operator_role == "expert"
                    and evidence.quality.confidence >= 0.9
                    else "safe"
                ),
                "uncertainty": {
                    "teaching_quality": evidence.quality.model_dump(mode="json"),
                    "trajectory_summary": evidence.summary.model_dump(mode="json"),
                    "observed_primitive": evidence.recommendation.recommended_primitive_id,
                    "promotion_warnings": list(evidence.promotion_warnings),
                },
            },
            deep=True,
        )
        primitive_catalog = [
            item.operation_name for item in get_default_registry().catalog()
        ]
        semantic, metadata = DemonstrationAnalyzer(self.settings).analyze(
            DemonstrationAnalysisInput(
                transcript_text=str(request.get("transcript_text", graph.name)),
                scene_summary={
                    "tool_id": "wiper_01",
                    "target_ids": ["table_surface_01"],
                },
                pose_summary=evidence.summary.model_dump(mode="json"),
                entity_catalog=["wiper_01", "table_surface_01"],
                primitive_catalog=primitive_catalog,
                approved_motion_profiles=list(self._motion_profiles()),
                approved_force_profiles=list(self._force_profiles()),
                motion_fitting_candidates=[
                    {"operation": item.primitive_id}
                    for item in evidence.recommendation.candidates
                ],
                confidence_summary={
                    "trajectory": evidence.quality.confidence,
                    "minimum_pose": evidence.summary.minimum_confidence,
                },
            )
        )
        promotion_warnings = list(evidence.promotion_warnings)
        if semantic.reteach_required:
            promotion_warnings.append(
                "semantic analysis recommends reteaching; stored as an inactive Candidate"
            )
        if promotion_warnings:
            graph = graph.model_copy(
                update={
                    "uncertainty": {
                        **graph.uncertainty,
                        "promotion_warnings": list(dict.fromkeys(promotion_warnings)),
                    }
                },
                deep=True,
            )
        version = self._persist_graph(
            graph,
            status="candidate",
            validation_status="pending",
            variant=str(request.get("variant", graph.operator_style or "default")),
        )
        validation = self.validate_skill(
            graph.skill_id, {"version": graph.version, "mode": "mock"}
        )
        if not validation["passed"]:
            raise ValueError("induced skill failed deterministic mock regression validation")
        result_row = self._find_version(graph.skill_id, graph.version)
        if not promotion_warnings:
            result = self.activate_skill(graph.skill_id, {"version": graph.version})
            result_status = result["status"]
            result_version = result["version"]
        else:
            result_status = result_row.status
            result_version = result_row.semantic_version
        return {
            "skill_id": graph.skill_id,
            "version": result_version,
            "status": result_status,
            "active": not promotion_warnings,
            "promotion_warnings": list(dict.fromkeys(promotion_warnings)),
            "source_demo": self._portable_artifact_reference(source),
            "fitted_operations": fitted_operations,
            "observed_primitive": evidence.recommendation.recommended_primitive_id,
            "teaching_quality": evidence.quality.confidence,
            "openai_trace_id": metadata.trace_id,
            "generated_code_uri": version.generated_code_uri,
            "generated_code_checksum_sha256": version.generated_code_checksum_sha256,
        }

    @staticmethod
    def _canonical_json_checksum(value: Any) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _put_content_addressed_json(
        self,
        *,
        recording_id: str,
        artifact_kind: str,
        value: dict[str, Any],
    ) -> Any:
        checksum = self._canonical_json_checksum(value)
        return self.store.put_json(
            f"demonstrations/{recording_id}/induction_v1/"
            f"{artifact_kind}_{checksum}.json",
            value,
        )

    def _approved_training_transcript(
        self, request: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any]]:
        transcript = request.get("transcript_text")
        artifact_uri = request.get("transcript_artifact_uri")
        artifact_checksum = request.get("transcript_artifact_checksum_sha256")
        if (artifact_uri is None) is not (artifact_checksum is None):
            raise ValueError("transcript artifact URI and checksum must be provided together")
        if transcript is not None and artifact_uri is not None:
            raise ValueError("provide exactly one training transcript source")
        if transcript is not None:
            normalized = str(transcript).strip()
            if normalized:
                if len(normalized) > 2_000:
                    raise ValueError("training transcript exceeds 2000 characters")
                return normalized, {"kind": "request", "approved": True}

        if artifact_uri is None and artifact_checksum is None:
            return None, {"kind": "missing", "approved": False}
        if not isinstance(artifact_uri, str) or not isinstance(artifact_checksum, str):
            raise ValueError("transcript artifact URI and checksum must be strings")
        with self.database.session() as session:
            record = session.scalar(
                select(TeachingSessionRecord).where(
                    TeachingSessionRecord.artifact_uri == artifact_uri,
                    TeachingSessionRecord.artifact_checksum_sha256
                    == artifact_checksum,
                    TeachingSessionRecord.ended_at_ns.is_not(None),
                )
            )
        if record is None or record.status not in {"finished", "completed"}:
            raise ValueError(
                "transcript artifact must be the checksum-pinned output of a finalized "
                "teaching session"
            )
        try:
            payload = json.loads(
                self.store.read_bytes(
                    artifact_uri,
                    expected_checksum_sha256=artifact_checksum,
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("approved transcript artifact is not UTF-8 JSON") from exc
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise ValueError("approved transcript artifact does not contain a successful session")
        text_value = payload.get("transcript_text") or payload.get(
            "operator_instruction"
        )
        if not isinstance(text_value, str) or not text_value.strip():
            return None, {
                "kind": "finalized_teaching_artifact",
                "approved": True,
                "artifact_uri": artifact_uri,
                "artifact_checksum_sha256": artifact_checksum,
            }
        normalized = text_value.strip()
        if len(normalized) > 2_000:
            raise ValueError("approved training transcript exceeds 2000 characters")
        return normalized, {
            "kind": "finalized_teaching_artifact",
            "approved": True,
            "artifact_uri": artifact_uri,
            "artifact_checksum_sha256": artifact_checksum,
        }

    def _resolve_training_semantics(
        self, transcript: str
    ) -> tuple[TrainingTaskSemantics, dict[str, Any]]:
        objects = [
            item
            for item in self.repository.list_catalog_entries(kind="object")
            if item.status != "retired"
        ]
        actions = [
            item
            for item in self.repository.list_catalog_entries(kind="action")
            if item.status != "retired"
        ]
        object_catalog = {
            item.canonical_id: tuple(
                dict.fromkeys([item.display_name, *item.aliases_json])
            )
            for item in objects
        }
        action_catalog = {
            item.canonical_id: tuple(
                dict.fromkeys([item.display_name, *item.aliases_json])
            )
            for item in actions
        }
        action_required_roles: dict[str, tuple[str, ...]] = {}
        for item in actions:
            raw_roles = item.metadata_json.get("required_roles", [])
            action_required_roles[item.canonical_id] = tuple(
                role for role in raw_roles if isinstance(role, str)
            ) if isinstance(raw_roles, list) else ()
        semantics, metadata = TrainingSemanticResolver(self.settings).resolve(
            transcript,
            object_catalog=object_catalog,
            action_catalog=action_catalog,
            action_required_roles=action_required_roles,
        )
        return semantics, metadata.model_dump(mode="json")

    def _ensure_rgbd_semantic_catalogs(
        self,
        *,
        semantics: TrainingTaskSemantics,
        manifest_checksum_sha256: str,
    ) -> dict[str, dict[str, Any]]:
        if semantics.object_class_id is None or semantics.action_id is None:
            raise ValueError("resolved training semantics require object and action IDs")
        catalog: dict[str, dict[str, Any]] = {}
        for kind, identifier in (
            ("object", semantics.object_class_id),
            ("action", semantics.action_id),
        ):
            entry = self.repository.get_catalog_entry(
                kind=kind, identifier=identifier, active_only=False
            )
            created = entry is None
            if entry is None:
                metadata: dict[str, Any] = {
                    "discovered_by": "rgbd_training_semantics",
                    "source_manifest_checksum_sha256": manifest_checksum_sha256,
                    "semantic_only": True,
                    "runtime_exposed": False,
                    "hardware_compatible": False,
                }
                if kind == "action":
                    metadata.update(
                        {
                            "required_roles": list(semantics.required_roles),
                            "input_contract": {},
                            "output_contract": {},
                            "anchor_policy": {},
                        }
                    )
                entry = self.repository.create_catalog_entry(
                    kind=kind,
                    canonical_id=identifier,
                    display_name=identifier,
                    aliases=[],
                    metadata=metadata,
                )
            summary = self._catalog_entry_summary(entry)
            summary["created_from_training"] = created
            catalog[kind] = summary
        return catalog

    def _rgbd_component_candidates(
        self,
        *,
        dataset: Any,
        segmentation: RGBDDatasetSegmentation,
        segmentation_report: Any,
        semantics: TrainingTaskSemantics,
        catalog: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if semantics.object_class_id is None or semantics.action_id is None:
            raise ValueError("component candidates require resolved semantics")
        common_source = {
            "recording_id": dataset.manifest.recording_id,
            "manifest_checksum_sha256": dataset.manifest.manifest_checksum_sha256,
            "segmentation_report_uri": segmentation_report.uri,
            "segmentation_report_checksum_sha256": segmentation_report.checksum_sha256,
            "candidate_only": True,
            "executable": False,
            "hardware_compatible": False,
        }
        grip_payload = {
            "schema_version": "1.0",
            "component_type": "grip_profile_evidence",
            "object_class_id": semantics.object_class_id,
            "frame_policy": "camera_relative_observation_only",
            "grip_interval": segmentation.segmentation.grip.model_dump(mode="json"),
            "stable_close_evidence": (
                segmentation.segmentation.evidence.first_stable_close.model_dump(
                    mode="json"
                )
            ),
            "source": common_source,
            "missing_hardware_evidence": [
                "object_relative_6d_pose",
                "approach_axis",
                "hand_eye_and_task_plane_revision",
                "rg2_fingertip_calibration",
                "post_lift_object_following_verification",
            ],
        }
        grip_artifact = self._put_content_addressed_json(
            recording_id=dataset.manifest.recording_id,
            artifact_kind="grip_candidate",
            value=grip_payload,
        )
        grip_version = self.repository.register_grip_profile_version(
            object_class_id=semantics.object_class_id,
            semantic_version=(
                f"0.0.0-{grip_artifact.checksum_sha256[:12]}-candidate"
            ),
            artifact_uri=grip_artifact.uri,
            artifact_checksum_sha256=grip_artifact.checksum_sha256,
            object_frame_policy="camera_relative_observation_only",
            object_frame_revision=dataset.manifest.manifest_checksum_sha256,
            status="candidate",
            validation_status="pending",
            hardware_compatible=False,
            auto_activation_allowed=False,
            gripper_calibration_profile_id=None,
            metadata={
                "source": "raw_rgbd_local_segmentation",
                "diagnostic_only": True,
                "normalized_2d_is_not_an_execution_target": True,
            },
        )

        action_payload = {
            "schema_version": "1.0",
            "component_type": "action_interval_evidence",
            "action_id": semantics.action_id,
            "required_roles": list(semantics.required_roles),
            "action_interval": segmentation.segmentation.action.model_dump(mode="json"),
            "intermediate_gripper_transitions": [
                item.model_dump(mode="json")
                for item in (
                    segmentation.segmentation.evidence.action_intermediate_transitions
                )
            ],
            "motion_block_policy": {
                "allowed_motion_operations": sorted(
                    RECORDING_BLOCK_MOTION_OPERATIONS
                ),
                "maximum_motion_blocks": MAXIMUM_ACTION_MOTION_BLOCKS,
                "geometry_owner": "deterministic_local_trajectory_simplifier",
            },
            "source": common_source,
        }
        action_artifact = self._put_content_addressed_json(
            recording_id=dataset.manifest.recording_id,
            artifact_kind="action_candidate",
            value=action_payload,
        )
        end_payload = {
            "schema_version": "1.0",
            "component_type": "end_motion_interval_evidence",
            "action_id": semantics.action_id,
            "end_motion_id": None,
            "end_motion_interval": segmentation.segmentation.end_motion.model_dump(
                mode="json"
            ),
            "final_stable_open": (
                segmentation.segmentation.evidence.final_stable_open.model_dump(
                    mode="json"
                )
            ),
            "motion_block_policy": {
                "allowed_motion_operations": sorted(
                    RECORDING_BLOCK_MOTION_OPERATIONS
                ),
                "maximum_motion_blocks": MAXIMUM_END_MOTION_BLOCKS,
                "geometry_owner": "deterministic_local_trajectory_simplifier",
            },
            "source": common_source,
        }
        end_artifact = self._put_content_addressed_json(
            recording_id=dataset.manifest.recording_id,
            artifact_kind="end_motion_candidate",
            value=end_payload,
        )
        return {
            "grip": {
                **self._grip_profile_version_summary(grip_version),
                "object_class_id": semantics.object_class_id,
                "catalog_status": catalog["object"]["status"],
                "component_status": "evidence_candidate",
                "blockers": grip_payload["missing_hardware_evidence"],
            },
            "action": {
                "action_id": semantics.action_id,
                "catalog_status": catalog["action"]["status"],
                "component_status": "segmented_evidence_only",
                "definition_created": False,
                "artifact_uri": action_artifact.uri,
                "artifact_checksum_sha256": action_artifact.checksum_sha256,
                "hardware_compatible": False,
                "blockers": [
                    "anchor-relative action geometry is not available",
                    "an executable Action SkillGraph has not been compiled",
                ],
            },
            "end_motion": {
                "end_motion_id": None,
                "component_status": "segmented_evidence_only",
                "definition_created": False,
                "mapping_created": False,
                "artifact_uri": end_artifact.uri,
                "artifact_checksum_sha256": end_artifact.checksum_sha256,
                "hardware_compatible": False,
                "blockers": [
                    "an approved anchor-relative EndMotion SkillGraph is required",
                    "Action-to-End mapping must be selected locally after validation",
                ],
            },
        }

    def _induce_rgbd_components(
        self, source: Path, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Integrity-load and split raw RGB-D evidence without fabricating a skill."""

        dataset = load_rgbd_dataset(source)
        segmentation_config = GripActionEndSegmentationConfig(
            finger_close_threshold_m=self.settings.finger_close_threshold_m,
            finger_state_stable_frames=self.settings.finger_state_stable_frames,
        )
        segmentation: RGBDDatasetSegmentation | None = None
        segmentation_status = "succeeded"
        segmentation_failure: dict[str, Any] | None = None
        try:
            segmentation = segment_rgbd_dataset(
                dataset,
                config=segmentation_config,
            )
            report_payload: dict[str, Any] = {
                "schema_version": "1.0",
                "report_type": "rgbd_grip_action_end_segmentation",
                "status": "succeeded",
                "source": {
                    "recording_id": dataset.manifest.recording_id,
                    "manifest_checksum_sha256": (
                        dataset.manifest.manifest_checksum_sha256
                    ),
                    "frame_count": dataset.manifest.frame_count,
                    "recording_fps": dataset.manifest.recording_fps,
                },
                "configuration": segmentation_config.model_dump(mode="json"),
                "observations": [
                    item.model_dump(mode="json") for item in segmentation.observations
                ],
                "transitions": [
                    item.model_dump(mode="json") for item in segmentation.transitions
                ],
                "segmentation": segmentation.segmentation.model_dump(mode="json"),
                "executable": False,
                "hardware_compatible": False,
            }
        except GripActionEndSegmentationError as exc:
            segmentation_status = "structural_failure"
            segmentation_failure = exc.failure.model_dump(mode="json")
            report_payload = {
                "schema_version": "1.0",
                "report_type": "rgbd_grip_action_end_segmentation",
                "status": segmentation_status,
                "source": {
                    "recording_id": dataset.manifest.recording_id,
                    "manifest_checksum_sha256": (
                        dataset.manifest.manifest_checksum_sha256
                    ),
                    "frame_count": dataset.manifest.frame_count,
                    "recording_fps": dataset.manifest.recording_fps,
                },
                "configuration": segmentation_config.model_dump(mode="json"),
                "failure": segmentation_failure,
                "executable": False,
                "hardware_compatible": False,
            }
        except NotConfiguredError as exc:
            segmentation_status = "configuration_failure"
            segmentation_failure = {
                "code": "mediapipe_not_configured",
                "message": str(exc),
            }
            report_payload = {
                "schema_version": "1.0",
                "report_type": "rgbd_grip_action_end_segmentation",
                "status": segmentation_status,
                "source": {
                    "recording_id": dataset.manifest.recording_id,
                    "manifest_checksum_sha256": (
                        dataset.manifest.manifest_checksum_sha256
                    ),
                    "frame_count": dataset.manifest.frame_count,
                    "recording_fps": dataset.manifest.recording_fps,
                },
                "configuration": segmentation_config.model_dump(mode="json"),
                "failure": segmentation_failure,
                "executable": False,
                "hardware_compatible": False,
            }
        segmentation_report = self._put_content_addressed_json(
            recording_id=dataset.manifest.recording_id,
            artifact_kind="segmentation_report",
            value=report_payload,
        )

        transcript, semantic_source = self._approved_training_transcript(request)
        semantics: TrainingTaskSemantics | None = None
        semantic_trace: dict[str, Any] | None = None
        semantic_status = "resolved"
        semantic_blockers: list[str] = []
        catalog: dict[str, dict[str, Any]] = {}
        if transcript is None:
            semantic_status = "semantics_missing"
            semantic_blockers.append(
                "transcript_text or a checksum-pinned finalized teaching artifact is required"
            )
        else:
            semantics, semantic_trace = self._resolve_training_semantics(transcript)
            if semantics.ambiguity:
                semantic_status = "semantics_unresolved"
                semantic_blockers.extend(semantics.unresolved_ambiguities)
            else:
                catalog = self._ensure_rgbd_semantic_catalogs(
                    semantics=semantics,
                    manifest_checksum_sha256=(
                        dataset.manifest.manifest_checksum_sha256
                    ),
                )

        blockers = list(semantic_blockers)
        if segmentation_failure is not None:
            blockers.append(str(segmentation_failure["message"]))
        components: dict[str, Any] = {
            "grip": {"component_status": "not_created"},
            "action": {"component_status": "not_created"},
            "end_motion": {"component_status": "not_created"},
        }
        if (
            segmentation is not None
            and semantics is not None
            and not semantics.ambiguity
        ):
            components = self._rgbd_component_candidates(
                dataset=dataset,
                segmentation=segmentation,
                segmentation_report=segmentation_report,
                semantics=semantics,
                catalog=catalog,
            )
            blockers.extend(
                [
                    "Grip evidence lacks an object-relative 6D execution pose",
                    "Action and EndMotion lack approved anchor-relative SkillGraphs",
                    "an active Action-to-End mapping has not been created",
                ]
            )

        if segmentation_status == "configuration_failure":
            result_status = "configuration_failure"
        elif segmentation_status != "succeeded" or semantic_status != "resolved":
            result_status = "structural_failure"
        else:
            result_status = "candidate"
        segmentation_summary: dict[str, Any] = {
            "status": segmentation_status,
            "report_uri": segmentation_report.uri,
            "report_checksum_sha256": segmentation_report.checksum_sha256,
            "failure": segmentation_failure,
        }
        if segmentation is not None:
            segmentation_summary.update(
                {
                    "grip": segmentation.segmentation.grip.model_dump(mode="json"),
                    "action": segmentation.segmentation.action.model_dump(mode="json"),
                    "end_motion": segmentation.segmentation.end_motion.model_dump(
                        mode="json"
                    ),
                    "warnings": [
                        item.model_dump(mode="json")
                        for item in segmentation.segmentation.warnings
                    ],
                }
            )
        return {
            "skill_id": None,
            "status": result_status,
            "active": False,
            "executable": False,
            "hardware_compatible": False,
            "source_demo": self._portable_artifact_reference(source),
            "dataset": {
                "recording_id": dataset.manifest.recording_id,
                "frame_count": dataset.manifest.frame_count,
                "manifest_checksum_sha256": (
                    dataset.manifest.manifest_checksum_sha256
                ),
                "integrity_verified": True,
            },
            "segmentation": segmentation_summary,
            "segmentation_report_uri": segmentation_report.uri,
            "segmentation_report_checksum_sha256": (
                segmentation_report.checksum_sha256
            ),
            "semantic_classification": {
                "status": semantic_status,
                "source": semantic_source,
                "result": (
                    semantics.model_dump(mode="json") if semantics is not None else None
                ),
                "trace": semantic_trace,
            },
            "catalog": catalog,
            "selected_components": components,
            "task_flow_manifest": None,
            "fitted_operations": [],
            "promotion_warnings": list(dict.fromkeys(blockers)),
            "blockers": list(dict.fromkeys(blockers)),
        }

    def search_skills(self, request: dict[str, Any]) -> dict[str, Any]:
        query_text = str(request["query"])
        limit = int(request.get("limit", 5))
        embedding_service = SkillEmbeddingService(self.settings)
        try:
            query_embedding = tuple(embedding_service.embed_text(query_text).vector)
            candidates: list[SkillCandidate] = []
            for skill, version in self._active_rows():
                graph = SkillGraph.model_validate(version.graph_json)
                document = SkillSearchDocument(
                    name=skill.name,
                    description=skill.description,
                    target_objects=tuple(graph.required_entity_roles.values()),
                    tools=tuple(graph.required_tools),
                    style=graph.operator_style or "normal",
                    preconditions=tuple(graph.preconditions),
                )
                stored_embedding = self.repository.get_skill_embedding(
                    skill_version_id=version.id,
                    model=self.settings.openai_embedding_model,
                )
                if stored_embedding is None:
                    embedded = embedding_service.embed_document(document)
                    stored_embedding = self.repository.put_skill_embedding(
                        skill_version_id=version.id,
                        model=embedded.model,
                        vector=embedded.vector,
                        source_checksum_sha256=version.graph_checksum_sha256,
                    )
                vector = tuple(stored_embedding.embedding_json)
                candidates.append(
                    SkillCandidate(
                        skill_id=graph.skill_id,
                        version=version.semantic_version,
                        name=skill.name,
                        description=skill.description,
                        embedding=vector,
                        intent=skill.intent,
                        tool_classes=tuple(graph.required_tools),
                        target_types=tuple(graph.required_entity_roles.values()),
                        contact=graph.skill_type.value == "contact",
                        operator_style=graph.operator_style or "normal",
                        status=version.status,
                        validation_status=version.validation_status,
                        hardware_compatible=version.hardware_compatible,
                    )
                )
            ranked = rank_skills(
                SkillSearchQuery(
                    embedding=query_embedding,
                    intent=request.get("intent"),
                    tool_class=request.get("tool_class"),
                    target_type=request.get("target_type"),
                    operator_style=request.get("style"),
                    require_hardware_compatibility=self.settings.robot_execution_mode
                    is SettingsExecutionMode.HARDWARE,
                ),
                candidates,
                limit=limit,
            )
            return {
                "mode": "embedding",
                "results": [
                    {
                        "skill_id": item.candidate.skill_id,
                        "version": item.candidate.version,
                        "name": item.candidate.name,
                        "score": item.score,
                        "factors": item.factors,
                    }
                    for item in ranked
                ],
            }
        except Exception as exc:
            # This is a deliberate provider boundary: keyword fallback is allowed, execution
            # validation is not bypassed, and no credential/raw request is logged here.
            rows = self.repository.search_active_skills_keyword(query_text, limit=limit)
            return {
                "mode": "keyword_fallback",
                "fallback_reason": type(exc).__name__,
                "results": [self._version_summary(row) for row in rows],
            }

    def list_skills(self) -> dict[str, Any]:
        """Return registry rows required by the operator UI without provider calls."""

        with self.database.session() as session:
            rows = list(
                session.execute(
                    select(SkillVersionRecord, SkillRecord)
                    .join(SkillRecord, SkillVersionRecord.skill_id == SkillRecord.id)
                    .order_by(SkillVersionRecord.created_at.desc())
                ).tuples()
            )
        response: dict[str, Any] = {
            "skills": [self._skill_registry_summary(version, skill) for version, skill in rows]
        }
        hierarchy = self.list_task_flow_hierarchy()
        if hierarchy["objects"]:
            response["task_flow_catalog"] = hierarchy
        return response

    def _skill_registry_summary(
        self, version: SkillVersionRecord, skill: SkillRecord
    ) -> dict[str, Any]:
        summary = {
            **self._version_summary(version, include_graph=True),
            "name": skill.name,
            "intent": skill.intent,
            "variant": skill.variant,
            "description": skill.description,
            "node_count": len(version.graph_json.get("nodes", [])),
        }
        repair = self._builtin_wipe_repair_metadata(version)
        if repair is not None:
            summary["repair"] = repair
        return summary

    @staticmethod
    def _builtin_wipe_repair_metadata(
        version: SkillVersionRecord,
    ) -> dict[str, Any] | None:
        """Identify only the known 1.0 relative-target/MoveJ registry regression."""

        graph = version.graph_json
        if (
            graph.get("skill_id") != "wipe_surface"
            or version.status != "active"
            or version.validation_status != "failed"
        ):
            return None
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            return None
        legacy_node = next(
            (
                node
                for node in nodes
                if isinstance(node, dict) and node.get("node_id") == "approach_joint"
            ),
            None,
        )
        if not isinstance(legacy_node, dict):
            return None
        arguments = legacy_node.get("arguments")
        if (
            legacy_node.get("operation") != "motion.move_j"
            or not isinstance(arguments, dict)
            or "target" not in arguments
            or "target_joint_positions_rad" in arguments
        ):
            return None
        return {
            "available": True,
            "repair_id": "wipe_surface_relative_pre_approach_v1",
            "reason": (
                "legacy approach_joint stores an anchor-relative target under motion.move_j"
            ),
            "mock_only": True,
            "preserves_parent": True,
        }

    def repair_builtin_wipe_skill(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create, validate, and activate a corrected Mock-only child version."""

        if request.get("acknowledge_mock_only") is not True:
            raise ValueError("built-in wipe repair requires Mock-only acknowledgement")
        parent = self._find_version("wipe_surface")
        expected_checksum = str(request.get("expected_parent_checksum_sha256") or "")
        if expected_checksum != parent.graph_checksum_sha256:
            raise ValueError("wipe repair parent checksum no longer matches the registry")

        repair = self._builtin_wipe_repair_metadata(parent)
        if repair is None:
            if (
                parent.status == "active"
                and parent.validation_status == "passed"
                and SkillGraphValidator().inspect(
                    SkillGraph.model_validate(parent.graph_json)
                ).valid
            ):
                return {
                    "repaired": False,
                    "already_ready": True,
                    "mock_only": True,
                    "active": self._version_summary(parent, include_graph=True),
                }
            raise ValueError(
                "selected wipe_surface version is not the recognized built-in regression"
            )

        parent_graph = SkillGraph.model_validate(parent.graph_json)
        repaired_nodes: list[SkillNode] = []
        for node in parent_graph.nodes:
            updates: dict[str, Any] = {}
            if node.node_id == "approach_joint":
                updates.update(
                    {
                        "node_id": "pre_approach_linear",
                        "operation": "motion.move_l",
                        "arguments": {
                            "target": node.arguments["target"],
                            "motion_profile_id": "linear_slow",
                        },
                    }
                )
            if node.on_success == "approach_joint":
                updates["on_success"] = "pre_approach_linear"
            if node.on_failure == "approach_joint":
                updates["on_failure"] = "pre_approach_linear"
            repaired_nodes.append(
                node.model_copy(update=updates, deep=True) if updates else node
            )
        repaired_motion_profiles = list(parent_graph.motion_profiles)
        if "linear_slow" not in repaired_motion_profiles:
            repaired_motion_profiles.append("linear_slow")
        if not any(
            node.arguments.get("motion_profile_id") == "joint_safe"
            for node in repaired_nodes
        ):
            repaired_motion_profiles = [
                profile
                for profile in repaired_motion_profiles
                if profile != "joint_safe"
            ]
        existing_versions = {row.semantic_version for row in self._versions("wipe_surface")}
        candidate_version = next_candidate_version(parent.semantic_version)
        while (
            candidate_version in existing_versions
            or stable_version(candidate_version) in existing_versions
        ):
            parsed = SemanticVersion.parse(candidate_version)
            candidate_version = str(
                SemanticVersion(parsed.major, parsed.minor + 1, 0, "candidate")
            )
        candidate_graph = parent_graph.model_copy(
            update={
                "version": candidate_version,
                "parent_version": parent.semantic_version,
                "nodes": repaired_nodes,
                "motion_profiles": repaired_motion_profiles,
                "validation_status": ValidationStatus.UNVALIDATED,
                "lifecycle_status": SkillLifecycleStatus.CANDIDATE,
                "uncertainty": {
                    **parent_graph.uncertainty,
                    "builtin_repair": {
                        "repair_id": repair["repair_id"],
                        "parent_version": parent.semantic_version,
                        "parent_checksum_sha256": parent.graph_checksum_sha256,
                        "mock_only": True,
                    },
                },
            },
            deep=True,
        )
        candidate = self._persist_graph(
            candidate_graph,
            status="candidate",
            validation_status="pending",
            variant=self._skill_variant(parent),
            parent_version_id=parent.id,
        )
        validation = self.validate_skill(
            "wipe_surface", {"version": candidate.semantic_version, "mode": "mock"}
        )
        if not validation["passed"]:
            return {
                "repaired": False,
                "already_ready": False,
                "mock_only": True,
                "candidate": self._version_summary(
                    self._find_version("wipe_surface", candidate.semantic_version),
                    include_graph=True,
                ),
                "validation": validation,
            }
        active = self.activate_skill(
            "wipe_surface", {"version": candidate.semantic_version}
        )
        return {
            "repaired": True,
            "already_ready": False,
            "mock_only": True,
            "parent_version": parent.semantic_version,
            "candidate_version": candidate.semantic_version,
            "validation": validation,
            "active": active,
        }

    def delete_skill(self, skill_id: str) -> dict[str, Any]:
        """Delete an inactive skill identity and all of its candidate versions."""

        graph_skill_ids = {
            str(row.graph_json.get("skill_id")) for row in self._versions(skill_id)
        }
        result = self.repository.delete_skill(skill_id)
        deleted_artifacts = sum(
            self.store.delete_tree(f"skills/{graph_skill_id}")
            for graph_skill_id in graph_skill_ids
        )
        return {
            "deleted": True,
            **result,
            "deleted_artifacts": deleted_artifacts,
        }

    def deactivate_skill(self, skill_id: str) -> dict[str, Any]:
        """Retire the currently active version without deleting its history."""

        retired = self.repository.deactivate_skill(skill_id)
        return {
            "deactivated": True,
            **self._version_summary(retired, include_graph=False),
        }

    @staticmethod
    def _catalog_entry_summary(entry: SemanticCatalogRecord) -> dict[str, Any]:
        return {
            "id": entry.id,
            "kind": entry.kind,
            "canonical_id": entry.canonical_id,
            "display_name": entry.display_name,
            "aliases": list(entry.aliases_json),
            "status": entry.status,
            "metadata": dict(entry.metadata_json),
        }

    def _catalog_entry(self, kind: str, identifier: str) -> SemanticCatalogRecord:
        entry = self.repository.get_catalog_entry(
            kind=kind, identifier=identifier, active_only=False
        )
        if entry is None:
            raise KeyError(f"unknown {kind} catalog entry {identifier!r}")
        return entry

    def _create_catalog_entry(
        self,
        *,
        kind: str,
        canonical_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(request.get("metadata", {}))
        if kind in {"action", "end_motion"}:
            metadata["required_roles"] = list(request.get("required_roles", []))
            metadata["input_contract"] = dict(request.get("input_contract", {}))
            metadata["output_contract"] = dict(request.get("output_contract", {}))
            metadata["anchor_policy"] = dict(request.get("anchor_policy", {}))
        entry = self.repository.create_catalog_entry(
            kind=kind,
            canonical_id=canonical_id,
            display_name=str(request["display_name"]),
            aliases=list(request.get("aliases", [])),
            metadata=metadata,
        )
        return self._catalog_entry_summary(entry)

    def create_catalog_object(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._create_catalog_entry(
            kind="object",
            canonical_id=str(request["object_class_id"]),
            request=request,
        )

    def create_catalog_action(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._create_catalog_entry(
            kind="action",
            canonical_id=str(request["action_id"]),
            request=request,
        )

    def create_catalog_end_motion(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._create_catalog_entry(
            kind="end_motion",
            canonical_id=str(request["end_motion_id"]),
            request=request,
        )

    def _list_catalog_entries(
        self, *, kind: str, status: str | None, response_key: str
    ) -> dict[str, Any]:
        return {
            response_key: [
                self._catalog_entry_summary(entry)
                for entry in self.repository.list_catalog_entries(
                    kind=kind, status=status
                )
            ]
        }

    def list_catalog_objects(self, *, status: str | None = None) -> dict[str, Any]:
        return self._list_catalog_entries(
            kind="object", status=status, response_key="objects"
        )

    def list_catalog_actions(self, *, status: str | None = None) -> dict[str, Any]:
        return self._list_catalog_entries(
            kind="action", status=status, response_key="actions"
        )

    def list_catalog_end_motions(
        self, *, status: str | None = None
    ) -> dict[str, Any]:
        return self._list_catalog_entries(
            kind="end_motion", status=status, response_key="end_motions"
        )

    def get_catalog_object(self, object_class_id: str) -> dict[str, Any]:
        summary = self._catalog_entry_summary(
            self._catalog_entry("object", object_class_id)
        )
        summary["grip_profile_versions"] = self.list_grip_profiles(
            object_class_id
        )["grip_profile_versions"]
        return summary

    def get_catalog_action(self, action_id: str) -> dict[str, Any]:
        summary = self._catalog_entry_summary(self._catalog_entry("action", action_id))
        summary["definitions"] = [
            self._stage_definition_summary(stage)
            for stage in self.repository.list_stage_definitions(
                kind="action", canonical_id=summary["canonical_id"]
            )
        ]
        mapping = self.repository.get_action_end_mapping(
            action_id=str(summary["canonical_id"]), active_only=False
        )
        summary["active_or_latest_end_mapping"] = (
            self._action_end_mapping_summary(mapping) if mapping is not None else None
        )
        return summary

    def get_catalog_end_motion(self, end_motion_id: str) -> dict[str, Any]:
        summary = self._catalog_entry_summary(
            self._catalog_entry("end_motion", end_motion_id)
        )
        summary["definitions"] = [
            self._stage_definition_summary(stage)
            for stage in self.repository.list_stage_definitions(
                kind="end_motion", canonical_id=summary["canonical_id"]
            )
        ]
        return summary

    def _update_catalog_entry(
        self, kind: str, identifier: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        entry = self._catalog_entry(kind, identifier)
        updated = self.repository.update_catalog_entry(entry.id, **request)
        return self._catalog_entry_summary(updated)

    def update_catalog_object(
        self, object_class_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._update_catalog_entry("object", object_class_id, request)

    def update_catalog_action(
        self, action_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._update_catalog_entry("action", action_id, request)

    def _set_catalog_entry_status(
        self, kind: str, identifier: str, status: str
    ) -> dict[str, Any]:
        entry = self._catalog_entry(kind, identifier)
        updated = self.repository.set_catalog_entry_status(entry.id, status=status)
        return self._catalog_entry_summary(updated)

    def set_catalog_object_status(
        self, object_class_id: str, status: str
    ) -> dict[str, Any]:
        return self._set_catalog_entry_status("object", object_class_id, status)

    def set_catalog_action_status(
        self, action_id: str, status: str
    ) -> dict[str, Any]:
        return self._set_catalog_entry_status("action", action_id, status)

    def set_catalog_end_motion_status(
        self, end_motion_id: str, status: str
    ) -> dict[str, Any]:
        return self._set_catalog_entry_status("end_motion", end_motion_id, status)

    @staticmethod
    def _grip_profile_version_summary(
        version: GripProfileVersionRecord,
    ) -> dict[str, Any]:
        return {
            "id": version.id,
            "version": version.semantic_version,
            "status": version.status,
            "validation_status": version.validation_status,
            "hardware_compatible": version.hardware_compatible,
            "auto_activation_allowed": version.auto_activation_allowed,
            "object_frame_policy": version.object_frame_policy,
            "object_frame_revision": version.object_frame_revision,
            "gripper_calibration_profile_id": (
                version.gripper_calibration_profile_id
            ),
            "artifact_uri": version.artifact_uri,
            "artifact_checksum_sha256": version.artifact_checksum_sha256,
            "metadata": dict(version.metadata_json),
        }

    def list_grip_profiles(self, object_class_id: str) -> dict[str, Any]:
        entry = self._catalog_entry("object", object_class_id)
        return {
            "object_class_id": entry.canonical_id,
            "grip_profile_versions": [
                self._grip_profile_version_summary(version)
                for version in self.repository.list_grip_profile_versions(
                    object_class_id=entry.canonical_id
                )
            ],
        }

    def activate_grip_profile_version(self, version_id: str) -> dict[str, Any]:
        return self._grip_profile_version_summary(
            self.repository.activate_grip_profile_version(
                version_id, automatic=False
            )
        )

    @staticmethod
    def _stage_definition_summary(stage: StageDefinitionRecord) -> dict[str, Any]:
        return {
            "id": stage.id,
            "kind": stage.stage_type,
            "version": stage.semantic_version,
            "skill_version_id": stage.skill_version_id,
            "status": stage.status,
            "validation_status": stage.validation_status,
            "hardware_compatible": stage.hardware_compatible,
            "graph_checksum_sha256": stage.graph_checksum_sha256,
            "required_roles": list(stage.required_roles_json),
            "input_contract": dict(stage.input_contract_json),
            "output_contract": dict(stage.output_contract_json),
            "anchor_policy": dict(stage.anchor_policy_json),
            "metadata": dict(stage.metadata_json),
        }

    def _create_stage_definition(
        self, *, kind: str, canonical_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        entry = self._catalog_entry(kind, canonical_id)
        skill_version = self._find_version(
            str(request["skill_id"]), str(request["skill_version"])
        )
        linked_graph = SkillGraph.model_validate(skill_version.graph_json)
        if linked_graph.skill_id != request["skill_id"]:
            raise ValueError("stage SkillGraph identity does not match the request")
        if linked_graph.version != request["component_version"]:
            raise ValueError(
                "component_version must equal the exact linked SkillGraph version"
            )
        passed = skill_version.validation_status == "passed"
        stage = self.repository.register_stage_definition(
            kind=kind,
            canonical_id=entry.canonical_id,
            skill_version_id=skill_version.id,
            semantic_version=str(request["component_version"]),
            status="validated" if passed else "candidate",
            validation_status="passed" if passed else "pending",
            required_roles=list(request.get("required_roles", [])),
            input_contract=dict(request.get("input_contract", {})),
            output_contract=dict(request.get("output_contract", {})),
            anchor_policy=dict(request.get("anchor_policy", {})),
            metadata=dict(request.get("metadata", {})),
        )
        return self._stage_definition_summary(stage)

    def create_action_definition(
        self, action_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create_stage_definition(
            kind="action", canonical_id=action_id, request=request
        )

    def create_end_motion_definition(
        self, end_motion_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create_stage_definition(
            kind="end_motion", canonical_id=end_motion_id, request=request
        )

    def activate_stage_definition(self, definition_id: str) -> dict[str, Any]:
        return self._stage_definition_summary(
            self.repository.activate_stage_definition(definition_id)
        )

    def _action_end_mapping_summary(
        self, mapping: ActionEndMappingRecord
    ) -> dict[str, Any]:
        with self.database.session() as session:
            action_stage = session.get(
                StageDefinitionRecord, mapping.action_stage_definition_id
            )
            end_stage = session.get(
                StageDefinitionRecord, mapping.end_motion_stage_definition_id
            )
            if action_stage is None or end_stage is None:
                raise ValueError("action/end mapping references a missing stage")
            action_entry = session.get(
                SemanticCatalogRecord, action_stage.catalog_entry_id
            )
            end_entry = session.get(SemanticCatalogRecord, end_stage.catalog_entry_id)
            if action_entry is None or end_entry is None:
                raise ValueError("action/end mapping references a missing catalog entry")
            action_id = action_entry.canonical_id
            end_motion_id = end_entry.canonical_id
            action_definition = self._stage_definition_summary(action_stage)
            end_motion_definition = self._stage_definition_summary(end_stage)
        return {
            "id": mapping.id,
            "action_id": action_id,
            "action_definition_id": mapping.action_stage_definition_id,
            "end_motion_id": end_motion_id,
            "end_motion_definition_id": mapping.end_motion_stage_definition_id,
            "revision": mapping.revision,
            "status": mapping.status,
            "mapping_checksum_sha256": mapping.mapping_checksum_sha256,
            "action_definition": action_definition,
            "end_motion_definition": end_motion_definition,
            "metadata": dict(mapping.metadata_json),
        }

    def map_action_end_motion(
        self, action_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        mapping = self.repository.map_action_to_end_motion(
            action_id=self._catalog_entry("action", action_id).canonical_id,
            end_motion_id=str(request["end_motion_id"]),
            revision=request.get("revision"),
            activate=True,
        )
        return self._action_end_mapping_summary(mapping)

    def import_grip_point_candidates(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_path = Path(str(request.get("artifact_path") or ""))
        path = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (self.settings.repo_root / raw_path).resolve()
        )
        allowed_root = (self.settings.repo_root / "data/test/grip_point").resolve()
        if path.suffix != ".json" or not path.is_relative_to(allowed_root):
            raise ValueError(
                "grip-point imports are restricted to JSON under data/test/grip_point"
            )
        imported = GripPointResultImporter(self.repository, self.store).import_file(path)
        return {
            "source_checksum_sha256": imported.source_checksum_sha256,
            "hardware_compatible": False,
            "auto_activation_allowed": False,
            "profiles": [
                {
                    "object_class_id": item.object_class_id,
                    "catalog_entry_id": item.catalog_entry_id,
                    "grip_profile_id": item.grip_profile_id,
                    "grip_profile_version_id": item.grip_profile_version_id,
                    "artifact_uri": item.artifact_uri,
                    "artifact_checksum_sha256": (
                        item.artifact_checksum_sha256
                    ),
                }
                for item in imported.profiles
            ],
        }

    def list_task_flow_hierarchy(self) -> dict[str, Any]:
        """List the semantic cross-product; composition decides actual compatibility."""

        objects = self.repository.list_catalog_entries(kind="object")
        actions = self.repository.list_catalog_entries(kind="action")
        action_components: dict[str, tuple[StageDefinitionRecord | None, Any]] = {}
        for action in actions:
            action_components[action.canonical_id] = (
                self.repository.active_stage_definition(
                    kind="action", canonical_id=action.canonical_id
                ),
                self.repository.get_action_end_mapping(
                    action_id=action.canonical_id, active_only=True
                ),
            )
        object_items: list[dict[str, Any]] = []
        for object_entry in objects:
            active_grip = self.repository.active_grip_profile_version(
                object_class_id=object_entry.canonical_id
            )
            object_items.append(
                {
                    **self._catalog_entry_summary(object_entry),
                    "active_grip_profile": (
                        self._grip_profile_version_summary(active_grip)
                        if active_grip is not None
                        else None
                    ),
                    "actions": [
                        self._task_flow_hierarchy_action(
                            object_entry=object_entry,
                            action_entry=action_entry,
                            active_grip=active_grip,
                            action_stage=action_components[action_entry.canonical_id][0],
                            mapping=action_components[action_entry.canonical_id][1],
                        )
                        for action_entry in actions
                    ],
                }
            )
        return {
            "selection_policy": "active_grip_x_active_action_then_compose",
            "has_static_object_action_allowlist": False,
            "objects": object_items,
        }

    def _task_flow_hierarchy_action(
        self,
        *,
        object_entry: SemanticCatalogRecord,
        action_entry: SemanticCatalogRecord,
        active_grip: GripProfileVersionRecord | None,
        action_stage: StageDefinitionRecord | None,
        mapping: ActionEndMappingRecord | None,
    ) -> dict[str, Any]:
        blockers: list[str] = []
        if object_entry.status != "active":
            blockers.append("object catalog entry is not active")
        if active_grip is None:
            blockers.append("active, passed GripProfile is missing")
        if action_entry.status != "active":
            blockers.append("action catalog entry is not active")
        if action_stage is None:
            blockers.append("active, passed ActionDefinition is missing")
        if mapping is None:
            blockers.append("active ActionEndMotionMapping is missing")
        return {
            **self._catalog_entry_summary(action_entry),
            "active_action_definition": (
                self._stage_definition_summary(action_stage)
                if action_stage is not None
                else None
            ),
            "mapped_end_motion": (
                self._action_end_mapping_summary(mapping)
                if mapping is not None
                else None
            ),
            "executable": not blockers,
            "blockers": blockers,
        }

    def get_skill_editor_catalog(self) -> dict[str, Any]:
        """Expose only canonical, locally owned primitive schemas and profile ids."""

        registry = get_default_registry()
        motion_profiles = self._motion_profiles()
        force_profiles = self._force_profiles()
        primitives: list[dict[str, Any]] = []
        for operation in registry.operation_names():
            if registry.canonical_operation_name(operation) != operation:
                continue
            metadata = registry.metadata(operation)
            motion_profile_ids = [
                profile_id
                for profile_id, profile in motion_profiles.items()
                if self._editor_motion_profile_supports(operation, profile.motion_kinds)
            ]
            force_profile_ids = (
                sorted(force_profiles)
                if "force_profile_id" in metadata.typed_parameter_schema.get(
                    "properties", {}
                )
                else []
            )
            profile_options = {
                "motion_profile_ids": motion_profile_ids,
                "force_profile_ids": force_profile_ids,
                "recovery_profile_ids": (
                    ["safe_retract_default"]
                    if "recovery_profile_id"
                    in metadata.typed_parameter_schema.get("properties", {})
                    else []
                ),
            }
            primitives.append(
                {
                    **metadata.model_dump(mode="json"),
                    "approved_profile_ids": profile_options,
                    "default_arguments": self._editor_default_arguments(
                        operation,
                        metadata.typed_parameter_schema,
                        profile_options,
                    ),
                }
            )
        return {
            "schema_version": "1.0",
            "primitives": primitives,
            "approved_profiles": {
                "motion_profile_ids": sorted(motion_profiles),
                "force_profile_ids": sorted(force_profiles),
                "recovery_profile_ids": ["safe_retract_default"],
            },
            "constraints": {
                "sequential_only": True,
                "loops_allowed": False,
                "branches_allowed": False,
                "free_form_code_allowed": False,
                "free_form_json_editor_allowed": False,
                "inline_velocity_acceleration_force_allowed": False,
                "candidate_only": True,
                "hardware_compatible": False,
                "generated_grip_action_end": {
                    "allowed_motion_operations": sorted(
                        RECORDING_BLOCK_MOTION_OPERATIONS
                    ),
                    "maximum_action_motion_blocks": (
                        MAXIMUM_ACTION_MOTION_BLOCKS
                    ),
                    "maximum_end_motion_blocks": MAXIMUM_END_MOTION_BLOCKS,
                },
            },
        }

    def preview_skill_editor_blocks(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and compile a sequential graph without writing any artifact."""

        try:
            graph = self._build_editor_skill_graph(request, version="0.1.0-candidate")
            report = SkillGraphValidator().inspect(graph)
            if not report.valid:
                return {
                    "valid": False,
                    "errors": list(report.errors),
                    "warnings": list(report.warnings),
                    "persisted": False,
                }
            compiled = SkillCompiler().compile(graph)
        except (RobotSkillError, ValueError) as error:
            return {
                "valid": False,
                "errors": [str(error)],
                "warnings": [],
                "persisted": False,
            }
        return {
            "valid": compiled.validation_report.valid,
            "errors": [],
            "warnings": list(report.warnings),
            "persisted": False,
            "graph_checksum_sha256": SkillCompiler.graph_checksum(graph),
            "skill_graph": graph.model_dump(mode="json"),
            "normalized_blocks": [
                {
                    "node_id": node.node_id,
                    "operation": node.operation,
                    "arguments": report.normalized_arguments[node.node_id],
                }
                for node in graph.nodes
            ],
        }

    def create_skill_editor_candidate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Persist an independently created sequential block skill as a Candidate."""

        promotion = PromotionPolicy().evaluate_block_candidate(
            operator_confirmed=request.get("acknowledge_mock_only") is True
        )
        if not promotion.eligible:
            raise ValueError(
                "block candidate promotion blocked: "
                + "; ".join(promotion.blockers)
            )
        skill_id = str(request["skill_id"])
        if self._find_version_optional(skill_id) is not None:
            raise ValueError(
                "skill_id already exists; use the checksum-guarded parameter Candidate API"
            )
        version = self._next_editor_candidate_version(skill_id)
        try:
            graph = self._build_editor_skill_graph(request, version=version)
            graph = graph.model_copy(
                update={
                    "uncertainty": {
                        **graph.uncertainty,
                        "promotion_policy": promotion.as_dict(),
                    }
                },
                deep=True,
            )
            row = self._persist_graph(
                graph,
                status="candidate",
                validation_status="pending",
                variant=f"block_{skill_id}",
            )
        except RobotSkillError as error:
            raise ValueError(str(error)) from error
        validation = self.validate_skill(
            skill_id,
            {"version": row.semantic_version, "mode": "mock"},
        )
        created = self._find_version(skill_id, row.semantic_version)
        return {
            "created": True,
            "candidate": self._version_summary(created, include_graph=True),
            "validation": validation,
            "mock_validation_passed": validation["passed"],
            "hardware_compatible": False,
            "promotion_policy": promotion.as_dict(),
        }

    def create_skill_editor_revision_candidate(
        self,
        skill_id: str,
        version: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a full Blockly child revision without mutating its parent graph."""

        promotion = PromotionPolicy().evaluate_block_candidate(
            operator_confirmed=request.get("acknowledge_mock_only") is True
        )
        if not promotion.eligible:
            raise ValueError(
                "block revision Candidate promotion blocked: "
                + "; ".join(promotion.blockers)
            )
        if str(request["skill_id"]) != skill_id:
            raise ValueError("request skill_id must match the parent skill_id")

        parent = self._find_version(skill_id, version)
        expected_checksum = str(request["expected_parent_checksum_sha256"])
        if expected_checksum != parent.graph_checksum_sha256:
            raise ValueError(
                "parent SkillGraph checksum changed; reload before editing"
            )
        parent_graph = SkillGraph.model_validate(parent.graph_json)
        if SkillCompiler.graph_checksum(parent_graph) != parent.graph_checksum_sha256:
            raise ValueError("stored parent SkillGraph checksum verification failed")

        immutable_metadata = {
            "name": parent_graph.name,
            "description": parent_graph.description,
            "skill_type": parent_graph.skill_type.value,
        }
        for field_name, expected_value in immutable_metadata.items():
            supplied_value = request[field_name]
            if hasattr(supplied_value, "value"):
                supplied_value = supplied_value.value
            if str(supplied_value) != expected_value:
                raise ValueError(
                    f"{field_name} cannot be changed in a Blockly child revision"
                )

        candidate_version = self._next_editor_candidate_version(
            skill_id,
            parent_version=parent_graph.version,
        )
        try:
            editor_graph = self._build_editor_skill_graph(
                request,
                version=candidate_version,
            )
            candidate_graph = parent_graph.model_copy(
                update={
                    "version": candidate_version,
                    "parent_version": parent_graph.version,
                    "required_tools": editor_graph.required_tools,
                    "required_entity_roles": editor_graph.required_entity_roles,
                    "bindings": editor_graph.bindings,
                    "nodes": editor_graph.nodes,
                    "edges": [],
                    "start_node": editor_graph.start_node,
                    "terminal_nodes": editor_graph.terminal_nodes,
                    "motion_profiles": editor_graph.motion_profiles,
                    "force_profiles": editor_graph.force_profiles,
                    "uncertainty": {
                        **parent_graph.uncertainty,
                        "authoring_method": "sequential_block_editor_revision",
                        "blockly_revision": {
                            "parent_graph_checksum_sha256": expected_checksum,
                            "original_node_count": len(parent_graph.nodes),
                            "revised_node_count": len(editor_graph.nodes),
                        },
                        "promotion_policy": promotion.as_dict(),
                    },
                    "validation_status": ValidationStatus.UNVALIDATED,
                    "lifecycle_status": SkillLifecycleStatus.CANDIDATE,
                },
                deep=True,
            )
            row = self._persist_graph(
                candidate_graph,
                status="candidate",
                validation_status="pending",
                variant=self._skill_variant(parent),
                parent_version_id=parent.id,
            )
        except RobotSkillError as error:
            raise ValueError(str(error)) from error

        validation = self.validate_skill(
            skill_id,
            {"version": row.semantic_version, "mode": "mock"},
        )
        unchanged_parent = self._find_version(skill_id, parent.semantic_version)
        return {
            "created": True,
            "skill_id": skill_id,
            "parent_version": parent.semantic_version,
            "parent_checksum_sha256": parent.graph_checksum_sha256,
            "parent_unchanged": (
                unchanged_parent.graph_checksum_sha256
                == parent.graph_checksum_sha256
            ),
            "candidate": self._version_summary(
                self._find_version(skill_id, row.semantic_version),
                include_graph=True,
            ),
            "validation": validation,
            "mock_validation_passed": validation["passed"],
            "hardware_compatible": False,
            "promotion_policy": promotion.as_dict(),
        }

    def create_skill_parameter_candidate(
        self,
        skill_id: str,
        version: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a checksum-guarded child graph while leaving its parent untouched."""

        if request.get("acknowledge_mock_only") is not True:
            raise ValueError(
                "parameter Candidate creation requires Mock-only acknowledgement"
            )
        parent = self._find_version(skill_id, version)
        expected_checksum = str(request["expected_parent_checksum_sha256"])
        if expected_checksum != parent.graph_checksum_sha256:
            raise ValueError("parent SkillGraph checksum changed; reload before editing")
        parent_graph = SkillGraph.model_validate(parent.graph_json)
        if SkillCompiler.graph_checksum(parent_graph) != parent.graph_checksum_sha256:
            raise ValueError("stored parent SkillGraph checksum verification failed")

        edits = {
            str(edit["node_id"]): dict(edit["arguments"])
            for edit in request["edits"]
        }
        known_node_ids = {node.node_id for node in parent_graph.nodes}
        unknown_node_ids = sorted(set(edits) - known_node_ids)
        if unknown_node_ids:
            raise ValueError(f"parameter edits reference unknown nodes {unknown_node_ids}")

        registry = get_default_registry()
        updated_nodes: list[SkillNode] = []
        normalized_arguments: list[tuple[str, dict[str, Any]]] = []
        for node in parent_graph.nodes:
            if node.node_id not in edits:
                updated_nodes.append(node.model_copy(deep=True))
                normalized_arguments.append((node.operation, dict(node.arguments)))
                continue
            try:
                validated = registry.validate_operation(
                    node.operation,
                    edits[node.node_id],
                    skill_type=parent_graph.skill_type.value,
                )
            except RobotSkillError as error:
                raise ValueError(str(error)) from error
            arguments = validated.model_dump(mode="json", exclude_none=True)
            self._validate_editor_profiles(node.operation, arguments)
            updated_nodes.append(
                node.model_copy(update={"arguments": arguments}, deep=True)
            )
            normalized_arguments.append((node.operation, arguments))

        placeholder_hints = self._editor_placeholder_hints(
            [arguments for _, arguments in normalized_arguments]
        )
        for placeholder, hint in placeholder_hints.items():
            binding = parent_graph.bindings.get(placeholder)
            if binding is None:
                raise ValueError(
                    f"parameter edits cannot introduce new binding {placeholder!r}"
                )
            if hint is not None and binding.entity_kind is not hint:
                raise ValueError(
                    f"binding {placeholder!r} conflicts with edited anchor/tool type"
                )

        motion_profile_ids, force_profile_ids = self._editor_declared_profiles(
            normalized_arguments
        )
        candidate_version = self._next_editor_candidate_version(
            skill_id,
            parent_version=parent_graph.version,
        )
        candidate_graph = parent_graph.model_copy(
            update={
                "version": candidate_version,
                "parent_version": parent_graph.version,
                "nodes": updated_nodes,
                "motion_profiles": list(
                    dict.fromkeys([*parent_graph.motion_profiles, *motion_profile_ids])
                ),
                "force_profiles": list(
                    dict.fromkeys([*parent_graph.force_profiles, *force_profile_ids])
                ),
                "uncertainty": {
                    **parent_graph.uncertainty,
                    "parameter_candidate": {
                        "parent_graph_checksum_sha256": expected_checksum,
                        "edited_node_ids": sorted(edits),
                    },
                },
                "validation_status": ValidationStatus.UNVALIDATED,
                "lifecycle_status": SkillLifecycleStatus.CANDIDATE,
            },
            deep=True,
        )
        try:
            row = self._persist_graph(
                candidate_graph,
                status="candidate",
                validation_status="pending",
                variant=self._skill_variant(parent),
                parent_version_id=parent.id,
            )
        except RobotSkillError as error:
            raise ValueError(str(error)) from error
        validation = self.validate_skill(
            skill_id,
            {"version": row.semantic_version, "mode": "mock"},
        )
        unchanged_parent = self._find_version(skill_id, parent.semantic_version)
        return {
            "created": True,
            "skill_id": skill_id,
            "parent_version": parent.semantic_version,
            "parent_checksum_sha256": parent.graph_checksum_sha256,
            "parent_unchanged": (
                unchanged_parent.graph_checksum_sha256 == parent.graph_checksum_sha256
            ),
            "candidate": self._version_summary(
                self._find_version(skill_id, row.semantic_version), include_graph=True
            ),
            "validation": validation,
            "mock_validation_passed": validation["passed"],
            "hardware_compatible": False,
        }

    def _build_editor_skill_graph(
        self,
        request: dict[str, Any],
        *,
        version: str,
    ) -> SkillGraph:
        registry = get_default_registry()
        skill_type = SkillType(str(request.get("skill_type", SkillType.COMPOSITE.value)))
        normalized_blocks: list[tuple[str, dict[str, Any]]] = []
        for raw_block in request["blocks"]:
            operation = str(raw_block["operation"])
            try:
                if registry.canonical_operation_name(operation) != operation:
                    raise ValueError(
                        f"compatibility alias {operation!r} is not allowed in new block skills"
                    )
                validated = registry.validate_operation(
                    operation,
                    dict(raw_block.get("arguments", {})),
                    skill_type=skill_type.value,
                )
            except RobotSkillError as error:
                raise ValueError(str(error)) from error
            arguments = validated.model_dump(mode="json", exclude_none=True)
            self._validate_editor_profiles(operation, arguments)
            normalized_blocks.append((operation, arguments))

        supplied_bindings = {
            str(key): BindingSpec.model_validate(value)
            for key, value in dict(request.get("bindings", {})).items()
        }
        bindings = self._editor_bindings(
            [arguments for _, arguments in normalized_blocks],
            supplied_bindings,
        )
        nodes = [
            SkillNode(
                node_id=f"block_{index + 1:03d}",
                operation=operation,
                arguments=arguments,
                on_success=(
                    f"block_{index + 2:03d}"
                    if index + 1 < len(normalized_blocks)
                    else None
                ),
            )
            for index, (operation, arguments) in enumerate(normalized_blocks)
        ]
        motion_profile_ids, force_profile_ids = self._editor_declared_profiles(
            normalized_blocks
        )
        required_tools = sorted(
            {
                binding.class_name
                for binding in bindings.values()
                if binding.entity_kind is EntityKind.TOOL
                and binding.class_name is not None
            }
        )
        required_roles = {
            placeholder: binding.role
            for placeholder, binding in bindings.items()
            if binding.role is not None
        }
        source_recording_ids = [
            str(recording_id) for recording_id in request.get("source_recording_ids", [])
        ]
        source_demonstrations = [
            str(
                self.camera_controller.get_recording_manifest(
                    recording_id, include_frames=False
                )["manifest_uri"]
            )
            for recording_id in source_recording_ids
        ]
        return SkillGraph(
            skill_id=str(request["skill_id"]),
            version=version,
            name=str(request["name"]),
            description=str(request["description"]),
            skill_type=skill_type,
            source_demonstrations=source_demonstrations,
            required_tools=required_tools,
            required_entity_roles=required_roles,
            bindings=bindings,
            nodes=nodes,
            start_node=nodes[0].node_id,
            terminal_nodes=[nodes[-1].node_id],
            motion_profiles=motion_profile_ids,
            force_profiles=force_profile_ids,
            uncertainty={
                "authoring_method": "sequential_block_editor",
                "demonstration_required": False,
                "hardware_compatible": False,
                "source_recording_ids": source_recording_ids,
            },
            validation_status=ValidationStatus.UNVALIDATED,
            lifecycle_status=SkillLifecycleStatus.CANDIDATE,
        )

    def _validate_editor_profiles(
        self,
        operation: str,
        arguments: dict[str, Any],
    ) -> None:
        motion_profile_id = arguments.get("motion_profile_id")
        if isinstance(motion_profile_id, str):
            profiles = self._motion_profiles()
            profile = profiles.get(motion_profile_id)
            if profile is None:
                raise ValueError(
                    f"motion profile {motion_profile_id!r} is not locally approved"
                )
            if not self._editor_motion_profile_supports(
                operation, profile.motion_kinds
            ):
                raise ValueError(
                    f"motion profile {motion_profile_id!r} does not support {operation!r}"
                )
        force_profile_id = arguments.get("force_profile_id")
        if (
            isinstance(force_profile_id, str)
            and force_profile_id not in self._force_profiles()
        ):
            raise ValueError(
                f"force profile {force_profile_id!r} is not locally approved"
            )
        recovery_profile_id = arguments.get("recovery_profile_id")
        if recovery_profile_id not in (None, "safe_retract_default"):
            raise ValueError(
                f"recovery profile {recovery_profile_id!r} is not locally approved"
            )

    @staticmethod
    def _editor_motion_profile_supports(
        operation: str,
        motion_kinds: tuple[str, ...],
    ) -> bool:
        supported_kinds = {
            "motion.move_j": {"move_j"},
            "motion.move_l": {"move_l"},
            "motion.move_c": {"move_c"},
            "motion.move_spline": {"move_spline"},
            "motion.move_periodic": {"move_periodic"},
            "contact.follow_path": {"move_l", "move_spline"},
        }.get(operation)
        return supported_kinds is None or not supported_kinds.isdisjoint(motion_kinds)

    @staticmethod
    def _editor_declared_profiles(
        blocks: list[tuple[str, dict[str, Any]]],
    ) -> tuple[list[str], list[str]]:
        motion_profile_ids = list(
            dict.fromkeys(
                value
                for _, arguments in blocks
                if isinstance((value := arguments.get("motion_profile_id")), str)
            )
        )
        force_profile_ids = list(
            dict.fromkeys(
                value
                for _, arguments in blocks
                if isinstance((value := arguments.get("force_profile_id")), str)
            )
        )
        return motion_profile_ids, force_profile_ids

    @staticmethod
    def _editor_placeholder_hints(
        arguments_list: list[dict[str, Any]],
    ) -> dict[str, EntityKind | None]:
        hints: dict[str, EntityKind | None] = {}
        anchor_type_kinds = {
            "object": EntityKind.OBJECT,
            "tool": EntityKind.TOOL,
            "surface": EntityKind.SURFACE,
            "fixture": EntityKind.SURFACE,
            "workspace_region": EntityKind.WORKSPACE,
        }

        def visit(value: Any, key: str | None = None) -> None:
            if isinstance(value, dict):
                anchor_id = value.get("anchor_id")
                anchor_type = value.get("anchor_type")
                if isinstance(anchor_id, str) and anchor_id.startswith("$"):
                    inferred = (
                        anchor_type_kinds.get(str(anchor_type))
                        if anchor_type is not None
                        else None
                    )
                    previous = hints.get(anchor_id)
                    if previous is not None and inferred is not None and previous is not inferred:
                        raise ValueError(
                            f"binding {anchor_id!r} is used with conflicting anchor types"
                        )
                    hints[anchor_id] = previous or inferred
                for child_key, child in value.items():
                    visit(child, str(child_key))
                return
            if isinstance(value, list):
                for child in value:
                    visit(child, key)
                return
            if not isinstance(value, str) or not value.startswith("$"):
                return
            inferred = (
                {
                    "tool": EntityKind.TOOL,
                    "surface": EntityKind.SURFACE,
                    "region": EntityKind.WORKSPACE,
                    "anchor_id": None,
                }.get(key)
                if key is not None
                else None
            )
            previous = hints.get(value)
            if previous is not None and inferred is not None and previous is not inferred:
                raise ValueError(f"binding {value!r} has conflicting parameter roles")
            hints[value] = previous or inferred

        for arguments in arguments_list:
            visit(arguments)
        return hints

    def _editor_bindings(
        self,
        arguments_list: list[dict[str, Any]],
        supplied: dict[str, BindingSpec],
    ) -> dict[str, BindingSpec]:
        hints = self._editor_placeholder_hints(arguments_list)
        unused = sorted(set(supplied) - set(hints))
        if unused:
            raise ValueError(f"block skill declares unused bindings {unused}")
        bindings: dict[str, BindingSpec] = {}
        conventional = {
            "$tool": EntityKind.TOOL,
            "$surface": EntityKind.SURFACE,
            "$object": EntityKind.OBJECT,
            "$workspace": EntityKind.WORKSPACE,
        }
        for placeholder, hint in hints.items():
            existing = supplied.get(placeholder)
            if existing is not None:
                if existing.variable != placeholder:
                    raise ValueError("binding dictionary key must match binding variable")
                if hint is not None and existing.entity_kind is not hint:
                    raise ValueError(
                        f"binding {placeholder!r} conflicts with its parameter role"
                    )
                bindings[placeholder] = existing
                continue
            entity_kind = hint or conventional.get(placeholder)
            if entity_kind is None:
                raise ValueError(
                    f"binding {placeholder!r} needs an explicit typed binding specification"
                )
            bindings[placeholder] = BindingSpec(
                variable=placeholder,
                entity_kind=entity_kind,
                must_be_attached=True if entity_kind is EntityKind.TOOL else None,
            )
        return bindings

    def _next_editor_candidate_version(
        self,
        skill_id: str,
        *,
        parent_version: str | None = None,
    ) -> str:
        parent = SemanticVersion.parse(parent_version) if parent_version else None
        major = parent.major if parent is not None else 0
        starting_minor = parent.minor + 1 if parent is not None else 1
        for minor in range(starting_minor, starting_minor + 1000):
            candidate = f"{major}.{minor}.0-candidate"
            if self._find_version_optional(skill_id, candidate) is None:
                return candidate
        raise ValueError("no editor Candidate semantic-version slot remains")

    def _editor_default_arguments(
        self,
        operation: str,
        schema: dict[str, Any],
        profile_options: dict[str, list[str]],
    ) -> dict[str, Any]:
        def resolve(item: dict[str, Any]) -> dict[str, Any]:
            reference = item.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                definitions = schema.get("$defs")
                resolved = (
                    definitions.get(reference.rsplit("/", 1)[-1])
                    if isinstance(definitions, dict)
                    else None
                )
                return resolved if isinstance(resolved, dict) else item
            variants = item.get("anyOf")
            if isinstance(variants, list):
                selected = next(
                    (
                        variant
                        for variant in variants
                        if isinstance(variant, dict) and variant.get("type") != "null"
                    ),
                    item,
                )
                return resolve(selected)
            return item

        def default_value(item: dict[str, Any], field_name: str) -> Any:
            resolved = resolve(item)
            if field_name == "target_joint_positions_rad":
                return [0.0] * 6
            if field_name == "motion_profile_id":
                options = profile_options["motion_profile_ids"]
                return options[0] if options else ""
            if field_name == "force_profile_id":
                options = profile_options["force_profile_ids"]
                return options[0] if options else ""
            if field_name == "recovery_profile_id":
                return "safe_retract_default"
            if field_name == "anchor_id":
                return "$surface"
            if field_name == "anchor_type":
                return "surface"
            if field_name == "tool":
                return "$tool"
            if field_name == "surface":
                return "$surface"
            if field_name == "region":
                return "$workspace"
            if "default" in resolved:
                return resolved["default"]
            enum = resolved.get("enum")
            if isinstance(enum, list) and enum:
                return enum[0]
            item_type = resolved.get("type")
            if item_type == "object" or "properties" in resolved:
                required = set(resolved.get("required", []))
                return {
                    key: default_value(child, key)
                    for key, child in resolved.get("properties", {}).items()
                    if key in required or "default" in resolve(child)
                }
            if item_type == "array":
                count = int(resolved.get("minItems", 0))
                return [
                    default_value(dict(resolved.get("items", {})), field_name)
                    for _ in range(count)
                ]
            if item_type == "integer":
                return int(resolved.get("minimum", 1))
            if item_type == "number":
                if field_name == "width_m":
                    return 0.05
                if field_name == "w":
                    return 1.0
                if "exclusiveMinimum" in resolved:
                    return float(resolved["exclusiveMinimum"]) + 1.0
                return float(resolved.get("minimum", 0.0))
            if item_type == "boolean":
                return False
            return ""

        generated = default_value(schema, "arguments")
        arguments: dict[str, Any] = generated if isinstance(generated, dict) else {}
        if operation == "motion.move_periodic":
            arguments.setdefault("amplitude_m", {})["x"] = 0.01
        try:
            validated = get_default_registry().validate_arguments(operation, arguments)
        except RobotSkillError:
            return arguments
        return validated.model_dump(mode="json", exclude_none=True)

    def get_skill(self, skill_id: str, version: str | None = None) -> dict[str, Any]:
        row = self._find_version(skill_id, version)
        return self._version_summary(row, include_graph=True)

    def get_skill_versions(self, skill_id: str) -> dict[str, Any]:
        rows = self._versions(skill_id)
        return {"skill_id": skill_id, "versions": [self._version_summary(row) for row in rows]}

    def compile_skill(self, skill_id: str, request: dict[str, Any]) -> dict[str, Any]:
        row = self._find_version(skill_id, request.get("version"))
        graph = SkillGraph.model_validate(row.graph_json)
        artifact = SkillCompiler().compile(graph)
        stored = self.store.put_text(
            f"skills/{graph.skill_id}/{graph.version}/compiled_skill.py",
            artifact.source,
            media_type="text/x-python; charset=utf-8",
        )
        return {
            "skill_id": skill_id,
            "version": graph.version,
            "compiled_skill_uri": stored.uri,
            "checksum_sha256": stored.checksum_sha256,
            "ast_valid": artifact.validation_report.valid,
            "py_compile_passed": artifact.validation_report.py_compile_passed,
        }

    def validate_skill(self, skill_id: str, request: dict[str, Any]) -> dict[str, Any]:
        row = self._find_version(skill_id, request.get("version"))
        graph = SkillGraph.model_validate(row.graph_json)
        report = SkillGraphValidator().inspect(graph)
        errors = list(report.errors)
        runtime_evidence: dict[str, Any] = {
            "passed": False,
            "robot_commands": [],
            "events": [],
        }
        try:
            compilation = SkillCompiler().compile(graph) if report.valid else None
            if compilation is not None and compilation.validation_report.valid:
                runtime_evidence = self._mock_regression_validate(row, graph)
        except Exception as exc:
            errors.append(f"mock regression failed: {type(exc).__name__}: {exc}")
        passed = report.valid and runtime_evidence["passed"] is True
        with self.database.session() as session:
            attached = session.get(SkillVersionRecord, row.id)
            if attached is None:
                raise KeyError(row.id)
            attached.validation_status = "passed" if passed else "failed"
            if passed and attached.status in {"candidate", "rejected"}:
                attached.status = "validated"
            elif not passed and attached.status == "candidate":
                attached.status = "rejected"
        validation_result = {
            "skill_id": skill_id,
            "version": row.semantic_version,
            "passed": passed,
            "mock_validation": True,
            "hardware_validated": False,
            "errors": errors,
            "warnings": list(report.warnings),
            "runtime": runtime_evidence,
        }
        self.repository.record_validation_run(
            skill_version_id=row.id,
            status="passed" if passed else "failed",
            result=validation_result,
        )
        return validation_result

    def activate_skill(self, skill_id: str, request: dict[str, Any]) -> dict[str, Any]:
        row = self._find_version(skill_id, str(request["version"]))
        if row.validation_status != "passed":
            raise ValueError("only a passed version can be activated")
        if row.semantic_version.endswith("-candidate"):
            candidate_graph = SkillGraph.model_validate(row.graph_json)
            promoted_version = stable_version(row.semantic_version)
            promoted_graph = candidate_graph.model_copy(
                update={
                    "version": promoted_version,
                    "parent_version": candidate_graph.version,
                    "validation_status": ValidationStatus.PENDING,
                    "lifecycle_status": SkillLifecycleStatus.CANDIDATE,
                },
                deep=True,
            )
            promoted = self._persist_graph(
                promoted_graph,
                status="candidate",
                validation_status="pending",
                variant=self._skill_variant(row),
                parent_version_id=row.id,
            )
            validation = self.validate_skill(
                skill_id, {"version": promoted.semantic_version, "mode": "mock"}
            )
            if not validation["passed"]:
                raise ValueError("promoted stable version failed mock regression validation")
            return self.activate_skill(
                skill_id, {"version": promoted.semantic_version}
            )
        if row.status not in {"validated", "retired", "active"}:
            raise ValueError("only validated or previously active stable versions can activate")
        with self.database.session() as session:
            attached = session.get(SkillVersionRecord, row.id)
            if attached is None:
                raise KeyError(row.id)
            skill = session.get(SkillRecord, attached.skill_id)
            if skill is None:
                raise KeyError(attached.skill_id)
            self._retire_current_in_session(session, skill, except_version_id=attached.id)
            attached.status = "active"
            skill.active_version_id = attached.id
        return self._version_summary(self._find_version(skill_id, row.semantic_version))

    def rollback_skill(self, skill_id: str, request: dict[str, Any]) -> dict[str, Any]:
        target = self._find_version(skill_id, str(request["version"]))
        if target.validation_status != "passed":
            raise ValueError("rollback target must have passed validation")
        if target.semantic_version.endswith("-candidate") or target.status not in {
            "active",
            "retired",
        }:
            raise ValueError("rollback target must be a previously active stable version")
        with self.database.session() as session:
            attached = session.get(SkillVersionRecord, target.id)
            if attached is None:
                raise KeyError(target.id)
            skill = session.get(SkillRecord, attached.skill_id)
            if skill is None:
                raise KeyError(attached.skill_id)
            self._retire_current_in_session(session, skill, except_version_id=attached.id)
            attached.status = "active"
            skill.active_version_id = attached.id
        updated = self._find_version(skill_id, target.semantic_version)
        return {"rolled_back": True, **self._version_summary(updated)}

    def update_skill(self, skill_id: str, request: dict[str, Any]) -> dict[str, Any]:
        demo_path = self._safe_demo_path(str(request["demo_path"]))
        evidence = self._demonstration_evidence(demo_path)
        parent = self._find_version(skill_id, request.get("base_version"))
        parent_graph = SkillGraph.model_validate(parent.graph_json)
        has_force = bool(request.get("has_force_measurements", False))
        operator_role = str(
            request.get("operator_role")
            or evidence.trajectory.operator_role
            or "operator"
        )
        is_measured_expert = operator_role == "expert" and evidence.quality.confidence >= 0.9
        normalized_path = self._normalized_path(evidence.processed)
        proposal = SkillUpdater().propose(
            parent_graph,
            UpdateEvidence(
                source_demonstration_id=self._portable_artifact_reference(demo_path),
                operator_role=operator_role,
                operator_style="expert_precise" if is_measured_expert else "normal",
                normalized_path_m=normalized_path,
                timing_motion_profile_id="linear_expert" if is_measured_expert else None,
                has_robot_force_log=has_force,
                node_argument_updates=self._wipe_path_updates(
                    parent_graph, normalized_path
                ),
                quality_score=evidence.quality.confidence,
            ),
            parent_normalized_path_m=self._parent_normalized_path(parent_graph),
        )
        candidate_graph = proposal.candidate_graph
        if evidence.promotion_warnings:
            candidate_graph = candidate_graph.model_copy(
                update={
                    "uncertainty": {
                        **candidate_graph.uncertainty,
                        "promotion_warnings": list(evidence.promotion_warnings),
                    }
                },
                deep=True,
            )
        version = self._persist_graph(
            candidate_graph,
            status="candidate",
            validation_status="pending",
            variant=proposal.variant,
            parent_version_id=parent.id,
        )
        return {
            "skill_id": skill_id,
            "parent_version": parent.semantic_version,
            "candidate_version": version.semantic_version,
            "status": version.status,
            "variant": self._skill_variant(version),
            "force_profile_preserved": proposal.force_profile_preserved,
            "components": [component.value for component in proposal.components],
            "promotion_warnings": list(evidence.promotion_warnings),
        }

    def resolve_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = self._scene(str(request["scene_id"])) if request.get("scene_id") else None
        active_objects = self.repository.list_catalog_entries(
            kind="object", status="active"
        )
        active_actions = self.repository.list_catalog_entries(
            kind="action", status="active"
        )
        if active_objects or active_actions:
            if not active_objects or not active_actions:
                raise ValueError(
                    "task-flow runtime requires both active object and action catalogs"
                )
            if scene is None:
                raise ValueError("task-flow runtime resolution requires a current scene_id")
            return self._resolve_task_flow_intent(
                text=str(request["text"]),
                scene=scene,
                active_objects=active_objects,
                active_actions=active_actions,
            )
        entity_ids: list[str] = []
        if scene is not None:
            entity_ids = [
                *[item.instance_id for item in scene.objects],
                *[item.instance_id for item in scene.tools],
                *[item.instance_id for item in scene.surfaces],
            ]
        intent, metadata = RuntimeIntentResolver(
            self.settings,
            tool_dispatcher=self._function_dispatcher(scene),
        ).resolve(
            str(request["text"]),
            entity_ids=entity_ids,
            approved_motion_profiles=list(self._motion_profiles()),
            approved_force_profiles=list(self._force_profiles()),
        )
        local_match = self._local_command_skill_match(str(request["text"]))
        if local_match is not None:
            matched_row, matched_name = local_match
            intent = intent.model_copy(
                update={
                    "skill_query": str(matched_row.graph_json["skill_id"]),
                    "tool_query": None,
                    "target_query": None,
                    "speed_profile_request": None,
                    "force_profile_request": None,
                    "confidence": 1.0,
                    "unresolved_ambiguities": [],
                    "ambiguity": False,
                    "confidence_rationale": "exact local command phrase matched an active skill",
                },
                deep=True,
            )
            return {
                "intent": intent.model_dump(mode="json"),
                "trace": metadata.model_dump(mode="json"),
                "skill_candidates": {
                    "mode": "local_command_phrase",
                    "results": [
                        {
                            "skill_id": str(matched_row.graph_json["skill_id"]),
                            "version": matched_row.semantic_version,
                            "name": matched_name,
                            "score": 1.0,
                            "factors": {
                                "local_command_phrase": 1.0,
                                "embedding_used": False,
                            },
                        }
                    ],
                },
                "resolver_mode": "local_command_phrase",
            }
        matches = self.search_skills(
            {
                "query": intent.skill_query or intent.intent,
                "intent": intent.intent,
                "tool_class": "wiper" if intent.tool_query else None,
                "target_type": "contact_target" if intent.target_query else None,
                "style": intent.style.value,
                "limit": 5,
            }
        )
        return {
            "intent": intent.model_dump(mode="json"),
            "trace": metadata.model_dump(mode="json"),
            "skill_candidates": matches,
            "resolver_mode": "legacy_monolithic_compatibility",
        }

    def _local_command_skill_match(
        self, text: str
    ) -> tuple[SkillVersionRecord, str] | None:
        """Match an exact normalized command phrase embedded in one active skill.

        This deterministic path is intentionally narrow: short fragments do not match, and
        ambiguity returns no result so the normal semantic resolver remains authoritative.
        """

        normalized_text = self._normalize_command_phrase(text)
        if len(normalized_text) < 4:
            return None
        with self.database.session() as session:
            rows = list(
                session.execute(
                    select(SkillVersionRecord, SkillRecord)
                    .join(SkillRecord, SkillVersionRecord.skill_id == SkillRecord.id)
                    .where(
                        SkillRecord.active_version_id == SkillVersionRecord.id,
                        SkillVersionRecord.status == "active",
                        SkillVersionRecord.validation_status == "passed",
                    )
                ).tuples()
            )
        matches: list[tuple[SkillVersionRecord, str]] = []
        for version, skill in rows:
            graph = SkillGraph.model_validate(version.graph_json)
            uncertainty_aliases = graph.uncertainty.get("command_aliases", [])
            aliases = (
                [str(value) for value in uncertainty_aliases]
                if isinstance(uncertainty_aliases, list)
                else []
            )
            searchable = [graph.name, graph.description, skill.name, skill.description, *aliases]
            if any(
                normalized_text in self._normalize_command_phrase(value)
                for value in searchable
            ):
                matches.append((version, skill.name))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _normalize_command_phrase(value: str) -> str:
        return "".join(character.casefold() for character in value if character.isalnum())

    @staticmethod
    def _task_scene_entities(
        scene: SceneSnapshot,
    ) -> dict[str, dict[str, str]]:
        entities: dict[str, dict[str, str]] = {}
        for object_item in scene.objects:
            entities[object_item.instance_id] = {
                "kind": "object",
                "class_or_role": object_item.class_name,
            }
        for tool_item in scene.tools:
            entities[tool_item.instance_id] = {
                "kind": "tool",
                "class_or_role": tool_item.tool_class,
            }
        for surface_item in scene.surfaces:
            entities[surface_item.instance_id] = {
                "kind": "surface",
                "class_or_role": surface_item.role.value,
            }
        for region_item in scene.workspace_regions:
            entities[region_item.region_id] = {
                "kind": "workspace_region",
                "class_or_role": region_item.role.value,
            }
        return entities

    def _resolve_task_flow_intent(
        self,
        *,
        text: str,
        scene: SceneSnapshot,
        active_objects: list[SemanticCatalogRecord],
        active_actions: list[SemanticCatalogRecord],
    ) -> dict[str, Any]:
        object_catalog = {
            item.canonical_id: tuple(
                dict.fromkeys([item.display_name, *item.aliases_json])
            )
            for item in active_objects
        }
        action_catalog = {
            item.canonical_id: tuple(
                dict.fromkeys([item.display_name, *item.aliases_json])
            )
            for item in active_actions
        }
        stages = {
            item.canonical_id: self.repository.active_stage_definition(
                kind="action", canonical_id=item.canonical_id
            )
            for item in active_actions
        }
        if any(stage is None for stage in stages.values()):
            raise ValueError("active action catalog contains no active ActionDefinition")
        required_roles = {
            action_id: tuple(stage.required_roles_json)
            for action_id, stage in stages.items()
            if stage is not None
        }
        intent, metadata = TaskIntentResolver(self.settings).resolve(
            text,
            object_catalog=object_catalog,
            action_catalog=action_catalog,
            scene_entities=self._task_scene_entities(scene),
            required_roles=required_roles,
        )
        grip = self.repository.active_grip_profile_version(
            object_class_id=intent.object_class_id
        )
        action_stage = stages[intent.action_id]
        mapping = self.repository.get_action_end_mapping(
            action_id=intent.action_id, active_only=True
        )
        blockers: list[str] = []
        if grip is None:
            blockers.append("active, passed GripProfile is missing")
        if action_stage is None:
            blockers.append("active, passed ActionDefinition is missing")
        if mapping is None:
            blockers.append("active ActionEndMotionMapping is missing")
        if intent.ambiguity:
            blockers.extend(intent.unresolved_ambiguities or ["task intent is ambiguous"])
        selected = {
            "grip": (
                self._grip_profile_version_summary(grip)
                if grip is not None
                else None
            ),
            "action": (
                self._stage_definition_summary(action_stage)
                if action_stage is not None
                else None
            ),
            "end_motion_mapping": (
                self._action_end_mapping_summary(mapping)
                if mapping is not None
                else None
            ),
        }
        task_flow_manifest: dict[str, Any] | None = None
        results: list[dict[str, Any]] = []
        if not blockers:
            try:
                materialized = TaskFlowMaterializer(
                    self.repository, self.store
                ).materialize(
                    object_class_id=intent.object_class_id,
                    action_id=intent.action_id,
                    resolved_role_bindings=intent.role_bindings,
                    require_hardware_compatible=False,
                )
                row = self._persist_graph(
                    materialized.graph,
                    status="active",
                    validation_status="passed",
                    variant="task_flow_composite",
                    index_for_legacy_retrieval=False,
                )
                suggested_bindings = dict(intent.role_bindings)
                if intent.object_instance_id is not None:
                    suggested_bindings["$object"] = intent.object_instance_id
                preflight = self._preflight_materialized_flow(
                    row=row,
                    graph=materialized.graph,
                    scene=scene,
                    binding_hints=suggested_bindings,
                )
                preflight_blockers = [
                    f"{item['name']}: {item.get('detail') or 'failed'}"
                    for item in preflight["checks"]
                    if not item["passed"]
                ]
                blockers.extend(preflight_blockers)
                storage_plan = self.repository.record_task_flow_plan(
                    grip_profile_version_id=(
                        materialized.selection.grip_profile_version_id
                    ),
                    action_end_mapping_id=(
                        materialized.selection.action_end_mapping_record_id
                    ),
                    composer_version=materialized.manifest.composer_version,
                    composite_skill_version_id=row.id,
                    validation_status=(
                        "passed" if preflight["passed"] else "failed"
                    ),
                    hardware_compatible=False,
                    metadata={
                        "composer_plan_checksum_sha256": (
                            materialized.manifest.plan_checksum_sha256
                        )
                    },
                )
                task_flow_manifest = {
                    **materialized.manifest.model_dump(mode="json"),
                    "storage_plan_id": storage_plan.id,
                    "storage_plan_checksum_sha256": (
                        storage_plan.plan_checksum_sha256
                    ),
                    "selection": materialized.selection.model_dump(mode="json"),
                    "preflight": preflight,
                }
                results.append(
                    {
                        "skill_id": materialized.graph.skill_id,
                        "version": materialized.graph.version,
                        "name": materialized.graph.name,
                        "score": 1.0,
                        "factors": {
                            "selection": "exact active Grip + Action + mapped End",
                            "embedding_used": False,
                        },
                        "bindings": suggested_bindings,
                        "executable": preflight["passed"],
                        "blockers": preflight_blockers,
                    }
                )
            except (
                RobotSkillError,
                RuntimeErrorBase,
                ValueError,
            ) as exc:
                blockers.append(f"composition rejected: {exc}")
        return {
            "intent": intent.model_dump(mode="json"),
            "trace": metadata.model_dump(mode="json"),
            "resolver_mode": "grip_action_end_hierarchical",
            "selected_components": selected,
            "task_flow_manifest": task_flow_manifest,
            "skill_candidates": {
                "mode": "hierarchical_component_selection",
                "results": results,
                "blockers": blockers,
            },
        }

    def _preflight_materialized_flow(
        self,
        *,
        row: SkillVersionRecord,
        graph: SkillGraph,
        scene: SceneSnapshot,
        binding_hints: dict[str, str],
    ) -> dict[str, Any]:
        requirements = [
            requirement.model_copy(
                update={"instance_id": binding_hints.get(requirement.variable)}
            )
            if requirement.variable in binding_hints
            else requirement
            for requirement in graph.binding_requirements()
        ]
        bindings = EntityBinder().bind_entities(
            scene,
            requirements,
            maximum_scene_age_ms=self.settings.scene_freshness_ms,
        )
        robot = MockRobotAdapter()
        robot.connect()
        report = PreflightValidator().validate(
            scene=scene,
            skill=graph,
            bindings=bindings,
            robot=robot,
            execution_mode=ExecutionMode.MOCK,
            skill_validation_status=row.validation_status,
            expected_calibration_id=scene.calibration_id,
            expected_tool_class=(
                graph.required_tools[0] if graph.required_tools else None
            ),
            skill_uses_force=bool(graph.force_profiles),
            motion_profiles=self._motion_profiles(),
            force_profiles=self._force_profiles(),
            verification_profiles=self._verification_profiles(),
            safety_policy=self._safety_policy(),
            robot_backend="mock",
            enable_real_robot=False,
            dry_run=True,
            raise_on_failure=False,
        )
        return report.as_dict()

    def _scene_with_graph_task_plane(
        self, scene: SceneSnapshot, graph: SkillGraph
    ) -> SceneSnapshot:
        """Attach the graph's exact calibrated task plane to a fresh Scene."""

        uncertainty = graph.uncertainty
        anchor_id = uncertainty.get("task_plane_anchor_id")
        camera_transform_payload = uncertainty.get("camera_to_task_plane")
        source_frame = uncertainty.get("task_plane_source_frame")
        if not isinstance(anchor_id, str) or not isinstance(
            camera_transform_payload, dict
        ):
            return scene
        transform_payload: dict[str, Any] | None = None
        transform_source = "camera_relative_task_plane"
        base_chain = uncertainty.get("base_chain")
        if (
            scene.reference_frame in {"base", "base_link", "robot_base"}
            and isinstance(base_chain, dict)
            and (
                base_chain.get("verified") is True
                or base_chain.get("operator_fixed_workspace_reuse") is True
            )
            and isinstance(base_chain.get("base_to_task_plane"), dict)
        ):
            transform_payload = base_chain["base_to_task_plane"]
            transform_source = "verified_base_task_plane_chain"
        elif scene.reference_frame == source_frame:
            transform_payload = camera_transform_payload
        if transform_payload is None:
            return scene
        transform = RigidTransform.model_validate(transform_payload)
        normal = rotate_vector(
            transform.rotation_xyzw,
            Vector3(x=0.0, y=0.0, z=1.0),
        )
        allowed_contact_links = sorted(
            {
                link
                for surface in scene.surfaces
                for link in surface.allowed_contact_links
            }
        )
        task_plane = SurfaceInstance(
            instance_id=anchor_id,
            role=SurfaceRole.CONTACT_TARGET,
            pose=Pose(
                frame_id=scene.reference_frame,
                position_m=transform.translation_m,
                orientation_xyzw=transform.rotation_xyzw,
                timestamp_ns=scene.timestamp_ns,
                source=transform_source,
                confidence=0.99,
            ),
            center_m=transform.translation_m,
            normal=normal,
            boundary_m=[],
            confidence=0.99,
            allowed_contact_links=allowed_contact_links,
            material="operator_confirmed_task_plane",
        )
        surfaces = [
            surface for surface in scene.surfaces if surface.instance_id != anchor_id
        ]
        surfaces.append(task_plane)
        workspace_regions = list(scene.workspace_regions)
        observed_path = uncertainty.get("observed_trajectory_task_plane_m")
        if isinstance(observed_path, list) and observed_path:
            observed_points: list[Vector3] = []
            for value in observed_path:
                if not isinstance(value, list) or len(value) != 3:
                    observed_points = []
                    break
                relative = Vector3(
                    x=float(value[0]), y=float(value[1]), z=float(value[2])
                )
                rotated = rotate_vector(transform.rotation_xyzw, relative)
                observed_points.append(
                    Vector3(
                        x=transform.translation_m.x + rotated.x,
                        y=transform.translation_m.y + rotated.y,
                        z=transform.translation_m.z + rotated.z,
                    )
                )
            # These narrow boxes are derived from the operator-confirmed RGB-D
            # trajectory and cover only each measured segment. They extend Mock
            # observed-space evidence; collision volumes and hardware gates remain
            # unchanged and are still evaluated independently.
            segment_points = (
                list(zip(observed_points, observed_points[1:], strict=False))
                if len(observed_points) >= 2
                else [(observed_points[0], observed_points[0])]
                if observed_points
                else []
            )
            for index, (start, end) in enumerate(segment_points):
                padding_m = 0.002
                workspace_regions.append(
                    WorkspaceRegion(
                        region_id=f"{anchor_id}_observed_segment_{index:03d}",
                        role=WorkspaceRole.FREE_SPACE,
                        geometry={
                            "type": "box",
                            "center_m": [
                                (start.x + end.x) / 2.0,
                                (start.y + end.y) / 2.0,
                                (start.z + end.z) / 2.0,
                            ],
                            "size_m": [
                                max(abs(end.x - start.x), padding_m),
                                max(abs(end.y - start.y), padding_m),
                                max(abs(end.z - start.z), padding_m),
                            ],
                        },
                        frame_id=scene.reference_frame,
                        minimum_clearance_m=0.0,
                        access_policy=AccessPolicy.ALLOWED,
                        confidence=min(
                            0.99,
                            float(
                                (uncertainty.get("trajectory_quality") or {}).get(
                                    "mean_confidence", 0.0
                                )
                            ),
                        ),
                    )
                )
        return scene.model_copy(
            update={
                "surfaces": surfaces,
                "workspace_regions": workspace_regions,
                "calibration_id": str(
                    uncertainty.get("task_plane_calibration_id")
                    or scene.calibration_id
                ),
            },
            deep=True,
        )

    def bind_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        row = self._find_version(str(request["skill_id"]), request.get("version"))
        graph = SkillGraph.model_validate(row.graph_json)
        if str(request.get("mode", "mock")) == "hardware":
            _, base_to_plane = self._acquire_fixed_hardware_session()
            graph = self._graph_with_aruco_base_plane(graph, base_to_plane)
        scene = self._scene_with_graph_task_plane(
            self._scene(str(request["scene_id"])), graph
        )
        scene = self._scene_with_mock_binding_fixtures(scene, graph)
        hints = dict(request.get("entity_hints", {}))
        requirements = [
            requirement.model_copy(update={"instance_id": hints.get(requirement.variable)})
            if requirement.variable in hints
            else requirement
            for requirement in graph.binding_requirements()
        ]
        bindings = EntityBinder().bind_entities(
            scene,
            requirements,
            maximum_scene_age_ms=self.settings.scene_freshness_ms,
        )
        return {
            "skill_id": graph.skill_id,
            "version": graph.version,
            "scene_id": scene.scene_id,
            "bindings": {name: value.entity_id for name, value in bindings.items()},
        }

    def preflight_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = ExecutionMode(str(request.get("mode", "mock")))
        row, graph, scene, bindings, robot, motion, force = self._runtime_parts(request)
        safety_policy = self._safety_policy()
        geometry_validator = (
            DoosanFixedPlaneGeometryValidator(robot, minimum_clearance_m=0.0)
            if mode is ExecutionMode.HARDWARE
            else None
        )
        preflight_policy = (
            PreflightPolicy(maximum_scene_age_ms=1_800_000, minimum_clearance_m=0.0)
            if mode is ExecutionMode.HARDWARE
            else None
        )
        report = PreflightValidator(
            geometry_validator=geometry_validator, policy=preflight_policy
        ).validate(
            scene=scene,
            skill=graph,
            bindings=bindings,
            robot=robot,
            execution_mode=mode,
            enable_hardware_execution=self.settings.hardware_enabled,
            skill_validation_status=row.validation_status,
            expected_calibration_id=scene.calibration_id,
            expected_tool_class=graph.required_tools[0] if graph.required_tools else None,
            skill_uses_force=bool(graph.force_profiles),
            motion_profiles=motion,
            force_profiles=force,
            verification_profiles=self._verification_profiles(),
            safety_policy=safety_policy,
            robot_backend=self.settings.robot_backend,
            enable_real_robot=self.settings.enable_real_robot,
            dry_run=self.settings.dry_run,
            hardware_workspace_monitor_verified=mode is ExecutionMode.HARDWARE,
            hardware_scene_monitor_verified=mode is ExecutionMode.HARDWARE,
        )
        return report.as_dict()

    def execute_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.jog_controller.enabled:
            raise ValueError("stop and disable jog before executing a skill")
        requested_mode = str(request.get("mode", "mock"))
        mock_override = request.get("mock_override")
        if mock_override is not None:
            if requested_mode != "mock":
                raise ValueError("validation override is available only in mode=mock")
            if (
                mock_override.get("override_all_overridable") is not True
                or mock_override.get("acknowledge_mock_only") is not True
                or not str(mock_override.get("operator_id") or "").strip()
                or not str(mock_override.get("reason") or "").strip()
            ):
                raise ValueError(
                    "Mock override requires operator, reason, and explicit acknowledgements"
                )
        if requested_mode == "hardware" and not self.settings.hardware_enabled:
            raise ValueError("hardware execution gates are not all enabled")
        row, graph, scene, bindings, robot, motion, force = self._runtime_parts(
            request, allow_mock_candidate=mock_override is not None
        )
        safety_policy = self._safety_policy()
        geometry_validator = (
            DoosanFixedPlaneGeometryValidator(robot, minimum_clearance_m=0.0)
            if requested_mode == "hardware"
            else None
        )
        preflight_policy = (
            PreflightPolicy(maximum_scene_age_ms=1_800_000, minimum_clearance_m=0.0)
            if requested_mode == "hardware"
            else None
        )
        report = PreflightValidator(
            geometry_validator=geometry_validator, policy=preflight_policy
        ).validate(
            scene=scene,
            skill=graph,
            bindings=bindings,
            robot=robot,
            execution_mode=ExecutionMode(requested_mode),
            skill_validation_status=(
                "passed" if mock_override is not None else row.validation_status
            ),
            expected_tool_class=graph.required_tools[0] if graph.required_tools else None,
            skill_uses_force=bool(graph.force_profiles),
            motion_profiles=motion,
            force_profiles=force,
            verification_profiles=self._verification_profiles(),
            safety_policy=safety_policy,
            enable_hardware_execution=self.settings.hardware_enabled,
            robot_backend=self.settings.robot_backend,
            enable_real_robot=self.settings.enable_real_robot,
            dry_run=self.settings.dry_run,
            hardware_workspace_monitor_verified=requested_mode == "hardware",
            hardware_scene_monitor_verified=requested_mode == "hardware",
        )
        run = self._load_verified_run(row, SkillGraph.model_validate(row.graph_json))
        gripper = (
            OnRobotRG2Adapter(
                host=self.settings.rg2_modbus_host,
                port=self.settings.rg2_modbus_port,
                unit_id=self.settings.rg2_modbus_unit_id,
                force_n=self.settings.rg2_grip_force_n,
                open_width_m=self.settings.rg2_open_width_m,
                closed_width_m=self.settings.rg2_closed_width_m,
                execution_mode="hardware",
                hardware_enabled=self.settings.hardware_enabled,
            )
            if requested_mode == "hardware"
            else MockGripperAdapter()
        )
        gripper.connect()
        event_sink = InMemoryEventSink()
        if mock_override is not None:
            bypassed_check_ids = [
                check_id
                for check_id, bypassed
                in (
                    ("component_lifecycle_active", row.status != "active"),
                    ("skill_validation", row.validation_status != "passed"),
                )
                if bypassed
            ]
            event_sink.record(
                "mock_validation_override",
                {
                    "operator_id": str(mock_override["operator_id"]),
                    "reason": str(mock_override["reason"]),
                    "override_all_overridable": True,
                    "acknowledge_mock_only": True,
                    "bypassed_check_ids": bypassed_check_ids,
                    "component_activation_changed": False,
                    "hardware_compatibility_changed": False,
                },
                severity="warning",
            )
        context = RuntimeContext(
            scene=scene,
            bindings=bindings,
            motion_profiles=motion,
            force_profiles=force,
            verification_profiles=self._verification_profiles(),
            execution_mode=ExecutionMode(requested_mode),
            skill=graph,
            safety_policy=safety_policy,
            command_text=request.get("text"),
            preflight_report=report,
        )
        contact_links = tuple(
            link for surface in scene.surfaces for link in surface.allowed_contact_links
        )
        force_supervisor = GlobalForceSupervisor(
            robot,
            force,
            allowed_contact_links=contact_links,
            maximum_force_n=safety_policy.maximum_force_n,
        )
        safety_supervisor = GlobalSafetySupervisor()
        workspace_monitor = (
            DoosanStateMonitor(robot) if requested_mode == "hardware" else None
        )
        scene_monitor = (
            FixedReferenceSceneMonitor(maximum_scene_age_ms=1_800_000)
            if requested_mode == "hardware"
            else None
        )
        executor = RuntimeExecutor(
            context=context,
            robot=robot,
            gripper=gripper,
            binder=EntityBinder(),
            workspace_supervisor=GlobalWorkspaceSupervisor(
                monitor=workspace_monitor,
                minimum_clearance_m=safety_policy.minimum_clearance_m
            ),
            force_supervisor=force_supervisor,
            safety_supervisor=safety_supervisor,
            primitive_registry=get_default_registry(),
            event_sink=event_sink,
            scene_monitor=scene_monitor,
        )
        execution = self.repository.start_execution(
            started_at_ns=time.time_ns(),
            execution_mode=requested_mode,
            execution_run_id=request.get("run_id"),
            skill_version_id=row.id,
            scene_id=scene.scene_id,
            command_text=request.get("text"),
            preflight=report.as_dict(),
            bindings={name: value.entity_id for name, value in bindings.items()},
        )
        with self._active_execution_lock:
            if execution.id in self._active_executions:
                raise ValueError(f"execution run {execution.id!r} is already active")
            self._active_executions[execution.id] = ActiveExecution(
                robot=robot,
                force_supervisor=force_supervisor,
                safety_supervisor=safety_supervisor,
            )
        error: BaseException | None = None
        try:
            asyncio.run(executor.execute_compiled(run))
        except BaseException as exc:
            error = exc
        finally:
            with self._active_execution_lock:
                active = self._active_executions.pop(execution.id, None)
                if (
                    error is None
                    and active is not None
                    and active.safety_supervisor.abort_requested
                ):
                    error = ExecutionAbortedError(
                        active.safety_supervisor.abort_reason or "operator_request"
                    )
            for event in event_sink.events:
                self.repository.append_execution_event(
                    execution.id,
                    timestamp_ns=event.timestamp_ns,
                    event_type=event.event_type,
                    details=event.details,
                    severity=event.severity,
                )
            self.repository.finish_execution(
                execution.id,
                status="succeeded" if error is None else "failed",
                ended_at_ns=time.time_ns(),
                error_code=None if error is None else type(error).__name__,
                error_message=None if error is None else str(error),
            )
            try:
                gripper.disconnect()
            finally:
                if requested_mode == "hardware":
                    robot.disconnect()
        if error is not None:
            raise error
        return {
            "run_id": execution.id,
            "status": "succeeded",
            "mode": requested_mode,
            "preflight": report.as_dict(),
            "bindings": {name: value.entity_id for name, value in bindings.items()},
            "robot_commands": [command.operation for command in robot.commands],
            "events": [event.event_type for event in event_sink.events],
        }

    def abort_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        run_id = str(request["run_id"])
        reason = str(request.get("reason", "operator_request"))
        with self._active_execution_lock:
            active = self._active_executions.get(run_id)
            if active is not None:
                # Mark the request while holding the same lifecycle lock used by
                # execution finalization.  This makes abort-vs-completion races
                # deterministic: either this run fails as aborted or it has
                # already left the active set and is reported as completed.
                active.safety_supervisor.request_abort(reason)
        if active is None:
            with self.database.session() as session:
                known = session.get(ExecutionRunRecord, run_id)
            if known is None:
                raise KeyError(f"unknown execution run {run_id!r}")
            return {
                "run_id": run_id,
                "abort_requested": False,
                "reason": reason,
                "status": known.status,
            }
        active.robot.stop(reason=f"operator_abort:{reason}")
        if active.force_supervisor.force_active:
            direction = active.force_supervisor.emergency_release()
            if direction is not None:
                active.robot.safe_retract(direction_xyz=direction, distance_m=0.05)
        return {"run_id": run_id, "abort_requested": True, "reason": reason}

    def get_runtime_run(self, run_id: str) -> dict[str, Any]:
        with self.database.session() as session:
            row = session.get(ExecutionRunRecord, run_id)
            if row is None:
                raise KeyError(f"unknown execution run {run_id!r}")
            return {
                "run_id": row.id,
                "status": row.status,
                "mode": row.execution_mode,
                "scene_id": row.scene_id,
                "skill_version_id": row.skill_version_id,
                "preflight": row.preflight_json,
                "bindings": row.bindings_json,
                "events": [
                    {
                        "sequence": event.sequence,
                        "timestamp_ns": event.timestamp_ns,
                        "event_type": event.event_type,
                        "severity": event.severity,
                        "details": event.details_json,
                    }
                    for event in row.events
                ],
            }

    def _runtime_parts(
        self,
        request: dict[str, Any],
        *,
        allow_mock_candidate: bool = False,
    ) -> tuple[
        SkillVersionRecord,
        SkillGraph,
        SceneSnapshot,
        dict[str, Any],
        Any,
        dict[str, Any],
        dict[str, Any],
    ]:
        mode = str(request.get("mode", "mock"))
        row = self._find_version(str(request["skill_id"]), request.get("version"))
        if row.status != "active" or row.validation_status != "passed":
            hardware_validated_candidate = (
                mode == "hardware"
                and row.status == "validated"
                and row.validation_status == "passed"
            )
            if not allow_mock_candidate:
                if not hardware_validated_candidate:
                    raise ValueError(
                        "runtime accepts active skills or passed validated Candidates"
                    )
            elif row.status not in {"candidate", "validated"}:
                raise ValueError(
                    "Mock override cannot run rejected or retired skill versions"
                )
            if row.validation_status == "failed":
                raise ValueError(
                    "Mock override cannot bypass a failed deterministic validation"
                )
        verify_skill_checksum(row.graph_json, row.graph_checksum_sha256)
        graph = SkillGraph.model_validate(row.graph_json)
        SkillGraphValidator().validate(graph)
        SkillCompiler().compile(graph)
        fixed_workspace_orientation_xyzw: tuple[float, float, float, float] | None = None
        initial_hardware_pose: BoundTargetPose | None = None
        if mode == "hardware":
            aruco_robot, base_to_plane = self._acquire_fixed_hardware_session()
            initial_tcp_matrix = aruco_robot.get_base_to_tcp_matrix()
            fixed_workspace_orientation_xyzw = rotation_matrix_to_quaternion_xyzw(
                initial_tcp_matrix[:3, :3]
            )
            initial_hardware_pose = BoundTargetPose(
                frame_id="base",
                position_m=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in initial_tcp_matrix[:3, 3]),
                ),
                orientation_xyzw=fixed_workspace_orientation_xyzw,
                anchor_entity_id="fixed_workspace_run_start_tcp",
                timestamp_ns=time.time_ns(),
            )
            graph = self._graph_with_aruco_base_plane(graph, base_to_plane)
            scene_source = self._scene(str(request["scene_id"]))
            self._scenes[scene_source.scene_id] = scene_source.model_copy(
                update={"timestamp_ns": time.time_ns(), "valid_for_ms": 1_800_000},
                deep=True,
            )
        scene = self._scene_with_graph_task_plane(
            self._scene(str(request["scene_id"])), graph
        )
        if mode != "hardware":
            scene = self._scene_with_mock_binding_fixtures(scene, graph)
        hints = dict(request.get("bindings", {}))
        requirements = [
            requirement.model_copy(update={"instance_id": hints.get(requirement.variable)})
            if requirement.variable in hints
            else requirement
            for requirement in graph.binding_requirements()
        ]
        bindings = EntityBinder().bind_entities(
            scene,
            requirements,
            maximum_scene_age_ms=self.settings.scene_freshness_ms,
        )
        robot = (
            DoosanM0609Adapter(
                aruco_robot,
                fixed_workspace_orientation_xyzw=fixed_workspace_orientation_xyzw,
                initial_pose=initial_hardware_pose,
            )
            if mode == "hardware"
            else MockRobotAdapter()
        )
        robot.connect()
        return row, graph, scene, bindings, robot, self._motion_profiles(), self._force_profiles()

    @staticmethod
    def _graph_with_aruco_base_plane(graph: SkillGraph, base_to_plane: Any) -> SkillGraph:
        """Bind the recorded task-plane anchor to the current acknowledged base frame."""

        uncertainty = dict(graph.uncertainty)
        transform = rigid_transform_from_matrix(base_to_plane, label="ArUco T_base_plane")
        uncertainty["base_chain"] = {
            "verified": True,
            "verification_source": "operator_acknowledged_aruco_reference_session",
            "base_to_task_plane": transform.model_dump(mode="json"),
        }
        return graph.model_copy(update={"uncertainty": uncertainty}, deep=True)

    def _load_fixed_base_plane(self) -> tuple[Any, str]:
        """Load the operator-frozen base/workspace transform without re-running ArUco."""

        with self._fixed_hardware_session_lock:
            cached = self._fixed_base_plane_cache
            if cached is not None:
                return cached[0].copy(), cached[1]
        source = (
            self.settings.repo_root
            / "aruco/results/d435i_plane_scans/scan_20260807_123803/"
            "base_workspace_urdf_candidate.json"
        ).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        transforms = payload.get("transforms")
        if not isinstance(transforms, dict):
            raise ValueError("fixed base workspace artifact has no transforms")
        matrix = transforms.get(
            "T_base_plane_from_urdf_camera_and_reference_T_camera_plane"
        )
        transform = rigid_transform_from_matrix(matrix, label="fixed T_base_plane")
        # Round-trip through the typed transform to reject malformed matrices, then
        # retain the exact matrix values selected by the operator.
        del transform
        import numpy as np

        result = np.asarray(matrix, dtype=np.float64), str(source)
        with self._fixed_hardware_session_lock:
            self._fixed_base_plane_cache = result
        return result[0].copy(), result[1]

    def _acquire_fixed_hardware_session(self) -> tuple[DoosanArucoExperimentRobot, Any]:
        """Reuse one server-owned Doosan session and the frozen workspace transform."""

        if not self.settings.hardware_enabled:
            raise ValueError("서버가 hardware 모드로 시작되지 않았습니다")
        base_to_plane, _ = self._load_fixed_base_plane()
        with self._fixed_hardware_session_lock:
            robot = self._fixed_hardware_robot
            if robot is None:
                robot = DoosanArucoExperimentRobot(
                    robot_id=self.settings.doosan_robot_id,
                    robot_model=self.settings.doosan_robot_model,
                    execution_mode="hardware",
                    hardware_enabled=True,
                    expected_tcp_name=self.settings.aruco_experiment_expected_tcp,
                )
                robot.connect()
                self._fixed_hardware_robot = robot
            return robot, base_to_plane.copy()

    def _mock_regression_validate(
        self, row: SkillVersionRecord, graph: SkillGraph
    ) -> dict[str, Any]:
        """Compile, bind, preflight, and execute a candidate against a fresh mock scene."""

        scene = self._scene_with_graph_task_plane(
            capture_mock_scene(frame_count=self.settings.scene_burst_frame_count),
            graph,
        )
        scene = self._scene_with_mock_binding_fixtures(scene, graph)
        bindings = EntityBinder().bind_entities(
            scene,
            graph.binding_requirements(),
            maximum_scene_age_ms=self.settings.scene_freshness_ms,
        )
        robot = MockRobotAdapter()
        robot.connect()
        gripper = MockGripperAdapter()
        gripper.connect()
        motion = self._motion_profiles()
        force = self._force_profiles()
        safety_policy = self._safety_policy()
        preflight = PreflightValidator().validate(
            scene=scene,
            skill=graph,
            bindings=bindings,
            robot=robot,
            execution_mode=ExecutionMode.MOCK,
            skill_validation_status="passed",
            expected_tool_class=graph.required_tools[0] if graph.required_tools else None,
            skill_uses_force=bool(graph.force_profiles),
            motion_profiles=motion,
            force_profiles=force,
            verification_profiles=self._verification_profiles(),
            safety_policy=safety_policy,
            robot_backend="mock",
            enable_real_robot=False,
            dry_run=True,
        )
        event_sink = InMemoryEventSink()
        context = RuntimeContext(
            scene=scene,
            bindings=bindings,
            motion_profiles=motion,
            force_profiles=force,
            verification_profiles=self._verification_profiles(),
            execution_mode=ExecutionMode.MOCK,
            skill=graph,
            safety_policy=safety_policy,
            command_text="candidate regression validation",
            preflight_report=preflight,
        )
        contact_links = tuple(
            link for surface in scene.surfaces for link in surface.allowed_contact_links
        )
        executor = RuntimeExecutor(
            context=context,
            robot=robot,
            gripper=gripper,
            binder=EntityBinder(),
            workspace_supervisor=GlobalWorkspaceSupervisor(
                minimum_clearance_m=safety_policy.minimum_clearance_m
            ),
            force_supervisor=GlobalForceSupervisor(
                robot,
                force,
                allowed_contact_links=contact_links,
                maximum_force_n=safety_policy.maximum_force_n,
            ),
            safety_supervisor=GlobalSafetySupervisor(),
            primitive_registry=get_default_registry(),
            event_sink=event_sink,
        )
        asyncio.run(executor.execute_compiled(self._load_verified_run(row, graph)))
        gripper_state = gripper.get_state()
        return {
            "passed": True,
            "preflight": preflight.as_dict(),
            "robot_commands": [command.operation for command in robot.commands],
            "gripper_commands": [operation for operation, _ in gripper.commands],
            "final_gripper_state": {
                "connected": gripper_state.connected,
                "width_m": gripper_state.width_m,
                "is_holding": gripper_state.is_holding,
                "fault_code": gripper_state.fault_code,
            },
            "events": [event.event_type for event in event_sink.events],
        }

    @staticmethod
    def _scene_with_mock_binding_fixtures(
        scene: SceneSnapshot, graph: SkillGraph
    ) -> SceneSnapshot:
        """Supply declared object classes only inside the explicit built-in Mock scene.

        This makes class-specific block skills reproducible without weakening real-scene
        binding.  The cloned object retains the Mock perception pose, including its complete
        quaternion, and is visibly marked as a generated test fixture.
        """

        camera = scene.camera_metadata
        if (
            scene.calibration_id != "mock_calibration_v1"
            or camera is None
            or camera.camera_id != "mock_d435i"
        ):
            return scene
        objects = list(scene.objects)
        if not objects:
            return scene
        existing_ids = {
            entity.instance_id
            for collection in (scene.objects, scene.tools, scene.surfaces)
            for entity in collection
        }
        for binding in graph.bindings.values():
            class_name = binding.class_name
            if binding.entity_kind is not EntityKind.OBJECT or class_name is None:
                continue
            if any(item.class_name == class_name for item in objects):
                continue
            identifier_class = re.sub(r"[^A-Za-z0-9_.:-]+", "_", class_name).strip("_")
            if not identifier_class:
                raise ValueError("Mock object binding class has no safe identifier")
            base_identifier = f"mock_{identifier_class}_01"
            identifier = base_identifier
            suffix = 2
            while identifier in existing_ids:
                identifier = f"mock_{identifier_class}_{suffix:02d}"
                suffix += 1
            seed = objects[0]
            fixture = seed.model_copy(
                update={
                    "instance_id": identifier,
                    "class_name": class_name,
                    "attributes": {
                        **seed.attributes,
                        "mock_binding_fixture": True,
                        "declared_skill_id": graph.skill_id,
                    },
                    "pose_source": "mock_declared_binding_fixture_6d",
                },
                deep=True,
            )
            objects.append(fixture)
            existing_ids.add(identifier)
        if objects == scene.objects:
            return scene
        return scene.model_copy(update={"objects": objects}, deep=True)

    def _load_verified_run(
        self, row: SkillVersionRecord, graph: SkillGraph
    ) -> CompiledRun:
        """Verify the registry graph, manifest association, and code immediately before import."""

        if not row.generated_code_uri or not row.generated_code_checksum_sha256:
            raise ValueError("skill version has no compiled artifact/checksum")
        if SkillCompiler.graph_checksum(graph) != row.graph_checksum_sha256:
            raise ValueError("registry graph checksum does not match the validated SkillGraph")
        code_uri = PurePosixPath(row.generated_code_uri)
        manifest_uri = str(code_uri.parent / "manifest.json")
        manifest = SkillManifest.model_validate_json(self.store.read_bytes(manifest_uri))
        expected = {
            "skill_id": graph.skill_id,
            "version": graph.version,
            "compiled_skill_uri": row.generated_code_uri,
            "compiled_skill_checksum_sha256": row.generated_code_checksum_sha256,
            "skill_graph_checksum_sha256": row.graph_checksum_sha256,
        }
        actual = {
            "skill_id": manifest.skill_id,
            "version": manifest.version,
            "compiled_skill_uri": manifest.compiled_skill_uri,
            "compiled_skill_checksum_sha256": manifest.compiled_skill_checksum_sha256,
            "skill_graph_checksum_sha256": manifest.skill_graph_checksum_sha256,
        }
        if actual != expected:
            raise ValueError("skill manifest is not associated with the selected registry version")
        if not SkillCompiler.verify_manifest(manifest, self.settings.artifact_root):
            raise ValueError("skill manifest artifact checksum verification failed")
        return load_compiled_run(
            Path(row.generated_code_uri),
            artifact_root=self.settings.artifact_root,
            expected_checksum_sha256=row.generated_code_checksum_sha256,
        )

    def _motion_profiles(self) -> dict[str, Any]:
        return load_motion_profiles(
            self.settings.repo_root / "configs/motion_profiles/default.json"
        )

    def _force_profiles(self) -> dict[str, Any]:
        return load_force_profiles(
            self.settings.repo_root / "configs/force_profiles/default.json"
        )

    def _verification_profiles(self) -> dict[str, Any]:
        return load_grasp_verification_profiles(
            self.settings.repo_root
            / "configs/grasp_verification_profiles/default.json"
        )

    def _safety_policy(self) -> SafetyPolicy:
        policies = load_safety_policies(
            self.settings.repo_root / "configs/safety_policies/default.json"
        )
        return policies["global_default"]

    def _persist_graph(
        self,
        graph: SkillGraph,
        *,
        status: str,
        validation_status: str,
        variant: str,
        parent_version_id: str | None = None,
        index_for_legacy_retrieval: bool = True,
    ) -> SkillVersionRecord:
        graph_report = SkillGraphValidator().validate(graph)
        compiled = SkillCompiler().compile(graph)
        base_uri = f"skills/{graph.skill_id}/{graph.version}"
        graph_artifact = self.store.put_json(
            f"{base_uri}/skill_graph.json", graph.model_dump(mode="json")
        )
        code_artifact = self.store.put_text(
            f"{base_uri}/compiled_skill.py",
            compiled.source,
            media_type="text/x-python; charset=utf-8",
        )
        report = ValidationReport(
            skill_id=graph.skill_id,
            version=graph.version,
            passed=graph_report.valid and compiled.validation_report.valid,
            checks={
                "schema": True,
                "graph": graph_report.valid,
                "ast": compiled.validation_report.valid,
                "py_compile": compiled.validation_report.py_compile_passed,
            },
            graph_checksum_sha256=graph_artifact.checksum_sha256,
            generated_code_checksum_sha256=code_artifact.checksum_sha256,
            mock_validation=False,
            hardware_validated=False,
            timestamp_ns=0,
        )
        report_artifact = self.store.put_json(
            f"{base_uri}/validation_report.json", report.model_dump(mode="json")
        )
        manifest = SkillManifest(
            skill_id=graph.skill_id,
            version=graph.version,
            skill_graph_uri=graph_artifact.uri,
            skill_graph_checksum_sha256=graph_artifact.checksum_sha256,
            compiled_skill_uri=code_artifact.uri,
            compiled_skill_checksum_sha256=code_artifact.checksum_sha256,
            validation_report_uri=report_artifact.uri,
            validation_report_checksum_sha256=report_artifact.checksum_sha256,
            source_demonstration_uris=list(graph.source_demonstrations),
        )
        self.store.put_json(f"{base_uri}/manifest.json", manifest.model_dump(mode="json"))
        if not SkillCompiler.verify_manifest(manifest, self.settings.artifact_root):
            raise ValueError("skill manifest checksum verification failed")
        existing = self._find_version_optional(graph.skill_id, graph.version)
        if existing is not None:
            if existing.graph_checksum_sha256 != graph_artifact.checksum_sha256:
                raise ValueError(
                    "existing skill identity/version has a different graph checksum"
                )
            if index_for_legacy_retrieval:
                self._ensure_skill_embedding(existing, graph)
            return existing
        registered = self.repository.register_skill_version(
            name=graph.name,
            intent=graph.skill_id,
            semantic_version=graph.version,
            graph=graph,
            status=status,
            variant=variant,
            description=graph.description,
            parent_version_id=parent_version_id,
            generated_code_uri=code_artifact.uri,
            generated_code_checksum_sha256=code_artifact.checksum_sha256,
            validation_status=validation_status,
            hardware_compatible=False,
        )
        if index_for_legacy_retrieval:
            self._ensure_skill_embedding(registered, graph)
        return registered

    def _ensure_skill_embedding(
        self, row: SkillVersionRecord, graph: SkillGraph
    ) -> None:
        if self.repository.get_skill_embedding(
            skill_version_id=row.id,
            model=self.settings.openai_embedding_model,
        ) is not None:
            return
        document = SkillSearchDocument(
            name=graph.name,
            description=graph.description,
            target_objects=tuple(graph.required_entity_roles.values()),
            tools=tuple(graph.required_tools),
            style=graph.operator_style or "normal",
            preconditions=tuple(graph.preconditions),
        )
        embedded = SkillEmbeddingService(self.settings).embed_document(document)
        self.repository.put_skill_embedding(
            skill_version_id=row.id,
            model=embedded.model,
            vector=embedded.vector,
            source_checksum_sha256=row.graph_checksum_sha256,
        )

    def _active_rows(self) -> list[tuple[SkillRecord, SkillVersionRecord]]:
        with self.database.session() as session:
            statement = (
                select(SkillRecord, SkillVersionRecord)
                .join(
                    SkillVersionRecord,
                    SkillRecord.active_version_id == SkillVersionRecord.id,
                )
                .where(
                    SkillVersionRecord.status == "active",
                    SkillVersionRecord.validation_status == "passed",
                )
            )
            return list(session.execute(statement).tuples())

    def _function_dispatcher(
        self, scene: SceneSnapshot | None = None
    ) -> SafeFunctionDispatcher:
        """Snapshot validated read-only registry/Scene data for live model tools."""

        with self.database.session() as session:
            rows = list(
                session.execute(
                    select(SkillVersionRecord, SkillRecord).join(
                        SkillRecord, SkillVersionRecord.skill_id == SkillRecord.id
                    )
                ).tuples()
            )
        skill_versions: list[FunctionSkillVersion] = []
        manifests: list[FunctionSkillManifest] = []
        for version, skill in rows:
            graph = SkillGraph.model_validate(version.graph_json)
            skill_versions.append(
                FunctionSkillVersion(
                    skill_id=graph.skill_id,
                    name=skill.name,
                    intent=skill.intent,
                    version=version.semantic_version,
                    lifecycle_status=version.status,
                    validation_status=version.validation_status,
                    variant=skill.variant,
                    description=skill.description,
                    graph_checksum_sha256=version.graph_checksum_sha256,
                    generated_code_checksum_sha256=(
                        version.generated_code_checksum_sha256
                    ),
                    hardware_compatible=version.hardware_compatible,
                )
            )
            if not version.generated_code_uri:
                continue
            manifest_uri = str(
                PurePosixPath(version.generated_code_uri).parent / "manifest.json"
            )
            try:
                manifest = SkillManifest.model_validate_json(
                    self.store.read_bytes(manifest_uri)
                )
            except (FileNotFoundError, ValueError):
                continue
            if (
                manifest.skill_id == graph.skill_id
                and manifest.version == graph.version
                and SkillCompiler.verify_manifest(manifest, self.settings.artifact_root)
            ):
                manifests.append(FunctionSkillManifest.from_manifest(manifest))
        provider = LocalReadOnlyToolProvider(
            skill_versions=skill_versions,
            manifests=manifests,
            scenes=(() if scene is None else (FunctionScene.from_scene(scene),)),
            primitive_registry=get_default_registry(),
        )
        return SafeFunctionDispatcher(provider)

    def _versions(self, skill_id: str) -> list[SkillVersionRecord]:
        with self.database.session() as session:
            rows = list(session.scalars(select(SkillVersionRecord)))
        matched = [row for row in rows if row.graph_json.get("skill_id") == skill_id]
        if not matched:
            raise KeyError(f"unknown skill {skill_id!r}")
        return sorted(matched, key=lambda row: (row.created_at, row.semantic_version))

    def _skill_variant(self, row: SkillVersionRecord) -> str:
        with self.database.session() as session:
            skill = session.get(SkillRecord, row.skill_id)
            if skill is None:
                raise KeyError(row.skill_id)
            return skill.variant

    def _find_version(
        self, skill_id: str, version: str | None = None
    ) -> SkillVersionRecord:
        row = self._find_version_optional(skill_id, version)
        if row is None:
            label = f" version {version!r}" if version else ""
            raise KeyError(f"unknown skill {skill_id!r}{label}")
        return row

    def _find_version_optional(
        self, skill_id: str, version: str | None = None
    ) -> SkillVersionRecord | None:
        with self.database.session() as session:
            rows = list(session.scalars(select(SkillVersionRecord)))
        matches = [
            row
            for row in rows
            if row.graph_json.get("skill_id") == skill_id
            and (version is None or row.semantic_version == version)
        ]
        if version is None:
            active = [row for row in matches if row.status == "active"]
            if active:
                return max(active, key=lambda row: row.created_at)
        return max(matches, key=lambda row: row.created_at) if matches else None

    @staticmethod
    def _version_summary(
        row: SkillVersionRecord, *, include_graph: bool = False
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "skill_id": row.graph_json.get("skill_id"),
            "version": row.semantic_version,
            "status": row.status,
            "validation_status": row.validation_status,
            "graph_checksum_sha256": row.graph_checksum_sha256,
            "generated_code_uri": row.generated_code_uri,
            "generated_code_checksum_sha256": row.generated_code_checksum_sha256,
            "hardware_compatible": row.hardware_compatible,
        }
        if include_graph:
            summary["skill_graph"] = row.graph_json
        return summary

    def _retire_other_active_versions(self, skill_record_id: str, keep_id: str) -> None:
        with self.database.session() as session:
            skill = session.get(SkillRecord, skill_record_id)
            if skill is None:
                return
            self._retire_current_in_session(session, skill, except_version_id=keep_id)

    @staticmethod
    def _retire_current_in_session(
        session: Any, skill: SkillRecord, *, except_version_id: str
    ) -> None:
        if skill.active_version_id and skill.active_version_id != except_version_id:
            current = session.get(SkillVersionRecord, skill.active_version_id)
            if current is not None:
                current.status = "retired"

    def _demonstration_evidence(self, source: Path) -> DemonstrationEvidence:
        trajectory = self._load_demonstration_source(source)
        if trajectory.successful is False:
            raise ValueError("failed demonstrations cannot induce or update a skill")
        warnings: list[str] = []
        try:
            processed = preprocess_trajectory(trajectory)
        except TrajectoryPreprocessingError as exc:
            # Preserve a locally analyzable Candidate for operator review even
            # when every pose falls below the normal observation threshold.
            # This fallback changes no robot safety/profile threshold.
            processed = preprocess_trajectory(
                trajectory,
                replace(PreprocessingConfig(), confidence_threshold=0.0),
            )
            warnings.append(f"low-quality preprocessing fallback: {exc}")
        quality = assess_trajectory_quality(processed)
        segmentation = segment_trajectory(trajectory)
        if quality.reteach_required:
            warnings.append(
                quality.reason or "local quality assessment recommends reteaching"
            )
        if segmentation.reteach_required:
            warnings.append(
                segmentation.reason or "local segmentation recommends reteaching"
            )
        return DemonstrationEvidence(
            source=source,
            trajectory=trajectory,
            processed=processed,
            quality=quality,
            summary=summarize_trajectory(processed),
            recommendation=recommend_primitive(processed),
            promotion_warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _load_demonstration_source(source: Path) -> DemonstrationTrajectory:
        if source.is_dir() or source.suffix == ".jsonl":
            return load_demonstration(source)
        if source.suffix != ".json":
            raise ValueError(
                "demonstration must be a fixture JSON, pose JSONL, or session directory"
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        fixture_kind = payload.get("fixture_kind")
        generators = {
            "novice_wipe": generate_novice_wipe_trajectory,
            "expert_wipe": generate_expert_wipe_trajectory,
            "periodic_wipe": generate_periodic_trajectory,
        }
        generator = generators.get(fixture_kind)
        if generator is None:
            raise ValueError(f"unsupported demonstration fixture kind: {fixture_kind!r}")
        return generator()

    @staticmethod
    def _normalized_path(trajectory: ProcessedTrajectory) -> list[tuple[float, float, float]]:
        origin = trajectory.samples[0].position_m
        return [
            (
                sample.position_m[0] - origin[0],
                sample.position_m[1] - origin[1],
                sample.position_m[2] - origin[2],
            )
            for sample in trajectory.samples
        ]

    @staticmethod
    def _wipe_path_updates(
        graph: SkillGraph, path: list[tuple[float, float, float]]
    ) -> dict[str, dict[str, Any]]:
        """Map measured normalized wipe evidence onto the fixed MVP wipe topology."""

        if len(path) < 4:
            return {}

        def target(fraction: float) -> dict[str, Any]:
            index = min(len(path) - 1, round((len(path) - 1) * fraction))
            x_m, y_m, z_m = path[index]
            return {
                "anchor_id": "$surface",
                "anchor_type": "surface",
                "position_m": {"x": x_m, "y": y_m, "z": z_m},
                "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }

        node_ids = {node.node_id for node in graph.nodes}
        updates: dict[str, dict[str, Any]] = {}
        if "validate_path" in node_ids:
            updates["validate_path"] = {
                "path": [target(index / 11.0) for index in range(12)]
            }
        if "approach_linear" in node_ids:
            updates["approach_linear"] = {"target": target(0.0)}
        if "stroke_forward" in node_ids:
            updates["stroke_forward"] = {"target": target(1.0 / 3.0)}
        if "turn_arc" in node_ids:
            updates["turn_arc"] = {
                "via": target(0.5),
                "target": target(2.0 / 3.0),
            }
        if "stroke_backward" in node_ids:
            updates["stroke_backward"] = {"target": target(1.0)}
        return updates

    @staticmethod
    def _apply_node_argument_updates(
        graph: SkillGraph, updates: dict[str, dict[str, Any]]
    ) -> SkillGraph:
        nodes = []
        for node in graph.nodes:
            arguments = dict(node.arguments)
            arguments.update(updates.get(node.node_id, {}))
            nodes.append(node.model_copy(update={"arguments": arguments}, deep=True))
        return graph.model_copy(update={"nodes": nodes}, deep=True)

    def _parent_normalized_path(
        self, parent_graph: SkillGraph
    ) -> list[tuple[float, float, float]]:
        for reference in reversed(parent_graph.source_demonstrations):
            try:
                source = self._safe_demo_path(reference)
                return self._normalized_path(
                    preprocess_trajectory(self._load_demonstration_source(source))
                )
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
        return []

    def _portable_artifact_reference(self, source: Path) -> str:
        resolved = source.resolve()
        for root in (self.settings.repo_root.resolve(), self.settings.artifact_root.resolve()):
            if resolved == root or resolved.is_relative_to(root):
                return resolved.relative_to(root).as_posix()
        raise ValueError("demonstration source is outside configured roots")

    def _safe_demo_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            candidate = path.resolve()
        else:
            repo_candidate = (self.settings.repo_root / path).resolve()
            artifact_candidate = (self.settings.artifact_root / path).resolve()
            candidate = repo_candidate if repo_candidate.exists() else artifact_candidate
        allowed_roots = (
            (self.settings.repo_root / "tests/fixtures").resolve(),
            (self.settings.repo_root / "data/demonstrations").resolve(),
            (self.settings.artifact_root / "demonstrations").resolve(),
        )
        if not any(candidate == root or root in candidate.parents for root in allowed_roots):
            raise ValueError("demonstration path escapes approved fixture/artifact roots")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate


def create_application(settings: Settings | None = None) -> MVPApplication:
    """Create the default service with environment-derived mock-safe settings."""

    return MVPApplication(settings or Settings.from_env())
