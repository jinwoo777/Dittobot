"""Synchronous application service shared by FastAPI and the CLI."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from sqlalchemy import select

from robot_skill_system.adapters.mock_robot import MockGripperAdapter, MockRobotAdapter
from robot_skill_system.calibration.controller import HandEyeCalibrationController
from robot_skill_system.calibration.robot import DoosanHandEyeCalibrationRobot
from robot_skill_system.capture.realsense_capture import (
    RealSenseCapture,
    RealSenseCaptureConfig,
)
from robot_skill_system.capture.rgb_frame_transport import (
    build_rgbd_contact_sheet_pdf,
    build_rgbd_keyframe_zip,
)
from robot_skill_system.capture.rgbd_recording import RGBDCameraController
from robot_skill_system.demonstrations.models import (
    DemonstrationTrajectory,
    ProcessedTrajectory,
)
from robot_skill_system.demonstrations.models import (
    PrimitiveRecommendation as LocalPrimitiveRecommendation,
)
from robot_skill_system.demonstrations.preprocessing import preprocess_trajectory
from robot_skill_system.demonstrations.primitive_fitter import recommend_primitive
from robot_skill_system.demonstrations.quality import (
    QualityAssessment,
    assess_trajectory_quality,
)
from robot_skill_system.demonstrations.recorder import load_demonstration
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
from robot_skill_system.demonstrations.synthetic import (
    generate_expert_wipe_trajectory,
    generate_novice_wipe_trajectory,
    generate_periodic_trajectory,
)
from robot_skill_system.demonstrations.trajectory import (
    TrajectorySummary,
    summarize_trajectory,
)
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
from robot_skill_system.openai_integration.recording_skill_analyzer import (
    ImageInputRejectedError,
    RecordingSkillDraftAnalyzer,
)
from robot_skill_system.openai_integration.schemas import (
    DemonstrationAnalysisInput,
    RecordingSkillDraftInput,
)
from robot_skill_system.perception.hand_pose import MediaPipeHandPoseEstimator
from robot_skill_system.primitives.models import SafetyPolicy
from robot_skill_system.primitives.profiles import (
    load_force_profiles,
    load_motion_profiles,
    load_safety_policies,
)
from robot_skill_system.primitives.registry import get_default_registry
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import ExecutionAbortedError
from robot_skill_system.runtime.event_log import InMemoryEventSink
from robot_skill_system.runtime.executor import RuntimeExecutor
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.integrity import verify_skill_checksum
from robot_skill_system.runtime.models import ExecutionMode, RuntimeContext
from robot_skill_system.runtime.preflight import PreflightValidator
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor
from robot_skill_system.scene.models import SceneSnapshot
from robot_skill_system.scene.transforms import RigidTransform
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
from robot_skill_system.skills.retrieval import (
    SkillCandidate,
    SkillSearchQuery,
    rank_skills,
)
from robot_skill_system.skills.updater import SkillUpdater, UpdateEvidence
from robot_skill_system.skills.versioning import stable_version
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import Database, StorageRepository
from robot_skill_system.storage.orm import (
    ExecutionRunRecord,
    SceneRecord,
    SkillRecord,
    SkillVersionRecord,
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


@dataclass(frozen=True, slots=True)
class ActiveExecution:
    """Abort handles for a currently running mock/dry-run execution."""

    robot: MockRobotAdapter
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

    def close(self) -> None:
        """Release database resources."""

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
        if mode not in {"mock", "single", "burst"}:
            raise ValueError("MVP scene capture supports mock/single/burst only")
        frame_count = 1 if mode == "single" else self.settings.scene_burst_frame_count
        scene = capture_mock_scene(mode=mode, frame_count=frame_count)
        self.repository.record_scene(scene)
        self.store.put_json(
            f"scenes/{scene.scene_id}.json", scene.model_dump(mode="json")
        )
        self._scenes[scene.scene_id] = scene
        return scene.model_dump(mode="json")

    def get_scene(self, scene_id: str) -> dict[str, Any]:
        return self._scene(scene_id).model_dump(mode="json")

    def get_camera_status(self) -> dict[str, Any]:
        return self.camera_controller.status()

    def get_handeye_calibration_status(self) -> dict[str, Any]:
        return self.calibration_controller.status()

    def start_handeye_calibration(self, request: dict[str, Any]) -> dict[str, Any]:
        required = ("operator_confirmed", "board_secured", "workspace_cleared", "estop_ready")
        if not all(request.get(key) is True for key in required):
            raise ValueError("all hand-eye calibration safety acknowledgements are required")
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
        return self.camera_controller.start_recording(
            maximum_duration_s=float(request.get("maximum_duration_s", 30.0))
        )

    def stop_camera_recording(self, recording_id: str) -> dict[str, Any]:
        return self.camera_controller.stop_recording(recording_id)

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
            "image_detail": self.settings.openai_image_detail,
            "uploads_rgb_keyframes_only": False,
            "uploads_rgb_and_aligned_depth_pairs": True,
            "image_pair_order": "rgb_then_aligned_depth_per_keyframe",
            "tcp_proxy_mode": "two_finger_gripper_midpoint_semantic_only",
            "provider_video_input_supported": False,
            "direct_image_transport": True,
            "fallback_analysis_transport": "pdf_contact_sheet",
            "creates_zip_archive_on_fallback": True,
            "zip_is_not_used_as_vision_input": True,
            "returns_frame_complete_tcp_audit": True,
            "returns_semantic_scene_regions": True,
            "automatic_surface_plane_backend": "local_numpy_ransac_raw_depth",
            "npy_validation_blocks_mock_candidate": False,
            "stores_openai_response": False,
            "creates_executable_skill": False,
        }

    def create_recording_skill_draft(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create a non-executable semantic draft from bounded RGB-D frame pairs."""

        recording_id = str(request["recording_id"])
        maximum_keyframes = min(
            int(request.get("keyframe_count", 100)),
            self.settings.openai_max_keyframes,
        )
        selected = self.camera_controller.select_recording_rgbd_keyframes(
            recording_id, maximum_keyframes
        )
        manifest = self.camera_controller.get_recording_manifest(
            recording_id, include_frames=False
        )
        primitive_catalog = list(get_default_registry().operation_names())
        entity_role_catalog = [
            "tool",
            "target_object",
            "target_surface",
            "fixture",
            "workspace_region",
        ]
        keyframe_indices = [index for index, _rgb_path, _depth_path in selected]
        analysis_input = RecordingSkillDraftInput(
            recording_id=recording_id,
            name_hint=str(request.get("name_hint", "recorded_skill")),
            operator_instruction=str(request["operator_instruction"]),
            recording_summary={
                "duration_s": manifest.get("duration_s"),
                "frame_count": manifest.get("frame_count"),
                "recording_fps": manifest.get("recording_fps"),
                "depth_aligned_to_color": manifest.get("depth_aligned_to_color"),
                "timestamps_preserved": manifest.get("timestamps_preserved"),
            },
            primitive_catalog=primitive_catalog,
            entity_role_catalog=entity_role_catalog,
            keyframe_indices=keyframe_indices,
            limitations=[
                "RGB-D keyframes contain no trusted robot-base pose trajectory or TF chain.",
                "Depth NPZ artifacts remain local and are not uploaded to OpenAI.",
                "Aligned depth is uploaded only as a qualitative TURBO color visualization.",
                "The midpoint of two visible fingertips is a semantic TCP proxy, not a pose.",
                "Normalized OpenAI regions are hints; local raw depth and intrinsics "
                "own metric geometry.",
                "A missing TCP landmark trajectory must be reported with an explicit reason.",
                "Force, velocity, acceleration, and execution permission remain local-only.",
            ],
        )
        draft_id = f"draft_{uuid.uuid4().hex}"
        analyzer = RecordingSkillDraftAnalyzer(self.settings)
        transport: dict[str, Any] = {
            "mode": "direct_rgbd_images",
            "keyframe_pair_count": len(selected),
            "image_count": len(selected) * 2,
            "pair_order": "rgb_then_aligned_depth_per_keyframe",
            "fallback_used": False,
        }
        try:
            draft, metadata = analyzer.analyze(
                analysis_input,
                rgb_paths=[rgb_path for _index, rgb_path, _depth_path in selected],
                depth_paths=[depth_path for _index, _rgb_path, depth_path in selected],
            )
        except ImageInputRejectedError:
            transport_root = (
                f"demonstrations/{recording_id}/skill_drafts/{draft_id}_transport"
            )
            zip_artifact = self.store.put_bytes(
                f"{transport_root}/rgbd_keyframes.zip",
                build_rgbd_keyframe_zip(recording_id, selected),
                media_type="application/zip",
            )
            pdf_artifact = self.store.put_bytes(
                f"{transport_root}/rgbd_contact_sheet.pdf",
                build_rgbd_contact_sheet_pdf(selected),
                media_type="application/pdf",
            )
            draft, metadata = analyzer.analyze_pdf(
                analysis_input,
                contact_sheet_path=self.store.path_for(pdf_artifact.uri),
            )
            transport = {
                "mode": "pdf_contact_sheet",
                "keyframe_pair_count": len(selected),
                "image_count": len(selected) * 2,
                "pair_order": "rgb_then_aligned_depth_per_keyframe",
                "fallback_used": True,
                "reason": "direct_image_input_rejected",
                "zip_archive": {
                    "uri": zip_artifact.uri,
                    "checksum_sha256": zip_artifact.checksum_sha256,
                    "size_bytes": zip_artifact.size_bytes,
                },
                "analysis_pdf": {
                    "uri": pdf_artifact.uri,
                    "checksum_sha256": pdf_artifact.checksum_sha256,
                    "size_bytes": pdf_artifact.size_bytes,
                },
            }
        artifact_payload = {
            "schema_version": "1.0",
            "draft_id": draft_id,
            "status": "semantic_draft",
            "created_at_ns": time.time_ns(),
            "source_recording_id": recording_id,
            "keyframe_indices": keyframe_indices,
            "openai_mode": self.settings.openai_mode.value,
            "openai_model": self.settings.openai_reasoning_model,
            "openai_trace_id": metadata.trace_id,
            "transport": transport,
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

        transform, diagnostics = calibrate_surface_from_three_points(
            frame,
            origin_px=pixel("origin_px"),
            positive_x_px=pixel("positive_x_px"),
            positive_y_px=pixel("positive_y_px"),
        )
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
            "method": "operator_three_point_aligned_depth",
            "transform_convention": "T_camera_surface",
            "camera_to_surface": transform.model_dump(mode="json"),
            "diagnostics": diagnostics,
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
            "method": "local_depth_ransac_plane",
            "semantic_surface_hint": surface or None,
            "transform_convention": "T_camera_surface",
            "camera_to_surface": transform.model_dump(mode="json"),
            "diagnostics": diagnostics,
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
            calibration["camera_to_surface"]
        )
        source_frame = str(calibration["source_frame"])
        method = str(request["method"])
        samples: list[ManualTCPPathSample] = []
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
            if len(samples) < 4:
                detail = "; ".join(extraction_failures[:4]) or "no valid local depth"
                raise ValueError(
                    "GPT returned a TCP landmark trajectory, but local aligned depth "
                    f"reconstructed only {len(samples)} valid samples: {detail}"
                )
        elif method == "mediapipe_rgbd":
            keyframe_indices = [
                int(index) for index in draft_payload.get("keyframe_indices") or []
            ]
            attempted_count = len(keyframe_indices)
            estimator = MediaPipeHandPoseEstimator()
            for frame_index in keyframe_indices:
                frame = self.camera_controller.load_recording_rgbd_frame(
                    recording_id, frame_index
                )
                if frame.reference_frame != source_frame:
                    raise ValueError("recording frame does not match calibration source frame")
                estimate = estimator.estimate(frame)
                if estimate is None:
                    continue
                surface_to_tcp = transform_camera_pose_to_surface(
                    camera_to_surface,
                    RigidTransform(
                        translation_m=estimate.pose.position_m,
                        rotation_xyzw=estimate.pose.orientation_xyzw,
                    ),
                )
                samples.append(
                    ManualTCPPathSample(
                        frame_index=frame_index,
                        timestamp_ns=frame.timestamp_ns,
                        position_surface_m=surface_to_tcp.translation_m.as_tuple(),
                        orientation_surface_xyzw=(
                            surface_to_tcp.rotation_xyzw.as_tuple()
                        ),
                        gripper_width_m=estimate.gripper_width_m,
                        confidence=estimate.confidence,
                    )
                )
        else:
            raise ValueError("unsupported TCP trajectory extraction method")
        quality = validate_surface_relative_path(samples)
        coverage_ratio = len(samples) / max(1, attempted_count)
        if method == "mediapipe_rgbd" and coverage_ratio < 0.5:
            raise ValueError(
                "MediaPipe detected a valid two-finger pose in fewer than 50% of frames"
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
                "mediapipe_palm_orientation"
                if method == "mediapipe_rgbd"
                else "jaw_axis_x_surface_normal_z"
            ),
            "semantic_draft_artifact_uri": (
                draft_path.relative_to(self.store.root).as_posix()
                if method == "openai_rgbd"
                else None
            ),
            "extraction_failures": extraction_failures,
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
        graph = self._recording_candidate_graph(
            draft_payload,
            calibration=calibration,
            trajectory=trajectory,
            handeye_transform=handeye_transform,
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
        has_close = "gripper.close" in primitive_operations
        has_open = "gripper.open" in primitive_operations
        samples = trajectory.get("samples") or []
        if not isinstance(samples, list) or len(samples) < 4:
            raise ValueError("TCP trajectory has insufficient materialization samples")
        maximum_points = 256 if has_contact else 128
        if len(samples) > maximum_points:
            selected_indices = sorted(
                {
                    round(position * (len(samples) - 1) / (maximum_points - 1))
                    for position in range(maximum_points)
                }
            )
            samples = [samples[index] for index in selected_indices]
        path = [
            {
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
            for sample in samples
        ]
        node_specs: list[tuple[str, str, dict[str, Any]]] = [
            ("validate_path", "workspace.validate_path", {"path": path})
        ]
        if has_close:
            node_specs.append(("gripper_close", "gripper.close", {"tool": "$tool"}))
        if has_contact:
            node_specs.extend(
                [
                    (
                        "contact_search",
                        "contact.search_surface",
                        {
                            "surface": "$surface",
                            "force_profile_id": "contact_search_soft",
                        },
                    ),
                    (
                        "force_enable",
                        "contact.enable_force",
                        {
                            "surface": "$surface",
                            "force_profile_id": "contact_search_soft",
                        },
                    ),
                    (
                        "follow_path",
                        "contact.follow_path",
                        {"path": path, "motion_profile_id": "linear_slow"},
                    ),
                    ("force_disable", "contact.disable_force", {}),
                ]
            )
        else:
            node_specs.append(
                (
                    "follow_path",
                    "motion.move_spline",
                    {"waypoints": path, "motion_profile_id": "linear_slow"},
                )
            )
        if has_open:
            node_specs.append(("gripper_open", "gripper.open", {"tool": "$tool"}))
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
                role="contact_target",
                minimum_confidence=0.8,
            )
        }
        if has_close or has_open or has_contact:
            bindings["$tool"] = BindingSpec(
                variable="$tool",
                entity_kind=EntityKind.TOOL,
                class_name=contact_tool_class,
                minimum_confidence=0.8,
                must_be_attached=True,
            )
        skill_type = (
            SkillType.COMPOSITE
            if (has_close or has_open) and has_contact
            else SkillType.CONTACT
            if has_contact
            else SkillType.MANIPULATION
            if has_close or has_open
            else SkillType.MOTION
        )
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
            motion_profiles=["linear_slow"],
            force_profiles=["contact_search_soft"] if has_contact else [],
            preconditions=[
                "fresh_scene",
                "surface_binding_verified",
                "operator_review_required",
            ],
            postconditions=["mock_validation_only"],
            uncertainty={
                "hardware_validated": False,
                "camera_to_surface_calibration_id": calibration["calibration_id"],
                "tcp_trajectory_id": trajectory["trajectory_id"],
                "trajectory_quality": trajectory["quality"],
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
        has_tcp_schema = isinstance(tcp_observation, dict)
        tcp_detected = (
            bool(tcp_observation.get("detected"))
            if isinstance(tcp_observation, dict)
            else False
        )
        pair_count = int(transport.get("keyframe_pair_count") or 0)
        has_rgbd_pairs = pair_count > 0 and transport.get("pair_order") == (
            "rgb_then_aligned_depth_per_keyframe"
        )
        calibration = self._latest_draft_evidence(
            payload, "surface_calibration_*.json"
        )
        trajectory = self._latest_draft_evidence(payload, "tcp_trajectory_*.json")
        registration = self._latest_draft_evidence(
            payload, "candidate_registration_*.json"
        )
        handeye_transform = self._latest_legacy_handeye_transform()
        has_calibration = bool(
            isinstance(calibration, dict)
            and calibration.get("operator_confirmed") is True
            and calibration.get("transform_convention") == "T_camera_surface"
        )
        trajectory_quality = (
            trajectory.get("quality") if isinstance(trajectory, dict) else None
        )
        has_pose_trajectory = bool(
            has_calibration
            and isinstance(trajectory, dict)
            and isinstance(calibration, dict)
            and trajectory.get("operator_confirmed") is True
            and trajectory.get("calibration_id") == calibration.get("calibration_id")
            and isinstance(trajectory_quality, dict)
            and int(trajectory_quality.get("sample_count") or 0) >= 4
            and float(trajectory_quality.get("mean_confidence") or 0.0) >= 0.6
        )
        mock_validation_passed = bool(
            registration and registration.get("mock_validation_passed") is True
        )
        checks = [
            {
                "id": "semantic_analysis",
                "label": "구조화된 semantic draft",
                "passed": True,
                "detail": "GPT 결과가 로컬 스키마와 catalog 검사를 통과했습니다.",
            },
            {
                "id": "rgbd_evidence",
                "label": "시간 정렬 RGB + Depth 증거",
                "passed": has_rgbd_pairs,
                "detail": (
                    f"{pair_count}개 RGB-D 프레임 쌍"
                    if has_rgbd_pairs
                    else "RGB-D 쌍 분석으로 다시 생성해야 합니다."
                ),
            },
            {
                "id": "tcp_proxy",
                "label": "두 손가락 TCP 프록시 관찰",
                "passed": tcp_detected,
                "detail": (
                    "두 fingertip 중점의 정성적 동작이 관찰되었습니다."
                    if tcp_detected
                    else "두 fingertip이 함께 보이는 구간이 필요합니다."
                ),
            },
            {
                "id": "calibrated_transform",
                "label": "보정된 camera → surface/tool TF",
                "passed": has_calibration,
                "detail": (
                    f"{calibration['calibration_id']} · "
                    f"{calibration.get('method', 'RGB-D 표면 프레임')}"
                    if has_calibration and isinstance(calibration, dict)
                    else "Depth 평면을 자동 추출하거나 RGB에서 원점, +X, +Y를 지정하세요."
                ),
            },
            {
                "id": "handeye_transform_candidate",
                "label": "선택 증거: NPY 기반 flange → camera TF 후보",
                "passed": bool(
                    handeye_transform and handeye_transform.get("passed") is True
                ),
                "required": False,
                "blocking": False,
                "detail": (
                    f"{handeye_transform['import_id']} · "
                    + (
                        "관측 품질 검증 통과, 물리 검증은 별도 필요"
                        if handeye_transform.get("passed") is True
                        else "품질 기준 미통과 · Mock Candidate 등록은 차단하지 않음"
                    )
                    if handeye_transform
                    else "없음 · Mock Candidate 등록은 차단하지 않습니다."
                ),
            },
            {
                "id": "trusted_pose_trajectory",
                "label": "신뢰 가능한 tool/TCP pose trajectory",
                "passed": has_pose_trajectory,
                "detail": (
                    f"표면 상대 TCP {trajectory_quality['sample_count']}개 · "
                    f"경로 {float(trajectory_quality['path_length_m']):.3f} m"
                    if has_pose_trajectory and isinstance(trajectory_quality, dict)
                    else "두 fingertip 3D 경로를 추출하고 운영자가 확인해야 합니다."
                ),
            },
            {
                "id": "mock_validation",
                "label": "컴파일 및 Mock 회귀 검증",
                "passed": mock_validation_passed,
                "detail": (
                    f"{registration['skill_id']}@{registration['version']} Mock 검증 통과"
                    if mock_validation_passed and isinstance(registration, dict)
                    else "Candidate SkillGraph 등록 시 자동 실행됩니다."
                ),
            },
        ]
        if not has_rgbd_pairs or not has_tcp_schema:
            readiness_status = "needs_reanalysis"
        elif not tcp_detected:
            readiness_status = "needs_tcp_evidence"
        elif not has_calibration:
            readiness_status = "needs_calibration"
        elif not has_pose_trajectory:
            readiness_status = "needs_pose_evidence"
        elif mock_validation_passed:
            readiness_status = "candidate_registered"
        elif registration is not None:
            readiness_status = "candidate_validation_failed"
        else:
            readiness_status = "ready_for_candidate"
        can_register_candidate = (
            has_rgbd_pairs
            and has_tcp_schema
            and tcp_detected
            and has_calibration
            and has_pose_trajectory
            and not mock_validation_passed
        )
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
                "next_action": (
                    "RGB-D와 두 손가락 TCP 프록시로 다시 분석하세요."
                    if readiness_status == "needs_reanalysis"
                    else "두 fingertip이 함께 보이도록 다시 티칭하세요."
                    if readiness_status == "needs_tcp_evidence"
                    else "Depth 평면 자동 추출 또는 3점 방식으로 표면 TF를 보정하세요."
                    if readiness_status == "needs_calibration"
                    else "두 fingertip 경로를 자동 추출하거나 수동으로 지정하세요."
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
        if semantic.reteach_required:
            raise ValueError("semantic analysis requires the demonstration to be retaught")
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
        active = self.activate_skill(graph.skill_id, {"version": graph.version})
        return {
            "skill_id": graph.skill_id,
            "version": graph.version,
            "status": active["status"],
            "source_demo": self._portable_artifact_reference(source),
            "fitted_operations": fitted_operations,
            "observed_primitive": evidence.recommendation.recommended_primitive_id,
            "teaching_quality": evidence.quality.confidence,
            "openai_trace_id": metadata.trace_id,
            "generated_code_uri": version.generated_code_uri,
            "generated_code_checksum_sha256": version.generated_code_checksum_sha256,
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
        return {
            "skills": [
                {
                    **self._version_summary(version, include_graph=True),
                    "name": skill.name,
                    "intent": skill.intent,
                    "variant": skill.variant,
                    "description": skill.description,
                    "node_count": len(version.graph_json.get("nodes", [])),
                }
                for version, skill in rows
            ]
        }

    def create_skill(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create a new standalone skill without induction or recording."""

        name = str(request.get("name", "")).strip()
        intent = str(request.get("intent", "")).strip()
        variant = str(request.get("variant", "default")).strip() or "default"
        description = str(request.get("description", "")).strip()
        semantic_version = (
            str(request.get("semantic_version", "0.1.0")).strip() or "0.1.0"
        )

        if not name:
            raise ValueError("skill name is required")

        if not intent:
            raise ValueError("skill intent is required")

        # 새 스킬의 기본 그래프
        #
        # SkillGraph 스키마가 요구하는 메타데이터까지 모두 포함한다.
        # skill_id는 시스템 내부 식별자이므로 ASCII/숫자/underscore/hyphen만 사용한다.
        skill_id = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_")

        if not skill_id:
            skill_id = f"skill_{semantic_version.replace('.', '_')}"

        graph = {
            "schema_version": "1.0",
            "skill_id": skill_id,
            "version": semantic_version,
            "name": name,
            "description": description or name,
            "skill_type": "motion",

            "source_demonstrations": [],
            "operator_style": None,
            "required_tools": [],
            "required_entity_roles": {},
            "bindings": {},

            "nodes": [
                {
                    "node_id": "move_1",
                    "operation": "motion.move_l",
                    "arguments": {
                        "target": {
                            "anchor_id": "$surface",
                            "anchor_type": "surface",
                            "position_m": {
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                            },
                            "orientation_xyzw": {
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                                "w": 1.0,
                            },
                        },
                        "motion_profile_id": "linear_slow",
                    },
                }
            ],

            "edges": [],
            "start_node": "move_1",
            "terminal_nodes": ["move_1"],

            "motion_profiles": ["linear_slow"],
            "force_profiles": [],

            "preconditions": [],
            "postconditions": [],
            "recovery_policy": "global_safe_stop",

            "global_policy_requirements": [
                "global_safety_supervisor",
                "workspace_monitor",
                "force_supervisor",
                "emergency_stop_monitor",
            ],

            "uncertainty": {},
            "validation_status": "unvalidated",
            "lifecycle_status": "draft",
        }

        version = self.repository.register_skill_version(
            name=name,
            intent=intent,
            semantic_version=semantic_version,
            graph=graph,
            status="draft",
            variant=variant,
            description=description,
            validation_status="pending",
            hardware_compatible=False,
        )

        return {
            "skill_id": version.skill_id,
            "version": version.semantic_version,
            "name": name,
            "intent": intent,
            "variant": variant,
            "description": description,
            "status": version.status,
            "validation_status": version.validation_status,
            "message": "스킬이 생성되었습니다.",
        }

    def delete_skill(self, skill_id: str) -> dict[str, Any]:
        """Delete a skill from the registry."""

        result = self.repository.delete_skill(skill_id)

        return {
            "ok": True,
            **result,
        }

    def get_skill(self, skill_id: str, version: str | None = None) -> dict[str, Any]:
        row = self._find_version(skill_id, version)
        return self._version_summary(row, include_graph=True)

    def get_primitive_catalog(self) -> dict[str, Any]:
        """Return primitive metadata for the block-based skill editor."""

        primitives = []

        for metadata in get_default_registry().catalog():
            schema = metadata.typed_parameter_schema

            properties = schema.get("properties", {})
            required = schema.get("required", [])

            primitives.append(
                {
                    "operation_name": metadata.operation_name,
                    "description": metadata.description,
                    "parameter_schema": schema,
                    "required_parameters": required,
                    "allowed_skill_types": list(metadata.allowed_skill_types),
                    "required_preconditions": list(metadata.required_preconditions),
                    "side_effects": list(metadata.side_effects),
                    "maximum_timeout_s": metadata.maximum_timeout_s,
                    "recovery_operation": metadata.recovery_operation,
                    "hardware_support": metadata.hardware_support,
                    "simulation_support": metadata.simulation_support,
                    "mock_support": metadata.mock_support,
                }
            )

        return {
            "primitives": primitives,
        }

    def update_skill_node(
        self,
        skill_id: str,
        node_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Update one node's arguments in an editable skill version."""

        row = self._find_version(skill_id, request.get("version"))

        # 기존 DB graph를 최신 SkillGraph 모델로 강제 검증하지 않는다.
        # 구버전 graph도 UI에서 노드 arguments를 수정할 수 있도록
        # raw JSON을 직접 수정한다.
        graph = dict(row.graph_json)

        nodes = graph.get("nodes", [])

        if not isinstance(nodes, list):
            raise ValueError("skill graph nodes must be an array")

        target_node = None

        for node in nodes:
            if isinstance(node, dict) and node.get("node_id") == node_id:
                target_node = node
                break

        if target_node is None:
            raise KeyError(
                f"unknown node {node_id!r} in skill {skill_id!r}"
            )

        new_arguments = request.get("arguments")

        if not isinstance(new_arguments, dict):
            raise ValueError("arguments must be an object")

        existing_arguments = target_node.get("arguments", {})

        if not isinstance(existing_arguments, dict):
            existing_arguments = {}

        def deep_merge(
            original: dict[str, Any],
            updates: dict[str, Any],
        ) -> dict[str, Any]:
            result = dict(original)

            for key, value in updates.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value

            return result

        merged_arguments = deep_merge(
            existing_arguments,
            new_arguments,
        )

        operation = target_node.get("operation")

        if not isinstance(operation, str):
            raise ValueError("node operation is missing")

        registry = get_default_registry()

        normalized_arguments = registry.validate_arguments(
            operation,
            merged_arguments,
        )

        target_node["arguments"] = normalized_arguments.model_dump(
            mode="json"
        )

        graph["nodes"] = nodes

        self.repository.update_skill_version_graph(
            version_id=row.id,
            graph=graph,
        )

        updated_row = self._find_version(
            skill_id,
            row.semantic_version,
        )

        return self._version_summary(
            updated_row,
            include_graph=True,
        )

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
        version = self._persist_graph(
            proposal.candidate_graph,
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
        }

    def resolve_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = self._scene(str(request["scene_id"])) if request.get("scene_id") else None
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
        }

    def bind_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        row = self._find_version(str(request["skill_id"]), request.get("version"))
        graph = SkillGraph.model_validate(row.graph_json)
        scene = self._scene(str(request["scene_id"]))
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
        row, graph, scene, bindings, robot, motion, force = self._runtime_parts(request)
        mode = ExecutionMode(str(request.get("mode", "mock")))
        safety_policy = self._safety_policy()
        report = PreflightValidator().validate(
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
            safety_policy=safety_policy,
            robot_backend=self.settings.robot_backend,
            enable_real_robot=self.settings.enable_real_robot,
            dry_run=self.settings.dry_run,
        )
        return report.as_dict()

    def execute_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        requested_mode = str(request.get("mode", "mock"))
        if requested_mode == "hardware":
            if not self.settings.hardware_enabled:
                raise ValueError("hardware execution gates are not all enabled")
            raise ValueError("Doosan hardware adapter is not configured or physically validated")
        row, graph, scene, bindings, robot, motion, force = self._runtime_parts(request)
        safety_policy = self._safety_policy()
        report = PreflightValidator().validate(
            scene=scene,
            skill=graph,
            bindings=bindings,
            robot=robot,
            execution_mode=ExecutionMode(requested_mode),
            skill_validation_status=row.validation_status,
            expected_tool_class=graph.required_tools[0] if graph.required_tools else None,
            skill_uses_force=bool(graph.force_profiles),
            motion_profiles=motion,
            force_profiles=force,
            safety_policy=safety_policy,
            robot_backend="mock",
            enable_real_robot=False,
            dry_run=True,
        )
        run = self._load_verified_run(row, graph)
        gripper = MockGripperAdapter()
        gripper.connect()
        event_sink = InMemoryEventSink()
        context = RuntimeContext(
            scene=scene,
            bindings=bindings,
            motion_profiles=motion,
            force_profiles=force,
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
        executor = RuntimeExecutor(
            context=context,
            robot=robot,
            gripper=gripper,
            binder=EntityBinder(),
            workspace_supervisor=GlobalWorkspaceSupervisor(
                minimum_clearance_m=safety_policy.minimum_clearance_m
            ),
            force_supervisor=force_supervisor,
            safety_supervisor=safety_supervisor,
            primitive_registry=get_default_registry(),
            event_sink=event_sink,
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
        self, request: dict[str, Any]
    ) -> tuple[
        SkillVersionRecord,
        SkillGraph,
        SceneSnapshot,
        dict[str, Any],
        MockRobotAdapter,
        dict[str, Any],
        dict[str, Any],
    ]:
        row = self._find_version(str(request["skill_id"]), request.get("version"))
        if row.status != "active" or row.validation_status != "passed":
            raise ValueError("runtime accepts only active, validated skill versions")
        verify_skill_checksum(row.graph_json, row.graph_checksum_sha256)
        graph = SkillGraph.model_validate(row.graph_json)
        scene = self._scene(str(request["scene_id"]))
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
        robot = MockRobotAdapter()
        robot.connect()
        return row, graph, scene, bindings, robot, self._motion_profiles(), self._force_profiles()

    def _mock_regression_validate(
        self, row: SkillVersionRecord, graph: SkillGraph
    ) -> dict[str, Any]:
        """Compile, bind, preflight, and execute a candidate against a fresh mock scene."""

        scene = capture_mock_scene(frame_count=self.settings.scene_burst_frame_count)
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
        return {
            "passed": True,
            "preflight": preflight.as_dict(),
            "robot_commands": [command.operation for command in robot.commands],
            "events": [event.event_type for event in event_sink.events],
        }

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
        processed = preprocess_trajectory(trajectory)
        quality = assess_trajectory_quality(processed)
        segmentation = segment_trajectory(processed)
        if quality.reteach_required or segmentation.reteach_required:
            reason = quality.reason or segmentation.reason or "insufficient demonstration quality"
            raise ValueError(f"demonstration requires reteaching: {reason}")
        return DemonstrationEvidence(
            source=source,
            trajectory=trajectory,
            processed=processed,
            quality=quality,
            summary=summarize_trajectory(processed),
            recommendation=recommend_primitive(processed),
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
