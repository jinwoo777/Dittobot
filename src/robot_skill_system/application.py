"""Synchronous application service shared by FastAPI and the CLI."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select

from robot_skill_system.adapters.mock_robot import MockGripperAdapter, MockRobotAdapter
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
from robot_skill_system.openai_integration.schemas import DemonstrationAnalysisInput
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
from robot_skill_system.settings import ExecutionMode as SettingsExecutionMode
from robot_skill_system.settings import Settings
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.graph import SkillGraphValidator
from robot_skill_system.skills.loader import CompiledRun, load_compiled_run
from robot_skill_system.skills.models import (
    SkillGraph,
    SkillLifecycleStatus,
    SkillManifest,
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

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = LocalArtifactStore(settings.artifact_root)
        self.database = Database(settings.database_url)
        self.database.create_schema()
        self.repository = StorageRepository(self.database)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._scenes: dict[str, SceneSnapshot] = {}
        self._active_executions: dict[str, ActiveExecution] = {}
        self._active_execution_lock = threading.RLock()

    def close(self) -> None:
        """Release database resources."""

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
        scene = capture_mock_scene(frame_count=frame_count)
        self.repository.record_scene(scene)
        self.store.put_json(
            f"scenes/{scene.scene_id}.json", scene.model_dump(mode="json")
        )
        self._scenes[scene.scene_id] = scene
        return scene.model_dump(mode="json")

    def get_scene(self, scene_id: str) -> dict[str, Any]:
        return self._scene(scene_id).model_dump(mode="json")

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
