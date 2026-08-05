"""Deterministic offline substitutes for every OpenAI-dependent service."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .schemas import (
    APICallMetadata,
    ContactInterval,
    DemonstrationAnalysis,
    DemonstrationAnalysisInput,
    EmbeddingResult,
    HandShapeObservation,
    MotionStyle,
    NormalizedImagePoint,
    NormalizedImageRegion,
    PrimitiveRecommendation,
    RecordingSceneObservation,
    RecordingSkillDraft,
    RecordingSkillDraftInput,
    RecordingSkillDraftPrimitive,
    RuntimeIntent,
    SemanticPhase,
    SkillGraphProposal,
    SkillGraphProposalNode,
    TargetObjectObservation,
    TCPProxyFrameState,
    TCPProxyObservation,
    ToolShapeObservation,
    TranscriptResult,
    WorkSurfaceObservation,
)


class MockOpenAIClient:
    """Pure local semantic behavior used by tests, examples, and default runtime."""

    def analyze_demonstration(
        self, request: DemonstrationAnalysisInput, trace_id: str
    ) -> tuple[DemonstrationAnalysis, APICallMetadata]:
        requested_ops = [
            str(candidate.get("operation", "motion.move_l"))
            for candidate in request.motion_fitting_candidates
        ]
        candidate_ops = [item for item in requested_ops if item in request.primitive_catalog]
        if not candidate_ops and request.primitive_catalog:
            candidate_ops = [request.primitive_catalog[0]]
        entity_catalog = set(request.entity_catalog)
        target_ids = [
            str(item)
            for item in request.scene_summary.get("target_ids", ["table_surface_01"])
            if str(item) in entity_catalog
        ]
        requested_tool_id = str(request.scene_summary.get("tool_id", "wiper_01"))
        tool_id = requested_tool_id if requested_tool_id in entity_catalog else None
        involved_entity_ids = [*([tool_id] if tool_id is not None else []), *target_ids]
        motion_profile_id = _preferred_profile(
            request.approved_motion_profiles, "linear_normal"
        )
        force_profile_id = _preferred_profile(
            request.approved_force_profiles, "wipe_standard"
        )
        low_confidence = any(value < 0.5 for value in request.confidence_summary.values())
        phases = [
            SemanticPhase(
                interval_id=f"phase_{index}",
                label=operation,
                involved_entity_ids=involved_entity_ids,
                confidence=0.95 if not low_confidence else 0.45,
            )
            for index, operation in enumerate(candidate_ops)
        ]
        recommendations = [
            PrimitiveRecommendation(
                interval_id=f"phase_{index}",
                operation=operation,
                motion_profile_id=(motion_profile_id if operation.startswith("motion.") else None),
                force_profile_id=(force_profile_id if "contact" in operation else None),
                confidence=0.92,
            )
            for index, operation in enumerate(candidate_ops)
        ]
        analysis = DemonstrationAnalysis(
            task_name="wipe_surface",
            task_description="Bind a wiper to a contact surface and follow the taught path.",
            involved_entity_ids=involved_entity_ids,
            tool_id=tool_id,
            target_ids=target_ids,
            phase_labels=phases,
            contact_intervals=(
                [
                    ContactInterval(
                        interval_id="contact_0",
                        surface_id=target_ids[0],
                        likely_contact=True,
                        confidence=0.9,
                    )
                ]
                if target_ids
                else []
            ),
            motion_style=MotionStyle.EXPERT,
            repeated_patterns=["wipe_pass"],
            primitive_recommendations=recommendations,
            unresolved_ambiguities=(
                ["pose confidence below teaching threshold"] if low_confidence else []
            ),
            reteach_required=low_confidence,
            confidence=0.45 if low_confidence else 0.94,
            rationale_summary="Deterministic mock result combining local fit candidates.",
        )
        return analysis, APICallMetadata(trace_id=trace_id)

    def analyze_recording_skill_draft(
        self, request: RecordingSkillDraftInput, trace_id: str
    ) -> tuple[RecordingSkillDraft, APICallMetadata]:
        """Return a deterministic non-executable draft for offline UI/tests."""

        lowered = request.operator_instruction.lower()
        preferred_operations = (
            ["motion.move_l", "contact.search_surface", "contact.follow_path"]
            if "닦" in request.operator_instruction or "wipe" in lowered
            else ["motion.move_l"]
        )
        operations = [
            operation
            for operation in preferred_operations
            if operation in request.primitive_catalog
        ]
        if not operations:
            operations = [request.primitive_catalog[0]]
        preferred_roles = ["tool", "target_surface"]
        roles = [role for role in preferred_roles if role in request.entity_role_catalog]
        if not roles:
            roles = [request.entity_role_catalog[0]]
        frame_count = len(request.keyframe_indices)
        representative_index = (
            request.first_frame_index
            if request.visual_input_policy == "first_rgb_plus_local_fingertip_trace"
            and request.first_frame_index is not None
            else request.keyframe_indices[frame_count // 2]
        )
        tcp_states = [
            TCPProxyFrameState(
                frame_index=frame_index,
                gripper_state="pinching",
                landmarks_detected=True,
                jaw_tip_a_normalized=NormalizedImagePoint(
                    x=0.32 + 0.30 * position / max(1, frame_count - 1),
                    y=0.48,
                ),
                jaw_tip_b_normalized=NormalizedImagePoint(
                    x=0.38 + 0.30 * position / max(1, frame_count - 1),
                    y=0.48,
                ),
                midpoint_normalized=NormalizedImagePoint(
                    x=0.35 + 0.30 * position / max(1, frame_count - 1),
                    y=0.48,
                ),
                confidence=0.8,
            )
            for position, frame_index in enumerate(request.keyframe_indices)
        ]
        trajectory_usable = frame_count >= 4
        draft = RecordingSkillDraft(
            suggested_skill_id=request.name_hint,
            display_name=request.name_hint.replace("_", " "),
            task_description=request.operator_instruction,
            observed_task_summary=(
                f"Chronological review of {len(request.fingertip_trace)} local fingertip "
                "trace frames and one initial RGB frame."
                if request.visual_input_policy
                == "first_rgb_plus_local_fingertip_trace"
                else f"Chronological review of {len(request.keyframe_indices)} selected RGB frames."
            ),
            required_entity_roles=roles,
            primitive_sequence=[
                RecordingSkillDraftPrimitive(
                    operation=operation,
                    rationale="Allowed semantic operation suggested from the frame sequence.",
                    confidence=0.8,
                )
                for operation in operations
            ],
            scene_observation=RecordingSceneObservation(
                person_hand=HandShapeObservation(
                    detected=True,
                    shape="two_finger_gripper",
                    representative_frame_index=representative_index,
                    region_normalized=NormalizedImageRegion(
                        x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.70
                    ),
                    description="Mock hand uses two extended fingertips as gripper jaws.",
                    confidence=0.8,
                ),
                tool=ToolShapeObservation(
                    detected="tool" in roles,
                    shape="wiper" if "tool" in roles else "not_detected",
                    representative_frame_index=representative_index,
                    region_normalized=(
                        NormalizedImageRegion(
                            x_min=0.30, y_min=0.40, x_max=0.70, y_max=0.75
                        )
                        if "tool" in roles
                        else None
                    ),
                    description=(
                        "Mock elongated wiping tool."
                        if "tool" in roles
                        else "No separate tool is visible."
                    ),
                    confidence=0.75,
                ),
                target_object=TargetObjectObservation(
                    detected=False,
                    class_name=None,
                    representative_frame_index=representative_index,
                    region_normalized=None,
                    description="No separate target object is required for the mock surface task.",
                    confidence=0.7,
                ),
                work_surface=WorkSurfaceObservation(
                    detected=True,
                    shape="planar_rectangular",
                    representative_frame_index=representative_index,
                    region_normalized=NormalizedImageRegion(
                        x_min=0.05, y_min=0.30, x_max=0.95, y_max=0.95
                    ),
                    plane_likelihood=0.9,
                    description="Mock dominant workbench plane.",
                    confidence=0.9,
                ),
            ),
            tcp_proxy_observation=TCPProxyObservation(
                detected=True,
                observed_states=tcp_states,
                trajectory_status="complete",
                valid_landmark_frame_count=frame_count,
                usable_for_local_depth_path=trajectory_usable,
                failure_reason=(
                    None
                    if trajectory_usable
                    else "Fewer than four keyframes were supplied for a local depth path."
                ),
                depth_consistency="consistent",
                motion_summary=(
                    "Mock two-finger gripper proxy follows the demonstrated semantic path."
                ),
                confidence=0.8,
                semantic_only=True,
                robot_tcp_pose_available=False,
            ),
            unresolved_ambiguities=[
                "RGB frames do not provide robot-base pose, TF, or measured force evidence."
            ],
            confidence=0.8,
            executable=False,
            requires_pose_trajectory=True,
        )
        return draft, APICallMetadata(trace_id=trace_id)

    def resolve_intent(
        self,
        command: str,
        entity_ids: list[str],
        approved_motion_profiles: list[str],
        approved_force_profiles: list[str],
        trace_id: str,
    ) -> tuple[RuntimeIntent, APICallMetadata]:
        lowered = command.lower()
        style = (
            MotionStyle.EXPERT
            if "숙련" in command or "expert" in lowered
            else MotionStyle.NORMAL
        )
        tool = "blue wiper" if "파란" in command or "blue" in lowered else "wiper"
        target = (
            "right table surface"
            if "오른쪽" in command or "right" in lowered
            else "table surface"
        )
        repetitions = 2 if "두 번" in command or "twice" in lowered else 1
        intent = RuntimeIntent(
            intent="wipe_surface" if "닦" in command or "wipe" in lowered else "execute_skill",
            tool_query=tool,
            target_query=target,
            style=style,
            repetitions=repetitions,
            speed_profile_request=_preferred_profile(
                approved_motion_profiles, "linear_normal"
            ),
            force_profile_request=_preferred_profile(
                approved_force_profiles, "wipe_standard"
            ),
            candidate_entity_ids=entity_ids,
            confidence=0.96,
        )
        return intent, APICallMetadata(trace_id=trace_id)

    def compose_skill_graph(
        self,
        analysis: DemonstrationAnalysis,
        primitive_catalog: list[str],
        binding_ref_catalog: list[str],
        approved_motion_profiles: list[str],
        approved_force_profiles: list[str],
        trace_id: str,
    ) -> tuple[SkillGraphProposal, APICallMetadata]:
        requested = [item.operation for item in analysis.primitive_recommendations]
        operations = [operation for operation in requested if operation in primitive_catalog]
        binding_ref = binding_ref_catalog[0] if binding_ref_catalog else None
        motion_profile_id = _preferred_profile(approved_motion_profiles, "linear_normal")
        force_profile_id = _preferred_profile(approved_force_profiles, "wipe_standard")
        proposal = SkillGraphProposal(
            name=analysis.task_name,
            description=analysis.task_description,
            required_tools=[analysis.tool_id] if analysis.tool_id else [],
            required_entity_roles=(
                {binding_ref: "contact_target"} if binding_ref is not None else {}
            ),
            nodes=[
                SkillGraphProposalNode(
                    operation=operation,
                    binding_refs=[binding_ref] if binding_ref is not None else [],
                    motion_profile_id=(
                        motion_profile_id if operation.startswith("motion.") else None
                    ),
                    force_profile_id=(force_profile_id if "force" in operation else None),
                )
                for operation in operations
            ],
            unresolved_ambiguities=analysis.unresolved_ambiguities,
            confidence=analysis.confidence,
        )
        return proposal, APICallMetadata(trace_id=trace_id)

    def embed(self, text: str, model: str) -> EmbeddingResult:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(32)]
        magnitude = math.sqrt(sum(value * value for value in raw)) or 1.0
        return EmbeddingResult(vector=[value / magnitude for value in raw], model=model)

    def transcribe(self, audio_path: Path, model: str) -> TranscriptResult:
        sidecar = audio_path.with_suffix(".txt")
        text = (
            sidecar.read_text(encoding="utf-8").strip()
            if sidecar.exists()
            else "테이블을 닦아줘"
        )
        return TranscriptResult(
            text=text,
            language="ko",
            confidence=1.0,
            model=model,
            source_audio_uri=audio_path.name,
        )


def _preferred_profile(catalog: list[str], preferred: str) -> str | None:
    if preferred in catalog:
        return preferred
    return catalog[0] if catalog else None
