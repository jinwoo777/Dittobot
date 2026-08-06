"""Strict semantic schemas accepted from OpenAI or deterministic mocks."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES = 6_000


class StrictModel(BaseModel):
    """Base schema that rejects model-supplied fields outside the contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")


class MotionStyle(str, Enum):
    SAFE = "safe"
    NORMAL = "normal"
    EXPERT = "expert"
    EXPERT_FAST = "expert_fast"
    EXPERT_PRECISE = "expert_precise"


class SemanticPhase(StrictModel):
    """Semantic label layered onto locally detected motion boundaries."""

    interval_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    involved_entity_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ContactInterval(StrictModel):
    interval_id: str = Field(min_length=1)
    surface_id: str
    likely_contact: bool
    confidence: float = Field(ge=0.0, le=1.0)


class PrimitiveRecommendation(StrictModel):
    """A whitelist name/profile suggestion, never raw motion parameters."""

    interval_id: str
    operation: str
    motion_profile_id: str | None = None
    force_profile_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class DemonstrationAnalysis(StrictModel):
    task_name: str = Field(min_length=1)
    task_description: str = Field(min_length=1)
    involved_entity_ids: list[str]
    tool_id: str | None = None
    target_ids: list[str]
    phase_labels: list[SemanticPhase]
    contact_intervals: list[ContactInterval]
    motion_style: MotionStyle
    repeated_patterns: list[str]
    primitive_recommendations: list[PrimitiveRecommendation]
    unresolved_ambiguities: list[str]
    reteach_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: str = Field(max_length=1000)


class DemonstrationAnalysisInput(StrictModel):
    transcript_text: str
    scene_summary: dict[str, Any]
    pose_summary: dict[str, Any]
    entity_catalog: list[str]
    primitive_catalog: list[str]
    approved_motion_profiles: list[str]
    approved_force_profiles: list[str]
    motion_fitting_candidates: list[dict[str, Any]]
    confidence_summary: dict[str, float]
    keyframe_paths: list[str] = Field(default_factory=list, max_length=12)


class RecordingSkillDraftPrimitive(StrictModel):
    """One chronological primitive suggestion without geometry or safety numbers."""

    operation: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    rationale: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class TCPProxyTeachingDefinition(StrictModel):
    """Operator-approved visual convention for a non-executable TCP proxy."""

    proxy_type: Literal["two_finger_gripper"] = "two_finger_gripper"
    jaw_tip_landmarks: Literal["two_visible_fingertips"] = "two_visible_fingertips"
    tcp_proxy_rule: Literal["midpoint_between_fingertips"] = (
        "midpoint_between_fingertips"
    )
    coordinate_policy: Literal["semantic_observation_only"] = "semantic_observation_only"


class NormalizedImagePoint(StrictModel):
    """Non-metric image location; local RGB-D code must recover 3-D geometry."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class NormalizedImageRegion(StrictModel):
    """Normalized image ROI supplied only as a hint to local depth processing."""

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> NormalizedImageRegion:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("normalized image region bounds must be ordered")
        return self


class CompactFingertipTraceLandmark(BaseModel):
    """One locally detected fingertip location without depth or metric geometry."""

    model_config = ConfigDict(extra="forbid")

    landmark_index: Literal[4, 8]
    normalized_xy: tuple[float, float] | None = None
    pixel_xy: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _coordinates_are_consistent(self) -> CompactFingertipTraceLandmark:
        if (self.normalized_xy is None) is not (self.pixel_xy is None):
            raise ValueError("trace landmark normalized and pixel coordinates are all-or-none")
        if self.normalized_xy is not None and not all(
            0.0 <= component <= 1.0 for component in self.normalized_xy
        ):
            raise ValueError("trace normalized coordinates must be in [0, 1]")
        if self.pixel_xy is not None and any(component < 0 for component in self.pixel_xy):
            raise ValueError("trace pixel coordinates must be non-negative")
        return self


class CompactFingertipTraceFrame(BaseModel):
    """Thumb/index-only image-space evidence for one full-recording frame."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    thumb_tip: CompactFingertipTraceLandmark
    index_tip: CompactFingertipTraceLandmark
    status: Literal["valid", "uncertain"]

    @model_validator(mode="after")
    def _landmark_ids_and_status_match(self) -> CompactFingertipTraceFrame:
        if self.thumb_tip.landmark_index != 4 or self.index_tip.landmark_index != 8:
            raise ValueError("compact trace must use MediaPipe thumb=4 and index=8")
        detected = (
            self.thumb_tip.normalized_xy is not None
            and self.index_tip.normalized_xy is not None
        )
        if (self.status == "valid") is not detected:
            raise ValueError(
                "compact trace status must describe image-space fingertip availability"
            )
        return self


class TCPProxyFrameState(StrictModel):
    """Audited two-finger state for exactly one supplied RGB-D frame pair."""

    frame_index: int = Field(ge=0)
    gripper_state: Literal["open", "pinching", "closed", "occluded", "uncertain"]
    landmarks_detected: bool
    jaw_tip_a_normalized: NormalizedImagePoint | None = None
    jaw_tip_b_normalized: NormalizedImagePoint | None = None
    midpoint_normalized: NormalizedImagePoint | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _landmark_fields_are_all_or_none(self) -> TCPProxyFrameState:
        landmarks = (
            self.jaw_tip_a_normalized,
            self.jaw_tip_b_normalized,
            self.midpoint_normalized,
        )
        if self.landmarks_detected and any(item is None for item in landmarks):
            raise ValueError("detected fingertip landmarks require both tips and midpoint")
        if not self.landmarks_detected and any(item is not None for item in landmarks):
            raise ValueError("undetected fingertip landmarks cannot contain image positions")
        return self


class TCPProxyObservation(StrictModel):
    """Frame-complete visual TCP audit with non-metric image coordinates."""

    detected: bool
    observed_states: list[TCPProxyFrameState] = Field(min_length=1, max_length=300)
    trajectory_status: Literal["complete", "partial", "not_detected"]
    valid_landmark_frame_count: int = Field(ge=0, le=300)
    usable_for_local_depth_path: bool
    failure_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    depth_consistency: Literal["consistent", "ambiguous", "inconsistent", "unavailable"]
    motion_summary: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_only: Literal[True] = True
    robot_tcp_pose_available: Literal[False] = False

    @model_validator(mode="after")
    def _detection_requires_evidence(self) -> TCPProxyObservation:
        valid_count = sum(item.landmarks_detected for item in self.observed_states)
        if self.valid_landmark_frame_count != valid_count:
            raise ValueError("valid_landmark_frame_count does not match observed states")
        if self.detected is not (valid_count > 0):
            raise ValueError("TCP proxy detected flag does not match landmark evidence")
        if self.usable_for_local_depth_path is not (valid_count >= 4):
            raise ValueError("local depth path requires at least four landmark frames")
        expected_status = (
            "not_detected"
            if valid_count == 0
            else "complete"
            if valid_count == len(self.observed_states)
            else "partial"
        )
        if self.trajectory_status != expected_status:
            raise ValueError("trajectory_status does not match frame landmark coverage")
        if self.usable_for_local_depth_path and self.failure_reason is not None:
            raise ValueError("usable TCP trajectory cannot contain a failure reason")
        if not self.usable_for_local_depth_path and self.failure_reason is None:
            raise ValueError("unusable TCP trajectory requires an explicit failure reason")
        return self


class HandShapeObservation(StrictModel):
    detected: bool
    shape: Literal[
        "two_finger_gripper", "open_hand", "closed_hand", "other", "occluded", "not_detected"
    ]
    representative_frame_index: int = Field(ge=0)
    region_normalized: NormalizedImageRegion | None = None
    description: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class ToolShapeObservation(StrictModel):
    detected: bool
    shape: Literal[
        "gripper", "wiper", "brush", "driver", "container", "other", "not_detected"
    ]
    representative_frame_index: int = Field(ge=0)
    region_normalized: NormalizedImageRegion | None = None
    description: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class TargetObjectObservation(StrictModel):
    """First-frame semantic target ROI; local depth owns its metric anchor."""

    detected: bool
    class_name: str | None = Field(default=None, min_length=1, max_length=80)
    representative_frame_index: int = Field(ge=0)
    region_normalized: NormalizedImageRegion | None = None
    description: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _detected_object_requires_a_region(self) -> TargetObjectObservation:
        if self.detected and (
            self.class_name is None or self.region_normalized is None
        ):
            raise ValueError("detected target objects require class_name and image region")
        if not self.detected and (
            self.class_name is not None or self.region_normalized is not None
        ):
            raise ValueError("undetected target objects cannot contain class_name or region")
        return self


class WorkSurfaceObservation(StrictModel):
    detected: bool
    shape: Literal[
        "planar_rectangular", "planar_irregular", "curved", "ambiguous", "not_detected"
    ]
    representative_frame_index: int = Field(ge=0)
    region_normalized: NormalizedImageRegion | None = None
    plane_likelihood: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class RecordingSceneObservation(StrictModel):
    """Semantic image regions; never a metric plane or execution authority."""

    person_hand: HandShapeObservation
    tool: ToolShapeObservation
    target_object: TargetObjectObservation | None = None
    work_surface: WorkSurfaceObservation


class RecordingSkillDraft(StrictModel):
    """Non-executable semantic interpretation of chronological RGB-D evidence."""

    suggested_skill_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    display_name: str = Field(min_length=1, max_length=120)
    task_description: str = Field(min_length=1, max_length=2000)
    observed_task_summary: str = Field(min_length=1, max_length=2000)
    required_entity_roles: list[str] = Field(default_factory=list, max_length=8)
    primitive_sequence: list[RecordingSkillDraftPrimitive] = Field(
        default_factory=list, max_length=32
    )
    scene_observation: RecordingSceneObservation
    tcp_proxy_observation: TCPProxyObservation
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    executable: Literal[False] = False
    requires_pose_trajectory: Literal[True] = True


class RecordingSkillDraftInput(StrictModel):
    """Bounded metadata paired with chronological aligned RGB-depth image pairs."""

    recording_id: str = Field(pattern=r"^rgbd_[A-Za-z0-9_-]{1,96}$")
    name_hint: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    operator_instruction: str = Field(min_length=1, max_length=2000)
    recording_summary: dict[str, Any]
    primitive_catalog: list[str] = Field(min_length=1, max_length=64)
    entity_role_catalog: list[str] = Field(min_length=1, max_length=16)
    keyframe_indices: list[int] = Field(min_length=1, max_length=300)
    first_frame_index: int | None = Field(default=None, ge=0)
    fingertip_trace: list[CompactFingertipTraceFrame] = Field(
        default_factory=list, max_length=MAXIMUM_COMPACT_FINGERTIP_TRACE_FRAMES
    )
    visual_input_policy: Literal[
        "rgbd_keyframes",
        "first_rgb_plus_local_fingertip_trace",
    ] = "rgbd_keyframes"
    image_pair_order: Literal["rgb_then_aligned_depth_per_keyframe"] = (
        "rgb_then_aligned_depth_per_keyframe"
    )
    depth_visualization: Literal["turbo_colormap_near_warm_invalid_black"] = (
        "turbo_colormap_near_warm_invalid_black"
    )
    tcp_proxy_definition: TCPProxyTeachingDefinition = Field(
        default_factory=TCPProxyTeachingDefinition
    )
    limitations: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _validate_full_recording_trace_policy(self) -> RecordingSkillDraftInput:
        if self.visual_input_policy == "rgbd_keyframes":
            return self
        if self.first_frame_index is None:
            raise ValueError("first-frame trace analysis requires first_frame_index")
        if self.keyframe_indices != [self.first_frame_index]:
            raise ValueError("first-frame trace analysis supplies exactly the first RGB frame")
        if not self.fingertip_trace:
            raise ValueError("first-frame trace analysis requires the full fingertip trace")
        indices = [item.frame_index for item in self.fingertip_trace]
        timestamps = [item.timestamp_ns for item in self.fingertip_trace]
        if len(set(indices)) != len(indices) or indices != sorted(indices):
            raise ValueError("compact fingertip trace frame indices must be unique and ordered")
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ValueError("compact fingertip trace timestamps must be strictly increasing")
        expected_count = int(self.recording_summary.get("frame_count") or 0)
        if expected_count and len(self.fingertip_trace) != expected_count:
            raise ValueError("compact fingertip trace must cover every manifest frame")
        if indices[0] != self.first_frame_index:
            raise ValueError("compact fingertip trace must start at first_frame_index")
        return self


class RuntimeIntent(StrictModel):
    intent: str
    skill_query: str | None = None
    tool_query: str | None = None
    target_query: str | None = None
    style: MotionStyle = MotionStyle.NORMAL
    repetitions: int = Field(default=1, ge=1, le=100)
    speed_profile_request: str | None = None
    force_profile_request: str | None = None
    candidate_entity_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    ambiguity: bool = False
    confidence_rationale: str | None = None


class TaskIntent(StrictModel):
    """Catalog-bounded Grip -> Action intent; local code selects End Motion.

    This schema deliberately contains semantic identifiers only.  Geometry,
    control values, component versions, and execution authority are excluded.
    """

    object_class_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    action_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    object_instance_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
    )
    # Keep the cardinality bound in the local validator instead of Field(max_length).
    # Pydantic serializes the latter as JSON Schema ``maxProperties``, which the
    # Responses API structured-output dialect does not accept.
    role_bindings: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: bool = False
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=32)
    confidence_rationale: str | None = Field(default=None, max_length=1000)

    @field_validator("role_bindings")
    @classmethod
    def validate_role_bindings(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 16:
            raise ValueError("role_bindings cannot contain more than 16 entries")
        role_pattern = re.compile(r"^\$[A-Za-z][A-Za-z0-9_.-]{0,63}$")
        entity_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
        for role, entity_id in value.items():
            if not role_pattern.fullmatch(role):
                raise ValueError("role binding keys must use $name syntax")
            if not entity_pattern.fullmatch(entity_id):
                raise ValueError("role bindings must contain scene entity identifiers")
        return value


class SkillInvocation(StrictModel):
    skill_id: str
    version: str
    bindings: dict[str, str]
    repetitions: int = Field(default=1, ge=1, le=100)
    motion_profile_id: str | None = None
    force_profile_id: str | None = None


class SkillGraphProposalNode(StrictModel):
    operation: str
    binding_refs: list[str] = Field(default_factory=list)
    motion_profile_id: str | None = None
    force_profile_id: str | None = None


class SkillGraphProposal(StrictModel):
    """Semantic graph outline; local code supplies and validates all geometry/numbers."""

    name: str
    description: str
    required_tools: list[str]
    required_entity_roles: dict[str, str]
    nodes: list[SkillGraphProposalNode]
    unresolved_ambiguities: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptSegment(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str


class TranscriptResult(StrictModel):
    text: str
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_s: float | None = Field(default=None, ge=0.0)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str
    response_id: str | None = None
    source_audio_uri: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["succeeded", "failed"] = "succeeded"
    error: str | None = None


class EmbeddingResult(StrictModel):
    vector: list[float]
    model: str
    response_id: str | None = None
    input_tokens: int | None = None


class APICallMetadata(StrictModel):
    trace_id: str
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    attempts: int = Field(default=1, ge=1)


class SafeFunctionName(str, Enum):
    SEARCH_SKILL_REGISTRY = "search_skill_registry"
    GET_SKILL_MANIFEST = "get_skill_manifest"
    GET_PRIMITIVE_CATALOG = "get_primitive_catalog"
    QUERY_SCENE_ENTITIES = "query_scene_entities"
    GET_SCENE_SUMMARY = "get_scene_summary"
    COMPARE_SKILL_VERSIONS = "compare_skill_versions"


class SkillUpdateProposal(StrictModel):
    """Semantic update selection; local comparison supplies all numeric deltas."""

    skill_id: str
    parent_version: str
    suggested_variant: str
    update_spatial_path: bool = False
    update_orientation_path: bool = False
    update_timing_profile: bool = False
    update_gripper_profile: bool = False
    update_force_profile: bool = False
    update_recovery_policy: bool = False
    semantic_differences: list[str] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
