"""Strict semantic schemas accepted from OpenAI or deterministic mocks."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
