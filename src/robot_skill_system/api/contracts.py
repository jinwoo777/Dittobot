"""HTTP request contracts; domain responses retain their native schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robot_skill_system.skills.models import BindingSpec, SkillType

CanonicalCatalogID = Annotated[
    str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
]
SemanticAlias = Annotated[str, Field(min_length=1, max_length=200)]
RequiredRole = Annotated[
    str, Field(pattern=r"^\$[A-Za-z][A-Za-z0-9_.-]{0,63}$")
]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TeachingSessionCreate(APIModel):
    operator_id: str = "operator_mock"
    operator_role: Literal["novice", "operator", "expert"] = "operator"
    notes: str | None = None


class TeachingCaptureRequest(APIModel):
    mode: Literal["mock", "single", "burst"] = "mock"


class TeachingFinishRequest(APIModel):
    success: bool = True
    transcript_text: str = "테이블을 닦는다"


class SceneCaptureRequest(APIModel):
    mode: Literal["mock", "single", "burst"] = "mock"
    fixture_path: str | None = None


class CameraRecordingStartRequest(APIModel):
    maximum_duration_s: float = Field(default=30.0, ge=1.0, le=600.0)


class HandEyeCalibrationStartRequest(APIModel):
    operator_id: str = Field(default="operator", min_length=1, max_length=64)
    operator_confirmed: bool
    board_secured: bool
    workspace_cleared: bool
    estop_ready: bool

    @model_validator(mode="after")
    def validate_safety_acknowledgements(self) -> HandEyeCalibrationStartRequest:
        acknowledgements = (
            self.operator_confirmed,
            self.board_secured,
            self.workspace_cleared,
            self.estop_ready,
        )
        if not all(acknowledgements):
            raise ValueError("all hand-eye calibration safety acknowledgements are required")
        return self


class HandEyeCalibrationAbortRequest(APIModel):
    reason: str = Field(default="operator_request", min_length=1, max_length=128)


class JogEnableRequest(APIModel):
    operator_id: str = Field(default="ui_operator", min_length=1, max_length=64)
    workspace_cleared: bool
    estop_ready: bool
    acknowledge_direct_motion: bool

    @model_validator(mode="after")
    def validate_safety_acknowledgements(self) -> JogEnableRequest:
        if not all(
            (self.workspace_cleared, self.estop_ready, self.acknowledge_direct_motion)
        ):
            raise ValueError("all jog safety acknowledgements are required")
        return self


class JogJointMoveRequest(APIModel):
    joint_index: int = Field(ge=1, le=6)
    delta_deg: float = Field(ge=-5.0, le=5.0)

    @model_validator(mode="after")
    def validate_non_zero_delta(self) -> JogJointMoveRequest:
        if self.delta_deg == 0.0:
            raise ValueError("jog delta must be non-zero")
        return self


class JogStopRequest(APIModel):
    reason: str = Field(default="operator_request", min_length=1, max_length=128)


class HandEyeLegacyImportRequest(APIModel):
    operator_id: str = Field(default="operator", min_length=1, max_length=64)
    operator_confirmed: bool
    acknowledge_candidate_only: bool

    @model_validator(mode="after")
    def validate_candidate_acknowledgements(self) -> HandEyeLegacyImportRequest:
        if not self.operator_confirmed or not self.acknowledge_candidate_only:
            raise ValueError("legacy NPY import requires candidate-only acknowledgement")
        return self


class SkillInduceRequest(APIModel):
    demo_path: str
    name: str = "wipe_surface"
    variant: str = "default"
    transcript_text: str | None = Field(default=None, min_length=1, max_length=2000)
    transcript_artifact_uri: str | None = Field(default=None, min_length=1, max_length=500)
    transcript_artifact_checksum_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_training_semantic_source(self) -> SkillInduceRequest:
        artifact_pair = (
            self.transcript_artifact_uri,
            self.transcript_artifact_checksum_sha256,
        )
        if (artifact_pair[0] is None) is not (artifact_pair[1] is None):
            raise ValueError("transcript artifact URI and checksum must be provided together")
        if self.transcript_text is not None and artifact_pair[0] is not None:
            raise ValueError("provide exactly one training transcript source")
        return self


class RecordingSkillDraftRequest(APIModel):
    recording_id: str = Field(pattern=r"^rgbd_[A-Za-z0-9_-]{1,96}$")
    name_hint: str = Field(
        default="recorded_skill",
        pattern=r"^[a-z][a-z0-9_]{2,63}$",
    )
    operator_instruction: str = Field(min_length=1, max_length=2000)
    keyframe_count: int = Field(
        default=1,
        ge=1,
        le=300,
        deprecated=True,
        description=(
            "Deprecated and ignored. Drafting always sends the first manifest RGB frame and "
            "tracks every manifest frame locally."
        ),
    )


class PixelPointRequest(APIModel):
    x_px: float = Field(ge=0.0, le=8192.0)
    y_px: float = Field(ge=0.0, le=8192.0)


class DraftSurfaceCalibrationRequest(APIModel):
    frame_index: int = Field(ge=0)
    surface_anchor_id: str = Field(
        default="teaching_surface",
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
    )
    origin_px: PixelPointRequest
    positive_x_px: PixelPointRequest
    positive_y_px: PixelPointRequest
    operator_confirmed: bool


class DraftAutoSurfaceCalibrationRequest(APIModel):
    frame_index: int | None = Field(default=None, ge=0)
    surface_anchor_id: str = Field(
        default="teaching_surface",
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
    )
    operator_confirmed: bool


class TwoFingerFrameAnnotationRequest(APIModel):
    frame_index: int = Field(ge=0)
    jaw_tip_a_px: PixelPointRequest
    jaw_tip_b_px: PixelPointRequest


class DraftTCPPathRequest(APIModel):
    method: Literal["manual_two_fingertip", "mediapipe_rgbd", "openai_rgbd"] = (
        "manual_two_fingertip"
    )
    annotations: list[TwoFingerFrameAnnotationRequest] = Field(
        default_factory=list, max_length=256
    )
    operator_confirmed: bool

    @model_validator(mode="after")
    def validate_method_evidence(self) -> DraftTCPPathRequest:
        if self.method == "manual_two_fingertip" and len(self.annotations) < 2:
            raise ValueError("manual TCP teaching requires at least two annotated frames")
        if self.method in {"mediapipe_rgbd", "openai_rgbd"} and self.annotations:
            raise ValueError("automatic TCP extraction does not accept manual annotations")
        return self


class DraftCandidateRegistrationRequest(APIModel):
    acknowledge_mock_only: bool


class SkillSearchRequest(APIModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    tool_class: str | None = None
    target_type: str | None = None
    style: str | None = None


class SkillCompileRequest(APIModel):
    version: str | None = None


class SkillValidateRequest(APIModel):
    version: str | None = None
    mode: Literal["mock", "simulation"] = "mock"


class SkillActivateRequest(APIModel):
    version: str


class SkillRollbackRequest(APIModel):
    version: str


class SkillUpdateRequest(APIModel):
    demo_path: str
    base_version: str | None = None
    operator_role: str = "expert"
    has_force_measurements: bool = False


class SkillEditorBlockRequest(APIModel):
    """One operation in the deliberately sequential block editor."""

    operation: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    arguments: dict[str, Any] = Field(default_factory=dict)


class SkillEditorPreviewRequest(APIModel):
    """Typed, code-free input shared by block preview and Candidate creation."""

    skill_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=2000)
    skill_type: SkillType = SkillType.COMPOSITE
    blocks: list[SkillEditorBlockRequest] = Field(min_length=1, max_length=128)
    bindings: dict[str, BindingSpec] = Field(default_factory=dict)


class SkillEditorCandidateRequest(SkillEditorPreviewRequest):
    """Explicit acknowledgement required before persisting a Mock-only Candidate."""

    acknowledge_mock_only: bool

    @model_validator(mode="after")
    def validate_mock_acknowledgement(self) -> SkillEditorCandidateRequest:
        if not self.acknowledge_mock_only:
            raise ValueError("block Candidate creation requires Mock-only acknowledgement")
        return self


class SkillParameterEditRequest(APIModel):
    """Complete replacement arguments for one existing immutable graph node."""

    node_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")
    arguments: dict[str, Any] = Field(default_factory=dict)


class SkillParameterCandidateRequest(APIModel):
    """Checksum-guarded node argument edits that always create a child version."""

    expected_parent_checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    edits: list[SkillParameterEditRequest] = Field(min_length=1, max_length=128)
    acknowledge_mock_only: bool

    @model_validator(mode="after")
    def validate_parameter_candidate(self) -> SkillParameterCandidateRequest:
        if not self.acknowledge_mock_only:
            raise ValueError("parameter Candidate creation requires Mock-only acknowledgement")
        node_ids = [edit.node_id for edit in self.edits]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("parameter Candidate edits must have unique node_id values")
        return self


class RuntimeResolveRequest(APIModel):
    text: str
    scene_id: str | None = None


class RuntimeBindRequest(APIModel):
    skill_id: str
    version: str | None = None
    scene_id: str
    entity_hints: dict[str, str] = Field(default_factory=dict)


class RuntimePreflightRequest(APIModel):
    skill_id: str
    version: str | None = None
    scene_id: str
    bindings: dict[str, str] = Field(default_factory=dict)
    mode: Literal["mock", "dry_run", "simulation", "hardware"] = "dry_run"


class MockValidationOverrideRequest(APIModel):
    """Per-run acknowledgement for overridable Mock-only quality gates."""

    override_all_overridable: Literal[True]
    operator_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=500)
    acknowledge_mock_only: Literal[True]


class RuntimeExecuteRequest(RuntimePreflightRequest):
    text: str | None = None
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
    mock_override: MockValidationOverrideRequest | None = None


class RuntimeAbortRequest(APIModel):
    run_id: str
    reason: str = "operator_request"


JSONDict = dict[str, Any]


class ObjectCatalogCreateRequest(APIModel):
    object_class_id: CanonicalCatalogID
    display_name: str = Field(min_length=1, max_length=200)
    aliases: list[SemanticAlias] = Field(default_factory=list, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionCatalogCreateRequest(APIModel):
    action_id: CanonicalCatalogID
    display_name: str = Field(min_length=1, max_length=200)
    aliases: list[SemanticAlias] = Field(default_factory=list, max_length=128)
    required_roles: list[RequiredRole] = Field(default_factory=list, max_length=32)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    anchor_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EndMotionCatalogCreateRequest(APIModel):
    end_motion_id: CanonicalCatalogID
    display_name: str = Field(min_length=1, max_length=200)
    aliases: list[SemanticAlias] = Field(default_factory=list, max_length=128)
    required_roles: list[RequiredRole] = Field(default_factory=list, max_length=32)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    anchor_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CatalogEntryUpdateRequest(APIModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    aliases: list[SemanticAlias] | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


class CatalogStatusRequest(APIModel):
    status: Literal["active", "retired"]


class StageStateContractRequest(APIModel):
    attachment: Literal["any", "holding", "released"]
    force_mode: Literal["any", "disabled", "enabled"]


class StageDefinitionCreateRequest(APIModel):
    component_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-candidate)?$")
    skill_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
    skill_version: str = Field(min_length=1, max_length=64)
    required_roles: list[RequiredRole] = Field(default_factory=list, max_length=32)
    input_contract: StageStateContractRequest
    output_contract: StageStateContractRequest
    anchor_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionEndMappingCreateRequest(APIModel):
    end_motion_id: CanonicalCatalogID
    revision: int | None = Field(default=None, ge=1)


class GripPointImportRequest(APIModel):
    artifact_path: str = "data/test/grip_point/result.json"
