"""HTTP request contracts; domain responses retain their native schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class RecordingSkillDraftRequest(APIModel):
    recording_id: str = Field(pattern=r"^rgbd_[A-Za-z0-9_-]{1,96}$")
    name_hint: str = Field(
        default="recorded_skill",
        pattern=r"^[a-z][a-z0-9_]{2,63}$",
    )
    operator_instruction: str = Field(min_length=1, max_length=2000)
    keyframe_count: int = Field(default=100, ge=1, le=300)
    tcp_landmark_confidence_threshold: float = Field(
        default=0.20, ge=0.10, le=0.90
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
    method: Literal[
        "manual_two_fingertip",
        "mediapipe_rgbd",
        "openai_rgbd",
        "openai_rgbd_operator_confirmed",
        "openai_rgbd_low_confidence_mock",
    ] = "manual_two_fingertip"
    annotations: list[TwoFingerFrameAnnotationRequest] = Field(
        default_factory=list, max_length=256
    )
    operator_confirmed: bool
    acknowledge_two_fingertip_tcp_proxy: bool = False
    acknowledge_low_confidence_mock_only: bool = False

    @model_validator(mode="after")
    def validate_method_evidence(self) -> DraftTCPPathRequest:
        if self.method == "manual_two_fingertip" and len(self.annotations) < 4:
            raise ValueError("manual TCP teaching requires at least four annotated frames")
        automatic_methods = {
            "mediapipe_rgbd",
            "openai_rgbd",
            "openai_rgbd_operator_confirmed",
            "openai_rgbd_low_confidence_mock",
        }
        if self.method in automatic_methods and self.annotations:
            raise ValueError("automatic TCP extraction does not accept manual annotations")
        if (
            self.method == "openai_rgbd_operator_confirmed"
            and not self.acknowledge_two_fingertip_tcp_proxy
        ):
            raise ValueError(
                "operator-confirmed GPT geometry requires acknowledgement that the "
                "two-fingertip midpoint is the intended robot TCP proxy"
            )
        if (
            self.method != "openai_rgbd_operator_confirmed"
            and self.acknowledge_two_fingertip_tcp_proxy
        ):
            raise ValueError(
                "two-fingertip TCP acknowledgement is valid only for the "
                "operator-confirmed method"
            )
        if (
            self.method == "openai_rgbd_low_confidence_mock"
            and not self.acknowledge_low_confidence_mock_only
        ):
            raise ValueError(
                "low-confidence GPT geometry requires explicit Mock-only acknowledgement"
            )
        if (
            self.method != "openai_rgbd_low_confidence_mock"
            and self.acknowledge_low_confidence_mock_only
        ):
            raise ValueError(
                "low-confidence acknowledgement is valid only for the Mock-only method"
            )
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


class SkillCommissionRequest(APIModel):
    version: str
    operator_id: str = Field(default="operator", min_length=1, max_length=64)
    operator_confirmed: bool
    workspace_cleared: bool
    estop_ready: bool
    path_reviewed: bool


class SkillRollbackRequest(APIModel):
    version: str


class SkillUpdateRequest(APIModel):
    demo_path: str
    base_version: str | None = None
    operator_role: str = "expert"
    has_force_measurements: bool = False


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


class RuntimeExecuteRequest(RuntimePreflightRequest):
    text: str | None = None
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class RuntimeAbortRequest(APIModel):
    run_id: str
    reason: str = "operator_request"


JSONDict = dict[str, Any]
