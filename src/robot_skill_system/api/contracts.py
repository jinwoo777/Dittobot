"""HTTP request contracts; domain responses retain their native schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class SkillInduceRequest(APIModel):
    demo_path: str
    name: str = "wipe_surface"
    variant: str = "default"


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
