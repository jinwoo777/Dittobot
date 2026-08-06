"""Skill induction, retrieval, compilation, validation, and lifecycle routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    DraftAutoSurfaceCalibrationRequest,
    DraftCandidateRegistrationRequest,
    DraftSurfaceCalibrationRequest,
    DraftTCPPathRequest,
    RecordingSkillDraftRequest,
    SkillActivateRequest,
    SkillCompileRequest,
    SkillEditorCandidateRequest,
    SkillEditorPreviewRequest,
    SkillInduceRequest,
    SkillParameterCandidateRequest,
    SkillRollbackRequest,
    SkillSearchRequest,
    SkillUpdateRequest,
    SkillValidateRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("")
def list_skills(service: ServiceDependency) -> dict[str, Any]:
    return service.list_skills()


@router.get("/editor/catalog")
def get_skill_editor_catalog(service: ServiceDependency) -> dict[str, Any]:
    return service.get_skill_editor_catalog()


@router.post("/editor/preview")
def preview_skill_editor_blocks(
    request: SkillEditorPreviewRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.preview_skill_editor_blocks(request.model_dump(mode="json"))


@router.post("/editor/candidates")
def create_skill_editor_candidate(
    request: SkillEditorCandidateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.create_skill_editor_candidate(request.model_dump(mode="json"))


@router.post("/induce")
def induce_skill(
    request: SkillInduceRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.induce_skill(request.model_dump())


@router.get("/draft-from-recording/capabilities")
def recording_draft_capabilities(service: ServiceDependency) -> dict[str, Any]:
    return service.get_recording_skill_draft_capabilities()


@router.post("/draft-from-recording")
def create_recording_draft(
    request: RecordingSkillDraftRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.create_recording_skill_draft(request.model_dump())


@router.get("/drafts")
def list_recording_drafts(service: ServiceDependency) -> dict[str, Any]:
    return service.list_recording_skill_drafts()


@router.get("/drafts/{draft_id}")
def get_recording_draft(
    draft_id: str,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.get_recording_skill_draft(draft_id)


@router.post("/drafts/{draft_id}/surface-calibration")
def calibrate_recording_draft_surface(
    draft_id: str,
    request: DraftSurfaceCalibrationRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.calibrate_recording_draft_surface(draft_id, request.model_dump())


@router.post("/drafts/{draft_id}/surface-calibration/auto")
def auto_calibrate_recording_draft_surface(
    draft_id: str,
    request: DraftAutoSurfaceCalibrationRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.auto_calibrate_recording_draft_surface(
        draft_id, request.model_dump()
    )


@router.post("/drafts/{draft_id}/tcp-trajectory")
def create_recording_draft_tcp_trajectory(
    draft_id: str,
    request: DraftTCPPathRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.create_recording_draft_tcp_trajectory(
        draft_id, request.model_dump()
    )


@router.post("/drafts/{draft_id}/candidate")
def register_recording_draft_candidate(
    draft_id: str,
    request: DraftCandidateRegistrationRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.register_recording_draft_candidate(draft_id, request.model_dump())


@router.post("/search")
def search_skills(
    request: SkillSearchRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.search_skills(request.model_dump())


@router.get("/{skill_id}")
def get_skill(
    skill_id: str,
    service: ServiceDependency,
    version: str | None = None,
) -> dict[str, Any]:
    return service.get_skill(skill_id, version)


@router.get("/{skill_id}/versions")
def get_versions(skill_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_skill_versions(skill_id)


@router.delete("/{skill_id}")
def delete_skill(skill_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.delete_skill(skill_id)


@router.post("/{skill_id}/versions/{version}/parameter-candidates")
def create_skill_parameter_candidate(
    skill_id: str,
    version: str,
    request: SkillParameterCandidateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.create_skill_parameter_candidate(
        skill_id,
        version,
        request.model_dump(mode="json"),
    )


@router.post("/{skill_id}/compile")
def compile_skill(
    skill_id: str,
    request: SkillCompileRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.compile_skill(skill_id, request.model_dump())


@router.post("/{skill_id}/validate")
def validate_skill(
    skill_id: str,
    request: SkillValidateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.validate_skill(skill_id, request.model_dump())


@router.post("/{skill_id}/activate")
def activate_skill(
    skill_id: str,
    request: SkillActivateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.activate_skill(skill_id, request.model_dump())


@router.post("/{skill_id}/rollback")
def rollback_skill(
    skill_id: str,
    request: SkillRollbackRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.rollback_skill(skill_id, request.model_dump())


@router.post("/{skill_id}/update")
@router.post("/{skill_id}/updates", include_in_schema=True)
def update_skill(
    skill_id: str,
    request: SkillUpdateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.update_skill(skill_id, request.model_dump())


@router.post("/{skill_id}/versions/{version}/promote")
def promote_skill_version(
    skill_id: str, version: str, service: ServiceDependency
) -> dict[str, Any]:
    return service.activate_skill(skill_id, {"version": version})
