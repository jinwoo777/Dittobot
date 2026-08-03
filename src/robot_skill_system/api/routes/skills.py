"""Skill induction, retrieval, compilation, validation, and lifecycle routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    SkillActivateRequest,
    SkillCompileRequest,
    SkillInduceRequest,
    SkillRollbackRequest,
    SkillSearchRequest,
    SkillUpdateRequest,
    SkillValidateRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/skills", tags=["skills"])


@router.post("/induce")
def induce_skill(
    request: SkillInduceRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.induce_skill(request.model_dump())


@router.post("/search")
def search_skills(
    request: SkillSearchRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.search_skills(request.model_dump())


@router.get("/{skill_id}")
def get_skill(skill_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_skill(skill_id)


@router.get("/{skill_id}/versions")
def get_versions(skill_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_skill_versions(skill_id)


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
