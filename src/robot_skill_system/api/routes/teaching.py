"""Teaching-session routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    TeachingCaptureRequest,
    TeachingFinishRequest,
    TeachingSessionCreate,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/teaching/sessions", tags=["teaching"])


@router.post("")
def create_session(
    request: TeachingSessionCreate, service: ServiceDependency
) -> dict[str, Any]:
    return service.create_teaching_session(request.model_dump())


@router.post("/{session_id}/capture")
@router.post("/{session_id}/frames", include_in_schema=True)
def capture_session(
    session_id: str,
    request: TeachingCaptureRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.capture_teaching_session(session_id, request.model_dump())


@router.post("/{session_id}/finish")
@router.post("/{session_id}/finalize", include_in_schema=True)
def finish_session(
    session_id: str,
    request: TeachingFinishRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.finish_teaching_session(session_id, request.model_dump())


@router.get("/{session_id}")
def get_session(session_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_teaching_session(session_id)
