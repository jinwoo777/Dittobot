"""Fail-closed, discrete joint jog routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    JogEnableRequest,
    JogJointMoveRequest,
    JogStopRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/jog", tags=["jog"])


@router.get("/status")
def status(service: ServiceDependency) -> dict[str, Any]:
    return service.get_jog_status()


@router.post("/enable")
def enable(request: JogEnableRequest, service: ServiceDependency) -> dict[str, Any]:
    return service.enable_jog(request.model_dump())


@router.post("/joints")
def move_joint(
    request: JogJointMoveRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.move_jog_joint(request.model_dump())


@router.post("/stop")
def stop(request: JogStopRequest, service: ServiceDependency) -> dict[str, Any]:
    return service.stop_jog(request.model_dump())
