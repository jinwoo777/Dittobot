"""Operator-confirmed task-plane calibration catalog."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/calibration/task-planes", tags=["calibration"])


@router.get("")
def list_task_planes(service: ServiceDependency) -> dict[str, Any]:
    return service.list_task_planes()
