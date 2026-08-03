"""Scene capture/read routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import SceneCaptureRequest
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(tags=["scenes"])


@router.post("/scenes/capture")
@router.post("/perception/scene/capture", include_in_schema=True)
def capture_scene(
    request: SceneCaptureRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.capture_scene(request.model_dump())


@router.get("/scenes/{scene_id}")
@router.get("/perception/scenes/{scene_id}", include_in_schema=True)
def get_scene(scene_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_scene(scene_id)
