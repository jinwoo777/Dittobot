"""Read-only ditto_ws coordinate capture integration routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import Response

from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/coordinates/ditto", tags=["coordinates"])


@router.get("/status")
def status(service: ServiceDependency) -> dict[str, Any]:
    return service.get_ditto_coordinate_status()


@router.post("/start")
def start(service: ServiceDependency) -> dict[str, Any]:
    return service.start_ditto_coordinate_capture()


@router.post("/stop")
def stop(service: ServiceDependency) -> dict[str, Any]:
    return service.stop_ditto_coordinate_capture()


@router.get("/latest")
def latest(service: ServiceDependency) -> dict[str, Any]:
    return service.get_latest_ditto_coordinates()


@router.get("/frame.jpg")
def frame(service: ServiceDependency) -> Response:
    return Response(
        content=service.get_ditto_coordinate_frame(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/results/{stage}/{filename}")
def result(
    stage: Literal["raw", "smooth", "verify"],
    filename: str,
    service: ServiceDependency,
) -> Any:
    if filename.endswith(".json"):
        return service.get_ditto_coordinate_json(stage, filename)
    if filename.endswith(".png"):
        return Response(
            content=service.get_ditto_coordinate_image(stage, filename),
            media_type="image/png",
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
            },
        )
    raise ValueError("ditto coordinate result must be JSON or PNG")
