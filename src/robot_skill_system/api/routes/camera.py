"""Local RealSense RGB-D preview and bounded recording routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from robot_skill_system.api.contracts import CameraRecordingStartRequest
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/camera", tags=["camera"])


@router.get("/status")
def status(service: ServiceDependency) -> dict[str, Any]:
    return service.get_camera_status()


@router.post("/preview/start")
def start_preview(service: ServiceDependency) -> dict[str, Any]:
    return service.start_camera_preview()


@router.post("/preview/stop")
def stop_preview(service: ServiceDependency) -> dict[str, Any]:
    return service.stop_camera_preview()


@router.get("/streams/{kind}.mjpg")
def stream_preview(
    kind: Literal["rgb", "depth"], service: ServiceDependency
) -> StreamingResponse:
    status_payload = service.get_camera_status()
    if status_payload["state"] != "streaming":
        raise ValueError("RealSense preview is not streaming")
    return StreamingResponse(
        service.stream_camera_preview(kind),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/recordings")
def start_recording(
    request: CameraRecordingStartRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.start_camera_recording(request.model_dump())


@router.get("/recordings")
def list_recordings(service: ServiceDependency) -> dict[str, Any]:
    return service.list_camera_recordings()


@router.post("/recordings/{recording_id}/stop")
def stop_recording(recording_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.stop_camera_recording(recording_id)


@router.get("/recordings/{recording_id}")
def get_recording(recording_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_camera_recording(recording_id)


@router.get("/recordings/{recording_id}/frames/{frame_index}/{kind}.jpg")
def get_recording_frame(
    recording_id: str,
    frame_index: int,
    kind: Literal["rgb", "depth"],
    service: ServiceDependency,
) -> Response:
    return Response(
        content=service.get_camera_recording_frame(recording_id, frame_index, kind),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )
