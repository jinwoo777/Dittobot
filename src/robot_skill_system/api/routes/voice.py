"""Wake Word status and charge-aware transcription routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/voice", tags=["voice"])

_MAXIMUM_AUDIO_BYTES = 8 * 1024 * 1024


@router.get("/capabilities")
def capabilities(service: ServiceDependency) -> dict[str, Any]:
    """Inspect wiring only; this endpoint never captures audio or contacts OpenAI."""

    return service.get_monitor_voice_capabilities()


@router.post("/transcribe")
async def transcribe(request: Request, service: ServiceDependency) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            content_length_value = int(content_length)
        except ValueError as exc:
            raise ValueError("voice audio Content-Length is invalid") from exc
        if content_length_value > _MAXIMUM_AUDIO_BYTES:
            raise ValueError("voice audio exceeds the 8 MiB request limit")
    audio = await request.body()
    if len(audio) > _MAXIMUM_AUDIO_BYTES:
        raise ValueError("voice audio exceeds the 8 MiB request limit")
    acknowledged = (
        request.headers.get("x-acknowledge-openai-charges", "").strip().lower()
        == "true"
    )
    return await run_in_threadpool(
        service.transcribe_monitor_voice,
        audio,
        request.headers.get("content-type", ""),
        acknowledged,
    )
