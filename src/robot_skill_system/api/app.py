"""FastAPI application factory; importing the core does not require FastAPI."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


def create_app(service: Any | None = None) -> Any:
    """Create the HTTP application around an injected or default MVP service."""

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - environment-dependent guard
        raise RuntimeError("Install the 'api' optional dependencies to run FastAPI") from exc

    from robot_skill_system.api.routes import (
        aruco_experiment,
        calibration,
        camera,
        catalog,
        coordinates,
        jog,
        runtime,
        scenes,
        skills,
        task_planes,
        teaching,
        voice,
    )
    from robot_skill_system.capture.rgbd_recording import CameraStateError
    from robot_skill_system.exceptions import HardwareExecutionDenied, NotConfiguredError
    from robot_skill_system.runtime.errors import PreflightError

    owns_service = service is None
    if owns_service:
        from robot_skill_system.application import create_application
        from robot_skill_system.settings import Settings

        settings = Settings.from_env()
        # The HTTP server owns the operator-facing microphone monitor. Start its
        # free, local Wake Word detector automatically unless an operator has
        # explicitly disabled it. CLI and test-created services remain mock-safe.
        if "ENABLE_DITTO_WAKE_WORD" not in os.environ:
            settings = settings.model_copy(update={"enable_ditto_wake_word": True})
        service = create_application(settings)

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_service:
                close = getattr(service, "close", None)
                if callable(close):
                    close()

    app = FastAPI(
        title="Robot Skill System",
        version="0.1.0",
        description="Safety-bounded teaching, skill registry, and runtime API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8010",
            "http://localhost:8010",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Content-Type",
            "X-Acknowledge-OpenAI-Charges",
        ],
    )
    app.state.service = service

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "default_execution_mode": "mock"}

    app.include_router(teaching.router)
    app.include_router(scenes.router)
    app.include_router(skills.router)
    app.include_router(runtime.router)
    app.include_router(catalog.router)
    app.include_router(camera.router)
    app.include_router(coordinates.router)
    app.include_router(calibration.router)
    app.include_router(jog.router)
    app.include_router(aruco_experiment.router)
    app.include_router(task_planes.router)
    app.include_router(voice.router)

    settings = getattr(service, "settings", None)
    repository_root = getattr(settings, "repo_root", None)
    ui_root = Path(repository_root) / "dittobot-design" if repository_root else None
    if ui_root is not None and ui_root.is_dir():

        @app.get("/ui", include_in_schema=False)
        def ui_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

        app.mount("/ui", StaticFiles(directory=ui_root, html=True), name="ui")

    @app.exception_handler(KeyError)
    async def not_found(_request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_request(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(CameraStateError)
    async def camera_conflict(_request: Request, exc: CameraStateError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotConfiguredError)
    async def optional_adapter_missing(
        _request: Request, exc: NotConfiguredError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(HardwareExecutionDenied)
    async def hardware_denied(
        _request: Request, exc: HardwareExecutionDenied
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(PreflightError)
    async def preflight_rejected(
        _request: Request, exc: PreflightError
    ) -> JSONResponse:
        # A failed live validation is a request result, not an unhandled server
        # error.  The UI can surface this detail directly instead of showing a
        # misleading generic "Internal Server Error" banner.
        return JSONResponse(status_code=422, content={"detail": str(exc), "code": exc.code})

    # Retain FastAPI's type in the generated OpenAPI graph without importing it in core modules.
    _ = HTTPException
    return app
