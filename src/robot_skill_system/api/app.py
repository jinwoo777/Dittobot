"""FastAPI application factory; importing the core does not require FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_app(service: Any | None = None) -> Any:
    """Create the HTTP application around an injected or default MVP service."""

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - environment-dependent guard
        raise RuntimeError("Install the 'api' optional dependencies to run FastAPI") from exc

    from robot_skill_system.api.routes import calibration, camera, primitives, runtime, scenes, skills, teaching
    from robot_skill_system.capture.rgbd_recording import CameraStateError
    from robot_skill_system.exceptions import HardwareExecutionDenied, NotConfiguredError

    if service is None:
        from robot_skill_system.application import create_application

        service = create_application()
    app = FastAPI(
        title="Robot Skill System",
        version="0.1.0",
        description="Safety-bounded teaching, skill registry, and runtime API",
    )
    app.state.service = service

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "default_execution_mode": "mock"}

    app.include_router(teaching.router)
    app.include_router(scenes.router)
    app.include_router(skills.router)
    app.include_router(runtime.router)
    app.include_router(camera.router)
    app.include_router(calibration.router)
    app.include_router(primitives.router)

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

    # Retain FastAPI's type in the generated OpenAPI graph without importing it in core modules.
    _ = HTTPException
    return app
