"""Runtime intent, binding, preflight, execution, abort, and run routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    RuntimeAbortRequest,
    RuntimeBindRequest,
    RuntimeExecuteRequest,
    RuntimePreflightRequest,
    RuntimeResolveRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/capabilities")
def capabilities(service: ServiceDependency) -> dict[str, Any]:
    return service.get_runtime_capabilities()


@router.post("/resolve")
@router.post("/intent", include_in_schema=True)
def resolve(
    request: RuntimeResolveRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.resolve_runtime(request.model_dump())


@router.post("/bind")
def bind(request: RuntimeBindRequest, service: ServiceDependency) -> dict[str, Any]:
    return service.bind_runtime(request.model_dump())


@router.post("/preflight")
@router.post("/validate", include_in_schema=True)
def preflight(
    request: RuntimePreflightRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.preflight_runtime(request.model_dump())


@router.post("/execute")
def execute(
    request: RuntimeExecuteRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.execute_runtime(request.model_dump())


@router.post("/abort")
def abort(request: RuntimeAbortRequest, service: ServiceDependency) -> dict[str, Any]:
    return service.abort_runtime(request.model_dump())


@router.get("/runs/{run_id}")
def get_run(run_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_runtime_run(run_id)
