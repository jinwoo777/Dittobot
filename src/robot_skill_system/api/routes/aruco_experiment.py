"""Narrow, fail-closed routes for the supervised ArUco +Z experiment."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    ArucoExperimentEnableRequest,
    ArucoExperimentStopRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/aruco-experiment", tags=["aruco-experiment"])


@router.get("/status")
def status(service: ServiceDependency) -> dict[str, Any]:
    return service.get_aruco_experiment_status()


@router.post("/enable")
def enable(
    request: ArucoExperimentEnableRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.enable_aruco_experiment(request.model_dump())


@router.post("/move-reference")
def move_reference(service: ServiceDependency) -> dict[str, Any]:
    return service.move_aruco_reference()


@router.post("/move-plane-z-test")
def move_plane_z_test(service: ServiceDependency) -> dict[str, Any]:
    return service.move_aruco_plane_z_test()


@router.post("/stop")
def stop(
    request: ArucoExperimentStopRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.stop_aruco_experiment(request.model_dump())
