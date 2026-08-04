"""Fail-closed eye-in-hand calibration routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    HandEyeCalibrationAbortRequest,
    HandEyeCalibrationStartRequest,
    HandEyeLegacyImportRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/calibration/hand-eye", tags=["calibration"])


@router.get("/status")
def status(service: ServiceDependency) -> dict[str, Any]:
    return service.get_handeye_calibration_status()


@router.post("/start")
def start(
    request: HandEyeCalibrationStartRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.start_handeye_calibration(request.model_dump())


@router.post("/abort")
def abort(
    request: HandEyeCalibrationAbortRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.abort_handeye_calibration(request.model_dump())


@router.post("/import-legacy-npy")
def import_legacy_npy(
    request: HandEyeLegacyImportRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.import_legacy_handeye_npy(request.model_dump())
