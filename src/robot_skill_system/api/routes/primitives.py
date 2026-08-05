"""Primitive catalog routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.dependencies import ServiceDependency
from robot_skill_system.primitives.registry import get_default_registry

router = APIRouter(prefix="/primitives", tags=["primitives"])


@router.get("/catalog")
def get_primitive_catalog(
    service: ServiceDependency,
) -> dict[str, Any]:
    """Return primitive metadata and parameter schemas for the UI."""

    registry = get_default_registry()

    return {
        "primitives": [
            metadata.model_dump(mode="json")
            for metadata in registry.catalog()
        ]
    }
