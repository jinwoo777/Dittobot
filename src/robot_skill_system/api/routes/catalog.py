"""Semantic component catalog and Grip→Action→End hierarchy routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from robot_skill_system.api.contracts import (
    ActionCatalogCreateRequest,
    ActionEndMappingCreateRequest,
    CatalogEntryUpdateRequest,
    CatalogStatusRequest,
    EndMotionCatalogCreateRequest,
    GripPointImportRequest,
    ObjectCatalogCreateRequest,
    StageDefinitionCreateRequest,
)
from robot_skill_system.api.dependencies import ServiceDependency

router = APIRouter(prefix="/catalog", tags=["task-flow-catalog"])


@router.get("/objects")
def list_objects(
    service: ServiceDependency, status: str | None = None
) -> dict[str, Any]:
    return service.list_catalog_objects(status=status)


@router.post("/objects")
def create_object(
    request: ObjectCatalogCreateRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.create_catalog_object(request.model_dump(mode="json"))


@router.get("/objects/{object_class_id}")
def get_object(object_class_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_catalog_object(object_class_id)


@router.patch("/objects/{object_class_id}")
def update_object(
    object_class_id: str,
    request: CatalogEntryUpdateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.update_catalog_object(
        object_class_id, request.model_dump(mode="json", exclude_none=True)
    )


@router.post("/objects/{object_class_id}/status")
def set_object_status(
    object_class_id: str,
    request: CatalogStatusRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.set_catalog_object_status(object_class_id, request.status)


@router.get("/objects/{object_class_id}/grip-profiles")
def list_grip_profiles(
    object_class_id: str, service: ServiceDependency
) -> dict[str, Any]:
    return service.list_grip_profiles(object_class_id)


@router.get("/actions")
def list_actions(
    service: ServiceDependency, status: str | None = None
) -> dict[str, Any]:
    return service.list_catalog_actions(status=status)


@router.post("/actions")
def create_action(
    request: ActionCatalogCreateRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.create_catalog_action(request.model_dump(mode="json"))


@router.get("/actions/{action_id}")
def get_action(action_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_catalog_action(action_id)


@router.patch("/actions/{action_id}")
def update_action(
    action_id: str,
    request: CatalogEntryUpdateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.update_catalog_action(
        action_id, request.model_dump(mode="json", exclude_none=True)
    )


@router.post("/actions/{action_id}/definitions")
def create_action_definition(
    action_id: str,
    request: StageDefinitionCreateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.create_action_definition(action_id, request.model_dump(mode="json"))


@router.put("/actions/{action_id}/end-motion")
def map_action_end_motion(
    action_id: str,
    request: ActionEndMappingCreateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.map_action_end_motion(action_id, request.model_dump(mode="json"))


@router.post("/actions/{action_id}/status")
def set_action_status(
    action_id: str,
    request: CatalogStatusRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.set_catalog_action_status(action_id, request.status)


@router.get("/end-motions")
def list_end_motions(
    service: ServiceDependency, status: str | None = None
) -> dict[str, Any]:
    return service.list_catalog_end_motions(status=status)


@router.post("/end-motions")
def create_end_motion(
    request: EndMotionCatalogCreateRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.create_catalog_end_motion(request.model_dump(mode="json"))


@router.get("/end-motions/{end_motion_id}")
def get_end_motion(end_motion_id: str, service: ServiceDependency) -> dict[str, Any]:
    return service.get_catalog_end_motion(end_motion_id)


@router.post("/end-motions/{end_motion_id}/definitions")
def create_end_motion_definition(
    end_motion_id: str,
    request: StageDefinitionCreateRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.create_end_motion_definition(
        end_motion_id, request.model_dump(mode="json")
    )


@router.post("/stage-definitions/{definition_id}/activate")
def activate_stage_definition(
    definition_id: str, service: ServiceDependency
) -> dict[str, Any]:
    return service.activate_stage_definition(definition_id)


@router.post("/grip-profile-versions/{version_id}/activate")
def activate_grip_profile_version(
    version_id: str, service: ServiceDependency
) -> dict[str, Any]:
    return service.activate_grip_profile_version(version_id)


@router.post("/end-motions/{end_motion_id}/status")
def set_end_motion_status(
    end_motion_id: str,
    request: CatalogStatusRequest,
    service: ServiceDependency,
) -> dict[str, Any]:
    return service.set_catalog_end_motion_status(end_motion_id, request.status)


@router.post("/grip-profiles/import-grip-points")
def import_grip_points(
    request: GripPointImportRequest, service: ServiceDependency
) -> dict[str, Any]:
    return service.import_grip_point_candidates(request.model_dump(mode="json"))


@router.get("/task-flows")
def list_task_flows(service: ServiceDependency) -> dict[str, Any]:
    return service.list_task_flow_hierarchy()
