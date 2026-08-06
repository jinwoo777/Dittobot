from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robot_skill_system.api.app import create_app
from robot_skill_system.api.contracts import (
    ActionCatalogCreateRequest,
    CatalogStatusRequest,
    ObjectCatalogCreateRequest,
)
from robot_skill_system.api.routes import catalog
from robot_skill_system.application import MVPApplication
from robot_skill_system.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )


def test_object_and_action_catalog_create_and_query_api(tmp_path: Path) -> None:
    service = MVPApplication(_settings(tmp_path))
    try:
        created_object = catalog.create_object(
            ObjectCatalogCreateRequest.model_validate(
                {
                "object_class_id": "hammer",
                "display_name": "망치",
                "aliases": ["망치", "해머"],
                }
            ),
            service,
        )
        assert created_object["status"] == "draft"

        created_action = catalog.create_action(
            ActionCatalogCreateRequest.model_validate(
                {
                "action_id": "bring",
                "display_name": "가져오기",
                "aliases": ["가져와", "옮겨줘"],
                "required_roles": ["$destination"],
                "input_contract": {"attachment": "holding"},
                "output_contract": {"attachment": "holding"},
                "anchor_policy": {"allowed": ["destination"]},
                }
            ),
            service,
        )
        action = created_action
        assert action["canonical_id"] == "bring"
        assert action["metadata"]["required_roles"] == ["$destination"]

        assert catalog.list_objects(service)["objects"][0][
            "canonical_id"
        ] == "hammer"
        assert catalog.get_action("bring", service)["aliases"] == [
            "가져와",
            "옮겨줘",
        ]
        assert catalog.get_action("가져와", service)["canonical_id"] == "bring"

        hierarchy = catalog.list_task_flows(service)
        assert hierarchy["has_static_object_action_allowlist"] is False
        assert hierarchy["objects"][0]["actions"][0]["canonical_id"] == "bring"
        assert hierarchy["objects"][0]["actions"][0]["executable"] is False

        # A semantic row alone is never enough to expose a runtime-active component.
        with pytest.raises(ValueError, match="active, passed stage"):
            catalog.set_action_status(
                "bring", CatalogStatusRequest(status="active"), service
            )

        skills = service.list_skills()
        assert skills["skills"] == []
        assert skills["task_flow_catalog"]["objects"][0]["canonical_id"] == "hammer"
    finally:
        service.close()


def test_catalog_rejects_alias_collision_and_unknown_fields(tmp_path: Path) -> None:
    service = MVPApplication(_settings(tmp_path))
    try:
        first = catalog.create_object(
            ObjectCatalogCreateRequest.model_validate(
                {
                "object_class_id": "cup",
                "display_name": "컵",
                "aliases": ["잔"],
                }
            ),
            service,
        )
        assert first["status"] == "draft"
        with pytest.raises(ValueError, match="already owned"):
            catalog.create_object(
                ObjectCatalogCreateRequest.model_validate(
                    {
                        "object_class_id": "glass",
                        "display_name": "유리잔",
                        "aliases": ["잔"],
                    }
                ),
                service,
            )

        with pytest.raises(ValidationError):
            ActionCatalogCreateRequest.model_validate(
                {
                "action_id": "wipe",
                "display_name": "닦기",
                "end_motion_id": "model_must_not_choose_this",
                }
            )

        paths = create_app(service).openapi()["paths"]
        assert "/catalog/objects" in paths
        assert "/catalog/actions/{action_id}" in paths
    finally:
        service.close()
