from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import robot_skill_system.openai_integration.task_intent_resolver as resolver_module
from robot_skill_system.exceptions import SemanticCatalogViolationError
from robot_skill_system.openai_integration.schemas import APICallMetadata, TaskIntent
from robot_skill_system.openai_integration.semantic_validation import validate_task_intent
from robot_skill_system.openai_integration.task_intent_resolver import TaskIntentResolver
from robot_skill_system.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env({"ARTIFACT_ROOT": str(tmp_path)}, root=tmp_path)


def test_task_intent_contains_semantics_but_no_end_or_geometry() -> None:
    intent = TaskIntent(
        object_class_id="hammer",
        action_id="bring",
        object_instance_id="hammer_01",
        role_bindings={"$destination": "handoff_region"},
        confidence=0.9,
    )

    assert set(intent.model_dump()) == {
        "schema_version",
        "object_class_id",
        "action_id",
        "object_instance_id",
        "role_bindings",
        "confidence",
        "ambiguity",
        "unresolved_ambiguities",
        "confidence_rationale",
    }
    with pytest.raises(ValidationError):
        TaskIntent.model_validate(
            {
                **intent.model_dump(),
                "end_motion_id": "release_and_retract",
            }
        )
    with pytest.raises(ValidationError):
        TaskIntent.model_validate(
            {
                **intent.model_dump(),
                "grasp_pose": {"x": 1.0},
            }
        )


def test_task_intent_keeps_role_bound_out_of_openai_json_schema() -> None:
    schema = TaskIntent.model_json_schema()

    assert "maxProperties" not in schema["properties"]["role_bindings"]
    with pytest.raises(ValidationError, match="more than 16"):
        TaskIntent(
            object_class_id="hammer",
            action_id="bring",
            role_bindings={f"$role_{index}": f"entity_{index}" for index in range(17)},
            confidence=0.9,
        )


def test_mock_task_intent_resolves_catalog_ids_and_scene_roles(tmp_path: Path) -> None:
    intent, metadata = TaskIntentResolver(_settings(tmp_path)).resolve(
        "망치를 가져와",
        object_catalog={"hammer": ("망치",)},
        action_catalog={"bring": ("가져와", "가져오기")},
        scene_entities={
            "hammer_01": {"kind": "object", "class_or_role": "hammer"},
            "handoff_region": {
                "kind": "workspace_region",
                "class_or_role": "destination",
            },
        },
        required_roles={"bring": ("$destination",)},
    )

    assert intent.object_class_id == "hammer"
    assert intent.action_id == "bring"
    assert intent.object_instance_id == "hammer_01"
    assert intent.role_bindings == {"$destination": "handoff_region"}
    assert intent.ambiguity is False
    assert metadata.trace_id


def test_live_wire_role_array_is_converted_to_internal_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path),
            "OPENAI_MODE": "live",
            "OPENAI_API_KEY": "test-key",
        },
        root=tmp_path,
    )

    def fake_parse_structured_response(**kwargs: object) -> tuple[object, APICallMetadata]:
        output_type = kwargs["output_type"]
        response = output_type.model_validate(  # type: ignore[union-attr]
            {
                "object_class_id": "hammer",
                "action_id": "bring",
                "object_instance_id": "hammer_01",
                "role_bindings": [
                    {"role": "$destination", "entity_id": "handoff_region"}
                ],
                "confidence": 0.98,
                "ambiguity": False,
                "unresolved_ambiguities": [],
                "confidence_rationale": "catalog match",
            }
        )
        return response, APICallMetadata(trace_id="live-wire-test")

    monkeypatch.setattr(
        resolver_module, "parse_structured_response", fake_parse_structured_response
    )
    intent, _metadata = TaskIntentResolver(settings, client=object()).resolve(  # type: ignore[arg-type]
        "망치 가져와",
        object_catalog={"hammer": ("망치",)},
        action_catalog={"bring": ("가져와",)},
        scene_entities={
            "hammer_01": {"kind": "object", "class_or_role": "hammer"},
            "handoff_region": {
                "kind": "workspace_region",
                "class_or_role": "destination",
            },
        },
        required_roles={"bring": ("$destination",)},
    )

    assert intent.role_bindings == {"$destination": "handoff_region"}


def test_task_intent_catalog_validation_rejects_unknown_selection() -> None:
    intent = TaskIntent(
        object_class_id="knife",
        action_id="wipe",
        object_instance_id="knife_01",
        role_bindings={"$surface": "surface_01"},
        confidence=0.8,
    )

    with pytest.raises(SemanticCatalogViolationError, match="object_class_id"):
        validate_task_intent(
            intent,
            object_catalog={"hammer"},
            action_catalog={"wipe"},
            entity_catalog={"knife_01", "surface_01"},
            allowed_roles={"$surface"},
        )
