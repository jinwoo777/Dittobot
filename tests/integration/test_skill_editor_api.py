from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robot_skill_system.api.app import create_app
from robot_skill_system.api.contracts import SkillEditorPreviewRequest
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


@pytest.fixture
def service(tmp_path: Path) -> MVPApplication:
    application = MVPApplication(_settings(tmp_path))
    try:
        yield application
    finally:
        application.close()


def _gripper_skill_payload() -> dict[str, object]:
    return {
        "skill_id": "block_gripper_demo",
        "name": "순차 그리퍼 데모",
        "description": "승인된 그리퍼 primitive만 순서대로 실행한다.",
        "skill_type": "composite",
        "blocks": [
            {"operation": "gripper.open", "arguments": {"tool": "$tool"}},
            {
                "operation": "gripper.move_width",
                "arguments": {"tool": "$tool", "width_m": 0.02},
            },
        ],
        "bindings": {},
    }


def test_editor_catalog_exposes_canonical_typed_primitives_and_profile_ids(
    service: MVPApplication,
) -> None:
    paths = create_app(service).openapi()["paths"]
    assert "/skills/editor/catalog" in paths
    assert "/skills/editor/preview" in paths
    assert "/skills/editor/candidates" in paths
    assert "/skills/{skill_id}/versions/{version}/parameter-candidates" in paths
    assert "delete" in paths["/skills/{skill_id}"]
    catalog = service.get_skill_editor_catalog()
    operations = {item["operation_name"]: item for item in catalog["primitives"]}
    assert "motion.move_l" in operations
    assert "gripper.open" in operations
    assert "motion.move_sx" not in operations
    assert operations["motion.move_l"]["typed_parameter_schema"]["properties"][
        "target"
    ]
    assert operations["motion.move_l"]["approved_profile_ids"][
        "motion_profile_ids"
    ] == ["linear_slow", "linear_normal", "linear_expert"]
    assert catalog["approved_profiles"]["motion_profile_ids"]
    assert all(
        isinstance(profile_id, str)
        for profile_id in catalog["approved_profiles"]["force_profile_ids"]
    )
    assert catalog["constraints"] == {
        "sequential_only": True,
        "loops_allowed": False,
        "branches_allowed": False,
        "free_form_code_allowed": False,
        "free_form_json_editor_allowed": False,
        "inline_velocity_acceleration_force_allowed": False,
        "candidate_only": True,
        "hardware_compatible": False,
    }


def test_editor_preview_is_non_persisting_and_rejects_aliases_and_bad_profiles(
    service: MVPApplication,
) -> None:
    payload = _gripper_skill_payload()
    result = service.preview_skill_editor_blocks(payload)
    assert result["valid"] is True
    assert result["persisted"] is False
    assert [item["node_id"] for item in result["normalized_blocks"]] == [
        "block_001",
        "block_002",
    ]
    assert result["skill_graph"]["nodes"][0]["on_success"] == "block_002"
    assert result["skill_graph"]["terminal_nodes"] == ["block_002"]
    assert service.list_skills()["skills"] == []

    alias_payload = {
        **payload,
        "blocks": [
            {
                "operation": "gripper.set_width",
                "arguments": {"width_m": 0.02},
            }
        ],
    }
    alias_preview = service.preview_skill_editor_blocks(alias_payload)
    assert alias_preview["valid"] is False
    assert "compatibility alias" in alias_preview["errors"][0]

    pose = {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": 0.0, "y": 0.0, "z": 0.02},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    bad_profile_payload = {
        **payload,
        "blocks": [
            {
                "operation": "motion.move_l",
                "arguments": {"target": pose, "motion_profile_id": "joint_safe"},
            }
        ],
    }
    bad_profile_preview = service.preview_skill_editor_blocks(bad_profile_payload)
    assert bad_profile_preview["valid"] is False
    assert "does not support" in bad_profile_preview["errors"][0]

    loop_payload = {**payload, "loop": {"count": 2}}
    with pytest.raises(ValidationError):
        SkillEditorPreviewRequest.model_validate(loop_payload)


def test_editor_preview_preserves_explicit_typed_custom_binding(
    service: MVPApplication,
) -> None:
    payload = {
        **_gripper_skill_payload(),
        "skill_id": "custom_binding_demo",
        "blocks": [
            {"operation": "gripper.open", "arguments": {"tool": "$vacuum_tool"}}
        ],
        "bindings": {
            "$vacuum_tool": {
                "variable": "$vacuum_tool",
                "entity_kind": "tool",
                "class_name": "vacuum_gripper",
                "role": "active_end_effector",
                "minimum_confidence": 0.85,
                "minimum_visible_fraction": 0.6,
                "must_be_attached": True,
            }
        },
    }

    result = service.preview_skill_editor_blocks(payload)

    assert result["valid"] is True
    binding = result["skill_graph"]["bindings"]["$vacuum_tool"]
    assert binding["entity_kind"] == "tool"
    assert binding["class_name"] == "vacuum_gripper"
    assert binding["role"] == "active_end_effector"
    assert binding["minimum_confidence"] == 0.85
    assert binding["must_be_attached"] is True


def test_block_and_parameter_candidates_are_immutable_and_mock_validated(
    service: MVPApplication,
) -> None:
    payload = {**_gripper_skill_payload(), "acknowledge_mock_only": True}
    created = service.create_skill_editor_candidate(payload)
    assert created["candidate"]["version"] == "0.1.0-candidate"
    assert created["candidate"]["hardware_compatible"] is False
    assert created["mock_validation_passed"] is True
    assert created["validation"]["runtime"]["gripper_commands"] == [
        "connect",
        "open",
        "move_width",
    ]
    assert created["validation"]["runtime"]["final_gripper_state"]["width_m"] == 0.02

    parent = service.get_skill("block_gripper_demo", "0.1.0-candidate")
    parent_checksum = parent["graph_checksum_sha256"]
    with pytest.raises(ValueError, match="Mock-only acknowledgement"):
        service.create_skill_parameter_candidate(
            "block_gripper_demo",
            "0.1.0-candidate",
            {
                "expected_parent_checksum_sha256": parent_checksum,
                "edits": [
                    {
                        "node_id": "block_002",
                        "arguments": {"tool": "$tool", "width_m": 0.05},
                    }
                ],
                "acknowledge_mock_only": False,
            },
        )
    edited = service.create_skill_parameter_candidate(
        "block_gripper_demo",
        "0.1.0-candidate",
        {
            "expected_parent_checksum_sha256": parent_checksum,
            "edits": [
                {
                    "node_id": "block_002",
                    "arguments": {"tool": "$tool", "width_m": 0.05},
                }
            ],
            "acknowledge_mock_only": True,
        },
    )
    assert edited["parent_unchanged"] is True
    assert edited["candidate"]["version"] == "0.2.0-candidate"
    assert edited["candidate"]["skill_graph"]["parent_version"] == "0.1.0-candidate"
    assert edited["mock_validation_passed"] is True
    assert edited["validation"]["runtime"]["final_gripper_state"]["width_m"] == 0.05

    unchanged = service.get_skill("block_gripper_demo", "0.1.0-candidate")
    assert unchanged["graph_checksum_sha256"] == parent_checksum
    assert unchanged["skill_graph"]["nodes"][1]["arguments"]["width_m"] == 0.02

    with pytest.raises(ValueError, match="checksum changed"):
        service.create_skill_parameter_candidate(
            "block_gripper_demo",
            "0.1.0-candidate",
            {
                "expected_parent_checksum_sha256": "0" * 64,
                "edits": [
                    {
                        "node_id": "block_002",
                        "arguments": {"tool": "$tool", "width_m": 0.05},
                    }
                ],
                "acknowledge_mock_only": True,
            },
        )


def test_parameter_child_does_not_replace_active_parent(
    service: MVPApplication,
) -> None:
    payload = {
        **_gripper_skill_payload(),
        "skill_id": "active_block_gripper",
        "name": "활성 부모 불변 검증",
        "acknowledge_mock_only": True,
    }
    created = service.create_skill_editor_candidate(payload)
    activated = service.activate_skill(
        "active_block_gripper",
        {"version": created["candidate"]["version"]},
    )
    assert activated["version"] == "0.1.0"
    assert activated["status"] == "active"
    active_parent = service.get_skill("active_block_gripper")
    active_checksum = active_parent["graph_checksum_sha256"]

    child = service.create_skill_parameter_candidate(
        "active_block_gripper",
        "0.1.0",
        {
            "expected_parent_checksum_sha256": active_checksum,
            "edits": [
                {
                    "node_id": "block_002",
                    "arguments": {"tool": "$tool", "width_m": 0.04},
                }
            ],
            "acknowledge_mock_only": True,
        },
    )

    assert child["candidate"]["version"] == "0.2.0-candidate"
    assert child["candidate"]["status"] == "validated"
    unchanged_active = service.get_skill("active_block_gripper")
    assert unchanged_active["version"] == "0.1.0"
    assert unchanged_active["status"] == "active"
    assert unchanged_active["graph_checksum_sha256"] == active_checksum
    assert unchanged_active["skill_graph"]["nodes"][1]["arguments"]["width_m"] == 0.02


def test_inactive_block_skill_can_be_deleted_with_all_candidate_versions(
    service: MVPApplication,
) -> None:
    payload = {**_gripper_skill_payload(), "acknowledge_mock_only": True}
    created = service.create_skill_editor_candidate(payload)
    parent = created["candidate"]
    child = service.create_skill_parameter_candidate(
        "block_gripper_demo",
        parent["version"],
        {
            "expected_parent_checksum_sha256": parent["graph_checksum_sha256"],
            "edits": [
                {
                    "node_id": "block_002",
                    "arguments": {"tool": "$tool", "width_m": 0.04},
                }
            ],
            "acknowledge_mock_only": True,
        },
    )
    assert child["candidate"]["version"] == "0.2.0-candidate"
    artifact_directory = service.store.path_for("skills/block_gripper_demo")
    assert artifact_directory.is_dir()

    deleted = service.delete_skill("block_gripper_demo")

    assert deleted["deleted"] is True
    assert deleted["skill_id"] == "block_gripper_demo"
    assert deleted["deleted_versions"] == 2
    assert deleted["deleted_artifacts"] >= 8
    assert not artifact_directory.exists()
    assert service.list_skills()["skills"] == []
    with pytest.raises(KeyError, match="unknown skill"):
        service.get_skill("block_gripper_demo")

    recreated = service.create_skill_editor_candidate(payload)
    assert recreated["candidate"]["version"] == "0.1.0-candidate"


def test_active_block_skill_cannot_be_deleted(service: MVPApplication) -> None:
    payload = {**_gripper_skill_payload(), "acknowledge_mock_only": True}
    created = service.create_skill_editor_candidate(payload)
    service.activate_skill(
        "block_gripper_demo",
        {"version": created["candidate"]["version"]},
    )

    with pytest.raises(ValueError, match="active skill cannot be deleted"):
        service.delete_skill("block_gripper_demo")
