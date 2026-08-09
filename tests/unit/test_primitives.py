"""Unit tests for the closed primitive registry and approved profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_skill_system.exceptions import PrimitiveValidationError, UnknownPrimitiveError
from robot_skill_system.primitives.profiles import (
    load_force_profiles,
    load_grasp_verification_profiles,
    load_motion_profiles,
    load_safety_policies,
)
from robot_skill_system.primitives.registry import PrimitiveRegistry

ROOT = Path(__file__).resolve().parents[2]


def _relative_target() -> dict[str, object]:
    return {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": 0.1, "y": 0.0, "z": 0.05},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def test_catalog_contains_required_metadata_and_aliases() -> None:
    registry = PrimitiveRegistry.default()

    for operation in (
        "motion.move_j",
        "motion.rotate_joint_6_relative",
        "motion.move_l",
        "motion.move_c",
        "motion.move_spline",
        "motion.move_periodic",
        "motion.wait",
        "gripper.open",
        "contact.enable_force",
        "contact.disable_force",
        "workspace.validate_path",
        "recovery.safe_retract",
        "motion.move_sx",
        "gripper.set_width",
        "contact.follow_surface_path",
    ):
        metadata = registry.metadata(operation)
        assert metadata.typed_parameter_schema
        assert metadata.maximum_timeout_s > 0.0
        assert metadata.mock_support
    assert registry.canonical_operation_name("workspace.check_path") == "workspace.validate_path"


def test_unknown_primitive_is_rejected_immediately() -> None:
    with pytest.raises(UnknownPrimitiveError, match="whitelist"):
        PrimitiveRegistry.default().validate_arguments("robot.raw_move", {})


def test_operation_specific_arguments_reject_model_owned_speed() -> None:
    arguments = {
        "target": _relative_target(),
        "motion_profile_id": "linear_normal",
        "velocity_m_s": 99.0,
    }

    with pytest.raises(PrimitiveValidationError, match="velocity_m_s"):
        PrimitiveRegistry.default().validate_arguments("motion.move_l", arguments)


def test_relative_motion_arguments_are_typed_and_normalized() -> None:
    validated = PrimitiveRegistry.default().validate_arguments(
        "motion.move_l",
        {"target": _relative_target(), "profile_id": "linear_normal"},
    )

    assert validated.model_dump(mode="json")["motion_profile_id"] == "linear_normal"


def test_relative_j6_rotation_is_bounded_and_profile_owned() -> None:
    registry = PrimitiveRegistry.default()
    validated = registry.validate_arguments(
        "motion.rotate_joint_6_relative",
        {"delta_rad": 0.5, "motion_profile_id": "joint_safe"},
    )

    assert validated.model_dump(mode="json") == {
        "delta_rad": 0.5,
        "motion_profile_id": "joint_safe",
    }
    with pytest.raises(PrimitiveValidationError):
        registry.validate_arguments(
            "motion.rotate_joint_6_relative",
            {"delta_rad": 2.0, "motion_profile_id": "joint_safe"},
        )


def test_approved_profile_files_validate() -> None:
    motion = load_motion_profiles(ROOT / "configs/motion_profiles/default.json")
    force = load_force_profiles(ROOT / "configs/force_profiles/default.json")
    grasp = load_grasp_verification_profiles(
        ROOT / "configs/grasp_verification_profiles/default.json"
    )
    safety = load_safety_policies(ROOT / "configs/safety_policies/default.json")

    assert set(motion) >= {"joint_safe", "linear_normal", "periodic_safe"}
    assert set(force) >= {"wipe_light", "wipe_standard", "contact_search_soft"}
    assert grasp["grasp_default"].minimum_released_width_m == pytest.approx(0.05)
    assert safety["global_default"].unknown_space_is_occupied
    assert all(len(profile.force_control_axes) in {3, 6} for profile in force.values())
