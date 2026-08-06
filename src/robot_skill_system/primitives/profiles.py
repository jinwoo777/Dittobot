"""Load approved local motion, force, and safety profile JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from robot_skill_system.exceptions import ConfigurationError, ProfileNotFoundError
from robot_skill_system.primitives.models import (
    ForceProfile,
    GraspVerificationProfile,
    MotionProfile,
    SafetyPolicy,
)

ProfileT = TypeVar("ProfileT", bound=BaseModel)


def _load_profile_list(path: Path, key: str, model: type[ProfileT]) -> dict[str, ProfileT]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read approved profile file {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ConfigurationError(f"profile file {path} must contain a {key!r} array")
    profiles: dict[str, ProfileT] = {}
    try:
        for item in payload[key]:
            profile = model.model_validate(item)
            profile_data = profile.model_dump(mode="python")
            profile_id = profile_data.get("profile_id") or profile_data.get("policy_id")
            if not isinstance(profile_id, str):
                raise ConfigurationError(f"profile in {path} has no string identifier")
            if profile_id in profiles:
                raise ConfigurationError(f"duplicate profile id {profile_id!r} in {path}")
            profiles[profile_id] = profile
    except (ValidationError, TypeError, AttributeError) as error:
        raise ConfigurationError(f"invalid approved profile in {path}: {error}") from error
    return profiles


def load_motion_profiles(path: Path) -> dict[str, MotionProfile]:
    """Load and validate a motion-profile JSON file."""

    return _load_profile_list(path, "motion_profiles", MotionProfile)


def load_force_profiles(path: Path) -> dict[str, ForceProfile]:
    """Load and validate a force-profile JSON file."""

    return _load_profile_list(path, "force_profiles", ForceProfile)


def load_grasp_verification_profiles(path: Path) -> dict[str, GraspVerificationProfile]:
    """Load locally approved grasp/release interpretation profiles."""

    return _load_profile_list(
        path, "grasp_verification_profiles", GraspVerificationProfile
    )


def load_safety_policies(path: Path) -> dict[str, SafetyPolicy]:
    """Load and validate a safety-policy JSON file."""

    return _load_profile_list(path, "safety_policies", SafetyPolicy)


def require_profile(profiles: dict[str, ProfileT], profile_id: str) -> ProfileT:
    """Return a loaded approved profile or raise a domain-specific error."""

    try:
        return profiles[profile_id]
    except KeyError as error:
        raise ProfileNotFoundError(f"approved profile {profile_id!r} was not loaded") from error
