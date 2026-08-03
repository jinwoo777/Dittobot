"""Canonical graph-manifest verification before execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from robot_skill_system.runtime.errors import SkillHashMismatchError


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_skill_checksum_sha256(skill: Any) -> str:
    payload = json.dumps(
        _jsonable(skill), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_skill_checksum(skill: Any, expected_checksum_sha256: str) -> str:
    actual = canonical_skill_checksum_sha256(skill)
    if actual != expected_checksum_sha256:
        raise SkillHashMismatchError(
            "skill manifest checksum mismatch: "
            f"expected={expected_checksum_sha256}, actual={actual}"
        )
    return actual


__all__ = ["canonical_skill_checksum_sha256", "verify_skill_checksum"]
