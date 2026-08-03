"""Skill version state transitions and semantic-version helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from robot_skill_system.skills.models import SkillLifecycleStatus

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")

_ALLOWED_TRANSITIONS: dict[SkillLifecycleStatus, frozenset[SkillLifecycleStatus]] = {
    SkillLifecycleStatus.DRAFT: frozenset(
        {SkillLifecycleStatus.CANDIDATE, SkillLifecycleStatus.REJECTED}
    ),
    SkillLifecycleStatus.CANDIDATE: frozenset(
        {SkillLifecycleStatus.VALIDATED, SkillLifecycleStatus.REJECTED}
    ),
    SkillLifecycleStatus.VALIDATED: frozenset(
        {SkillLifecycleStatus.ACTIVE, SkillLifecycleStatus.REJECTED}
    ),
    SkillLifecycleStatus.ACTIVE: frozenset({SkillLifecycleStatus.RETIRED}),
    SkillLifecycleStatus.RETIRED: frozenset({SkillLifecycleStatus.ACTIVE}),
    SkillLifecycleStatus.REJECTED: frozenset(),
}


@dataclass(frozen=True, order=True, slots=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int
    prerelease: str | None = None

    @classmethod
    def parse(cls, value: str) -> SemanticVersion:
        """Parse the supported Semantic Version subset."""

        match = _SEMVER.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid semantic version: {value!r}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
        )

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base


def next_candidate_version(parent_version: str) -> str:
    """Create a new minor candidate version without overwriting the parent."""

    parent = SemanticVersion.parse(parent_version)
    return str(SemanticVersion(parent.major, parent.minor + 1, 0, "candidate"))


def stable_version(candidate_version: str) -> str:
    """Drop the candidate prerelease label during validated promotion."""

    parsed = SemanticVersion.parse(candidate_version)
    if parsed.prerelease not in {"candidate", None}:
        raise ValueError("only candidate versions may be stabilized")
    return str(SemanticVersion(parsed.major, parsed.minor, parsed.patch))


def ensure_transition(
    current: SkillLifecycleStatus | str, requested: SkillLifecycleStatus | str
) -> None:
    """Reject lifecycle jumps that could bypass candidate validation."""

    source = current if isinstance(current, SkillLifecycleStatus) else SkillLifecycleStatus(current)
    target = (
        requested
        if isinstance(requested, SkillLifecycleStatus)
        else SkillLifecycleStatus(requested)
    )
    if target not in _ALLOWED_TRANSITIONS[source]:
        raise ValueError(f"invalid skill lifecycle transition: {source.value} -> {target.value}")
