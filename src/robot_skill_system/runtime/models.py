"""Small runtime-only value objects; domain schemas remain in scene/skills modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    MOCK = "mock"
    DRY_RUN = "dry_run"
    SIMULATION = "simulation"
    HARDWARE = "hardware"


class RuntimeState(str, Enum):
    IDLE = "idle"
    RESOLVING = "resolving"
    CAPTURING = "capturing"
    BINDING = "binding"
    PREFLIGHT = "preflight"
    READY = "ready"
    EXECUTING = "executing"
    STOPPED = "stopped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EntityKind(str, Enum):
    OBJECT = "object"
    TOOL = "tool"
    SURFACE = "surface"
    WORKSPACE = "workspace"


@dataclass(frozen=True, slots=True)
class EntityRequirement:
    placeholder: str
    entity_kind: EntityKind
    instance_id: str | None = None
    class_name: str | None = None
    role: str | None = None
    minimum_confidence: float = 0.7
    minimum_visible_fraction: float = 0.5
    must_be_attached: bool | None = None
    compatible_skill: str | None = None


@dataclass(frozen=True, slots=True)
class EntityBinding:
    placeholder: str
    entity_id: str
    entity_kind: EntityKind
    confidence: float
    entity: Any


@dataclass(frozen=True, slots=True)
class BoundTargetPose:
    frame_id: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    anchor_entity_id: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    passed: bool
    source: str
    detail: str = ""
    is_mock: bool = False


@dataclass(frozen=True, slots=True)
class PreflightReport:
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "source": check.source,
                    "detail": check.detail,
                    "is_mock": check.is_mock,
                }
                for check in self.checks
            ],
        }


@dataclass(frozen=True, slots=True)
class ObstacleObservation:
    obstacle_id: str
    timestamp_ns: int
    distance_m: float
    dynamic: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    timestamp_ns: int
    event_type: str
    severity: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeContext:
    scene: Any
    bindings: Mapping[str, EntityBinding]
    motion_profiles: Mapping[str, Any]
    force_profiles: Mapping[str, Any]
    execution_mode: ExecutionMode
    verification_profiles: Mapping[str, Any] = field(default_factory=dict)
    attachment_states: dict[str, str] = field(default_factory=dict)
    skill: Any = None
    preflight_report: PreflightReport | None = None
    safety_policy: Any = None
    command_text: str | None = None
    execution_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    success: bool
    status: RuntimeState
    execution_run_id: str | None
    events: tuple[RuntimeEvent, ...]
    error_code: str | None = None
    error_message: str | None = None
