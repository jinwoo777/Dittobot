"""Typed, vendor-neutral robot adapter protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RobotOperationalState(str, Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MOVING = "moving"
    STOPPED = "stopped"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class RobotState:
    operational_state: RobotOperationalState
    connected: bool
    emergency_stop_active: bool
    protective_stop_active: bool = False
    fault_code: str | None = None


@dataclass(frozen=True, slots=True)
class RobotCommand:
    sequence: int
    timestamp_ns: int
    operation: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RobotEvent:
    sequence: int
    timestamp_ns: int
    event_type: str
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RobotAdapter(Protocol):
    """Required capabilities for runtime execution.

    Vendor-specific pose conversion belongs inside a concrete adapter. Core/runtime poses stay
    in metres and quaternion ``xyzw`` form.
    """

    @property
    def adapter_name(self) -> str: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_state(self) -> RobotState: ...

    def get_current_pose(self) -> Any: ...

    def get_joint_positions(self) -> tuple[float, ...]: ...

    def move_j(
        self,
        target: Any,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None: ...

    def move_l(
        self,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None: ...

    def move_c(
        self,
        via_pose: Any,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None: ...

    def move_periodic(
        self,
        center_pose: Any,
        amplitude_m: Any,
        *,
        repetitions: int,
        velocity_m_s: float,
        acceleration_m_s2: float,
    ) -> None: ...

    def stop(self, *, reason: str) -> None: ...

    def start_compliance(self, *, stiffness_n_m: Sequence[float]) -> None: ...

    def set_desired_force(
        self, *, force_vector_n: Sequence[float], force_control_axes: Sequence[bool]
    ) -> None: ...

    def release_force(self, *, release_time_s: float) -> None: ...

    def release_compliance(self) -> None: ...

    def read_tool_force(self) -> tuple[float, ...]: ...

    def is_emergency_stop_active(self) -> bool: ...
