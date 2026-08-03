"""Deterministic, offline robot and gripper adapters."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from robot_skill_system.adapters.gripper import GripperState
from robot_skill_system.adapters.robot import (
    RobotCommand,
    RobotEvent,
    RobotOperationalState,
    RobotState,
)


class MockRobotAdapter:
    """Stateful fake that records every command and safety-relevant event."""

    adapter_name = "mock"

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        initial_pose: Any = None,
        joint_count: int = 6,
    ) -> None:
        self._clock_ns = clock_ns
        self._connected = False
        self._operational_state = RobotOperationalState.DISCONNECTED
        self._emergency_stop_active = False
        self._protective_stop_active = False
        self._fault_code: str | None = None
        self._current_pose = initial_pose
        self._joint_positions_rad = tuple(0.0 for _ in range(joint_count))
        self._tool_force_n: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._compliance_active = False
        self._force_active = False
        self.commands: list[RobotCommand] = []
        self.events: list[RobotEvent] = []
        self._fail_next: dict[str, BaseException] = {}

    @property
    def compliance_active(self) -> bool:
        return self._compliance_active

    @property
    def force_active(self) -> bool:
        return self._force_active

    def connect(self) -> None:
        self._connected = True
        self._operational_state = RobotOperationalState.IDLE
        self._record_command("connect")
        self._record_event("connected")

    def disconnect(self) -> None:
        self._record_command("disconnect")
        self._connected = False
        self._operational_state = RobotOperationalState.DISCONNECTED
        self._record_event("disconnected")

    def get_state(self) -> RobotState:
        return RobotState(
            operational_state=self._operational_state,
            connected=self._connected,
            emergency_stop_active=self._emergency_stop_active,
            protective_stop_active=self._protective_stop_active,
            fault_code=self._fault_code,
        )

    def get_current_pose(self) -> Any:
        return self._current_pose

    def get_joint_positions(self) -> tuple[float, ...]:
        return self._joint_positions_rad

    def move_j(
        self,
        target: Any,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        self._prepare_motion("move_j")
        normalized_target: Any
        if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
            normalized_target = tuple(float(value) for value in target)
        else:
            normalized_target = target
        self._record_command(
            "move_j",
            target=normalized_target,
            velocity_rad_s=velocity_rad_s,
            acceleration_rad_s2=acceleration_rad_s2,
            blend_radius_m=blend_radius_m,
        )
        if isinstance(normalized_target, tuple):
            self._joint_positions_rad = normalized_target
        else:
            self._current_pose = normalized_target
        self._finish_motion("move_j")

    def move_l(
        self,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        self._prepare_motion("move_l")
        self._record_command(
            "move_l",
            target_pose=target_pose,
            velocity_m_s=velocity_m_s,
            acceleration_m_s2=acceleration_m_s2,
            blend_radius_m=blend_radius_m,
        )
        self._current_pose = target_pose
        self._finish_motion("move_l")

    def move_c(
        self,
        via_pose: Any,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        self._prepare_motion("move_c")
        self._record_command(
            "move_c",
            via_pose=via_pose,
            target_pose=target_pose,
            velocity_m_s=velocity_m_s,
            acceleration_m_s2=acceleration_m_s2,
            blend_radius_m=blend_radius_m,
        )
        self._current_pose = target_pose
        self._finish_motion("move_c")

    def move_periodic(
        self,
        center_pose: Any,
        amplitude_m: Any,
        *,
        repetitions: int,
        velocity_m_s: float,
        acceleration_m_s2: float,
    ) -> None:
        self._prepare_motion("move_periodic")
        self._record_command(
            "move_periodic",
            center_pose=center_pose,
            amplitude_m=amplitude_m,
            repetitions=repetitions,
            velocity_m_s=velocity_m_s,
            acceleration_m_s2=acceleration_m_s2,
        )
        self._finish_motion("move_periodic")

    def search_surface(
        self,
        *,
        surface_id: str,
        contact_search_speed_m_s: float,
        maximum_search_distance_m: float,
    ) -> bool:
        self._require_ready("search_surface")
        self._raise_injected("search_surface")
        self._record_command(
            "search_surface",
            surface_id=surface_id,
            contact_search_speed_m_s=contact_search_speed_m_s,
            maximum_search_distance_m=maximum_search_distance_m,
        )
        return True

    def wait(self, *, duration_s: float) -> None:
        self._require_ready("wait")
        self._raise_injected("wait")
        self._record_command("wait", duration_s=duration_s)
        # Preserve the timing semantics of the real adapter so concurrency,
        # timeout, and operator-abort paths can be exercised offline.
        time.sleep(duration_s)

    def stop(self, *, reason: str) -> None:
        self._record_command("stop", reason=reason)
        self._operational_state = RobotOperationalState.STOPPED
        self._record_event("stopped", reason=reason)

    def start_compliance(self, *, stiffness_n_m: Sequence[float]) -> None:
        self._require_ready("start_compliance")
        self._raise_injected("start_compliance")
        self._record_command(
            "start_compliance", stiffness_n_m=tuple(float(value) for value in stiffness_n_m)
        )
        self._compliance_active = True

    def set_desired_force(
        self, *, force_vector_n: Sequence[float], force_control_axes: Sequence[bool]
    ) -> None:
        self._require_ready("set_desired_force")
        if not self._compliance_active:
            raise RuntimeError("compliance must be active before desired force")
        self._raise_injected("set_desired_force")
        vector = tuple(float(value) for value in force_vector_n)
        self._record_command(
            "set_desired_force",
            force_vector_n=vector,
            force_control_axes=tuple(bool(value) for value in force_control_axes),
        )
        self._tool_force_n = vector + tuple(0.0 for _ in range(max(0, 6 - len(vector))))
        self._tool_force_n = self._tool_force_n[:6]
        self._force_active = True

    def release_force(self, *, release_time_s: float) -> None:
        self._record_command("release_force", release_time_s=release_time_s)
        self._force_active = False
        self._tool_force_n = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def release_compliance(self) -> None:
        self._record_command("release_compliance")
        self._compliance_active = False

    def read_tool_force(self) -> tuple[float, ...]:
        return self._tool_force_n

    def is_emergency_stop_active(self) -> bool:
        return self._emergency_stop_active

    def safe_retract(self, *, direction_xyz: Sequence[float], distance_m: float) -> None:
        self._require_connected()
        self._record_command(
            "safe_retract",
            direction_xyz=tuple(float(value) for value in direction_xyz),
            distance_m=distance_m,
        )
        self._operational_state = RobotOperationalState.IDLE

    def inject_failure(self, operation: str, error: BaseException) -> None:
        self._fail_next[operation] = error

    def set_emergency_stop(self, active: bool) -> None:
        self._emergency_stop_active = active
        if active:
            self._operational_state = RobotOperationalState.STOPPED
        self._record_event("emergency_stop_changed", active=active)

    def set_tool_force(self, force_n: Sequence[float]) -> None:
        vector = tuple(float(value) for value in force_n)
        self._tool_force_n = vector + tuple(0.0 for _ in range(max(0, 6 - len(vector))))
        self._tool_force_n = self._tool_force_n[:6]

    def _prepare_motion(self, operation: str) -> None:
        self._require_ready(operation)
        self._raise_injected(operation)
        self._operational_state = RobotOperationalState.MOVING

    def _finish_motion(self, operation: str) -> None:
        self._operational_state = RobotOperationalState.IDLE
        self._record_event("motion_completed", operation=operation)

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("mock robot is not connected")

    def _require_ready(self, operation: str) -> None:
        self._require_connected()
        if self._emergency_stop_active or self._protective_stop_active:
            raise RuntimeError(f"cannot {operation}: stop is active")
        if self._fault_code is not None:
            raise RuntimeError(f"cannot {operation}: robot fault {self._fault_code}")

    def _raise_injected(self, operation: str) -> None:
        error = self._fail_next.pop(operation, None)
        if error is not None:
            raise error

    def _record_command(self, operation: str, **arguments: Any) -> None:
        self.commands.append(
            RobotCommand(len(self.commands) + 1, self._clock_ns(), operation, arguments)
        )

    def _record_event(self, event_type: str, **details: Any) -> None:
        self.events.append(RobotEvent(len(self.events) + 1, self._clock_ns(), event_type, details))


class MockGripperAdapter:
    adapter_name = "mock_rg2"

    def __init__(self, *, maximum_width_m: float = 0.11) -> None:
        self.maximum_width_m = maximum_width_m
        self._connected = False
        self._width_m = maximum_width_m
        self.commands: list[tuple[str, Mapping[str, Any]]] = []

    def connect(self) -> None:
        self._connected = True
        self.commands.append(("connect", {}))

    def disconnect(self) -> None:
        self.commands.append(("disconnect", {}))
        self._connected = False

    def get_state(self) -> GripperState:
        return GripperState(self._connected, self._width_m, self._width_m < 0.01)

    def open(self) -> None:
        self._require_connected()
        self._width_m = self.maximum_width_m
        self.commands.append(("open", {}))

    def close(self) -> None:
        self._require_connected()
        self._width_m = 0.0
        self.commands.append(("close", {}))

    def move_width(self, width_m: float) -> None:
        self._require_connected()
        if not 0.0 <= width_m <= self.maximum_width_m:
            raise ValueError("gripper width is outside configured range")
        self._width_m = width_m
        self.commands.append(("move_width", {"width_m": width_m}))

    def stop(self) -> None:
        self.commands.append(("stop", {}))

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("mock gripper is not connected")
