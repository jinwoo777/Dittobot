"""Safe boundary for a future verified Doosan M0609 binding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

from robot_skill_system.adapters.errors import NotConfiguredError, require_hardware_authorization
from robot_skill_system.adapters.robot import RobotState


class DoosanM0609Adapter:
    """Reject use until the target cell's DSR_ROBOT2 signatures are verified.

    No vendor method is guessed here. A deployment integration must replace each method after
    confirming package version, namespace, ROS services/actions, and function signatures.
    """

    adapter_name = "doosan_m0609_unconfigured"

    def __init__(self, *, execution_mode: str = "mock", hardware_enabled: bool = False) -> None:
        self.execution_mode = execution_mode
        self.hardware_enabled = hardware_enabled

    def connect(self) -> NoReturn:
        self._unconfigured()

    def disconnect(self) -> NoReturn:
        self._unconfigured()

    def get_state(self) -> RobotState:
        self._unconfigured()

    def get_current_pose(self) -> Any:
        self._unconfigured()

    def get_joint_positions(self) -> tuple[float, ...]:
        self._unconfigured()

    def move_j(
        self,
        target: Any,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
        blend_radius_m: float = 0.0,
    ) -> NoReturn:
        self._unconfigured()

    def move_l(
        self,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> NoReturn:
        self._unconfigured()

    def move_c(
        self,
        via_pose: Any,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> NoReturn:
        self._unconfigured()

    def move_periodic(
        self,
        center_pose: Any,
        amplitude_m: Any,
        *,
        repetitions: int,
        velocity_m_s: float,
        acceleration_m_s2: float,
    ) -> NoReturn:
        self._unconfigured()

    def stop(self, *, reason: str) -> NoReturn:
        self._unconfigured()

    def start_compliance(self, *, stiffness_n_m: Sequence[float]) -> NoReturn:
        self._unconfigured()

    def set_desired_force(
        self, *, force_vector_n: Sequence[float], force_control_axes: Sequence[bool]
    ) -> NoReturn:
        self._unconfigured()

    def release_force(self, *, release_time_s: float) -> NoReturn:
        self._unconfigured()

    def release_compliance(self) -> NoReturn:
        self._unconfigured()

    def read_tool_force(self) -> tuple[float, ...]:
        self._unconfigured()

    def is_emergency_stop_active(self) -> bool:
        self._unconfigured()

    def _unconfigured(self) -> NoReturn:
        require_hardware_authorization(
            execution_mode=self.execution_mode, enabled=self.hardware_enabled
        )
        raise NotConfiguredError(
            "Doosan M0609 binding is unavailable: verify the installed DSR_ROBOT2 version, "
            "ROS 2 namespace/services/actions, and callable signatures on the target cell"
        )
