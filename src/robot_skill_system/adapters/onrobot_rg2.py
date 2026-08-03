"""Safe boundary for a future verified OnRobot RG2 binding."""

from __future__ import annotations

from typing import NoReturn

from robot_skill_system.adapters.errors import NotConfiguredError, require_hardware_authorization
from robot_skill_system.adapters.gripper import GripperState


class OnRobotRG2Adapter:
    adapter_name = "onrobot_rg2_unconfigured"

    def __init__(self, *, execution_mode: str = "mock", hardware_enabled: bool = False) -> None:
        self.execution_mode = execution_mode
        self.hardware_enabled = hardware_enabled

    def connect(self) -> NoReturn:
        self._unconfigured()

    def disconnect(self) -> NoReturn:
        self._unconfigured()

    def get_state(self) -> GripperState:
        self._unconfigured()

    def open(self) -> NoReturn:
        self._unconfigured()

    def close(self) -> NoReturn:
        self._unconfigured()

    def move_width(self, width_m: float) -> NoReturn:
        self._unconfigured()

    def stop(self) -> NoReturn:
        self._unconfigured()

    def _unconfigured(self) -> NoReturn:
        require_hardware_authorization(
            execution_mode=self.execution_mode, enabled=self.hardware_enabled
        )
        raise NotConfiguredError(
            "OnRobot RG2 binding is unavailable: verify the installed driver transport and "
            "its callable signatures on the target cell"
        )
