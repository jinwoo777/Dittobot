"""Lazy adapter for the installed standard-library OnRobot RG2 client."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from robot_skill_system.adapters.errors import NotConfiguredError, require_hardware_authorization
from robot_skill_system.adapters.gripper import GripperState


class OnRobotRG2Adapter:
    adapter_name = "onrobot_rg2_modbus_tcp"

    def __init__(self, *, host: str, port: int = 502, unit_id: int = 65,
                 force_n: float = 20.0, open_width_m: float = 0.110,
                 closed_width_m: float = 0.0, timeout_s: float = 1.0,
                 execution_mode: str = "hardware", hardware_enabled: bool = True,
                 client_factory: Callable[..., Any] | None = None) -> None:
        require_hardware_authorization(execution_mode=execution_mode, enabled=hardware_enabled)
        self.host, self.port, self.unit_id = host, port, unit_id
        self.force_n, self.open_width_m, self.closed_width_m = force_n, open_width_m, closed_width_m
        self.timeout_s = timeout_s
        self._client_factory = client_factory
        self._client: Any | None = None

    def connect(self) -> None:
        factory = self._client_factory
        if factory is None:
            try:
                factory = importlib.import_module("rokey.onrobot_rg2").RG2Client
            except (ImportError, AttributeError) as exc:
                raise NotConfiguredError(
                    "RG2 client import failed; source the ws_dsr install that provides "
                    "rokey.onrobot_rg2 before starting the API"
                ) from exc
        client = factory(self.host, port=self.port, unit_id=self.unit_id, timeout=self.timeout_s)
        client.connect()
        client.get_status()
        self._client = client

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None

    def _required(self) -> Any:
        if self._client is None:
            raise NotConfiguredError("RG2 adapter is not connected")
        return self._client

    def get_state(self) -> GripperState:
        client = self._required()
        status = client.get_status()
        width_m = float(client.get_width_mm()) / 1000.0
        return GripperState(True, width_m, bool(status.grip_detected),
                            "rg2_safety_fault" if status.has_safety_fault else None)

    def move_width(self, width_m: float) -> None:
        self._required().move_to_width(width_m * 1000.0, self.force_n)

    def open(self) -> None:
        self.move_width(self.open_width_m)

    def close(self) -> None:
        self.move_width(self.closed_width_m)

    def stop(self) -> None:
        self._required().stop()


__all__ = ["OnRobotRG2Adapter"]
