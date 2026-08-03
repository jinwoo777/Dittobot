"""Vendor-neutral gripper boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GripperState:
    connected: bool
    width_m: float
    is_holding: bool
    fault_code: str | None = None


@runtime_checkable
class GripperAdapter(Protocol):
    @property
    def adapter_name(self) -> str: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_state(self) -> GripperState: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def move_width(self, width_m: float) -> None: ...

    def stop(self) -> None: ...
