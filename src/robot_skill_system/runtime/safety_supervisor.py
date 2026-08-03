"""Global policy supervisor state shared by every skill."""

from __future__ import annotations

from dataclasses import dataclass

from robot_skill_system.runtime.errors import RuntimeSafetyError


@dataclass(slots=True)
class GlobalSafetySupervisor:
    active: bool = True
    abort_requested: bool = False
    abort_reason: str | None = None

    def assert_ready(self) -> None:
        if not self.active:
            raise RuntimeSafetyError("global safety supervisor is inactive")
        if self.abort_requested:
            raise RuntimeSafetyError(f"global abort is active: {self.abort_reason or 'unknown'}")

    def request_abort(self, reason: str) -> None:
        self.abort_requested = True
        self.abort_reason = reason

    def clear_abort(self) -> None:
        self.abort_requested = False
        self.abort_reason = None
