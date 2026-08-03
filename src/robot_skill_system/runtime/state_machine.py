"""Strict runtime lifecycle transitions."""

from __future__ import annotations

from robot_skill_system.runtime.models import RuntimeState

_ALLOWED: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.IDLE: frozenset({RuntimeState.RESOLVING, RuntimeState.FAILED}),
    RuntimeState.RESOLVING: frozenset({RuntimeState.CAPTURING, RuntimeState.FAILED}),
    RuntimeState.CAPTURING: frozenset({RuntimeState.BINDING, RuntimeState.FAILED}),
    RuntimeState.BINDING: frozenset({RuntimeState.PREFLIGHT, RuntimeState.FAILED}),
    RuntimeState.PREFLIGHT: frozenset({RuntimeState.READY, RuntimeState.FAILED}),
    RuntimeState.READY: frozenset({RuntimeState.EXECUTING, RuntimeState.FAILED}),
    RuntimeState.EXECUTING: frozenset(
        {RuntimeState.STOPPED, RuntimeState.SUCCEEDED, RuntimeState.FAILED}
    ),
    RuntimeState.STOPPED: frozenset({RuntimeState.FAILED}),
    RuntimeState.SUCCEEDED: frozenset({RuntimeState.IDLE}),
    RuntimeState.FAILED: frozenset({RuntimeState.IDLE}),
}


class RuntimeStateMachine:
    def __init__(self) -> None:
        self.state = RuntimeState.IDLE

    def transition(self, next_state: RuntimeState) -> None:
        if next_state not in _ALLOWED[self.state]:
            raise RuntimeError(f"invalid runtime transition: {self.state} -> {next_state}")
        self.state = next_state

    def reset(self) -> None:
        if self.state not in {RuntimeState.SUCCEEDED, RuntimeState.FAILED}:
            raise RuntimeError(f"cannot reset runtime from {self.state}")
        self.transition(RuntimeState.IDLE)
