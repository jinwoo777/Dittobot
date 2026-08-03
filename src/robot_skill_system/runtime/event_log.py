"""Structured runtime event sinks."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from robot_skill_system.runtime.models import RuntimeEvent
from robot_skill_system.storage.database import StorageRepository


class EventSink(Protocol):
    @property
    def events(self) -> tuple[RuntimeEvent, ...]: ...

    def record(
        self, event_type: str, details: Mapping[str, Any] | None = None, *, severity: str = "info"
    ) -> RuntimeEvent: ...


class InMemoryEventSink:
    def __init__(self, *, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self._clock_ns = clock_ns
        self._events: list[RuntimeEvent] = []

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    def record(
        self, event_type: str, details: Mapping[str, Any] | None = None, *, severity: str = "info"
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            sequence=len(self._events) + 1,
            timestamp_ns=self._clock_ns(),
            event_type=event_type,
            severity=severity,
            details=dict(details or {}),
        )
        self._events.append(event)
        return event


class StorageEventSink(InMemoryEventSink):
    def __init__(
        self,
        repository: StorageRepository,
        execution_run_id: str,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        super().__init__(clock_ns=clock_ns)
        self.repository = repository
        self.execution_run_id = execution_run_id

    def record(
        self, event_type: str, details: Mapping[str, Any] | None = None, *, severity: str = "info"
    ) -> RuntimeEvent:
        event = super().record(event_type, details, severity=severity)
        self.repository.append_execution_event(
            self.execution_run_id,
            timestamp_ns=event.timestamp_ns,
            event_type=event.event_type,
            details=event.details,
            severity=event.severity,
        )
        return event
