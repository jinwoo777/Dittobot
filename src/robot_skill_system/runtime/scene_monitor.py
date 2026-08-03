"""Continuous camera/scene freshness gate used between primitives."""

from __future__ import annotations

import time
from typing import Any

from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import RuntimeSafetyError


class CameraSceneMonitor:
    hardware_verified = False

    def __init__(
        self,
        *,
        maximum_scene_age_ms: int = 5_000,
        maximum_camera_heartbeat_age_ms: int = 2_000,
        clock_ns: Any = time.time_ns,
        active: bool = True,
    ) -> None:
        self.maximum_scene_age_ms = maximum_scene_age_ms
        self.maximum_camera_heartbeat_age_ms = maximum_camera_heartbeat_age_ms
        self._clock_ns = clock_ns
        self.active = active
        self._last_camera_heartbeat_ns: int | None = None
        self._binder = EntityBinder(clock_ns=clock_ns)

    def update_camera_heartbeat(self, timestamp_ns: int | None = None) -> None:
        self._last_camera_heartbeat_ns = self._clock_ns() if timestamp_ns is None else timestamp_ns

    def assert_fresh(self, scene: Any) -> None:
        if not self.active:
            raise RuntimeSafetyError("camera/scene freshness monitor is inactive")
        self._binder.ensure_scene_fresh(scene, maximum_age_ms=self.maximum_scene_age_ms)
        if self._last_camera_heartbeat_ns is None:
            # A static fixture scene has no continuous camera stream. Its own validity is still
            # enforced above and the absence is explicit rather than silently synthesized.
            return
        age_ns = self._clock_ns() - self._last_camera_heartbeat_ns
        if age_ns < 0 or age_ns > self.maximum_camera_heartbeat_age_ms * 1_000_000:
            raise RuntimeSafetyError("camera heartbeat is stale")


SceneFreshnessMonitor = CameraSceneMonitor

__all__ = ["CameraSceneMonitor", "SceneFreshnessMonitor"]
