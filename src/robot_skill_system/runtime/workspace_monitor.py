"""Global workspace hooks and obstacle recovery policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any, Protocol

from robot_skill_system.runtime.errors import ObstacleDetectedError, RuntimeSafetyError
from robot_skill_system.runtime.geometry import DeterministicSceneGeometry, GeometryAssessment
from robot_skill_system.runtime.models import ObstacleObservation


class ObstacleMonitor(Protocol):
    @property
    def active(self) -> bool: ...

    @property
    def hardware_verified(self) -> bool: ...

    def poll(self) -> ObstacleObservation | None: ...


class MockObstacleMonitor:
    hardware_verified = False

    def __init__(
        self, observations: Iterable[ObstacleObservation] = (), *, active: bool = True
    ) -> None:
        self.active = active
        self._observations = deque(observations)

    def inject(self, observation: ObstacleObservation) -> None:
        self._observations.append(observation)

    def poll(self) -> ObstacleObservation | None:
        if not self.active or not self._observations:
            return None
        return self._observations.popleft()


class GlobalWorkspaceSupervisor:
    """Wrap all primitives with fail-closed workspace monitoring hooks."""

    def __init__(
        self,
        monitor: ObstacleMonitor | None = None,
        *,
        active: bool = True,
        retract_distance_m: float = 0.05,
        geometry_validator: DeterministicSceneGeometry | None = None,
        minimum_clearance_m: float = 0.03,
    ) -> None:
        self.monitor = monitor or MockObstacleMonitor()
        self.active = active
        self.retract_distance_m = retract_distance_m
        self.geometry_validator = geometry_validator or DeterministicSceneGeometry(
            minimum_clearance_m=minimum_clearance_m
        )
        self.current_checkpoint: str | None = None
        self.last_obstacle: ObstacleObservation | None = None

    def before_skill(self) -> None:
        if not self.active or not self.monitor.active:
            raise RuntimeSafetyError("workspace supervisor/monitor is inactive")
        self.last_obstacle = None

    def before_primitive(self, operation: str, checkpoint: str | None = None) -> None:
        if checkpoint is not None:
            self.current_checkpoint = checkpoint
        self._raise_if_obstacle(operation)

    def during_primitive(self, operation: str) -> None:
        self._raise_if_obstacle(operation)

    def after_primitive(self, operation: str) -> None:
        self._raise_if_obstacle(operation)

    def validate_target(self, *, target: Any, scene: Any) -> None:
        """Validate an already-bound TCP target against the current scene."""

        self._raise_if_geometry_failed(
            self.geometry_validator.validate_target(target=target, scene=scene),
            operation="workspace.validate_target",
        )

    def validate_path(self, *, path: Any, scene: Any) -> None:
        """Validate all explicit waypoints and segments against the current scene."""

        self._raise_if_geometry_failed(
            self.geometry_validator.validate_path(path=path, scene=scene),
            operation="workspace.validate_path",
        )

    def on_obstacle(self, robot: Any, force_supervisor: Any) -> None:
        """Stop, then release force/compliance and retract before reporting failure."""

        observation = self.last_obstacle
        obstacle_id = observation.obstacle_id if observation else "unknown"
        robot.stop(reason=f"workspace_obstacle:{obstacle_id}")
        if force_supervisor.force_active:
            retract_direction = force_supervisor.emergency_release()
            if retract_direction is None:
                raise RuntimeSafetyError("force cleanup did not provide a safe retract direction")
            safe_retract = getattr(robot, "safe_retract", None)
            if safe_retract is None:
                raise RuntimeSafetyError(
                    "adapter has no verified safe_retract capability after contact obstacle"
                )
            safe_retract(
                direction_xyz=retract_direction, distance_m=self.retract_distance_m
            )

    def on_scene_stale(self, robot: Any) -> None:
        robot.stop(reason="scene_stale")

    def on_abort(self, robot: Any, reason: str) -> None:
        robot.stop(reason=reason)

    def _raise_if_obstacle(self, operation: str) -> None:
        observation = self.monitor.poll()
        if observation is None:
            return
        self.last_obstacle = observation
        raise ObstacleDetectedError(
            f"obstacle {observation.obstacle_id!r} detected around {operation} "
            f"at {observation.distance_m:.3f} m"
        )

    def _raise_if_geometry_failed(
        self, assessment: GeometryAssessment, *, operation: str
    ) -> None:
        if assessment.passed:
            return
        obstacle_id = assessment.obstacle_id or "unverified_scene_geometry"
        distance_m = assessment.minimum_distance_m
        if distance_m is None or distance_m < 0.0:
            distance_m = 0.0
        self.last_obstacle = ObstacleObservation(
            obstacle_id=obstacle_id,
            timestamp_ns=0,
            distance_m=distance_m,
            dynamic=False,
            details={"operation": operation, "reason": assessment.detail},
        )
        raise ObstacleDetectedError(f"{operation} rejected: {assessment.detail}")


WorkspaceSupervisor = GlobalWorkspaceSupervisor

__all__ = [
    "GlobalWorkspaceSupervisor",
    "MockObstacleMonitor",
    "ObstacleMonitor",
    "WorkspaceSupervisor",
]
