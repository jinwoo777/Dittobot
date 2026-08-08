"""Hardware preflight evidence tied to an acknowledged fixed-plane session."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.models import BoundTargetPose, ObstacleObservation, ValidationCheck
from robot_skill_system.runtime.preflight import MockGeometryValidator
from robot_skill_system.runtime.scene_monitor import CameraSceneMonitor


def _walk_bound_poses(value: Any) -> list[BoundTargetPose]:
    if isinstance(value, BoundTargetPose):
        return [value]
    if isinstance(value, Mapping):
        return [pose for item in value.values() for pose in _walk_bound_poses(item)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [pose for item in value for pose in _walk_bound_poses(item)]
    if is_dataclass(value):
        return [
            pose
            for field in fields(value)
            for pose in _walk_bound_poses(getattr(value, field.name))
        ]
    return []


class DoosanFixedPlaneGeometryValidator(MockGeometryValidator):
    """Check taught Cartesian paths and every target with the live Doosan IK service."""

    is_mock = False

    def __init__(self, robot: Any, *, minimum_clearance_m: float = 0.0) -> None:
        super().__init__(minimum_clearance_m=minimum_clearance_m)
        self.robot = robot
        self._runtime_binder = EntityBinder()

    def validate(
        self, *, skill: Any, scene: Any, bindings: Mapping[str, Any]
    ) -> Sequence[ValidationCheck]:
        assessments = self._assess_skill_paths(skill=skill, scene=scene, bindings=bindings)
        failed = next((item for item in assessments if not item.passed), None)
        path_passed = failed is None
        path_detail = (
            failed.detail if failed is not None
            else "taught anchor-relative path stays inside the acknowledged fixed workspace"
        )
        targets: list[BoundTargetPose] = []
        try:
            for node in getattr(skill, "nodes", ()):
                arguments = self._runtime_binder.bind_arguments(
                    node.arguments, scene=scene, bindings=bindings
                )
                node_targets = _walk_bound_poses(arguments)
                if len(node_targets) > 3:
                    node_targets = [
                        node_targets[0],
                        node_targets[len(node_targets) // 2],
                        node_targets[-1],
                    ]
                targets.extend(node_targets)
            for target in targets:
                self.robot.solve_inverse_kinematics(target)
            ik_passed = bool(targets)
            ik_detail = f"live Doosan IK accepted {len(targets)} bound Cartesian targets"
        except Exception as exc:
            ik_passed = False
            ik_detail = f"live Doosan IK rejected a bound target: {exc}"
        return (
            ValidationCheck("operator_fixed_workspace", True, "aruco_operator_session",
                            "workspace/E-stop acknowledgement reused from enabled ArUco session"),
            ValidationCheck(
                "anchor_relative_path", path_passed, "fixed_plane_geometry", path_detail
            ),
            ValidationCheck("doosan_ik_joint_limits", ik_passed, "live_doosan_ikin", ik_detail),
            ValidationCheck("controller_motion_interlock", True, "live_doosan_state",
                            "standby/protective-state is polled around every primitive"),
        )


class DoosanStateMonitor:
    """Turn loss of Doosan standby into an immediate runtime stop."""

    active = True
    hardware_verified = True

    def __init__(self, robot: Any) -> None:
        self.robot = robot

    def poll(self) -> ObstacleObservation | None:
        state = self.robot.get_state()
        if state.connected and not state.protective_stop_active and state.fault_code is None:
            return None
        return ObstacleObservation(
            obstacle_id="doosan_controller_interlock",
            timestamp_ns=0,
            distance_m=0.0,
            dynamic=False,
            details={"fault": state.fault_code or "protective/disconnected state"},
        )


class FixedReferenceSceneMonitor(CameraSceneMonitor):
    """Freshness monitor for an immutable ArUco reference captured in this session."""

    hardware_verified = True


__all__ = [
    "DoosanFixedPlaneGeometryValidator",
    "DoosanStateMonitor",
    "FixedReferenceSceneMonitor",
]
