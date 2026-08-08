"""Audited runtime bridge for the connected Doosan M0609 session."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.adapters.errors import NotConfiguredError, require_hardware_authorization
from robot_skill_system.adapters.robot import (
    RobotCommand,
    RobotOperationalState,
    RobotState,
)
from robot_skill_system.perception._geometry import rotation_matrix_to_quaternion_xyzw
from robot_skill_system.runtime.models import BoundTargetPose

JointVector = tuple[float, float, float, float, float, float]


def _get(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name)


def _rotation_from_xyzw(value: Any) -> NDArray[np.float64]:
    x, y, z, w = (float(component) for component in value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _zyz_deg(rotation: NDArray[np.float64]) -> tuple[float, float, float]:
    second = math.acos(max(-1.0, min(1.0, float(rotation[2, 2]))))
    if abs(math.sin(second)) > 1e-8:
        first = math.atan2(float(rotation[1, 2]), float(rotation[0, 2]))
        third = math.atan2(float(rotation[2, 1]), -float(rotation[2, 0]))
    else:
        first = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        third = 0.0
    return cast(
        tuple[float, float, float],
        tuple(math.degrees(value) for value in (first, second, third)),
    )


def runtime_pose_to_doosan(
    target: Any,
    *,
    fixed_workspace_orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Convert a bound metre/xyzw runtime pose to Doosan mm/ZYZ degrees."""

    position = _get(target, "position_m")
    orientation = (
        fixed_workspace_orientation_xyzw
        if fixed_workspace_orientation_xyzw is not None
        else _get(target, "orientation_xyzw")
    )
    xyz = (
        tuple(float(item) for item in position)
        if isinstance(position, Sequence)
        else tuple(float(_get(position, name)) for name in ("x", "y", "z"))
    )
    xyzw = (
        tuple(float(item) for item in orientation)
        if isinstance(orientation, Sequence)
        else tuple(float(_get(orientation, name)) for name in ("x", "y", "z", "w"))
    )
    first, second, third = _zyz_deg(_rotation_from_xyzw(xyzw))
    return (xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0, first, second, third)


class DoosanM0609Adapter:
    """Runtime adapter that reuses the safety-acknowledged ArUco robot session."""

    adapter_name = "doosan_m0609_runtime"

    def __init__(
        self,
        robot: Any | None = None,
        *,
        execution_mode: str = "mock",
        hardware_enabled: bool = False,
        fixed_workspace_orientation_xyzw: tuple[float, float, float, float] | None = None,
        initial_pose: BoundTargetPose | None = None,
    ) -> None:
        self.robot = robot
        self.execution_mode = execution_mode
        self.hardware_enabled = hardware_enabled
        self.fixed_workspace_orientation_xyzw = fixed_workspace_orientation_xyzw
        # The application obtains this once while it acquires the acknowledged
        # fixed-base session.  Reusing it during a graph run avoids repeatedly
        # calling the vendor ``get_current_posx`` wrapper, which can return an
        # empty response even while the controller remains in standby.
        self._last_pose = initial_pose
        self.commands: list[RobotCommand] = []
        self._sequence = 0

    def _robot(self) -> Any:
        if self.robot is None:
            require_hardware_authorization(
                execution_mode=self.execution_mode, enabled=self.hardware_enabled
            )
            raise NotConfiguredError(
                "Doosan runtime requires an enabled, reference-captured ArUco session"
            )
        return self.robot

    def _record(self, operation: str, **arguments: Any) -> None:
        self._sequence += 1
        self.commands.append(RobotCommand(self._sequence, time.time_ns(), operation, arguments))

    def connect(self) -> None:
        self._robot()._require_standby()
        self._record("connect_reused_aruco_session")

    def disconnect(self) -> None:
        # The ArUco controller owns the ROS session and its Stop button closes it.
        self._record("disconnect_runtime_lease")

    def get_state(self) -> RobotState:
        try:
            self._robot()._require_standby()
        except Exception as exc:
            return RobotState(RobotOperationalState.FAULT, True, False, True, str(exc))
        return RobotState(RobotOperationalState.IDLE, True, False, False, None)

    def get_current_pose(self) -> Any:
        if self._last_pose is not None:
            return self._last_pose
        matrix = np.asarray(self._robot().get_base_to_tcp_matrix(), dtype=np.float64)
        quaternion = rotation_matrix_to_quaternion_xyzw(matrix[:3, :3])
        self._last_pose = BoundTargetPose(
            frame_id="base",
            position_m=cast(
                tuple[float, float, float],
                tuple(float(value) for value in matrix[:3, 3]),
            ),
            orientation_xyzw=quaternion,
            anchor_entity_id="live_active_tcp",
            timestamp_ns=time.time_ns(),
        )
        return self._last_pose

    def get_joint_positions(self) -> tuple[float, ...]:
        return cast(tuple[float, ...], self._robot().get_joint_positions_rad())

    def solve_inverse_kinematics(self, target_pose: Any) -> tuple[float, ...]:
        return cast(
            tuple[float, ...],
            self._robot().solve_inverse_kinematics(
                runtime_pose_to_doosan(
                    target_pose,
                    fixed_workspace_orientation_xyzw=self.fixed_workspace_orientation_xyzw,
                )
            ),
        )

    def move_j(self, target: Any, *, velocity_rad_s: float, acceleration_rad_s2: float,
               blend_radius_m: float = 0.0) -> None:
        del blend_radius_m
        if not isinstance(target, Sequence) or len(target) != 6:
            raise ValueError("Doosan MoveJ runtime target must contain six joint radians")
        values = cast(JointVector, tuple(float(item) for item in target))
        self._robot().move_joints(values, velocity_rad_s=velocity_rad_s,
                                  acceleration_rad_s2=acceleration_rad_s2)
        self._record("move_j", target=values)

    def move_l(self, target_pose: Any, *, velocity_m_s: float,
               acceleration_m_s2: float, blend_radius_m: float = 0.0) -> None:
        del blend_radius_m
        target = runtime_pose_to_doosan(
            target_pose,
            fixed_workspace_orientation_xyzw=self.fixed_workspace_orientation_xyzw,
        )
        self._robot().solve_inverse_kinematics(target)
        self._robot().move_linear(
            target,
            linear_velocity_m_s=velocity_m_s,
            linear_acceleration_m_s2=acceleration_m_s2,
            angular_velocity_rad_s=math.radians(15.0),
            angular_acceleration_rad_s2=math.radians(30.0),
        )
        if isinstance(target_pose, BoundTargetPose):
            orientation = (
                self.fixed_workspace_orientation_xyzw
                if self.fixed_workspace_orientation_xyzw is not None
                else target_pose.orientation_xyzw
            )
            self._last_pose = replace(
                target_pose, orientation_xyzw=orientation, timestamp_ns=time.time_ns()
            )
        self._record("move_l", target_pose=target_pose)

    def move_c(self, via_pose: Any, target_pose: Any, *, velocity_m_s: float,
               acceleration_m_s2: float, blend_radius_m: float = 0.0) -> None:
        # The installed audited boundary exposes MoveL and IK. Preserve both taught
        # circle waypoints as two checked linear segments instead of guessing MoveC fields.
        self.move_l(via_pose, velocity_m_s=velocity_m_s,
                    acceleration_m_s2=acceleration_m_s2, blend_radius_m=blend_radius_m)
        self.move_l(target_pose, velocity_m_s=velocity_m_s,
                    acceleration_m_s2=acceleration_m_s2, blend_radius_m=blend_radius_m)
        self._record("move_c_as_two_move_l")

    def move_periodic(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError("hardware periodic motion is not configured")

    def wait(self, *, duration_s: float) -> None:
        time.sleep(duration_s)
        self._record("wait", duration_s=duration_s)

    def stop(self, *, reason: str) -> None:
        self._robot().stop(reason=reason)
        self._record("stop", reason=reason)

    def is_emergency_stop_active(self) -> bool:
        try:
            self._robot()._require_standby()
        except Exception:
            return True
        return False

    @property
    def compliance_active(self) -> bool:
        return False

    @property
    def force_active(self) -> bool:
        return False

    def start_compliance(self, **kwargs: Any) -> None:
        raise ValueError("hardware force control is not configured for this runtime")

    def set_desired_force(self, **kwargs: Any) -> None:
        raise ValueError("hardware force control is not configured for this runtime")

    def release_force(self, **kwargs: Any) -> None:
        self._record("release_force_noop")

    def release_compliance(self) -> None:
        self._record("release_compliance_noop")

    def read_tool_force(self) -> tuple[float, ...]:
        raise ValueError("hardware tool-force feedback is not configured")


__all__ = ["DoosanM0609Adapter", "runtime_pose_to_doosan"]
