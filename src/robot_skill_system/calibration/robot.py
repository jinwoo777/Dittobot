"""Calibration-only robot boundary with fixed, reviewable joint waypoints."""

from __future__ import annotations

import importlib
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.adapters.errors import require_hardware_authorization
from robot_skill_system.exceptions import NotConfiguredError

Matrix44 = NDArray[np.float64]
JointVector = tuple[float, float, float, float, float, float]

DOOSAN_ROBOT_STATE_NAMES = {
    0: "STATE_INITIALIZING",
    1: "STATE_STANDBY",
    2: "STATE_MOVING",
    3: "STATE_SAFE_OFF",
    4: "STATE_TEACHING",
    5: "STATE_SAFE_STOP",
    6: "STATE_EMERGENCY_STOP",
    7: "STATE_HOMMING",
    8: "STATE_RECOVERY",
    9: "STATE_SAFE_STOP2",
    10: "STATE_SAFE_OFF2",
}


def _set_module_attribute(module: ModuleType, name: str, value: Any) -> None:
    """Set a runtime-only vendor module attribute without static stub assumptions."""

    setattr(module, name, value)


def _empty_pose_response(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 0
    )


@dataclass(frozen=True, slots=True)
class CalibrationWaypoint:
    waypoint_id: str
    joint_positions_rad: JointVector

    def as_dict(self) -> dict[str, Any]:
        return {
            "waypoint_id": self.waypoint_id,
            "joint_positions_deg": [
                round(math.degrees(value), 3) for value in self.joint_positions_rad
            ],
        }


@runtime_checkable
class HandEyeCalibrationRobot(Protocol):
    adapter_name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions_rad(self) -> JointVector: ...

    def get_base_to_flange_matrix(self) -> Matrix44: ...

    def get_base_to_tcp_matrix(self) -> Matrix44: ...

    def get_active_tcp_name(self) -> str: ...

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None: ...

    def stop(self, *, reason: str) -> None: ...


def approved_eye_in_hand_waypoints() -> tuple[CalibrationWaypoint, ...]:
    """Return the fixed M0609 micro-motion plan with J1/J2 locked at zero."""

    offsets_deg = (
        (0.0, 0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0, 0.0),
        (5.0, 5.0, 0.0, 0.0),
        (0.0, 5.0, 0.0, 0.0),
        (0.0, 5.0, 5.0, 0.0),
        (0.0, 0.0, 5.0, 0.0),
        (0.0, 0.0, 5.0, 5.0),
        (0.0, 0.0, 0.0, 5.0),
        (0.0, 0.0, 0.0, 0.0),
        (-5.0, 0.0, 0.0, 0.0),
        (-5.0, -5.0, 0.0, 0.0),
        (0.0, -5.0, 0.0, 0.0),
        (0.0, -5.0, -5.0, 0.0),
        (0.0, 0.0, -5.0, 0.0),
        (0.0, 0.0, -5.0, -5.0),
        (0.0, 0.0, 0.0, -5.0),
        (0.0, 0.0, 0.0, 0.0),
        (5.0, -5.0, 5.0, -5.0),
        (0.0, 0.0, 0.0, 0.0),
        (-5.0, 5.0, -5.0, 5.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    result: list[CalibrationWaypoint] = []
    for index, (joint_3, joint_4, joint_5, joint_6) in enumerate(offsets_deg):
        degrees = (0.0, 0.0, 90.0 + joint_3, joint_4, 90.0 + joint_5, joint_6)
        result.append(
            CalibrationWaypoint(
                waypoint_id=f"pose_{index:02d}",
                joint_positions_rad=cast(
                    JointVector,
                    tuple(math.radians(value) for value in degrees),
                ),
            )
        )
    return tuple(result)


class DoosanHandEyeCalibrationRobot:
    """Lazily bind the exact DSR functions required by the approved calibration flow."""

    adapter_name = "doosan_m0609_handeye"
    _ACTIVE_TCP_POSE_ATTEMPTS = 3
    _ACTIVE_TCP_POSE_TIMEOUT_S = 1.0
    _ACTIVE_TCP_POSE_RETRY_DELAY_S = 0.05

    def __init__(
        self,
        *,
        robot_id: str,
        robot_model: str,
        execution_mode: str,
        hardware_enabled: bool,
    ) -> None:
        self.robot_id = robot_id
        self.robot_model = robot_model
        self.execution_mode = execution_mode
        self.hardware_enabled = hardware_enabled
        self._rclpy: ModuleType | None = None
        self._node: Any | None = None
        self._owns_rclpy = False
        self._get_current_posj: Callable[..., Any] | None = None
        self._get_current_tool_flange_posx: Callable[..., Any] | None = None
        self._get_current_posx: Callable[..., Any] | None = None
        self._current_posx_client: Any | None = None
        self._current_posx_type: Any | None = None
        self._current_posx_lock = Lock()
        self._get_tcp: Callable[..., Any] | None = None
        self._get_robot_state: Callable[..., Any] | None = None
        self._movej: Callable[..., Any] | None = None
        self._mwait: Callable[..., Any] | None = None
        self._posj: Callable[..., Any] | None = None
        self._move_stop_client: Any | None = None
        self._move_stop_type: Any | None = None

    def connect(self) -> None:
        require_hardware_authorization(
            execution_mode=self.execution_mode,
            enabled=self.hardware_enabled,
        )
        try:
            rclpy = importlib.import_module("rclpy")
            dr_init = importlib.import_module("DR_init")
        except ImportError as exc:
            missing_module = exc.name or "rclpy/DR_init"
            raise NotConfiguredError(
                "hand-eye robot control cannot import "
                f"{missing_module!r}; source ROS 2 and the Doosan workspace, then preserve "
                "the sourced PYTHONPATH when adding this repository's src directory"
            ) from exc
        self._rclpy = rclpy
        try:
            if not bool(rclpy.ok()):
                rclpy.init(args=None)
                self._owns_rclpy = True
            node = rclpy.create_node(
                "dittobot_handeye_calibration", namespace=self.robot_id
            )
        except Exception as exc:
            cyclone_uri = os.environ.get("CYCLONEDDS_URI", "")
            invalid_interface_hint = (
                " CYCLONEDDS_URI contains an empty NetworkInterface name; set it to the "
                "robot-network interface before starting the API."
                if 'name=""' in cyclone_uri
                else ""
            )
            self.disconnect()
            raise NotConfiguredError(
                f"failed to create the calibration ROS 2 node: {exc}.{invalid_interface_hint}"
            ) from exc
        _set_module_attribute(dr_init, "__dsr__id", self.robot_id)
        # The supplied tutorial contains both spellings; set the legacy alias before import too.
        _set_module_attribute(dr_init, "dsr__id", self.robot_id)
        _set_module_attribute(dr_init, "__dsr__model", self.robot_model)
        _set_module_attribute(dr_init, "__dsr__node", node)
        self._node = node
        try:
            existing_dsr = sys.modules.get("DSR_ROBOT2")
            dsr = (
                importlib.reload(existing_dsr)
                if isinstance(existing_dsr, ModuleType)
                else importlib.import_module("DSR_ROBOT2")
            )
            dr_common = importlib.import_module("DR_common2")
            dsr_services = importlib.import_module("dsr_msgs2.srv")
        except ImportError as exc:
            self.disconnect()
            missing_module = exc.name or "DSR_ROBOT2/DR_common2/dsr_msgs2"
            raise NotConfiguredError(
                "hand-eye robot control cannot import "
                f"{missing_module!r}; source the installed Doosan workspace and preserve its "
                "PYTHONPATH"
            ) from exc
        required = {
            "get_current_posj": getattr(dsr, "get_current_posj", None),
            "get_current_tool_flange_posx": getattr(
                dsr, "get_current_tool_flange_posx", None
            ),
            "get_current_posx": getattr(dsr, "get_current_posx", None),
            "get_tcp": getattr(dsr, "get_tcp", None),
            "get_robot_state": getattr(dsr, "get_robot_state", None),
            "movej": getattr(dsr, "movej", None),
            "mwait": getattr(dsr, "mwait", None),
            "posj": getattr(dr_common, "posj", None),
        }
        missing = sorted(name for name, value in required.items() if not callable(value))
        if missing:
            self.disconnect()
            raise NotConfiguredError(
                "installed DSR_ROBOT2 calibration API is incomplete: " + ", ".join(missing)
            )
        self._get_current_posj = required["get_current_posj"]
        self._get_current_tool_flange_posx = required["get_current_tool_flange_posx"]
        self._get_current_posx = required["get_current_posx"]
        self._get_tcp = required["get_tcp"]
        self._get_robot_state = required["get_robot_state"]
        self._movej = required["movej"]
        self._mwait = required["mwait"]
        self._posj = required["posj"]
        service_types = {
            "aux_control/get_current_posj": getattr(
                dsr_services, "GetCurrentPosj", None
            ),
            "aux_control/get_current_tool_flange_posx": getattr(
                dsr_services, "GetCurrentToolFlangePosx", None
            ),
            "aux_control/get_current_posx": getattr(
                dsr_services, "GetCurrentPosx", None
            ),
            "tcp/get_current_tcp": getattr(dsr_services, "GetCurrentTcp", None),
            "system/get_robot_state": getattr(dsr_services, "GetRobotState", None),
            "motion/move_joint": getattr(dsr_services, "MoveJoint", None),
            "motion/move_wait": getattr(dsr_services, "MoveWait", None),
            "motion/move_stop": getattr(dsr_services, "MoveStop", None),
        }
        missing_types = sorted(name for name, value in service_types.items() if value is None)
        if missing_types:
            self.disconnect()
            raise NotConfiguredError(
                "installed dsr_msgs2 is missing calibration service types: "
                + ", ".join(missing_types)
            )
        probe_clients: list[Any] = []
        try:
            for service_name, service_type in service_types.items():
                client = node.create_client(service_type, service_name)
                probe_clients.append(client)
                if not client.wait_for_service(timeout_sec=2.0):
                    raise NotConfiguredError(
                        f"Doosan bringup service /{self.robot_id}/{service_name} is unavailable"
                    )
            self._move_stop_type = service_types["motion/move_stop"]
            self._move_stop_client = node.create_client(
                self._move_stop_type, "motion/move_stop"
            )
            self._current_posx_type = service_types["aux_control/get_current_posx"]
            self._current_posx_client = node.create_client(
                self._current_posx_type, "aux_control/get_current_posx"
            )
        except Exception:
            self.disconnect()
            raise
        finally:
            for client in probe_clients:
                with suppress(Exception):
                    node.destroy_client(client)
        # Read-only calls prove that the configured namespace exposes both required frames.
        try:
            self._require_standby()
            self.get_joint_positions_rad()
            self.get_base_to_flange_matrix()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        node, rclpy = self._node, self._rclpy
        self._node = None
        if node is not None:
            for client in (self._move_stop_client, self._current_posx_client):
                if client is not None:
                    node.destroy_client(client)
            self._move_stop_client = None
            self._move_stop_type = None
            self._current_posx_client = None
            self._current_posx_type = None
            node.destroy_node()
        if self._owns_rclpy and rclpy is not None and bool(rclpy.ok()):
            rclpy.shutdown()
        self._owns_rclpy = False
        self._rclpy = None

    def get_joint_positions_rad(self) -> JointVector:
        function = self._required(self._get_current_posj, "get_current_posj")
        values = _pose_values(function(), label="joint position")
        return cast(JointVector, tuple(math.radians(value) for value in values))

    def get_base_to_flange_matrix(self) -> Matrix44:
        function = self._required(
            self._get_current_tool_flange_posx, "get_current_tool_flange_posx"
        )
        return _cartesian_pose_matrix(function(), label="tool flange pose")

    def get_base_to_tcp_matrix(self) -> Matrix44:
        return _cartesian_pose_matrix(
            self._read_active_tcp_pose(), label="active TCP pose"
        )

    def _read_active_tcp_pose(self) -> JointVector:
        with self._current_posx_lock:
            if (
                self._rclpy is not None
                and self._node is not None
                and self._current_posx_client is not None
                and self._current_posx_type is not None
            ):
                return self._read_active_tcp_pose_service()
            return self._read_active_tcp_pose_driver()

    def _read_active_tcp_pose_service(self) -> JointVector:
        rclpy = self._rclpy
        node = self._node
        client = self._current_posx_client
        service_type = self._current_posx_type
        if rclpy is None or node is None or client is None or service_type is None:
            raise NotConfiguredError("Doosan active-TCP pose service is not configured")
        last_error = "empty response"
        for attempt in range(1, self._ACTIVE_TCP_POSE_ATTEMPTS + 1):
            request = service_type.Request()
            request.ref = 0
            future = client.call_async(request)
            rclpy.spin_until_future_complete(
                node,
                future,
                timeout_sec=self._ACTIVE_TCP_POSE_TIMEOUT_S,
            )
            if not future.done():
                with suppress(Exception):
                    future.cancel()
                last_error = (
                    f"no response within {self._ACTIVE_TCP_POSE_TIMEOUT_S:g}s"
                )
            else:
                try:
                    response = future.result()
                except Exception as exc:
                    last_error = f"service error: {exc}"
                else:
                    task_positions = getattr(response, "task_pos_info", ())
                    success = getattr(response, "success", False)
                    if success is True and task_positions:
                        values = getattr(task_positions[0], "data", ())
                        if len(values) >= 6:
                            return _pose_values(
                                list(values[:6]), label="Doosan active TCP pose"
                            )
                    last_error = "controller returned success=false or an empty pose"
            if attempt < self._ACTIVE_TCP_POSE_ATTEMPTS:
                time.sleep(self._ACTIVE_TCP_POSE_RETRY_DELAY_S)
        raise NotConfiguredError(
            "Doosan active-TCP pose service failed after "
            f"{self._ACTIVE_TCP_POSE_ATTEMPTS} bounded attempts ({last_error})"
        )

    def _read_active_tcp_pose_driver(self) -> JointVector:
        function = self._required(self._get_current_posx, "get_current_posx")
        last_error = "empty response"
        for attempt in range(1, self._ACTIVE_TCP_POSE_ATTEMPTS + 1):
            try:
                value = function()
                pose = (
                    value[0]
                    if isinstance(value, tuple) and len(value) == 2
                    else value
                )
                if _empty_pose_response(pose):
                    last_error = "empty active-TCP response"
                    if attempt < self._ACTIVE_TCP_POSE_ATTEMPTS:
                        time.sleep(self._ACTIVE_TCP_POSE_RETRY_DELAY_S)
                    continue
                return _pose_values(pose, label="Doosan active TCP pose")
            except IndexError:
                last_error = "empty active-TCP response"
            if attempt < self._ACTIVE_TCP_POSE_ATTEMPTS:
                time.sleep(self._ACTIVE_TCP_POSE_RETRY_DELAY_S)
        raise NotConfiguredError(
            "Doosan get_current_posx failed after "
            f"{self._ACTIVE_TCP_POSE_ATTEMPTS} bounded attempts ({last_error})"
        )

    def get_active_tcp_name(self) -> str:
        function = self._required(self._get_tcp, "get_tcp")
        value = function()
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("Doosan did not return a valid active TCP name")
        return value.strip()

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None:
        if velocity_rad_s <= 0.0 or acceleration_rad_s2 <= 0.0:
            raise ValueError("calibration joint speed and acceleration must be positive")
        self._require_standby()
        movej = self._required(self._movej, "movej")
        mwait = self._required(self._mwait, "mwait")
        posj = self._required(self._posj, "posj")
        target_deg = [math.degrees(value) for value in target_rad]
        move_result = movej(
            posj(target_deg),
            vel=math.degrees(velocity_rad_s),
            acc=math.degrees(acceleration_rad_s2),
        )
        if move_result != 0:
            raise RuntimeError("Doosan movej rejected the calibration waypoint")
        wait_result = mwait()
        if wait_result != 0:
            raise RuntimeError("Doosan mwait did not confirm calibration motion completion")
        self._require_standby()

    def stop(self, *, reason: str) -> None:
        del reason
        if self._node is None or self._rclpy is None:
            raise NotConfiguredError("Doosan calibration adapter is not connected: move_stop")
        client = self._move_stop_client
        service_type = self._move_stop_type
        if client is None or service_type is None:
            raise NotConfiguredError("Doosan MoveStop service is not configured")
        if not client.wait_for_service(timeout_sec=1.0):
            raise NotConfiguredError("Doosan MoveStop service is unavailable")
        request = service_type.Request()
        request.stop_mode = 0  # STOP_TYPE_QUICK_STO / DR_QSTOP_STO
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        if not future.done():
            raise RuntimeError("Doosan MoveStop did not respond within five seconds")
        result = future.result()
        if result is None or result.success is not True:
            raise RuntimeError("Doosan MoveStop rejected the calibration stop request")

    def _require_standby(self) -> None:
        function = self._required(self._get_robot_state, "get_robot_state")
        state = int(function())
        if state != 1:  # DRFC.STATE_STANDBY
            state_name = DOOSAN_ROBOT_STATE_NAMES.get(state, f"UNKNOWN_STATE_{state}")
            raise ValueError(
                "Doosan robot must be in STATE_STANDBY; "
                f"current state is {state_name} ({state})"
            )

    @staticmethod
    def _required(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
        if value is None:
            raise NotConfiguredError(f"Doosan calibration adapter is not connected: {name}")
        return value


def _cartesian_pose_matrix(value: Any, *, label: str) -> Matrix44:
    x_mm, y_mm, z_mm, first_deg, second_deg, third_deg = _pose_values(
        value, label=label
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rotation_zyz(
        math.radians(first_deg),
        math.radians(second_deg),
        math.radians(third_deg),
    )
    matrix[:3, 3] = np.asarray([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return matrix


def _pose_values(value: Any, *, label: str) -> JointVector:
    candidate = value
    if isinstance(value, Sequence) and len(value) == 2 and isinstance(value[0], Sequence):
        candidate = value[0]
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        raise NotConfiguredError(f"DSR {label} response is not a six-value sequence")
    if len(candidate) != 6:
        raise NotConfiguredError(f"DSR {label} response must contain six values")
    result = tuple(float(item) for item in candidate)
    if not all(math.isfinite(item) for item in result):
        raise NotConfiguredError(f"DSR {label} response contains a non-finite value")
    return cast(JointVector, result)


def _rotation_zyz(first: float, second: float, third: float) -> NDArray[np.float64]:
    """Match the ZYZ convention used by the supplied M0609 tutorial."""

    cos_a, sin_a = math.cos(first), math.sin(first)
    cos_b, sin_b = math.cos(second), math.sin(second)
    cos_c, sin_c = math.cos(third), math.sin(third)
    rotate_z_first = np.asarray(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
    )
    rotate_y = np.asarray(
        [[cos_b, 0.0, sin_b], [0.0, 1.0, 0.0], [-sin_b, 0.0, cos_b]]
    )
    rotate_z_third = np.asarray(
        [[cos_c, -sin_c, 0.0], [sin_c, cos_c, 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(rotate_z_first @ rotate_y @ rotate_z_third, dtype=np.float64)


__all__ = [
    "CalibrationWaypoint",
    "DoosanHandEyeCalibrationRobot",
    "HandEyeCalibrationRobot",
    "JointVector",
    "approved_eye_in_hand_waypoints",
]
