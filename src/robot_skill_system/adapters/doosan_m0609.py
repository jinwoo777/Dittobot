"""ROS 2/DSR_ROBOT2 binding for the commissioned Doosan M0609 cell.

The adapter keeps the rest of the application in SI units.  Conversion to the
Doosan API's millimetres, degrees and ZYZ task-pose convention happens only at
this boundary.  Imports and ROS node creation are deliberately lazy so Mock
mode remains usable without ROS installed or sourced.
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from types import ModuleType
from typing import Any, cast

from robot_skill_system.adapters.errors import (
    NotConfiguredError,
    require_hardware_authorization,
)
from robot_skill_system.adapters.robot import (
    RobotCommand,
    RobotOperationalState,
    RobotState,
)
from robot_skill_system.runtime.models import BoundTargetPose


def _set_module_attribute(module: ModuleType, name: str, value: Any) -> None:
    setattr(module, name, value)


class DoosanM0609Adapter:
    """Blocking, fail-closed adapter for the installed DSR_ROBOT2 API.

    The robot controller remains responsible for its configured joint, speed,
    collision and safety limits.  Application preflight must pass before any
    method that produces motion is called.
    """

    adapter_name = "doosan_m0609_dsr_robot2"

    def __init__(
        self,
        *,
        robot_id: str = "dsr01",
        robot_model: str = "m0609",
        execution_mode: str = "mock",
        hardware_enabled: bool = False,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.robot_id = robot_id
        self.robot_model = robot_model
        self.execution_mode = execution_mode
        self.hardware_enabled = hardware_enabled
        self._clock_ns = clock_ns
        self._rclpy: ModuleType | None = None
        self._node: Any | None = None
        self._owns_rclpy = False
        self._connected = False
        self._dsr: ModuleType | None = None
        self._posj: Callable[..., Any] | None = None
        self._posx: Callable[..., Any] | None = None
        self._move_stop_client: Any | None = None
        self._move_stop_type: Any | None = None
        self.commands: list[RobotCommand] = []

    def connect(self) -> None:
        require_hardware_authorization(
            execution_mode=self.execution_mode,
            enabled=self.hardware_enabled,
        )
        if self._connected:
            return
        try:
            rclpy = importlib.import_module("rclpy")
            dr_init = importlib.import_module("DR_init")
        except ImportError as exc:
            raise NotConfiguredError(
                "Doosan control requires sourced ROS 2 and DSR_ROBOT2; source "
                "/opt/ros/humble/setup.bash and the Doosan workspace before the API"
            ) from exc
        self._rclpy = rclpy
        try:
            if not bool(rclpy.ok()):
                rclpy.init(args=None)
                self._owns_rclpy = True
            self._node = rclpy.create_node(
                "dittobot_skill_runtime", namespace=self.robot_id
            )
            _set_module_attribute(dr_init, "__dsr__id", self.robot_id)
            _set_module_attribute(dr_init, "dsr__id", self.robot_id)
            _set_module_attribute(dr_init, "__dsr__model", self.robot_model)
            _set_module_attribute(dr_init, "__dsr__node", self._node)
            existing = sys.modules.get("DSR_ROBOT2")
            self._dsr = (
                importlib.reload(existing)
                if isinstance(existing, ModuleType)
                else importlib.import_module("DSR_ROBOT2")
            )
            dr_common = importlib.import_module("DR_common2")
            services = importlib.import_module("dsr_msgs2.srv")
            self._posj = self._require_callable(dr_common, "posj")
            self._posx = self._require_callable(dr_common, "posx")
            for name in (
                "get_robot_state",
                "get_current_posj",
                "get_current_posx",
                "get_tool_force",
                "movej",
                "movel",
                "movec",
                "move_periodic",
                "mwait",
                "task_compliance_ctrl",
                "set_desired_force",
                "release_force",
                "release_compliance_ctrl",
            ):
                self._require_callable(self._dsr, name)
            self._probe_services(services)
            self._connected = True
            state = self.get_state()
            if not state.connected or state.emergency_stop_active:
                raise RuntimeError(
                    f"Doosan is not ready after connection: {state.operational_state.value}"
                )
            # These read-only calls verify the response shapes before execution.
            self.get_joint_positions()
            self.get_current_pose()
            self._record("connect", robot_id=self.robot_id, robot_model=self.robot_model)
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        node, rclpy = self._node, self._rclpy
        self._connected = False
        self._dsr = None
        self._posj = None
        self._posx = None
        self._node = None
        if node is not None:
            if self._move_stop_client is not None:
                with suppress(Exception):
                    node.destroy_client(self._move_stop_client)
            with suppress(Exception):
                node.destroy_node()
        self._move_stop_client = None
        self._move_stop_type = None
        if self._owns_rclpy and rclpy is not None and bool(rclpy.ok()):
            with suppress(Exception):
                rclpy.shutdown()
        self._owns_rclpy = False
        self._rclpy = None

    def get_state(self) -> RobotState:
        if not self._connected or self._dsr is None:
            return RobotState(
                operational_state=RobotOperationalState.DISCONNECTED,
                connected=False,
                emergency_stop_active=False,
            )
        state_number = int(self._call("get_robot_state"))
        state_map = {
            1: RobotOperationalState.IDLE,
            2: RobotOperationalState.MOVING,
            5: RobotOperationalState.STOPPED,
            6: RobotOperationalState.STOPPED,
            9: RobotOperationalState.STOPPED,
        }
        fault = None if state_number in {1, 2, 4, 5, 6, 7, 8, 9} else str(state_number)
        return RobotState(
            operational_state=state_map.get(state_number, RobotOperationalState.FAULT),
            connected=True,
            emergency_stop_active=state_number == 6,
            protective_stop_active=state_number in {5, 9},
            fault_code=fault,
        )

    def get_current_pose(self) -> BoundTargetPose:
        value = self._call("get_current_posx", 0)
        pose = value[0] if isinstance(value, tuple) and len(value) == 2 else value
        values = _six_values(pose, label="current task pose")
        quaternion = _zyz_degrees_to_quaternion(values[3], values[4], values[5])
        return BoundTargetPose(
            frame_id="base",
            position_m=(values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0),
            orientation_xyzw=quaternion,
            anchor_entity_id="robot_base",
            timestamp_ns=self._clock_ns(),
        )

    def get_joint_positions(self) -> tuple[float, ...]:
        values = _six_values(self._call("get_current_posj"), label="joint position")
        return tuple(math.radians(value) for value in values)

    def move_j(
        self,
        target: Any,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        self._require_standby()
        values = _six_values(target, label="joint target")
        target_deg = [math.degrees(value) for value in values]
        result = self._call(
            "movej",
            self._required_value(self._posj, "posj")(target_deg),
            vel=math.degrees(velocity_rad_s),
            acc=math.degrees(acceleration_rad_s2),
            radius=blend_radius_m * 1000.0,
        )
        self._finish_motion(result, "move_j")
        self._record("move_j", target_deg=target_deg)

    def move_l(
        self,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        self._require_standby()
        target = _target_pose_to_dsr(target_pose)
        result = self._call(
            "movel",
            self._required_value(self._posx, "posx")(target),
            vel=velocity_m_s * 1000.0,
            acc=acceleration_m_s2 * 1000.0,
            radius=blend_radius_m * 1000.0,
            ref=0,
        )
        self._finish_motion(result, "move_l")
        self._record("move_l", target_mm_deg=target)

    def move_c(
        self,
        via_pose: Any,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        self._require_standby()
        via = _target_pose_to_dsr(via_pose)
        target = _target_pose_to_dsr(target_pose)
        posx = self._required_value(self._posx, "posx")
        result = self._call(
            "movec",
            posx(via),
            posx(target),
            vel=velocity_m_s * 1000.0,
            acc=acceleration_m_s2 * 1000.0,
            radius=blend_radius_m * 1000.0,
            ref=0,
        )
        self._finish_motion(result, "move_c")
        self._record("move_c", via_mm_deg=via, target_mm_deg=target)

    def move_periodic(
        self,
        center_pose: Any,
        amplitude_m: Any,
        *,
        repetitions: int,
        velocity_m_s: float,
        acceleration_m_s2: float,
    ) -> None:
        # Move to the validated center first; Doosan periodic amplitude is tool-relative.
        self.move_l(
            center_pose,
            velocity_m_s=velocity_m_s,
            acceleration_m_s2=acceleration_m_s2,
        )
        amplitudes = _sequence(amplitude_m, label="periodic amplitude")
        if len(amplitudes) not in {3, 6}:
            raise ValueError("periodic amplitude must contain three or six values")
        dsr_amplitude = [value * 1000.0 for value in amplitudes[:3]]
        dsr_amplitude.extend(
            math.degrees(value) for value in (amplitudes[3:] if len(amplitudes) == 6 else (0, 0, 0))
        )
        period_s = max(0.2, 4.0 * max(abs(value) for value in amplitudes[:3]) / velocity_m_s)
        result = self._call(
            "move_periodic",
            dsr_amplitude,
            period_s,
            atime=min(1.0, velocity_m_s / acceleration_m_s2),
            repeat=repetitions,
            ref=1,
        )
        self._finish_motion(result, "move_periodic")
        self._record("move_periodic", amplitude_mm_deg=dsr_amplitude)

    def wait(self, *, duration_s: float) -> None:
        if duration_s < 0.0:
            raise ValueError("wait duration cannot be negative")
        time.sleep(duration_s)
        self._record("wait", duration_s=duration_s)

    def stop(self, *, reason: str) -> None:
        node, rclpy = self._node, self._rclpy
        client, service_type = self._move_stop_client, self._move_stop_type
        if node is None or rclpy is None or client is None or service_type is None:
            raise NotConfiguredError("Doosan MoveStop service is not connected")
        request = service_type.Request()
        request.stop_mode = 0
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result() if future.done() else None
        if result is None or result.success is not True:
            raise RuntimeError("Doosan MoveStop did not confirm a quick stop")
        self._record("stop", reason=reason)

    def start_compliance(self, *, stiffness_n_m: Sequence[float]) -> None:
        values = list(_six_values(stiffness_n_m, label="compliance stiffness"))
        self._accepted(self._call("task_compliance_ctrl", stx=values), "start_compliance")

    def set_desired_force(
        self, *, force_vector_n: Sequence[float], force_control_axes: Sequence[bool]
    ) -> None:
        force = list(_six_values(force_vector_n, label="desired force"))
        axes = list(_six_values(force_control_axes, label="force control axes"))
        self._accepted(
            self._call("set_desired_force", fd=force, dir=[int(bool(v)) for v in axes]),
            "set_desired_force",
        )

    def release_force(self, *, release_time_s: float) -> None:
        self._accepted(self._call("release_force", time=release_time_s), "release_force")

    def release_compliance(self) -> None:
        self._accepted(self._call("release_compliance_ctrl"), "release_compliance")

    def read_tool_force(self) -> tuple[float, ...]:
        return _six_values(self._call("get_tool_force", 0), label="tool force")

    def is_emergency_stop_active(self) -> bool:
        return self.get_state().emergency_stop_active

    def safe_retract(self, *, direction_xyz: Sequence[float], distance_m: float) -> None:
        direction = _sequence(direction_xyz, label="retract direction")
        if len(direction) != 3 or not 0.0 < distance_m <= 0.10:
            raise ValueError("safe retract requires a 3D direction and distance in (0, 0.10] m")
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-9:
            raise ValueError("safe retract direction cannot be zero")
        current = self.get_current_pose()
        target = BoundTargetPose(
            frame_id="base",
            position_m=(
                current.position_m[0] + direction[0] / norm * distance_m,
                current.position_m[1] + direction[1] / norm * distance_m,
                current.position_m[2] + direction[2] / norm * distance_m,
            ),
            orientation_xyzw=current.orientation_xyzw,
            anchor_entity_id="robot_base",
            timestamp_ns=self._clock_ns(),
        )
        self.move_l(target, velocity_m_s=0.02, acceleration_m_s2=0.05)

    def _probe_services(self, services: ModuleType) -> None:
        if self._node is None:
            raise NotConfiguredError("Doosan ROS node was not created")
        required = {
            "system/get_robot_state": "GetRobotState",
            "aux_control/get_current_posj": "GetCurrentPosj",
            "aux_control/get_current_posx": "GetCurrentPosx",
            "motion/move_joint": "MoveJoint",
            "motion/move_line": "MoveLine",
            "motion/move_wait": "MoveWait",
            "motion/move_stop": "MoveStop",
        }
        clients: list[Any] = []
        try:
            for service_name, type_name in required.items():
                service_type = getattr(services, type_name, None)
                if service_type is None:
                    raise NotConfiguredError(f"dsr_msgs2 is missing {type_name}")
                client = self._node.create_client(service_type, service_name)
                clients.append(client)
                if not client.wait_for_service(timeout_sec=2.0):
                    raise NotConfiguredError(
                        f"Doosan bringup service /{self.robot_id}/{service_name} is unavailable"
                    )
                if service_name == "motion/move_stop":
                    self._move_stop_type = service_type
            if self._move_stop_type is None:
                raise NotConfiguredError("MoveStop service type is unavailable")
            self._move_stop_client = self._node.create_client(
                self._move_stop_type, "motion/move_stop"
            )
        finally:
            for client in clients:
                with suppress(Exception):
                    self._node.destroy_client(client)

    def _require_standby(self) -> None:
        state = self.get_state()
        if state.operational_state is not RobotOperationalState.IDLE:
            raise RuntimeError(
                "Doosan must be in STATE_STANDBY before motion, got "
                f"{state.operational_state.value}"
            )

    def _finish_motion(self, result: Any, operation: str) -> None:
        self._accepted(result, operation)
        self._accepted(self._call("mwait"), f"{operation}/mwait")
        self._require_standby()

    @staticmethod
    def _accepted(result: Any, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"Doosan rejected {operation}")

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if not self._connected or self._dsr is None:
            raise NotConfiguredError(f"Doosan adapter is not connected: {name}")
        return self._require_callable(self._dsr, name)(*args, **kwargs)

    @staticmethod
    def _require_callable(module: ModuleType, name: str) -> Callable[..., Any]:
        value = getattr(module, name, None)
        if not callable(value):
            raise NotConfiguredError(f"installed DSR_ROBOT2 is missing callable {name}")
        return cast(Callable[..., Any], value)

    @staticmethod
    def _required_value(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
        if value is None:
            raise NotConfiguredError(f"Doosan adapter is not connected: {name}")
        return value

    def _record(self, operation: str, **arguments: Any) -> None:
        self.commands.append(
            RobotCommand(
                sequence=len(self.commands) + 1,
                timestamp_ns=self._clock_ns(),
                operation=operation,
                arguments=arguments,
            )
        )


def _sequence(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a numeric sequence")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _six_values(value: Any, *, label: str) -> tuple[float, float, float, float, float, float]:
    result = _sequence(value, label=label)
    if len(result) != 6:
        raise ValueError(f"{label} must contain six values")
    return result


def _get(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError("position must contain three values")
        return float(value[0]), float(value[1]), float(value[2])
    return float(_get(value, "x")), float(_get(value, "y")), float(_get(value, "z"))


def _xyzw(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise ValueError("orientation must contain four values")
        result = tuple(float(item) for item in value)
    else:
        result = (
            float(_get(value, "x")),
            float(_get(value, "y")),
            float(_get(value, "z")),
            float(_get(value, "w")),
        )
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1e-12:
        raise ValueError("orientation quaternion cannot be zero")
    return cast(tuple[float, float, float, float], tuple(item / norm for item in result))


def _target_pose_to_dsr(value: Any) -> list[float]:
    position = _xyz(_get(value, "position_m", "position"))
    quaternion = _xyzw(_get(value, "orientation_xyzw", "orientation"))
    zyz = _quaternion_to_zyz_degrees(quaternion)
    return [position[0] * 1000.0, position[1] * 1000.0, position[2] * 1000.0, *zyz]


def _quaternion_to_zyz_degrees(
    quaternion: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    beta = math.acos(max(-1.0, min(1.0, rotation[2][2])))
    if abs(math.sin(beta)) > 1e-9:
        alpha = math.atan2(rotation[1][2], rotation[0][2])
        gamma = math.atan2(rotation[2][1], -rotation[2][0])
    else:
        alpha = math.atan2(rotation[1][0], rotation[0][0])
        gamma = 0.0
    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


def _zyz_degrees_to_quaternion(
    alpha_deg: float, beta_deg: float, gamma_deg: float
) -> tuple[float, float, float, float]:
    alpha, beta, gamma = map(math.radians, (alpha_deg, beta_deg, gamma_deg))
    half_sum = 0.5 * (alpha + gamma)
    half_difference = 0.5 * (alpha - gamma)
    half_beta = 0.5 * beta
    return (
        -math.sin(half_difference) * math.sin(half_beta),
        math.cos(half_difference) * math.sin(half_beta),
        math.sin(half_sum) * math.cos(half_beta),
        math.cos(half_sum) * math.cos(half_beta),
    )


__all__ = ["DoosanM0609Adapter"]
