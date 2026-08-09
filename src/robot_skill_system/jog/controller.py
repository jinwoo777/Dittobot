"""Fail-closed MoveJ and MoveL jog controller for the Doosan M0609."""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, cast

from robot_skill_system.calibration.robot import DoosanHandEyeCalibrationRobot
from robot_skill_system.exceptions import HardwareExecutionDenied, NotConfiguredError

JointVector = tuple[float, float, float, float, float, float]
CartesianPose = tuple[float, float, float, float, float, float]

# Conservative default safety angle ranges published for the M-series controller.
M0609_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-360.0, 360.0),
    (-95.0, 95.0),
    (-135.0, 135.0),
    (-360.0, 360.0),
    (-135.0, 135.0),
    (-360.0, 360.0),
)
MAXIMUM_JOG_STEP_DEG = 5.0
MAXIMUM_MOVEL_TRANSLATION_MM = 100.0
MAXIMUM_MOVEL_ROTATION_DEG = 5.0
MAXIMUM_TCP_BASE_RADIUS_MM = 1000.0


def _cartesian_pose_values(value: object, *, label: str) -> CartesianPose:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain six finite values")
    try:
        values = tuple(float(item) for item in cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain six finite values") from exc
    if len(values) != 6 or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain six finite values")
    return cast(CartesianPose, values)


class JogRobot(Protocol):
    adapter_name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions_rad(self) -> JointVector: ...

    def get_tcp_pose_base_mm_zyz_deg(self) -> CartesianPose: ...

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None: ...

    def move_linear(
        self,
        target_pose_base_mm_zyz_deg: CartesianPose,
        *,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None: ...

    def stop(self, *, reason: str) -> None: ...


class MockJogRobot:
    """In-memory robot used unless every real-hardware gate is enabled."""

    adapter_name = "mock_m0609_web_jog"

    def __init__(self) -> None:
        self.connected = False
        self.joint_positions_rad: JointVector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.tcp_pose_base_mm_zyz_deg: CartesianPose = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_positions_rad(self) -> JointVector:
        if not self.connected:
            raise RuntimeError("mock jog robot is not connected")
        return self.joint_positions_rad

    def get_tcp_pose_base_mm_zyz_deg(self) -> CartesianPose:
        if not self.connected:
            raise RuntimeError("mock jog robot is not connected")
        return self.tcp_pose_base_mm_zyz_deg

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None:
        if not self.connected:
            raise RuntimeError("mock jog robot is not connected")
        if velocity_rad_s <= 0.0 or acceleration_rad_s2 <= 0.0:
            raise ValueError("jog motion profile limits must be positive")
        self.joint_positions_rad = target_rad

    def move_linear(
        self,
        target_pose_base_mm_zyz_deg: CartesianPose,
        *,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None:
        if not self.connected:
            raise RuntimeError("mock jog robot is not connected")
        if min(
            linear_velocity_m_s,
            linear_acceleration_m_s2,
            angular_velocity_rad_s,
            angular_acceleration_rad_s2,
        ) <= 0.0:
            raise ValueError("jog MoveL profile limits must be positive")
        self.tcp_pose_base_mm_zyz_deg = target_pose_base_mm_zyz_deg

    def stop(self, *, reason: str) -> None:
        _ = reason


class DoosanJogRobot(DoosanHandEyeCalibrationRobot):
    """Add the supplied DSR MoveL API to the audited Doosan jog boundary."""

    adapter_name = "doosan_m0609_web_jog"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._movel: Callable[..., Any] | None = None
        self._posx: Callable[..., Any] | None = None
        self._dr_base: int | None = None

    def connect(self) -> None:
        super().connect()
        try:
            dsr = importlib.import_module("DSR_ROBOT2")
            dr_common = importlib.import_module("DR_common2")
            movel = getattr(dsr, "movel", None)
            posx = getattr(dr_common, "posx", None)
            dr_base = getattr(dsr, "DR_BASE", None)
            if not callable(movel) or not callable(posx) or not isinstance(dr_base, int):
                raise NotConfiguredError(
                    "installed DSR_ROBOT2 jog API requires movel, posx, and DR_BASE"
                )
            self._movel = movel
            self._posx = posx
            self._dr_base = dr_base
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        self._movel = None
        self._posx = None
        self._dr_base = None
        super().disconnect()

    def get_tcp_pose_base_mm_zyz_deg(self) -> CartesianPose:
        function = self._required(self._get_current_posx, "get_current_posx")
        value = function()
        pose = value[0] if isinstance(value, tuple) and len(value) == 2 else value
        return _cartesian_pose_values(pose, label="Doosan active TCP pose")

    def move_linear(
        self,
        target_pose_base_mm_zyz_deg: CartesianPose,
        *,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None:
        if min(
            linear_velocity_m_s,
            linear_acceleration_m_s2,
            angular_velocity_rad_s,
            angular_acceleration_rad_s2,
        ) <= 0.0:
            raise ValueError("jog MoveL profile limits must be positive")
        self._require_standby()
        movel = self._required(self._movel, "movel")
        posx = self._required(self._posx, "posx")
        mwait = self._required(self._mwait, "mwait")
        if self._dr_base is None:
            raise NotConfiguredError("Doosan DR_BASE reference is not configured")
        result = movel(
            posx(list(target_pose_base_mm_zyz_deg)),
            vel=[
                linear_velocity_m_s * 1000.0,
                math.degrees(angular_velocity_rad_s),
            ],
            acc=[
                linear_acceleration_m_s2 * 1000.0,
                math.degrees(angular_acceleration_rad_s2),
            ],
            ref=self._dr_base,
        )
        if result != 0:
            raise RuntimeError("Doosan movel rejected the jog target")
        if mwait() != 0:
            raise RuntimeError("Doosan mwait did not confirm jog MoveL completion")
        self._require_standby()


class JogController:
    """Own one acknowledged jog session and execute validated MoveJ/MoveL targets."""

    def __init__(
        self,
        *,
        robot_factory: Callable[[], JogRobot],
        mode: str,
        hardware_authorized: bool,
        gate_summary: dict[str, bool],
        joint_velocity_rad_s: float,
        joint_acceleration_rad_s2: float,
        linear_velocity_m_s: float | None = None,
        linear_acceleration_m_s2: float | None = None,
        angular_velocity_rad_s: float | None = None,
        angular_acceleration_rad_s2: float | None = None,
        maximum_step_deg: float = MAXIMUM_JOG_STEP_DEG,
    ) -> None:
        if mode not in {"mock", "hardware"}:
            raise ValueError("jog mode must be mock or hardware")
        if joint_velocity_rad_s <= 0.0 or joint_acceleration_rad_s2 <= 0.0:
            raise ValueError("jog motion profile limits must be positive")
        if maximum_step_deg <= 0.0 or maximum_step_deg > MAXIMUM_JOG_STEP_DEG:
            raise ValueError("maximum jog step must be within (0, 5] degrees")
        cartesian_profile = (
            linear_velocity_m_s,
            linear_acceleration_m_s2,
            angular_velocity_rad_s,
            angular_acceleration_rad_s2,
        )
        if any(value is not None for value in cartesian_profile) and (
            any(value is None for value in cartesian_profile)
            or any(cast(float, value) <= 0.0 for value in cartesian_profile)
        ):
            raise ValueError("complete positive MoveL profile limits are required")
        self._robot_factory = robot_factory
        self.mode = mode
        self.hardware_authorized = hardware_authorized
        self.gate_summary = dict(gate_summary)
        self.joint_velocity_rad_s = joint_velocity_rad_s
        self.joint_acceleration_rad_s2 = joint_acceleration_rad_s2
        self.linear_velocity_m_s = linear_velocity_m_s
        self.linear_acceleration_m_s2 = linear_acceleration_m_s2
        self.angular_velocity_rad_s = angular_velocity_rad_s
        self.angular_acceleration_rad_s2 = angular_acceleration_rad_s2
        self.maximum_step_deg = maximum_step_deg
        self._lock = threading.RLock()
        self._robot: JogRobot | None = None
        self._enabled = False
        self._operator_id: str | None = None
        self._enabled_at_ns: int | None = None
        self._last_move_at_ns: int | None = None
        self._last_error: str | None = None
        self._joint_positions_rad: JointVector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._tcp_pose_base_mm_zyz_deg: CartesianPose = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def _capabilities(self) -> dict[str, Any]:
        movel_available = all(
            value is not None
            for value in (
                self.linear_velocity_m_s,
                self.linear_acceleration_m_s2,
                self.angular_velocity_rad_s,
                self.angular_acceleration_rad_s2,
            )
        )
        return {
            "mode": self.mode,
            "hardware_authorized": self.hardware_authorized,
            "gate_summary": dict(self.gate_summary),
            "failed_gates": [name for name, passed in self.gate_summary.items() if not passed],
            "maximum_step_deg": self.maximum_step_deg,
            "joint_limits_deg": [
                {"joint_index": index, "minimum": limits[0], "maximum": limits[1]}
                for index, limits in enumerate(M0609_JOINT_LIMITS_DEG, start=1)
            ],
            "motion_profile_id": "joint_safe",
            "joint_velocity_deg_s": round(math.degrees(self.joint_velocity_rad_s), 3),
            "joint_acceleration_deg_s2": round(
                math.degrees(self.joint_acceleration_rad_s2), 3
            ),
            "movel_available": movel_available,
            "movel_motion_profile_id": "linear_slow" if movel_available else None,
            "maximum_movel_translation_mm": MAXIMUM_MOVEL_TRANSLATION_MM,
            "maximum_movel_rotation_deg": MAXIMUM_MOVEL_ROTATION_DEG,
            "maximum_tcp_base_radius_mm": MAXIMUM_TCP_BASE_RADIUS_MM,
            "tcp_pose_reference": "DR_BASE",
            "tcp_pose_units": ["mm", "mm", "mm", "deg", "deg", "deg"],
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._enabled and self._robot is not None:
                try:
                    self._joint_positions_rad = self._robot.get_joint_positions_rad()
                    self._tcp_pose_base_mm_zyz_deg = (
                        self._robot.get_tcp_pose_base_mm_zyz_deg()
                    )
                    self._last_error = None
                except Exception as exc:
                    self._last_error = str(exc)
            return {
                "enabled": self._enabled,
                "operator_id": self._operator_id,
                "enabled_at_ns": self._enabled_at_ns,
                "last_move_at_ns": self._last_move_at_ns,
                "last_error": self._last_error,
                "adapter_name": self._robot.adapter_name if self._robot else None,
                "joint_positions_deg": [
                    round(math.degrees(value), 3) for value in self._joint_positions_rad
                ],
                "tcp_pose_base_mm_zyz_deg": [
                    round(value, 3) for value in self._tcp_pose_base_mm_zyz_deg
                ],
                "capabilities": self._capabilities(),
            }

    def enable(
        self,
        *,
        operator_id: str,
        workspace_cleared: bool,
        estop_ready: bool,
        acknowledge_direct_motion: bool,
    ) -> dict[str, Any]:
        if not operator_id.strip():
            raise ValueError("operator_id is required")
        if not all((workspace_cleared, estop_ready, acknowledge_direct_motion)):
            raise ValueError("all jog safety acknowledgements are required")
        with self._lock:
            if self.mode == "hardware" and not self.hardware_authorized:
                failed = ", ".join(
                    name for name, passed in self.gate_summary.items() if not passed
                )
                raise HardwareExecutionDenied(
                    "web jog hardware gates are closed" + (f": {failed}" if failed else "")
                )
            if self._enabled:
                if self._operator_id != operator_id.strip():
                    raise ValueError("jog is already enabled by another operator")
                return self.status()
            robot = self._robot_factory()
            try:
                robot.connect()
                current = robot.get_joint_positions_rad()
                self._validate_joint_vector(current)
                current_tcp = robot.get_tcp_pose_base_mm_zyz_deg()
                self._validate_cartesian_pose(current_tcp)
            except Exception:
                robot.disconnect()
                raise
            self._robot = robot
            self._joint_positions_rad = current
            self._tcp_pose_base_mm_zyz_deg = current_tcp
            self._operator_id = operator_id.strip()
            self._enabled_at_ns = time.time_ns()
            self._last_error = None
            self._enabled = True
            return self.status()

    def move_joint(self, *, joint_index: int, delta_deg: float) -> dict[str, Any]:
        if joint_index < 1 or joint_index > 6:
            raise ValueError("joint_index must be between 1 and 6")
        if not math.isfinite(delta_deg) or delta_deg == 0.0:
            raise ValueError("jog delta must be a finite non-zero value")
        if abs(delta_deg) > self.maximum_step_deg:
            raise ValueError(
                f"jog delta cannot exceed {self.maximum_step_deg:g} degrees per request"
            )
        with self._lock:
            if not self._enabled or self._robot is None:
                raise ValueError("enable jog and acknowledge safety before moving")
            current = self._robot.get_joint_positions_rad()
            self._validate_joint_vector(current)
            target = list(current)
            target[joint_index - 1] += math.radians(delta_deg)
            target_vector = cast(JointVector, tuple(target))
            self._validate_joint_limits(target_vector)
            self._robot.move_joints(
                target_vector,
                velocity_rad_s=self.joint_velocity_rad_s,
                acceleration_rad_s2=self.joint_acceleration_rad_s2,
            )
            self._joint_positions_rad = self._robot.get_joint_positions_rad()
            self._last_move_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def move_to_joint_positions(
        self, *, target_joint_positions_deg: tuple[float, ...]
    ) -> dict[str, Any]:
        """Execute one MoveJ for a complete operator-entered six-axis target."""

        if len(target_joint_positions_deg) != 6 or any(
            not math.isfinite(value) for value in target_joint_positions_deg
        ):
            raise ValueError("movej target must contain six finite joint angles")
        target_vector = cast(
            JointVector,
            tuple(math.radians(value) for value in target_joint_positions_deg),
        )
        self._validate_joint_limits(target_vector)
        with self._lock:
            if not self._enabled or self._robot is None:
                raise ValueError("enable jog and acknowledge safety before moving")
            current = self._robot.get_joint_positions_rad()
            self._validate_joint_vector(current)
            self._robot.move_joints(
                target_vector,
                velocity_rad_s=self.joint_velocity_rad_s,
                acceleration_rad_s2=self.joint_acceleration_rad_s2,
            )
            self._joint_positions_rad = self._robot.get_joint_positions_rad()
            self._last_move_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def move_to_cartesian_pose(
        self, *, target_tcp_pose_base_mm_zyz_deg: tuple[float, ...]
    ) -> dict[str, Any]:
        """Execute one base-referenced MoveL for a complete Cartesian target."""

        target = _cartesian_pose_values(
            target_tcp_pose_base_mm_zyz_deg,
            label="movel target",
        )
        self._validate_cartesian_pose(target)
        with self._lock:
            if not self._enabled or self._robot is None:
                raise ValueError("enable jog and acknowledge safety before moving")
            profile = (
                self.linear_velocity_m_s,
                self.linear_acceleration_m_s2,
                self.angular_velocity_rad_s,
                self.angular_acceleration_rad_s2,
            )
            if any(value is None for value in profile):
                raise ValueError("jog MoveL profile is not configured")
            current = self._robot.get_tcp_pose_base_mm_zyz_deg()
            self._validate_cartesian_pose(current)
            self._validate_movel_delta(current, target)
            self._robot.move_linear(
                target,
                linear_velocity_m_s=cast(float, self.linear_velocity_m_s),
                linear_acceleration_m_s2=cast(float, self.linear_acceleration_m_s2),
                angular_velocity_rad_s=cast(float, self.angular_velocity_rad_s),
                angular_acceleration_rad_s2=cast(
                    float, self.angular_acceleration_rad_s2
                ),
            )
            self._tcp_pose_base_mm_zyz_deg = (
                self._robot.get_tcp_pose_base_mm_zyz_deg()
            )
            self._joint_positions_rad = self._robot.get_joint_positions_rad()
            self._last_move_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def stop(self, *, reason: str = "operator_request") -> dict[str, Any]:
        with self._lock:
            robot = self._robot
            self._enabled = False
            self._robot = None
            self._operator_id = None
            self._enabled_at_ns = None
            if robot is not None:
                try:
                    robot.stop(reason=reason)
                finally:
                    robot.disconnect()
            return self.status()

    def close(self) -> None:
        self.stop(reason="application_shutdown")

    @staticmethod
    def _validate_joint_vector(values: tuple[float, ...]) -> None:
        if len(values) != 6 or any(not math.isfinite(value) for value in values):
            raise ValueError("robot joint feedback must contain six finite angles")

    @staticmethod
    def _validate_joint_limits(values: JointVector) -> None:
        JogController._validate_joint_vector(values)
        for index, (value_rad, limits) in enumerate(
            zip(values, M0609_JOINT_LIMITS_DEG, strict=True), start=1
        ):
            value_deg = math.degrees(value_rad)
            if value_deg < limits[0] or value_deg > limits[1]:
                raise ValueError(
                    f"J{index} target {value_deg:.3f}° is outside the approved "
                    f"range {limits[0]:g}°..{limits[1]:g}°"
                )

    @staticmethod
    def _validate_cartesian_pose(values: CartesianPose) -> None:
        _cartesian_pose_values(values, label="TCP pose")
        radius_mm = math.sqrt(sum(value * value for value in values[:3]))
        if radius_mm > MAXIMUM_TCP_BASE_RADIUS_MM:
            raise ValueError(
                f"MoveL TCP target radius {radius_mm:.3f} mm exceeds the approved "
                f"{MAXIMUM_TCP_BASE_RADIUS_MM:g} mm base radius"
            )
        if any(abs(value) > 360.0 for value in values[3:]):
            raise ValueError("MoveL orientation must stay within +/-360 degrees")

    @staticmethod
    def _validate_movel_delta(current: CartesianPose, target: CartesianPose) -> None:
        translation_mm = math.sqrt(
            sum((target[index] - current[index]) ** 2 for index in range(3))
        )
        if translation_mm > MAXIMUM_MOVEL_TRANSLATION_MM:
            raise ValueError(
                f"MoveL translation {translation_mm:.3f} mm exceeds the approved "
                f"{MAXIMUM_MOVEL_TRANSLATION_MM:g} mm per request"
            )
        rotation_deltas = [
            abs((target[index] - current[index] + 180.0) % 360.0 - 180.0)
            for index in range(3, 6)
        ]
        if max(rotation_deltas) > MAXIMUM_MOVEL_ROTATION_DEG:
            raise ValueError(
                "MoveL orientation change exceeds the approved "
                f"{MAXIMUM_MOVEL_ROTATION_DEG:g} degrees per axis per request"
            )
