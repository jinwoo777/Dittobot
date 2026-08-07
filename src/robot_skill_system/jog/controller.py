"""Discrete, fail-closed joint jog controller for the Doosan M0609."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, cast

from robot_skill_system.calibration.robot import DoosanHandEyeCalibrationRobot
from robot_skill_system.exceptions import HardwareExecutionDenied

JointVector = tuple[float, float, float, float, float, float]

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


class JogRobot(Protocol):
    adapter_name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions_rad(self) -> JointVector: ...

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None: ...

    def stop(self, *, reason: str) -> None: ...


class MockJogRobot:
    """In-memory robot used unless every real-hardware gate is enabled."""

    adapter_name = "mock_m0609_web_jog"

    def __init__(self) -> None:
        self.connected = False
        self.joint_positions_rad: JointVector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_positions_rad(self) -> JointVector:
        if not self.connected:
            raise RuntimeError("mock jog robot is not connected")
        return self.joint_positions_rad

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

    def stop(self, *, reason: str) -> None:
        _ = reason


class DoosanJogRobot(DoosanHandEyeCalibrationRobot):
    """Web-jog label for the fixed, lazily imported Doosan joint adapter."""

    adapter_name = "doosan_m0609_web_jog"


class JogController:
    """Own one acknowledged jog session and execute validated joint targets."""

    def __init__(
        self,
        *,
        robot_factory: Callable[[], JogRobot],
        mode: str,
        hardware_authorized: bool,
        gate_summary: dict[str, bool],
        joint_velocity_rad_s: float,
        joint_acceleration_rad_s2: float,
        maximum_step_deg: float = MAXIMUM_JOG_STEP_DEG,
    ) -> None:
        if mode not in {"mock", "hardware"}:
            raise ValueError("jog mode must be mock or hardware")
        if joint_velocity_rad_s <= 0.0 or joint_acceleration_rad_s2 <= 0.0:
            raise ValueError("jog motion profile limits must be positive")
        if maximum_step_deg <= 0.0 or maximum_step_deg > MAXIMUM_JOG_STEP_DEG:
            raise ValueError("maximum jog step must be within (0, 5] degrees")
        self._robot_factory = robot_factory
        self.mode = mode
        self.hardware_authorized = hardware_authorized
        self.gate_summary = dict(gate_summary)
        self.joint_velocity_rad_s = joint_velocity_rad_s
        self.joint_acceleration_rad_s2 = joint_acceleration_rad_s2
        self.maximum_step_deg = maximum_step_deg
        self._lock = threading.RLock()
        self._robot: JogRobot | None = None
        self._enabled = False
        self._operator_id: str | None = None
        self._enabled_at_ns: int | None = None
        self._last_move_at_ns: int | None = None
        self._last_error: str | None = None
        self._joint_positions_rad: JointVector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def _capabilities(self) -> dict[str, Any]:
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
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._enabled and self._robot is not None:
                try:
                    self._joint_positions_rad = self._robot.get_joint_positions_rad()
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
            except Exception:
                robot.disconnect()
                raise
            self._robot = robot
            self._joint_positions_rad = current
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
