"""Fail-closed UI boundary for the fixed ArUco-plane +Z experiment.

The frozen plane uses metres and ``T_A_B`` means coordinates in B transformed
into A.  At the exact reference joint pose we bind the already-frozen plane to
the current robot base as ``T_base_plane = T_base_tcp @ T_tcp_plane``.  This
does not re-fit or re-align the ArUco frame.

Only two motions are exposed: the fixed M0609 reference joint pose and one
20 mm linear TCP move along plane +Z (toward the frozen reference camera and
away from the table).  Unsafe values are rejected and are never projected or
clamped into the workspace.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from aruco.object_width_workspace import (
    RuntimeWorkspace,
    build_runtime_workspace,
    load_reference,
    require_tcp_point_plane,
    save_runtime_npz,
)
from robot_skill_system.calibration.robot import DoosanHandEyeCalibrationRobot
from robot_skill_system.exceptions import HardwareExecutionDenied, NotConfiguredError
from robot_skill_system.jog.controller import M0609_JOINT_LIMITS_DEG

Matrix44 = NDArray[np.float64]
JointVector = tuple[float, float, float, float, float, float]
CartesianPose = tuple[float, float, float, float, float, float]

REFERENCE_JOINT_DEG: JointVector = (0.0, 0.0, 90.0, 0.0, 90.0, -90.0)
REFERENCE_JOINT_RAD: JointVector = cast(
    JointVector, tuple(math.radians(value) for value in REFERENCE_JOINT_DEG)
)
REFERENCE_JOINT_TOLERANCE_DEG = 0.5
PLANE_Z_TEST_DISTANCE_M = 0.020
MAXIMUM_BASE_RADIUS_M = 1.0
MAXIMUM_TCP_FEEDBACK_ERROR_M = 0.001
PATH_SAMPLE_STEP_M = 0.002


class ArucoExperimentRobot(Protocol):
    """Minimal robot boundary used by the supervised experiment."""

    adapter_name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions_rad(self) -> JointVector: ...

    def get_base_to_tcp_matrix(self) -> Matrix44: ...

    def get_tcp_pose_base_mm_zyz_deg(self) -> CartesianPose: ...

    def get_active_tcp_name(self) -> str: ...

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None: ...

    def solve_inverse_kinematics(self, target_pose: CartesianPose) -> JointVector: ...

    def move_linear(
        self,
        target_pose: CartesianPose,
        *,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None: ...

    def stop(self, *, reason: str) -> None: ...


def _rigid_matrix(value: object, *, label: str) -> Matrix44:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{label} rotation determinant must be +1")
    return np.asarray(matrix, dtype=np.float64)


def _pose_values(value: object, *, label: str) -> CartesianPose:
    # ROS 2 generated fixed-size float arrays are numpy.ndarray instances, which
    # are iterable but deliberately do not register as ``Sequence``.  Convert
    # through numpy so the controller accepts both ordinary Python sequences and
    # the Doosan service's ``float64[6]`` response without weakening the shape or
    # finite-value validation below.
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain six values")
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain six values") from exc
    if values.shape != (6,):
        raise ValueError(f"{label} must contain six values")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return cast(CartesianPose, result)


def _joint_values(value: object, *, label: str) -> JointVector:
    return _pose_values(value, label=label)


def _validate_joint_limits(values_rad: JointVector) -> None:
    for index, (value_rad, limits_deg) in enumerate(
        zip(values_rad, M0609_JOINT_LIMITS_DEG, strict=True), start=1
    ):
        value_deg = math.degrees(value_rad)
        if not limits_deg[0] <= value_deg <= limits_deg[1]:
            raise ValueError(
                f"IK J{index}={value_deg:.3f} deg is outside "
                f"{limits_deg[0]:g}..{limits_deg[1]:g} deg"
            )


def _within_reference_joint(values_rad: JointVector) -> bool:
    return all(
        abs(math.degrees(actual - expected)) <= REFERENCE_JOINT_TOLERANCE_DEG
        for actual, expected in zip(values_rad, REFERENCE_JOINT_RAD, strict=True)
    )


def _point_from_transform(transform: Matrix44) -> NDArray[np.float64]:
    return np.asarray(transform[:3, 3], dtype=np.float64)


def _transform_point(transform: Matrix44, point_xyz_m: NDArray[np.float64]) -> NDArray[np.float64]:
    homogeneous = np.concatenate((np.asarray(point_xyz_m, dtype=np.float64), [1.0]))
    return np.asarray((transform @ homogeneous)[:3], dtype=np.float64)


def _require_base_radius(point_base_m: NDArray[np.float64]) -> None:
    radius_m = float(np.linalg.norm(point_base_m))
    if not math.isfinite(radius_m) or radius_m > MAXIMUM_BASE_RADIUS_M:
        raise ValueError(
            f"TCP base radius {radius_m:.4f} m exceeds the approved "
            f"{MAXIMUM_BASE_RADIUS_M:.1f} m experiment limit"
        )


class MockArucoExperimentRobot:
    """Deterministic CPU-only robot; no ROS, hardware, or network access."""

    adapter_name = "mock_m0609_aruco_experiment"

    def __init__(self, *, active_tcp_name: str = "GripperDA_v1") -> None:
        self.connected = False
        self.active_tcp_name = active_tcp_name
        self.joints_rad: JointVector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.base_to_tcp = np.eye(4, dtype=np.float64)
        self.base_to_tcp[:3, 3] = [0.40, 0.0, 0.50]

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("mock ArUco experiment robot is not connected")

    def get_joint_positions_rad(self) -> JointVector:
        self._require_connected()
        return self.joints_rad

    def get_base_to_tcp_matrix(self) -> Matrix44:
        self._require_connected()
        return self.base_to_tcp.copy()

    def get_tcp_pose_base_mm_zyz_deg(self) -> CartesianPose:
        self._require_connected()
        xyz_mm = self.base_to_tcp[:3, 3] * 1000.0
        return (float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2]), 0.0, 0.0, 0.0)

    def get_active_tcp_name(self) -> str:
        self._require_connected()
        return self.active_tcp_name

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None:
        self._require_connected()
        if velocity_rad_s <= 0.0 or acceleration_rad_s2 <= 0.0:
            raise ValueError("mock joint profile must be positive")
        self.joints_rad = target_rad

    def solve_inverse_kinematics(self, target_pose: CartesianPose) -> JointVector:
        self._require_connected()
        _pose_values(target_pose, label="mock IK target")
        return self.joints_rad

    def move_linear(
        self,
        target_pose: CartesianPose,
        *,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None:
        self._require_connected()
        if min(
            linear_velocity_m_s,
            linear_acceleration_m_s2,
            angular_velocity_rad_s,
            angular_acceleration_rad_s2,
        ) <= 0.0:
            raise ValueError("mock Cartesian profile must be positive")
        pose = _pose_values(target_pose, label="mock MoveL target")
        self.base_to_tcp[:3, 3] = np.asarray(pose[:3], dtype=np.float64) / 1000.0

    def stop(self, *, reason: str) -> None:
        del reason
        self._require_connected()


class DoosanArucoExperimentRobot(DoosanHandEyeCalibrationRobot):
    """Lazily add audited MoveL and IK services to the existing Doosan adapter."""

    adapter_name = "doosan_m0609_aruco_experiment"

    def __init__(
        self, *, expected_tcp_name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._expected_tcp_name = (expected_tcp_name or "").strip()
        self._move_line_client: Any | None = None
        self._move_line_type: Any | None = None
        self._ikin_client: Any | None = None
        self._ikin_type: Any | None = None

    def get_active_tcp_name(self) -> str:
        """Read TCP when the driver exposes it, otherwise reuse configured TCP.

        The deployed M0609 driver returns an empty string from ``get_tcp()``
        even while ``get_current_posx()`` reports the configured active TCP.
        ArUco is already tied to one configured TCP, so an empty value is not a
        name mismatch; use that explicit configured name.  A non-empty driver
        answer still remains authoritative and is checked by the controller.
        """

        try:
            return super().get_active_tcp_name()
        except RuntimeError as exc:
            if (
                str(exc) == "Doosan did not return a valid active TCP name"
                and self._expected_tcp_name
            ):
                return self._expected_tcp_name
            raise

    def connect(self) -> None:
        super().connect()
        node = self._node
        if node is None:
            raise NotConfiguredError("Doosan ArUco experiment node was not created")
        try:
            import importlib

            services = importlib.import_module("dsr_msgs2.srv")
            move_line_type = getattr(services, "MoveLine", None)
            ikin_type = getattr(services, "Ikin", None)
            if move_line_type is None or ikin_type is None:
                raise NotConfiguredError("dsr_msgs2 lacks MoveLine or Ikin")
            move_line_client = node.create_client(move_line_type, "motion/move_line")
            ikin_client = node.create_client(ikin_type, "motion/ikin")
            if not move_line_client.wait_for_service(timeout_sec=2.0):
                raise NotConfiguredError("Doosan motion/move_line service is unavailable")
            if not ikin_client.wait_for_service(timeout_sec=2.0):
                raise NotConfiguredError("Doosan motion/ikin service is unavailable")
            self._move_line_type = move_line_type
            self._move_line_client = move_line_client
            self._ikin_type = ikin_type
            self._ikin_client = ikin_client
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        node = self._node
        if node is not None:
            for client in (self._move_line_client, self._ikin_client):
                if client is not None:
                    with suppress(Exception):
                        node.destroy_client(client)
        self._move_line_client = None
        self._move_line_type = None
        self._ikin_client = None
        self._ikin_type = None
        super().disconnect()

    def get_tcp_pose_base_mm_zyz_deg(self) -> CartesianPose:
        function = self._required(self._get_current_posx, "get_current_posx")
        value = function()
        candidate = value[0] if isinstance(value, tuple) and len(value) == 2 else value
        return _pose_values(candidate, label="Doosan active TCP pose")

    def _call_service(self, client: Any, request: Any, *, label: str, timeout_s: float) -> Any:
        if self._rclpy is None or self._node is None or client is None:
            raise NotConfiguredError(f"Doosan {label} service is not configured")
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
        if not future.done():
            raise RuntimeError(f"Doosan {label} did not respond within {timeout_s:g} seconds")
        result = future.result()
        if result is None:
            raise RuntimeError(f"Doosan {label} returned no response")
        return result

    def solve_inverse_kinematics(self, target_pose: CartesianPose) -> JointVector:
        self._require_standby()
        if self._ikin_type is None or self._ikin_client is None:
            raise NotConfiguredError("Doosan Ikin service is not configured")
        pose = _pose_values(target_pose, label="Doosan IK target")
        current = self.get_joint_positions_rad()
        candidates: list[JointVector] = []
        for solution_space in range(8):
            request = self._ikin_type.Request()
            request.pos = list(pose)
            request.sol_space = solution_space
            request.ref = 0
            response = self._call_service(
                self._ikin_client, request, label="Ikin", timeout_s=3.0
            )
            if response.success is not True:
                continue
            candidate = cast(
                JointVector,
                tuple(math.radians(value) for value in _joint_values(
                    response.conv_posj, label="Doosan IK response"
                )),
            )
            try:
                _validate_joint_limits(candidate)
            except ValueError:
                continue
            candidates.append(candidate)
        if not candidates:
            raise ValueError("Doosan IK found no in-limit solution for the +Z target")
        return min(
            candidates,
            key=lambda values: sum(
                abs(math.atan2(math.sin(a - b), math.cos(a - b)))
                for a, b in zip(values, current, strict=True)
            ),
        )

    def move_linear(
        self,
        target_pose: CartesianPose,
        *,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None:
        if self._move_line_type is None or self._move_line_client is None:
            raise NotConfiguredError("Doosan MoveLine service is not configured")
        if min(
            linear_velocity_m_s,
            linear_acceleration_m_s2,
            angular_velocity_rad_s,
            angular_acceleration_rad_s2,
        ) <= 0.0:
            raise ValueError("MoveL profile limits must be positive")
        self._require_standby()
        request = self._move_line_type.Request()
        request.pos = list(_pose_values(target_pose, label="Doosan MoveL target"))
        request.vel = [
            linear_velocity_m_s * 1000.0,
            math.degrees(angular_velocity_rad_s),
        ]
        request.acc = [
            linear_acceleration_m_s2 * 1000.0,
            math.degrees(angular_acceleration_rad_s2),
        ]
        request.time = 0.0
        request.radius = 0.0
        request.ref = 0
        request.mode = 0
        request.blend_type = 0
        request.sync_type = 0
        response = self._call_service(
            self._move_line_client, request, label="MoveLine", timeout_s=30.0
        )
        if response.success is not True:
            raise RuntimeError("Doosan MoveLine rejected the +Z target")
        self._require_standby()


class ArucoExperimentController:
    """Own one acknowledged session for a fixed reference and +Z test."""

    def __init__(
        self,
        *,
        robot_factory: Callable[[], ArucoExperimentRobot],
        mode: str,
        hardware_authorized: bool,
        gate_summary: dict[str, bool],
        reference_npz: Path,
        runtime_npz: Path,
        expected_tcp_name: str,
        joint_velocity_rad_s: float,
        joint_acceleration_rad_s2: float,
        linear_velocity_m_s: float,
        linear_acceleration_m_s2: float,
        angular_velocity_rad_s: float,
        angular_acceleration_rad_s2: float,
    ) -> None:
        if mode not in {"mock", "hardware"}:
            raise ValueError("ArUco experiment mode must be mock or hardware")
        if not expected_tcp_name.strip():
            raise ValueError("ArUco experiment expected TCP name is required")
        limits = (
            joint_velocity_rad_s,
            joint_acceleration_rad_s2,
            linear_velocity_m_s,
            linear_acceleration_m_s2,
            angular_velocity_rad_s,
            angular_acceleration_rad_s2,
        )
        if min(limits) <= 0.0 or not all(math.isfinite(value) for value in limits):
            raise ValueError("ArUco experiment motion profile limits must be finite and positive")
        self._robot_factory = robot_factory
        self.mode = mode
        self.hardware_authorized = hardware_authorized
        self.gate_summary = dict(gate_summary)
        self.reference_npz = reference_npz.expanduser().resolve()
        self.runtime_npz = runtime_npz.expanduser().resolve()
        self.expected_tcp_name = expected_tcp_name.strip()
        self.joint_velocity_rad_s = joint_velocity_rad_s
        self.joint_acceleration_rad_s2 = joint_acceleration_rad_s2
        self.linear_velocity_m_s = linear_velocity_m_s
        self.linear_acceleration_m_s2 = linear_acceleration_m_s2
        self.angular_velocity_rad_s = angular_velocity_rad_s
        self.angular_acceleration_rad_s2 = angular_acceleration_rad_s2
        self._lock = threading.RLock()
        self._robot: ArucoExperimentRobot | None = None
        self._enabled = False
        self._operator_id: str | None = None
        self._enabled_at_ns: int | None = None
        self._reference_captured = False
        self._z_test_completed = False
        self._base_to_plane: Matrix44 | None = None
        self._base_to_tcp_reference: Matrix44 | None = None
        self._runtime: RuntimeWorkspace | None = None
        self._runtime_width_model: str | None = None
        self._last_action: str | None = None
        self._last_action_at_ns: int | None = None
        self._last_error: str | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def _reference_summary(self) -> dict[str, Any]:
        reference = load_reference(self.reference_npz)
        camera_xyz = np.asarray(reference["camera_reference_plane_xyz_m"], dtype=np.float64)
        tcp_xyz = np.asarray(reference["tcp_reference_plane_xyz_m"], dtype=np.float64)
        polygon = np.asarray(reference["workspace_xy_m"], dtype=np.float64)
        return {
            "path": str(self.reference_npz),
            "checksum_sha256": str(reference["_reference_sha256"]),
            "frame_name": str(reference["frame_name"]),
            "reference_joint_deg": list(REFERENCE_JOINT_DEG),
            "reference_tcp_plane_xyz_m": tcp_xyz.tolist(),
            "reference_camera_plane_xyz_m": camera_xyz.tolist(),
            "plane_to_camera_z_range_m": [0.0, float(camera_xyz[2])],
            "workspace_vertex_count": int(len(polygon)),
        }

    def _capabilities(self) -> dict[str, Any]:
        failed_gates = [name for name, passed in self.gate_summary.items() if not passed]
        try:
            reference = self._reference_summary()
            reference_error = None
        except Exception as exc:
            reference = None
            reference_error = str(exc)
        return {
            "mode": self.mode,
            "hardware_authorized": self.hardware_authorized,
            "gate_summary": dict(self.gate_summary),
            "failed_gates": failed_gates,
            "reference": reference,
            "reference_error": reference_error,
            "expected_tcp_name": self.expected_tcp_name,
            "reference_joint_tolerance_deg": REFERENCE_JOINT_TOLERANCE_DEG,
            "plane_z_test_distance_m": PLANE_Z_TEST_DISTANCE_M,
            "maximum_base_radius_m": MAXIMUM_BASE_RADIUS_M,
            "unsafe_targets_are_clamped": False,
            "obstacle_policy": "operator-cleared cell; base radius <= 1 m",
            "joint_profile_id": "joint_safe",
            "linear_profile_id": "linear_slow",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            joints_deg: list[float] | None = None
            tcp_base_xyz_m: list[float] | None = None
            active_tcp_name: str | None = None
            if self._enabled and self._robot is not None:
                try:
                    joints = self._robot.get_joint_positions_rad()
                    joints_deg = [round(math.degrees(value), 4) for value in joints]
                    tcp = _rigid_matrix(
                        self._robot.get_base_to_tcp_matrix(), label="T_base_tcp feedback"
                    )
                    tcp_base_xyz_m = _point_from_transform(tcp).tolist()
                    active_tcp_name = self._robot.get_active_tcp_name()
                except Exception as exc:
                    self._last_error = str(exc)
            runtime = self._runtime
            return {
                "enabled": self._enabled,
                "operator_id": self._operator_id,
                "enabled_at_ns": self._enabled_at_ns,
                "reference_captured": self._reference_captured,
                "z_test_completed": self._z_test_completed,
                "active_tcp_name": active_tcp_name,
                "joint_positions_deg": joints_deg,
                "tcp_base_xyz_m": tcp_base_xyz_m,
                "adapter_name": self._robot.adapter_name if self._robot else None,
                "last_action": self._last_action,
                "last_action_at_ns": self._last_action_at_ns,
                "last_error": self._last_error,
                "runtime_workspace": (
                    {
                        "path": str(self.runtime_npz),
                        "object_width_mm": runtime.object_width_mm,
                        "width_model": runtime.width_model,
                        "theta_deg": runtime.theta_deg,
                        "opening_offset_m": runtime.opening_offset_m,
                        "z_min_plane_m": runtime.z_min_plane_m,
                        "z_max_plane_m": runtime.z_max_plane_m,
                    }
                    if runtime is not None
                    else None
                ),
                "capabilities": self._capabilities(),
            }

    def acquire_runtime_session(self) -> tuple[ArucoExperimentRobot, Matrix44]:
        """Lease the connected robot and captured base/plane transform for one skill run."""

        with self._lock:
            robot = self._require_enabled_robot()
            if not self._reference_captured or self._base_to_plane is None:
                raise ValueError("ArUco 기준 자세(1번)를 먼저 실행해 좌표계를 고정하세요")
            if self.mode != "hardware" or not self.hardware_authorized:
                raise ValueError("현재 ArUco 세션은 실제 하드웨어 세션이 아닙니다")
            return robot, self._base_to_plane.copy()

    def enable(
        self,
        *,
        operator_id: str,
        workspace_cleared: bool,
        estop_ready: bool,
        acknowledge_direct_motion: bool,
        object_width_mm: float,
        width_model: str,
    ) -> dict[str, Any]:
        if not operator_id.strip():
            raise ValueError("operator_id is required")
        if not all((workspace_cleared, estop_ready, acknowledge_direct_motion)):
            raise ValueError("all ArUco experiment safety acknowledgements are required")
        with self._lock:
            if self.mode == "hardware" and not self.hardware_authorized:
                failed = ", ".join(
                    name for name, passed in self.gate_summary.items() if not passed
                )
                raise HardwareExecutionDenied(
                    "ArUco experiment hardware gates are closed"
                    + (f": {failed}" if failed else "")
                )
            reference = load_reference(self.reference_npz)
            runtime = build_runtime_workspace(reference, object_width_mm, width_model)
            save_runtime_npz(self.runtime_npz, reference, runtime)
            if self._enabled:
                if self._operator_id != operator_id.strip():
                    raise ValueError("ArUco experiment is enabled by another operator")
                self._runtime = runtime
                self._runtime_width_model = width_model
                return self.status()
            robot = self._robot_factory()
            try:
                robot.connect()
                joints = _joint_values(
                    robot.get_joint_positions_rad(), label="robot joint feedback"
                )
                _validate_joint_limits(joints)
                tcp = _rigid_matrix(
                    robot.get_base_to_tcp_matrix(), label="T_base_tcp feedback"
                )
                _require_base_radius(_point_from_transform(tcp))
                active_tcp_name = robot.get_active_tcp_name()
                if active_tcp_name != self.expected_tcp_name:
                    raise ValueError(
                        f"active TCP {active_tcp_name!r} does not match frozen "
                        f"workspace TCP {self.expected_tcp_name!r}"
                    )
            except Exception:
                robot.disconnect()
                raise
            self._robot = robot
            self._enabled = True
            self._operator_id = operator_id.strip()
            self._enabled_at_ns = time.time_ns()
            self._runtime = runtime
            self._runtime_width_model = width_model
            self._reference_captured = False
            self._z_test_completed = False
            self._base_to_plane = None
            self._base_to_tcp_reference = None
            self._last_action = "enabled"
            self._last_action_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def record_failure(self, action: str, exc: Exception) -> None:
        """Keep a motion/enable failure visible across subsequent status polling."""

        with self._lock:
            self._last_action = f"{action}_failed"
            self._last_action_at_ns = time.time_ns()
            self._last_error = f"{type(exc).__name__}: {exc}"

    def move_to_reference(self) -> dict[str, Any]:
        with self._lock:
            robot = self._require_enabled_robot()
            robot.move_joints(
                REFERENCE_JOINT_RAD,
                velocity_rad_s=self.joint_velocity_rad_s,
                acceleration_rad_s2=self.joint_acceleration_rad_s2,
            )
            joints = _joint_values(robot.get_joint_positions_rad(), label="joint feedback")
            if not _within_reference_joint(joints):
                raise RuntimeError("robot did not settle at the fixed reference joint pose")
            active_tcp_name = robot.get_active_tcp_name()
            if active_tcp_name != self.expected_tcp_name:
                raise ValueError("active TCP changed while moving to the reference pose")
            base_to_tcp = _rigid_matrix(
                robot.get_base_to_tcp_matrix(), label="reference T_base_tcp"
            )
            _require_base_radius(_point_from_transform(base_to_tcp))
            reference = load_reference(self.reference_npz)
            tcp_to_plane = _rigid_matrix(reference["T_tcp_plane"], label="T_tcp_plane")
            base_to_plane = _rigid_matrix(
                base_to_tcp @ tcp_to_plane, label="derived T_base_plane"
            )
            reference_tcp_plane = np.asarray(
                reference["tcp_reference_plane_xyz_m"], dtype=np.float64
            )
            mapped_reference_tcp = _transform_point(base_to_plane, reference_tcp_plane)
            if not np.allclose(
                mapped_reference_tcp,
                _point_from_transform(base_to_tcp),
                atol=1e-6,
                rtol=0.0,
            ):
                raise RuntimeError("reference TCP and frozen plane transform chain disagree")
            if self._runtime is None:
                raise RuntimeError("generate a width-based runtime workspace first")
            require_tcp_point_plane(reference, self._runtime, reference_tcp_plane)
            self._base_to_tcp_reference = base_to_tcp
            self._base_to_plane = base_to_plane
            self._reference_captured = True
            self._z_test_completed = False
            self._last_action = "moved_to_reference"
            self._last_action_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def move_plane_z_test(self) -> dict[str, Any]:
        with self._lock:
            robot = self._require_enabled_robot()
            if not self._reference_captured:
                raise ValueError("move to and capture the fixed reference pose first")
            if self._z_test_completed:
                raise ValueError("the +Z 20 mm test already completed; return to reference first")
            if self._runtime is None or self._base_to_plane is None:
                raise RuntimeError("runtime workspace or base-to-plane binding is missing")
            current = _rigid_matrix(robot.get_base_to_tcp_matrix(), label="current T_base_tcp")
            reference_tcp = self._base_to_tcp_reference
            if reference_tcp is None:
                raise RuntimeError("reference TCP capture is missing")
            reference_feedback_error_m = float(
                np.linalg.norm(
                    _point_from_transform(current) - _point_from_transform(reference_tcp)
                )
            )
            if reference_feedback_error_m > MAXIMUM_TCP_FEEDBACK_ERROR_M:
                raise ValueError("TCP is no longer at the captured reference position")
            reference = load_reference(self.reference_npz)
            reference_tcp_plane = np.asarray(
                reference["tcp_reference_plane_xyz_m"], dtype=np.float64
            )
            target_plane = reference_tcp_plane + np.asarray(
                [0.0, 0.0, PLANE_Z_TEST_DISTANCE_M], dtype=np.float64
            )
            sample_count = max(
                2, math.ceil(PLANE_Z_TEST_DISTANCE_M / PATH_SAMPLE_STEP_M) + 1
            )
            for fraction in np.linspace(0.0, 1.0, sample_count):
                point = reference_tcp_plane + fraction * (target_plane - reference_tcp_plane)
                require_tcp_point_plane(reference, self._runtime, point)
            target_base = _transform_point(self._base_to_plane, target_plane)
            _require_base_radius(target_base)
            current_pose = robot.get_tcp_pose_base_mm_zyz_deg()
            target_pose: CartesianPose = (
                float(target_base[0] * 1000.0),
                float(target_base[1] * 1000.0),
                float(target_base[2] * 1000.0),
                current_pose[3],
                current_pose[4],
                current_pose[5],
            )
            ik_solution = robot.solve_inverse_kinematics(target_pose)
            _validate_joint_limits(ik_solution)
            robot.move_linear(
                target_pose,
                linear_velocity_m_s=self.linear_velocity_m_s,
                linear_acceleration_m_s2=self.linear_acceleration_m_s2,
                angular_velocity_rad_s=self.angular_velocity_rad_s,
                angular_acceleration_rad_s2=self.angular_acceleration_rad_s2,
            )
            feedback = _rigid_matrix(
                robot.get_base_to_tcp_matrix(), label="post-MoveL T_base_tcp"
            )
            feedback_error_m = float(np.linalg.norm(_point_from_transform(feedback) - target_base))
            if feedback_error_m > MAXIMUM_TCP_FEEDBACK_ERROR_M:
                with suppress(Exception):
                    robot.stop(reason="aruco_z_feedback_error")
                raise RuntimeError(
                    f"TCP +Z feedback error {feedback_error_m * 1000.0:.3f} mm exceeds "
                    f"{MAXIMUM_TCP_FEEDBACK_ERROR_M * 1000.0:.1f} mm"
                )
            self._z_test_completed = True
            self._last_action = "plane_z_plus_20mm_completed"
            self._last_action_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def stop(self, *, reason: str = "operator_request") -> dict[str, Any]:
        with self._lock:
            robot = self._robot
            self._enabled = False
            self._robot = None
            self._operator_id = None
            self._enabled_at_ns = None
            self._reference_captured = False
            self._base_to_plane = None
            self._base_to_tcp_reference = None
            if robot is not None:
                try:
                    robot.stop(reason=reason)
                finally:
                    robot.disconnect()
            self._last_action = "stopped"
            self._last_action_at_ns = time.time_ns()
            self._last_error = None
            return self.status()

    def close(self) -> None:
        self.stop(reason="application_shutdown")

    def _require_enabled_robot(self) -> ArucoExperimentRobot:
        if not self._enabled or self._robot is None:
            raise ValueError("enable the ArUco experiment and acknowledge safety first")
        return self._robot


__all__ = [
    "ArucoExperimentController",
    "ArucoExperimentRobot",
    "DoosanArucoExperimentRobot",
    "MAXIMUM_BASE_RADIUS_M",
    "MockArucoExperimentRobot",
    "PLANE_Z_TEST_DISTANCE_M",
    "REFERENCE_JOINT_DEG",
]
