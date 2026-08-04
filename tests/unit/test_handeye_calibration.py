from __future__ import annotations

import math
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import robot_skill_system.calibration.robot as robot_module
from robot_skill_system.calibration.controller import HandEyeCalibrationController
from robot_skill_system.calibration.handeye import (
    CheckerboardObservation,
    EyeInHandCalibrationConfig,
    solve_eye_in_hand,
)
from robot_skill_system.calibration.robot import (
    DoosanHandEyeCalibrationRobot,
    JointVector,
    approved_eye_in_hand_waypoints,
)
from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.storage.artifact_store import LocalArtifactStore


def _rotation_xyz(x_rad: float, y_rad: float, z_rad: float) -> np.ndarray:
    cx, sx = math.cos(x_rad), math.sin(x_rad)
    cy, sy = math.cos(y_rad), math.sin(y_rad)
    cz, sz = math.cos(z_rad), math.sin(z_rad)
    rotation_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def _transform(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _known_flange_to_camera() -> np.ndarray:
    return _transform(
        _rotation_xyz(math.radians(4), math.radians(-7), math.radians(11)),
        (0.075, -0.012, 0.045),
    )


def _synthetic_observations(count: int = 21) -> list[CheckerboardObservation]:
    flange_to_camera = _known_flange_to_camera()
    base_to_board = _transform(_rotation_xyz(0.0, 0.0, 0.1), (0.55, 0.0, 0.18))
    observations: list[CheckerboardObservation] = []
    for index in range(count):
        phase = 2.0 * math.pi * index / count
        base_to_flange = _transform(
            _rotation_xyz(
                math.radians(18.0) * math.sin(phase),
                math.radians(16.0) * math.cos(phase),
                math.radians(12.0) * math.sin(2.0 * phase),
            ),
            (
                0.32 + 0.07 * math.sin(phase),
                -0.04 + 0.06 * math.cos(phase),
                0.28 + 0.05 * math.sin(2.0 * phase),
            ),
        )
        camera_to_board = np.linalg.inv(base_to_flange @ flange_to_camera) @ base_to_board
        observations.append(
            CheckerboardObservation(
                waypoint_id=f"pose_{index:02d}",
                frame_number=index,
                captured_at_ns=index + 1,
                joint_positions_rad=(0.0, 0.0, 1.57, 0.0, 1.57, 0.0),
                base_to_flange=base_to_flange,
                camera_to_board=camera_to_board,
                reprojection_rms_px=0.1,
            )
        )
    return observations


def test_eye_in_hand_solver_recovers_flange_camera_and_passes_metrics() -> None:
    result = solve_eye_in_hand(_synthetic_observations(), EyeInHandCalibrationConfig())

    assert result.passed is True
    assert np.allclose(result.flange_to_camera, _known_flange_to_camera(), atol=1.0e-7)
    assert result.translation_rms_m < 1.0e-8
    assert result.rotation_rms_deg < 1.0e-5
    assert result.translation_span_m > 0.05
    assert result.rotation_span_deg > 15.0


def test_approved_plan_keeps_joints_one_and_two_locked() -> None:
    waypoints = approved_eye_in_hand_waypoints()
    reference = waypoints[0].joint_positions_rad

    assert len(waypoints) == 21
    assert all(item.joint_positions_rad[:2] == (0.0, 0.0) for item in waypoints)
    assert waypoints[0].joint_positions_rad == (
        0.0,
        0.0,
        math.pi / 2.0,
        0.0,
        math.pi / 2.0,
        0.0,
    )
    assert max(
        abs(value - reference[joint_index])
        for waypoint in waypoints
        for joint_index, value in enumerate(waypoint.joint_positions_rad)
    ) <= math.radians(5.0) + 1.0e-12
    assert max(
        abs(current_value - previous.joint_positions_rad[joint_index])
        for previous, current in zip(waypoints[:-1], waypoints[1:], strict=True)
        for joint_index, current_value in enumerate(current.joint_positions_rad)
    ) <= math.radians(5.0) + 1.0e-12


def test_doosan_node_failure_reports_empty_cyclonedds_interface_and_resets_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"initialized": False, "shutdown": False}
    fake_rclpy = ModuleType("rclpy")
    fake_dr_init = ModuleType("DR_init")

    def ok() -> bool:
        return state["initialized"] and not state["shutdown"]

    def init(*, args: object) -> None:
        assert args is None
        state["initialized"] = True

    def create_node(_name: str, *, namespace: str) -> object:
        assert namespace == "dsr01"
        raise RuntimeError("rcl node's rmw handle is invalid")

    def shutdown() -> None:
        state["shutdown"] = True

    fake_rclpy.ok = ok  # type: ignore[attr-defined]
    fake_rclpy.init = init  # type: ignore[attr-defined]
    fake_rclpy.create_node = create_node  # type: ignore[attr-defined]
    fake_rclpy.shutdown = shutdown  # type: ignore[attr-defined]

    def import_module(name: str) -> Any:
        if name == "rclpy":
            return fake_rclpy
        if name == "DR_init":
            return fake_dr_init
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(robot_module.importlib, "import_module", import_module)
    monkeypatch.setenv(
        "CYCLONEDDS_URI",
        '<CycloneDDS><NetworkInterface name=""/></CycloneDDS>',
    )
    robot = DoosanHandEyeCalibrationRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
    )

    with pytest.raises(robot_module.NotConfiguredError, match="empty NetworkInterface"):
        robot.connect()

    assert state["shutdown"] is True


class _FakeRobot:
    adapter_name = "fake_handeye_robot"

    def __init__(self) -> None:
        self.current: JointVector = approved_eye_in_hand_waypoints()[0].joint_positions_rad
        self.connected = False
        self.stopped = False
        self.moves: list[JointVector] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_positions_rad(self) -> JointVector:
        return self.current

    def get_base_to_flange_matrix(self) -> np.ndarray:
        _, _, joint_3, joint_4, joint_5, joint_6 = self.current
        delta_3 = joint_3 - math.pi / 2.0
        delta_5 = joint_5 - math.pi / 2.0
        return _transform(
            _rotation_xyz(joint_4, delta_5, joint_6),
            (0.32 + delta_3 * 0.6, -0.04 + joint_4 * 0.25, 0.28 + delta_5 * 0.3),
        )

    def get_base_to_tcp_matrix(self) -> np.ndarray:
        flange_to_tcp = _transform(np.eye(3), (0.0, 0.0, 0.24))
        return self.get_base_to_flange_matrix() @ flange_to_tcp

    def get_active_tcp_name(self) -> str:
        return "2FG_TCP"

    def move_joints(
        self,
        target_rad: JointVector,
        *,
        velocity_rad_s: float,
        acceleration_rad_s2: float,
    ) -> None:
        assert velocity_rad_s > 0.0
        assert acceleration_rad_s2 > 0.0
        self.moves.append(target_rad)
        self.current = target_rad

    def stop(self, *, reason: str) -> None:
        assert reason
        self.stopped = True


class _FakeCamera:
    def __init__(self) -> None:
        self.timestamp_ns = time.time_ns()

    def status(self) -> dict[str, object]:
        return {"state": "streaming", "recording": None}

    def get_latest_frame(
        self, *, after_timestamp_ns: int | None = None, timeout_s: float = 2.0
    ) -> SynchronizedRGBDFrame:
        assert timeout_s > 0.0
        self.timestamp_ns = max(self.timestamp_ns + 1, (after_timestamp_ns or 0) + 1)
        return SynchronizedRGBDFrame(
            color_image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth_image_m=np.ones((480, 640), dtype=np.float32),
            color_timestamp_ns=self.timestamp_ns,
            depth_timestamp_ns=self.timestamp_ns,
            color_intrinsics=CameraIntrinsics(640, 480, 600.0, 600.0, 320.0, 240.0),
            frame_number=self.timestamp_ns % 1_000_000,
        )


def test_controller_collects_plan_and_persists_validated_result(tmp_path: Path) -> None:
    robot = _FakeRobot()
    robot.current = (
        0.0,
        0.0,
        math.radians(85.0),
        math.radians(5.0),
        math.radians(85.0),
        math.radians(-5.0),
    )
    camera = _FakeCamera()
    flange_to_camera = _known_flange_to_camera()
    base_to_board = _transform(_rotation_xyz(0.0, 0.0, 0.1), (0.55, 0.0, 0.18))

    def detector(
        _frame: SynchronizedRGBDFrame, _config: EyeInHandCalibrationConfig
    ) -> tuple[np.ndarray, float]:
        camera_to_board = np.linalg.inv(
            robot.get_base_to_flange_matrix() @ flange_to_camera
        ) @ base_to_board
        return camera_to_board, 0.1

    source_path = tmp_path / "T_gripper2camera.npy"
    flange_to_tcp = _transform(np.eye(3), (0.0, 0.0, 0.24))
    tcp_to_camera_mm = np.linalg.inv(flange_to_tcp) @ flange_to_camera
    tcp_to_camera_mm[:3, 3] *= 1000.0
    np.save(source_path, tcp_to_camera_mm, allow_pickle=False)
    source_bytes_before = source_path.read_bytes()
    controller = HandEyeCalibrationController(
        store=LocalArtifactStore(tmp_path),
        camera=camera,  # type: ignore[arg-type]
        robot_factory=lambda: robot,
        hardware_authorized=True,
        gate_summary={"test_gate": True},
        detector=detector,
        settle_time_s=0.0,
        legacy_npy_path=source_path,
        legacy_expected_tcp_name="2FG_TCP",
    )

    started = controller.start(operator_id="test_operator")
    assert started["session"]["status"] in {"starting", "running"}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        status = controller.status()
        if status["session"]["status"] in {"passed", "failed", "aborted"}:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("calibration controller did not finish")

    assert status["session"]["status"] == "passed", status
    result = status["latest_result"]
    assert result["passed"] is True
    assert robot.moves[0] == approved_eye_in_hand_waypoints()[0].joint_positions_rad
    assert np.allclose(result["flange_to_camera"], flange_to_camera, atol=1.0e-7)
    assert (tmp_path / result["artifact_uri"]).is_file()
    assert (tmp_path / result["matrix_artifact"]["uri"]).is_file()

    imported = controller.import_legacy_npy(operator_id="test_operator")
    legacy = imported["legacy_transform"]
    assert legacy["passed"] is True
    assert legacy["candidate_only"] is True
    assert legacy["hardware_validated"] is False
    assert legacy["runtime_authorized"] is False
    assert legacy["source_session_id"] == result["session_id"]
    assert np.allclose(legacy["flange_to_camera"], flange_to_camera, atol=1.0e-7)
    assert source_path.read_bytes() == source_bytes_before
    backup_path = tmp_path / legacy["source_backup"]["uri"]
    assert backup_path != source_path
    assert backup_path.read_bytes() == source_bytes_before
    assert (tmp_path / legacy["matrix_artifact"]["uri"]).is_file()


def test_controller_refuses_to_rotate_joint_one_or_two_to_reference(tmp_path: Path) -> None:
    robot = _FakeRobot()
    robot.current = (
        math.radians(1.0),
        0.0,
        math.pi / 2.0,
        0.0,
        math.pi / 2.0,
        0.0,
    )
    controller = HandEyeCalibrationController(
        store=LocalArtifactStore(tmp_path),
        camera=_FakeCamera(),  # type: ignore[arg-type]
        robot_factory=lambda: robot,
        hardware_authorized=True,
        gate_summary={"test_gate": True},
        settle_time_s=0.0,
    )

    controller.start(operator_id="test_operator")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = controller.status()
        if status["session"]["status"] == "failed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("calibration controller did not reject unlocked J1")

    assert "automatic reference motion never rotates J1/J2" in status["session"]["error"]
    assert robot.moves == []
