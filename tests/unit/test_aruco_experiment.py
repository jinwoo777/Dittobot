from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_skill_system.aruco_experiment.controller import (
    PLANE_Z_TEST_DISTANCE_M,
    REFERENCE_JOINT_DEG,
    ArucoExperimentController,
    DoosanArucoExperimentRobot,
    MockArucoExperimentRobot,
    _joint_values,
)
from robot_skill_system.exceptions import NotConfiguredError


def _controller(
    tmp_path: Path,
    *,
    robot: MockArucoExperimentRobot | None = None,
    expected_tcp_name: str = "GripperDA_v1",
) -> ArucoExperimentController:
    instance = robot or MockArucoExperimentRobot(active_tcp_name=expected_tcp_name)
    return ArucoExperimentController(
        robot_factory=lambda: instance,
        mode="mock",
        hardware_authorized=False,
        gate_summary={"ROBOT_EXECUTION_MODE=hardware": False},
        reference_npz=Path("aruco/fixed_workspace_reference.npz").resolve(),
        runtime_npz=tmp_path / "runtime_workspace.npz",
        expected_tcp_name=expected_tcp_name,
        joint_velocity_rad_s=0.175,
        joint_acceleration_rad_s2=0.25,
        linear_velocity_m_s=0.0135,
        linear_acceleration_m_s2=0.036,
        angular_velocity_rad_s=0.09,
        angular_acceleration_rad_s2=0.18,
    )


def _enable(controller: ArucoExperimentController, *, width_mm: float = 80.0) -> None:
    controller.enable(
        operator_id="test_operator",
        workspace_cleared=True,
        estop_ready=True,
        acknowledge_direct_motion=True,
        object_width_mm=width_mm,
        width_model="full-opening",
    )


def test_joint_values_accept_ros_fixed_size_numpy_array() -> None:
    response = np.asarray([0.0, 1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)

    assert _joint_values(response, label="Doosan IK response") == (
        0.0,
        1.0,
        -2.0,
        3.0,
        -4.0,
        5.0,
    )


def test_aruco_robot_uses_configured_tcp_when_driver_returns_empty_name() -> None:
    robot = DoosanArucoExperimentRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
        expected_tcp_name="GripperDA_v1",
    )
    robot._get_tcp = lambda: ""

    assert robot.get_active_tcp_name() == "GripperDA_v1"


def test_aruco_robot_reports_named_non_standby_controller_state() -> None:
    robot = DoosanArucoExperimentRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
    )
    robot._get_robot_state = lambda: 3

    with pytest.raises(ValueError, match=r"STATE_SAFE_OFF \(3\)"):
        robot._require_standby()


def test_aruco_robot_reports_empty_vendor_tcp_pose_response() -> None:
    robot = DoosanArucoExperimentRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
    )

    def empty_vendor_response() -> object:
        raise IndexError("list index out of range")

    robot._get_current_posx = empty_vendor_response

    robot._ACTIVE_TCP_POSE_RETRY_DELAY_S = 0.0

    with pytest.raises(NotConfiguredError, match="failed after 3 bounded attempts"):
        robot.get_base_to_tcp_matrix()


def test_aruco_robot_retries_empty_vendor_tcp_pose_response() -> None:
    robot = DoosanArucoExperimentRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
    )
    responses: list[object] = [IndexError(), IndexError(), ([1, 2, 3, 4, 5, 6], 0)]

    def transient_vendor_response() -> object:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    robot._get_current_posx = transient_vendor_response
    robot._ACTIVE_TCP_POSE_RETRY_DELAY_S = 0.0

    assert robot.get_tcp_pose_base_mm_zyz_deg() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert responses == []


@pytest.mark.parametrize("empty_response", [None, [], (), np.asarray([])])
def test_aruco_robot_retries_empty_list_tcp_pose_response(
    empty_response: object,
) -> None:
    robot = DoosanArucoExperimentRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
    )
    responses = [empty_response, ([1, 2, 3, 4, 5, 6], 0)]

    def transient_empty_response() -> object:
        return responses.pop(0)

    robot._get_current_posx = transient_empty_response
    robot._ACTIVE_TCP_POSE_RETRY_DELAY_S = 0.0

    assert robot.get_tcp_pose_base_mm_zyz_deg() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert responses == []


def test_aruco_robot_uses_bounded_direct_tcp_pose_service() -> None:
    robot = DoosanArucoExperimentRobot(
        robot_id="dsr01",
        robot_model="m0609",
        execution_mode="hardware",
        hardware_enabled=True,
    )

    class Request:
        ref = -1

    class Future:
        def __init__(self, response: object) -> None:
            self.response = response

        def done(self) -> bool:
            return True

        def result(self) -> object:
            return self.response

    responses = [
        SimpleNamespace(success=False, task_pos_info=[]),
        SimpleNamespace(success=False, task_pos_info=[]),
        SimpleNamespace(
            success=True,
            task_pos_info=[SimpleNamespace(data=[10, 20, 30, 40, 50, 60, 0])],
        ),
    ]
    requests: list[Request] = []

    class Client:
        def call_async(self, request: Request) -> Future:
            requests.append(request)
            return Future(responses.pop(0))

    robot._rclpy = SimpleNamespace(
        spin_until_future_complete=lambda *args, **kwargs: None
    )
    robot._node = object()
    robot._current_posx_client = Client()
    robot._current_posx_type = SimpleNamespace(Request=Request)
    robot._ACTIVE_TCP_POSE_RETRY_DELAY_S = 0.0

    assert robot.get_tcp_pose_base_mm_zyz_deg() == (
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    )
    assert [request.ref for request in requests] == [0, 0, 0]


def test_mock_two_step_experiment_uses_camera_z_upper_bound(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _enable(controller)

    enabled = controller.status()
    runtime = enabled["runtime_workspace"]
    camera_z = enabled["capabilities"]["reference"]["plane_to_camera_z_range_m"][1]
    assert runtime["z_max_plane_m"] == pytest.approx(camera_z)
    assert (tmp_path / "runtime_workspace.npz").is_file()
    with np.load(tmp_path / "runtime_workspace.npz", allow_pickle=False) as artifact:
        assert artifact["workspace_point_z_bounds_plane_m"].tolist() == pytest.approx(
            [0.0, camera_z]
        )
        assert artifact["tcp_z_bounds_plane_m"].tolist() == pytest.approx(
            [runtime["z_min_plane_m"], camera_z]
        )

    referenced = controller.move_to_reference()
    assert referenced["reference_captured"] is True
    assert referenced["joint_positions_deg"] == pytest.approx(REFERENCE_JOINT_DEG)
    before = np.asarray(referenced["tcp_base_xyz_m"], dtype=np.float64)

    completed = controller.move_plane_z_test()
    after = np.asarray(completed["tcp_base_xyz_m"], dtype=np.float64)
    assert completed["z_test_completed"] is True
    assert np.linalg.norm(after - before) == pytest.approx(PLANE_Z_TEST_DISTANCE_M)


def test_z_test_rejects_wrong_order_and_duplicate(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _enable(controller)
    with pytest.raises(ValueError, match="reference pose first"):
        controller.move_plane_z_test()
    controller.move_to_reference()
    controller.move_plane_z_test()
    with pytest.raises(ValueError, match="already completed"):
        controller.move_plane_z_test()


def test_enable_rejects_active_tcp_mismatch(tmp_path: Path) -> None:
    robot = MockArucoExperimentRobot(active_tcp_name="wrong_tcp")
    controller = _controller(
        tmp_path,
        robot=robot,
        expected_tcp_name="GripperDA_v1",
    )
    with pytest.raises(ValueError, match="does not match frozen workspace TCP"):
        _enable(controller)
    assert robot.connected is False


def test_recorded_failure_survives_status_poll_until_success_or_stop(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    _enable(controller)
    controller.record_failure(
        "move_to_reference", ValueError("motion interlock rejected the request")
    )

    failed = controller.status()
    assert failed["last_action"] == "move_to_reference_failed"
    assert failed["last_error"] == (
        "ValueError: motion interlock rejected the request"
    )

    recovered = controller.move_to_reference()
    assert recovered["last_error"] is None
    controller.record_failure("move_plane_z_test", RuntimeError("adapter failure"))
    assert controller.stop()["last_error"] is None


def test_status_uses_cached_feedback_without_another_robot_call(tmp_path: Path) -> None:
    robot = MockArucoExperimentRobot()
    controller = _controller(tmp_path, robot=robot)
    _enable(controller)
    expected = controller.status()["tcp_base_xyz_m"]

    def unexpected_status_poll() -> tuple[float, float, float, float, float, float]:
        raise AssertionError("status must not call the robot")

    robot.get_joint_positions_rad = unexpected_status_poll  # type: ignore[method-assign]
    robot.get_base_to_tcp_matrix = unexpected_status_poll  # type: ignore[method-assign]
    assert controller.status()["tcp_base_xyz_m"] == expected
    assert controller.status()["feedback_source"] == "last_serialized_aruco_action"


def test_enable_rejects_tcp_outside_one_metre_base_radius(tmp_path: Path) -> None:
    robot = MockArucoExperimentRobot()
    robot.base_to_tcp[:3, 3] = [1.01, 0.0, 0.0]
    controller = _controller(tmp_path, robot=robot)
    with pytest.raises(ValueError, match="exceeds the approved 1.0 m"):
        _enable(controller)


def test_width_runtime_matches_requested_model(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _enable(controller, width_mm=80.0)
    runtime = controller.status()["runtime_workspace"]
    assert runtime["theta_deg"] == pytest.approx(
        math.degrees(math.asin((80.0 / 2.0) / 110.0))
    )
    assert runtime["z_min_plane_m"] < runtime["z_max_plane_m"]
