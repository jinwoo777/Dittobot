from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from robot_skill_system.api.contracts import JogMoveJRequest
from robot_skill_system.jog.controller import JogController, MockJogRobot


def _controller() -> tuple[JogController, MockJogRobot]:
    robot = MockJogRobot()
    controller = JogController(
        robot_factory=lambda: robot,
        mode="mock",
        hardware_authorized=False,
        gate_summary={"ROBOT_EXECUTION_MODE=hardware": False},
        joint_velocity_rad_s=math.radians(10.0),
        joint_acceleration_rad_s2=math.radians(20.0),
    )
    return controller, robot


def _enable(controller: JogController) -> None:
    controller.enable(
        operator_id="test_operator",
        workspace_cleared=True,
        estop_ready=True,
        acknowledge_direct_motion=True,
    )


def test_movej_executes_one_complete_six_axis_target() -> None:
    controller, robot = _controller()
    _enable(controller)

    result = controller.move_to_joint_positions(
        target_joint_positions_deg=(10.0, -20.0, 30.0, 40.0, -50.0, 60.0)
    )

    assert result["joint_positions_deg"] == pytest.approx(
        [10.0, -20.0, 30.0, 40.0, -50.0, 60.0]
    )
    robot_positions_deg = tuple(
        math.degrees(value) for value in robot.joint_positions_rad
    )
    assert robot_positions_deg == pytest.approx(
        (10.0, -20.0, 30.0, 40.0, -50.0, 60.0)
    )
    assert result["last_move_at_ns"] is not None


def test_movej_rejects_closed_session_and_out_of_limit_target() -> None:
    controller, robot = _controller()
    with pytest.raises(ValueError, match="enable jog"):
        controller.move_to_joint_positions(
            target_joint_positions_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    assert robot.connected is False

    _enable(controller)
    with pytest.raises(ValueError, match="J2 target"):
        controller.move_to_joint_positions(
            target_joint_positions_deg=(0.0, 96.0, 0.0, 0.0, 0.0, 0.0)
        )
    assert controller.status()["joint_positions_deg"] == pytest.approx([0.0] * 6)


def test_movej_schema_rejects_wrong_length_nonfinite_and_schema_envelope() -> None:
    with pytest.raises(ValidationError):
        JogMoveJRequest(target_joint_positions_deg=[0.0] * 5)
    with pytest.raises(ValidationError):
        JogMoveJRequest(
            target_joint_positions_deg=[0.0, 0.0, math.nan, 0.0, 0.0, 0.0]
        )
    with pytest.raises(ValidationError):
        JogMoveJRequest(
            target_joint_positions_deg=[361.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
