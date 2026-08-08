from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_skill_system.adapters.doosan_m0609 import (
    DoosanM0609Adapter,
    runtime_pose_to_doosan,
)
from robot_skill_system.adapters.onrobot_rg2 import OnRobotRG2Adapter
from robot_skill_system.runtime.models import BoundTargetPose


class _DoosanSession:
    def __init__(self) -> None:
        self.moves: list[tuple[str, Any]] = []

    def _require_standby(self) -> None:
        return None

    def get_base_to_tcp_matrix(self) -> np.ndarray:
        return np.eye(4)

    def get_joint_positions_rad(self) -> tuple[float, ...]:
        return (0.0,) * 6

    def solve_inverse_kinematics(self, target: Any) -> tuple[float, ...]:
        self.moves.append(("ik", target))
        return (0.0,) * 6

    def move_joints(self, target: Any, **kwargs: Any) -> None:
        self.moves.append(("move_j", target))

    def move_linear(self, target: Any, **kwargs: Any) -> None:
        self.moves.append(("move_l", target))

    def stop(self, *, reason: str) -> None:
        self.moves.append(("stop", reason))


class _RG2Status:
    grip_detected = True
    has_safety_fault = False


class _RG2Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.connected = False
        self.width_mm = 55.0
        self.moves: list[tuple[float, float]] = []

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def get_status(self) -> _RG2Status:
        return _RG2Status()

    def get_width_mm(self) -> float:
        return self.width_mm

    def move_to_width(self, width_mm: float, force_n: float) -> None:
        self.width_mm = width_mm
        self.moves.append((width_mm, force_n))

    def stop(self) -> None:
        return None


def _target() -> BoundTargetPose:
    return BoundTargetPose(
        frame_id="base",
        position_m=(0.4, -0.1, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        anchor_entity_id="plane",
        timestamp_ns=1,
    )


def test_doosan_runtime_uses_live_ik_then_move_l() -> None:
    session = _DoosanSession()
    adapter = DoosanM0609Adapter(session)
    adapter.connect()
    adapter.move_l(_target(), velocity_m_s=0.02, acceleration_m_s2=0.04)

    pose = runtime_pose_to_doosan(_target())
    assert pose[:3] == pytest.approx((400.0, -100.0, 300.0))
    assert [operation for operation, _ in session.moves] == ["ik", "move_l"]


def test_doosan_runtime_uses_fixed_workspace_tcp_orientation_for_ik_and_motion() -> None:
    session = _DoosanSession()
    fixed_orientation = (-0.70710678, 0.70710678, 0.0, 0.0)
    adapter = DoosanM0609Adapter(
        session,
        fixed_workspace_orientation_xyzw=fixed_orientation,
    )
    adapter.connect()
    adapter.move_l(_target(), velocity_m_s=0.02, acceleration_m_s2=0.04)

    expected = runtime_pose_to_doosan(
        _target(), fixed_workspace_orientation_xyzw=fixed_orientation
    )
    assert session.moves[0] == ("ik", pytest.approx(expected))
    assert session.moves[1] == ("move_l", pytest.approx(expected))


def test_doosan_runtime_reuses_acquired_pose_during_fixed_workspace_run() -> None:
    session = _DoosanSession()
    initial = _target()
    adapter = DoosanM0609Adapter(session, initial_pose=initial)

    assert adapter.get_current_pose() == initial
    assert session.moves == []

    adapter.move_l(_target(), velocity_m_s=0.02, acceleration_m_s2=0.04)
    assert adapter.get_current_pose().timestamp_ns >= initial.timestamp_ns


def test_rg2_runtime_preserves_installed_client_and_commands_width() -> None:
    adapter = OnRobotRG2Adapter(
        host="192.168.1.1",
        execution_mode="hardware",
        hardware_enabled=True,
        client_factory=_RG2Client,
    )
    adapter.connect()
    adapter.close()
    assert adapter.get_state().width_m == pytest.approx(0.0)
    assert adapter.get_state().is_holding is True
    adapter.open()
    assert adapter.get_state().width_m == pytest.approx(0.110)
