"""Robot and tool adapter boundaries."""

from robot_skill_system.adapters.doosan_m0609 import DoosanM0609Adapter
from robot_skill_system.adapters.errors import (
    AdapterError,
    HardwareExecutionDisabledError,
    NotConfiguredError,
)
from robot_skill_system.adapters.gripper import GripperAdapter, GripperState
from robot_skill_system.adapters.mock_robot import MockGripperAdapter, MockRobotAdapter
from robot_skill_system.adapters.onrobot_rg2 import OnRobotRG2Adapter
from robot_skill_system.adapters.robot import (
    RobotAdapter,
    RobotCommand,
    RobotEvent,
    RobotOperationalState,
    RobotState,
)
from robot_skill_system.adapters.simulation_robot import SimulationRobotAdapter

__all__ = [
    "AdapterError",
    "DoosanM0609Adapter",
    "GripperAdapter",
    "GripperState",
    "HardwareExecutionDisabledError",
    "MockGripperAdapter",
    "MockRobotAdapter",
    "NotConfiguredError",
    "OnRobotRG2Adapter",
    "RobotAdapter",
    "RobotCommand",
    "RobotEvent",
    "RobotOperationalState",
    "RobotState",
    "SimulationRobotAdapter",
]
