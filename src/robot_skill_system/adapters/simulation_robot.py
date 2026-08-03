"""Simulation adapter placeholder.

The MVP simulation path intentionally aliases the deterministic mock until a verified simulator
transport is configured.
"""

from robot_skill_system.adapters.mock_robot import MockRobotAdapter

SimulationRobotAdapter = MockRobotAdapter

__all__ = ["SimulationRobotAdapter"]
