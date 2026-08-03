from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from robot_skill_system.adapters import MockRobotAdapter
from robot_skill_system.primitives.profiles import load_force_profiles, load_motion_profiles
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import (
    ForceSafetyError,
    ObstacleDetectedError,
    ProfileNotApprovedError,
)
from robot_skill_system.runtime.executor import RuntimeExecutor
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.models import (
    BoundTargetPose,
    EntityBinding,
    EntityKind,
    ExecutionMode,
    PreflightReport,
    RuntimeContext,
    ValidationCheck,
)
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor
from robot_skill_system.scene.models import (
    ConfidenceSummary,
    ObstacleInstance,
    Pose,
    Quaternion,
    SceneSnapshot,
    SurfaceInstance,
    SurfaceRole,
    ToolInstance,
    Vector3,
)

NOW_NS = 11_000_000_000_000
ROOT = Path(__file__).resolve().parents[2]


def _pose(x: float = 0.5, y: float = 0.0, z: float = 0.7) -> Pose:
    return Pose(
        frame_id="base",
        position_m=Vector3(x=x, y=y, z=z),
        orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        timestamp_ns=NOW_NS,
        source="safety_hardening_fixture",
        confidence=0.99,
    )


def _scene(*, allowed_links: list[str] | None = None, obstacle: bool = False) -> SceneSnapshot:
    return SceneSnapshot(
        scene_id="hardening_scene",
        timestamp_ns=NOW_NS,
        reference_frame="base",
        valid_for_ms=5_000,
        tools=[
            ToolInstance(
                instance_id="wiper_01",
                tool_class="wiper",
                attached=True,
                tcp_frame="wiper_01_tcp",
                pose=_pose(),
                verification_confidence=0.99,
            )
        ],
        surfaces=[
            SurfaceInstance(
                instance_id="surface_01",
                role=SurfaceRole.CONTACT_TARGET,
                center_m=Vector3(x=0.5, y=0.0, z=0.7),
                normal=Vector3(x=0.0, y=0.0, z=1.0),
                confidence=0.99,
                allowed_contact_links=(
                    ["wiper_01_tcp"] if allowed_links is None else allowed_links
                ),
            )
        ],
        static_obstacles=(
            [
                ObstacleInstance(
                    instance_id="fixture_block",
                    pose=_pose(0.6, 0.0, 0.8),
                    geometry={"type": "box", "size_m": [0.02, 0.02, 0.02]},
                    confidence=0.99,
                )
            ]
            if obstacle
            else []
        ),
        calibration_id="hardening_calibration",
        confidence_summary=ConfidenceSummary(overall=0.99),
    )


def _bindings(scene: SceneSnapshot) -> dict[str, EntityBinding]:
    return {
        "$tool": EntityBinding(
            "$tool", "wiper_01", EntityKind.TOOL, 0.99, scene.tools[0]
        ),
        "$surface": EntityBinding(
            "$surface", "surface_01", EntityKind.SURFACE, 0.99, scene.surfaces[0]
        ),
    }


def _target(x: float = 0.6) -> BoundTargetPose:
    return BoundTargetPose(
        frame_id="base",
        position_m=(x, 0.0, 0.8),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        anchor_entity_id="surface_01",
        timestamp_ns=NOW_NS,
    )


def _geometry_report() -> PreflightReport:
    return PreflightReport(
        tuple(
            ValidationCheck(name, True, "test_preflight", is_mock=True)
            for name in ("ik", "joint_limits", "singularity_margin")
        )
    )


def _profiles() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_motion_profiles(ROOT / "configs/motion_profiles/default.json"),
        load_force_profiles(ROOT / "configs/force_profiles/default.json"),
    )


def _enabled_supervisor(robot: MockRobotAdapter) -> GlobalForceSupervisor:
    _, forces = _profiles()
    supervisor = GlobalForceSupervisor(
        robot, forces, allowed_contact_links=["wiper_01_tcp"]
    )
    supervisor.enable(
        "wipe_light",
        tool_class="wiper",
        surface_role="contact_target",
        target_surface_id="surface_01",
        surface_normal_xyz=(0.0, 0.0, 1.0),
        contact_link="wiper_01_tcp",
        contact_search_succeeded=True,
        ik_passed=True,
    )
    return supervisor


def test_force_monitor_rejects_contact_loss_tangential_spike_and_wrong_direction() -> None:
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    robot.connect()
    supervisor = _enabled_supervisor(robot)

    robot.set_tool_force((0.0, 0.0, -0.5))
    with pytest.raises(ForceSafetyError, match="contact force lost"):
        supervisor.monitor()

    robot.set_tool_force((4.1, 0.0, -5.0))
    with pytest.raises(ForceSafetyError, match="tangential force exceeded"):
        supervisor.monitor()

    robot.set_tool_force((0.0, 0.0, 5.0))
    with pytest.raises(ForceSafetyError, match="opposite"):
        supervisor.monitor()
    supervisor.disable()


def test_force_enable_requires_global_link_allowlist() -> None:
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    robot.connect()
    _, forces = _profiles()
    supervisor = GlobalForceSupervisor(robot, forces)

    with pytest.raises(ForceSafetyError, match="no globally allowed"):
        supervisor.enable(
            "wipe_light",
            tool_class="wiper",
            surface_role="contact_target",
            target_surface_id="surface_01",
            surface_normal_xyz=(0.0, 0.0, 1.0),
            contact_link="wiper_01_tcp",
            contact_search_succeeded=True,
            ik_passed=True,
        )


class _CountingForceRobot(MockRobotAdapter):
    def __init__(self) -> None:
        super().__init__(clock_ns=lambda: NOW_NS)
        self.force_read_count = 0

    def read_tool_force(self) -> tuple[float, ...]:
        self.force_read_count += 1
        return super().read_tool_force()


class _ContactDropsAfterMoveRobot(MockRobotAdapter):
    def move_l(
        self,
        target_pose: Any,
        *,
        velocity_m_s: float,
        acceleration_m_s2: float,
        blend_radius_m: float = 0.0,
    ) -> None:
        super().move_l(
            target_pose,
            velocity_m_s=velocity_m_s,
            acceleration_m_s2=acceleration_m_s2,
            blend_radius_m=blend_radius_m,
        )
        self.set_tool_force((0.0, 0.0, 0.0))


def test_each_force_active_blocking_cartesian_call_is_polled_before_and_after() -> None:
    robot = _CountingForceRobot()
    robot.connect()
    force = _enabled_supervisor(robot)
    motion, forces = _profiles()
    scene = _scene()
    executor = RuntimeExecutor(
        context=RuntimeContext(
            scene=scene,
            bindings=_bindings(scene),
            motion_profiles=motion,
            force_profiles=forces,
            execution_mode=ExecutionMode.MOCK,
        ),
        robot=robot,
        binder=EntityBinder(clock_ns=lambda: NOW_NS),
        workspace_supervisor=GlobalWorkspaceSupervisor(),
        force_supervisor=force,
        safety_supervisor=GlobalSafetySupervisor(),
    )

    async def run_calls() -> None:
        await executor.execute_primitive(
            "motion.move_l",
            {"target": _target(), "motion_profile_id": "linear_slow"},
            None,
            None,
        )
        await executor.execute_primitive(
            "motion.move_c",
            {
                "via": _target(0.61),
                "target": _target(0.62),
                "motion_profile_id": "circular_normal",
            },
            None,
            None,
        )
        await executor.execute_primitive(
            "motion.move_periodic",
            {
                "center": _target(0.62),
                "amplitude_m": (0.01, 0.0, 0.0),
                "repetitions": 1,
                "motion_profile_id": "periodic_safe",
            },
            None,
            None,
        )
        await executor.execute_primitive(
            "contact.follow_path",
            {"path": [_target(0.63)], "motion_profile_id": "linear_slow"},
            None,
            None,
        )

    asyncio.run(run_calls())

    assert robot.force_read_count == 8
    force.disable()


def test_contact_loss_after_blocking_move_stops_then_cleans_up_force() -> None:
    robot = _ContactDropsAfterMoveRobot(clock_ns=lambda: NOW_NS)
    robot.connect()
    force = _enabled_supervisor(robot)
    motion, forces = _profiles()
    scene = _scene()
    executor = RuntimeExecutor(
        context=RuntimeContext(
            scene=scene,
            bindings=_bindings(scene),
            motion_profiles=motion,
            force_profiles=forces,
            execution_mode=ExecutionMode.MOCK,
        ),
        robot=robot,
        binder=EntityBinder(clock_ns=lambda: NOW_NS),
        workspace_supervisor=GlobalWorkspaceSupervisor(),
        force_supervisor=force,
        safety_supervisor=GlobalSafetySupervisor(),
    )

    async def compiled_run(runtime: RuntimeExecutor) -> None:
        await runtime.execute_primitive(
            "motion.move_l",
            {"target": _target(), "motion_profile_id": "linear_slow"},
            None,
            None,
        )

    with pytest.raises(ForceSafetyError, match="contact force lost"):
        asyncio.run(executor.execute_compiled(compiled_run))

    operations = [command.operation for command in robot.commands]
    cleanup_start = operations.index("stop")
    assert operations[cleanup_start : cleanup_start + 4] == [
        "stop",
        "release_force",
        "release_compliance",
        "safe_retract",
    ]
    assert not robot.force_active
    assert not robot.compliance_active


def test_executor_force_enable_uses_preflight_evidence_and_exact_surface_link() -> None:
    motion, forces = _profiles()
    scene = _scene()
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    robot.connect()

    def executor(report: PreflightReport | None) -> RuntimeExecutor:
        return RuntimeExecutor(
            context=RuntimeContext(
                scene=scene,
                bindings=_bindings(scene),
                motion_profiles=motion,
                force_profiles=forces,
                execution_mode=ExecutionMode.MOCK,
                preflight_report=report,
            ),
            robot=robot,
            binder=EntityBinder(clock_ns=lambda: NOW_NS),
            workspace_supervisor=GlobalWorkspaceSupervisor(),
            force_supervisor=GlobalForceSupervisor(
                robot, forces, allowed_contact_links=["wiper_01_tcp"]
            ),
            safety_supervisor=GlobalSafetySupervisor(),
        )

    unvalidated = executor(None)
    unvalidated.contact_search_succeeded = True
    with pytest.raises(ForceSafetyError, match="IK/singularity"):
        asyncio.run(
            unvalidated.execute_primitive(
                "contact.enable_force",
                {"surface": "$surface", "force_profile_id": "wipe_light"},
                None,
                None,
            )
        )

    validated = executor(_geometry_report())
    validated.contact_search_succeeded = True
    assert asyncio.run(
        validated.execute_primitive(
            "contact.enable_force",
            {"surface": "$surface", "force_profile_id": "wipe_light"},
            None,
            None,
        )
    )
    validated.force_supervisor.disable()

    mismatched_scene = _scene(allowed_links=["different_tcp"])
    mismatched = RuntimeExecutor(
        context=RuntimeContext(
            scene=mismatched_scene,
            bindings=_bindings(mismatched_scene),
            motion_profiles=motion,
            force_profiles=forces,
            execution_mode=ExecutionMode.MOCK,
            preflight_report=_geometry_report(),
        ),
        robot=robot,
        binder=EntityBinder(clock_ns=lambda: NOW_NS),
        workspace_supervisor=GlobalWorkspaceSupervisor(),
        force_supervisor=GlobalForceSupervisor(
            robot, forces, allowed_contact_links=["wiper_01_tcp"]
        ),
        safety_supervisor=GlobalSafetySupervisor(),
    )
    mismatched.contact_search_succeeded = True
    with pytest.raises(ForceSafetyError, match="not explicitly allowed"):
        asyncio.run(
            mismatched.execute_primitive(
                "contact.enable_force",
                {"surface": "$surface", "force_profile_id": "wipe_light"},
                None,
                None,
            )
        )


def test_workspace_validate_target_is_not_a_noop() -> None:
    motion, forces = _profiles()
    scene = _scene(obstacle=True)
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    robot.connect()
    executor = RuntimeExecutor(
        context=RuntimeContext(
            scene=scene,
            bindings=_bindings(scene),
            motion_profiles=motion,
            force_profiles=forces,
            execution_mode=ExecutionMode.MOCK,
        ),
        robot=robot,
        binder=EntityBinder(clock_ns=lambda: NOW_NS),
        workspace_supervisor=GlobalWorkspaceSupervisor(),
        force_supervisor=GlobalForceSupervisor(robot, forces),
        safety_supervisor=GlobalSafetySupervisor(),
    )

    with pytest.raises(ObstacleDetectedError, match="fixture_block"):
        asyncio.run(
            executor.execute_primitive(
                "workspace.validate_target",
                {
                    "target": {
                        "anchor_id": "$surface",
                        "position_m": {"x": 0.1, "y": 0.0, "z": 0.1},
                        "orientation_xyzw": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0,
                        },
                    }
                },
                None,
                None,
            )
        )
    assert [command.operation for command in robot.commands][-1] == "stop"


def test_loaded_safety_policy_caps_motion_and_force_profiles() -> None:
    motion, forces = _profiles()
    scene = _scene()
    robot = MockRobotAdapter(clock_ns=lambda: NOW_NS)
    robot.connect()
    context = RuntimeContext(
        scene=scene,
        bindings=_bindings(scene),
        motion_profiles=motion,
        force_profiles=forces,
        execution_mode=ExecutionMode.MOCK,
        safety_policy={"maximum_linear_velocity_m_s": 0.001},
    )
    executor = RuntimeExecutor(
        context=context,
        robot=robot,
        binder=EntityBinder(clock_ns=lambda: NOW_NS),
        workspace_supervisor=GlobalWorkspaceSupervisor(),
        force_supervisor=GlobalForceSupervisor(robot, forces),
        safety_supervisor=GlobalSafetySupervisor(),
    )

    with pytest.raises(ProfileNotApprovedError, match="exceeds loaded safety policy"):
        asyncio.run(
            executor.execute_primitive(
                "motion.move_l",
                {"target": _target(), "motion_profile_id": "linear_slow"},
                None,
                None,
            )
        )

    force = GlobalForceSupervisor(
        robot,
        forces,
        allowed_contact_links=["wiper_01_tcp"],
        maximum_force_n=5.0,
    )
    with pytest.raises(ForceSafetyError, match="safety policy maximum"):
        force.enable(
            "wipe_light",
            tool_class="wiper",
            surface_role="contact_target",
            target_surface_id="surface_01",
            surface_normal_xyz=(0.0, 0.0, 1.0),
            contact_link="wiper_01_tcp",
            contact_search_succeeded=True,
            ik_passed=True,
        )
