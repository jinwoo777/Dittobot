"""Fixed, reusable orchestration flow for skill resolution through execution logging."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import Any

from robot_skill_system.adapters.errors import HardwareExecutionDisabledError
from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import PreflightError
from robot_skill_system.runtime.event_log import InMemoryEventSink, StorageEventSink
from robot_skill_system.runtime.executor import RuntimeExecutor
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.integrity import verify_skill_checksum
from robot_skill_system.runtime.models import (
    EntityBinding,
    EntityRequirement,
    ExecutionMode,
    ExecutionResult,
    PreflightReport,
    RuntimeContext,
    RuntimeState,
)
from robot_skill_system.runtime.preflight import (
    MockGeometryValidator,
    PreflightPolicy,
    PreflightValidator,
)
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.scene_monitor import CameraSceneMonitor
from robot_skill_system.runtime.state_machine import RuntimeStateMachine
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor
from robot_skill_system.storage.database import StorageRepository


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class RuntimeOrchestrator:
    """One process-level orchestrator reused across skill executions."""

    def __init__(
        self,
        *,
        robot: Any,
        motion_profiles: Mapping[str, Any],
        force_profiles: Mapping[str, Any],
        safety_policy: Any | None = None,
        execution_mode: ExecutionMode | str = ExecutionMode.DRY_RUN,
        enable_hardware_execution: bool = False,
        robot_backend: str = "mock",
        enable_real_robot: bool = False,
        dry_run: bool = True,
        gripper: Any | None = None,
        primitive_registry: Any | None = None,
        binder: EntityBinder | None = None,
        preflight_validator: PreflightValidator | None = None,
        safety_supervisor: GlobalSafetySupervisor | None = None,
        workspace_supervisor: GlobalWorkspaceSupervisor | None = None,
        force_supervisor: GlobalForceSupervisor | None = None,
        scene_monitor: CameraSceneMonitor | None = None,
        storage_repository: StorageRepository | None = None,
        command_resolver: Callable[[str], Any | Awaitable[Any]] | None = None,
        skill_retriever: Callable[[Any], Any | Awaitable[Any]] | None = None,
        scene_capture: Callable[[], Any | Awaitable[Any]] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.robot = robot
        self.gripper = gripper
        self.motion_profiles = motion_profiles
        self.force_profiles = force_profiles
        self.safety_policy = safety_policy
        self.execution_mode = (
            execution_mode
            if isinstance(execution_mode, ExecutionMode)
            else ExecutionMode(_text(execution_mode))
        )
        self.enable_hardware_execution = enable_hardware_execution
        self.robot_backend = robot_backend
        self.enable_real_robot = enable_real_robot
        self.dry_run = dry_run
        self.primitive_registry = primitive_registry
        self.binder = binder or EntityBinder(clock_ns=clock_ns)
        minimum_clearance_m = float(
            _get(safety_policy, "minimum_clearance_m", default=0.03)
        )
        geometry_validator = MockGeometryValidator(
            minimum_clearance_m=minimum_clearance_m
        )
        self.preflight_validator = preflight_validator or PreflightValidator(
            policy=PreflightPolicy(minimum_clearance_m=minimum_clearance_m),
            geometry_validator=geometry_validator,
            clock_ns=clock_ns,
        )
        self.safety_supervisor = safety_supervisor or GlobalSafetySupervisor()
        self.workspace_supervisor = workspace_supervisor or GlobalWorkspaceSupervisor(
            geometry_validator=geometry_validator.geometry,
            minimum_clearance_m=minimum_clearance_m,
        )
        self.force_supervisor = force_supervisor or GlobalForceSupervisor(
            robot,
            force_profiles,
            maximum_force_n=_get(safety_policy, "maximum_force_n"),
        )
        self.scene_monitor = scene_monitor or CameraSceneMonitor(clock_ns=clock_ns)
        self.storage_repository = storage_repository
        self.command_resolver = command_resolver
        self.skill_retriever = skill_retriever
        self.scene_capture_provider = scene_capture
        self._clock_ns = clock_ns
        self.state_machine = RuntimeStateMachine()
        self.event_sink = InMemoryEventSink(clock_ns=clock_ns)
        self.last_context: RuntimeContext | None = None
        self.last_preflight: PreflightReport | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        robot: Any,
        motion_profiles: Mapping[str, Any],
        force_profiles: Mapping[str, Any],
        **kwargs: Any,
    ) -> RuntimeOrchestrator:
        return cls(
            robot=robot,
            motion_profiles=motion_profiles,
            force_profiles=force_profiles,
            execution_mode=_text(_get(settings, "robot_execution_mode", default="dry_run")),
            enable_hardware_execution=bool(
                _get(settings, "enable_hardware_execution", default=False)
            ),
            robot_backend=str(_get(settings, "robot_backend", default="mock")),
            enable_real_robot=bool(_get(settings, "enable_real_robot", default=False)),
            dry_run=bool(_get(settings, "dry_run", default=True)),
            **kwargs,
        )

    async def run(
        self,
        *,
        command_text: str,
        skill: Any | None = None,
        scene: Any | None = None,
        requirements: Sequence[EntityRequirement | Mapping[str, Any] | Any] | None = None,
        expected_skill_checksum_sha256: str | None = None,
        skill_version_id: str | None = None,
        persisted_scene_id: str | None = None,
        compiled_run: Callable[[RuntimeExecutor], Awaitable[Any]] | None = None,
        raise_on_error: bool = False,
    ) -> ExecutionResult:
        if self.state_machine.state in {RuntimeState.SUCCEEDED, RuntimeState.FAILED}:
            self.state_machine.reset()
        if self.state_machine.state is not RuntimeState.IDLE:
            raise RuntimeError("runtime orchestrator is already executing a skill")
        self.event_sink = InMemoryEventSink(clock_ns=self._clock_ns)
        execution_run_id: str | None = None
        error: BaseException | None = None
        try:
            self.state_machine.transition(RuntimeState.RESOLVING)
            resolved_command = await self.resolve_command(command_text)
            selected_skill = await self.retrieve_skill(resolved_command, supplied_skill=skill)
            if expected_skill_checksum_sha256 is None:
                expected_skill_checksum_sha256 = _get(
                    selected_skill, "graph_checksum_sha256", "manifest_checksum_sha256"
                )
            if expected_skill_checksum_sha256 is not None:
                verify_skill_checksum(selected_skill, expected_skill_checksum_sha256)

            self.state_machine.transition(RuntimeState.CAPTURING)
            current_scene = await self.capture_scene(supplied_scene=scene)
            self.state_machine.transition(RuntimeState.BINDING)
            bindings = self.bind_entities(
                selected_skill, current_scene, requirements=requirements
            )
            self._authorize_hardware_before_connect()
            self._connect_adapters_if_needed()
            self.state_machine.transition(RuntimeState.PREFLIGHT)
            report = self.validate_preconditions(selected_skill, current_scene, bindings)
            self.validate_ik_and_collision(report)
            self.last_preflight = report
            self.state_machine.transition(RuntimeState.READY)

            if self.storage_repository is not None:
                run_record = self.storage_repository.start_execution(
                    started_at_ns=self._clock_ns(),
                    execution_mode=self.execution_mode.value,
                    skill_version_id=skill_version_id,
                    scene_id=persisted_scene_id,
                    command_text=command_text,
                    preflight=report.as_dict(),
                    bindings={key: value.entity_id for key, value in bindings.items()},
                )
                execution_run_id = run_record.id
                self.event_sink = StorageEventSink(
                    self.storage_repository,
                    execution_run_id,
                    clock_ns=self._clock_ns,
                )

            context = self.create_execution_context(
                selected_skill,
                current_scene,
                bindings,
                command_text=command_text,
                execution_run_id=execution_run_id,
            )
            self.last_context = context
            self.state_machine.transition(RuntimeState.EXECUTING)
            self.event_sink.record(
                "execution_started",
                {
                    "mode": self.execution_mode.value,
                    "skill_id": _get(selected_skill, "skill_id", default="unknown"),
                },
            )
            await self.execute_skill(selected_skill, context, compiled_run=compiled_run)
            self.monitor_execution(current_scene)
            self.state_machine.transition(RuntimeState.SUCCEEDED)
            self.event_sink.record("execution_succeeded")
        except BaseException as exc:
            error = exc
            # The executor already stopped/released/retracted on workspace/contact failures.
            self.state_machine.transition(RuntimeState.FAILED)
            self.event_sink.record(
                "execution_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
                severity="error",
            )
            # Covers failures that occur between force enablement and executor cleanup boundaries.
            self.force_supervisor.disable()
        finally:
            if execution_run_id is not None and self.storage_repository is not None:
                self.finalize_log(execution_run_id, error=error)

        success = error is None
        result = ExecutionResult(
            success=success,
            status=self.state_machine.state,
            execution_run_id=execution_run_id,
            events=self.event_sink.events,
            error_code=None if error is None else getattr(error, "code", "runtime_error"),
            error_message=None if error is None else str(error),
        )
        if error is not None and raise_on_error:
            raise error
        return result

    async def resolve_command(self, command_text: str) -> Any:
        if not command_text.strip():
            raise ValueError("runtime command cannot be empty")
        self.event_sink.record("command_resolving")
        if self.command_resolver is None:
            return command_text.strip()
        return await _await_if_needed(self.command_resolver(command_text))

    async def retrieve_skill(self, resolved_command: Any, *, supplied_skill: Any | None) -> Any:
        if supplied_skill is not None:
            selected = supplied_skill
        elif self.skill_retriever is not None:
            selected = await _await_if_needed(self.skill_retriever(resolved_command))
        else:
            raise LookupError("no supplied skill or configured skill retriever")
        if selected is None:
            raise LookupError("no active validated skill matched the command")
        lifecycle = _get(selected, "lifecycle_status", "status", default="active")
        validation = _get(selected, "validation_status", default="passed")
        if _text(lifecycle) not in {"active"} or _text(validation) not in {"passed"}:
            raise PreflightError("runtime accepts only active, validated skill versions")
        self.event_sink.record(
            "skill_retrieved", {"skill_id": _get(selected, "skill_id", default="unknown")}
        )
        return selected

    async def capture_scene(self, *, supplied_scene: Any | None) -> Any:
        if supplied_scene is not None:
            captured = supplied_scene
        elif self.scene_capture_provider is not None:
            captured = await _await_if_needed(self.scene_capture_provider())
        else:
            raise LookupError("no supplied scene or configured scene capture provider")
        if captured is None:
            raise LookupError("scene capture produced no SceneSnapshot")
        self.event_sink.record(
            "scene_captured", {"scene_id": _get(captured, "scene_id", default="unknown")}
        )
        return captured

    def bind_entities(
        self,
        skill: Any,
        scene: Any,
        *,
        requirements: Sequence[EntityRequirement | Mapping[str, Any] | Any] | None = None,
    ) -> dict[str, EntityBinding]:
        if requirements is None:
            requirement_method = getattr(skill, "binding_requirements", None)
            if callable(requirement_method):
                requirements = requirement_method()
            else:
                raw = _get(skill, "bindings", default={})
                requirements = tuple(raw.values()) if isinstance(raw, Mapping) else tuple(raw)
        bindings = self.binder.bind_entities(scene, tuple(requirements))
        self.event_sink.record(
            "entities_bound", {key: value.entity_id for key, value in bindings.items()}
        )
        return bindings

    def validate_preconditions(
        self, skill: Any, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> PreflightReport:
        skill_uses_force = self._skill_uses_force(skill)
        required_tools = tuple(_get(skill, "required_tools", default=()) or ())
        return self.preflight_validator.validate(
            scene=scene,
            skill=skill,
            bindings=bindings,
            robot=self.robot,
            execution_mode=self.execution_mode,
            enable_hardware_execution=self.enable_hardware_execution,
            skill_validation_status=_text(_get(skill, "validation_status", default="passed")),
            expected_tool_class=required_tools[0] if len(required_tools) == 1 else None,
            global_safety_active=self.safety_supervisor.active,
            workspace_monitor_active=(
                self.workspace_supervisor.active and self.workspace_supervisor.monitor.active
            ),
            force_supervisor_active=self.force_supervisor.active,
            skill_uses_force=skill_uses_force,
            motion_profiles=self.motion_profiles,
            force_profiles=self.force_profiles,
            safety_policy=self.safety_policy,
            robot_backend=self.robot_backend,
            enable_real_robot=self.enable_real_robot,
            dry_run=self.dry_run,
            hardware_workspace_monitor_verified=bool(
                getattr(self.workspace_supervisor.monitor, "hardware_verified", False)
            ),
            hardware_scene_monitor_verified=bool(
                getattr(self.scene_monitor, "hardware_verified", False)
            ),
        )

    @staticmethod
    def validate_ik_and_collision(report: PreflightReport) -> None:
        names = {"ik", "joint_limits", "self_collision", "environment_collision"}
        checks = [check for check in report.checks if check.name in names]
        if len(checks) != len(names) or not all(check.passed for check in checks):
            raise PreflightError("IK/collision validation did not pass")

    def create_execution_context(
        self,
        skill: Any,
        scene: Any,
        bindings: Mapping[str, EntityBinding],
        *,
        command_text: str,
        execution_run_id: str | None,
    ) -> RuntimeContext:
        if self._skill_uses_force(skill):
            contact_links = tuple(
                str(link)
                for binding in bindings.values()
                if binding.entity_kind.value == "surface"
                for link in (_get(binding.entity, "allowed_contact_links", default=()) or ())
            )
            self.force_supervisor.authorize_contact_links(contact_links)
        return RuntimeContext(
            scene=scene,
            bindings=bindings,
            motion_profiles=self.motion_profiles,
            force_profiles=self.force_profiles,
            execution_mode=self.execution_mode,
            skill=skill,
            preflight_report=self.last_preflight,
            safety_policy=self.safety_policy,
            command_text=command_text,
            execution_run_id=execution_run_id,
        )

    async def execute_skill(
        self,
        skill: Any,
        context: RuntimeContext,
        *,
        compiled_run: Callable[[RuntimeExecutor], Awaitable[Any]] | None = None,
    ) -> None:
        executor = RuntimeExecutor(
            context=context,
            robot=self.robot,
            gripper=self.gripper,
            binder=self.binder,
            workspace_supervisor=self.workspace_supervisor,
            force_supervisor=self.force_supervisor,
            safety_supervisor=self.safety_supervisor,
            primitive_registry=self.primitive_registry,
            event_sink=self.event_sink,
            scene_monitor=self.scene_monitor,
        )
        if compiled_run is None:
            await executor.execute_graph(skill)
        else:
            await executor.execute_compiled(compiled_run)

    def monitor_execution(self, scene: Any) -> None:
        self.scene_monitor.assert_fresh(scene)
        self.safety_supervisor.assert_ready()
        self.force_supervisor.assert_released()

    def finalize_log(self, execution_run_id: str, *, error: BaseException | None) -> None:
        if self.storage_repository is None:
            return
        self.storage_repository.finish_execution(
            execution_run_id,
            status="succeeded" if error is None else "failed",
            ended_at_ns=self._clock_ns(),
            error_code=None if error is None else getattr(error, "code", "runtime_error"),
            error_message=None if error is None else str(error),
        )

    def _authorize_hardware_before_connect(self) -> None:
        if self.execution_mode is not ExecutionMode.HARDWARE:
            return
        if not (
            self.enable_hardware_execution
            and self.robot_backend == "doosan"
            and self.enable_real_robot
            and not self.dry_run
        ):
            raise HardwareExecutionDisabledError(
                "hardware requires mode=hardware, ENABLE_HARDWARE_EXECUTION=true, "
                "ROBOT_BACKEND=doosan, ENABLE_REAL_ROBOT=true, and DRY_RUN=false"
            )
        if not bool(
            getattr(self.workspace_supervisor.monitor, "hardware_verified", False)
        ) or not bool(getattr(self.scene_monitor, "hardware_verified", False)):
            raise HardwareExecutionDisabledError(
                "hardware requires verified dynamic obstacle and continuous scene monitors"
            )

    def _connect_adapters_if_needed(self) -> None:
        state = self.robot.get_state()
        if not bool(_get(state, "connected", default=False)):
            self.robot.connect()
        if self.gripper is not None:
            gripper_state = self.gripper.get_state()
            if not bool(_get(gripper_state, "connected", default=False)):
                self.gripper.connect()

    @staticmethod
    def _skill_uses_force(skill: Any) -> bool:
        if _get(skill, "force_profiles", default=()):
            return True
        return any(
            str(_get(node, "operation", default=""))
            in {"contact.enable_force", "contact.follow_path", "contact.verify_force"}
            for node in (_get(skill, "nodes", default=()) or ())
        )


Runtime = RuntimeOrchestrator

__all__ = ["Runtime", "RuntimeOrchestrator"]
