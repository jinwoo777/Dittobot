"""Fixed primitive interpreter used by generated ``run(runtime)`` functions and graphs."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import (
    ExecutionAbortedError,
    ForceSafetyError,
    ObstacleDetectedError,
    ProfileNotApprovedError,
    RuntimeSafetyError,
)
from robot_skill_system.runtime.event_log import EventSink, InMemoryEventSink
from robot_skill_system.runtime.force_supervisor import GlobalForceSupervisor
from robot_skill_system.runtime.models import EntityBinding, EntityKind, RuntimeContext
from robot_skill_system.runtime.safety_supervisor import GlobalSafetySupervisor
from robot_skill_system.runtime.scene_monitor import CameraSceneMonitor
from robot_skill_system.runtime.workspace_monitor import GlobalWorkspaceSupervisor

_SUPPORTED_OPERATIONS = frozenset(
    {
        "motion.move_j",
        "motion.rotate_joint_6_relative",
        "motion.move_l",
        "motion.move_c",
        "motion.move_spline",
        "motion.move_periodic",
        "motion.wait",
        "gripper.open",
        "gripper.close",
        "gripper.move_width",
        "gripper.verify_state",
        "grasp.verify_holding",
        "grasp.verify_released",
        "contact.search_surface",
        "contact.enable_force",
        "contact.follow_path",
        "contact.disable_force",
        "contact.verify_force",
        "workspace.validate_target",
        "workspace.validate_path",
        "workspace.wait_until_clear",
        "workspace.request_replan",
        "recovery.safe_stop",
        "recovery.safe_retract",
        "recovery.return_checkpoint",
        "recovery.abort",
    }
)

_FORBIDDEN_MODEL_NUMBERS = frozenset(
    {
        "velocity",
        "velocity_m_s",
        "velocity_rad_s",
        "acceleration",
        "acceleration_m_s2",
        "acceleration_rad_s2",
        "target_force",
        "target_force_vector",
        "target_force_vector_n",
        "maximum_force",
        "maximum_force_n",
        "stiffness",
        "stiffness_n_m",
    }
)


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


class RuntimeExecutor:
    """Execute only whitelisted primitives through explicit adapter calls.

    The method signature of :meth:`execute_primitive` intentionally matches deterministic
    compiler output. There is no dynamic import or user-selected callable lookup.
    """

    def __init__(
        self,
        *,
        context: RuntimeContext,
        robot: Any,
        binder: EntityBinder,
        workspace_supervisor: GlobalWorkspaceSupervisor,
        force_supervisor: GlobalForceSupervisor,
        safety_supervisor: GlobalSafetySupervisor,
        gripper: Any | None = None,
        primitive_registry: Any | None = None,
        event_sink: EventSink | None = None,
        scene_monitor: CameraSceneMonitor | None = None,
    ) -> None:
        self.context = context
        self.robot = robot
        self.gripper = gripper
        self.binder = binder
        self.workspace_supervisor = workspace_supervisor
        self.force_supervisor = force_supervisor
        self.safety_supervisor = safety_supervisor
        self.primitive_registry = primitive_registry
        self.event_sink = event_sink or InMemoryEventSink()
        self.scene_monitor = scene_monitor
        self.contact_search_succeeded = False
        self._checkpoint_poses: dict[str, Any] = {}

    async def execute_graph(self, graph: Any) -> None:
        """Interpret a validated graph by following result-conditioned edges.

        This path exists for local tests and integrations that have not persisted a compiled
        artifact yet.  It deliberately uses the same normalized arguments and transitions as
        the deterministic compiler; walking every node in topological order would execute both
        sides of a recovery branch.
        """

        from robot_skill_system.skills.graph import SkillGraphValidator

        report = SkillGraphValidator(self.primitive_registry).validate(graph)
        nodes = _get(graph, "nodes", default=())
        node_by_id = {str(_get(node, "node_id", "id")): node for node in nodes}
        current_node: str | None = str(_get(graph, "start_node"))
        self.workspace_supervisor.before_skill()
        self.event_sink.record("skill_execution_started")
        completed_normally = False
        try:
            while current_node is not None:
                node = node_by_id[current_node]
                operation = str(_get(node, "operation", "operation_name"))
                canonicalizer = getattr(
                    self.primitive_registry, "canonical_operation_name", None
                )
                if canonicalizer is not None:
                    operation = str(canonicalizer(operation))
                arguments = report.normalized_arguments[current_node]
                timeout_s = report.resolved_timeouts_s[current_node]
                checkpoint_value = _get(node, "checkpoint", "checkpoint_id", default=None)
                if isinstance(checkpoint_value, bool):
                    checkpoint = str(_get(node, "node_id", "id")) if checkpoint_value else None
                else:
                    checkpoint = None if checkpoint_value is None else str(checkpoint_value)
                primitive_ok = await self.execute_primitive(
                    operation, arguments, timeout_s, checkpoint
                )
                transition = report.transitions[current_node]
                next_node = transition.success if primitive_ok else transition.failure
                if not primitive_ok and next_node is None:
                    raise ExecutionAbortedError(
                        f"primitive {operation!r} failed without a recovery transition"
                    )
                current_node = next_node
            if self.force_supervisor.force_active:
                raise ForceSafetyError("skill ended while force mode was still active")
            completed_normally = True
            self.event_sink.record("skill_execution_completed")
        finally:
            # Safety cleanup is unconditional, including validation errors, timeouts, and adapter
            # failures. It is deliberately idempotent.
            retract_direction = (
                self.force_supervisor.current.surface_normal_xyz
                if not completed_normally and self.force_supervisor.current is not None
                else None
            )
            self.force_supervisor.disable()
            if retract_direction is not None:
                safe_retract = getattr(self.robot, "safe_retract", None)
                if safe_retract is None:
                    self.robot.stop(reason="force_cleanup_missing_safe_retract")
                    self.event_sink.record(
                        "force_cleanup_retract_unavailable", severity="error"
                    )
                else:
                    safe_retract(
                        direction_xyz=retract_direction,
                        distance_m=self.workspace_supervisor.retract_distance_m,
                    )
                    self.event_sink.record("force_cleanup_retracted")
            if not completed_normally:
                self.event_sink.record("skill_execution_cleanup", severity="warning")

    async def execute_compiled(self, run_callable: Any) -> None:
        """Run a trusted compiler-created callable inside the same global safety envelope.

        The callable is passed directly by local code; this method never loads a user-selected
        module or evaluates generated text.
        """

        if not callable(run_callable):
            raise TypeError("compiled skill entrypoint must be callable")
        self.workspace_supervisor.before_skill()
        self.event_sink.record("compiled_skill_execution_started")
        completed_normally = False
        try:
            outcome = await run_callable(self)
            if outcome is False:
                raise ExecutionAbortedError("compiled skill returned failure")
            if self.force_supervisor.force_active:
                raise ForceSafetyError("compiled skill ended while force mode was still active")
            completed_normally = True
            self.event_sink.record("compiled_skill_execution_completed")
        finally:
            retract_direction = (
                self.force_supervisor.current.surface_normal_xyz
                if not completed_normally and self.force_supervisor.current is not None
                else None
            )
            self.force_supervisor.disable()
            if retract_direction is not None:
                safe_retract = getattr(self.robot, "safe_retract", None)
                if safe_retract is None:
                    self.robot.stop(reason="force_cleanup_missing_safe_retract")
                    self.event_sink.record(
                        "force_cleanup_retract_unavailable", severity="error"
                    )
                else:
                    safe_retract(
                        direction_xyz=retract_direction,
                        distance_m=self.workspace_supervisor.retract_distance_m,
                    )
                    self.event_sink.record("force_cleanup_retracted")
            if not completed_normally:
                self.event_sink.record("compiled_skill_execution_cleanup", severity="warning")

    async def execute_primitive(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        timeout_s: float | None,
        checkpoint: str | None,
    ) -> bool:
        """Compiler-facing primitive entrypoint with the exact stable four-argument API."""

        if operation not in _SUPPORTED_OPERATIONS:
            raise RuntimeSafetyError(f"runtime refuses unknown primitive {operation!r}")
        validated_arguments = self._validate_arguments(operation, arguments)
        if any(name in validated_arguments for name in _FORBIDDEN_MODEL_NUMBERS):
            forbidden = sorted(
                name for name in validated_arguments if name in _FORBIDDEN_MODEL_NUMBERS
            )
            raise RuntimeSafetyError(
                "primitive may reference approved profiles only; "
                f"numeric controls present: {forbidden}"
            )
        bound_arguments = self.binder.bind_arguments(
            validated_arguments, scene=self.context.scene, bindings=self.context.bindings
        )
        self.safety_supervisor.assert_ready()
        self.event_sink.record(
            "primitive_started",
            {"operation": operation, "checkpoint": checkpoint, "timeout_s": timeout_s},
        )
        try:
            if self.scene_monitor is not None:
                try:
                    self.scene_monitor.assert_fresh(self.context.scene)
                except BaseException:
                    self.workspace_supervisor.on_scene_stale(self.robot)
                    raise
            self.workspace_supervisor.before_primitive(operation, checkpoint)
            coroutine = self._execute_operation(operation, bound_arguments)
            if timeout_s is None:
                await coroutine
            else:
                if timeout_s <= 0:
                    raise RuntimeSafetyError("primitive timeout must be positive")
                await asyncio.wait_for(coroutine, timeout=float(timeout_s))
            self.workspace_supervisor.during_primitive(operation)
            self.workspace_supervisor.after_primitive(operation)
            if checkpoint is not None:
                self._checkpoint_poses[checkpoint] = self.robot.get_current_pose()
            self.event_sink.record("primitive_completed", {"operation": operation})
            return True
        except ObstacleDetectedError as exc:
            self.event_sink.record(
                "workspace_obstacle",
                {"operation": operation, "error": str(exc)},
                severity="error",
            )
            self.workspace_supervisor.on_obstacle(self.robot, self.force_supervisor)
            raise
        except ForceSafetyError as exc:
            # A force violation is an unexpected contact condition.  Stop first;
            # the enclosing executor finally block owns release/compliance/retract.
            self.robot.stop(reason=f"force_safety:{type(exc).__name__}")
            self.event_sink.record(
                "force_safety_violation",
                {"operation": operation, "error": str(exc)},
                severity="error",
            )
            raise
        except BaseException as exc:
            self.event_sink.record(
                "primitive_failed",
                {"operation": operation, "error_type": type(exc).__name__, "error": str(exc)},
                severity="error",
            )
            raise

    async def _execute_operation(self, operation: str, arguments: Mapping[str, Any]) -> None:
        if operation == "motion.move_j":
            profile = self._motion_profile(arguments)
            self.robot.move_j(
                _get(arguments, "target_joint_positions_rad", "joint_positions_rad", "target"),
                velocity_rad_s=self._motion_limit(profile, "joint_velocity_rad_s"),
                acceleration_rad_s2=self._motion_limit(
                    profile, "joint_acceleration_rad_s2"
                ),
                blend_radius_m=self._profile_number(profile, "blend_radius_m", default=0.0),
            )
        elif operation == "motion.rotate_joint_6_relative":
            profile = self._motion_profile(arguments)
            current = tuple(float(value) for value in self.robot.get_joint_positions())
            if len(current) != 6 or any(not math.isfinite(value) for value in current):
                raise RuntimeSafetyError("robot did not return six finite joint positions")
            delta_rad = float(_get(arguments, "delta_rad"))
            target = (*current[:5], current[5] + delta_rad)
            if abs(target[5]) > 2.0 * math.pi:
                raise RuntimeSafetyError("relative J6 target exceeds the ±360 degree envelope")
            self.robot.move_j(
                target,
                velocity_rad_s=self._motion_limit(profile, "joint_velocity_rad_s"),
                acceleration_rad_s2=self._motion_limit(
                    profile, "joint_acceleration_rad_s2"
                ),
                blend_radius_m=self._profile_number(
                    profile, "blend_radius_m", default=0.0
                ),
            )
        elif operation == "motion.move_l":
            self._move_l(_get(arguments, "target_pose", "target"), arguments)
        elif operation == "motion.move_c":
            profile = self._motion_profile(arguments)
            via_pose = _get(arguments, "via_pose", "via")
            target_pose = _get(arguments, "target_pose", "target")
            current_pose = self.robot.get_current_pose()
            checked_path = (
                (current_pose, via_pose, target_pose)
                if current_pose is not None
                else (via_pose, target_pose)
            )
            self.workspace_supervisor.validate_path(
                path=checked_path, scene=self.context.scene
            )
            self._force_guarded_call(
                lambda: self.robot.move_c(
                    via_pose,
                    target_pose,
                    velocity_m_s=self._motion_limit(profile, "linear_velocity_m_s"),
                    acceleration_m_s2=self._motion_limit(
                        profile, "linear_acceleration_m_s2"
                    ),
                    blend_radius_m=self._profile_number(
                        profile, "blend_radius_m", default=0.0
                    ),
                )
            )
        elif operation == "motion.move_spline":
            poses = _get(arguments, "target_poses", "waypoints", "poses", default=())
            if not poses:
                raise RuntimeSafetyError("spline requires at least one pose")
            for pose in poses:
                self._move_l(pose, arguments)
        elif operation == "motion.move_periodic":
            profile = self._motion_profile(arguments)
            self._force_guarded_call(
                lambda: self.robot.move_periodic(
                    _get(arguments, "center", "center_pose"),
                    _get(arguments, "amplitudes_m", "amplitude_m"),
                    repetitions=int(_get(arguments, "repetitions")),
                    velocity_m_s=self._motion_limit(profile, "linear_velocity_m_s"),
                    acceleration_m_s2=self._motion_limit(
                        profile, "linear_acceleration_m_s2"
                    ),
                )
            )
        elif operation == "motion.wait":
            duration_s = float(_get(arguments, "duration_s"))
            if duration_s < 0:
                raise RuntimeSafetyError("wait duration cannot be negative")
            wait_method = getattr(self.robot, "wait", None)
            if wait_method is not None:
                wait_method(duration_s=duration_s)
            else:
                await asyncio.sleep(duration_s)
        elif operation == "gripper.open":
            self._require_gripper().open()
        elif operation == "gripper.close":
            self._require_gripper().close()
        elif operation == "gripper.move_width":
            self._require_gripper().move_width(float(_get(arguments, "width_m")))
        elif operation == "gripper.verify_state":
            expected = _text(_get(arguments, "expected_state", "state"))
            state = self._require_gripper().get_state()
            actual = "closed" if state.width_m < 0.01 else "open"
            if expected not in {actual, "holding" if state.is_holding else actual}:
                raise RuntimeSafetyError(
                    f"gripper state mismatch: expected={expected}, actual={actual}"
                )
        elif operation in {"grasp.verify_holding", "grasp.verify_released"}:
            self._verify_grasp(operation, arguments)
        elif operation == "contact.search_surface":
            self._search_surface(arguments)
        elif operation == "contact.enable_force":
            self._enable_force(arguments)
        elif operation == "contact.follow_path":
            if not self.force_supervisor.force_active:
                raise ForceSafetyError("contact path requires active force mode")
            for pose in _get(arguments, "target_poses", "path", "waypoints", default=()):
                self._move_l(pose, arguments)
        elif operation == "contact.disable_force":
            self.force_supervisor.disable()
        elif operation == "contact.verify_force":
            self.force_supervisor.monitor(
                expected_profile_id=_get(arguments, "force_profile_id", "profile_id")
            )
        elif operation == "workspace.validate_target":
            self.workspace_supervisor.validate_target(
                target=_get(arguments, "target", "target_pose"),
                scene=self.context.scene,
            )
        elif operation == "workspace.validate_path":
            self.workspace_supervisor.validate_path(
                path=_get(arguments, "path", "waypoints", "target_poses"),
                scene=self.context.scene,
            )
        elif operation == "workspace.wait_until_clear":
            observation = self.workspace_supervisor.monitor.poll()
            if observation is not None:
                self.workspace_supervisor.last_obstacle = observation
                raise ObstacleDetectedError(
                    f"workspace remains occupied by {observation.obstacle_id!r}"
                )
        elif operation == "workspace.request_replan":
            raise ExecutionAbortedError(
                "no verified planner is configured; remaining path cannot be replanned"
            )
        elif operation == "recovery.safe_stop":
            self.robot.stop(reason=str(_get(arguments, "reason", default="skill_recovery")))
        elif operation == "recovery.safe_retract":
            self._safe_retract(arguments)
        elif operation == "recovery.return_checkpoint":
            checkpoint = str(_get(arguments, "checkpoint", "checkpoint_id"))
            pose = self._checkpoint_poses.get(checkpoint)
            if pose is None:
                raise RuntimeSafetyError(f"unknown runtime checkpoint {checkpoint!r}")
            self._move_l(pose, arguments)
        elif operation == "recovery.abort":
            reason = str(
                _get(arguments, "reason", "reason_code", default="skill_requested_abort")
            )
            self.robot.stop(reason=reason)
            raise ExecutionAbortedError(reason)
        else:  # pragma: no cover - guarded by the whitelist above
            raise RuntimeSafetyError(f"unsupported primitive {operation!r}")

    def _move_l(self, target_pose: Any, arguments: Mapping[str, Any]) -> None:
        if target_pose is None:
            raise RuntimeSafetyError("linear motion target is missing")
        current_pose = self.robot.get_current_pose()
        if current_pose is None:
            self.workspace_supervisor.validate_target(
                target=target_pose, scene=self.context.scene
            )
        else:
            self.workspace_supervisor.validate_path(
                path=(current_pose, target_pose), scene=self.context.scene
            )
        profile = self._motion_profile(arguments)
        self._force_guarded_call(
            lambda: self.robot.move_l(
                target_pose,
                velocity_m_s=self._motion_limit(profile, "linear_velocity_m_s"),
                acceleration_m_s2=self._motion_limit(
                    profile, "linear_acceleration_m_s2"
                ),
                blend_radius_m=self._profile_number(
                    profile, "blend_radius_m", default=0.0
                ),
            )
        )

    def _force_guarded_call(self, adapter_call: Any) -> None:
        """Poll force immediately before and after one blocking adapter call.

        Vendor motion calls in the current adapter protocol are blocking.  These
        two checks deliberately do not claim continuous polling while the call is
        in progress; hardware backends need a separate safety-rated streaming
        monitor or interruptible motion API.
        """

        if self.force_supervisor.force_active:
            self.force_supervisor.monitor(require_contact=True)
        adapter_call()
        if self.force_supervisor.force_active:
            self.force_supervisor.monitor(require_contact=True)

    def _enable_force(self, arguments: Mapping[str, Any]) -> None:
        profile_id = str(_get(arguments, "force_profile_id", "profile_id"))
        tool = self._bound_entity(
            _get(arguments, "tool_id", "tool", default="$tool"), EntityKind.TOOL
        )
        if not bool(_get(tool.entity, "attached", default=False)):
            raise ForceSafetyError("force control requires the verified tool to be attached")
        surface = self._bound_entity(
            _get(arguments, "surface_id", "surface", "target_surface_id", default="$surface"),
            EntityKind.SURFACE,
        )
        normal_value = _get(surface.entity, "normal", "normal_xyz")
        if normal_value is None:
            raise ForceSafetyError("contact surface has no trusted normal")
        normal = self._xyz(normal_value)
        allowed_links = tuple(_get(surface.entity, "allowed_contact_links", default=()) or ())
        tool_tcp_frame = _get(tool.entity, "tcp_frame")
        if not allowed_links:
            raise ForceSafetyError("contact surface declares no allowed contact link")
        if not isinstance(tool_tcp_frame, str) or tool_tcp_frame not in allowed_links:
            raise ForceSafetyError(
                "attached tool TCP is not explicitly allowed for the contact surface"
            )
        contact_link = tool_tcp_frame
        self.force_supervisor.enable(
            profile_id,
            tool_class=str(_get(tool.entity, "tool_class", "class_name")),
            surface_role=_text(_get(surface.entity, "role")),
            target_surface_id=surface.entity_id,
            surface_normal_xyz=normal,
            contact_link=contact_link,
            contact_search_succeeded=self.contact_search_succeeded,
            ik_passed=self._preflight_geometry_passed(),
        )

    def _preflight_geometry_passed(self) -> bool:
        report = self.context.preflight_report
        if report is None or not report.passed:
            return False
        required = {"ik", "joint_limits", "singularity_margin"}
        checks = {
            check.name: check
            for check in report.checks
            if check.name in required
        }
        return set(checks) == required and all(check.passed for check in checks.values())

    def _search_surface(self, arguments: Mapping[str, Any]) -> None:
        profile_id = str(_get(arguments, "force_profile_id", "profile_id"))
        profile = self.context.force_profiles.get(profile_id)
        if profile is None:
            raise ProfileNotApprovedError(f"force profile {profile_id!r} is not approved")
        surface = self._bound_entity(
            _get(arguments, "surface_id", "surface", default="$surface"),
            EntityKind.SURFACE,
        )
        if _text(_get(surface.entity, "role")) != "contact_target":
            raise ForceSafetyError("contact search requires a contact_target surface")
        search = getattr(self.robot, "search_surface", None)
        if search is None:
            raise RuntimeSafetyError("adapter has no verified contact-search capability")
        self.contact_search_succeeded = bool(
            search(
                surface_id=surface.entity_id,
                contact_search_speed_m_s=self._profile_number(
                    profile, "contact_search_speed_m_s"
                ),
                maximum_search_distance_m=self._profile_number(
                    profile, "maximum_search_distance_m"
                ),
            )
        )
        if not self.contact_search_succeeded:
            raise ForceSafetyError("contact search did not find the requested surface")

    def _safe_retract(self, arguments: Mapping[str, Any]) -> None:
        safe_retract = getattr(self.robot, "safe_retract", None)
        if safe_retract is None:
            raise RuntimeSafetyError("robot adapter has no verified safe_retract operation")
        surface_reference = _get(
            arguments, "surface_id", "surface", "target_surface_id", default="$surface"
        )
        surface = self._bound_entity(surface_reference, EntityKind.SURFACE)
        normal = self._xyz(_get(surface.entity, "normal", "normal_xyz"))
        distance_m = float(_get(arguments, "distance_m", default=0.05))
        if not 0.0 < distance_m <= 0.25:
            raise RuntimeSafetyError("safe retract distance is outside the globally bounded range")
        norm = math.sqrt(sum(component * component for component in normal))
        if norm <= 1e-12:
            raise RuntimeSafetyError("surface normal is zero")
        safe_retract(
            direction_xyz=tuple(component / norm for component in normal), distance_m=distance_m
        )

    def _verify_grasp(self, operation: str, arguments: Mapping[str, Any]) -> None:
        """Interpret gripper evidence through an approved local profile.

        Binding replaces ``$object``/``$gripper`` before this method runs, but
        :meth:`_bound_entity` deliberately accepts either placeholders or exact
        bound IDs.  No geometry or thresholds come from the primitive arguments.
        """

        if self.force_supervisor.force_active:
            raise ForceSafetyError("grasp verification requires force mode to be disabled")
        profile_id = str(_get(arguments, "verification_profile_id"))
        profile = self.context.verification_profiles.get(profile_id)
        if profile is None:
            raise ProfileNotApprovedError(
                f"grasp verification profile {profile_id!r} is not approved"
            )
        configured_profile_id = _get(profile, "profile_id")
        if configured_profile_id is not None and str(configured_profile_id) != profile_id:
            raise ProfileNotApprovedError(
                "grasp verification profile registry key does not match profile_id"
            )
        object_binding = self._bound_entity(
            _get(arguments, "object", "object_id", default="$object"), EntityKind.OBJECT
        )
        tool_binding = self._bound_entity(
            _get(arguments, "tool", "tool_id", default="$gripper"), EntityKind.TOOL
        )
        state = self._require_gripper().get_state()
        if not bool(_get(state, "connected", default=False)):
            raise RuntimeSafetyError("grasp verification requires a connected gripper")

        details = {
            "object_id": object_binding.entity_id,
            "tool_id": tool_binding.entity_id,
            "verification_profile_id": profile_id,
        }
        recorded_tool_id = self.context.attachment_states.get(object_binding.entity_id)
        if recorded_tool_id is not None and recorded_tool_id != tool_binding.entity_id:
            raise RuntimeSafetyError(
                f"object {object_binding.entity_id!r} is tracked by a different tool"
            )
        if operation == "grasp.verify_holding":
            if not bool(_get(profile, "require_holding_signal", default=True)):
                raise ProfileNotApprovedError(
                    "current runtime supports only profiles requiring a holding signal"
                )
            if not bool(_get(state, "is_holding", default=False)):
                raise RuntimeSafetyError(
                    f"grasp holding verification failed for {object_binding.entity_id!r}"
                )
            self.context.attachment_states[object_binding.entity_id] = tool_binding.entity_id
            self.event_sink.record("grasp_holding_verified", details)
            return

        minimum_width_m = self._profile_number(profile, "minimum_released_width_m")
        observed_width_m = float(_get(state, "width_m"))
        if observed_width_m < minimum_width_m:
            raise RuntimeSafetyError(
                "grasp release verification failed: observed gripper width is below "
                "the approved profile threshold"
            )
        self.context.attachment_states.pop(object_binding.entity_id, None)
        self.event_sink.record("grasp_release_verified", details)

    def _bound_entity(self, reference: Any, kind: EntityKind) -> EntityBinding:
        if isinstance(reference, str) and reference in self.context.bindings:
            binding = self.context.bindings[reference]
            if binding.entity_kind is not kind:
                raise RuntimeSafetyError(f"{reference!r} is not a {kind.value} binding")
            return binding
        reference_id = str(reference)
        for binding in self.context.bindings.values():
            if binding.entity_kind is kind and binding.entity_id == reference_id:
                return binding
        raise RuntimeSafetyError(f"{kind.value} binding {reference_id!r} is unavailable")

    def _motion_profile(self, arguments: Mapping[str, Any]) -> Any:
        profile_id = _get(arguments, "motion_profile_id", "profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ProfileNotApprovedError("motion primitive requires an approved profile ID")
        profile = self.context.motion_profiles.get(profile_id)
        if profile is None:
            raise ProfileNotApprovedError(f"motion profile {profile_id!r} is not approved")
        return profile

    @staticmethod
    def _profile_number(profile: Any, name: str, *, default: float | None = None) -> float:
        value = _get(profile, name, default=default)
        if value is None:
            raise ProfileNotApprovedError(f"approved profile is missing {name}")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ProfileNotApprovedError(f"approved profile has invalid {name}")
        return number

    def _motion_limit(self, profile: Any, name: str) -> float:
        value = self._profile_number(profile, name)
        safety_scale = self._profile_number(profile, "safety_scale", default=1.0)
        if not 0.0 < safety_scale <= 1.0:
            raise ProfileNotApprovedError("approved motion profile has invalid safety_scale")
        effective = value * safety_scale
        cap_name = {
            "linear_velocity_m_s": "maximum_linear_velocity_m_s",
            "linear_acceleration_m_s2": "maximum_linear_acceleration_m_s2",
            "joint_velocity_rad_s": "maximum_joint_velocity_rad_s",
            "joint_acceleration_rad_s2": "maximum_joint_acceleration_rad_s2",
        }.get(name)
        cap = _get(self.context.safety_policy, cap_name) if cap_name is not None else None
        if cap is not None and effective > float(cap) + 1e-12:
            raise ProfileNotApprovedError(
                f"approved motion profile {name}={effective:.6g} exceeds "
                f"loaded safety policy {cap_name}={float(cap):.6g}"
            )
        return effective

    def _validate_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(arguments, Mapping):
            if hasattr(arguments, "model_dump"):
                arguments = arguments.model_dump(mode="python")
            else:
                raise RuntimeSafetyError("primitive arguments must be a mapping")
        if self.primitive_registry is None:
            return arguments
        validator = getattr(self.primitive_registry, "validate_arguments", None)
        if validator is None:
            raise RuntimeSafetyError("primitive registry has no validate_arguments API")
        validated = validator(operation, arguments)
        if validated is None:
            return arguments
        if hasattr(validated, "model_dump"):
            dumped = validated.model_dump(mode="python")
            if isinstance(dumped, Mapping):
                return dumped
            raise RuntimeSafetyError("primitive registry dumped non-mapping arguments")
        if isinstance(validated, Mapping):
            return validated
        raise RuntimeSafetyError("primitive registry returned non-mapping arguments")

    @staticmethod
    def _xyz(value: Any) -> tuple[float, float, float]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 3:
                raise RuntimeSafetyError("expected a three-dimensional vector")
            return float(value[0]), float(value[1]), float(value[2])
        return float(_get(value, "x")), float(_get(value, "y")), float(_get(value, "z"))

    def _require_gripper(self) -> Any:
        if self.gripper is None:
            raise RuntimeSafetyError("skill requires a configured gripper adapter")
        return self.gripper

    @staticmethod
    def _ordered_nodes(graph: Any) -> Sequence[Any]:
        ordered_nodes = getattr(graph, "ordered_nodes", None)
        if callable(ordered_nodes):
            return tuple(ordered_nodes())
        topological_order = getattr(graph, "topological_order", None)
        nodes = _get(graph, "nodes", default=())
        if callable(topological_order):
            ordered = tuple(topological_order())
            if ordered and isinstance(ordered[0], str):
                by_id = {str(_get(node, "node_id", "id")): node for node in nodes}
                return tuple(by_id[node_id] for node_id in ordered)
            return ordered
        return tuple(nodes)


PrimitiveExecutor = RuntimeExecutor

__all__ = ["PrimitiveExecutor", "RuntimeExecutor"]
