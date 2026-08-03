"""Fail-closed preflight checks, including explicit mock geometry labels."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from robot_skill_system.runtime.binder import EntityBinder
from robot_skill_system.runtime.errors import PreflightError, SceneStaleError
from robot_skill_system.runtime.geometry import (
    DeterministicSceneGeometry,
    GeometryAssessment,
)
from robot_skill_system.runtime.models import (
    EntityBinding,
    EntityKind,
    ExecutionMode,
    PreflightReport,
    ValidationCheck,
)


class GeometryValidator(Protocol):
    @property
    def is_mock(self) -> bool: ...

    def validate(
        self, *, skill: Any, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> Sequence[ValidationCheck]: ...


class MockGeometryValidator:
    """Deterministic TCP geometry gate that remains explicitly mock evidence.

    Robot-link IK, joint-limit, self-collision, and singularity results remain mock
    placeholders.  Unlike those unavailable checks, explicit anchor-relative TCP
    targets and paths are deterministically bound and checked against the current
    scene.  A malformed/unsupported relevant path fails the environment gate.
    """

    is_mock = True

    def __init__(self, *, minimum_clearance_m: float = 0.03) -> None:
        self.geometry = DeterministicSceneGeometry(
            minimum_clearance_m=minimum_clearance_m
        )
        self._binder = EntityBinder()

    def validate_target(self, *, target: Any, scene: Any) -> GeometryAssessment:
        return self.geometry.validate_target(target=target, scene=scene)

    def validate_path(self, *, path: Any, scene: Any) -> GeometryAssessment:
        return self.geometry.validate_path(path=path, scene=scene)

    def validate(
        self, *, skill: Any, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> Sequence[ValidationCheck]:
        unavailable_detail = "offline mock result; not hardware or MoveIt validation"
        checks = [
            ValidationCheck(name, True, "mock_geometry", unavailable_detail, is_mock=True)
            for name in (
                "start_pose",
                "ik",
                "joint_limits",
                "self_collision",
                "singularity_margin",
            )
        ]
        assessments = self._assess_skill_paths(skill=skill, scene=scene, bindings=bindings)
        failed = next((assessment for assessment in assessments if not assessment.passed), None)
        if failed is not None:
            passed = False
            detail = failed.detail
        elif assessments:
            passed = True
            detail = "; ".join(assessment.detail for assessment in assessments)
        else:
            passed = True
            detail = "skill has no explicit Cartesian TCP target/path to assess"
        checks.extend(
            (
                ValidationCheck(
                    "environment_collision",
                    passed,
                    "deterministic_mock_geometry",
                    detail,
                    is_mock=True,
                ),
                ValidationCheck(
                    "minimum_clearance",
                    passed,
                    "deterministic_mock_geometry",
                    detail,
                    is_mock=True,
                ),
            )
        )
        return tuple(checks)

    def _assess_skill_paths(
        self, *, skill: Any, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> tuple[GeometryAssessment, ...]:
        assessments: list[GeometryAssessment] = []
        previous_cartesian_target: Any | None = None
        for node in _get(skill, "nodes", default=()) or ():
            operation = str(_get(node, "operation", "operation_name", default=""))
            raw_arguments = _get(node, "arguments", "parameters", default={})
            if not isinstance(raw_arguments, Mapping):
                if self._is_geometry_relevant(operation):
                    assessments.append(
                        GeometryAssessment(
                            False,
                            f"{operation or 'primitive'} path cannot be checked: "
                            "arguments are not a mapping",
                        )
                    )
                continue
            try:
                arguments = self._binder.bind_arguments(
                    raw_arguments, scene=scene, bindings=bindings
                )
            except Exception as exc:
                if self._is_geometry_relevant(operation):
                    assessments.append(
                        GeometryAssessment(
                            False,
                            f"{operation} path cannot be bound for checking: {exc}",
                        )
                    )
                continue
            if operation in {
                "motion.move_j",
                "workspace.validate_target",
                "workspace.check_target",
                "workspace.request_replan",
            }:
                target = _get(arguments, "target", "target_pose")
                assessments.append(self.geometry.validate_target(target=target, scene=scene))
                if operation == "motion.move_j":
                    previous_cartesian_target = None
            elif operation == "motion.move_l":
                target = _get(arguments, "target", "target_pose")
                if previous_cartesian_target is None:
                    assessments.append(
                        self.geometry.validate_target(target=target, scene=scene)
                    )
                else:
                    assessments.append(
                        self.geometry.validate_path(
                            path=(previous_cartesian_target, target), scene=scene
                        )
                    )
                previous_cartesian_target = target
            elif operation in {"motion.move_c"}:
                via = _get(arguments, "via", "via_pose")
                target = _get(arguments, "target", "target_pose")
                path = (
                    (previous_cartesian_target, via, target)
                    if previous_cartesian_target is not None
                    else (via, target)
                )
                assessments.append(self.geometry.validate_path(path=path, scene=scene))
                previous_cartesian_target = target
            elif operation in {
                "motion.move_spline",
                "contact.follow_path",
                "workspace.validate_path",
                "workspace.check_path",
            }:
                path = _get(
                    arguments,
                    "path",
                    "waypoints",
                    "target_poses",
                    "poses",
                )
                checked_path = path
                if (
                    operation in {"motion.move_spline", "contact.follow_path"}
                    and previous_cartesian_target is not None
                    and isinstance(path, Sequence)
                    and not isinstance(path, (str, bytes))
                ):
                    checked_path = (previous_cartesian_target, *path)
                assessments.append(
                    self.geometry.validate_path(path=checked_path, scene=scene)
                )
                if (
                    operation in {"motion.move_spline", "contact.follow_path"}
                    and isinstance(path, Sequence)
                    and not isinstance(path, (str, bytes))
                    and path
                ):
                    previous_cartesian_target = path[-1]
            elif operation == "motion.move_periodic":
                center = _get(arguments, "center", "center_pose")
                assessments.append(
                    self.geometry.validate_periodic_envelope(
                        center=center,
                        amplitude_m=_get(arguments, "amplitude_m", "amplitudes_m"),
                        scene=scene,
                    )
                )
                previous_cartesian_target = center
        return tuple(assessments)

    @staticmethod
    def _is_geometry_relevant(operation: str) -> bool:
        return operation in {
            "motion.move_j",
            "motion.move_l",
            "motion.move_c",
            "motion.move_spline",
            "motion.move_periodic",
            "contact.follow_path",
            "workspace.validate_target",
            "workspace.validate_path",
            "workspace.check_target",
            "workspace.check_path",
            "workspace.request_replan",
        }


@dataclass(frozen=True, slots=True)
class PreflightPolicy:
    maximum_scene_age_ms: int = 5_000
    minimum_clearance_m: float = 0.03
    minimum_object_confidence: float = 0.7
    minimum_tool_confidence: float = 0.8
    minimum_visible_fraction: float = 0.5
    require_validated_skill: bool = True


@dataclass(frozen=True, slots=True)
class HardwareInterlocks:
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    enable_hardware_execution: bool = False


class PreflightValidator:
    def __init__(
        self,
        *,
        policy: PreflightPolicy | None = None,
        geometry_validator: GeometryValidator | None = None,
        clock_ns: Any = time.time_ns,
    ) -> None:
        self.policy = policy or PreflightPolicy()
        self.geometry_validator = geometry_validator or MockGeometryValidator(
            minimum_clearance_m=self.policy.minimum_clearance_m
        )
        self._binder = EntityBinder(clock_ns=clock_ns)

    def validate(
        self,
        *,
        scene: Any,
        skill: Any,
        bindings: Mapping[str, EntityBinding],
        robot: Any,
        execution_mode: ExecutionMode | str,
        enable_hardware_execution: bool = False,
        skill_validation_status: str = "passed",
        expected_calibration_id: str | None = None,
        expected_tool_id: str | None = None,
        expected_tool_class: str | None = None,
        global_safety_active: bool = True,
        workspace_monitor_active: bool = True,
        force_supervisor_active: bool = True,
        skill_uses_force: bool = False,
        motion_profiles: Mapping[str, Any] | None = None,
        force_profiles: Mapping[str, Any] | None = None,
        safety_policy: Any | None = None,
        robot_backend: str = "doosan",
        enable_real_robot: bool = True,
        dry_run: bool = False,
        hardware_workspace_monitor_verified: bool = False,
        hardware_scene_monitor_verified: bool = False,
        raise_on_failure: bool = True,
    ) -> PreflightReport:
        mode = (
            execution_mode
            if isinstance(execution_mode, ExecutionMode)
            else ExecutionMode(execution_mode)
        )
        checks: list[ValidationCheck] = []
        checks.append(self._scene_freshness_check(scene))
        actual_calibration = _get(scene, "calibration_id")
        calibration_passed = (
            bool(actual_calibration)
            and (expected_calibration_id is None or actual_calibration == expected_calibration_id)
        )
        checks.append(
            ValidationCheck(
                "calibration",
                calibration_passed,
                "scene",
                "calibration ID present and matches requirement"
                if calibration_passed
                else "calibration ID missing or mismatched",
            )
        )
        checks.extend(self._binding_checks(bindings, expected_tool_id, expected_tool_class))
        skill_passed = not self.policy.require_validated_skill or skill_validation_status in {
            "passed",
            "validated",
            "active",
        }
        checks.append(
            ValidationCheck(
                "skill_validation",
                skill_passed,
                "registry",
                f"validation_status={skill_validation_status}",
            )
        )
        checks.extend(
            (
                ValidationCheck(
                    "global_safety_supervisor", global_safety_active, "runtime_supervisor"
                ),
                ValidationCheck(
                    "workspace_monitor", workspace_monitor_active, "runtime_supervisor"
                ),
                ValidationCheck(
                    "force_supervisor",
                    force_supervisor_active or not skill_uses_force,
                    "runtime_supervisor",
                    "required by contact skill" if skill_uses_force else "not required by skill",
                ),
            )
        )
        state = robot.get_state()
        checks.extend(
            (
                ValidationCheck(
                    "robot_connected",
                    bool(_get(state, "connected", default=False)),
                    "robot",
                ),
                ValidationCheck(
                    "emergency_stop",
                    not bool(robot.is_emergency_stop_active()),
                    "robot",
                    "E-stop inactive" if not robot.is_emergency_stop_active() else "E-stop active",
                ),
                ValidationCheck(
                    "protective_stop",
                    not bool(_get(state, "protective_stop_active", default=False)),
                    "robot",
                ),
                ValidationCheck(
                    "robot_fault",
                    _get(state, "fault_code") is None,
                    "robot",
                    str(_get(state, "fault_code", default="")),
                ),
            )
        )
        if mode is ExecutionMode.HARDWARE:
            checks.extend(
                (
                    ValidationCheck("hardware_mode_flag", True, "environment"),
                    ValidationCheck(
                        "hardware_enable_flag", enable_hardware_execution, "environment"
                    ),
                    ValidationCheck(
                        "hardware_backend",
                        robot_backend == "doosan",
                        "environment",
                        f"robot_backend={robot_backend}",
                    ),
                    ValidationCheck(
                        "enable_real_robot", enable_real_robot, "environment"
                    ),
                    ValidationCheck("dry_run_disabled", not dry_run, "environment"),
                    ValidationCheck(
                        "hardware_geometry_validator",
                        not self.geometry_validator.is_mock,
                        "geometry_validator",
                        "mock IK/collision evidence is forbidden for hardware execution",
                    ),
                    ValidationCheck(
                        "hardware_workspace_monitor",
                        hardware_workspace_monitor_verified,
                        "runtime_supervisor",
                        "hardware execution requires a verified dynamic obstacle monitor",
                    ),
                    ValidationCheck(
                        "hardware_scene_monitor",
                        hardware_scene_monitor_verified,
                        "runtime_supervisor",
                        "hardware execution requires a verified continuous camera/scene monitor",
                    ),
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    "hardware_disabled",
                    True,
                    "environment",
                    f"execution_mode={mode.value}",
                )
            )
        checks.extend(self.geometry_validator.validate(skill=skill, scene=scene, bindings=bindings))
        checks.extend(
            self._profile_checks(
                skill,
                bindings,
                motion_profiles or {},
                force_profiles or {},
                skill_uses_force=skill_uses_force,
            )
        )
        checks.extend(
            self._safety_policy_checks(
                skill,
                motion_profiles or {},
                force_profiles or {},
                safety_policy,
            )
        )
        report = PreflightReport(tuple(checks))
        if raise_on_failure and not report.passed:
            failures = ", ".join(
                f"{check.name}: {check.detail or 'failed'}" for check in report.failed_checks
            )
            raise PreflightError(f"preflight rejected execution: {failures}")
        return report

    def _scene_freshness_check(self, scene: Any) -> ValidationCheck:
        try:
            self._binder.ensure_scene_fresh(
                scene, maximum_age_ms=self.policy.maximum_scene_age_ms
            )
        except SceneStaleError as exc:
            return ValidationCheck("scene_freshness", False, "scene", str(exc))
        return ValidationCheck("scene_freshness", True, "scene")

    def _binding_checks(
        self,
        bindings: Mapping[str, EntityBinding],
        expected_tool_id: str | None,
        expected_tool_class: str | None,
    ) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        tool_bindings = [
            binding for binding in bindings.values() if binding.entity_kind is EntityKind.TOOL
        ]
        for placeholder, binding in bindings.items():
            threshold = (
                self.policy.minimum_tool_confidence
                if binding.entity_kind is EntityKind.TOOL
                else self.policy.minimum_object_confidence
            )
            checks.append(
                ValidationCheck(
                    f"binding_confidence:{placeholder}",
                    binding.confidence >= threshold,
                    "scene",
                    f"confidence={binding.confidence:.3f}, minimum={threshold:.3f}",
                )
            )
            visible = float(_get(binding.entity, "visible_fraction", default=1.0))
            checks.append(
                ValidationCheck(
                    f"visible_fraction:{placeholder}",
                    visible >= self.policy.minimum_visible_fraction,
                    "scene",
                    f"visible_fraction={visible:.3f}",
                )
            )
        if expected_tool_id is not None:
            actual_ids = {binding.entity_id for binding in tool_bindings}
            checks.append(
                ValidationCheck(
                    "tool_identity",
                    expected_tool_id in actual_ids,
                    "scene",
                    f"expected={expected_tool_id}, actual={sorted(actual_ids)}",
                )
            )
        if expected_tool_class is not None:
            actual_classes = {
                str(_get(binding.entity, "tool_class", "class_name"))
                for binding in tool_bindings
            }
            checks.append(
                ValidationCheck(
                    "tool_class",
                    expected_tool_class in actual_classes,
                    "scene",
                    f"expected={expected_tool_class}, actual={sorted(actual_classes)}",
                )
            )
        return checks

    def _profile_checks(
        self,
        skill: Any,
        bindings: Mapping[str, EntityBinding],
        motion_profiles: Mapping[str, Any],
        force_profiles: Mapping[str, Any],
        *,
        skill_uses_force: bool,
    ) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        required_motion = set(_get(skill, "motion_profiles", default=()) or ())
        required_force = set(_get(skill, "force_profiles", default=()) or ())
        for node in _get(skill, "nodes", default=()) or ():
            operation = str(_get(node, "operation", default=""))
            arguments = _get(node, "arguments", "parameters", default={})
            motion_id = _get(arguments, "motion_profile_id", "profile_id")
            force_id = _get(arguments, "force_profile_id")
            if (
                operation.startswith("motion.") or operation == "contact.follow_path"
            ) and isinstance(motion_id, str):
                required_motion.add(motion_id)
            if operation.startswith("contact.") and isinstance(force_id, str):
                required_force.add(force_id)
        missing_motion = sorted(required_motion - set(motion_profiles))
        missing_force = sorted(required_force - set(force_profiles))
        checks.append(
            ValidationCheck(
                "motion_profiles_loaded",
                not missing_motion,
                "approved_configuration",
                f"missing={missing_motion}" if missing_motion else "all required profiles loaded",
            )
        )
        checks.append(
            ValidationCheck(
                "force_profiles_loaded",
                not missing_force and (bool(required_force) or not skill_uses_force),
                "approved_configuration",
                (
                    f"missing={missing_force}"
                    if missing_force
                    else "contact skill declares no force profile"
                    if skill_uses_force and not required_force
                    else "all required profiles loaded"
                ),
            )
        )
        if not skill_uses_force:
            return checks
        surfaces = [
            binding for binding in bindings.values() if binding.entity_kind is EntityKind.SURFACE
        ]
        contact_surfaces = [
            binding
            for binding in surfaces
            if _enum_text(_get(binding.entity, "role")) == "contact_target"
        ]
        checks.append(
            ValidationCheck(
                "contact_target",
                bool(contact_surfaces),
                "scene",
                "force control requires a bound contact_target surface",
            )
        )
        tools = [
            binding for binding in bindings.values() if binding.entity_kind is EntityKind.TOOL
        ]
        contact_links_ok = bool(contact_surfaces) and bool(tools) and all(
            bool(_get(tool.entity, "attached", default=False))
            and isinstance(_get(tool.entity, "tcp_frame"), str)
            and _get(tool.entity, "tcp_frame")
            in {
                str(link)
                for surface in contact_surfaces
                for link in (
                    _get(surface.entity, "allowed_contact_links", default=()) or ()
                )
            }
            for tool in tools
        )
        checks.append(
            ValidationCheck(
                "contact_link_authorized",
                contact_links_ok,
                "scene",
                "attached tool TCP is explicitly allowed by the contact surface"
                if contact_links_ok
                else "contact surface has no explicit allowance for the attached tool TCP",
            )
        )
        for profile_id in sorted(required_force & set(force_profiles)):
            profile = force_profiles[profile_id]
            allowed_roles = {
                _enum_text(value)
                for value in _get(profile, "allowed_surface_roles", default=())
            }
            allowed_tools = {
                str(value) for value in _get(profile, "allowed_tool_classes", default=())
            }
            roles_ok = bool(contact_surfaces) and all(
                _enum_text(_get(binding.entity, "role")) in allowed_roles
                for binding in contact_surfaces
            )
            tools_ok = bool(tools) and all(
                str(_get(binding.entity, "tool_class", "class_name")) in allowed_tools
                for binding in tools
            )
            checks.append(
                ValidationCheck(
                    f"force_profile_compatibility:{profile_id}",
                    roles_ok and tools_ok,
                    "approved_configuration",
                    f"surface_roles_ok={roles_ok}, tool_classes_ok={tools_ok}",
                )
            )
        return checks

    def _safety_policy_checks(
        self,
        skill: Any,
        motion_profiles: Mapping[str, Any],
        force_profiles: Mapping[str, Any],
        safety_policy: Any | None,
    ) -> list[ValidationCheck]:
        """Reject approved profiles that exceed a loaded global policy."""

        if safety_policy is None:
            return []
        required_motion = set(_get(skill, "motion_profiles", default=()) or ())
        required_force = set(_get(skill, "force_profiles", default=()) or ())
        policy_pairs = {
            "linear_velocity_m_s": "maximum_linear_velocity_m_s",
            "linear_acceleration_m_s2": "maximum_linear_acceleration_m_s2",
            "joint_velocity_rad_s": "maximum_joint_velocity_rad_s",
            "joint_acceleration_rad_s2": "maximum_joint_acceleration_rad_s2",
        }
        violations: list[str] = []
        for profile_id in sorted(required_motion & set(motion_profiles)):
            profile = motion_profiles[profile_id]
            scale = float(_get(profile, "safety_scale", default=1.0))
            for profile_field, policy_field in policy_pairs.items():
                value = _get(profile, profile_field)
                cap = _get(safety_policy, policy_field)
                if value is not None and cap is not None and float(value) * scale > float(cap):
                    violations.append(
                        f"{profile_id}.{profile_field}={float(value) * scale:.6g}>"
                        f"{policy_field}={float(cap):.6g}"
                    )
        maximum_force = _get(safety_policy, "maximum_force_n")
        for profile_id in sorted(required_force & set(force_profiles)):
            profile_force = _get(force_profiles[profile_id], "maximum_force_n")
            if (
                maximum_force is not None
                and profile_force is not None
                and float(profile_force) > float(maximum_force)
            ):
                violations.append(
                    f"{profile_id}.maximum_force_n={float(profile_force):.6g}>"
                    f"maximum_force_n={float(maximum_force):.6g}"
                )
        policy_flags_ok = (
            bool(_get(safety_policy, "unknown_space_is_occupied", default=False))
            and bool(_get(safety_policy, "emergency_stop_required", default=False))
            and bool(_get(safety_policy, "workspace_monitor_required", default=False))
            and bool(_get(safety_policy, "force_supervisor_required", default=False))
        )
        checks = [
            ValidationCheck(
                "safety_policy_caps",
                not violations,
                "approved_configuration",
                "; ".join(violations) if violations else "required profiles satisfy global caps",
            ),
            ValidationCheck(
                "safety_policy_fail_closed",
                policy_flags_ok,
                "approved_configuration",
                "unknown-space and global supervisor requirements are enabled"
                if policy_flags_ok
                else "loaded safety policy disables a required fail-closed control",
            ),
        ]
        return checks


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _enum_text(value: Any) -> str:
    return str(value.value) if hasattr(value, "value") else str(value)


MockIKCollisionValidator = MockGeometryValidator

__all__ = [
    "GeometryValidator",
    "HardwareInterlocks",
    "MockGeometryValidator",
    "MockIKCollisionValidator",
    "PreflightPolicy",
    "PreflightValidator",
]
