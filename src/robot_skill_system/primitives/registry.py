"""Closed primitive whitelist and operation-specific argument validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from robot_skill_system.exceptions import PrimitiveValidationError, UnknownPrimitiveError
from robot_skill_system.primitives.models import (
    ALL_SKILL_TYPES,
    ContactFollowPathArguments,
    ContactSearchArguments,
    DisableForceArguments,
    EnableForceArguments,
    GraspVerifyHoldingArguments,
    GraspVerifyReleasedArguments,
    GripperCloseArguments,
    GripperMoveWidthArguments,
    GripperOpenArguments,
    GripperVerifyStateArguments,
    MoveCArguments,
    MoveJArguments,
    MoveLArguments,
    MovePeriodicArguments,
    MoveSplineArguments,
    PrimitiveDefinition,
    PrimitiveMetadata,
    RecoveryAbortArguments,
    RecoveryReturnCheckpointArguments,
    RecoverySafeRetractArguments,
    RecoverySafeStopArguments,
    RotateJoint6RelativeArguments,
    VerifyForceArguments,
    WaitArguments,
    WorkspaceRequestReplanArguments,
    WorkspaceValidatePathArguments,
    WorkspaceValidateTargetArguments,
    WorkspaceWaitUntilClearArguments,
)

_CANONICAL_OPERATION_NAMES: dict[str, str] = {
    "motion.move_sx": "motion.move_spline",
    "motion.move_sj": "motion.move_spline",
    "gripper.set_width": "gripper.move_width",
    "contact.follow_surface_path": "contact.follow_path",
    "workspace.check_path": "workspace.validate_path",
    "workspace.stop_and_replan": "workspace.request_replan",
}


def _definition(
    operation_name: str,
    description: str,
    argument_model: type[BaseModel],
    *,
    allowed_skill_types: tuple[str, ...] = ALL_SKILL_TYPES,
    required_preconditions: tuple[str, ...] = (),
    side_effects: tuple[str, ...] = (),
    maximum_timeout_s: float = 30.0,
    recovery_operation: str = "recovery.safe_stop",
    hardware_support: bool = False,
) -> PrimitiveDefinition:
    return PrimitiveDefinition(
        operation_name=operation_name,
        description=description,
        argument_model=argument_model,
        allowed_skill_types=allowed_skill_types,
        required_preconditions=required_preconditions,
        side_effects=side_effects,
        maximum_timeout_s=maximum_timeout_s,
        recovery_operation=recovery_operation,
        hardware_support=hardware_support,
    )


def builtin_primitive_definitions() -> tuple[PrimitiveDefinition, ...]:
    """Build the fixed MVP primitive catalog.

    ``hardware_support`` remains false until adapter signatures and behavior are
    verified on the target cell.  Mock and simulation support are true by
    construction in :class:`PrimitiveDefinition`.
    """

    motion_types = ("motion", "manipulation", "contact", "composite")
    contact_types = ("contact", "composite")
    recovery_types = ("recovery", "motion", "manipulation", "contact", "composite")
    return (
        _definition(
            "motion.move_j",
            "Execute a joint-space path to six validated joint angles.",
            MoveJArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
        ),
        _definition(
            "motion.rotate_joint_6_relative",
            "Rotate only wrist joint 6 by a bounded runtime-relative angle.",
            RotateJoint6RelativeArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
        ),
        _definition(
            "motion.move_l",
            "Execute a linear TCP path to an anchor-relative goal.",
            MoveLArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
        ),
        _definition(
            "motion.move_c",
            "Execute a circular TCP segment through an anchor-relative via point.",
            MoveCArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
        ),
        _definition(
            "motion.move_spline",
            "Execute a bounded spline through anchor-relative waypoints.",
            MoveSplineArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "motion.move_periodic",
            "Execute approved periodic geometry around an anchor-relative center.",
            MovePeriodicArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
            maximum_timeout_s=120.0,
        ),
        _definition(
            "motion.wait",
            "Wait for a bounded duration while global monitoring remains active.",
            WaitArguments,
            maximum_timeout_s=300.0,
        ),
        _definition(
            "gripper.open",
            "Open the bound gripper through its approved adapter.",
            GripperOpenArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("tool_verified",),
            side_effects=("gripper_motion", "object_release"),
        ),
        _definition(
            "gripper.close",
            "Close the bound gripper through its approved adapter.",
            GripperCloseArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("tool_verified",),
            side_effects=("gripper_motion", "object_grasp"),
        ),
        _definition(
            "gripper.move_width",
            "Move the bound gripper to a validated opening width.",
            GripperMoveWidthArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("tool_verified",),
            side_effects=("gripper_motion",),
        ),
        _definition(
            "gripper.verify_state",
            "Verify the bound gripper state without moving it.",
            GripperVerifyStateArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("tool_verified",),
        ),
        _definition(
            "grasp.verify_holding",
            "Verify and record object attachment using an approved local profile.",
            GraspVerifyHoldingArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("object_verified", "tool_verified"),
            side_effects=("attachment_state_verified",),
        ),
        _definition(
            "grasp.verify_released",
            "Verify and record object release using an approved local profile.",
            GraspVerifyReleasedArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("object_verified", "tool_verified"),
            side_effects=("release_state_verified",),
        ),
        _definition(
            "contact.search_surface",
            "Search for a permitted surface using an approved force profile.",
            ContactSearchArguments,
            allowed_skill_types=contact_types,
            required_preconditions=("surface_verified", "force_supervisor_active"),
            side_effects=("robot_motion", "possible_contact"),
            recovery_operation="recovery.safe_retract",
        ),
        _definition(
            "contact.enable_force",
            "Enter supervised force mode using only an approved profile.",
            EnableForceArguments,
            allowed_skill_types=contact_types,
            required_preconditions=(
                "contact_search_succeeded",
                "surface_verified",
                "force_supervisor_active",
            ),
            side_effects=("force_mode_enabled",),
            recovery_operation="contact.disable_force",
        ),
        _definition(
            "contact.follow_path",
            "Follow an anchor-relative path under active global force supervision.",
            ContactFollowPathArguments,
            allowed_skill_types=contact_types,
            required_preconditions=("force_mode_enabled", "workspace_valid"),
            side_effects=("robot_motion", "surface_contact"),
            recovery_operation="contact.disable_force",
            maximum_timeout_s=120.0,
        ),
        _definition(
            "contact.disable_force",
            "Ramp down force and leave compliance mode.",
            DisableForceArguments,
            allowed_skill_types=contact_types,
            required_preconditions=("force_supervisor_active",),
            side_effects=("force_mode_disabled",),
            recovery_operation="recovery.safe_stop",
        ),
        _definition(
            "contact.verify_force",
            "Verify measured force against the active approved profile.",
            VerifyForceArguments,
            allowed_skill_types=contact_types,
            required_preconditions=("force_mode_enabled", "force_supervisor_active"),
            recovery_operation="contact.disable_force",
        ),
        _definition(
            "workspace.validate_target",
            "Validate one bound target against global workspace policy.",
            WorkspaceValidateTargetArguments,
            required_preconditions=("workspace_monitor_active",),
        ),
        _definition(
            "workspace.validate_path",
            "Validate a bound path against collision and workspace policy.",
            WorkspaceValidatePathArguments,
            required_preconditions=("workspace_monitor_active",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "workspace.wait_until_clear",
            "Wait until the named region is clear while continuously monitoring it.",
            WorkspaceWaitUntilClearArguments,
            required_preconditions=("workspace_monitor_active",),
            maximum_timeout_s=300.0,
        ),
        _definition(
            "workspace.request_replan",
            "Request a locally generated collision-free path to a relative target.",
            WorkspaceRequestReplanArguments,
            required_preconditions=("workspace_monitor_active",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "recovery.safe_stop",
            "Stop motion through the global safety supervisor.",
            RecoverySafeStopArguments,
            allowed_skill_types=recovery_types,
            side_effects=("robot_stop",),
            maximum_timeout_s=10.0,
            recovery_operation="recovery.abort",
        ),
        _definition(
            "recovery.safe_retract",
            "Retract using an approved recovery profile and direction.",
            RecoverySafeRetractArguments,
            allowed_skill_types=recovery_types,
            required_preconditions=("workspace_monitor_active",),
            side_effects=("robot_motion",),
            recovery_operation="recovery.safe_stop",
        ),
        _definition(
            "recovery.return_checkpoint",
            "Return to a runtime-recorded, revalidated checkpoint.",
            RecoveryReturnCheckpointArguments,
            allowed_skill_types=recovery_types,
            required_preconditions=("checkpoint_valid", "workspace_monitor_active"),
            side_effects=("robot_motion",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "recovery.abort",
            "Abort the skill through the fixed runtime state machine.",
            RecoveryAbortArguments,
            allowed_skill_types=recovery_types,
            side_effects=("skill_abort",),
            maximum_timeout_s=10.0,
            recovery_operation="recovery.safe_stop",
        ),
        _definition(
            "motion.move_sx",
            "Compatibility name for an anchor-relative Cartesian spline.",
            MoveSplineArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "motion.move_sj",
            "Compatibility name for a locally solved spline through relative goals.",
            MoveSplineArguments,
            allowed_skill_types=motion_types,
            required_preconditions=("workspace_valid", "motion_profile_loaded"),
            side_effects=("robot_motion",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "gripper.set_width",
            "Compatibility name for a validated gripper opening width.",
            GripperMoveWidthArguments,
            allowed_skill_types=("manipulation", "composite"),
            required_preconditions=("tool_verified",),
            side_effects=("gripper_motion",),
        ),
        _definition(
            "contact.follow_surface_path",
            "Compatibility name for a supervised anchor-relative contact path.",
            ContactFollowPathArguments,
            allowed_skill_types=contact_types,
            required_preconditions=("force_mode_enabled", "workspace_valid"),
            side_effects=("robot_motion", "surface_contact"),
            recovery_operation="contact.disable_force",
            maximum_timeout_s=120.0,
        ),
        _definition(
            "workspace.check_path",
            "Compatibility name for global workspace path validation.",
            WorkspaceValidatePathArguments,
            required_preconditions=("workspace_monitor_active",),
            maximum_timeout_s=60.0,
        ),
        _definition(
            "workspace.stop_and_replan",
            "Stop through the supervisor and request a local replan to a relative goal.",
            WorkspaceRequestReplanArguments,
            required_preconditions=("workspace_monitor_active",),
            side_effects=("robot_stop",),
            maximum_timeout_s=60.0,
        ),
    )


class PrimitiveRegistry:
    """A non-dynamic registry of explicitly registered primitive definitions."""

    def __init__(
        self,
        definitions: Iterable[PrimitiveDefinition] | None = None,
        *,
        include_defaults: bool = True,
    ) -> None:
        self._definitions: dict[str, PrimitiveDefinition] = {}
        if include_defaults:
            definitions_to_add: Iterable[PrimitiveDefinition] = builtin_primitive_definitions()
        else:
            definitions_to_add = ()
        for definition in definitions_to_add:
            self.register(definition)
        if definitions is not None:
            for definition in definitions:
                self.register(definition)

    @classmethod
    def default(cls) -> PrimitiveRegistry:
        """Return a fresh registry containing exactly the built-in whitelist."""

        return cls()

    def register(self, definition: PrimitiveDefinition) -> None:
        """Register a code-owned definition and reject duplicate names."""

        operation = definition.metadata.operation_name
        if operation in self._definitions:
            raise PrimitiveValidationError(f"primitive {operation!r} is already registered")
        self._definitions[operation] = definition

    def __contains__(self, operation_name: object) -> bool:
        return operation_name in self._definitions

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._definitions)

    def get(self, operation_name: str) -> PrimitiveDefinition:
        """Return a whitelisted definition or reject the unknown operation."""

        try:
            return self._definitions[operation_name]
        except KeyError as error:
            raise UnknownPrimitiveError(
                f"operation {operation_name!r} is not in the primitive whitelist"
            ) from error

    def has_operation(self, operation_name: str) -> bool:
        """Return whether an operation is in this closed registry."""

        return operation_name in self._definitions

    def operation_names(self) -> tuple[str, ...]:
        """Return whitelisted names in stable lexical order."""

        return tuple(sorted(self._definitions))

    def metadata(self, operation_name: str) -> PrimitiveMetadata:
        """Return serializable metadata for one operation."""

        return self.get(operation_name).metadata

    def canonical_operation_name(self, operation_name: str) -> str:
        """Resolve a supported compatibility operation to its runtime canonical name."""

        self.get(operation_name)
        return _CANONICAL_OPERATION_NAMES.get(operation_name, operation_name)

    def catalog(self) -> tuple[PrimitiveMetadata, ...]:
        """Return stable operation metadata sorted by operation name."""

        return tuple(
            self._definitions[name].metadata for name in sorted(self._definitions)
        )

    def validate_arguments(
        self,
        operation_name: str,
        arguments: Mapping[str, Any] | BaseModel,
    ) -> BaseModel:
        """Validate and normalize arguments with the fixed operation model."""

        definition = self.get(operation_name)
        raw_arguments: Any
        if isinstance(arguments, BaseModel):
            raw_arguments = arguments.model_dump(mode="python")
        else:
            raw_arguments = dict(arguments)
        try:
            return definition.argument_model.model_validate(raw_arguments)
        except ValidationError as error:
            raise PrimitiveValidationError(
                f"invalid arguments for {operation_name!r}: {error}"
            ) from error

    def validate_timeout(self, operation_name: str, timeout_s: float | None) -> float:
        """Resolve an omitted timeout and reject values above the primitive maximum."""

        maximum = self.metadata(operation_name).maximum_timeout_s
        if timeout_s is None:
            return maximum
        if timeout_s <= 0.0 or timeout_s > maximum:
            raise PrimitiveValidationError(
                f"timeout_s for {operation_name!r} must be in (0, {maximum}]"
            )
        return timeout_s

    def validate_operation(
        self,
        operation_name: str,
        arguments: Mapping[str, Any],
        *,
        skill_type: str | None = None,
        timeout_s: float | None = None,
    ) -> BaseModel:
        """Validate one complete invocation against metadata and typed arguments."""

        metadata = self.metadata(operation_name)
        if skill_type is not None and skill_type not in metadata.allowed_skill_types:
            raise PrimitiveValidationError(
                f"operation {operation_name!r} is not allowed for skill type {skill_type!r}"
            )
        self.validate_timeout(operation_name, timeout_s)
        return self.validate_arguments(operation_name, arguments)


DEFAULT_PRIMITIVE_REGISTRY = PrimitiveRegistry.default()


def get_default_registry() -> PrimitiveRegistry:
    """Return a fresh default registry so callers cannot mutate a process singleton."""

    return PrimitiveRegistry.default()
