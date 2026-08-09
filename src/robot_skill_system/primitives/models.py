"""Typed primitive arguments and approved profile schemas.

Primitive argument models intentionally contain geometry and profile *ids*, not
velocity, acceleration, or force magnitudes.  Numeric safety parameters live in
locally approved configuration profiles and are injected by the runtime.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from robot_skill_system.scene.models import StrictModel, Vector3
from robot_skill_system.scene.transforms import RelativePose

_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_BINDING_OR_ID_PATTERN = re.compile(
    r"^(?:\$[A-Za-z][A-Za-z0-9_]*|[A-Za-z0-9][A-Za-z0-9_.:-]*)$"
)


def validate_profile_id(value: str) -> str:
    """Validate a stable local configuration profile id."""

    if not _PROFILE_ID_PATTERN.fullmatch(value):
        raise ValueError("profile id must be lower-case and path-independent")
    return value


def validate_binding_or_id(value: str) -> str:
    """Validate a concrete scene id or a graph binding variable."""

    if not _BINDING_OR_ID_PATTERN.fullmatch(value):
        raise ValueError("value must be a scene id or a $binding reference")
    return value


class PrimitiveArguments(StrictModel):
    """Base for operation-specific, extra-forbidding argument models."""


class MotionTargetArguments(PrimitiveArguments):
    """Arguments shared by point-to-point and linear TCP motion."""

    target: RelativePose = Field(validation_alias=AliasChoices("target", "target_pose"))
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)


class MoveJArguments(PrimitiveArguments):
    """Six-axis joint goal in radians with profile-owned motion limits."""

    target_joint_positions_rad: list[float] = Field(min_length=6, max_length=6)
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)

    @field_validator("target_joint_positions_rad")
    @classmethod
    def validate_joint_positions(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(position_rad) for position_rad in value):
            raise ValueError("move_j joint positions must be finite")
        if any(abs(position_rad) > 2.0 * math.pi for position_rad in value):
            raise ValueError("move_j joint positions exceed the ±360 degree schema envelope")
        return value


class RotateJoint6RelativeArguments(PrimitiveArguments):
    """Runtime-relative wrist rotation with profile-owned motion limits."""

    delta_rad: float = Field(ge=-math.pi / 2.0, le=math.pi / 2.0)
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)

    @field_validator("delta_rad")
    @classmethod
    def validate_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("relative J6 rotation must be finite")
        return value


class MoveLArguments(MotionTargetArguments):
    """Anchor-relative straight TCP motion goal."""


class MoveCArguments(PrimitiveArguments):
    """Anchor-relative via and end targets for a circular TCP segment."""

    via: RelativePose = Field(validation_alias=AliasChoices("via", "via_pose"))
    target: RelativePose = Field(validation_alias=AliasChoices("target", "target_pose"))
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)

    @model_validator(mode="after")
    def validate_shared_anchor(self) -> MoveCArguments:
        """Keep circular geometry in one semantic frame."""

        if self.via.anchor_id != self.target.anchor_id:
            raise ValueError("move_c via and target must use the same anchor_id")
        return self


class MoveSplineArguments(PrimitiveArguments):
    """A bounded sequence of anchor-relative spline waypoints."""

    waypoints: list[RelativePose] = Field(
        min_length=2,
        max_length=128,
        validation_alias=AliasChoices("waypoints", "target_poses", "poses"),
    )
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)

    @field_validator("waypoints")
    @classmethod
    def validate_shared_anchor(cls, value: list[RelativePose]) -> list[RelativePose]:
        """Require all waypoints to use one persisted anchor."""

        if len({waypoint.anchor_id for waypoint in value}) != 1:
            raise ValueError("spline waypoints must use the same anchor_id")
        return value


class MovePeriodicArguments(PrimitiveArguments):
    """Bounded periodic geometry; timing and speed remain profile-owned."""

    center: RelativePose
    amplitude_m: Vector3 = Field(
        validation_alias=AliasChoices("amplitude_m", "amplitudes_m")
    )
    repetitions: int = Field(ge=1, le=100)
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)

    @field_validator("amplitude_m")
    @classmethod
    def validate_amplitude(cls, value: Vector3) -> Vector3:
        """Reject an empty periodic path and cap geometry to a local workspace scale."""

        if value.norm() <= 1e-9:
            raise ValueError("periodic amplitude_m cannot be zero")
        if max(abs(value.x), abs(value.y), abs(value.z)) > 1.0:
            raise ValueError("periodic amplitude_m exceeds the schema safety envelope")
        return value


class WaitArguments(PrimitiveArguments):
    """A bounded dwell independent of robot motion profiles."""

    duration_s: float = Field(
        gt=0.0,
        le=300.0,
        validation_alias=AliasChoices("duration_s", "duration"),
    )


class ToolBindingArguments(PrimitiveArguments):
    """Optional tool binding shared by simple gripper commands."""

    tool: str = "$tool"

    _validate_tool = field_validator("tool")(validate_binding_or_id)


class GripperOpenArguments(ToolBindingArguments):
    """Open the bound gripper using adapter-approved settings."""


class GripperCloseArguments(ToolBindingArguments):
    """Close the bound gripper using adapter-approved settings."""


class GripperMoveWidthArguments(ToolBindingArguments):
    """Move to a geometrically requested opening width in metres."""

    width_m: float = Field(ge=0.0, le=0.2)


class GripperState(str, Enum):
    """Observable gripper states accepted by verification."""

    OPEN = "open"
    CLOSED = "closed"
    HOLDING = "holding"


class GripperVerifyStateArguments(ToolBindingArguments):
    """Verify the bound gripper's observable state."""

    expected_state: GripperState


class GraspVerificationArguments(PrimitiveArguments):
    """Verify attachment state using code-owned object/tool bindings and a profile.

    The arguments intentionally contain no thresholds.  Sensor interpretation
    thresholds belong to the locally approved :class:`GraspVerificationProfile`.
    """

    object: str = "$object"
    tool: str = "$gripper"
    verification_profile_id: str

    _validate_object = field_validator("object")(validate_binding_or_id)
    _validate_tool = field_validator("tool")(validate_binding_or_id)
    _validate_profile = field_validator("verification_profile_id")(validate_profile_id)


class GraspVerifyHoldingArguments(GraspVerificationArguments):
    """Verify and record that the bound tool is holding the bound object."""


class GraspVerifyReleasedArguments(GraspVerificationArguments):
    """Verify and record that the bound object has been released by the tool."""


class ContactSearchArguments(PrimitiveArguments):
    """Search for an approved surface with a profile-owned speed and force."""

    surface: str = Field(
        default="$surface",
        validation_alias=AliasChoices("surface", "surface_id"),
    )
    force_profile_id: str

    _validate_surface = field_validator("surface")(validate_binding_or_id)
    _validate_profile = field_validator("force_profile_id")(validate_profile_id)


class EnableForceArguments(ContactSearchArguments):
    """Enable force mode for an approved surface and force profile."""


class ContactFollowPathArguments(PrimitiveArguments):
    """Follow an anchor-relative path while global force supervision is active."""

    path: list[RelativePose] = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("path", "waypoints", "target_poses"),
    )
    motion_profile_id: str = Field(
        validation_alias=AliasChoices("motion_profile_id", "profile_id")
    )

    _validate_profile = field_validator("motion_profile_id")(validate_profile_id)

    @field_validator("path")
    @classmethod
    def validate_shared_anchor(cls, value: list[RelativePose]) -> list[RelativePose]:
        """Keep the complete contact path relative to one surface/object anchor."""

        if len({point.anchor_id for point in value}) != 1:
            raise ValueError("contact path points must use the same anchor_id")
        return value


class DisableForceArguments(PrimitiveArguments):
    """Disable force mode; the runtime remembers the active approved profile."""


class VerifyForceArguments(PrimitiveArguments):
    """Verify force tracking against the active local profile."""

    force_profile_id: str

    _validate_profile = field_validator("force_profile_id")(validate_profile_id)


class WorkspaceValidateTargetArguments(PrimitiveArguments):
    """Ask the global workspace supervisor to validate one relative target."""

    target: RelativePose = Field(validation_alias=AliasChoices("target", "target_pose"))


class WorkspaceValidatePathArguments(PrimitiveArguments):
    """Ask the global workspace supervisor to validate a relative path."""

    path: list[RelativePose] = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("path", "waypoints", "target_poses"),
    )


class WorkspaceWaitUntilClearArguments(PrimitiveArguments):
    """Wait for a named workspace region to become clear."""

    region: str = Field(validation_alias=AliasChoices("region", "region_id"))

    _validate_region = field_validator("region")(validate_binding_or_id)


class WorkspaceRequestReplanArguments(PrimitiveArguments):
    """Request a locally planned replacement path to a relative goal."""

    target: RelativePose = Field(validation_alias=AliasChoices("target", "target_pose"))


class RecoverySafeStopArguments(PrimitiveArguments):
    """Stop using the globally configured safe-stop behavior."""


class RecoverySafeRetractArguments(PrimitiveArguments):
    """Retract using a prevalidated recovery profile, not model-provided values."""

    recovery_profile_id: str = "safe_retract_default"
    surface: str = "$surface"

    _validate_profile = field_validator("recovery_profile_id")(validate_profile_id)
    _validate_surface = field_validator("surface")(validate_binding_or_id)


class RecoveryReturnCheckpointArguments(PrimitiveArguments):
    """Return to a runtime-recorded checkpoint."""

    checkpoint_id: str | None = None


class RecoveryAbortArguments(PrimitiveArguments):
    """Abort with an optional structured reason label."""

    reason_code: str = Field(default="skill_aborted", min_length=1, max_length=80)


class PrimitiveMetadata(StrictModel):
    """Serializable whitelist metadata for one local operation."""

    operation_name: str
    description: str = Field(min_length=1)
    typed_parameter_schema: dict[str, Any]
    allowed_skill_types: tuple[str, ...]
    required_preconditions: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    maximum_timeout_s: float = Field(gt=0.0)
    recovery_operation: str
    hardware_support: bool
    simulation_support: bool
    mock_support: bool

    @field_validator("operation_name", "recovery_operation")
    @classmethod
    def validate_operation_name(cls, value: str) -> str:
        """Require a stable ``namespace.operation`` name."""

        if not _OPERATION_PATTERN.fullmatch(value):
            raise ValueError("operation names must use lower-case namespace.operation syntax")
        return value


class MotionProfile(StrictModel):
    """Approved velocity/acceleration limits for one category of motion."""

    profile_id: str
    motion_kinds: tuple[str, ...] = Field(min_length=1)
    joint_velocity_rad_s: float | None = Field(default=None, gt=0.0)
    joint_acceleration_rad_s2: float | None = Field(default=None, gt=0.0)
    linear_velocity_m_s: float | None = Field(default=None, gt=0.0)
    linear_acceleration_m_s2: float | None = Field(default=None, gt=0.0)
    angular_velocity_rad_s: float | None = Field(default=None, gt=0.0)
    angular_acceleration_rad_s2: float | None = Field(default=None, gt=0.0)
    safety_scale: float = Field(gt=0.0, le=1.0)
    blend_radius_m: float = Field(default=0.0, ge=0.0)

    _validate_profile = field_validator("profile_id")(validate_profile_id)

    @model_validator(mode="after")
    def validate_kind_limits(self) -> MotionProfile:
        """Require joint or Cartesian limits appropriate to the listed motions."""

        has_joint = any(kind == "move_j" for kind in self.motion_kinds)
        has_cartesian = any(kind != "move_j" for kind in self.motion_kinds)
        if has_joint and (
            self.joint_velocity_rad_s is None or self.joint_acceleration_rad_s2 is None
        ):
            raise ValueError("move_j profiles require joint velocity and acceleration")
        if has_cartesian and (
            self.linear_velocity_m_s is None or self.linear_acceleration_m_s2 is None
        ):
            raise ValueError("Cartesian profiles require linear velocity and acceleration")
        return self


class GraspVerificationProfile(StrictModel):
    """Locally approved interpretation policy for grasp/release observations.

    These values are configuration-owned.  They must never be copied from LLM
    output or embedded in a SkillGraph primitive invocation.
    """

    profile_id: str
    require_holding_signal: bool = True
    minimum_released_width_m: float = Field(ge=0.0, le=0.2)

    _validate_profile = field_validator("profile_id")(validate_profile_id)


class UnexpectedContactPolicy(str, Enum):
    """Approved reaction to unexpected contact."""

    STOP_RELEASE_RETRACT = "stop_release_retract"
    ABORT = "abort"


class ForceProfile(StrictModel):
    """Approved force-control parameters; never populated from model output."""

    profile_id: str
    allowed_tool_classes: tuple[str, ...] = Field(min_length=1)
    allowed_surface_roles: tuple[str, ...] = Field(min_length=1)
    target_force_vector_n: Vector3
    force_control_axes: (
        tuple[bool, bool, bool] | tuple[bool, bool, bool, bool, bool, bool]
    )
    maximum_force_n: float = Field(gt=0.0)
    minimum_contact_force_n: float = Field(ge=0.0)
    tangential_force_limit_n: float = Field(gt=0.0)
    stiffness_n_m: Vector3
    force_ramp_time_s: float = Field(gt=0.0)
    force_release_time_s: float = Field(gt=0.0)
    contact_search_speed_m_s: float = Field(gt=0.0)
    maximum_search_distance_m: float = Field(gt=0.0)
    unexpected_contact_policy: UnexpectedContactPolicy

    _validate_profile = field_validator("profile_id")(validate_profile_id)

    @model_validator(mode="after")
    def validate_force_bounds(self) -> ForceProfile:
        """Keep target/contact thresholds within the configured maximum force."""

        if self.target_force_vector_n.norm() > self.maximum_force_n + 1e-9:
            raise ValueError("target force magnitude exceeds maximum_force_n")
        if self.minimum_contact_force_n > self.maximum_force_n:
            raise ValueError("minimum_contact_force_n exceeds maximum_force_n")
        if self.tangential_force_limit_n > self.maximum_force_n:
            raise ValueError("tangential_force_limit_n exceeds maximum_force_n")
        if not any(self.force_control_axes):
            raise ValueError("at least one force-control axis must be enabled")
        return self


class SafetyPolicy(StrictModel):
    """Global limits that generated skill code cannot override."""

    policy_id: str
    minimum_clearance_m: float = Field(gt=0.0)
    maximum_linear_velocity_m_s: float = Field(gt=0.0)
    maximum_linear_acceleration_m_s2: float = Field(gt=0.0)
    maximum_joint_velocity_rad_s: float = Field(gt=0.0)
    maximum_joint_acceleration_rad_s2: float = Field(gt=0.0)
    maximum_force_n: float = Field(gt=0.0)
    unknown_space_is_occupied: bool
    dynamic_obstacle_policy: str
    emergency_stop_required: bool
    workspace_monitor_required: bool
    force_supervisor_required: bool

    _validate_profile = field_validator("policy_id")(validate_profile_id)


class PrimitiveDefinition:
    """In-memory pairing of serializable metadata and a fixed Pydantic model."""

    __slots__ = ("argument_model", "metadata")

    argument_model: type[BaseModel]
    metadata: PrimitiveMetadata

    def __init__(
        self,
        *,
        operation_name: str,
        description: str,
        argument_model: type[BaseModel],
        allowed_skill_types: tuple[str, ...],
        required_preconditions: tuple[str, ...] = (),
        side_effects: tuple[str, ...] = (),
        maximum_timeout_s: float,
        recovery_operation: str,
        hardware_support: bool = False,
        simulation_support: bool = True,
        mock_support: bool = True,
    ) -> None:
        self.argument_model = argument_model
        self.metadata = PrimitiveMetadata(
            operation_name=operation_name,
            description=description,
            typed_parameter_schema=argument_model.model_json_schema(),
            allowed_skill_types=allowed_skill_types,
            required_preconditions=required_preconditions,
            side_effects=side_effects,
            maximum_timeout_s=maximum_timeout_s,
            recovery_operation=recovery_operation,
            hardware_support=hardware_support,
            simulation_support=simulation_support,
            mock_support=mock_support,
        )


ALL_SKILL_TYPES: tuple[str, ...] = (
    "motion",
    "manipulation",
    "contact",
    "inspection",
    "recovery",
    "composite",
)
