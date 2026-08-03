"""Strict, unit-explicit Pydantic models for a captured robot scene.

All orientations use normalized quaternions in ``xyzw`` order.  Positions are
metres and timestamps are nanoseconds.  These models contain perception facts;
they do not make motion or safety decisions.
"""

from __future__ import annotations

import math
import re
import time
from enum import Enum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

_FRAME_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)*$")
_ENTITY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def validate_frame_id(value: str) -> str:
    """Return a valid TF-style frame id or raise ``ValueError``.

    Leading slashes, empty path components, whitespace, and parent-directory
    components are rejected so a frame id can never be confused with a path.
    """

    if not _FRAME_ID_PATTERN.fullmatch(value) or ".." in value:
        raise ValueError(
            "frame_id must be a non-empty relative TF name without whitespace or '..'"
        )
    return value


def validate_entity_id(value: str) -> str:
    """Validate a concrete scene entity identifier."""

    if not _ENTITY_ID_PATTERN.fullmatch(value):
        raise ValueError("entity id contains unsupported characters")
    return value


class StrictModel(BaseModel):
    """Shared Pydantic behavior for semantic and geometry schemas."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class Vector3(StrictModel):
    """A finite three-dimensional vector."""

    x: float
    y: float
    z: float

    def norm(self) -> float:
        """Return the Euclidean norm."""

        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(x, y, z)``."""

        return (self.x, self.y, self.z)


class Quaternion(StrictModel):
    """A normalized quaternion in ``xyzw`` component order."""

    x: float
    y: float
    z: float
    w: float

    @model_validator(mode="after")
    def validate_unit_quaternion(self) -> Quaternion:
        """Reject zero, non-finite, and materially non-unit quaternions."""

        squared_norm = self.x**2 + self.y**2 + self.z**2 + self.w**2
        if not math.isfinite(squared_norm) or squared_norm <= 1e-12:
            raise ValueError("orientation_xyzw must be a finite non-zero quaternion")
        if not math.isclose(squared_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("orientation_xyzw quaternion must be normalized")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return components in the documented ``xyzw`` order."""

        return (self.x, self.y, self.z, self.w)


class Pose(StrictModel):
    """A timestamped six-degree-of-freedom pose."""

    frame_id: str
    position_m: Vector3
    orientation_xyzw: Quaternion
    timestamp_ns: int = Field(ge=0)
    source: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    covariance: tuple[float, ...] | None = None
    uncertainty: dict[str, float] | None = None

    _COVARIANCE_LENGTHS = {6, 36}

    _validate_frame = field_validator("frame_id")(validate_frame_id)

    @field_validator("covariance")
    @classmethod
    def validate_covariance(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        """Accept a diagonal-6 or flattened 6x6 covariance vector."""

        if value is not None:
            if len(value) not in cls._COVARIANCE_LENGTHS:
                raise ValueError("covariance must contain 6 diagonal or 36 matrix values")
            if any(not math.isfinite(component) for component in value):
                raise ValueError("covariance values must be finite")
        return value


class BoundingBox2D(StrictModel):
    """Axis-aligned pixel bounding box."""

    x_min_px: float = Field(ge=0.0, validation_alias=AliasChoices("x_min_px", "x_min"))
    y_min_px: float = Field(ge=0.0, validation_alias=AliasChoices("y_min_px", "y_min"))
    x_max_px: float = Field(ge=0.0, validation_alias=AliasChoices("x_max_px", "x_max"))
    y_max_px: float = Field(ge=0.0, validation_alias=AliasChoices("y_max_px", "y_max"))

    @model_validator(mode="after")
    def validate_extents(self) -> BoundingBox2D:
        """Ensure minimum corners precede maximum corners."""

        if self.x_max_px <= self.x_min_px or self.y_max_px <= self.y_min_px:
            raise ValueError("2D bounding box must have positive width and height")
        return self


class BoundingBox3D(StrictModel):
    """Oriented 3D box represented by a pose and positive dimensions."""

    center_pose: Pose = Field(validation_alias=AliasChoices("center_pose", "pose"))
    size_m: Vector3

    @field_validator("size_m")
    @classmethod
    def validate_size(cls, value: Vector3) -> Vector3:
        """Reject zero or negative extents."""

        if min(value.x, value.y, value.z) <= 0.0:
            raise ValueError("3D bounding box size_m components must be positive")
        return value


class ContactGeometry(StrictModel):
    """Tool contact geometry described in a calibrated tool frame."""

    geometry_type: str = Field(min_length=1)
    frame_id: str
    parameters_m: dict[str, float] = Field(default_factory=dict)

    _validate_frame = field_validator("frame_id")(validate_frame_id)


class PlaneGeometry(StrictModel):
    """A locally estimated plane in a named frame."""

    frame_id: str
    center_m: Vector3 = Field(validation_alias=AliasChoices("center_m", "center"))
    normal: Vector3
    boundary_m: list[Vector3] = Field(
        default_factory=list,
        validation_alias=AliasChoices("boundary_m", "boundary"),
    )

    _validate_frame = field_validator("frame_id")(validate_frame_id)

    @field_validator("normal")
    @classmethod
    def validate_normal(cls, value: Vector3) -> Vector3:
        """Require a unit surface normal."""

        if not math.isclose(value.norm(), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("surface normal must be normalized")
        return value


class SurfaceRole(str, Enum):
    """Safety-relevant semantic role of a perceived surface."""

    CONTACT_TARGET = "contact_target"
    SUPPORT_SURFACE = "support_surface"
    FORBIDDEN_SURFACE = "forbidden_surface"
    UNKNOWN_SURFACE = "unknown_surface"


class WorkspaceRole(str, Enum):
    """Safety-relevant semantic role of a workspace volume."""

    TASK_REGION = "task_region"
    FREE_SPACE = "free_space"
    CONTACT_TARGET = "contact_target"
    GRASP_TARGET = "grasp_target"
    FORBIDDEN_REGION = "forbidden_region"
    UNKNOWN_REGION = "unknown_region"


class AccessPolicy(str, Enum):
    """Default access behavior for a workspace region."""

    ALLOWED = "allowed"
    SUPERVISED = "supervised"
    FORBIDDEN = "forbidden"
    OCCUPIED = "occupied"


class ObjectInstance(StrictModel):
    """A locally perceived object instance."""

    instance_id: str
    class_name: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    pose: Pose
    bounding_box_2d: BoundingBox2D | None = None
    bounding_box_3d: BoundingBox3D | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    visible_fraction: float = Field(ge=0.0, le=1.0)
    pose_source: str = Field(min_length=1)
    object_definition_id: str | None = None

    _validate_id = field_validator("instance_id")(validate_entity_id)


class ToolInstance(StrictModel):
    """A calibrated or perceived physical tool."""

    instance_id: str
    tool_class: str = Field(min_length=1)
    attached: bool
    tcp_frame: str
    pose: Pose
    contact_geometry: ContactGeometry | None = None
    compatible_skills: list[str] = Field(default_factory=list)
    verification_confidence: float = Field(ge=0.0, le=1.0)

    _validate_id = field_validator("instance_id")(validate_entity_id)
    _validate_tcp_frame = field_validator("tcp_frame")(validate_frame_id)

    @property
    def confidence(self) -> float:
        """Expose a uniform confidence attribute for binding code."""

        return self.verification_confidence


class SurfaceInstance(StrictModel):
    """A surface and its contact permissions."""

    instance_id: str
    role: SurfaceRole
    pose: Pose | None = Field(
        default=None,
        validation_alias=AliasChoices("pose", "anchor_pose", "frame_pose"),
    )
    plane: PlaneGeometry | None = None
    mesh_uri: str | None = None
    center_m: Vector3 = Field(validation_alias=AliasChoices("center_m", "center"))
    normal: Vector3
    boundary_m: list[Vector3] = Field(
        default_factory=list,
        validation_alias=AliasChoices("boundary_m", "boundary"),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    allowed_contact_links: list[str] = Field(default_factory=list)
    material: str | None = Field(
        default=None,
        validation_alias=AliasChoices("material", "surface_type"),
    )

    _validate_id = field_validator("instance_id")(validate_entity_id)

    @field_validator("normal")
    @classmethod
    def validate_normal(cls, value: Vector3) -> Vector3:
        """Require a unit surface normal."""

        if not math.isclose(value.norm(), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("surface normal must be normalized")
        return value

    @model_validator(mode="after")
    def validate_anchor_pose(self) -> SurfaceInstance:
        """Keep an optional calibrated surface frame consistent with its plane facts.

        Legacy perception backends may provide only ``center_m`` and ``normal``; those retain
        translation-only binding.  A backend that supplies ``pose`` opts into full 6D anchor
        rebinding, so the pose origin and its local +Z axis must agree with the surface facts.
        """

        if self.pose is None:
            return self
        center = self.center_m.as_tuple()
        origin = self.pose.position_m.as_tuple()
        if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(center, origin, strict=True)):
            raise ValueError("surface pose origin must match center_m")
        qx, qy, qz, qw = self.pose.orientation_xyzw.as_tuple()
        local_z = (
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        if any(
            not math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)
            for a, b in zip(local_z, self.normal.as_tuple(), strict=True)
        ):
            raise ValueError("surface pose local +Z axis must match normal")
        return self

    @property
    def center(self) -> Vector3:
        """Runtime-compatible alias for the surface center."""

        return self.center_m


class WorkspaceRegion(StrictModel):
    """A named workspace volume. Unknown regions are never implicitly free."""

    region_id: str
    role: WorkspaceRole
    geometry: dict[str, Any]
    frame_id: str
    minimum_clearance_m: float = Field(ge=0.0)
    access_policy: AccessPolicy
    confidence: float = Field(ge=0.0, le=1.0)

    _validate_id = field_validator("region_id")(validate_entity_id)
    _validate_frame = field_validator("frame_id")(validate_frame_id)

    @model_validator(mode="after")
    def enforce_unknown_is_not_allowed(self) -> WorkspaceRegion:
        """Make the conservative unknown-space policy explicit in the schema."""

        if self.role is WorkspaceRole.UNKNOWN_REGION and self.access_policy not in {
            AccessPolicy.FORBIDDEN,
            AccessPolicy.OCCUPIED,
        }:
            raise ValueError("unknown workspace regions must be occupied or forbidden")
        return self


class ObstacleInstance(StrictModel):
    """A static or dynamic obstacle estimate."""

    instance_id: str
    pose: Pose
    geometry: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    velocity_m_s: Vector3 | None = None

    _validate_id = field_validator("instance_id")(validate_entity_id)


class CameraMetadata(StrictModel):
    """RGB-D capture metadata needed to interpret local geometry."""

    camera_id: str = Field(min_length=1)
    color_frame_id: str
    depth_frame_id: str
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    depth_scale_m: float = Field(gt=0.0)
    capture_mode: str = "burst"
    frame_count: int = Field(default=1, gt=0)

    _validate_color_frame = field_validator("color_frame_id")(validate_frame_id)
    _validate_depth_frame = field_validator("depth_frame_id")(validate_frame_id)


class ConfidenceSummary(StrictModel):
    """Aggregate scene confidence values, each bounded to ``[0, 1]``."""

    overall: float = Field(ge=0.0, le=1.0)
    perception: float | None = Field(default=None, ge=0.0, le=1.0)
    geometry: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration: float | None = Field(default=None, ge=0.0, le=1.0)


class SceneSnapshot(StrictModel):
    """An immutable-in-meaning snapshot used for binding and preflight."""

    schema_version: str = Field(default="1.0", min_length=1)
    scene_id: str
    timestamp_ns: int = Field(ge=0)
    reference_frame: str
    valid_for_ms: int = Field(gt=0)
    objects: list[ObjectInstance] = Field(default_factory=list)
    tools: list[ToolInstance] = Field(default_factory=list)
    surfaces: list[SurfaceInstance] = Field(default_factory=list)
    workspace_regions: list[WorkspaceRegion] = Field(default_factory=list)
    static_obstacles: list[ObstacleInstance] = Field(default_factory=list)
    dynamic_obstacles: list[ObstacleInstance] = Field(default_factory=list)
    occupancy_map_uri: str | None = None
    occupancy_map_checksum_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    camera_metadata: CameraMetadata | None = None
    calibration_id: str = Field(min_length=1)
    confidence_summary: ConfidenceSummary

    _entity_index: dict[str, object] = PrivateAttr(default_factory=dict)

    _validate_scene_id = field_validator("scene_id")(validate_entity_id)
    _validate_reference_frame = field_validator("reference_frame")(validate_frame_id)

    @model_validator(mode="after")
    def validate_unique_entity_ids(self) -> SceneSnapshot:
        """Reject ambiguous identifiers across all bindable scene collections."""

        identifiers = [entity.instance_id for entity in self.objects]
        identifiers.extend(tool.instance_id for tool in self.tools)
        identifiers.extend(surface.instance_id for surface in self.surfaces)
        identifiers.extend(region.region_id for region in self.workspace_regions)
        identifiers.extend(obstacle.instance_id for obstacle in self.static_obstacles)
        identifiers.extend(obstacle.instance_id for obstacle in self.dynamic_obstacles)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scene entity identifiers must be globally unique")
        if (self.occupancy_map_uri is None) is not (
            self.occupancy_map_checksum_sha256 is None
        ):
            raise ValueError("occupancy map URI and SHA-256 checksum must be provided together")
        return self

    @property
    def workspaces(self) -> list[WorkspaceRegion]:
        """Compatibility alias for workspace binding code."""

        return self.workspace_regions

    def age_ms(self, now_ns: int | None = None) -> float:
        """Return snapshot age in milliseconds using a caller-supplied clock if given."""

        current_ns = time.time_ns() if now_ns is None else now_ns
        return (current_ns - self.timestamp_ns) / 1_000_000.0

    def is_fresh(self, now_ns: int | None = None, required_freshness_ms: int | None = None) -> bool:
        """Return whether this snapshot satisfies both declared and requested freshness."""

        maximum_age_ms = self.valid_for_ms
        if required_freshness_ms is not None:
            if required_freshness_ms <= 0:
                return False
            maximum_age_ms = min(maximum_age_ms, required_freshness_ms)
        age_ms = self.age_ms(now_ns)
        return 0.0 <= age_ms <= maximum_age_ms

    def entity_by_id(self, entity_id: str) -> object | None:
        """Look up any bindable scene entity without guessing its collection."""

        if not self._entity_index:
            entities: list[object] = [*self.objects, *self.tools, *self.surfaces]
            entities.extend(self.workspace_regions)
            entities.extend(self.static_obstacles)
            entities.extend(self.dynamic_obstacles)
            for entity in entities:
                identifier = getattr(entity, "instance_id", None) or getattr(
                    entity, "region_id", None
                )
                if isinstance(identifier, str):
                    self._entity_index[identifier] = entity
        return self._entity_index.get(entity_id)


# Descriptive aliases used in API schemas.
Position3D = Vector3
QuaternionXYZW = Quaternion
