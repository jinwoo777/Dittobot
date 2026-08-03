"""Deterministic rigid-transform helpers for anchor-relative skill targets."""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

from pydantic import field_validator, model_validator

from robot_skill_system.exceptions import BindingError
from robot_skill_system.scene.models import Pose, Quaternion, StrictModel, Vector3

_ANCHOR_REFERENCE_PATTERN = re.compile(
    r"^(?:\$[A-Za-z][A-Za-z0-9_]*|[A-Za-z0-9][A-Za-z0-9_.:-]*)$"
)
_BASE_ABSOLUTE_ANCHORS = frozenset(
    {"base", "base_link", "robot_base", "robot_base_link", "world", "map"}
)


class AnchorType(str, Enum):
    """Permitted persistent reference-frame categories for skill targets."""

    OBJECT = "object"
    TOOL = "tool"
    SURFACE = "surface"
    FIXTURE = "fixture"
    WORKSPACE_REGION = "workspace_region"


def validate_anchor_id(value: str) -> str:
    """Validate an entity id or a ``$binding`` reference and reject base frames."""

    if not _ANCHOR_REFERENCE_PATTERN.fullmatch(value):
        raise ValueError("anchor_id must be a scene entity id or a $binding reference")
    if value.casefold() in _BASE_ABSOLUTE_ANCHORS:
        raise ValueError("robot-base/world absolute anchors cannot be persisted in skills")
    return value


class RigidTransform(StrictModel):
    """A translation in metres and normalized quaternion rotation in ``xyzw``."""

    translation_m: Vector3
    rotation_xyzw: Quaternion

    @classmethod
    def identity(cls) -> RigidTransform:
        """Return an identity rigid transform."""

        return cls(
            translation_m=Vector3(x=0.0, y=0.0, z=0.0),
            rotation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )


class RelativePose(StrictModel):
    """A target transform persisted relative to a semantic scene anchor.

    The canonical JSON shape contains ``anchor_id``, ``position_m`` and
    ``orientation_xyzw``.  For ergonomic interoperability the model also accepts
    a nested ``transform`` (``RigidTransform`` shape) or nested ``pose`` on input,
    but always serializes to the canonical anchor-relative form.
    """

    anchor_id: str
    anchor_type: AnchorType | None = None
    position_m: Vector3
    orientation_xyzw: Quaternion

    _validate_anchor = field_validator("anchor_id")(validate_anchor_id)

    @model_validator(mode="before")
    @classmethod
    def flatten_supported_shapes(cls, value: Any) -> Any:
        """Flatten explicit ``transform``/``pose`` wrappers without losing the anchor."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        transform = normalized.pop("transform", None)
        nested_pose = normalized.pop("pose", None)
        if transform is not None and nested_pose is not None:
            raise ValueError("relative target must provide only one of transform or pose")
        if transform is not None:
            transform_data = (
                transform.model_dump() if isinstance(transform, RigidTransform) else transform
            )
            if not isinstance(transform_data, dict):
                raise ValueError("transform must be a RigidTransform object")
            normalized.setdefault("position_m", transform_data.get("translation_m"))
            normalized.setdefault("orientation_xyzw", transform_data.get("rotation_xyzw"))
        if nested_pose is not None:
            pose_data = nested_pose.model_dump() if isinstance(nested_pose, Pose) else nested_pose
            if not isinstance(pose_data, dict):
                raise ValueError("pose must be a Pose object")
            nested_frame = pose_data.get("frame_id")
            anchor_id = normalized.get("anchor_id")
            if nested_frame is not None and anchor_id is not None and nested_frame != anchor_id:
                raise ValueError("nested relative pose frame_id must equal anchor_id")
            normalized.setdefault("position_m", pose_data.get("position_m"))
            normalized.setdefault("orientation_xyzw", pose_data.get("orientation_xyzw"))
        return normalized

    @property
    def transform(self) -> RigidTransform:
        """Expose this relative pose as a rigid transform."""

        return RigidTransform(
            translation_m=self.position_m,
            rotation_xyzw=self.orientation_xyzw,
        )

    @property
    def pose(self) -> RelativePose:
        """Runtime-compatible pose payload retaining the relative coordinates.

        A ``$binding`` is not a concrete TF frame and therefore cannot safely be
        converted to an absolute :class:`Pose` before scene binding.
        """

        return self


# More explicit name retained for callers that prefer it.
AnchorRelativePose = RelativePose


def _normalized_quaternion(x: float, y: float, z: float, w: float) -> Quaternion:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12 or not math.isfinite(norm):
        raise BindingError("quaternion composition produced an invalid rotation")
    return Quaternion(x=x / norm, y=y / norm, z=z / norm, w=w / norm)


def multiply_quaternions(left: Quaternion, right: Quaternion) -> Quaternion:
    """Compose two rotations using Hamilton multiplication in ``xyzw`` order."""

    lx, ly, lz, lw = left.as_tuple()
    rx, ry, rz, rw = right.as_tuple()
    return _normalized_quaternion(
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def inverse_quaternion(rotation: Quaternion) -> Quaternion:
    """Return the inverse of a normalized quaternion."""

    return Quaternion(x=-rotation.x, y=-rotation.y, z=-rotation.z, w=rotation.w)


def rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector without converting it to Euler angles."""

    qx, qy, qz, qw = rotation.as_tuple()
    # Equivalent to q * (v, 0) * conjugate(q), expanded to avoid accepting a
    # non-unit pure-vector quaternion in the strict Quaternion model.
    tx = 2.0 * (qy * vector.z - qz * vector.y)
    ty = 2.0 * (qz * vector.x - qx * vector.z)
    tz = 2.0 * (qx * vector.y - qy * vector.x)
    return Vector3(
        x=vector.x + qw * tx + (qy * tz - qz * ty),
        y=vector.y + qw * ty + (qz * tx - qx * tz),
        z=vector.z + qw * tz + (qx * ty - qy * tx),
    )


def compose_transforms(parent: RigidTransform, child: RigidTransform) -> RigidTransform:
    """Return ``parent * child`` using standard rigid-transform composition."""

    rotated_translation = rotate_vector(parent.rotation_xyzw, child.translation_m)
    return RigidTransform(
        translation_m=Vector3(
            x=parent.translation_m.x + rotated_translation.x,
            y=parent.translation_m.y + rotated_translation.y,
            z=parent.translation_m.z + rotated_translation.z,
        ),
        rotation_xyzw=multiply_quaternions(parent.rotation_xyzw, child.rotation_xyzw),
    )


def invert_transform(transform: RigidTransform) -> RigidTransform:
    """Return the mathematical inverse of a rigid transform."""

    inverse_rotation = inverse_quaternion(transform.rotation_xyzw)
    negative_translation = Vector3(
        x=-transform.translation_m.x,
        y=-transform.translation_m.y,
        z=-transform.translation_m.z,
    )
    return RigidTransform(
        translation_m=rotate_vector(inverse_rotation, negative_translation),
        rotation_xyzw=inverse_rotation,
    )


def pose_as_transform(pose: Pose) -> RigidTransform:
    """Discard pose metadata and return its geometric transform."""

    return RigidTransform(
        translation_m=pose.position_m,
        rotation_xyzw=pose.orientation_xyzw,
    )


def bind_relative_pose(
    relative_pose: RelativePose,
    anchor_pose: Pose,
    *,
    resolved_anchor_id: str | None = None,
    target_frame_id: str | None = None,
    timestamp_ns: int | None = None,
    source: str = "skill_binding",
) -> Pose:
    """Bind an anchor-relative target using ``T_base_anchor * T_anchor_target``.

    ``anchor_pose`` must already represent the currently resolved scene anchor.
    Its ``frame_id`` names the coordinate system in which the anchor pose is
    expressed (normally base/camera), not the anchor entity id.  Callers may pass
    ``resolved_anchor_id`` to make entity resolution consistency explicit.
    """

    if resolved_anchor_id is not None and not relative_pose.anchor_id.startswith("$") and (
        relative_pose.anchor_id != resolved_anchor_id
    ):
        raise BindingError(
            f"resolved anchor {resolved_anchor_id!r} does not match relative anchor "
            f"{relative_pose.anchor_id!r}"
        )
    composed = compose_transforms(pose_as_transform(anchor_pose), relative_pose.transform)
    return Pose(
        frame_id=target_frame_id or anchor_pose.frame_id,
        position_m=composed.translation_m,
        orientation_xyzw=composed.rotation_xyzw,
        timestamp_ns=anchor_pose.timestamp_ns if timestamp_ns is None else timestamp_ns,
        source=source,
        confidence=anchor_pose.confidence,
    )


def relative_pose_from_absolute(
    target_pose: Pose,
    anchor_pose: Pose,
    *,
    anchor_id: str,
    anchor_type: AnchorType | None = None,
) -> RelativePose:
    """Convert a measured target into a persistable anchor-relative transform."""

    if target_pose.frame_id != anchor_pose.frame_id:
        raise BindingError("target_pose and anchor_pose must be expressed in the same frame")
    relative = compose_transforms(
        invert_transform(pose_as_transform(anchor_pose)),
        pose_as_transform(target_pose),
    )
    return RelativePose(
        anchor_id=anchor_id,
        anchor_type=anchor_type,
        position_m=relative.translation_m,
        orientation_xyzw=relative.rotation_xyzw,
    )


# Compatibility verb with an explicit safety meaning.
bind_anchor_relative_pose = bind_relative_pose
