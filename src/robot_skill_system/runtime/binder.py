"""Deterministic entity selection and anchor-relative pose rebinding."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from robot_skill_system.runtime.errors import (
    AmbiguousBindingError,
    BindingError,
    SceneStaleError,
)
from robot_skill_system.runtime.models import (
    BoundTargetPose,
    EntityBinding,
    EntityKind,
    EntityRequirement,
)


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise BindingError("position must contain exactly three values")
        return float(value[0]), float(value[1]), float(value[2])
    return float(_get(value, "x")), float(_get(value, "y")), float(_get(value, "z"))


def _xyzw(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise BindingError("quaternion must contain exactly four values")
        result = tuple(float(item) for item in value)
    else:
        result = (
            float(_get(value, "x")),
            float(_get(value, "y")),
            float(_get(value, "z")),
            float(_get(value, "w")),
        )
    norm = math.sqrt(sum(component * component for component in result))
    if norm <= 1e-12:
        raise BindingError("quaternion norm is zero")
    return tuple(component / norm for component in result)  # type: ignore[return-value]


def _multiply_quaternion(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    # Optimized q * (v, 0) * conjugate(q).
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def bind_relative_pose(
    anchor_pose: Any,
    relative_pose: Any,
    *,
    anchor_entity_id: str,
    timestamp_ns: int,
    target_frame_id: str | None = None,
) -> BoundTargetPose:
    """Compute ``T_base_anchor_current * T_anchor_target_stored`` locally."""

    anchor_position = _xyz(_get(anchor_pose, "position_m", "position"))
    anchor_orientation = _xyzw(
        _get(anchor_pose, "orientation_xyzw", "orientation", default=(0.0, 0.0, 0.0, 1.0))
    )
    relative_position = _xyz(_get(relative_pose, "position_m", "position"))
    relative_orientation = _xyzw(
        _get(relative_pose, "orientation_xyzw", "orientation", default=(0.0, 0.0, 0.0, 1.0))
    )
    offset = _rotate_vector(anchor_orientation, relative_position)
    position = tuple(anchor_position[index] + offset[index] for index in range(3))
    orientation = _multiply_quaternion(anchor_orientation, relative_orientation)
    return BoundTargetPose(
        frame_id=target_frame_id or str(_get(anchor_pose, "frame_id", default="base")),
        position_m=position,  # type: ignore[arg-type]
        orientation_xyzw=orientation,
        anchor_entity_id=anchor_entity_id,
        timestamp_ns=timestamp_ns,
    )


class EntityBinder:
    def __init__(self, *, clock_ns: Any = time.time_ns) -> None:
        self._clock_ns = clock_ns

    def ensure_scene_fresh(self, scene: Any, *, maximum_age_ms: int | None = None) -> None:
        timestamp_ns = int(_get(scene, "timestamp_ns"))
        valid_for_ms = int(_get(scene, "valid_for_ms", default=0))
        allowed_ms = valid_for_ms if maximum_age_ms is None else min(valid_for_ms, maximum_age_ms)
        age_ns = self._clock_ns() - timestamp_ns
        if age_ns < 0:
            raise SceneStaleError("scene timestamp is in the future")
        if allowed_ms <= 0 or age_ns > allowed_ms * 1_000_000:
            raise SceneStaleError(
                f"scene is stale: age_ms={age_ns / 1_000_000:.1f}, allowed_ms={allowed_ms}"
            )

    def bind_entities(
        self,
        scene: Any,
        requirements: Sequence[EntityRequirement | Mapping[str, Any] | Any],
        *,
        require_fresh_scene: bool = True,
        maximum_scene_age_ms: int | None = None,
    ) -> dict[str, EntityBinding]:
        if require_fresh_scene:
            self.ensure_scene_fresh(scene, maximum_age_ms=maximum_scene_age_ms)
        bindings: dict[str, EntityBinding] = {}
        selected_ids: set[str] = set()
        for raw_requirement in requirements:
            requirement = self._normalize_requirement(raw_requirement)
            if not requirement.placeholder.startswith("$"):
                raise BindingError("binding placeholders must start with '$'")
            candidates = [
                candidate
                for candidate in self._collection(scene, requirement.entity_kind)
                if self._matches(candidate, requirement)
            ]
            if requirement.instance_id is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if self._entity_id(candidate, requirement.entity_kind)
                    == requirement.instance_id
                ]
            if not candidates:
                raise BindingError(f"no scene entity satisfies {requirement.placeholder}")
            if len(candidates) > 1:
                identifiers = sorted(
                    self._entity_id(candidate, requirement.entity_kind) for candidate in candidates
                )
                raise AmbiguousBindingError(
                    f"ambiguous binding {requirement.placeholder}: "
                    f"choose an instance ID from {identifiers}"
                )
            candidate = candidates[0]
            entity_id = self._entity_id(candidate, requirement.entity_kind)
            if entity_id in selected_ids:
                raise BindingError(f"entity {entity_id!r} was selected for multiple placeholders")
            selected_ids.add(entity_id)
            bindings[requirement.placeholder] = EntityBinding(
                placeholder=requirement.placeholder,
                entity_id=entity_id,
                entity_kind=requirement.entity_kind,
                confidence=float(
                    _get(candidate, "verification_confidence", "confidence", default=0.0)
                ),
                entity=candidate,
            )
        return bindings

    def bind_arguments(
        self, arguments: Mapping[str, Any], *, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> dict[str, Any]:
        return {
            str(key): self._bind_value(value, scene=scene, bindings=bindings)
            for key, value in arguments.items()
        }

    def bind_pose(
        self, relative_target: Any, *, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> BoundTargetPose:
        anchor_reference = _get(
            relative_target,
            "anchor_entity_id",
            "anchor_id",
            "anchor_frame_id",
            "entity_id",
        )
        if not isinstance(anchor_reference, str) or not anchor_reference:
            raise BindingError("relative target is missing an anchor entity ID")
        if anchor_reference in bindings:
            entity = bindings[anchor_reference].entity
            anchor_id = bindings[anchor_reference].entity_id
        else:
            entity = self._find_entity(scene, anchor_reference)
            anchor_id = anchor_reference
        anchor_pose = _get(entity, "pose", "anchor_pose", "center_pose")
        if anchor_pose is None:
            center = _get(entity, "center_m", "center")
            if center is None:
                raise BindingError(f"anchor {anchor_id!r} has no pose")
            anchor_pose = {
                "frame_id": _get(
                    entity,
                    "frame_id",
                    default=_get(scene, "reference_frame", default="base"),
                ),
                "position_m": center,
                "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
            }
        pose_in_anchor = _get(
            relative_target,
            "pose_in_anchor",
            "relative_pose",
            "pose",
            "transform",
            default=relative_target,
        )
        return bind_relative_pose(
            anchor_pose,
            pose_in_anchor,
            anchor_entity_id=anchor_id,
            timestamp_ns=int(_get(scene, "timestamp_ns", default=self._clock_ns())),
            target_frame_id=str(_get(scene, "reference_frame", default="base")),
        )

    def _bind_value(
        self, value: Any, *, scene: Any, bindings: Mapping[str, EntityBinding]
    ) -> Any:
        if isinstance(value, str) and value in bindings:
            return bindings[value].entity_id
        if self._looks_like_relative_pose(value):
            return self.bind_pose(value, scene=scene, bindings=bindings)
        if isinstance(value, Mapping):
            return {
                str(key): self._bind_value(item, scene=scene, bindings=bindings)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._bind_value(item, scene=scene, bindings=bindings) for item in value)
        if isinstance(value, list):
            return [self._bind_value(item, scene=scene, bindings=bindings) for item in value]
        if is_dataclass(value):
            # Domain dataclasses which are not relative poses remain intact. Primitive schemas can
            # still consume them without lossy reconstruction.
            _ = fields(value)
        return value

    @staticmethod
    def _looks_like_relative_pose(value: Any) -> bool:
        class_name = type(value).__name__.lower()
        if class_name == "relativepose" or class_name == "relative_pose":
            return True
        if isinstance(value, Mapping):
            has_anchor = any(
                key in value
                for key in ("anchor_entity_id", "anchor_id", "anchor_frame_id", "entity_id")
            )
            has_pose = any(
                key in value
                for key in ("pose_in_anchor", "relative_pose", "pose", "transform", "position_m")
            )
            return has_anchor and has_pose
        return hasattr(value, "anchor_entity_id") or hasattr(value, "anchor_id")

    @staticmethod
    def _normalize_requirement(
        raw: EntityRequirement | Mapping[str, Any] | Any,
    ) -> EntityRequirement:
        if isinstance(raw, EntityRequirement):
            return raw
        kind_value = _get(raw, "entity_kind", "kind")
        if isinstance(kind_value, Enum):
            kind_value = kind_value.value
        try:
            kind = kind_value if isinstance(kind_value, EntityKind) else EntityKind(str(kind_value))
        except ValueError as exc:
            raise BindingError(f"unknown entity kind: {kind_value!r}") from exc
        return EntityRequirement(
            placeholder=str(_get(raw, "placeholder", "variable")),
            entity_kind=kind,
            instance_id=_get(raw, "instance_id", "entity_id"),
            class_name=_get(raw, "class_name", "tool_class", "target_class"),
            role=_text(_get(raw, "role")),
            minimum_confidence=float(_get(raw, "minimum_confidence", default=0.7)),
            minimum_visible_fraction=float(
                _get(raw, "minimum_visible_fraction", default=0.5)
            ),
            must_be_attached=_get(raw, "must_be_attached"),
            compatible_skill=_get(raw, "compatible_skill"),
        )

    @staticmethod
    def _collection(scene: Any, kind: EntityKind) -> Sequence[Any]:
        names = {
            EntityKind.OBJECT: ("objects",),
            EntityKind.TOOL: ("tools",),
            EntityKind.SURFACE: ("surfaces",),
            EntityKind.WORKSPACE: ("workspace_regions", "workspaces"),
        }[kind]
        collection = _get(scene, *names, default=())
        if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes)):
            return collection
        raise BindingError(f"scene {kind.value} collection is not a sequence")

    @staticmethod
    def _entity_id(entity: Any, kind: EntityKind) -> str:
        names = {
            EntityKind.OBJECT: ("instance_id", "object_id", "id"),
            EntityKind.TOOL: ("instance_id", "tool_id", "id"),
            EntityKind.SURFACE: ("instance_id", "surface_id", "id"),
            EntityKind.WORKSPACE: ("region_id", "workspace_id", "id"),
        }[kind]
        identifier = _get(entity, *names)
        if not isinstance(identifier, str) or not identifier:
            raise BindingError(f"{kind.value} entity has no stable ID")
        return identifier

    @staticmethod
    def _matches(entity: Any, requirement: EntityRequirement) -> bool:
        confidence = float(_get(entity, "verification_confidence", "confidence", default=0.0))
        if confidence < requirement.minimum_confidence:
            return False
        visible = float(_get(entity, "visible_fraction", default=1.0))
        if visible < requirement.minimum_visible_fraction:
            return False
        if requirement.class_name is not None:
            actual_class = _text(
                _get(entity, "class_name", "tool_class", "surface_type", "object_class")
            )
            if actual_class != requirement.class_name:
                return False
        if requirement.role is not None and _text(_get(entity, "role")) != requirement.role:
            return False
        if requirement.must_be_attached is not None and (
            bool(_get(entity, "attached", default=False)) is not requirement.must_be_attached
        ):
            return False
        if requirement.compatible_skill is not None:
            compatible = set(_get(entity, "compatible_skills", default=()))
            if requirement.compatible_skill not in compatible:
                return False
        return True

    def _find_entity(self, scene: Any, entity_id: str) -> Any:
        for kind in EntityKind:
            for entity in self._collection(scene, kind):
                if self._entity_id(entity, kind) == entity_id:
                    return entity
        raise BindingError(f"anchor entity {entity_id!r} is absent from current scene")


RuntimeBinder = EntityBinder

__all__ = ["EntityBinder", "RuntimeBinder", "bind_relative_pose"]
