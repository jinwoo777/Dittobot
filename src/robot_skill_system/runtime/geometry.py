"""Deterministic offline point/path checks against conservative scene geometry.

This module is intentionally small and dependency free.  It is not a robot-link
collision checker and therefore never claims to provide hardware evidence.  It
does, however, make the mock runtime useful for safety tests: anchor-relative TCP
targets and explicit paths are checked against every supported static/dynamic
obstacle and forbidden workspace volume.  Unsupported relevant geometry fails
closed instead of being treated as free space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _finite_xyz(value: Any, *, label: str) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{label} must contain exactly three values")
        result = (float(value[0]), float(value[1]), float(value[2]))
    else:
        result = (
            float(_get(value, "x")),
            float(_get(value, "y")),
            float(_get(value, "z")),
        )
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _finite_xyzw(value: Any) -> tuple[float, float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0, 1.0)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise ValueError("obstacle orientation must contain exactly four values")
        result = tuple(float(component) for component in value)
    else:
        result = (
            float(_get(value, "x")),
            float(_get(value, "y")),
            float(_get(value, "z")),
            float(_get(value, "w")),
        )
    if not all(math.isfinite(component) for component in result):
        raise ValueError("obstacle orientation must contain only finite values")
    norm = math.sqrt(sum(component * component for component in result))
    if norm <= 1e-12:
        raise ValueError("obstacle orientation quaternion is zero")
    return tuple(component / norm for component in result)  # type: ignore[return-value]


def _inverse_rotate(
    quaternion_xyzw: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion_xyzw
    # Rotation by the quaternion conjugate.
    cx, cy, cz = -qx, -qy, -qz
    vx, vy, vz = vector
    tx = 2.0 * (cy * vz - cz * vy)
    ty = 2.0 * (cz * vx - cx * vz)
    tz = 2.0 * (cx * vy - cy * vx)
    return (
        vx + qw * tx + (cy * tz - cz * ty),
        vy + qw * ty + (cz * tx - cx * tz),
        vz + qw * tz + (cx * ty - cy * tx),
    )


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GeometryAssessment:
    """Result from a deterministic target/path assessment."""

    passed: bool
    detail: str
    minimum_distance_m: float | None = None
    obstacle_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Volume:
    identifier: str
    shape: str
    center_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    dimensions_m: tuple[float, ...]
    additional_clearance_m: float = 0.0

    def local_point(
        self, point_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _inverse_rotate(self.orientation_xyzw, _subtract(point_m, self.center_m))


class DeterministicSceneGeometry:
    """Conservative offline geometry checks for explicit TCP targets and paths.

    Supported obstacle/region geometry is ``sphere`` (``radius_m``), ``box``
    (``size_m``), and ``aabb`` (``min_m``/``max_m``).  Boxes may use the pose
    quaternion.  Dynamic obstacle volumes are inflated by their declared speed
    over the scene validity window.  Meshes, unreadable occupancy maps, frame
    mismatches, and malformed geometry fail closed whenever a relevant target or
    path is checked.
    """

    def __init__(self, *, minimum_clearance_m: float = 0.03) -> None:
        if not math.isfinite(minimum_clearance_m) or minimum_clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be finite and non-negative")
        self.minimum_clearance_m = minimum_clearance_m

    def validate_target(self, *, target: Any, scene: Any) -> GeometryAssessment:
        try:
            point = self._target_point(target)
        except (TypeError, ValueError) as exc:
            return GeometryAssessment(False, f"target cannot be checked: {exc}")
        return self._validate_points_and_segments((point,), (), scene)

    def validate_path(self, *, path: Any, scene: Any) -> GeometryAssessment:
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or not path:
            return GeometryAssessment(False, "path cannot be checked: no explicit waypoints")
        try:
            points = tuple(self._target_point(target) for target in path)
        except (TypeError, ValueError) as exc:
            return GeometryAssessment(False, f"path cannot be checked: {exc}")
        segments = tuple(zip(points, points[1:], strict=False))
        return self._validate_points_and_segments(points, segments, scene)

    def validate_periodic_envelope(
        self, *, center: Any, amplitude_m: Any, scene: Any
    ) -> GeometryAssessment:
        """Conservatively treat a periodic trajectory as its full axis-aligned envelope."""

        try:
            center_point = self._target_point(center)
            amplitude = _finite_xyz(amplitude_m, label="periodic amplitude")
        except (TypeError, ValueError) as exc:
            return GeometryAssessment(False, f"periodic path cannot be checked: {exc}")
        amplitude = (abs(amplitude[0]), abs(amplitude[1]), abs(amplitude[2]))
        corners = tuple(
            (
                center_point[0] + sx * amplitude[0],
                center_point[1] + sy * amplitude[1],
                center_point[2] + sz * amplitude[2],
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        )
        # Twelve box edges plus the centre/extrema are a conservative collision
        # proxy only when obstacle volumes are inflated by testing box overlap.
        # `_validate_envelope` performs that exact conservative overlap check.
        return self._validate_envelope(center_point, amplitude, corners, scene)

    @staticmethod
    def _target_point(target: Any) -> tuple[float, float, float]:
        position = _get(target, "position_m", "position")
        if position is None:
            raise ValueError("target has no bound position_m")
        return _finite_xyz(position, label="target position_m")

    def _validate_points_and_segments(
        self,
        points: Sequence[tuple[float, float, float]],
        segments: Sequence[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ],
        scene: Any,
    ) -> GeometryAssessment:
        preparation = self._prepare_scene(scene)
        if isinstance(preparation, GeometryAssessment):
            return preparation
        forbidden, allowed, requires_observed_space = preparation
        for point in points:
            space = self._validate_observed_space(point, allowed, requires_observed_space)
            if space is not None:
                return space
        minimum_distance = math.inf
        closest_id: str | None = None
        for volume in forbidden:
            clearance = self.minimum_clearance_m + volume.additional_clearance_m
            for point in points:
                distance = self._point_distance(point, volume)
                if distance < minimum_distance:
                    minimum_distance, closest_id = distance, volume.identifier
                if distance < clearance:
                    return GeometryAssessment(
                        False,
                        f"target is {distance:.4f} m from {volume.identifier!r}; "
                        f"required clearance is {clearance:.4f} m",
                        distance,
                        volume.identifier,
                    )
            for start, end in segments:
                if self._segment_intersects_inflated_volume(start, end, volume, clearance):
                    return GeometryAssessment(
                        False,
                        f"path intersects the {clearance:.4f} m clearance volume of "
                        f"{volume.identifier!r}",
                        0.0,
                        volume.identifier,
                    )
        reported = None if math.isinf(minimum_distance) else minimum_distance
        detail = "all explicit targets/path segments satisfy offline scene clearance"
        if reported is not None and closest_id is not None:
            detail += f"; nearest={closest_id!r} at {reported:.4f} m"
        return GeometryAssessment(True, detail, reported, closest_id)

    def _validate_envelope(
        self,
        center: tuple[float, float, float],
        half_size: tuple[float, float, float],
        corners: Sequence[tuple[float, float, float]],
        scene: Any,
    ) -> GeometryAssessment:
        preparation = self._prepare_scene(scene)
        if isinstance(preparation, GeometryAssessment):
            return preparation
        forbidden, allowed, requires_observed_space = preparation
        for point in (center, *corners):
            space = self._validate_observed_space(point, allowed, requires_observed_space)
            if space is not None:
                return space
        for volume in forbidden:
            clearance = self.minimum_clearance_m + volume.additional_clearance_m
            if self._aabb_overlaps_inflated_volume(center, half_size, volume, clearance):
                return GeometryAssessment(
                    False,
                    f"periodic path envelope intersects the {clearance:.4f} m clearance "
                    f"volume of {volume.identifier!r}",
                    0.0,
                    volume.identifier,
                )
        return GeometryAssessment(
            True, "periodic path envelope satisfies offline scene clearance"
        )

    def _prepare_scene(
        self, scene: Any
    ) -> tuple[list[_Volume], list[_Volume], bool] | GeometryAssessment:
        if _get(scene, "occupancy_map_uri") is not None:
            return GeometryAssessment(
                False,
                "scene occupancy map cannot be checked by the deterministic mock validator",
            )
        reference_frame = str(_get(scene, "reference_frame", default=""))
        forbidden: list[_Volume] = []
        allowed: list[_Volume] = []
        requires_observed_space = False
        try:
            for dynamic, collection_name in (
                (False, "static_obstacles"),
                (True, "dynamic_obstacles"),
            ):
                collection = _get(scene, collection_name, default=()) or ()
                if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes)):
                    raise ValueError(f"scene {collection_name} is not a sequence")
                for obstacle in collection:
                    forbidden.append(
                        self._obstacle_volume(
                            obstacle,
                            reference_frame=reference_frame,
                            dynamic=dynamic,
                            valid_for_ms=int(_get(scene, "valid_for_ms", default=0)),
                        )
                    )
            regions = _get(scene, "workspace_regions", "workspaces", default=()) or ()
            if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
                raise ValueError("scene workspace regions are not a sequence")
            for region in regions:
                policy = _text(_get(region, "access_policy", default="forbidden"))
                geometry = _get(region, "geometry", default={})
                shape = _text(_get(geometry, "type", "shape", default=""))
                role = _text(_get(region, "role", default=""))
                if shape == "complement_of_observed_frustum":
                    if policy in {"forbidden", "occupied"} or role == "unknown_region":
                        requires_observed_space = True
                        continue
                    raise ValueError("observed-space complement cannot be an allowed region")
                volume = self._workspace_volume(region, reference_frame=reference_frame)
                if policy in {"allowed", "supervised"}:
                    allowed.append(volume)
                else:
                    forbidden.append(volume)
        except (TypeError, ValueError) as exc:
            return GeometryAssessment(False, f"scene geometry cannot be checked: {exc}")
        if requires_observed_space and not allowed:
            return GeometryAssessment(
                False,
                "unknown space is forbidden but no supported observed-space volume is available",
            )
        return forbidden, allowed, requires_observed_space

    def _obstacle_volume(
        self,
        obstacle: Any,
        *,
        reference_frame: str,
        dynamic: bool,
        valid_for_ms: int,
    ) -> _Volume:
        identifier = str(_get(obstacle, "instance_id", "id", default="unknown_obstacle"))
        pose = _get(obstacle, "pose")
        if pose is None:
            raise ValueError(f"obstacle {identifier!r} has no pose")
        frame = str(_get(pose, "frame_id", default=""))
        if frame != reference_frame:
            raise ValueError(
                f"obstacle {identifier!r} frame {frame!r} does not match {reference_frame!r}"
            )
        geometry = _get(obstacle, "geometry")
        volume = self._volume_from_geometry(identifier, geometry, pose=pose)
        if not dynamic:
            return volume
        velocity = _get(obstacle, "velocity_m_s")
        if velocity is None:
            # A dynamic obstacle with unknown velocity is not predictable.  Inflate by
            # one global clearance rather than silently treating it as static.
            inflation = self.minimum_clearance_m
        else:
            speed = math.sqrt(
                sum(component * component for component in _finite_xyz(velocity, label="velocity"))
            )
            inflation = speed * max(valid_for_ms, 0) / 1_000.0
        return _Volume(
            identifier=volume.identifier,
            shape=volume.shape,
            center_m=volume.center_m,
            orientation_xyzw=volume.orientation_xyzw,
            dimensions_m=volume.dimensions_m,
            additional_clearance_m=inflation,
        )

    def _workspace_volume(self, region: Any, *, reference_frame: str) -> _Volume:
        identifier = str(_get(region, "region_id", "id", default="unknown_region"))
        frame = str(_get(region, "frame_id", default=""))
        if frame != reference_frame:
            raise ValueError(
                f"workspace {identifier!r} frame {frame!r} does not match {reference_frame!r}"
            )
        geometry = _get(region, "geometry")
        volume = self._volume_from_geometry(identifier, geometry, pose=None)
        return _Volume(
            identifier=volume.identifier,
            shape=volume.shape,
            center_m=volume.center_m,
            orientation_xyzw=volume.orientation_xyzw,
            dimensions_m=volume.dimensions_m,
            additional_clearance_m=float(_get(region, "minimum_clearance_m", default=0.0)),
        )

    @staticmethod
    def _volume_from_geometry(identifier: str, geometry: Any, *, pose: Any) -> _Volume:
        if not isinstance(geometry, Mapping):
            raise ValueError(f"geometry for {identifier!r} is not an object")
        shape = _text(_get(geometry, "type", "shape", default="")).lower()
        pose_position = _get(pose, "position_m", "position") if pose is not None else None
        center_value = _get(geometry, "center_m", "center", default=pose_position)
        orientation_value = (
            _get(pose, "orientation_xyzw", "orientation") if pose is not None else None
        )
        if shape in {"box", "cuboid"}:
            center = _finite_xyz(center_value, label=f"{identifier} center_m")
            size = _finite_xyz(
                _get(geometry, "size_m", "dimensions_m", "size"),
                label=f"{identifier} size_m",
            )
            if min(size) <= 0.0:
                raise ValueError(f"box {identifier!r} size_m must be positive")
            return _Volume(
                identifier,
                "box",
                center,
                _finite_xyzw(orientation_value),
                (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0),
            )
        if shape == "aabb":
            minimum = _finite_xyz(_get(geometry, "min_m"), label=f"{identifier} min_m")
            maximum = _finite_xyz(_get(geometry, "max_m"), label=f"{identifier} max_m")
            if any(maximum[index] <= minimum[index] for index in range(3)):
                raise ValueError(f"AABB {identifier!r} extents must be positive")
            center = (
                (minimum[0] + maximum[0]) / 2.0,
                (minimum[1] + maximum[1]) / 2.0,
                (minimum[2] + maximum[2]) / 2.0,
            )
            half = (
                (maximum[0] - minimum[0]) / 2.0,
                (maximum[1] - minimum[1]) / 2.0,
                (maximum[2] - minimum[2]) / 2.0,
            )
            return _Volume(
                identifier,
                "box",
                center,
                (0.0, 0.0, 0.0, 1.0),
                half,
            )
        if shape == "sphere":
            center = _finite_xyz(center_value, label=f"{identifier} center_m")
            radius = float(_get(geometry, "radius_m", "radius"))
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"sphere {identifier!r} radius_m must be positive")
            return _Volume(
                identifier,
                "sphere",
                center,
                (0.0, 0.0, 0.0, 1.0),
                (radius,),
            )
        raise ValueError(f"unsupported geometry type {shape!r} for {identifier!r}")

    @staticmethod
    def _point_distance(point: tuple[float, float, float], volume: _Volume) -> float:
        local = volume.local_point(point)
        if volume.shape == "sphere":
            return (
                math.sqrt(sum(component * component for component in local))
                - volume.dimensions_m[0]
            )
        half = volume.dimensions_m
        outside = tuple(max(abs(local[index]) - half[index], 0.0) for index in range(3))
        outside_distance = math.sqrt(sum(component * component for component in outside))
        if outside_distance > 0.0:
            return outside_distance
        return -min(half[index] - abs(local[index]) for index in range(3))

    @classmethod
    def _segment_intersects_inflated_volume(
        cls,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        volume: _Volume,
        clearance_m: float,
    ) -> bool:
        local_start = volume.local_point(start)
        local_end = volume.local_point(end)
        if volume.shape == "sphere":
            radius = volume.dimensions_m[0] + clearance_m
            direction = _subtract(local_end, local_start)
            length_squared = sum(component * component for component in direction)
            if length_squared <= 1e-18:
                return sum(component * component for component in local_start) <= radius**2
            projection = -sum(
                local_start[index] * direction[index] for index in range(3)
            ) / length_squared
            clamped = min(1.0, max(0.0, projection))
            closest = tuple(
                local_start[index] + clamped * direction[index] for index in range(3)
            )
            return sum(component * component for component in closest) <= radius**2
        half = (
            volume.dimensions_m[0] + clearance_m,
            volume.dimensions_m[1] + clearance_m,
            volume.dimensions_m[2] + clearance_m,
        )
        return cls._segment_intersects_aabb(local_start, local_end, half)

    @staticmethod
    def _segment_intersects_aabb(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        half_size: tuple[float, float, float],
    ) -> bool:
        t_min, t_max = 0.0, 1.0
        for axis in range(3):
            delta = end[axis] - start[axis]
            if abs(delta) <= 1e-15:
                if start[axis] < -half_size[axis] or start[axis] > half_size[axis]:
                    return False
                continue
            first = (-half_size[axis] - start[axis]) / delta
            second = (half_size[axis] - start[axis]) / delta
            if first > second:
                first, second = second, first
            t_min = max(t_min, first)
            t_max = min(t_max, second)
            if t_min > t_max:
                return False
        return True

    @classmethod
    def _aabb_overlaps_inflated_volume(
        cls,
        center: tuple[float, float, float],
        half_size: tuple[float, float, float],
        volume: _Volume,
        clearance_m: float,
    ) -> bool:
        # Checking all AABB corners and obstacle centre is conservative for an
        # oriented box only if we also use a circumscribed sphere.  This may reject
        # some clear periodic paths, but cannot miss an overlap in the mock gate.
        envelope_radius = math.sqrt(sum(component * component for component in half_size))
        distance = cls._point_distance(center, volume)
        return distance <= clearance_m + envelope_radius

    def _contains(self, point: tuple[float, float, float], volume: _Volume) -> bool:
        local = volume.local_point(point)
        # Workspace regions are coarse observed-space estimates.  Their declared
        # region clearance plus the global clearance is the bounded uncertainty
        # margin used for this membership check.
        margin = self.minimum_clearance_m + volume.additional_clearance_m
        if volume.shape == "sphere":
            return sum(component * component for component in local) <= (
                volume.dimensions_m[0] + margin
            ) ** 2
        return all(
            abs(local[index]) <= volume.dimensions_m[index] + margin + 1e-12
            for index in range(3)
        )

    def _validate_observed_space(
        self,
        point: tuple[float, float, float],
        allowed: Sequence[_Volume],
        required: bool,
    ) -> GeometryAssessment | None:
        if required and not any(self._contains(point, volume) for volume in allowed):
            return GeometryAssessment(
                False,
                "target/path enters unknown space outside all supported observed regions",
                0.0,
                "unknown_space",
            )
        return None


__all__ = ["DeterministicSceneGeometry", "GeometryAssessment"]
