"""Global force-mode policy, monitoring, and guaranteed cleanup."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from robot_skill_system.runtime.errors import ForceSafetyError, ProfileNotApprovedError


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _text(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _vector(value: Any, expected: int | None = None) -> tuple[float, ...]:
    result: tuple[float, ...]
    if all(hasattr(value, component) for component in ("x", "y", "z")):
        result = (float(value.x), float(value.y), float(value.z))
    else:
        result = tuple(float(item) for item in value)
    if expected is not None and len(result) != expected:
        raise ForceSafetyError(f"force profile vector must have {expected} values")
    if not all(math.isfinite(item) for item in result):
        raise ForceSafetyError("force profile contains a non-finite value")
    return result


@dataclass(slots=True)
class ActiveForceMode:
    profile_id: str
    target_surface_id: str
    contact_link: str
    maximum_force_n: float
    minimum_contact_force_n: float
    tangential_force_limit_n: float
    release_time_s: float
    surface_normal_xyz: tuple[float, float, float]
    expected_normal_direction: float


class GlobalForceSupervisor:
    """Resolve force numbers only from approved profiles and own force-mode lifetime."""

    def __init__(
        self,
        robot: Any,
        approved_profiles: Mapping[str, Any],
        *,
        active: bool = True,
        allowed_contact_links: Sequence[str] = (),
        maximum_force_n: float | None = None,
    ) -> None:
        self.robot = robot
        self.approved_profiles = approved_profiles
        self.active = active
        self._configured_contact_links = frozenset(allowed_contact_links)
        self.allowed_contact_links = self._configured_contact_links
        self.maximum_force_n = maximum_force_n
        self.current: ActiveForceMode | None = None

    @property
    def force_active(self) -> bool:
        return self.current is not None

    def enable(
        self,
        profile_id: str,
        *,
        tool_class: str,
        surface_role: str,
        target_surface_id: str,
        surface_normal_xyz: Sequence[float],
        contact_link: str,
        contact_search_succeeded: bool,
        ik_passed: bool,
    ) -> None:
        if not self.active:
            raise ForceSafetyError("force supervisor is inactive")
        if self.current is not None:
            raise ForceSafetyError("force mode is already active")
        profile = self.approved_profiles.get(profile_id)
        if profile is None:
            raise ProfileNotApprovedError(f"force profile {profile_id!r} is not approved")
        if surface_role != "contact_target":
            raise ForceSafetyError("force is allowed only on a contact_target surface")
        allowed_tools = {_text(item) for item in _get(profile, "allowed_tool_classes", default=())}
        if tool_class not in allowed_tools:
            raise ForceSafetyError(
                f"tool class {tool_class!r} is incompatible with force profile {profile_id!r}"
            )
        allowed_roles = {_text(item) for item in _get(profile, "allowed_surface_roles", default=())}
        if surface_role not in allowed_roles:
            raise ForceSafetyError(
                f"surface role {surface_role!r} is incompatible with force profile {profile_id!r}"
            )
        if not self.allowed_contact_links:
            raise ForceSafetyError("no globally allowed contact links are configured")
        if contact_link not in self.allowed_contact_links:
            raise ForceSafetyError(f"contact link {contact_link!r} is not globally allowed")
        if not contact_search_succeeded:
            raise ForceSafetyError("contact search has not succeeded")
        if not ik_passed:
            raise ForceSafetyError("IK/singularity validation has not passed")

        target_force = _vector(
            _get(profile, "target_force_vector_n", "target_force_vector"), expected=3
        )
        axes = tuple(
            bool(item)
            for item in _get(profile, "force_control_axes", default=(False, False, True))
        )
        if len(axes) == 3:
            axes = (*axes, False, False, False)
        if len(axes) != 6 or not any(axes):
            raise ForceSafetyError("force profile must define three or six force/compliance axes")
        normal = _vector(surface_normal_xyz, expected=3)
        normal_norm = math.sqrt(sum(item * item for item in normal))
        if normal_norm <= 1e-12:
            raise ForceSafetyError("surface normal is zero")
        normalized_normal = tuple(item / normal_norm for item in normal)
        active_alignment = sum(
            abs(normalized_normal[index])
            for index, enabled in enumerate(axes[:3])
            if enabled
        )
        inactive_force = sum(
            abs(target_force[index])
            for index, enabled in enumerate(axes[:3])
            if not enabled
        )
        if active_alignment < 0.5 or inactive_force > 1e-9:
            raise ForceSafetyError("force axes do not align with the trusted surface normal")

        target_normal_force = sum(
            target_force[index] * normalized_normal[index] for index in range(3)
        )
        target_tangential = math.sqrt(
            sum(
                (
                    target_force[index]
                    - target_normal_force * normalized_normal[index]
                )
                ** 2
                for index in range(3)
            )
        )
        if abs(target_normal_force) <= 1e-9:
            raise ForceSafetyError("target force has no component along the trusted normal")

        stiffness = _vector(_get(profile, "stiffness_n_m", "stiffness"))
        maximum_force_n = float(_get(profile, "maximum_force_n", "maximum_force"))
        minimum_force_n = float(
            _get(profile, "minimum_contact_force_n", "minimum_contact_force", default=0.0)
        )
        release_time_s = float(
            _get(profile, "force_release_time_s", "force_release_time", default=0.1)
        )
        tangential_limit = _get(profile, "tangential_force_limit_n")
        if tangential_limit is None:
            raise ForceSafetyError(
                "force profile has no approved tangential_force_limit_n"
            )
        tangential_force_limit_n = float(tangential_limit)
        if maximum_force_n <= 0 or minimum_force_n < 0 or minimum_force_n > maximum_force_n:
            raise ForceSafetyError("force profile limits are invalid")
        if (
            not math.isfinite(tangential_force_limit_n)
            or tangential_force_limit_n <= 0.0
            or tangential_force_limit_n > maximum_force_n
        ):
            raise ForceSafetyError("force profile tangential limit is invalid")
        if target_tangential > tangential_force_limit_n + 1e-9:
            raise ForceSafetyError(
                "target force exceeds the approved tangential force limit"
            )
        if self.maximum_force_n is not None and maximum_force_n > self.maximum_force_n:
            raise ForceSafetyError(
                f"force profile maximum {maximum_force_n:.3f} N exceeds loaded "
                f"safety policy maximum {self.maximum_force_n:.3f} N"
            )
        if release_time_s < 0:
            raise ForceSafetyError("force release time cannot be negative")

        self.robot.start_compliance(stiffness_n_m=stiffness)
        try:
            self.robot.set_desired_force(
                force_vector_n=target_force, force_control_axes=axes
            )
        except BaseException:
            self.robot.release_compliance()
            raise
        self.current = ActiveForceMode(
            profile_id=profile_id,
            target_surface_id=target_surface_id,
            contact_link=contact_link,
            maximum_force_n=maximum_force_n,
            minimum_contact_force_n=minimum_force_n,
            tangential_force_limit_n=tangential_force_limit_n,
            release_time_s=release_time_s,
            surface_normal_xyz=normalized_normal,  # type: ignore[arg-type]
            expected_normal_direction=math.copysign(1.0, target_normal_force),
        )

    def monitor(
        self, *, expected_profile_id: str | None = None, require_contact: bool = True
    ) -> None:
        if self.current is None:
            raise ForceSafetyError("force monitor invoked outside force mode")
        if expected_profile_id is not None and self.current.profile_id != expected_profile_id:
            raise ForceSafetyError("active force profile does not match expected profile")
        force = _vector(self.robot.read_tool_force())
        if len(force) < 3:
            raise ForceSafetyError("tool force measurement must provide at least three axes")
        translational = force[:3]
        force_norm = math.sqrt(sum(component**2 for component in translational))
        if force_norm > self.current.maximum_force_n:
            raise ForceSafetyError(
                f"maximum force exceeded: {force_norm:.3f} N > {self.current.maximum_force_n:.3f} N"
            )
        normal_force = sum(
            translational[index] * self.current.surface_normal_xyz[index]
            for index in range(3)
        )
        tangential_force = math.sqrt(
            sum(
                (
                    translational[index]
                    - normal_force * self.current.surface_normal_xyz[index]
                )
                ** 2
                for index in range(3)
            )
        )
        if tangential_force > self.current.tangential_force_limit_n:
            raise ForceSafetyError(
                f"tangential force exceeded: {tangential_force:.3f} N > "
                f"{self.current.tangential_force_limit_n:.3f} N"
            )
        if require_contact and abs(normal_force) < self.current.minimum_contact_force_n:
            raise ForceSafetyError(
                f"contact force lost: normal={abs(normal_force):.3f} N < "
                f"{self.current.minimum_contact_force_n:.3f} N"
            )
        if (
            require_contact
            and normal_force * self.current.expected_normal_direction <= 0.0
        ):
            raise ForceSafetyError(
                "measured force points opposite the approved surface-normal direction"
            )

    def authorize_contact_links(self, contact_links: Sequence[str]) -> None:
        """Install a non-empty trusted link set before force mode is entered."""

        if self.current is not None:
            raise ForceSafetyError("cannot change allowed contact links during force mode")
        normalized = frozenset(str(link) for link in contact_links if str(link))
        if not normalized:
            raise ForceSafetyError("no explicit allowed contact links were supplied")
        if self._configured_contact_links:
            normalized = self._configured_contact_links & normalized
            if not normalized:
                raise ForceSafetyError(
                    "scene contact links do not intersect the global contact-link policy"
                )
        self.allowed_contact_links = normalized

    def disable(self) -> None:
        current = self.current
        if current is None:
            return
        try:
            self.robot.release_force(release_time_s=current.release_time_s)
        finally:
            try:
                self.robot.release_compliance()
            finally:
                self.current = None

    def emergency_release(self) -> tuple[float, float, float] | None:
        """Release force/compliance and return the verified outward retract direction."""

        current = self.current
        if current is None:
            return None
        normal = current.surface_normal_xyz
        self.disable()
        return normal

    def assert_released(self) -> None:
        if self.current is not None:
            raise ForceSafetyError("force mode remained active after skill completion")


ForceSupervisor = GlobalForceSupervisor

__all__ = ["ActiveForceMode", "ForceSupervisor", "GlobalForceSupervisor"]
