"""Scene-level validation helpers used before binding and execution."""

from __future__ import annotations

from dataclasses import dataclass

from robot_skill_system.exceptions import SceneValidationError
from robot_skill_system.scene.models import SceneSnapshot


@dataclass(frozen=True)
class SceneValidationReport:
    """Result of deterministic, local scene validation."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class SceneValidator:
    """Validate freshness and minimum confidence without perception dependencies."""

    def __init__(self, *, minimum_confidence: float = 0.5) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between zero and one")
        self._minimum_confidence = minimum_confidence

    def inspect(
        self,
        scene: SceneSnapshot,
        *,
        now_ns: int | None = None,
        required_freshness_ms: int | None = None,
    ) -> SceneValidationReport:
        """Return all locally detectable scene errors without raising."""

        errors: list[str] = []
        warnings: list[str] = []
        if not scene.is_fresh(now_ns, required_freshness_ms):
            errors.append("scene snapshot is stale or timestamped in the future")
        if scene.confidence_summary.overall < self._minimum_confidence:
            errors.append("scene confidence is below the configured threshold")
        if scene.camera_metadata is None:
            warnings.append("scene has no camera metadata")
        if not scene.workspace_regions:
            warnings.append("scene has no explicit workspace regions")
        return SceneValidationReport(not errors, tuple(errors), tuple(warnings))

    def validate(
        self,
        scene: SceneSnapshot,
        *,
        now_ns: int | None = None,
        required_freshness_ms: int | None = None,
    ) -> SceneValidationReport:
        """Return a valid report or raise ``SceneValidationError``."""

        report = self.inspect(
            scene,
            now_ns=now_ns,
            required_freshness_ms=required_freshness_ms,
        )
        if not report.valid:
            raise SceneValidationError("; ".join(report.errors))
        return report


def validate_scene_snapshot(
    scene: SceneSnapshot,
    *,
    now_ns: int | None = None,
    required_freshness_ms: int | None = None,
    minimum_confidence: float = 0.5,
) -> SceneValidationReport:
    """Convenience wrapper around :class:`SceneValidator`."""

    return SceneValidator(minimum_confidence=minimum_confidence).validate(
        scene,
        now_ns=now_ns,
        required_freshness_ms=required_freshness_ms,
    )
