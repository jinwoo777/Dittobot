"""Strict offline importer for legacy grip-point experiment results.

The experiment contains useful 2D evidence, but not executable 6D object-relative targets.
Imported versions are therefore always candidate-only, non-hardware-compatible, and forbidden
from automatic activation. The detailed model/evidence document is written to the configured
artifact store; SQLite receives only catalog/version metadata and URI/checksum references.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import StorageRepository

_CANONICAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ImportedGripPointProfile:
    object_class_id: str
    catalog_entry_id: str
    grip_profile_id: str
    grip_profile_version_id: str
    artifact_uri: str
    artifact_checksum_sha256: str


@dataclass(frozen=True, slots=True)
class GripPointImportSummary:
    source_checksum_sha256: str
    profiles: tuple[ImportedGripPointProfile, ...]


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


class GripPointResultImporter:
    """Import bounded, data-only grip diagnostics without contacting a model or robot."""

    def __init__(
        self,
        repository: StorageRepository,
        artifact_store: LocalArtifactStore,
        *,
        maximum_source_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if maximum_source_bytes <= 0:
            raise ValueError("maximum_source_bytes must be positive")
        self.repository = repository
        self.artifact_store = artifact_store
        self.maximum_source_bytes = maximum_source_bytes

    def import_file(self, result_path: Path) -> GripPointImportSummary:
        source_path = result_path.resolve(strict=True)
        if not source_path.is_file():
            raise ValueError("grip-point result path must identify a regular file")
        source_size = source_path.stat().st_size
        if source_size > self.maximum_source_bytes:
            raise ValueError("grip-point result exceeds the configured import size limit")
        source_bytes = source_path.read_bytes()
        if len(source_bytes) != source_size:
            raise ValueError("grip-point result changed while it was being read")
        source_checksum = hashlib.sha256(source_bytes).hexdigest()
        try:
            decoded = source_bytes.decode("utf-8")
            payload = json.loads(
                decoded,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("grip-point result must be valid UTF-8 JSON") from exc
        root = _require_mapping(payload, field="result")
        objects = self._validate_root(root)

        imported: list[ImportedGripPointProfile] = []
        for object_class_id in sorted(objects):
            object_result = objects[object_class_id]
            self._validate_object(object_class_id, object_result)
            artifact_payload = {
                "schema_version": "grip-point-diagnostic-profile/1.0",
                "object_class_id": object_class_id,
                "source_result_checksum_sha256": source_checksum,
                "execution_mode": "offline_diagnostic_only",
                "hardware_compatible": False,
                "auto_activation_allowed": False,
                "approach_direction_estimated": False,
                "limitations": root.get("limitations", []),
                "evidence": object_result,
            }
            artifact_uri = (
                f"grip_profiles/candidates/{object_class_id}/"
                f"grip-point-{source_checksum[:16]}.json"
            )
            artifact = self.artifact_store.put_json(artifact_uri, artifact_payload)

            catalog = self.repository.get_catalog_entry(
                kind="object", identifier=object_class_id
            )
            if catalog is None:
                catalog = self.repository.create_catalog_entry(
                    kind="object",
                    canonical_id=object_class_id,
                    display_name=object_class_id,
                    metadata={"discovered_by": "grip_point_result_importer"},
                )
            profile = self.repository.ensure_grip_profile(object_class_id=object_class_id)
            version = self.repository.register_grip_profile_version(
                object_class_id=object_class_id,
                semantic_version=f"0.0.0+grip-point.{source_checksum[:12]}-candidate",
                artifact_uri=artifact.uri,
                artifact_checksum_sha256=artifact.checksum_sha256,
                object_frame_policy="dataset_object_axis_2d_diagnostic",
                object_frame_revision=str(root["schema_version"]),
                status="candidate",
                validation_status="pending",
                hardware_compatible=False,
                auto_activation_allowed=False,
                gripper_calibration_profile_id=None,
                metadata={
                    "source_result_checksum_sha256": source_checksum,
                    "importer": "grip_point_result_importer/1.0",
                    "diagnostic_only": True,
                    "operator_contact_confirmed": False,
                    "post_lift_holding_verified": False,
                    "rg2_calibrated": False,
                },
            )
            imported.append(
                ImportedGripPointProfile(
                    object_class_id=object_class_id,
                    catalog_entry_id=catalog.id,
                    grip_profile_id=profile.id,
                    grip_profile_version_id=version.id,
                    artifact_uri=artifact.uri,
                    artifact_checksum_sha256=artifact.checksum_sha256,
                )
            )
        return GripPointImportSummary(
            source_checksum_sha256=source_checksum,
            profiles=tuple(imported),
        )

    @staticmethod
    def _validate_root(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if root.get("schema_version") != "1.0":
            raise ValueError("unsupported grip-point result schema version")
        if root.get("execution_mode") != "offline_diagnostic_only":
            raise ValueError("only offline diagnostic grip-point results may be imported")
        if root.get("robot_or_gripper_called") is not False:
            raise ValueError("grip-point import requires robot_or_gripper_called=false")
        if root.get("approach_direction_estimated") is not False:
            raise ValueError("legacy importer accepts only results without an approach estimate")
        limitations = root.get("limitations")
        if not isinstance(limitations, list) or any(
            not isinstance(item, str) for item in limitations
        ):
            raise ValueError("grip-point limitations must be a list of strings")
        objects_value = _require_mapping(root.get("objects"), field="objects")
        if not objects_value or len(objects_value) > 256:
            raise ValueError("grip-point result must contain between 1 and 256 objects")
        objects: dict[str, dict[str, Any]] = {}
        for key, value in objects_value.items():
            if not _CANONICAL_ID_PATTERN.fullmatch(key):
                raise ValueError(f"unsafe object class ID in grip-point result: {key!r}")
            objects[key] = _require_mapping(value, field=f"objects.{key}")
        return objects

    @staticmethod
    def _validate_object(object_class_id: str, value: dict[str, Any]) -> None:
        model = _require_mapping(
            value.get("learned_grip_model"),
            field=f"objects.{object_class_id}.learned_grip_model",
        )
        if model.get("object") != object_class_id:
            raise ValueError(f"grip model object ID does not match {object_class_id!r}")
        normalized = _require_mapping(
            model.get("normalized_grasp_point"),
            field=f"objects.{object_class_id}.normalized_grasp_point",
        )
        for coordinate in ("lateral_median", "longitudinal_median"):
            number = _finite_number(
                normalized.get(coordinate),
                field=f"objects.{object_class_id}.{coordinate}",
            )
            if number < -1.5 or number > 1.5:
                raise ValueError("normalized diagnostic grasp coordinates are out of bounds")
        angle = _finite_number(
            model.get("jaw_relative_angle_deg"),
            field=f"objects.{object_class_id}.jaw_relative_angle_deg",
        )
        if angle < -180.0 or angle > 180.0:
            raise ValueError("diagnostic jaw angle is out of bounds")
        width_stats = _require_mapping(
            model.get("grip_width_statistics_m"),
            field=f"objects.{object_class_id}.grip_width_statistics_m",
        )
        median_width_m = _finite_number(
            width_stats.get("median"),
            field=f"objects.{object_class_id}.grip_width_statistics_m.median",
        )
        if median_width_m < 0.0 or median_width_m > 0.2:
            raise ValueError("diagnostic grip width is outside the bounded import range")


def import_grip_point_result(
    result_path: Path,
    *,
    repository: StorageRepository,
    artifact_store: LocalArtifactStore,
) -> GripPointImportSummary:
    """Convenience wrapper around :class:`GripPointResultImporter`."""

    return GripPointResultImporter(repository, artifact_store).import_file(result_path)


__all__ = [
    "GripPointImportSummary",
    "GripPointResultImporter",
    "ImportedGripPointProfile",
    "import_grip_point_result",
]
