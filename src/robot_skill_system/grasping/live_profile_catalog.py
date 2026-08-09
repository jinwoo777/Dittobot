"""Checksum-guarded registry entry for live RGB-D grip selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from robot_skill_system.perception.live_scene import LearnedGripPoint
from robot_skill_system.scene.models import StrictModel
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import StorageRepository
from robot_skill_system.storage.orm import GripProfileVersionRecord


class LiveGripRuntimeProfile(StrictModel):
    schema_version: Literal["live-grip-runtime/1.0"] = "live-grip-runtime/1.0"
    class_name: Literal["hammer"]
    normalized_longitudinal: float = Field(ge=-1.5, le=1.5)
    normalized_lateral: float = Field(ge=-1.5, le=1.5)
    jaw_relative_angle_rad: float = Field(ge=-3.141592653589793, le=3.141592653589793)
    target_gripper_width_m: float = Field(gt=0.0, le=0.110)
    source_checksum_sha256: str
    geometry_policy: Literal[
        "live_rgbd_contact_plus_aruco_plane_width_workspace"
    ] = "live_rgbd_contact_plus_aruco_plane_width_workspace"

    @field_validator("source_checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("source checksum must be lower-case SHA-256")
        return value

    def as_learned_grip_point(self, source_path: Path) -> LearnedGripPoint:
        return LearnedGripPoint(
            class_name=self.class_name,
            longitudinal=self.normalized_longitudinal,
            lateral=self.normalized_lateral,
            jaw_relative_angle_rad=self.jaw_relative_angle_rad,
            target_gripper_width_m=self.target_gripper_width_m,
            source_path=source_path,
        )


@dataclass(frozen=True, slots=True)
class RegisteredLiveGripProfile:
    record: GripProfileVersionRecord
    runtime_profile: LiveGripRuntimeProfile


def register_configured_live_grip_profile(
    repository: StorageRepository,
    store: LocalArtifactStore,
    configured_profile: LearnedGripPoint,
) -> RegisteredLiveGripProfile | None:
    object_entry = next(
        (
            entry
            for entry in repository.list_catalog_entries(kind="object")
            if entry.canonical_id == configured_profile.class_name
        ),
        None,
    )
    if object_entry is None:
        return None
    source_bytes = configured_profile.source_path.read_bytes()
    source_checksum = hashlib.sha256(source_bytes).hexdigest()
    runtime_profile = LiveGripRuntimeProfile(
        class_name="hammer",
        normalized_longitudinal=configured_profile.longitudinal,
        normalized_lateral=configured_profile.lateral,
        jaw_relative_angle_rad=configured_profile.jaw_relative_angle_rad,
        target_gripper_width_m=configured_profile.target_gripper_width_m,
        source_checksum_sha256=source_checksum,
    )
    version = f"1.0.0-live.{source_checksum[:12]}"
    artifact = store.put_json(
        f"grips/{configured_profile.class_name}/{version}.json",
        runtime_profile.model_dump(mode="json"),
    )
    record = repository.register_grip_profile_version(
        object_class_id=configured_profile.class_name,
        semantic_version=version,
        artifact_uri=artifact.uri,
        artifact_checksum_sha256=artifact.checksum_sha256,
        object_frame_policy="live_rgbd_contact_relative",
        object_frame_revision=source_checksum,
        status="validated",
        validation_status="passed",
        hardware_compatible=False,
        auto_activation_allowed=False,
        gripper_calibration_profile_id="rg2_default",
        metadata={
            "runtime_live_grip_profile": True,
            "hardware_validated": False,
            "selection_policy": "object_class_exact_match",
        },
    )
    repository.activate_grip_profile_version(record.id, automatic=False)
    if object_entry.status != "active":
        repository.activate_catalog_entry(object_entry.id)
    active = repository.active_grip_profile_version(
        object_class_id=configured_profile.class_name
    )
    if active is None or active.id != record.id:
        raise ValueError("registered live GripProfile was not activated")
    return RegisteredLiveGripProfile(record=active, runtime_profile=runtime_profile)


def load_registered_live_grip_profile(
    store: LocalArtifactStore,
    record: GripProfileVersionRecord,
    *,
    source_path: Path,
) -> LearnedGripPoint:
    payload = json.loads(
        store.read_bytes(
            record.artifact_uri,
            expected_checksum_sha256=record.artifact_checksum_sha256,
        )
    )
    runtime_profile = LiveGripRuntimeProfile.model_validate(payload)
    if runtime_profile.class_name != "hammer":
        raise ValueError("registered live GripProfile class is not hammer")
    source_checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if runtime_profile.source_checksum_sha256 != source_checksum:
        raise ValueError("registered live GripProfile source checksum changed")
    return runtime_profile.as_learned_grip_point(source_path)


__all__ = [
    "LiveGripRuntimeProfile",
    "RegisteredLiveGripProfile",
    "load_registered_live_grip_profile",
    "register_configured_live_grip_profile",
]
