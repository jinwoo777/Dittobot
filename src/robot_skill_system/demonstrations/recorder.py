"""Portable JSONL demonstration artifact loading and recording."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import DemonstrationTrajectory, PoseSample


class TeachingSessionMetadata(BaseModel):
    """Small metadata file stored beside raw demonstration artifacts."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    session_id: str
    operator_id: str | None = None
    operator_role: str | None = None
    successful: bool | None = None
    notes: str | None = None
    raw_capture_fps: float | None = Field(default=None, gt=0.0)
    pose_inference_fps: float | None = Field(default=None, gt=0.0)
    timestamps_preserved: bool = True


def _resolve_pose_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate / "poses.jsonl" if candidate.is_dir() else candidate


def load_pose_samples(path: Path | str) -> tuple[PoseSample, ...]:
    """Load strict pose records from a JSON Lines artifact."""

    pose_path = _resolve_pose_path(path)
    samples: list[PoseSample] = []
    with pose_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                samples.append(PoseSample.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(f"invalid pose at {pose_path}:{line_number}") from exc
    if not samples:
        raise ValueError(f"demonstration contains no pose samples: {pose_path}")
    return tuple(samples)


def load_demonstration(path: Path | str) -> DemonstrationTrajectory:
    """Load poses plus optional session metadata from a fixture/artifact directory."""

    candidate = Path(path)
    pose_path = _resolve_pose_path(candidate)
    directory = pose_path.parent
    metadata_path = directory / "metadata.json"
    metadata: TeachingSessionMetadata | None = None
    if metadata_path.exists():
        metadata = TeachingSessionMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
    return DemonstrationTrajectory(
        samples=load_pose_samples(pose_path),
        session_id=metadata.session_id if metadata else directory.name,
        operator_id=metadata.operator_id if metadata else None,
        operator_role=metadata.operator_role if metadata else None,
        successful=metadata.successful if metadata else None,
        notes=metadata.notes if metadata else None,
    )


def write_pose_samples(path: Path | str, samples: Iterable[PoseSample]) -> Path:
    """Write strict pose JSONL atomically enough for a local teaching artifact."""

    pose_path = Path(path)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = tuple(samples)
    if not materialized:
        raise ValueError("cannot record an empty pose series")
    temporary_path = pose_path.with_suffix(pose_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for sample in materialized:
            handle.write(sample.model_dump_json())
            handle.write("\n")
    temporary_path.replace(pose_path)
    return pose_path


class DemonstrationRecorder:
    """Record session artifacts beneath a caller-supplied root directory."""

    def __init__(self, artifact_root: Path | str) -> None:
        self.artifact_root = Path(artifact_root)

    def record(
        self,
        trajectory: DemonstrationTrajectory,
        metadata: TeachingSessionMetadata | None = None,
    ) -> Path:
        session_id = trajectory.session_id or (metadata.session_id if metadata else None)
        if not session_id:
            raise ValueError("a session_id is required to record a demonstration")
        if Path(session_id).name != session_id or session_id in {".", ".."}:
            raise ValueError("session_id must be one safe path component")
        session_directory = self.artifact_root / session_id
        session_directory.mkdir(parents=True, exist_ok=True)
        write_pose_samples(session_directory / "poses.jsonl", trajectory.samples)
        effective_metadata = metadata or TeachingSessionMetadata(
            session_id=session_id,
            operator_id=trajectory.operator_id,
            operator_role=trajectory.operator_role,
            successful=trajectory.successful,
            notes=trajectory.notes,
        )
        metadata_path = session_directory / "metadata.json"
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            effective_metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)
        return session_directory


__all__ = [
    "DemonstrationRecorder",
    "TeachingSessionMetadata",
    "load_demonstration",
    "load_pose_samples",
    "write_pose_samples",
]
