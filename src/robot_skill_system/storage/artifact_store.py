"""Filesystem artifact store with traversal protection and checksums."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    uri: str
    checksum_sha256: str
    size_bytes: int
    media_type: str


class LocalArtifactStore:
    """Store immutable-ish artifacts below one configured root.

    The returned URI is relative and portable. Callers persist only this metadata in SQLite.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self, relative_uri: str, content: bytes, *, media_type: str = "application/octet-stream"
    ) -> ArtifactMetadata:
        destination = self._resolve(relative_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256(content).hexdigest()
        if destination.exists():
            existing = destination.read_bytes()
            if hashlib.sha256(existing).hexdigest() != checksum:
                raise FileExistsError(
                    f"artifact URI {relative_uri!r} already contains different content"
                )
            return ArtifactMetadata(relative_uri, checksum, len(existing), media_type)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            Path(temporary_name).replace(destination)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return ArtifactMetadata(relative_uri, checksum, len(content), media_type)

    def put_text(
        self, relative_uri: str, content: str, *, media_type: str = "text/plain; charset=utf-8"
    ) -> ArtifactMetadata:
        return self.put_bytes(relative_uri, content.encode("utf-8"), media_type=media_type)

    def put_json(self, relative_uri: str, value: Any) -> ArtifactMetadata:
        content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return self.put_text(relative_uri, content, media_type="application/json")

    def read_bytes(
        self, relative_uri: str, *, expected_checksum_sha256: str | None = None
    ) -> bytes:
        content = self._resolve(relative_uri).read_bytes()
        if expected_checksum_sha256 is not None:
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected_checksum_sha256:
                raise ValueError(f"artifact checksum mismatch for {relative_uri!r}")
        return content

    def path_for(self, relative_uri: str) -> Path:
        return self._resolve(relative_uri)

    def delete_tree(self, relative_uri: str) -> int:
        """Delete one validated artifact subtree and return its file count."""

        target = self._resolve(relative_uri)
        if not target.exists():
            return 0
        if not target.is_dir():
            raise ValueError("artifact subtree URI must reference a directory")
        deleted_files = sum(1 for item in target.rglob("*") if item.is_file())
        shutil.rmtree(target)
        return deleted_files

    def _resolve(self, relative_uri: str) -> Path:
        if "\\" in relative_uri:
            raise ValueError("artifact URI must use POSIX separators")
        relative = PurePosixPath(relative_uri)
        invalid_part = any(part in {"", ".", ".."} for part in relative.parts)
        if relative.is_absolute() or not relative.parts or invalid_part:
            raise ValueError("artifact URI must be a normalized relative path")
        candidate = self.root.joinpath(*relative.parts).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("artifact URI escapes the configured root")
        return candidate
