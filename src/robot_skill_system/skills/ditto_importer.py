"""Read-only adapter from ``ditto_system/skills`` to safe Blockly blocks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,180}$")
_STAGE_PATTERNS = {
    "generate": re.compile(
        r"^(?P<source_id>[A-Za-z0-9][A-Za-z0-9_-]{0,180})_smoothed_move_plan\.json$"
    ),
    "classify": re.compile(
        r"^(?P<source_id>[A-Za-z0-9][A-Za-z0-9_-]{0,180})_smoothed_segments\.json$"
    ),
    "smooth": re.compile(
        r"^(?P<source_id>[A-Za-z0-9][A-Za-z0-9_-]{0,180})_smoothed\.json$"
    ),
    "raw": re.compile(r"^(?P<source_id>[A-Za-z0-9][A-Za-z0-9_-]{0,180})\.json$"),
}
_MAXIMUM_JSON_BYTES = 8 * 1024 * 1024
_MAXIMUM_BLOCKS = 128


class DittoSkillCatalog:
    """Discover and convert fixed external skill artifacts without modifying them."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.skills_root = (
            self.repository_root
            / "ditto_ws"
            / "src"
            / "ditto_system"
            / "skills"
        ).resolve()
        expected = self.repository_root / "ditto_ws" / "src" / "ditto_system" / "skills"
        if self.skills_root != expected.resolve():
            raise ValueError("ditto skill root must stay under repo_root/ditto_ws")

    def _sources(self) -> dict[str, dict[str, Path]]:
        sources: dict[str, dict[str, Path]] = {}
        if not self.skills_root.is_dir():
            return sources
        for stage, pattern in _STAGE_PATTERNS.items():
            stage_root = self.skills_root / stage
            if not stage_root.is_dir():
                continue
            for path in sorted(stage_root.glob("*.json")):
                match = pattern.fullmatch(path.name)
                if match is None or not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved.parent != stage_root.resolve():
                    continue
                sources.setdefault(match.group("source_id"), {})[stage] = resolved
        return sources

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        size = path.stat().st_size
        if size <= 0 or size > _MAXIMUM_JSON_BYTES:
            raise ValueError(f"unsupported ditto skill JSON size: {path.name}")
        return path.read_bytes()

    @classmethod
    def _read_json(cls, path: Path) -> dict[str, Any]:
        payload = json.loads(cls._read_bytes(path).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"ditto skill JSON must contain an object: {path.name}")
        return payload

    @staticmethod
    def _number(value: object, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ditto pose {field} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"ditto pose {field} must be finite")
        return result

    @classmethod
    def _pose_values(cls, pose: object) -> tuple[float, float, float, float]:
        if not isinstance(pose, dict):
            raise ValueError("ditto motion pose must be an object")
        yaw_value = pose.get("yaw", pose.get("rz", 0.0))
        return (
            cls._number(pose.get("x"), "x"),
            cls._number(pose.get("y"), "y"),
            cls._number(pose.get("z"), "z"),
            cls._number(yaw_value, "yaw/rz"),
        )

    @staticmethod
    def _relative_pose(
        pose: tuple[float, float, float, float],
        origin: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        x_mm, y_mm, z_mm, yaw_deg = pose
        origin_x_mm, origin_y_mm, origin_z_mm, origin_yaw_deg = origin
        yaw_delta_deg = (yaw_deg - origin_yaw_deg + 180.0) % 360.0 - 180.0
        half_yaw_rad = math.radians(yaw_delta_deg) / 2.0
        return {
            "anchor_id": "$workspace",
            "anchor_type": "workspace_region",
            "position_m": {
                "x": (x_mm - origin_x_mm) / 1000.0,
                "y": (y_mm - origin_y_mm) / 1000.0,
                "z": (z_mm - origin_z_mm) / 1000.0,
            },
            "orientation_xyzw": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(half_yaw_rad),
                "w": math.cos(half_yaw_rad),
            },
        }

    def _origin(self, files: dict[str, Path]) -> tuple[float, float, float, float]:
        for stage in ("smooth", "raw"):
            path = files.get(stage)
            if path is None:
                continue
            frames = self._read_json(path).get("frames")
            if isinstance(frames, list) and frames:
                return self._pose_values(frames[0])
        classify_path = files.get("classify")
        if classify_path is not None:
            segments = self._read_json(classify_path).get("segments")
            if isinstance(segments, list) and segments:
                first = segments[0]
                if isinstance(first, dict):
                    return self._pose_values(first.get("start_pose"))
        raise ValueError("ditto skill has no usable first waypoint anchor")

    def _generated_blocks(
        self,
        path: Path,
        origin: tuple[float, float, float, float],
    ) -> list[dict[str, Any]]:
        sequence = self._read_json(path).get("sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("ditto generated skill has no sequence")
        if len(sequence) > _MAXIMUM_BLOCKS:
            raise ValueError("ditto generated skill exceeds 128 Blockly blocks")
        # Generated plans bake the first orientation to 156.2 degrees. Position
        # remains in the original base frame and is made relative to the first
        # recorded waypoint below.
        generated_origin = (*origin[:3], 156.2)
        blocks: list[dict[str, Any]] = []
        for item in sequence:
            if not isinstance(item, dict):
                raise ValueError("ditto generated sequence item must be an object")
            target = self._relative_pose(
                self._pose_values(item.get("end_pose")), generated_origin
            )
            via_value = item.get("via_pose")
            if item.get("move_type") == "movec" and isinstance(via_value, dict):
                blocks.append(
                    {
                        "operation": "motion.move_c",
                        "arguments": {
                            "via": self._relative_pose(
                                self._pose_values(via_value), generated_origin
                            ),
                            "target": target,
                            "motion_profile_id": "circular_normal",
                        },
                    }
                )
            else:
                blocks.append(
                    {
                        "operation": "motion.move_l",
                        "arguments": {
                            "target": target,
                            "motion_profile_id": "linear_slow",
                        },
                    }
                )
        return blocks

    def _classified_blocks(
        self,
        path: Path,
        origin: tuple[float, float, float, float],
    ) -> list[dict[str, Any]]:
        segments = self._read_json(path).get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("ditto classified skill has no segments")
        if len(segments) > _MAXIMUM_BLOCKS:
            raise ValueError("ditto classified skill exceeds 128 Blockly blocks")
        blocks: list[dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("ditto segment must be an object")
            target = self._relative_pose(
                self._pose_values(segment.get("end_pose")), origin
            )
            via_value = segment.get("via_pose")
            if segment.get("type") == "arc" and isinstance(via_value, dict):
                blocks.append(
                    {
                        "operation": "motion.move_c",
                        "arguments": {
                            "via": self._relative_pose(
                                self._pose_values(via_value), origin
                            ),
                            "target": target,
                            "motion_profile_id": "circular_normal",
                        },
                    }
                )
            else:
                blocks.append(
                    {
                        "operation": "motion.move_l",
                        "arguments": {
                            "target": target,
                            "motion_profile_id": "linear_slow",
                        },
                    }
                )
        return blocks

    def _trajectory_blocks(
        self,
        path: Path,
        origin: tuple[float, float, float, float],
    ) -> list[dict[str, Any]]:
        frames = self._read_json(path).get("frames")
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("ditto trajectory requires at least two frames")
        if len(frames) > 128:
            raise ValueError("ditto trajectory exceeds 128 spline waypoints")
        return [
            {
                "operation": "motion.move_spline",
                "arguments": {
                    "waypoints": [
                        self._relative_pose(self._pose_values(frame), origin)
                        for frame in frames
                    ],
                    "motion_profile_id": "linear_slow",
                },
            }
        ]

    @staticmethod
    def _selected_stage(files: dict[str, Path]) -> str:
        for stage in ("generate", "classify", "smooth", "raw"):
            if stage in files:
                return stage
        raise ValueError("ditto skill does not have a supported JSON stage")

    @classmethod
    def _checksum(cls, files: dict[str, Path]) -> str:
        digest = hashlib.sha256()
        for stage, path in sorted(files.items()):
            digest.update(stage.encode("utf-8"))
            digest.update(b"\0")
            digest.update(cls._read_bytes(path))
            digest.update(b"\0")
        return digest.hexdigest()

    def editor_source(self, source_id: str) -> dict[str, Any]:
        if _SOURCE_ID.fullmatch(source_id) is None:
            raise ValueError("invalid ditto skill source id")
        files = self._sources().get(source_id)
        if files is None:
            raise KeyError(source_id)
        stage = self._selected_stage(files)
        origin = self._origin(files)
        if stage == "generate":
            blocks = self._generated_blocks(files[stage], origin)
        elif stage == "classify":
            blocks = self._classified_blocks(files[stage], origin)
        else:
            blocks = self._trajectory_blocks(files[stage], origin)
        source_checksum = self._checksum(files)
        tool_name = source_id.rsplit("_", 1)[0].removesuffix("ing") or "tool"
        return {
            "source_id": source_id,
            "source_stage": stage,
            "source_checksum_sha256": source_checksum,
            "source_files": {
                key: str(path.relative_to(self.repository_root))
                for key, path in sorted(files.items())
            },
            "skill_id": f"ditto_{source_id}"[:64],
            "name": f"Ditto {source_id}",
            "description": (
                f"ditto_system/skills {stage} 데이터를 첫 waypoint 기준 "
                "workspace-relative Blockly 블록으로 가져온 Mock-only Candidate"
            ),
            "skill_type": "motion",
            "tool_hint": tool_name,
            "blocks": blocks,
            "bindings": {
                "$workspace": {
                    "variable": "$workspace",
                    "entity_kind": "workspace",
                    "role": "ditto_demonstration_origin",
                    "minimum_confidence": 0.9,
                    "minimum_visible_fraction": 0.8,
                }
            },
            "warnings": [
                "robot-base absolute coordinates were discarded",
                "velocity, acceleration, force, and blend values were not imported",
                "local approved motion profiles replace external numeric settings",
                "gripper observations were not converted into actuator commands",
                "result is non-executable until edited and Mock-validated as a Candidate",
            ],
            "external_package_read_only": True,
            "persisted": False,
            "executable": False,
        }

    def list_sources(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for source_id, files in sorted(self._sources().items()):
            try:
                editor = self.editor_source(source_id)
                entries.append(
                    {
                        "source_id": source_id,
                        "source_stage": editor["source_stage"],
                        "source_checksum_sha256": editor["source_checksum_sha256"],
                        "block_count": len(editor["blocks"]),
                        "tool_hint": editor["tool_hint"],
                        "ready": True,
                        "blockers": [],
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                entries.append(
                    {
                        "source_id": source_id,
                        "source_stage": self._selected_stage(files),
                        "source_checksum_sha256": None,
                        "block_count": 0,
                        "tool_hint": None,
                        "ready": False,
                        "blockers": [str(exc)],
                    }
                )
        return {
            "integration": "ditto_system.skills",
            "available": self.skills_root.is_dir(),
            "external_package_read_only": True,
            "skills_root": str(self.skills_root.relative_to(self.repository_root))
            if self.skills_root.is_relative_to(self.repository_root)
            else None,
            "skills": entries,
        }
