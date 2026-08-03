"""JSON command-line interface for the offline-first robot skill MVP."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

from sqlalchemy import select

from robot_skill_system.application import MVPApplication, create_application
from robot_skill_system.openai_integration.transcription import TranscriptionService
from robot_skill_system.primitives.registry import get_default_registry
from robot_skill_system.settings import (
    ExecutionMode,
    OpenAIMode,
    Settings,
)
from robot_skill_system.storage.orm import SkillRecord, SkillVersionRecord
from robot_skill_system.vertical_slice import run_offline_demo

EXIT_OK = 0
EXIT_OPERATION_FAILED = 1
EXIT_USAGE = 2
EXIT_SAFETY_REJECTED = 3


class JsonArgumentParser(argparse.ArgumentParser):
    """Emit parse failures as machine-readable JSON."""

    def error(self, message: str) -> NoReturn:
        _emit(
            {
                "ok": False,
                "error": {"type": "usage_error", "message": message},
            },
            stream=sys.stderr,
        )
        raise SystemExit(EXIT_USAGE)


def offline_settings(settings: Settings | None = None) -> Settings:
    """Return settings guaranteed not to contact OpenAI or a real robot."""

    configured = settings or Settings.from_env()
    return configured.model_copy(
        update={
            "openai_mode": OpenAIMode.MOCK,
            "robot_execution_mode": ExecutionMode.MOCK,
            "enable_hardware_execution": False,
            "robot_backend": "mock",
            "enable_real_robot": False,
            "dry_run": True,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the dependency-free CLI parser."""

    parser = JsonArgumentParser(
        prog="robot-skill",
        description="Safety-bounded robot skill teaching and execution MVP",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="show safe configuration and primitive catalog")

    capture = subparsers.add_parser("capture-scene", help="capture and persist a scene")
    capture.add_argument("--backend", choices=("mock",), default="mock")
    capture.add_argument("--mode", choices=("mock", "single", "burst"), default="mock")

    subparsers.add_parser(
        "demo-e2e",
        aliases=("demo-offline",),
        help="run the complete deterministic offline demo",
    )

    transcribe = subparsers.add_parser(
        "transcribe", help="transcribe an audio file through the configured mock/live service"
    )
    transcribe.add_argument("audio_path", type=Path)

    induce = subparsers.add_parser("induce-skill", help="induce and register a skill")
    induce.add_argument("--demo", required=True, type=Path)
    induce.add_argument("--skill-id", default="wipe_surface")
    induce.add_argument("--variant", default="safe")

    update = subparsers.add_parser("update-skill", help="create an immutable candidate update")
    update.add_argument("--skill-id", "--skill", dest="skill_id", required=True)
    update.add_argument("--demo", required=True, type=Path)
    update.add_argument("--base-version")
    update.add_argument("--operator-role", default="expert")
    update.add_argument("--has-force-measurements", action="store_true")

    listing = subparsers.add_parser("list-skills", help="list registered skill versions")
    listing.add_argument("--skill-id")
    listing.add_argument("--active-only", action="store_true")

    execute = subparsers.add_parser("execute", help="resolve, bind, preflight, and execute")
    execute.add_argument("--text", required=True)
    execute.add_argument(
        "--mode",
        choices=("mock", "dry_run", "simulation", "hardware"),
        default="mock",
    )
    execute.add_argument("--skill-id", "--skill", dest="skill_id")
    execute.add_argument("--version")
    execute.add_argument("--scene-id")
    execute.add_argument(
        "--binding",
        action="append",
        default=[],
        metavar="PLACEHOLDER=ENTITY_ID",
    )

    execute_skill = subparsers.add_parser(
        "execute-skill", help="supplemental fail-closed dry-run skill command"
    )
    execute_skill.add_argument("--skill", dest="skill_id", required=True)
    execute_skill.add_argument("--dry-run", action="store_true", required=True)
    execute_skill.add_argument("--text", default="execute the selected skill")
    execute_skill.add_argument("--version")
    execute_skill.add_argument("--scene-id")
    execute_skill.add_argument(
        "--binding",
        action="append",
        default=[],
        metavar="PLACEHOLDER=ENTITY_ID",
    )
    execute_skill.set_defaults(mode="dry_run")

    rollback = subparsers.add_parser("rollback", help="reactivate a validated version")
    rollback.add_argument("--skill-id", "--skill", dest="skill_id", required=True)
    rollback.add_argument("--version", required=True)

    validate = subparsers.add_parser("validate-skill", help="validate a candidate locally")
    validate.add_argument("--skill-id", "--skill", dest="skill_id", required=True)
    validate.add_argument("--version")

    activate = subparsers.add_parser("activate-skill", help="activate a validated version")
    activate.add_argument("--skill-id", "--skill", dest="skill_id", required=True)
    activate.add_argument("--version", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
) -> int:
    """Run one command, printing exactly one JSON result or error document."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configured = settings or Settings.from_env()
    if args.command == "execute" and args.mode == "hardware":
        _emit_error(
            "hardware_execution_unavailable",
            "the MVP CLI has no physically validated Doosan/RG2 hardware adapter",
        )
        return EXIT_SAFETY_REJECTED
    try:
        result = _dispatch(args, configured)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        _emit_error(type(exc).__name__, str(exc))
        return EXIT_OPERATION_FAILED
    except Exception as exc:  # pragma: no cover - defensive process boundary
        _emit_error(type(exc).__name__, str(exc))
        return EXIT_OPERATION_FAILED
    _emit({"ok": True, "command": args.command, "result": result})
    return EXIT_OK


def _dispatch(args: argparse.Namespace, settings: Settings) -> Mapping[str, Any]:
    if args.command == "inspect":
        return _inspect(settings)
    if args.command in {"demo-e2e", "demo-offline"}:
        return run_offline_demo(offline_settings(settings)).model_dump(mode="json")
    if args.command == "transcribe":
        return TranscriptionService(settings).transcribe(args.audio_path).model_dump(
            mode="json"
        )

    app = create_application(settings)
    try:
        if args.command == "capture-scene":
            scene = app.capture_scene({"mode": args.mode})
            return {"backend": args.backend, **scene}
        if args.command == "induce-skill":
            return app.induce_skill(
                {
                    "demo_path": str(args.demo),
                    "name": args.skill_id,
                    "variant": args.variant,
                }
            )
        if args.command == "update-skill":
            request: dict[str, Any] = {
                "demo_path": str(args.demo),
                "operator_role": args.operator_role,
                "has_force_measurements": args.has_force_measurements,
            }
            if args.base_version:
                request["base_version"] = args.base_version
            return app.update_skill(args.skill_id, request)
        if args.command == "list-skills":
            return _list_skills(
                app,
                skill_id=args.skill_id,
                active_only=args.active_only,
            )
        if args.command in {"execute", "execute-skill"}:
            return _execute(app, args)
        if args.command == "rollback":
            return app.rollback_skill(args.skill_id, {"version": args.version})
        if args.command == "validate-skill":
            return app.validate_skill(args.skill_id, {"version": args.version})
        if args.command == "activate-skill":
            return app.activate_skill(args.skill_id, {"version": args.version})
        raise ValueError(f"unsupported command {args.command!r}")
    finally:
        app.close()


def _inspect(settings: Settings) -> dict[str, Any]:
    operations = sorted(item.operation_name for item in get_default_registry().catalog())
    return {
        "version": "0.1.0",
        "configuration": settings.safe_summary(),
        "hardware_enabled": settings.hardware_enabled,
        "primitive_operations": operations,
        "primitive_count": len(operations),
        "guarantees": {
            "model_generated_code_allowed": False,
            "absolute_base_targets_in_skills_allowed": False,
            "hardware_requires_all_gates": True,
        },
    }


def _list_skills(
    app: MVPApplication,
    *,
    skill_id: str | None,
    active_only: bool,
) -> dict[str, Any]:
    with app.database.session() as session:
        statement = (
            select(SkillRecord, SkillVersionRecord)
            .join(SkillVersionRecord, SkillVersionRecord.skill_id == SkillRecord.id)
            .order_by(SkillRecord.name, SkillVersionRecord.created_at, SkillVersionRecord.id)
        )
        rows = list(session.execute(statement).tuples())
    grouped: dict[str, dict[str, Any]] = {}
    for skill, version in rows:
        semantic_skill_id = str(version.graph_json.get("skill_id", skill.intent))
        if skill_id is not None and semantic_skill_id != skill_id:
            continue
        if active_only and version.id != skill.active_version_id:
            continue
        key = skill.id
        summary = grouped.setdefault(
            key,
            {
                "skill_id": semantic_skill_id,
                "name": skill.name,
                "variant": skill.variant,
                "active_version_id": skill.active_version_id,
                "versions": [],
            },
        )
        summary["versions"].append(
            {
                "version_id": version.id,
                "version": version.semantic_version,
                "status": version.status,
                "validation_status": version.validation_status,
                "active": version.id == skill.active_version_id,
                "hardware_compatible": version.hardware_compatible,
            }
        )
    skills = list(grouped.values())
    return {"count": len(skills), "skills": skills}


def _execute(app: MVPApplication, args: argparse.Namespace) -> dict[str, Any]:
    scene_id = args.scene_id
    if scene_id is None:
        scene_id = str(app.capture_scene({"mode": "mock"})["scene_id"])
    resolution = app.resolve_runtime({"text": args.text, "scene_id": scene_id})
    skill_id = args.skill_id or _first_resolved_skill_id(resolution)
    request: dict[str, Any] = {
        "text": args.text,
        "mode": args.mode,
        "skill_id": skill_id,
        "scene_id": scene_id,
        "bindings": _parse_bindings(args.binding),
    }
    if args.version:
        request["version"] = args.version
    execution = app.execute_runtime(request)
    return {"resolution": resolution, "execution": execution}


def _first_resolved_skill_id(resolution: Mapping[str, Any]) -> str:
    candidates = resolution.get("skill_candidates", {})
    if not isinstance(candidates, Mapping):
        raise ValueError("runtime resolver returned malformed skill candidates")
    results = candidates.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError(
            "no active validated skill matched; run induce-skill or demo-e2e first"
        )
    first = results[0]
    if not isinstance(first, Mapping) or not first.get("skill_id"):
        raise ValueError("runtime resolver returned a candidate without skill_id")
    return str(first["skill_id"])


def _parse_bindings(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        placeholder, separator, entity_id = value.partition("=")
        if not separator or not placeholder.startswith("$") or not entity_id:
            raise ValueError(
                f"invalid binding {value!r}; expected $placeholder=entity_id"
            )
        if placeholder in result:
            raise ValueError(f"duplicate binding for {placeholder!r}")
        result[placeholder] = entity_id
    return result


def _emit_error(error_type: str, message: str) -> None:
    _emit(
        {"ok": False, "error": {"type": error_type, "message": message}},
        stream=sys.stderr,
    )


def _emit(value: Mapping[str, Any], *, stream: Any | None = None) -> None:
    destination = sys.stdout if stream is None else stream
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default),
        file=destination,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
