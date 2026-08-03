"""Load only compiler-owned, checksum-verified artifacts from a configured root."""

from __future__ import annotations

import importlib.util
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from robot_skill_system.exceptions import CodeValidationError
from robot_skill_system.skills.code_validator import GeneratedCodeValidator
from robot_skill_system.skills.compiler import SkillCompiler

CompiledRun = Callable[[Any], Awaitable[bool]]


def _safe_artifact_path(path: Path, artifact_root: Path) -> Path:
    root = artifact_root.resolve()
    if path.is_absolute():
        raise CodeValidationError(
            "compiled skill path must be a registry-issued relative artifact URI"
        )
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise CodeValidationError("compiled skill path escapes the configured artifact root")
    if candidate.suffix != ".py" or not candidate.is_file():
        raise CodeValidationError("compiled skill must be an existing .py artifact")
    return candidate


def load_compiled_run(
    path: Path,
    *,
    artifact_root: Path,
    expected_checksum_sha256: str,
) -> CompiledRun:
    """Validate and import one trusted compiler artifact.

    The caller cannot choose a module/package name or bypass the configured root. The file is
    revalidated and checksum-matched immediately before import. This is intentionally narrower
    than a general dynamic module loader and must not be exposed as an OpenAI/function tool.
    """

    candidate = _safe_artifact_path(path, artifact_root)
    if not SkillCompiler.verify_checksum(candidate, expected_checksum_sha256):
        raise CodeValidationError("compiled skill checksum does not match its manifest")
    GeneratedCodeValidator().validate_file(candidate)
    module_name = f"_robot_skill_generated_{expected_checksum_sha256}"
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise CodeValidationError("could not create a loader for the compiled skill")
    module = importlib.util.module_from_spec(spec)
    _load_verified_module(spec.loader, module)
    run = getattr(module, "run", None)
    if run is None or not callable(run):
        raise CodeValidationError("compiled skill does not expose run(runtime)")
    return cast(CompiledRun, run)


def _load_verified_module(loader: Any, module: ModuleType) -> None:
    """Execute a compiler-owned module after all validation gates have passed."""

    loader.exec_module(module)
