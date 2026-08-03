"""Tests for AST-only deterministic skill compilation and code validation."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from robot_skill_system.exceptions import CodeValidationError, CompilationError
from robot_skill_system.skills.code_validator import GeneratedCodeValidator
from robot_skill_system.skills.compiler import SkillCompiler
from robot_skill_system.skills.loader import load_compiled_run
from robot_skill_system.skills.models import SkillEdge, SkillGraph, SkillNode, SkillType


def _target(x_m: float = 0.0) -> dict[str, object]:
    return {
        "anchor_id": "$surface",
        "anchor_type": "surface",
        "position_m": {"x": x_m, "y": 0.0, "z": 0.02},
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _motion_graph() -> SkillGraph:
    return SkillGraph(
        skill_id="compile_demo",
        version="1.0.0",
        name="compile demo",
        description="Compiler test graph.",
        skill_type=SkillType.MOTION,
        source_demonstrations=["demo_01"],
        bindings={"$surface": {"entity_kind": "surface"}},
        nodes=[
            SkillNode(
                node_id="first",
                operation="workspace.validate_target",
                arguments={"target": _target()},
            ),
            SkillNode(
                node_id="second",
                operation="motion.move_l",
                arguments={"motion_profile_id": "linear_normal", "target": _target(0.2)},
                checkpoint=True,
            ),
        ],
        edges=[SkillEdge(source_node="first", target_node="second")],
        start_node="first",
        terminal_nodes=["second"],
        motion_profiles=["linear_normal"],
    )


def _force_graph() -> SkillGraph:
    operations: list[tuple[str, str, dict[str, Any]]] = [
        (
            "search",
            "contact.search_surface",
            {"surface": "$surface", "force_profile_id": "wipe_light"},
        ),
        (
            "enable",
            "contact.enable_force",
            {"surface": "$surface", "force_profile_id": "wipe_light"},
        ),
        (
            "follow",
            "contact.follow_surface_path",
            {"path": [_target(), _target(0.2)], "motion_profile_id": "linear_slow"},
        ),
        ("disable", "contact.disable_force", {}),
    ]
    return SkillGraph(
        skill_id="force_compile_demo",
        version="1.0.0",
        name="force compile demo",
        description="Compiler force cleanup test graph.",
        skill_type=SkillType.CONTACT,
        source_demonstrations=["demo_02"],
        bindings={"$surface": {"entity_kind": "surface"}},
        nodes=[
            SkillNode(node_id=node_id, operation=operation, arguments=arguments)
            for node_id, operation, arguments in operations
        ],
        edges=[
            SkillEdge(source_node=source[0], target_node=target[0])
            for source, target in zip(operations, operations[1:], strict=False)
        ],
        start_node="search",
        terminal_nodes=["disable"],
        motion_profiles=["linear_slow"],
        force_profiles=["wipe_light"],
    )


def _import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("generated_skill_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingRuntime:
    def __init__(self, *, fail_operation: str | None = None) -> None:
        self.operations: list[str] = []
        self.fail_operation = fail_operation

    async def execute_primitive(
        self,
        operation: str,
        arguments: dict[str, Any],
        timeout_s: float | None,
        checkpoint: str | None,
    ) -> bool:
        del arguments, timeout_s, checkpoint
        self.operations.append(operation)
        if operation == self.fail_operation:
            raise RuntimeError("synthetic primitive failure")
        return True


def test_compiler_emits_parseable_import_free_deterministic_source() -> None:
    compiler = SkillCompiler()
    source = compiler.compile(_motion_graph())

    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    forbidden_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not ({"eval", "exec", "compile", "__import__"} & forbidden_calls)
    assert source == compiler.compile(_motion_graph())
    assert source.validation_report.py_compile_passed


def test_compiled_file_py_compiles_imports_and_runs_with_mock_runtime(tmp_path: Path) -> None:
    output = tmp_path / "compiled_skill.py"
    artifact = SkillCompiler().compile(_motion_graph(), output)
    module = _import_module(output)
    runtime = RecordingRuntime()

    result = asyncio.run(module.run(runtime))

    assert result is True
    assert runtime.operations == ["workspace.validate_target", "motion.move_l"]
    assert artifact.output_path == output
    assert len(artifact.checksum_sha256) == 64


def test_contact_compiler_uses_try_finally_and_releases_force_on_exception(
    tmp_path: Path,
) -> None:
    output = tmp_path / "compiled_force_skill.py"
    source = SkillCompiler().compile(_force_graph(), output)
    tree = ast.parse(source)
    assert any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(tree))
    module = _import_module(output)
    runtime = RecordingRuntime(fail_operation="contact.follow_path")

    with pytest.raises(RuntimeError, match="synthetic"):
        asyncio.run(module.run(runtime))

    assert runtime.operations[-2:] == ["contact.disable_force", "recovery.safe_retract"]
    assert runtime.operations.count("contact.disable_force") == 1


@pytest.mark.parametrize(
    "source",
    [
        "import os\nasync def run(runtime):\n    return True\n",
        "async def run(runtime):\n    eval('1 + 1')\n",
        "async def run(runtime):\n    return runtime.__class__\n",
    ],
)
def test_code_validator_rejects_import_eval_and_attribute_access(source: str) -> None:
    with pytest.raises(CodeValidationError):
        GeneratedCodeValidator().validate_source(source)


def test_compiler_rejects_unknown_operation() -> None:
    graph = _motion_graph()
    graph.nodes[1].operation = "motion.raw_move"

    with pytest.raises(CompilationError, match="whitelist"):
        SkillCompiler().compile(graph)


def test_generated_loader_accepts_relative_registry_artifact(tmp_path: Path) -> None:
    output = tmp_path / "compiled_skill.py"
    artifact = SkillCompiler().compile(_motion_graph(), output)

    run = load_compiled_run(
        Path(output.name),
        artifact_root=tmp_path,
        expected_checksum_sha256=artifact.checksum_sha256,
    )

    runtime = RecordingRuntime()
    assert asyncio.run(run(runtime)) is True


def test_generated_loader_rejects_absolute_python_path(tmp_path: Path) -> None:
    output = tmp_path / "compiled_skill.py"
    artifact = SkillCompiler().compile(_motion_graph(), output)

    with pytest.raises(CodeValidationError, match="relative artifact URI"):
        load_compiled_run(
            output,
            artifact_root=tmp_path,
            expected_checksum_sha256=artifact.checksum_sha256,
        )
