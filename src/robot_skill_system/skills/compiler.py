"""Deterministically compile validated SkillGraphs with Python's AST builder.

The compiler never accepts Python fragments.  Every emitted call name comes from
the primitive registry and every argument has passed an operation-specific
Pydantic schema.  Generated modules are import-free and contain only task order;
the runtime retains geometry binding, profiles, safety policy, and execution.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_skill_system.exceptions import CompilationError
from robot_skill_system.primitives.registry import PrimitiveRegistry, get_default_registry
from robot_skill_system.skills.code_validator import (
    GeneratedCodeValidationReport,
    GeneratedCodeValidator,
)
from robot_skill_system.skills.graph import SkillGraphValidationReport, SkillGraphValidator
from robot_skill_system.skills.models import SkillGraph, SkillManifest


class CompilationArtifact(str):
    """Generated source string plus checksums and optional persisted path.

    This is a ``str`` subclass so callers may directly pass the result to
    ``ast.parse`` or ``Path.write_text`` while still accessing artifact metadata.
    """

    source: str
    checksum_sha256: str
    graph_checksum_sha256: str
    output_path: Path | None
    validation_report: GeneratedCodeValidationReport

    def __new__(
        cls,
        source: str,
        *,
        graph_checksum_sha256: str,
        output_path: Path | None,
        validation_report: GeneratedCodeValidationReport,
    ) -> CompilationArtifact:
        instance = super().__new__(cls, source)
        instance.source = source
        instance.checksum_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        instance.graph_checksum_sha256 = graph_checksum_sha256
        instance.output_path = output_path
        instance.validation_report = validation_report
        return instance


def _literal(value: Any) -> ast.expr:
    """Convert validated JSON-compatible data into AST literals only."""

    if value is None or isinstance(value, (str, bool, int)):
        return ast.Constant(value=value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CompilationError("non-finite numeric literals are forbidden")
        return ast.Constant(value=value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CompilationError("generated argument dictionaries require string keys")
        sorted_items = sorted(value.items())
        return ast.Dict(
            keys=[ast.Constant(value=key) for key, _ in sorted_items],
            values=[_literal(item) for _, item in sorted_items],
        )
    if isinstance(value, tuple):
        return ast.Tuple(elts=[_literal(item) for item in value], ctx=ast.Load())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ast.List(elts=[_literal(item) for item in value], ctx=ast.Load())
    raise CompilationError(f"unsupported generated literal type {type(value).__name__}")


def _assign(name: str, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)


def _runtime_call(
    operation: str,
    arguments: Mapping[str, Any],
    timeout_s: float,
    checkpoint: str | None,
) -> ast.Await:
    return ast.Await(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="runtime", ctx=ast.Load()),
                attr="execute_primitive",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(arg="operation", value=ast.Constant(value=operation)),
                ast.keyword(arg="arguments", value=_literal(arguments)),
                ast.keyword(arg="timeout_s", value=ast.Constant(value=timeout_s)),
                ast.keyword(arg="checkpoint", value=ast.Constant(value=checkpoint)),
            ],
        )
    )


class SkillCompiler:
    """Compile a graph using a fixed registry and AST-only source generator."""

    def __init__(
        self,
        registry: PrimitiveRegistry | None = None,
        *,
        code_validator: GeneratedCodeValidator | None = None,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.graph_validator = SkillGraphValidator(self.registry)
        self.code_validator = code_validator or GeneratedCodeValidator()

    def compile_source(self, graph: SkillGraph | Mapping[str, Any]) -> str:
        """Validate ``graph`` and return deterministic import-free Python source."""

        validated_graph = (
            graph if isinstance(graph, SkillGraph) else SkillGraph.model_validate(graph)
        )
        report = self.graph_validator.validate(validated_graph)
        module = self._build_module(validated_graph, report)
        ast.fix_missing_locations(module)
        source = ast.unparse(module).rstrip() + "\n"
        self.code_validator.validate_source(source)
        return source

    def compile(
        self,
        graph: SkillGraph | Mapping[str, Any],
        output_path: Path | str | None = None,
    ) -> CompilationArtifact:
        """Compile, validate, optionally persist, and return source/checksum evidence."""

        try:
            validated_graph = (
                graph if isinstance(graph, SkillGraph) else SkillGraph.model_validate(graph)
            )
            report = self.graph_validator.validate(validated_graph)
            module = self._build_module(validated_graph, report)
            ast.fix_missing_locations(module)
            source = ast.unparse(module).rstrip() + "\n"
            code_report = self.code_validator.validate_source(source)
            graph_checksum = self.graph_checksum(validated_graph)
            resolved_path = Path(output_path) if output_path is not None else None
            if resolved_path is not None:
                if resolved_path.suffix != ".py":
                    raise CompilationError("compiled skill output path must end in .py")
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write(resolved_path, source)
                self.code_validator.validate_file(resolved_path)
            return CompilationArtifact(
                source,
                graph_checksum_sha256=graph_checksum,
                output_path=resolved_path,
                validation_report=code_report,
            )
        except CompilationError:
            raise
        except Exception as error:
            raise CompilationError(f"SkillGraph compilation failed: {error}") from error

    def _build_module(
        self,
        graph: SkillGraph,
        report: SkillGraphValidationReport,
    ) -> ast.Module:
        node_by_id = {node.node_id: node for node in graph.nodes}
        uses_force = any(
            self.registry.canonical_operation_name(node.operation)
            in {"contact.enable_force", "contact.follow_path", "contact.disable_force"}
            for node in graph.nodes
        )
        loop_body = self._build_dispatch_chain(graph, report, node_by_id, uses_force)
        loop = ast.While(
            test=ast.Compare(
                left=ast.Name(id="current_node", ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[loop_body],
            orelse=[],
        )
        function_body: list[ast.stmt] = [
            _assign("current_node", ast.Constant(value=graph.start_node))
        ]
        if uses_force:
            function_body.append(_assign("force_mode_active", ast.Constant(value=False)))
            cleanup_call = _runtime_call(
                "contact.disable_force",
                {},
                self.registry.metadata("contact.disable_force").maximum_timeout_s,
                None,
            )
            retract_arguments = self.registry.validate_arguments(
                "recovery.safe_retract", {}
            ).model_dump(mode="json", exclude_none=True)
            force_surface_references = {
                arguments["surface"]
                for node_id, arguments in report.normalized_arguments.items()
                if self.registry.canonical_operation_name(node_by_id[node_id].operation)
                == "contact.enable_force"
                and isinstance(arguments.get("surface"), str)
            }
            if len(force_surface_references) == 1:
                retract_arguments["surface"] = force_surface_references.pop()
            retract_call = _runtime_call(
                "recovery.safe_retract",
                retract_arguments,
                self.registry.metadata("recovery.safe_retract").maximum_timeout_s,
                None,
            )
            function_body.append(
                ast.Try(
                    body=[loop],
                    handlers=[],
                    orelse=[],
                    finalbody=[
                        ast.If(
                            test=ast.Name(id="force_mode_active", ctx=ast.Load()),
                            body=[
                                ast.Try(
                                    body=[ast.Expr(value=cleanup_call)],
                                    handlers=[],
                                    orelse=[],
                                    finalbody=[ast.Expr(value=retract_call)],
                                )
                            ],
                            orelse=[],
                        )
                    ],
                )
            )
        else:
            function_body.append(loop)
        function_body.append(ast.Return(value=ast.Constant(value=True)))
        run_function = ast.AsyncFunctionDef(
            name="run",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="runtime")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=function_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return ast.Module(
            body=[
                ast.Expr(
                    value=ast.Constant(
                        value=(
                            "Deterministically generated robot skill. "
                            "Safety policy and profiles are runtime-owned."
                        )
                    )
                ),
                run_function,
            ],
            type_ignores=[],
        )

    def _build_dispatch_chain(
        self,
        graph: SkillGraph,
        report: SkillGraphValidationReport,
        node_by_id: Mapping[str, Any],
        uses_force: bool,
    ) -> ast.If:
        fallback: list[ast.stmt] = [ast.Return(value=ast.Constant(value=False))]
        for node_id in reversed(report.topological_node_ids):
            node = node_by_id[node_id]
            canonical_operation = self.registry.canonical_operation_name(node.operation)
            invocation_statements: list[ast.stmt] = []
            if uses_force and canonical_operation == "contact.enable_force":
                # Set conservatively before awaiting: the adapter might enter
                # compliance and then raise before reporting success.
                invocation_statements.append(
                    _assign("force_mode_active", ast.Constant(value=True))
                )
            invocation_statements.append(
                _assign(
                    "primitive_ok",
                    _runtime_call(
                        canonical_operation,
                        report.normalized_arguments[node_id],
                        report.resolved_timeouts_s[node_id],
                        node.checkpoint_id,
                    ),
                )
            )
            success_body: list[ast.stmt] = []
            if uses_force and canonical_operation == "contact.disable_force":
                success_body.append(_assign("force_mode_active", ast.Constant(value=False)))
            success_target = report.transitions[node_id].success
            failure_target = report.transitions[node_id].failure
            if success_target is None:
                success_body.append(ast.Return(value=ast.Constant(value=True)))
            else:
                success_body.append(_assign("current_node", ast.Constant(value=success_target)))
            failure_body: list[ast.stmt]
            if failure_target is None:
                failure_body = [ast.Return(value=ast.Constant(value=False))]
            else:
                failure_body = [_assign("current_node", ast.Constant(value=failure_target))]
            invocation_statements.append(
                ast.If(
                    test=ast.Name(id="primitive_ok", ctx=ast.Load()),
                    body=success_body,
                    orelse=failure_body,
                )
            )
            fallback = [
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id="current_node", ctx=ast.Load()),
                        ops=[ast.Eq()],
                        comparators=[ast.Constant(value=node_id)],
                    ),
                    body=invocation_statements,
                    orelse=fallback,
                )
            ]
        first = fallback[0]
        if not isinstance(first, ast.If):
            raise CompilationError("validated SkillGraph contains no dispatchable nodes")
        return first

    @staticmethod
    def graph_checksum(graph: SkillGraph) -> str:
        """Return a canonical SHA-256 checksum of the validated graph schema."""

        encoded = json.dumps(
            graph.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, source: str) -> None:
        """Atomically replace a generated artifact from a same-directory temp file."""

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(source)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise CompilationError(
                f"cannot atomically write generated skill {path}: {error}"
            ) from error

    @staticmethod
    def file_checksum(path: Path) -> str:
        """Return the SHA-256 checksum for one artifact file."""

        digest = hashlib.sha256()
        try:
            with path.open("rb") as file_handle:
                while chunk := file_handle.read(64 * 1024):
                    digest.update(chunk)
        except OSError as error:
            raise CompilationError(f"cannot checksum artifact {path}: {error}") from error
        return digest.hexdigest()

    @classmethod
    def verify_checksum(cls, path: Path, expected_sha256: str) -> bool:
        """Verify one artifact against manifest checksum metadata."""

        return cls.file_checksum(path) == expected_sha256

    @classmethod
    def verify_manifest(cls, manifest: SkillManifest, artifact_root: Path) -> bool:
        """Verify all local files named by a trusted manifest.

        URI values must be relative filesystem paths.  Network schemes and path
        traversal are rejected; external artifact stores should verify content in
        their own adapter before materializing it locally.
        """

        checks = [
            (manifest.skill_graph_uri, manifest.skill_graph_checksum_sha256),
            (manifest.compiled_skill_uri, manifest.compiled_skill_checksum_sha256),
        ]
        if manifest.validation_report_uri and manifest.validation_report_checksum_sha256:
            checks.append(
                (
                    manifest.validation_report_uri,
                    manifest.validation_report_checksum_sha256,
                )
            )
        resolved_root = artifact_root.resolve()
        for uri, expected in checks:
            relative_path = Path(uri)
            if relative_path.is_absolute() or "://" in uri:
                raise CompilationError("manifest verification accepts only relative local URIs")
            candidate = (resolved_root / relative_path).resolve()
            if resolved_root not in candidate.parents:
                raise CompilationError("manifest URI escapes the artifact root")
            if not cls.verify_checksum(candidate, expected):
                return False
        return True


def compile_skill(
    graph: SkillGraph | Mapping[str, Any],
    output_path: Path | str | None = None,
    *,
    registry: PrimitiveRegistry | None = None,
) -> CompilationArtifact:
    """Compile a graph with the default closed whitelist."""

    return SkillCompiler(registry).compile(graph, output_path)


# Descriptive name used by design documents.
DeterministicSkillCompiler = SkillCompiler
