"""Validate deterministic compiler output without executing generated code."""

from __future__ import annotations

import ast
import hashlib
import py_compile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from robot_skill_system.exceptions import CodeValidationError

_MAX_SOURCE_BYTES = 1_000_000
_MAX_AST_NODES = 20_000
_BANNED_CALL_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
    }
)

_ALLOWED_AST_TYPES = (
    ast.Module,
    ast.Expr,
    ast.AsyncFunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.While,
    ast.Compare,
    ast.Eq,
    ast.IsNot,
    ast.If,
    ast.Await,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Return,
    ast.Try,
    ast.UnaryOp,
    ast.USub,
)


@dataclass(frozen=True)
class GeneratedCodeValidationReport:
    """Evidence produced by AST and bytecode syntax validation."""

    valid: bool
    checksum_sha256: str
    ast_node_count: int
    py_compile_passed: bool
    module_shape_valid: bool
    errors: tuple[str, ...] = ()


class _CompilerSubsetVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if not isinstance(node, _ALLOWED_AST_TYPES):
            self.errors.append(f"AST node {type(node).__name__} is not allowed")
            return
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.node_count += 1
        self.errors.append("generated skill modules may not import packages")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.node_count += 1
        self.errors.append("generated skill modules may not import packages")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALL_NAMES:
                self.errors.append(f"call to {node.func.id!r} is forbidden")
            else:
                self.errors.append(f"free function call {node.func.id!r} is not allowed")
        elif isinstance(node.func, ast.Attribute):
            valid_runtime_call = (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "runtime"
                and node.func.attr == "execute_primitive"
            )
            if not valid_runtime_call:
                self.errors.append("only runtime.execute_primitive calls are allowed")
        else:
            self.errors.append("indirect function calls are not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id == "runtime"
            and node.attr == "execute_primitive"
        ):
            self.errors.append("arbitrary attribute access is not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            self.errors.append(f"constant type {type(node.value).__name__} is not allowed")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:  # noqa: N802
        if not (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            self.errors.append("only negative numeric literals may use unary operators")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        if node.name != "run":
            self.errors.append("the only generated function must be named 'run'")
        if node.decorator_list:
            self.errors.append("generated functions may not have decorators")
        if node.returns is not None or any(
            argument.annotation is not None for argument in node.args.args
        ):
            self.errors.append("generated functions may not contain annotations")
        self.generic_visit(node)


class GeneratedCodeValidator:
    """Fail closed on any source outside the compiler-owned AST subset."""

    def inspect_source(
        self, source: str, *, run_py_compile: bool = True
    ) -> GeneratedCodeValidationReport:
        """Inspect source and return all findings without importing or executing it."""

        checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
        errors: list[str] = []
        if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            errors.append("generated source exceeds the validation size limit")
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as error:
            return GeneratedCodeValidationReport(
                valid=False,
                checksum_sha256=checksum,
                ast_node_count=0,
                py_compile_passed=False,
                module_shape_valid=False,
                errors=(f"syntax error: {error}",),
            )

        visitor = _CompilerSubsetVisitor()
        visitor.visit(tree)
        errors.extend(visitor.errors)
        if visitor.node_count > _MAX_AST_NODES:
            errors.append("generated source exceeds the AST node limit")
        shape_errors = self._module_shape_errors(tree)
        errors.extend(shape_errors)
        py_compile_passed = False
        if run_py_compile and not errors:
            try:
                self._py_compile(source)
                py_compile_passed = True
            except (OSError, py_compile.PyCompileError) as error:
                errors.append(f"py_compile failed: {error}")
        elif not run_py_compile:
            py_compile_passed = True
        return GeneratedCodeValidationReport(
            valid=not errors,
            checksum_sha256=checksum,
            ast_node_count=visitor.node_count,
            py_compile_passed=py_compile_passed,
            module_shape_valid=not shape_errors,
            errors=tuple(dict.fromkeys(errors)),
        )

    def validate_source(
        self, source: str, *, run_py_compile: bool = True
    ) -> GeneratedCodeValidationReport:
        """Return a report or raise ``CodeValidationError``."""

        report = self.inspect_source(source, run_py_compile=run_py_compile)
        if not report.valid:
            raise CodeValidationError("; ".join(report.errors))
        return report

    def validate_file(self, path: Path) -> GeneratedCodeValidationReport:
        """Read and validate a generated UTF-8 Python module."""

        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise CodeValidationError(f"cannot read generated module {path}: {error}") from error
        return self.validate_source(source)

    @staticmethod
    def _module_shape_errors(tree: ast.Module) -> list[str]:
        body = list(tree.body)
        if body and isinstance(body[0], ast.Expr):
            expression = body[0].value
            if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.AsyncFunctionDef):
            return ["generated module must contain only one async run(runtime) function"]
        function = body[0]
        positional = function.args.posonlyargs + function.args.args
        if (
            len(positional) != 1
            or positional[0].arg != "runtime"
            or function.args.vararg is not None
            or function.args.kwarg is not None
            or function.args.kwonlyargs
            or function.args.defaults
        ):
            return ["generated run function must accept exactly one runtime argument"]
        return []

    @staticmethod
    def _py_compile(source: str) -> None:
        with tempfile.TemporaryDirectory(prefix="robot_skill_validation_") as temporary_dir:
            source_path = Path(temporary_dir) / "compiled_skill.py"
            bytecode_path = Path(temporary_dir) / "compiled_skill.pyc"
            source_path.write_text(source, encoding="utf-8")
            py_compile.compile(
                str(source_path),
                cfile=str(bytecode_path),
                doraise=True,
            )


def validate_generated_code(source: str) -> GeneratedCodeValidationReport:
    """Validate deterministic compiler output."""

    return GeneratedCodeValidator().validate_source(source)


# Short aliases for callers and documentation.
CodeValidator = GeneratedCodeValidator
CodeValidationReport = GeneratedCodeValidationReport
