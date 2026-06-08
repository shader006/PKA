#!/usr/bin/env python3
"""
Extract a NumPy docstring/function dataset plus linked tests into JSONL files.

The script produces:
  1. problems.jsonl: one record per function with `nl` and `r`
  2. tests.jsonl: one record per matched test case linked by `problem_id`

It is intentionally conservative:
  - only Python source files are parsed
  - only functions with docstrings are considered problems
  - only problems with at least one matched test are written
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDES = {
    "benchmarks",
    "doc",
    "tools",
}

RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+([A-Za-z0-9_-]+)::\s*(.*)$")


@dataclass
class Problem:
    problem_id: str
    qualified_name: str
    aliases: list[str]
    module: str
    repo_path: str
    function_name: str
    parent_name: str | None
    signature: str
    nl: str
    r: str
    source: str
    source_type: str
    start_lineno: int
    end_lineno: int


@dataclass
class TestRecord:
    test_id: str
    problem_ids: list[str]
    test_path: str
    test_name: str
    class_name: str | None
    framework: str
    test_code: str
    start_lineno: int
    end_lineno: int


class SourceExtractor(ast.NodeVisitor):
    def __init__(self, module: str, repo_path: str, lines: list[str]) -> None:
        self.module = module
        self.repo_path = repo_path
        self.lines = lines
        self.problems: list[Problem] = []
        self.class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._maybe_add_problem(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._maybe_add_problem(node)
        self.generic_visit(node)

    def _maybe_add_problem(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not node.body:
            return

        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            return

        parent_name = ".".join(self.class_stack) if self.class_stack else None
        local_name = ".".join([*self.class_stack, node.name]) if self.class_stack else node.name
        qualified_name = f"{self.module}.{local_name}"
        aliases = {qualified_name}

        public_module = _get_set_module_value(node)
        if public_module:
            aliases.add(f"{public_module}.{node.name}")

        source = ast.get_source_segment("".join(self.lines), node) or _slice_source(
            self.lines, node.lineno, node.end_lineno
        )
        body_start = node.body[0].lineno
        if _is_docstring_expr(node.body[0]) and len(node.body) > 1:
            body_start = node.body[1].lineno
        r = _slice_source(self.lines, node.lineno, node.end_lineno, skip_until=body_start)
        signature = _extract_signature(source)

        self.problems.append(
            Problem(
                problem_id=qualified_name,
                qualified_name=qualified_name,
                aliases=sorted(aliases),
                module=self.module,
                repo_path=self.repo_path,
                function_name=node.name,
                parent_name=parent_name,
                signature=signature,
                nl=docstring,
                r=r,
                source=source,
                source_type="py",
                start_lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
            )
        )


class TestExtractor(ast.NodeVisitor):
    def __init__(
        self,
        module: str,
        repo_path: str,
        lines: list[str],
        alias_to_problem_ids: dict[str, set[str]],
    ) -> None:
        self.module = module
        self.repo_path = repo_path
        self.lines = lines
        self.alias_to_problem_ids = alias_to_problem_ids
        self.records: list[TestRecord] = []
        self.import_aliases: dict[str, str] = {}
        self.class_stack: list[str] = []
        self.helper_symbols: dict[str, set[str]] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target = alias.name
            local = alias.asname or target.split(".")[-1]
            self.import_aliases[local] = target

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module:
            return
        module = "." * node.level + node.module
        if node.level != 0:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            self.import_aliases[local] = f"{module}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._maybe_add_test(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._maybe_add_test(node)
        self.generic_visit(node)

    def _maybe_add_test(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not self._is_test_node(node):
            return

        matched_problem_ids = sorted(self._collect_problem_ids(node))
        if not matched_problem_ids:
            return

        test_code = ast.get_source_segment("".join(self.lines), node) or _slice_source(
            self.lines, node.lineno, node.end_lineno
        )
        class_name = ".".join(self.class_stack) if self.class_stack else None
        test_name = ".".join([*self.class_stack, node.name]) if self.class_stack else node.name
        test_id = f"{self.module}::{test_name}"

        self.records.append(
            TestRecord(
                test_id=test_id,
                problem_ids=matched_problem_ids,
                test_path=self.repo_path,
                test_name=node.name,
                class_name=class_name,
                framework="pytest",
                test_code=test_code,
                start_lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
            )
        )

    def _is_test_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        if node.name.startswith("test"):
            return True
        return bool(self.class_stack and self.class_stack[-1].startswith("Test"))

    def _collect_problem_ids(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        collector = SymbolCollector(self.import_aliases)
        collector.visit(node)
        problem_ids: set[str] = set()
        for symbol in collector.symbols:
            problem_ids.update(self.alias_to_problem_ids.get(symbol, set()))
        for helper_name in collector.local_calls:
            for symbol in self.helper_symbols.get(helper_name, set()):
                problem_ids.update(self.alias_to_problem_ids.get(symbol, set()))
        return problem_ids

    def build_helper_index(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if self._is_test_like_name(node.name):
                continue
            collector = SymbolCollector(self.import_aliases)
            for stmt in node.body:
                collector.visit(stmt)
            self.helper_symbols[node.name] = set(collector.symbols)

    @staticmethod
    def _is_test_like_name(name: str) -> bool:
        return name.startswith("test")


class SymbolCollector(ast.NodeVisitor):
    def __init__(self, import_aliases: dict[str, str]) -> None:
        self.import_aliases = import_aliases
        self.symbols: set[str] = set()
        self.local_calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        symbol = self._resolve_expr(node.func)
        if symbol:
            self.symbols.add(symbol)
        helper_name = self._resolve_local_name(node.func)
        if helper_name:
            self.local_calls.add(helper_name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        symbol = self.import_aliases.get(node.id)
        if symbol:
            self.symbols.add(symbol)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        symbol = self._resolve_expr(node)
        if symbol:
            self.symbols.add(symbol)
        self.generic_visit(node)

    def _resolve_expr(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.import_aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            parts = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                base = self.import_aliases.get(cur.id, cur.id)
                parts.append(base)
                return ".".join(reversed(parts))
        return None

    def _resolve_local_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name) and node.id not in self.import_aliases:
            return node.id
        return None


def _is_docstring_expr(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _get_set_module_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Name) and func.id == "set_module":
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                value = decorator.args[0].value
                if isinstance(value, str):
                    return value
    return None


def _extract_signature(source: str) -> str:
    lines = source.splitlines()
    collected: list[str] = []
    seen_def = False
    for line in lines:
        stripped = line.strip()
        if not seen_def:
            if stripped.startswith("def ") or stripped.startswith("async def "):
                seen_def = True
                collected.append(line)
                if stripped.endswith(":"):
                    break
            continue
        collected.append(line)
        if stripped.endswith(":"):
            break
    return "\n".join(collected).strip()


def _slice_source(
    lines: list[str], start_lineno: int, end_lineno: int | None, skip_until: int | None = None
) -> str:
    start = (skip_until or start_lineno) - 1
    end = end_lineno or start_lineno
    return "".join(lines[start:end]).rstrip()


def _module_name_from_path(package_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(package_root).with_suffix("")
    return ".".join((package_root.name, *rel.parts))


def _repo_relative(repo_root: Path, file_path: Path) -> str:
    return file_path.relative_to(repo_root).as_posix()


CYTHON_CLASS_RE = re.compile(r"^(\s*)(?:cdef\s+)?class\s+([A-Za-z_]\w*)\b")
CYTHON_FUNC_RE = re.compile(r"^(\s*)(?:cpdef|def)\b.*?\b([A-Za-z_]\w*)\s*\(")


def _extract_cython_problems(module: str, repo_path: str, text: str) -> list[Problem]:
    lines = text.splitlines(keepends=True)
    problems: list[Problem] = []
    class_stack: list[tuple[int, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = len(line) - len(line.lstrip(" "))
        while class_stack and indent <= class_stack[-1][0]:
            class_stack.pop()

        class_match = CYTHON_CLASS_RE.match(line)
        if class_match:
            class_stack.append((len(class_match.group(1)), class_match.group(2)))
            i += 1
            continue

        func_match = CYTHON_FUNC_RE.match(line)
        if not func_match:
            i += 1
            continue

        func_indent = len(func_match.group(1))
        start_idx = i
        sig_end = i
        while sig_end < len(lines) and not lines[sig_end].rstrip().endswith(":"):
            sig_end += 1
        if sig_end >= len(lines):
            i += 1
            continue

        body_first_idx = sig_end + 1
        while body_first_idx < len(lines):
            body_line = lines[body_first_idx]
            body_stripped = body_line.strip()
            if body_stripped and not body_stripped.startswith("#"):
                break
            body_first_idx += 1
        if body_first_idx >= len(lines):
            i = sig_end + 1
            continue

        doc_info = _extract_cython_docstring(lines, body_first_idx, func_indent)
        if not doc_info:
            i = sig_end + 1
            continue
        docstring, _, doc_end_idx = doc_info

        end_idx = _find_block_end(lines, start_idx, func_indent)
        parent_name = ".".join(name for _, name in class_stack) if class_stack else None
        function_name = func_match.group(2)
        class_names = [name for _, name in class_stack]
        local_name = ".".join([*class_names, function_name]) if class_names else function_name
        qualified_name = f"{module}.{local_name}"
        source = "".join(lines[start_idx:end_idx]).rstrip()
        r_start_idx = doc_end_idx + 1
        while r_start_idx < end_idx and not lines[r_start_idx].strip():
            r_start_idx += 1
        r = "".join(lines[r_start_idx:end_idx]).rstrip()
        signature = "".join(lines[start_idx : sig_end + 1]).rstrip()

        problems.append(
            Problem(
                problem_id=qualified_name,
                qualified_name=qualified_name,
                aliases=[qualified_name],
                module=module,
                repo_path=repo_path,
                function_name=function_name,
                parent_name=parent_name,
                signature=signature,
                nl=docstring,
                r=r,
                source=source,
                source_type="pyx",
                start_lineno=start_idx + 1,
                end_lineno=end_idx,
            )
        )
        i = end_idx

    return problems


def _extract_cython_docstring(
    lines: list[str], start_idx: int, func_indent: int
) -> tuple[str, int, int] | None:
    line = lines[start_idx]
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(" "))
    if indent <= func_indent:
        return None
    if not (stripped.startswith('"""') or stripped.startswith("'''")):
        return None

    quote = stripped[:3]
    doc_lines = [line[indent:]]
    if stripped.count(quote) >= 2 and len(stripped) > 3:
        try:
            value = ast.literal_eval("".join(doc_lines))
        except (SyntaxError, ValueError):
            return None
        if not isinstance(value, str):
            return None
        return value, start_idx, start_idx

    end_idx = start_idx + 1
    while end_idx < len(lines):
        doc_lines.append(lines[end_idx][indent:])
        if quote in lines[end_idx]:
            break
        end_idx += 1

    try:
        value = ast.literal_eval("".join(doc_lines))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, str):
        return None
    return value, start_idx, end_idx


def _find_block_end(lines: list[str], start_idx: int, base_indent: int) -> int:
    idx = start_idx + 1
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            indent = len(line) - len(line.lstrip(" "))
            if indent <= base_indent:
                break
        idx += 1
    return idx


def _normalize_doc_symbol(name: str, currentmodule: str | None) -> str | None:
    symbol = name.strip()
    if not symbol:
        return None
    symbol = symbol.lstrip("~")
    symbol = symbol.split()[0]
    if symbol.startswith(":"):
        return None
    if currentmodule and not symbol.startswith("numpy"):
        symbol = f"{currentmodule}.{symbol}"
    return symbol


def collect_documented_symbols(repo_root: Path) -> set[str]:
    reference_root = repo_root / "doc" / "source" / "reference"
    symbols: set[str] = set()
    if not reference_root.exists():
        return symbols

    for path in reference_root.rglob("*.rst"):
        rel_parts = path.relative_to(reference_root).parts
        if rel_parts and rel_parts[0] == "c-api":
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        currentmodule: str | None = None
        i = 0
        while i < len(lines):
            match = RST_DIRECTIVE_RE.match(lines[i])
            if not match:
                i += 1
                continue

            directive, value = match.groups()
            directive = directive.lower()
            value = value.strip()

            if directive == "currentmodule":
                currentmodule = value or None
                i += 1
                continue

            if directive in {"autofunction", "autoclass", "autoattribute", "automethod"}:
                symbol = _normalize_doc_symbol(value, currentmodule)
                if symbol:
                    symbols.add(symbol)
                i += 1
                continue

            if directive == "autosummary":
                base_indent = len(lines[i]) - len(lines[i].lstrip(" "))
                i += 1
                while i < len(lines):
                    line = lines[i]
                    stripped = line.strip()
                    indent = len(line) - len(line.lstrip(" "))
                    if stripped and indent <= base_indent:
                        break
                    if stripped and not stripped.startswith(":"):
                        symbol = _normalize_doc_symbol(stripped, currentmodule)
                        if symbol:
                            symbols.add(symbol)
                    i += 1
                continue

            i += 1

    return symbols


def _iter_source_files(root: Path, include_tests: bool) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".pyx"}:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in DEFAULT_EXCLUDES for part in rel_parts):
            continue
        if not include_tests and "tests" in rel_parts:
            continue
        if include_tests and "tests" not in rel_parts:
            continue
        yield path


def collect_problems(repo_root: Path, package_root: Path) -> list[Problem]:
    problems: list[Problem] = []
    for path in _iter_source_files(package_root, include_tests=False):
        if path.name == "conftest.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        module = _module_name_from_path(package_root, path)
        repo_path = _repo_relative(repo_root, path)
        if path.suffix == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            extractor = SourceExtractor(
                module=module,
                repo_path=repo_path,
                lines=text.splitlines(keepends=True),
            )
            extractor.visit(tree)
            problems.extend(extractor.problems)
            continue

        problems.extend(_extract_cython_problems(module, repo_path, text))
    return problems


def collect_tests(
    repo_root: Path,
    package_root: Path,
    alias_to_problem_ids: dict[str, set[str]],
) -> list[TestRecord]:
    tests: list[TestRecord] = []
    for path in _iter_source_files(package_root, include_tests=True):
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        extractor = TestExtractor(
            module=_module_name_from_path(package_root, path),
            repo_path=_repo_relative(repo_root, path),
            lines=text.splitlines(keepends=True),
            alias_to_problem_ids=alias_to_problem_ids,
        )
        extractor.visit(tree)
        extractor.build_helper_index(tree)
        extractor.records.clear()
        extractor.class_stack.clear()
        extractor.visit(tree)
        tests.extend(extractor.records)
    return tests


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _problem_is_documented(problem: Problem, documented_symbols: set[str]) -> bool:
    candidates = {
        problem.problem_id,
        problem.qualified_name,
        problem.function_name,
        *problem.aliases,
    }
    if problem.parent_name:
        candidates.add(f"numpy.{problem.parent_name}.{problem.function_name}")
    return any(candidate in documented_symbols for candidate in candidates)


def build_dataset(
    repo_root: Path,
    output_dir: Path,
    documented_only: bool = False,
) -> tuple[Path, Path, int, int]:
    package_root = repo_root / "numpy"
    problems = collect_problems(repo_root, package_root)

    alias_to_problem_ids: dict[str, set[str]] = defaultdict(set)
    problem_by_id: dict[str, Problem] = {}
    for problem in problems:
        problem_by_id[problem.problem_id] = problem
        alias_to_problem_ids[problem.problem_id].add(problem.problem_id)
        alias_to_problem_ids[problem.qualified_name].add(problem.problem_id)
        alias_to_problem_ids[problem.function_name].add(problem.problem_id)
        for alias in problem.aliases:
            alias_to_problem_ids[alias].add(problem.problem_id)

    tests = collect_tests(repo_root, package_root, alias_to_problem_ids)
    matched_problem_ids = {problem_id for test in tests for problem_id in test.problem_ids}
    if documented_only:
        documented_symbols = collect_documented_symbols(repo_root)
        matched_problem_ids = {
            problem_id
            for problem_id in matched_problem_ids
            if _problem_is_documented(problem_by_id[problem_id], documented_symbols)
        }

    problem_records = []
    for problem_id in sorted(matched_problem_ids):
        problem = problem_by_id[problem_id]
        problem_records.append(
            {
                "problem_id": problem.problem_id,
                "qualified_name": problem.qualified_name,
                "aliases": problem.aliases,
                "repo_path": problem.repo_path,
                "module": problem.module,
                "function_name": problem.function_name,
                "parent_name": problem.parent_name,
                "signature": problem.signature,
                "nl": problem.nl,
                "r": problem.r,
                "source": problem.source,
                "source_type": problem.source_type,
                "start_lineno": problem.start_lineno,
                "end_lineno": problem.end_lineno,
                "has_tests": True,
            }
        )

    test_records = []
    for test in tests:
        for problem_id in test.problem_ids:
            if problem_id not in matched_problem_ids:
                continue
            test_records.append(
                {
                    "problem_id": problem_id,
                    "test_id": test.test_id,
                    "test_path": test.test_path,
                    "test_name": test.test_name,
                    "class_name": test.class_name,
                    "framework": test.framework,
                    "test_code": test.test_code,
                    "start_lineno": test.start_lineno,
                    "end_lineno": test.end_lineno,
                }
            )

    problems_path = output_dir / "numpy_problems.jsonl"
    tests_path = output_dir / "numpy_tests.jsonl"
    write_jsonl(problems_path, problem_records)
    write_jsonl(tests_path, test_records)
    return problems_path, tests_path, len(problem_records), len(test_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract NumPy functions/docstrings and linked tests into JSONL files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("d:/code/web/numpy"),
        help="Path to the NumPy repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("d:/code/web/out"),
        help="Directory where JSONL files will be written.",
    )
    parser.add_argument(
        "--documented-only",
        action="store_true",
        help="Keep only problems that appear in NumPy's doc/source/reference docs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()

    problems_path, tests_path, problem_count, test_count = build_dataset(
        repo_root,
        output_dir,
        documented_only=args.documented_only,
    )
    print(f"Wrote {problem_count} problems to {problems_path}")
    print(f"Wrote {test_count} tests to {tests_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
