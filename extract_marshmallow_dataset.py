#!/usr/bin/env python3
"""
Extract a marshmallow docstring/function dataset plus linked tests into JSONL files.

Outputs:
  - marshmallow_problems.jsonl
  - marshmallow_tests.jsonl
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional UI dependency
    tqdm = None  # type: ignore[assignment]


RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+([A-Za-z0-9_-]+)::\s*(.*)$")


def _progress(iterable: Iterable, desc: str, unit: str = "it") -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)


def _progress_bar(total: int, desc: str, unit: str = "it"):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit)


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
    prompt: str
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
    def __init__(
        self,
        module: str,
        repo_path: str,
        lines: list[str],
        public_aliases: dict[str, set[str]],
    ) -> None:
        self.module = module
        self.repo_path = repo_path
        self.lines = lines
        self.public_aliases = public_aliases
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
        aliases = {qualified_name, *self.public_aliases.get(qualified_name, set())}

        source = ast.get_source_segment("".join(self.lines), node) or _slice_source(
            self.lines, node.lineno, node.end_lineno
        )
        body_start = node.body[0].lineno
        if _is_docstring_expr(node.body[0]) and len(node.body) > 1:
            body_start = node.body[1].lineno
        prompt_end = node.body[0].end_lineno if _is_docstring_expr(node.body[0]) else node.lineno
        prompt = _slice_source(self.lines, node.lineno, prompt_end) + "\n"
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
                prompt=prompt,
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
        if not node.module or node.level != 0:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            self.import_aliases[local] = f"{node.module}.{alias.name}"

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
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


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
                if _signature_line_ends(stripped):
                    break
            continue
        collected.append(line)
        if _signature_line_ends(stripped):
            break
    return "\n".join(collected).strip()


def _signature_line_ends(stripped_line: str) -> bool:
    return stripped_line.split("#", 1)[0].rstrip().endswith(":")


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


def _iter_py_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.py")


def collect_public_aliases(repo_root: Path) -> dict[str, set[str]]:
    init_path = repo_root / "src" / "marshmallow" / "__init__.py"
    aliases: dict[str, set[str]] = defaultdict(set)
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return aliases

    import_map: dict[str, str] = {}
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                import_map[local] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exported.add(elt.value)

    for name in exported:
        target = import_map.get(name)
        if target:
            aliases[target].add(f"marshmallow.{name}")
    return aliases


def collect_problems(repo_root: Path, package_root: Path) -> list[Problem]:
    public_aliases = collect_public_aliases(repo_root)
    problems: list[Problem] = []
    paths = [path for path in _iter_py_files(package_root) if "__pycache__" not in path.parts]
    for path in _progress(paths, "Collecting source functions", "file"):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        extractor = SourceExtractor(
            module=_module_name_from_path(package_root, path),
            repo_path=_repo_relative(repo_root, path),
            lines=text.splitlines(keepends=True),
            public_aliases=public_aliases,
        )
        extractor.visit(tree)
        problems.extend(extractor.problems)
    return problems


def collect_tests(
    repo_root: Path, tests_root: Path, alias_to_problem_ids: dict[str, set[str]]
) -> list[TestRecord]:
    tests: list[TestRecord] = []
    paths = [
        path
        for path in _iter_py_files(tests_root)
        if path.name != "conftest.py" and "mypy_test_cases" not in path.parts
    ]
    for path in _progress(paths, "Collecting tests", "file"):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
        extractor = TestExtractor(
            module=module,
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


class PytestTracePlugin:
    def __init__(self, repo_root: Path, problems: list[Problem]) -> None:
        self.repo_root = repo_root.resolve()
        self.problem_ids_by_test_id: dict[str, set[str]] = defaultdict(set)
        self.test_records_by_id: dict[str, TestRecord] = {}
        self.current_test_id: str | None = None
        self.problems_by_file_name: dict[tuple[str, str], list[Problem]] = defaultdict(list)
        self.progress_bar = None

        for problem in problems:
            source_path = (self.repo_root / problem.repo_path).resolve()
            key = (_trace_path_key(source_path), problem.function_name)
            self.problems_by_file_name[key].append(problem)

    def pytest_collection_finish(self, session) -> None:  # type: ignore[no-untyped-def]
        self.progress_bar = _progress_bar(len(session.items), "Tracing pytest", "test")

    def pytest_runtest_logfinish(self, nodeid, location) -> None:  # type: ignore[no-untyped-def]
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def pytest_sessionfinish(self, session, exitstatus) -> None:  # type: ignore[no-untyped-def]
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def pytest_runtest_setup(self, item) -> None:  # type: ignore[no-untyped-def]
        self._ensure_test_record(item)

    def pytest_runtest_call(self, item) -> None:  # type: ignore[no-untyped-def]
        self._ensure_test_record(item)
        self.current_test_id = _test_id_from_pytest_nodeid(item.nodeid)
        sys.settrace(self._trace)

    def pytest_runtest_teardown(self, item) -> None:  # type: ignore[no-untyped-def]
        sys.settrace(None)
        self.current_test_id = None

    def _trace(self, frame, event: str, arg):  # type: ignore[no-untyped-def]
        if event != "call" or self.current_test_id is None:
            return self._trace

        problem_id = self._problem_id_for_frame(frame)
        if problem_id:
            self.problem_ids_by_test_id[self.current_test_id].add(problem_id)
        return self._trace

    def _problem_id_for_frame(self, frame) -> str | None:  # type: ignore[no-untyped-def]
        try:
            filename = _trace_path_key(Path(frame.f_code.co_filename).resolve())
        except OSError:
            return None

        candidates = self.problems_by_file_name.get((filename, frame.f_code.co_name), [])
        if not candidates:
            return None

        first_lineno = frame.f_code.co_firstlineno
        for problem in candidates:
            if problem.start_lineno == first_lineno:
                return problem.problem_id

        for problem in candidates:
            if problem.start_lineno - 20 <= first_lineno <= problem.start_lineno:
                return problem.problem_id
        return None

    def _ensure_test_record(self, item) -> None:  # type: ignore[no-untyped-def]
        test_id = _test_id_from_pytest_nodeid(item.nodeid)
        if test_id in self.test_records_by_id:
            return

        path = Path(str(item.path))
        rel_path = _repo_relative(self.repo_root, path)
        test_name, class_name = _test_name_and_class_from_pytest_nodeid(item.nodeid)
        try:
            source_lines, start_lineno = inspect.getsourcelines(item.obj)
            test_code = "".join(source_lines).rstrip()
            end_lineno = start_lineno + len(source_lines) - 1
        except (OSError, TypeError):
            test_code = ""
            start_lineno = 0
            end_lineno = 0

        self.test_records_by_id[test_id] = TestRecord(
            test_id=test_id,
            problem_ids=[],
            test_path=rel_path,
            test_name=test_name,
            class_name=class_name,
            framework="pytest-trace",
            test_code=test_code,
            start_lineno=start_lineno,
            end_lineno=end_lineno,
        )

    def records(self) -> list[TestRecord]:
        records: list[TestRecord] = []
        for test_id, problem_ids in sorted(self.problem_ids_by_test_id.items()):
            record = self.test_records_by_id[test_id]
            records.append(
                TestRecord(
                    test_id=record.test_id,
                    problem_ids=sorted(problem_ids),
                    test_path=record.test_path,
                    test_name=record.test_name,
                    class_name=record.class_name,
                    framework=record.framework,
                    test_code=record.test_code,
                    start_lineno=record.start_lineno,
                    end_lineno=record.end_lineno,
                )
            )
        return records


def collect_tests_by_tracing(repo_root: Path, tests_root: Path, problems: list[Problem]) -> list[TestRecord]:
    try:
        import pytest
    except ImportError as error:
        raise RuntimeError(
            "pytest is required for --match-mode trace. "
            "Install marshmallow test dependencies or use --match-mode direct."
        ) from error

    plugin = PytestTracePlugin(repo_root, problems)
    old_cwd = Path.cwd()
    old_path = sys.path[:]
    try:
        os.chdir(repo_root)
        sys.path.insert(0, str((repo_root / "src").resolve()))
        sys.path.insert(0, str(repo_root.resolve()))
        exit_code = pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "--continue-on-collection-errors",
                "-k",
                "not test_from_timestamp_with_overflow_value",
                str(tests_root),
            ],
            plugins=[plugin],
        )
    finally:
        sys.settrace(None)
        sys.path[:] = old_path
        os.chdir(old_cwd)

    if exit_code == 1:
        print(
            "Warning: pytest reported test failures while tracing; "
            "writing coverage observed before/during those failures.",
            file=sys.stderr,
        )
    elif exit_code not in {0, 1}:
        raise RuntimeError(f"pytest tracing failed with exit code {exit_code}")
    return plugin.records()


def _trace_path_key(path: Path) -> str:
    return str(path).casefold()


def _test_id_from_pytest_nodeid(nodeid: str) -> str:
    path_part, *parts = nodeid.split("::")
    module = Path(path_part).with_suffix("").as_posix().replace("/", ".")
    if not parts:
        return module
    return f"{module}::{'.'.join(parts)}"


def _test_name_and_class_from_pytest_nodeid(nodeid: str) -> tuple[str, str | None]:
    _, *parts = nodeid.split("::")
    if not parts:
        return "", None
    return parts[-1], ".".join(parts[:-1]) or None


def _normalize_doc_symbol(name: str, currentmodule: str | None) -> str | None:
    symbol = name.strip().lstrip("~")
    if not symbol or symbol.startswith(":"):
        return None
    symbol = symbol.split()[0]
    if currentmodule and "." not in symbol:
        symbol = f"{currentmodule}.{symbol}"
    return symbol


def collect_documented_symbols(repo_root: Path, problems: list[Problem]) -> set[str]:
    docs_root = repo_root / "docs"
    documented: set[str] = set()
    by_module: dict[str, set[str]] = defaultdict(set)
    public_by_module: dict[str, set[str]] = defaultdict(set)
    public_by_class: dict[str, set[str]] = defaultdict(set)
    public_aliases: set[str] = set()

    for problem in problems:
        by_module[problem.module].add(problem.problem_id)
        for alias in problem.aliases:
            documented.add(alias) if False else None
            if alias not in {problem.problem_id, problem.qualified_name}:
                public_aliases.add(alias)
        is_public = not problem.function_name.startswith("_")
        if problem.parent_name:
            if is_public:
                public_by_class[f"{problem.module}.{problem.parent_name}"].add(problem.problem_id)
        elif is_public:
            public_by_module[problem.module].add(problem.problem_id)

    doc_paths = list(docs_root.rglob("*.rst"))
    for path in _progress(doc_paths, "Collecting documented API", "file"):
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
            base_indent = len(lines[i]) - len(lines[i].lstrip(" "))
            has_members = False
            exclude_members: set[str] = set()

            j = i + 1
            while j < len(lines):
                raw = lines[j]
                stripped = raw.strip()
                indent = len(raw) - len(raw.lstrip(" "))
                if stripped and indent <= base_indent:
                    break
                if stripped == ":members:":
                    has_members = True
                elif stripped.startswith(":exclude-members:"):
                    members = stripped.split(":", 2)[-1].strip()
                    exclude_members.update(x.strip() for x in members.split(",") if x.strip())
                j += 1

            if directive == "currentmodule":
                currentmodule = value or None
            elif directive in {"autofunction", "automethod", "autoclass", "autoattribute"}:
                symbol = _normalize_doc_symbol(value, currentmodule)
                if symbol:
                    documented.add(symbol)
                    if directive == "autoclass" and has_members:
                        documented.update(public_by_class.get(symbol, set()))
            elif directive == "automodule":
                module_name = _normalize_doc_symbol(value, currentmodule)
                if module_name:
                    documented.add(module_name)
                    if has_members:
                        for problem_id in public_by_module.get(module_name, set()):
                            if problem_id.rsplit(".", 1)[-1] not in exclude_members:
                                documented.add(problem_id)
                        if module_name == "marshmallow":
                            documented.update(
                                alias for alias in public_aliases if alias.startswith("marshmallow.")
                            )
            elif directive == "autosummary":
                k = i + 1
                while k < j:
                    stripped = lines[k].strip()
                    if stripped and not stripped.startswith(":"):
                        symbol = _normalize_doc_symbol(stripped, currentmodule)
                        if symbol:
                            documented.add(symbol)
                    k += 1
            i = j

    return documented


def _problem_is_documented(problem: Problem, documented_symbols: set[str]) -> bool:
    candidates = {
        problem.problem_id,
        problem.qualified_name,
        *problem.aliases,
    }
    return any(candidate in documented_symbols for candidate in candidates)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pretty_json(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_pretty_problem_json(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keyed_records = {record["task_id"]: record for record in records}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(keyed_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_dataset(
    repo_root: Path,
    output_dir: Path,
    documented_only: bool = False,
    match_mode: str = "trace",
    write_pretty: bool = True,
) -> tuple[Path, Path, int, int]:
    package_root = repo_root / "src" / "marshmallow"
    tests_root = repo_root / "tests"
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

    if match_mode == "trace":
        tests = collect_tests_by_tracing(repo_root, tests_root, problems)
    elif match_mode == "direct":
        tests = collect_tests(repo_root, tests_root, alias_to_problem_ids)
    else:
        raise ValueError(f"Unsupported match mode: {match_mode}")

    matched_problem_ids = {problem_id for test in tests for problem_id in test.problem_ids}
    documented_symbols: set[str] = set()
    if documented_only:
        documented_symbols = collect_documented_symbols(repo_root, problems)
        matched_problem_ids = {
            problem_id
            for problem_id in matched_problem_ids
            if _problem_is_documented(problem_by_id[problem_id], documented_symbols)
        }

    tests_by_problem_id: dict[str, list[TestRecord]] = defaultdict(list)
    for test in tests:
        for problem_id in test.problem_ids:
            if problem_id in matched_problem_ids:
                tests_by_problem_id[problem_id].append(test)

    problem_records = []
    sorted_problem_ids = sorted(matched_problem_ids)
    for index, problem_id in enumerate(_progress(sorted_problem_ids, "Building problem records", "problem")):
        problem = problem_by_id[problem_id]
        matched_tests = tests_by_problem_id[problem_id]
        problem_records.append(
            {
                "task_id": f"MyDataset/{index}",
                "problem_id": problem.problem_id,
                "prompt": problem.prompt,
                "canonical_solution": problem.r + "\n",
                "entry_point": problem.function_name,
                "base_input": [],
                "plus_input": [],
                "atol": 0,
            }
        )

    test_records = []
    for problem_id in _progress(sorted_problem_ids, "Building test records", "problem"):
        for test in tests_by_problem_id[problem_id]:
            test_records.append(
                {
                    "problem_id": problem_id,
                    "test_id": test.test_id,
                    "test_code": test.test_code,
                }
            )

    problems_path = output_dir / "marshmallow_problems.jsonl"
    tests_path = output_dir / "marshmallow_tests.jsonl"
    write_jsonl(problems_path, problem_records)
    write_jsonl(tests_path, test_records)
    if write_pretty:
        write_pretty_problem_json(output_dir / "marshmallow_problems.pretty.json", problem_records)
        write_pretty_json(output_dir / "marshmallow_tests.pretty.json", test_records)
    return problems_path, tests_path, len(problem_records), len(test_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract marshmallow functions/docstrings and linked tests into JSONL files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("d:/code/web/marshmallow"),
        help="Path to the marshmallow repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("d:/code/web/out_marshmallow"),
        help="Directory where JSONL files will be written.",
    )
    parser.add_argument(
        "--documented-only",
        action="store_true",
        help="Keep only problems that appear in marshmallow docs.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("trace", "direct"),
        default="trace",
        help=(
            "How to link tests to functions. 'trace' runs pytest and records executed "
            "marshmallow functions; 'direct' uses conservative static symbol matching."
        ),
    )
    parser.add_argument(
        "--no-pretty-json",
        action="store_true",
        help="Do not write pretty-printed .pretty.json files alongside JSONL output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems_path, tests_path, problem_count, test_count = build_dataset(
        args.repo_root.resolve(),
        args.output_dir.resolve(),
        documented_only=args.documented_only,
        match_mode=args.match_mode,
        write_pretty=not args.no_pretty_json,
    )
    print(f"Wrote {problem_count} problems to {problems_path}")
    print(f"Wrote {test_count} tests to {tests_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
