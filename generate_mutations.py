#!/usr/bin/env python3
"""Generate mutation variants from marshmallow dataset for mutation testing."""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, desc, unit="it"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)


@dataclass
class Mutation:
    mutation_id: str
    problem_id: str
    original_code: str
    mutated_code: str
    mutation_type: str
    mutation_desc: str


class MutationOperator(ast.NodeTransformer):
    """Base class for mutation operators."""

    def __init__(self):
        self.mutations_found = []

    def generic_visit(self, node):
        return super().generic_visit(node)


class ReturnValueMutation(MutationOperator):
    """Mutate return values: None -> value, value -> None, True -> False, etc."""

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.value is None:
            return node

        if isinstance(node.value, ast.Constant):
            original = node.value.value
            if original is True:
                self.mutations_found.append(("bool_true_to_false", ast.Constant(value=False)))
            elif original is False:
                self.mutations_found.append(("bool_false_to_true", ast.Constant(value=True)))
            elif original is None:
                self.mutations_found.append(("none_to_zero", ast.Constant(value=0)))
            elif isinstance(original, (int, float)):
                self.mutations_found.append(("num_to_zero", ast.Constant(value=0)))
                self.mutations_found.append(("num_negate", ast.UnaryOp(op=ast.USub(), operand=node.value)))
            elif isinstance(original, str):
                self.mutations_found.append(("str_to_empty", ast.Constant(value="")))
        elif isinstance(node.value, ast.Name) and node.value.id == "None":
            self.mutations_found.append(("return_none_to_value", ast.Constant(value=0)))

        return node


class ComparisonMutation(MutationOperator):
    """Mutate comparison operators."""

    COMPARISON_MAP = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.GtE,
        ast.GtE: ast.Lt,
        ast.LtE: ast.Gt,
        ast.Gt: ast.LtE,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        for i, op in enumerate(node.ops):
            if type(op) in self.COMPARISON_MAP:
                new_op = self.COMPARISON_MAP[type(op)]()
                self.mutations_found.append(
                    (f"comp_{type(op).__name__}_to_{type(new_op).__name__}", new_op)
                )
        return node


class ArithmeticMutation(MutationOperator):
    """Mutate arithmetic operators."""

    ARITH_MAP = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
        ast.FloorDiv: ast.Mult,
        ast.Mod: ast.Add,
    }

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        if type(node.op) in self.ARITH_MAP:
            new_op = self.ARITH_MAP[type(node.op)]()
            self.mutations_found.append(
                (f"arith_{type(node.op).__name__}_to_{type(new_op).__name__}", new_op)
            )
        return node


class BooleanMutation(MutationOperator):
    """Mutate boolean operators (and/or -> or/and)."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        if isinstance(node.op, ast.And):
            self.mutations_found.append(("bool_and_to_or", ast.Or()))
        elif isinstance(node.op, ast.Or):
            self.mutations_found.append(("bool_or_to_and", ast.And()))
        return node


class NoneCheckMutation(MutationOperator):
    """Mutate 'is None' / 'is not None' checks."""

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            if isinstance(op, ast.Is):
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    self.mutations_found.append(("is_none_to_is_not_none", ast.IsNot()))
            elif isinstance(op, ast.IsNot):
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    self.mutations_found.append(("is_not_none_to_is_none", ast.Is()))
        return node


class ExceptionMutation(MutationOperator):
    """Mutate exception types in raise/except."""

    EXCEPTION_MAP = {
        "ValueError": ["TypeError", "KeyError", "IndexError"],
        "TypeError": ["ValueError", "KeyError"],
        "KeyError": ["ValueError", "TypeError"],
        "AttributeError": ["TypeError", "ValueError"],
        "IndexError": ["KeyError", "ValueError"],
    }

    def visit_Raise(self, node: ast.Raise) -> ast.Raise:
        if node.exc and isinstance(node.exc, ast.Call):
            if isinstance(node.exc.func, ast.Name):
                exc_name = node.exc.func.id
                if exc_name in self.EXCEPTION_MAP:
                    for new_exc in self.EXCEPTION_MAP[exc_name]:
                        new_node = ast.Raise(
                            exc=ast.Call(
                                func=ast.Name(id=new_exc, ctx=ast.Load()),
                                args=node.exc.args[:],
                                keywords=[],
                            )
                        )
                        self.mutations_found.append(
                            (f"exc_{exc_name}_to_{new_exc}", new_node)
                        )
        return node


class ConditionalMutation(MutationOperator):
    """Mutate if conditions: negate conditions."""

    def visit_If(self, node: ast.If) -> ast.If:
        if isinstance(node.test, ast.NameConstant):
            if node.test.value is True:
                self.mutations_found.append(("if_true_to_false", ast.NameConstant(value=False)))
            elif node.test.value is False:
                self.mutations_found.append(("if_false_to_true", ast.NameConstant(value=True)))
        elif isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
            self.mutations_found.append(("negate_if_condition", node.test.operand))
        elif isinstance(node.test, ast.Compare):
            negated = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.mutations_found.append(("negate_if_condition", negated))
        return node


ALL_MUTATORS = [
    ReturnValueMutation,
    ComparisonMutation,
    ArithmeticMutation,
    BooleanMutation,
    NoneCheckMutation,
    ExceptionMutation,
    ConditionalMutation,
]


def apply_mutations(code: str) -> list[Mutation]:
    """Apply all mutation operators to code and return list of mutations."""
    mutations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return mutations

    for mutator_cls in ALL_MUTATORS:
        mutator = mutator_cls()
        mutator.visit(ast.parse(code))
        for desc, new_node in mutator.mutations_found:
            try:
                mutated = _apply_single_mutation(tree, new_node, desc)
                if mutated and mutated != code.strip():
                    mutations.append(Mutation(
                        mutation_id="",
                        problem_id="",
                        original_code=code.strip(),
                        mutated_code=mutated,
                        mutation_type=desc.split("_")[0],
                        mutation_desc=desc,
                    ))
            except Exception:
                continue

    return mutations


def _apply_single_mutation(original_tree: ast.AST, replacement, desc: str) -> str | None:
    """Try to apply a single mutation and return mutated code string."""
    try:
        source = ast.unparse(original_tree)
    except Exception:
        return None
    return source


def mutate_return_values(code: str) -> list[tuple[str, str]]:
    """Simple text-based return value mutations."""
    mutations = []
    lines = code.strip().split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()

        # return True -> return False
        if stripped == "return True":
            new_lines = lines[:i] + [line.replace("return True", "return False")] + lines[i+1:]
            mutations.append(("return_true_to_false", "\n".join(new_lines)))

        # return False -> return True
        if stripped == "return False":
            new_lines = lines[:i] + [line.replace("return False", "return True")] + lines[i+1:]
            mutations.append(("return_false_to_true", "\n".join(new_lines)))

        # return None -> return 0
        if stripped == "return None":
            new_lines = lines[:i] + [line.replace("return None", "return 0")] + lines[i+1:]
            mutations.append(("return_none_to_0", "\n".join(new_lines)))

        # return [] -> return None
        if stripped == "return []":
            new_lines = lines[:i] + [line.replace("return []", "return None")] + lines[i+1:]
            mutations.append(("return_empty_list_to_none", "\n".join(new_lines)))

        # return {} -> return None
        if stripped == "return {}":
            new_lines = lines[:i] + [line.replace("return {}", "return None")] + lines[i+1:]
            mutations.append(("return_empty_dict_to_none", "\n".join(new_lines)))

        # return value -> return None (for non-None returns)
        if stripped.startswith("return ") and not stripped.startswith("return None"):
            if "=" not in stripped.split("return ")[1][:5]:
                new_lines = lines[:i] + [line.replace(stripped, "return None")] + lines[i+1:]
                mutations.append(("return_value_to_none", "\n".join(new_lines)))

        # Mutation comparison operators
        for old, new in [("==", "!="), ("!=", "=="), ("<", ">="), (">", "<="), ("<=", ">"), (">=", "<")]:
            if old in stripped and "return" in stripped:
                new_lines = lines[:i] + [line.replace(old, new, 1)] + lines[i+1:]
                mutations.append((f"comp_{old}_to_{new}", "\n".join(new_lines)))

        # Mutation arithmetic operators
        for old, new in [("+", "-"), ("-", "+"), ("*", "/"), ("/", "*")]:
            if f" {old} " in stripped:
                new_lines = lines[:i] + [line.replace(f" {old} ", f" {new} ", 1)] + lines[i+1:]
                mutations.append((f"arith_{old}_to_{new}", "\n".join(new_lines)))

        # Mutation and/or
        if " and " in stripped:
            new_lines = lines[:i] + [line.replace(" and ", " or ")] + lines[i+1:]
            mutations.append(("and_to_or", "\n".join(new_lines)))
        if " or " in stripped:
            new_lines = lines[:i] + [line.replace(" or ", " and ")] + lines[i+1:]
            mutations.append(("or_to_and", "\n".join(new_lines)))

        # Mutation is/is not None
        if "is None" in stripped:
            new_lines = lines[:i] + [line.replace("is None", "is not None")] + lines[i+1:]
            mutations.append(("is_none_to_is_not_none", "\n".join(new_lines)))
        if "is not None" in stripped:
            new_lines = lines[:i] + [line.replace("is not None", "is None")] + lines[i+1:]
            mutations.append(("is_not_none_to_is_none", "\n".join(new_lines)))

        # Mutation if conditions
        if stripped.startswith("if ") and ":" in stripped:
            condition = stripped[3:].rstrip(":").strip()
            if not condition.startswith("not "):
                new_lines = lines[:i] + [line.replace(f"if {condition}", f"if not ({condition})")] + lines[i+1:]
                mutations.append(("negate_if", "\n".join(new_lines)))

    return mutations


def generate_mutations(
    problems_path: Path,
    output_dir: Path,
    max_mutations_per_problem: int = 5,
) -> tuple[Path, int]:
    """Generate mutation dataset from problems."""
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = []
    with problems_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    all_mutations = []
    problem_mutations = {}

    for problem in _progress(problems, "Generating mutations", "problem"):
        code = problem["canonical_solution"]
        problem_id = problem["problem_id"]

        mutations = mutate_return_values(code)

        if not mutations:
            continue

        # Limit mutations per problem
        if len(mutations) > max_mutations_per_problem:
            mutations = random.sample(mutations, max_mutations_per_problem)

        problem_mutations[problem_id] = []
        for mut_type, mutated_code in mutations:
            mutation = {
                "mutation_id": f"{problem_id}::{mut_type}",
                "problem_id": problem_id,
                "original_code": code.strip(),
                "mutated_code": mutated_code.strip(),
                "mutation_type": mut_type.split("_")[0],
                "mutation_desc": mut_type,
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "entry_point": problem["entry_point"],
            }
            all_mutations.append(mutation)
            problem_mutations[problem_id].append(mutation)

    # Write mutations JSONL
    mutations_path = output_dir / "marshmallow_mutations.jsonl"
    with mutations_path.open("w", encoding="utf-8", newline="\n") as f:
        for mutation in all_mutations:
            f.write(json.dumps(mutation, ensure_ascii=False) + "\n")

    # Write mutations JSON (grouped by problem)
    grouped_path = output_dir / "marshmallow_mutations_by_problem.json"
    with grouped_path.open("w", encoding="utf-8") as f:
        json.dump(problem_mutations, f, ensure_ascii=False, indent=2)

    # Write pretty version
    pretty_path = output_dir / "marshmallow_mutations.pretty.json"
    with pretty_path.open("w", encoding="utf-8") as f:
        json.dump(all_mutations, f, ensure_ascii=False, indent=2)

    return mutations_path, len(all_mutations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mutation variants from marshmallow dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("d:/code/web/out_marshmallow/marshmallow_problems.jsonl"),
        help="Input problems JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("d:/code/web/out_marshmallow"),
        help="Output directory for mutation files.",
    )
    parser.add_argument(
        "--max-per-problem",
        type=int,
        default=5,
        help="Maximum mutations per problem.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    mutations_path, mutation_count = generate_mutations(
        args.input.resolve(),
        args.output_dir.resolve(),
        max_mutations_per_problem=args.max_per_problem,
    )

    print(f"Generated {mutation_count} mutations")
    print(f"Written to {mutations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
