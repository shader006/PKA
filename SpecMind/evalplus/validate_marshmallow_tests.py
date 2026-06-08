import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from generate_and_test_postconditions_general import (
    is_pytest_nodeid_not_found,
    marshmallow_test_id_to_nodeid,
)


# Các biến cấu hình đường dẫn mặc định
DEFAULT_TESTS_PATH = "../../out_marshmallow_docfilter/marshmallow_tests.jsonl"
DEFAULT_MARSHMALLOW_REPO_ROOT = "../../marshmallow"
DEFAULT_FILTERED_OUTPUT = "../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl"
DEFAULT_INVALID_OUTPUT = "output/marshmallow/invalid_tests.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate marshmallow pytest nodeids referenced by marshmallow_tests.jsonl."
    )
    parser.add_argument(
        "--tests-path",
        type=Path,
        default=Path(DEFAULT_TESTS_PATH),
    )
    parser.add_argument(
        "--marshmallow-repo-root",
        type=Path,
        default=Path(DEFAULT_MARSHMALLOW_REPO_ROOT),
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path(DEFAULT_FILTERED_OUTPUT),
        help="Optional JSONL path for valid tests only.",
    )
    parser.add_argument(
        "--invalid-output",
        type=Path,
        default=Path(DEFAULT_INVALID_OUTPUT),
    )
    args = parser.parse_args()

    tests = load_jsonl(args.tests_path)
    repo_root = args.marshmallow_repo_root.resolve()

    # Chạy pytest collect-only một lần duy nhất cho toàn bộ thư mục tests
    print("Collecting all available tests using pytest...")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    
    collected_nodeids = set()
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or "::" not in line:
                continue
            # Bỏ phần tham số nếu có (ví dụ: test_foo[param] -> test_foo)
            if "[" in line:
                line = line.split("[")[0]
            collected_nodeids.add(line)
        print(f"Collected {len(collected_nodeids)} unique test methods/functions.")
    else:
        print("Error: Pytest failed to collect tests.")
        print(proc.stdout)
        sys.exit(1)

    valid_tests: list[dict[str, Any]] = []
    invalid_tests: list[dict[str, Any]] = []

    for index, test in enumerate(tests, start=1):
        nodeid = marshmallow_test_id_to_nodeid(test["test_id"])
        # Chuẩn hóa nodeid (loại bỏ phần tham số nếu có)
        normalized_nodeid = nodeid.split("[")[0] if "[" in nodeid else nodeid
        
        has_invalid_param = False
        if "[" in nodeid:
            param_part = nodeid.split("[", 1)[1]
            if "::" in param_part:
                has_invalid_param = True

        if normalized_nodeid in collected_nodeids and not has_invalid_param:
            valid_tests.append(test)
        else:
            invalid = dict(test)
            invalid["nodeid"] = nodeid
            if has_invalid_param:
                invalid["collect_log"] = f"Nodeid '{nodeid}' contains '::' inside parametrization brackets, which causes pytest collect error."
            else:
                invalid["collect_log"] = f"Nodeid '{nodeid}' (normalized: '{normalized_nodeid}') was not found in collected pytest tests."
            invalid_tests.append(invalid)

    write_jsonl(args.invalid_output, invalid_tests)
    if args.filtered_output is not None:
        write_jsonl(args.filtered_output, valid_tests)

    print(f"Valid tests: {len(valid_tests)}/{len(tests)}")
    print(f"Invalid tests: {len(invalid_tests)}/{len(tests)}")
    print(f"Wrote invalid report: {args.invalid_output}")
    if args.filtered_output is not None:
        print(f"Wrote filtered tests: {args.filtered_output}")


if __name__ == "__main__":
    main()
