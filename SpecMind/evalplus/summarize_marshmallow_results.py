import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--problems-path",
        type=Path,
        default=Path("../../out_marshmallow_docfilter/marshmallow_problems.jsonl"),
    )
    parser.add_argument(
        "--tests-path",
        type=Path,
        default=Path("../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl"),
    )
    args = parser.parse_args()

    # Lập map problem_id -> task_id từ problems.jsonl
    problem_to_task = {}
    if args.problems_path.exists():
        with args.problems_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    problem_to_task[item["problem_id"]] = item["task_id"]

    # Đếm số lượng test cases thực tế cho từng task_id từ tests_valid.jsonl
    tests_count_by_task = {}
    if args.tests_path.exists():
        with args.tests_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    problem_id = item["problem_id"]
                    task_id = problem_to_task.get(problem_id)
                    if task_id:
                        tests_count_by_task[task_id] = tests_count_by_task.get(task_id, 0) + 1

    rows = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0

    for path in sorted(args.results_dir.glob("postcondition_results_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        results = data.get("results")
        base = results.get("base") if results else None
        passed = 0
        total = 0
        if isinstance(base, list) and len(base) > 1 and isinstance(base[1], list):
            total = len(base[1])
            passed = sum(1 for item in base[1] if item)
        
        token_usage = data.get("token_usage", {})
        task_prompt = token_usage.get("prompt_tokens", 0)
        task_completion = token_usage.get("completion_tokens", 0)
        task_total = token_usage.get("total_tokens", 0)

        total_prompt_tokens += task_prompt
        total_completion_tokens += task_completion
        total_total_tokens += task_total

        # Tính số lượt submission (submission turns)
        raw_responses = data.get("raw_responses", [])
        has_solution_tag = any("<solution>" in res for res in raw_responses)
        if has_solution_tag:
            submission_turns = sum(1 for res in raw_responses if "<solution>" in res)
            if submission_turns == 0:
                submission_turns = 1
        else:
            submission_turns = data.get("attempts", 1)

        # Lấy completeness score
        power_eval = data.get("power_evaluation") or {}
        completeness = power_eval.get("completeness_score", 0.0)
        
        sub_turns = max(1, submission_turns)
        task_efficiency = completeness / sub_turns

        task_id = data.get("task_id", "")
        # Lấy số lượng test cases thực tế được định nghĩa cho task này
        task_dataset_total = tests_count_by_task.get(task_id, 0)
        if task_dataset_total == 0:
            task_dataset_total = total

        rows.append(
            {
                "task_id": task_id,
                "success": bool(data.get("success")),
                "attempts": data.get("attempts", 0),
                "submission_turns": sub_turns,
                "completeness": completeness,
                "efficiency": task_efficiency,
                "base_passed": passed,
                "base_total": task_dataset_total,
                "prompt_tokens": task_prompt,
                "completion_tokens": task_completion,
                "total_tokens": task_total,
                "result_file": path.name,
            }
        )

    task_count = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    corr = success_count / task_count if task_count else 0.0
    test_total = sum(row["base_total"] for row in rows)
    test_passed = sum(row["base_passed"] for row in rows)
    test_failed = test_total - test_passed

    avg_completeness = sum(row["completeness"] for row in rows) / task_count if task_count else 0.0
    avg_efficiency = sum(row["efficiency"] for row in rows) / task_count if task_count else 0.0

    summary = {
        "results_dir": str(args.results_dir),
        "task_count": task_count,
        "success_count": success_count,
        "corr": corr,
        "corr_percent": corr * 100,
        "avg_completeness": avg_completeness,
        "avg_completeness_percent": avg_completeness * 100,
        "avg_efficiency": avg_efficiency,
        "avg_efficiency_percent": avg_efficiency * 100,
        "test_passed": test_passed,
        "test_failed": test_failed,
        "test_total": test_total,
        "test_pass_rate": test_passed / test_total if test_total else 0.0,
        "total_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_total_tokens,
        },
        "tasks": rows,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "summary.json"
    csv_path = args.results_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "success",
                "attempts",
                "submission_turns",
                "completeness",
                "efficiency",
                "base_passed",
                "base_total",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "result_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Tasks: {success_count}/{task_count}")
    print(f"Corr: {corr * 100:.2f}%")
    print(f"Avg Completeness: {avg_completeness * 100:.2f}%")
    print(f"Avg Efficiency (E): {avg_efficiency * 100:.2f}%")
    print(f"Tests: Passed: {test_passed} | Failed: {test_failed} | Total: {test_total}")
    print(f"Total tokens used: Prompt: {total_prompt_tokens} | Completion: {total_completion_tokens} | Total: {total_total_tokens}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
