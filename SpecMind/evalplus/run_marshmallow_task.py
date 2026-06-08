#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

def load_env_file():
    for path in [Path(".env"), Path("../.env"), Path("../../.env")]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'").strip('"')
                        if key:
                            os.environ[key] = value
            break

# Các biến cấu hình mặc định (người dùng dễ dàng chỉnh sửa ở đây)
DEFAULT_TASK_ID = "MyDataset/0"
DEFAULT_MODE = "exploratory"  # Các chế độ hỗ trợ: "single-pass", "greedy", "exploratory"
DEFAULT_MAX_TURNS = 12
DEFAULT_MODEL = "meta-llama/llama-4-scout"
DEFAULT_PROBLEMS_PATH = "../../out_marshmallow_docfilter/marshmallow_problems.jsonl"
DEFAULT_TESTS_PATH = "../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl"
DEFAULT_REPO_ROOT = "../../marshmallow"
DEFAULT_OUTPUT_DIR = ""  # Nếu rỗng sẽ tự động tạo theo format output/marshmallow/{mode}/baseonly_mu{max_turns}
DEFAULT_COMPLETENESS_THRESHOLD = 90.0
DEFAULT_RUN_POWER_EVAL = True  # Bật/Tắt chạy kiểm thử mutant cho cả 3 mode


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Run a single marshmallow evaluation task using OpenRouter.")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID, help="Task ID to evaluate, e.g., 'MyDataset/0'.")
    parser.add_argument("--mode", choices=["single-pass", "greedy", "exploratory"], default=DEFAULT_MODE,
                        help="The experiment mode.")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum number of turns.")
    parser.add_argument("--run-power-eval", action="store_true", default=DEFAULT_RUN_POWER_EVAL,
                        help="Whether to run power evaluation (mutant testing).")
    parser.add_argument("--no-power-eval", dest="run_power_eval", action="store_false",
                        help="Disable power evaluation.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="The LLM model name.")
    parser.add_argument("--problems-path", type=Path, default=Path(DEFAULT_PROBLEMS_PATH),
                        help="Path to marshmallow_problems.jsonl.")
    parser.add_argument("--tests-path", type=Path, default=Path(DEFAULT_TESTS_PATH),
                        help="Path to marshmallow_tests_valid.jsonl.")
    parser.add_argument("--repo-root", type=Path, default=Path(DEFAULT_REPO_ROOT), help="Path to marshmallow repository.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory path.")
    args = parser.parse_args()

    # Configure OpenRouter environment variables
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["SPECMIND_MODEL"] = args.model

    # Use OPENROUTER_API_KEY if defined, fallback to OPENAI_API_KEY
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("Warning: Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set in environment.", file=sys.stderr)

    print(f"OPENAI_BASE_URL={os.environ['OPENAI_BASE_URL']}")
    print(f"SPECMIND_MODEL={os.environ['SPECMIND_MODEL']}")

    # Setup output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"output/marshmallow/{args.mode}/baseonly_mu{args.max_turns}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "generate_and_test_postconditions_general.py",
        "--dataset", "marshmallow",
        "--task-id", args.task_id,
        "--mode", args.mode,
        "--max-turns", str(args.max_turns),
        "--base-only",
        "--problems-path", str(args.problems_path),
        "--tests-path", str(args.tests_path),
        "--marshmallow-repo-root", str(args.repo_root),
        "--output-dir", str(output_dir)
    ]

    if not args.run_power_eval:
        cmd.append("--no-power-eval")
    else:
        if args.mode == "exploratory":
            cmd.extend([
                "--completeness-threshold", str(DEFAULT_COMPLETENESS_THRESHOLD),
                "--feedback-buggy-mutant"
            ])

    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Process failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
