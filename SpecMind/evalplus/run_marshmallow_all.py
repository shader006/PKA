#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
class Tee:
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, "a", encoding="utf-8")
        self.original_stream = original_stream

    def write(self, message):
        try:
            self.original_stream.write(message)
        except UnicodeEncodeError:
            # Fallback: replace characters that cannot be encoded by the terminal stream
            encoding = getattr(self.original_stream, "encoding", "utf-8") or "utf-8"
            safe_message = message.encode(encoding, errors="replace").decode(encoding)
            self.original_stream.write(safe_message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

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
DEFAULT_MODE = "exploratory"  # Các chế độ hỗ trợ: "single-pass", "greedy", "exploratory"
DEFAULT_MAX_TURNS = 12
DEFAULT_MODEL = "meta-llama/llama-4-scout"
DEFAULT_PROBLEMS_PATH = "../../out_marshmallow_docfilter/marshmallow_problems.jsonl"
DEFAULT_TESTS_PATH = "../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl"
DEFAULT_REPO_ROOT = "../../marshmallow"
DEFAULT_OUTPUT_DIR = ""  # Nếu rỗng sẽ tự động tạo theo format output/marshmallow/{mode}/baseonly_mu{max_turns}
DEFAULT_RESUME = False
DEFAULT_RETRIES = 2
DEFAULT_RETRY_DELAY = 30
DEFAULT_PYTEST_BATCH_SIZE = 100 #reconmand 50-100
DEFAULT_COMPLETENESS_THRESHOLD = 90.0
DEFAULT_RUN_POWER_EVAL = False  # Bật/Tắt chạy kiểm thử mutant


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Run all marshmallow evaluation tasks using OpenRouter.")
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
    parser.add_argument("--resume", action="store_true", default=DEFAULT_RESUME, help="Resume from previous runs by skipping existing output files.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Number of retries on failure.")
    parser.add_argument("--retry-delay", type=int, default=DEFAULT_RETRY_DELAY, help="Delay in seconds between retries.")
    parser.add_argument("--pytest-batch-size", type=int, default=DEFAULT_PYTEST_BATCH_SIZE, help="Pytest batch size.")
    args = parser.parse_args()

    # Configure OpenRouter environment variables
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["SPECMIND_MODEL"] = args.model
    os.environ["SPECMIND_MARSHMALLOW_PYTEST_BATCH_SIZE"] = str(args.pytest_batch_size)

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
        output_dir = f"output/marshmallow/{args.mode}/baseonly_mu12"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Enable terminal logging to output/run.log
    log_file_path = output_path / "run.log"
    sys.stdout = Tee(log_file_path, sys.stdout)
    sys.stderr = Tee(log_file_path, sys.stderr)
    print(f"Logging terminal output to {log_file_path}")

    # Load tasks
    if not args.problems_path.exists():
        print(f"Error: Problems path {args.problems_path} does not exist.", file=sys.stderr)
        return 1

    tasks = []
    with args.problems_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    tasks.append(json.loads(line)["task_id"])
                except Exception as e:
                    print(f"Failed to parse line as JSON: {e}", file=sys.stderr)

    total = len(tasks)
    print(f"Loaded {total} tasks from {args.problems_path}")

    # Run each task
    for index, task_id in enumerate(tasks, 1):
        # Format task filename replacing / with _
        result_file = output_path / f"postcondition_results_{task_id.replace('/', '_')}.json"
        
        if args.resume and result_file.exists():
            print(f"[{index}/{total}] Skipping {task_id} ({args.mode}); result already exists")
            continue

        print(f"[{index}/{total}] Running {task_id} ({args.mode})")

        cmd = [
            sys.executable,
            "generate_and_test_postconditions_general.py",
            "--dataset", "marshmallow",
            "--task-id", task_id,
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

        attempt = 0
        success = False
        while attempt <= args.retries:
            attempt += 1
            if attempt > 1:
                print(f"Retrying {task_id} attempt {attempt}/{args.retries + 1} after {args.retry_delay} seconds")
                time.sleep(args.retry_delay)
            
            try:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1) as proc:
                    if proc.stdout:
                        for line in proc.stdout:
                            sys.stdout.write(line)
                    proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
                success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"Process failed on attempt {attempt}: {e}", file=sys.stderr)
        
        if not success:
            print(f"Warning: Task {task_id} failed after {attempt} attempts.", file=sys.stderr)

    # Summarize results
    print("Summarizing results...")
    try:
        sum_cmd = [
            sys.executable,
            "summarize_marshmallow_results.py",
            "--results-dir", str(output_dir)
        ]
        with subprocess.Popen(sum_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1) as proc:
            if proc.stdout:
                for line in proc.stdout:
                    sys.stdout.write(line)
            proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, sum_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run summarize_marshmallow_results.py: {e}", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())
