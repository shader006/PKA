import argparse
import ast
import json
import os
import random
import re
import base64
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import openai
from tqdm import tqdm

# --- Configuration ---
# You can replace with your actual API key.
# OPENAI_API_KEY = "your-api-key"
MODEL_NAME = os.getenv("SPECMIND_MODEL", "meta-llama/llama-4-scout")
MAX_ATTEMPTS = 5
TEMPERATURE = 0
SUCCESS = "success"
MARSHMALLOW_PYTEST_BATCH_SIZE = int(os.getenv("SPECMIND_MARSHMALLOW_PYTEST_BATCH_SIZE", "50"))
MARSHMALLOW_PYTEST_MAX_COMMAND_CHARS = int(
    os.getenv("SPECMIND_MARSHMALLOW_PYTEST_MAX_COMMAND_CHARS", "24000")
)
MARSHMALLOW_PYTEST_TIMEOUT = int(os.getenv("SPECMIND_MARSHMALLOW_PYTEST_TIMEOUT", "300"))

# Client initialization
client = None


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


def get_openai_client():
    global client
    if client is None:
        load_env_file()
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url and os.getenv("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
        if base_url:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = openai.OpenAI(api_key=api_key)
    return client


MARSHMALLOW_PYTEST_PLUGIN = r'''
import base64
import functools
import importlib
import inspect
import os
import re


def _resolve_problem(problem_id):
    parts = problem_id.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        chain = parts[split_at:]
        if not chain:
            continue
        owner = module
        try:
            for name in chain[:-1]:
                owner = getattr(owner, name)
            return owner, chain[-1]
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot resolve problem_id: {problem_id}")


def pytest_sessionstart(session):
    problem_id = os.environ["SPECMIND_MARSHMALLOW_PROBLEM_ID"]
    postcondition = base64.b64decode(
        os.environ["SPECMIND_MARSHMALLOW_POSTCONDITION_B64"]
    ).decode("utf-8")
    owner, attr_name = _resolve_problem(problem_id)

    def run_postcondition(local_vars):
        exec_vars = {"re": re, "__builtins__": __builtins__}
        exec_vars.update(local_vars)
        # Use one namespace so comprehensions inside exec can see bound args.
        exec(postcondition, exec_vars, exec_vars)

    raw_descriptor = inspect.getattr_static(owner, attr_name)

    if isinstance(raw_descriptor, classmethod):
        original = raw_descriptor.__func__

        @functools.wraps(original)
        def wrapped(cls, *args, **kwargs):
            sig = inspect.signature(original)
            bound_args = sig.bind(cls, *args, **kwargs)
            bound_args.apply_defaults()
            local_vars = dict(bound_args.arguments)
            return_value = original(cls, *args, **kwargs)
            local_vars["return_value"] = return_value
            run_postcondition(local_vars)
            return return_value

        setattr(owner, attr_name, classmethod(wrapped))
        return

    if isinstance(raw_descriptor, staticmethod):
        original = raw_descriptor.__func__

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            sig = inspect.signature(original)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            local_vars = dict(bound_args.arguments)
            return_value = original(*args, **kwargs)
            local_vars["return_value"] = return_value
            run_postcondition(local_vars)
            return return_value

        setattr(owner, attr_name, staticmethod(wrapped))
        return

    original = getattr(owner, attr_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        sig = inspect.signature(original)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        local_vars = dict(bound_args.arguments)
        return_value = original(*args, **kwargs)
        local_vars["return_value"] = return_value
        run_postcondition(local_vars)
        return return_value

    setattr(owner, attr_name, wrapped)
'''


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_marshmallow_dataset(problems_path: Path, tests_path: Path) -> dict[str, dict[str, Any]]:
    problems = {record["task_id"]: record for record in load_jsonl(problems_path)}
    tests_by_problem_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for test in load_jsonl(tests_path):
        tests_by_problem_id[test["problem_id"]].append(test)
    for problem in problems.values():
        problem["dataset"] = "marshmallow"
        problem["marshmallow_tests"] = tests_by_problem_id.get(problem["problem_id"], [])
    return problems


def marshmallow_test_id_to_nodeid(test_id: str) -> str:
    module_name, test_name = test_id.split("::", 1)
    path = module_name.replace(".", "/") + ".py"
    return path + "::" + "::".join(test_name.split("."))


def is_pytest_nodeid_not_found(log: str) -> bool:
    return "ERROR: not found:" in log or "no match in any of" in log


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def pytest_nodeid_batches(nodeids: list[str], size: int, max_command_chars: int) -> list[list[str]]:
    batches = []
    current = []
    current_chars = 0
    max_items = max(1, size)

    for nodeid in nodeids:
        nodeid_chars = len(nodeid) + 3
        would_exceed_size = len(current) >= max_items
        would_exceed_chars = current and current_chars + nodeid_chars > max_command_chars
        if would_exceed_size or would_exceed_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(nodeid)
        current_chars += nodeid_chars

    if current:
        batches.append(current)
    return batches


def evaluate_marshmallow_postcondition(
    problem: Dict[str, Any], postcondition: str, repo_root: Path
) -> Tuple[bool, Dict[str, Any]]:
    tests = problem.get("marshmallow_tests", [])
    if not tests:
        return False, {"base": ("failed", [], None, ["No linked pytest tests found."]), "plus": None}

    with tempfile.TemporaryDirectory(prefix="specmind_marshmallow_") as temp_dir:
        plugin_path = Path(temp_dir) / "specmind_marshmallow_plugin.py"
        plugin_path.write_text(MARSHMALLOW_PYTEST_PLUGIN, encoding="utf-8")

        env = os.environ.copy()
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            [temp_dir, src_path, env.get("PYTHONPATH", "")]
        )
        env["SPECMIND_MARSHMALLOW_PROBLEM_ID"] = problem["problem_id"]
        env["SPECMIND_MARSHMALLOW_POSTCONDITION_B64"] = base64.b64encode(
            postcondition.encode("utf-8")
        ).decode("ascii")

        nodeids = [marshmallow_test_id_to_nodeid(test["test_id"]) for test in tests]
        test_results = []
        logs = []
        skipped_nodeids = []

        batches = pytest_nodeid_batches(
            nodeids,
            MARSHMALLOW_PYTEST_BATCH_SIZE,
            MARSHMALLOW_PYTEST_MAX_COMMAND_CHARS,
        )
        total_batches = len(batches)
        completed_nodeids = 0
        for batch_index, batch_nodeids in enumerate(batches, start=1):
            start_index = completed_nodeids + 1
            end_index = start_index + len(batch_nodeids) - 1
            print(
                f"Running marshmallow pytest batch {batch_index}/{total_batches} "
                f"({start_index}-{end_index}/{len(nodeids)})",
                flush=True,
            )
            completed_nodeids = end_index
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "specmind_marshmallow_plugin",
                *batch_nodeids,
            ]
            try:
                proc = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=MARSHMALLOW_PYTEST_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                log = (
                    f"TIMEOUT after {MARSHMALLOW_PYTEST_TIMEOUT}s while running "
                    f"marshmallow pytest batch {batch_index}/{total_batches}\n"
                    f"{exc.stdout or ''}"
                )
                test_results.extend([False] * len(batch_nodeids))
                logs.append(log)
                continue
            log = proc.stdout
            if is_pytest_nodeid_not_found(log):
                skipped_nodeids.extend(batch_nodeids)
                logs.append(
                    "SKIPPED invalid pytest nodeid batch: "
                    + ", ".join(batch_nodeids)
                    + "\n"
                    + log
                )
                continue
            test_results.extend([proc.returncode == 0] * len(batch_nodeids))
            logs.append(log)

    if skipped_nodeids:
        print(
            f"Skipped {len(skipped_nodeids)} invalid marshmallow pytest nodeid(s) "
            f"for {problem['task_id']}."
        )

    if not test_results:
        log = "No valid linked pytest tests found.\n" + "\n".join(logs)
        return False, {"base": ("failed", [], None, [log]), "plus": None}

    passed = all(test_results)
    result = (SUCCESS if passed else "failed", test_results, None, logs)
    return passed, {"base": result, "plus": None}


# --- Prompt Definition ---
PROMPTS = { 
    "exploratory" : (
"""
### Objective

You are an AI assistant tasked with verifying the correctness of a Python function based solely on its **docstring** and **implementation**.

Your goal is to write **symbolic postconditions** - Python `assert` statements that validate specific behavioral properties of the function's return value, assuming the function has been implemented correctly.

These symbolic postconditions must not reimplement the function, but instead express **concise, meaningful, and checkable properties** of the output.

---

### Exploration Process

You are allowed to iteratively reason and refine symbolic postconditions using the following tools:

#### 🔄 Turn Types

-   `<think>`: Reflect on the function's specification and infer its intended behavior.

<think>
…reasoning about the function's purpose, structure, expected output constraints, edge cases, etc…
</think>
    
-   `<assert>`: Propose one symbolic postconditions in Python. Must be a valid `assert` statement, preceded by a brief comment.
    
<assert>
# Checks that no output element exceeds input elements
assert all(x <= max(data) for x in return_value)
</assert>
    
-   `<observation>`: Receive feedback from the system about your assertions.
    
<observation>
Assertions are valid.
<reminder>You has {max_turns} turns remaining.</reminder>
</observation>
    
-   `<solution>`: When confident (or you only have 0 turn remaining) you must submit a solution, provide your finalized symbolic postcondition.\
Your final `<solution>` should ideally be submitted only when you ensure that it is the most refined and reliable postcondition to be deployed for bug detection in production.
    
<solution>...</solution>

---

### Interaction Limit

You have a maximum of {max_turns} total turns remaining (any mix of <think>, <assert>, or <observation>).
    - You may submit a `<solution>` at any time when you believe you have a strong postcondition. Multiple submissions are allowed, and each will be treated as a potential candidate for the final solution.
    - If you submit one or more `<solution>` blocks but are still required to continue, this indicates that your current postcondition is not yet fully correct or complete. You must then continue exploring and refining through additional reasoning and `<assert>` checks before submitting another `<solution>`.
    - Use the early rounds to carefully reason about the function and to issue `<assert>` checks that validate your understanding of its behavior.
    - In later rounds, refine your postconditions based on your reasoning and observations.
    - However, avoid submitting <solution> blocks too frequently. Use most of your turns for exploration (<think>, <assert>, <observation>) and only propose a new <solution> after substantial refinement or new insights.\
In particular, do not repeatedly submit <solution> blocks in the final turns. If turns remain after a submission, you are expected to keep exploring and strengthening your reasoning until the very end.
    - Before submitting any <solution>, you must test it internally to ensure: It has no syntax errors; Executing it will not raise an `AssertionError`; It faithfully reflects your reasoning so far. Only then should you submit it as a valid candidate.
    - If no `<solution>` has been submitted by the final turn (0 turns left), you must submit one at that point.
---

### Postcondition Rules

Your task is to write a symbolic postcondition for {entrypoint}. The postcondition should be in Python, and consist of exactly one assert statement. A Python comment explaining the postcondition's meaning should precede it. For variables, the postcondition should only use the input parameters defined in the function stub and a hypothetical return value of the function, which we'll assume is stored in a variable `return_value`.

For string manipulation, Python's `re` (regular expressions) library can be used. If other Python standard library functions are required, include the necessary imports. However, refrain from using external libraries or calling the function itself (in this case, {entrypoint}) within the postcondition.

If the postcondition calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure (i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

---

### Your Task

You will now be given a Python function `{entrypoint}`:

```python
{codeStubAndDocstring}{canonical_solution}
```

Begin by analyzing it with `<think>`, then proceed to propose assertions using `<assert>`, review feedback with `<observation>`, and finalize using `<solution>` — **all within a maximum of {max_turns} total turns.**

Let's begin.
"""
    ),
    "greedy" : (
"""
### Objective

You are an AI assistant tasked with verifying the correctness of a Python function based solely on its **docstring** and **implementation**.

Your goal is to write **symbolic postconditions** - Python `assert` statements that validate specific behavioral properties of the function's return value, assuming the function has been implemented correctly.

These symbolic postconditions must not reimplement the function, but instead express **concise, meaningful, and checkable properties** of the output.

---

### The format of your response should be:

-   `<think>`: Reflect on the function's specification and infer its intended behavior.

<think>
…reasoning about the function's purpose, structure, expected output constraints, edge cases, etc…
</think>
    
-   `<assert>`: Propose one symbolic postconditions in Python. Must be a valid `assert` statement, preceded by a brief comment.
    
<assert>
# Checks that no output element exceeds input elements
assert all(x <= max(data) for x in return_value)
</assert>
---

### Postcondition Rules

Your task is to write a symbolic postcondition for {entrypoint}. The postcondition should be in Python, and consist of exactly one assert statement. A Python comment explaining the postcondition's meaning should precede it. For variables, the postcondition should only use the input parameters defined in the function stub and a hypothetical return value of the function, which we'll assume is stored in a variable `return_value`.

For string manipulation, Python's `re` (regular expressions) library can be used. If other Python standard library functions are required, include the necessary imports. However, refrain from using external libraries or calling the function itself (in this case, {entrypoint}) within the postcondition.

If the postcondition calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure (i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

---

### Your Task

You will now be given a Python function `{entrypoint}`:

```python
{codeStubAndDocstring}{canonical_solution}
```

Begin by analyzing it with `<think>`, then proceed to propose assertions using `<assert>`.

Let's begin.
"""
    )
}
PROMPTS["single-pass"] = PROMPTS["greedy"]


def generate_postcondition(
    problem: Dict[str, Any],
    max_turns: int = 12,
    conversation_history: List[Dict[str, str]] = [],
    stop: Optional[List[str]] = None,
    mode: Literal["single-pass", "greedy", "exploratory"] = "exploratory"
) -> Tuple[str, Dict[str, int]]:
    """
    Generate a postcondition for the given problem using a language model.

    Args:
        problem: The problem dictionary containing task details.
        max_turns: The maximum number of turns allowed in the conversation.
        conversation_history: List of formatted conversation blocks to append to the prompt.
        stop: A list of strings to stop generation at.

    Returns:
        A tuple of (raw response string, token usage dict).
    """
    prompt = PROMPTS[mode].format(
        entrypoint=problem["entry_point"],
        codeStubAndDocstring=problem["prompt"],
        canonical_solution=problem["canonical_solution"],
        max_turns=max_turns,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a programming assistant that generates executable python only. You generate correct code, so you only generate code you are sure of. You have Python comments explaining your intent when possible.",
        },
        {"role": "user", "content": prompt},
    ]

    # Append any prior conversation history to the messages
    if conversation_history:
        messages.extend(conversation_history)

    response = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=1024,
        stop=stop,
    )
    if not response or not hasattr(response, "choices") or response.choices is None or len(response.choices) == 0:
        print(f"Warning: API returned no choices. Full response: {response}", file=sys.stderr)
        raw_content = ""
    else:
        raw_content = response.choices[0].message.content
        if raw_content is None:
            print(f"Warning: API returned None content. Full response: {response}", file=sys.stderr)
            raw_content = ""
        else:
            raw_content = raw_content.strip()
    
    usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(response, "usage") and response.usage:
        usage_dict["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0)
        usage_dict["completion_tokens"] = getattr(response.usage, "completion_tokens", 0)
        usage_dict["total_tokens"] = getattr(response.usage, "total_tokens", 0)
        
    return raw_content, usage_dict


def wrap_with_postcondition(code: str, postcondition: str, entry_point: str) -> str:
    """
    Wraps the original function code with the generated postcondition for testing.

    The wrapper function binds arguments and makes them available to the postcondition
    via local variables.

    Args:
        code: The original function code.
        postcondition: The generated postcondition as a string.
        entry_point: The name of the function to be tested.

    Returns:
        The combined, wrapped code as a string.
    """
    # Use AST to safely find function arguments
    func_def = None
    func_ast = ast.parse(code)
    for node in func_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            func_def = node
            break
    
    if not func_def:
        return code  # Fallback if the function definition can't be found
    
    arg_names = [arg.arg for arg in func_def.args.args]
    
    # Generate code to assign arguments to local variables
    assign_lines = [f"    {name} = bound_args.arguments['{name}']" for name in arg_names]
    assign_block = "\n".join(assign_lines)

    # Indent the postcondition correctly
    indented_postcondition = "\n".join(
        ["    " + line for line in postcondition.split("\n")]
    )

    # Construct the full wrapped code
    wrapped_code = f"""
{code}

def {entry_point}_wrapped(*args, **kwargs):
    import re
    import inspect
    sig = inspect.signature({entry_point})
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
{assign_block}
    return_value = {entry_point}(*args, **kwargs)
{indented_postcondition}
    return return_value
"""
    return wrapped_code.strip()


def evaluate_postcondition(
    problem: Dict[str, Any],
    postcondition: str,
    expected_output: Dict[str, List],
    base_only: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates a generated postcondition by running it against test inputs.

    Args:
        problem: The problem dictionary.
        postcondition: The postcondition string to be evaluated.
        expected_output: The ground truth outputs for base and plus inputs.
        base_only: If True, only evaluate against base inputs.

    Returns:
        A tuple containing:
        - A boolean indicating if the postcondition passed all tests.
        - A dictionary with detailed results for base and plus tests.
    """
    if problem.get("dataset") == "marshmallow":
        repo_root = Path(problem["marshmallow_repo_root"])
        return evaluate_marshmallow_postcondition(problem, postcondition, repo_root)

    from evalplus.eval import SUCCESS, untrusted_check

    # Combine function code and the postcondition wrapper
    wrapped_code = wrap_with_postcondition(
        problem["prompt"] + problem["canonical_solution"],
        postcondition,
        problem["entry_point"],
    )

    print(f"Evaluating postcondition for task {problem['task_id']}")
    print(f"Wrapped code:\n{wrapped_code}\n")

    # Test with base inputs
    base_result = untrusted_check(
        wrapped_code,
        problem["base_input"],
        problem["entry_point"] + "_wrapped",
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=False,
    )

    if base_only:
        return base_result[0] == SUCCESS, {"base": base_result, "plus": None}

    # Test with plus inputs
    plus_result = untrusted_check(
        wrapped_code,
        problem["plus_input"],
        problem["entry_point"] + "_wrapped",
        expected=expected_output["plus"],
        atol=problem["atol"],
        ref_time=expected_output["plus_time"],
        fast_check=False,
    )

    return plus_result[0] == SUCCESS, {"base": base_result, "plus": plus_result}


def double_check_postcondition(postcondition: str, entrypoint: str) -> str:
    """
    Extracts the postcondition logic from a potentially larger block of text.

    This function handles cases where the LLM might have included boilerplate code,
    like the `return_value` assignment. It ensures that only the final assertion
    and any necessary imports are returned.

    Args:
        postcondition: The raw postcondition text from the LLM.
        entrypoint: The name of the function being tested.

    Returns:
        A cleaned-up postcondition string.
    """
    lines = postcondition.strip().splitlines()
    result_lines = []
    assignment_index = None

    # Find the line where `return_value` is assigned
    for i, line in enumerate(lines):
        if re.match(rf"\s*return_value\s*=\s*{re.escape(entrypoint)}\s*\(.*\)", line):
            assignment_index = i
            break

    if assignment_index is None:
        return postcondition.strip()

    # Keep necessary imports from above the assignment line
    for line in lines[:assignment_index]:
        if re.match(r"\s*(import|from)\s+\S+", line):
            result_lines.append(line)

    # Append everything after the assignment line
    result_lines.extend(lines[assignment_index + 1 :])

    return "\n".join(result_lines).strip()


def parse_model_response(response: str):
    """
    Parses a model's response to extract content from specific tags.

    Args:
        response: The raw string response from the language model.

    Returns:
        A tuple of (think, assertions, solution) strings.
    """
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    assert_match = re.search(r"<assert>(.*?)</assert>", response, re.DOTALL)
    solution_match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else None
    assertion_content = assert_match.group(1).strip() if assert_match else None
    solution_content = solution_match.group(1).strip() if solution_match else None

    # Handle cases where the model stops mid-tag
    if not assert_match and response.strip().endswith("<assert>"):
        incomplete_assert_match = re.search(r"<assert>(.*)", response.strip(), re.DOTALL)
        if incomplete_assert_match:
            assertion_content = incomplete_assert_match.group(1).strip()

    if assertion_content is None and solution_content is None and "assert " in response:
        assertion_content = response.strip()

    return think_content, assertion_content, solution_content


# @traceable
def generate_and_test_postcondition(
    problem: Dict[str, Any],
    expected_output: Dict[str, List],
    max_turns: int = 12,
    base_only: bool = False,
    run_power_eval: bool = True,
    mode: Literal["exploratory", "greedy", "single-pass"] = "exploratory",
    completeness_threshold: Optional[float] = None,
    feedback_buggy_mutant: bool = False,
) -> Tuple[
    Optional[str],
    Optional[Dict[str, Any]],
    bool,
    List[Dict[str, str]],
    List[str],
    List[Tuple[float, int]],
    Dict[str, int],
]:
    """
    Generates and iteratively tests a postcondition using a conversational approach.

    Args:
        problem: The problem dictionary.
        expected_output: Ground truth outputs.
        max_turns: Maximum conversation turns.
        base_only: If True, only use base inputs for testing.
        run_power_eval: If True, calculate the completeness score.
        completeness_threshold: A float threshold to guide the LLM's refinement.

    Returns:
        A tuple containing the best postcondition, its evaluation result,
        success flag, full conversation history, raw model responses, completeness trend, and token usage.
    """
    conversation_history: List[Dict[str, str]] = []
    raw_responses: List[str] = []
    completeness_trend = []
    is_best_postcondition_correct = False
    best_postcondition = None
    best_result = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    
    for turn in range(1, max_turns + 1):
        print(f"Turn {turn}/{max_turns} for problem {problem['task_id']}")
        remaining_turns = max_turns - turn

        # Craft the reminder message for the LLM
        if mode == "exploratory":
            reminder = f"<reminder>You have {remaining_turns} turns remaining."
            if remaining_turns == 1:
                reminder += " This is your final turn. You must provide a solution now in the `<solution>` block."
            reminder += "</reminder>"
        else:
            reminder = ""
        

        # try:
        # Generate the next response. In singlepass/random-sampling mode, each
        # sample must be independent, so do not feed prior observations back.
        model_history = [] if mode == "single-pass" else conversation_history
        raw_response, usage = generate_postcondition(
            problem, max_turns, model_history, stop=["</assert>", "</solution>"], mode=mode
        )
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        total_total_tokens += usage.get("total_tokens", 0)
        print(raw_responses)
        raw_responses.append(raw_response)

        # Ensure tags are closed for history
        if "<assert>\n" in raw_response and not raw_response.strip().endswith("</assert>"):
            raw_response += "\n</assert>"
        
        if "<solution>\n" in raw_response and not raw_response.strip().endswith("</solution>"):
            raw_response += "\n</solution>"

        think_content, assertion_content, solution_content = parse_model_response(raw_response)
        
        # Add the model's full response to history
        conversation_history.append({"role": "assistant", "content": raw_response})

        postcondition_to_check = None
        is_final_solution = False

        if solution_content:
            print(f"Model Generated solution:\n{solution_content}")
            postcondition_to_check = solution_content
            is_final_solution = True
        elif assertion_content:
            print(f"Model Generated assertions:\n{assertion_content}")
            postcondition_to_check = assertion_content

        if mode in ["greedy", "single-pass"]:
            postcondition_to_check = assertion_content
            is_final_solution = True

        if not postcondition_to_check:
            print("No assertions or solution found in this turn.")
            observation = "No assertions were generated. Please provide assertions in <assert> tags or a final answer in <solution> tags."
            conversation_history.append(
                {"role": "user", "content": f"<observation>\n{observation}\n{reminder}\n</observation>"}
            )
            continue

        # Clean up the postcondition before testing
        cleaned_postcondition = double_check_postcondition(
            postcondition_to_check, problem["entry_point"]
        )
        print(f"Postprocessed Generated postcondition:\n{cleaned_postcondition}")

        is_correct, result = evaluate_postcondition(
            problem, cleaned_postcondition, expected_output, base_only
        )

        if is_correct and is_final_solution:
            # Run power evaluation for the final solution
            if run_power_eval:
                print("🔍 Calculating completeness (power evaluation)...")
                power_results = evaluate_postcondition_power_single(
                    problem, cleaned_postcondition, problem["entry_point"], 
                    feedback_buggy_mutant=feedback_buggy_mutant
                )
                print(f"Completeness score: {power_results['completeness_score']:.3f}")
                print(f"Tests killed: {power_results['num_bopi_killed']}/{power_results['num_bopi_run']}")
                print(f"Codes killed: {power_results['num_codes_killed']}/{power_results['num_codes_run']}")
                result["power_evaluation"] = power_results
            else:
                print("⏭️ Skipping power evaluation as requested")

        if 'power_evaluation' in result and 'completeness_score' in result['power_evaluation']:
            cur_completeness_score = result['power_evaluation']['completeness_score']
        else:
            cur_completeness_score = 0
        if is_final_solution:
            completeness_trend.append((cur_completeness_score, turn))

        if is_correct and is_final_solution:
            # Check against completeness threshold if required
            if (
                completeness_threshold is not None
                and power_results["completeness_score"] * 100 < completeness_threshold
            ):
                if feedback_buggy_mutant:
                    observation = f"""\
Current postconditions are sound but not complete. Current completeness score is {power_results['completeness_score'] * 100:.1f}% (Threshold is {completeness_threshold}%).
They cannot distinguish the original implementation from the following buggy mutant:
{power_results['buggy_mutant_to_fix']}
Please refine your postconditions so they can detect the bug in this mutant and all other possible bugs.\
"""
                else:
                    observation = f"""\
The postconditions are sound but not complete. Current completeness score is {power_results['completeness_score'] * 100:.1f}% (Threshold is {completeness_threshold}%).
While the current assertions pass for the correct implementation, they lack the comprehensiveness to catch potential bugs in other flawed implementations. A truly robust set of postconditions should be exhaustive enough to detect all incorrect behaviors.\
"""

                conversation_history.append(
                    {"role": "user", "content": f"<observation>\n{observation}\n</observation>"}
                )
            else:
                print("✅ Postconditions passed all tests and meet completeness requirements!")
                return cleaned_postcondition, result, True, conversation_history, raw_responses, completeness_trend, {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens, "total_tokens": total_total_tokens}
        
        # Formulate observation message based on test results
        if is_correct:
            print("✅ Postcondition passed all tests!")
            observation = "Assertions are valid."
        else:
            logs = [log for log in result["base"][-1] + (result["plus"][-1] if result["plus"] else []) if log is not None]
            traceback_log = logs[0] if logs else ""
            print(f"❌ Postcondition failed with traceback:\n{traceback_log}")
            observation = f"Assertions failed.\nTraceback log:\n{traceback_log}"
        
        # Add a specific message if a final solution failed
        if is_final_solution and mode == "exploratory":
            if not is_correct:
                observation += ("\nThe postconditions are unsound. They are overly strict and raise an AssertionError even when "
                    "the code is implemented correctly. This indicates a flaw in the postconditions themselves, "
                    "making them unreliable for verifying code correctness.")
            
        # Keep track of the best attempt from assertion blocks
        if (best_postcondition is None and turn == max_turns) or (is_final_solution and is_correct and (best_result is None or ('power_evaluation' in result and result['power_evaluation']['completeness_score'] >= best_result['power_evaluation']['completeness_score']))):
            is_best_postcondition_correct = is_correct
            best_postcondition = cleaned_postcondition
            best_result = result
        
        # Append the observation to the history for the next turn
        observation_msg = f"<observation>\n{observation}\n{reminder}\n</observation>"
        conversation_history.append({"role": "user", "content": observation_msg})

        # except Exception as e:
        #     print(f"Error generating/testing postcondition: {str(e)}")
        #     observation = f"An error occurred: {str(e)}"
        #     observation_msg = f"<observation>\n{observation}\n{reminder}\n</observation>"
        #     conversation_history.append({"role": "user", "content": observation_msg})
        #     continue

    print("⚠️ Reached max turns without finding a perfect solution.")
    return best_postcondition, best_result, is_best_postcondition_correct, conversation_history, raw_responses, completeness_trend, {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens, "total_tokens": total_total_tokens}


def evaluate_postcondition_power_single(
    problem: Dict[str, Any],
    postcondition: str,
    entry_point: str,
    n_workers: int = 1,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
    feedback_buggy_mutant: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates the "completeness" or "power" of a postcondition against buggy code mutants.
    This is a simplified version of the full evaluation process.

    Args:
        problem: The problem dictionary.
        postcondition: The postcondition to test.
        entry_point: The function entry point name.
        n_workers: Number of workers for evaluation (set to 1 for simplicity).
        min_time_limit: Minimum time limit for tests.
        gt_time_limit_factor: Factor for ground truth time limit.

    Returns:
        A dictionary of power evaluation results.
    """
    from response_preprocessing import code_sanitize, wrap_code_solution
    from run_postcondition_evaluation import evaluate_post_condition_power

    if "MyDataset" in problem["task_id"]:
        buggy_codes_file = "d:/code/web/out_marshmallow_docfilter/marshmallow_mutations.jsonl"
        if not os.path.exists(buggy_codes_file):
            print(f"Warning: Buggy codes file {buggy_codes_file} not found. Skipping power evaluation.")
            return {
                "buggy_mutant_to_fix": None,
                "num_bopi_run": 0, "num_bopi_killed": 0, "num_codes_run": 0,
                "num_codes_killed": 0, "completeness_score": 0.0,
            }
        with open(buggy_codes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    buggy_code = json.loads(line)
                    if buggy_code["task_id"] == problem["task_id"]:
                        buggy_code["solution"] = buggy_code.get("solution") or buggy_code["mutated_code"]
                        buggy_codes.append(buggy_code)
    else:
        buggy_codes_file = "code_mutants/all_code_mutants_with_bad_output.jsonl.zip"
        if not os.path.exists(buggy_codes_file):
            print(f"Warning: Buggy codes file {buggy_codes_file} not found. Skipping power evaluation.")
            return {
                "buggy_mutant_to_fix": None,
                "num_bopi_run": 0, "num_bopi_killed": 0, "num_codes_run": 0,
                "num_codes_killed": 0, "completeness_score": 0.0,
            }
        with zipfile.ZipFile(buggy_codes_file, "r") as zip_ref:
            for filename in zip_ref.namelist():
                if filename.endswith(".jsonl"):
                    with zip_ref.open(filename) as f:
                        for line in f:
                            buggy_code = json.loads(line.decode("utf-8"))
                            if buggy_code["task_id"] == problem["task_id"]:
                                buggy_codes.append(buggy_code)

    if not buggy_codes:
        print(f"No buggy codes found for task {problem['task_id']}")
        return {
            "buggy_mutant_to_fix": None,
            "num_bopi_run": 0, "num_bopi_killed": 0, "num_codes_run": 0,
            "num_codes_killed": 0, "completeness_score": 0.0,
        }

    # Sanitize and wrap postcondition with each buggy code
    sanitized_postcondition = code_sanitize(postcondition)
    if not sanitized_postcondition:
        print("Postcondition could not be sanitized.")
        return {
            "buggy_mutant_to_fix": None,
            "num_bopi_run": 0, "num_bopi_killed": 0, "num_codes_run": 0,
            "num_codes_killed": 0, "completeness_score": 0.0,
        }

    wrapped_codes = []
    for buggy_code in buggy_codes:
        wrapped_code = wrap_code_solution(None, buggy_code["solution"], entry_point, sanitized_postcondition)
        buggy_code["wrapped"] = wrapped_code
        wrapped_codes.append(buggy_code)

    # Prepare data for the evaluation function
    postcondition_info = {
        "task_id": problem["task_id"],
        "response_num": 0,
        "entry_point": entry_point,
        "all_time_limits": [10] * 100,
    }

    # Create a mock Flags class to pass arguments
    class Flags:
        def __init__(self):
            self.min_time_limit = min_time_limit
            self.gt_time_limit_factor = gt_time_limit_factor
            self.i_just_wanna_run = True

    flags = Flags()

    def print_and_log(msg):
        print(msg)

    # Run the power evaluation
    try:
        power_results_raw = evaluate_post_condition_power(
            wrapped_codes, postcondition_info, n_workers, flags, print_and_log
        )
        
        task_id = problem["task_id"]
        if task_id in power_results_raw:
            result_data = power_results_raw[task_id]
            num_tests_run = result_data["num_tests_run"]
            num_tests_killed = result_data["num_tests_killed"]
            num_codes_run = len(result_data["test_results"])
            num_codes_killed = len(
                [x for x in result_data["test_results"] if x[0] == "killed at least one mutant"]
            )
            
            # Add buggy mutant feedback only when user passes --feedback-buggy-mutant
            buggy_mutant_to_fix = None
            if feedback_buggy_mutant:
                # Find all buggy mutants that were not killed
                surviving_mutants = []
                for i in range(len(wrapped_codes)):
                    if result_data['test_results'][i][0] != "killed at least one mutant":
                        surviving_mutants.append((i, wrapped_codes[i]))
                
                # Randomly sample one surviving mutant
                if surviving_mutants:
                    selected_idx, selected_mutant = random.choice(surviving_mutants)
                    buggy_mutant_to_fix = selected_mutant['solution']
                    print(f"Selected buggy mutant {selected_idx + 1} out of {len(surviving_mutants)} surviving mutants:")
                    print(selected_mutant['wrapped'])

            completeness_score = num_tests_killed / num_tests_run if num_tests_run > 0 else 0.0
            return {
                "buggy_mutant_to_fix": buggy_mutant_to_fix,
                "num_bopi_run": num_tests_run,
                "num_bopi_killed": num_tests_killed,
                "num_codes_run": num_codes_run,
                "num_codes_killed": num_codes_killed,
                "completeness_score": completeness_score,
            }
        else:
            return {
                "buggy_mutant_to_fix": None,
                "num_bopi_run": 0, "num_bopi_killed": 0, "num_codes_run": 0,
                "num_codes_killed": 0, "completeness_score": 0.0,
            }
    except Exception as e:
        print(f"Error in power evaluation: {str(e)}")
        return {
            "buggy_mutant_to_fix": None,
            "num_bopi_run": 0, "num_bopi_killed": 0, "num_codes_run": 0,
            "num_codes_killed": 0, "completeness_score": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate postconditions for EvalPlus or Marshmallow problems.")
    parser.add_argument("--task-id", required=True, help="Task ID to evaluate, e.g., 'HumanEval/20' or 'MyDataset/0'.")
    parser.add_argument("--dataset", choices=["evalplus", "marshmallow"], default="evalplus")
    parser.add_argument(
        "--problems-path",
        type=Path,
        default=Path("../../out_marshmallow_docfilter/marshmallow_problems.jsonl"),
        help="Marshmallow problems JSONL path.",
    )
    parser.add_argument(
        "--tests-path",
        type=Path,
        default=Path("../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl"),
        help="Marshmallow tests JSONL path.",
    )
    parser.add_argument(
        "--marshmallow-repo-root",
        type=Path,
        default=Path("../../marshmallow"),
        help="Path to the marshmallow repository root.",
    )
    parser.add_argument("--max-turns", type=int, default=12,
                        help="Maximum conversation turns to generate a correct postcondition.")
    parser.add_argument("--base-only", action="store_true",
                        help="Only test with base HumanEval inputs.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save the output JSON file.")
    parser.add_argument("--no-power-eval", action="store_true",
                        help="Skip power evaluation (completeness calculation).")
    parser.add_argument("--completeness-threshold", type=float, default=None,
                        help="Force LLM to continue generating assertions if the completeness does not exceed the threshold.")
    parser.add_argument('--mode', type=str, choices=['exploratory', 'greedy', 'single-pass'], default='exploratory', 
                        help='The experiment mode')
    parser.add_argument("--feedback-buggy-mutant", action="store_true",
                        help="Provide feedback with buggy mutant code when completeness threshold is not met.")

    args = parser.parse_args()

    # If a completeness threshold is set, power evaluation is required
    if args.completeness_threshold is not None:
        args.no_power_eval = False

    if args.dataset == "evalplus":
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        from run_postcondition_evaluation import get_groundtruth

        problematic_tasks = {"HumanEval/36", "HumanEval/83", "HumanEval/139", "HumanEval/160", "HumanEval/32"}
        if args.task_id in problematic_tasks:
            print(f"Skipping task {args.task_id} as it is known to have issues.")
            return

        problems = get_human_eval_plus()
        print(f"Loaded {len(problems)} problems from HumanEval+")
        if args.task_id not in problems:
            print(f"Error: Task ID '{args.task_id}' not found.")
            return

        problem = problems[args.task_id]
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash)[args.task_id]
    else:
        problems = load_marshmallow_dataset(args.problems_path, args.tests_path)
        print(f"Loaded {len(problems)} problems from Marshmallow dataset")
        if args.task_id not in problems:
            by_problem_id = {problem["problem_id"]: problem for problem in problems.values()}
            if args.task_id not in by_problem_id:
                print(f"Error: Task ID or problem_id '{args.task_id}' not found.")
                return
            problem = by_problem_id[args.task_id]
        else:
            problem = problems[args.task_id]
        problem["marshmallow_repo_root"] = str(args.marshmallow_repo_root.resolve())
        expected_output = {"base": [], "plus": [], "base_time": [], "plus_time": []}
        args.base_only = True

    # Generate and test the postcondition
    postcondition, result, success, conversation_history, raw_responses, completeness_trend, token_usage = generate_and_test_postcondition(
        problem=problem,
        expected_output=expected_output,
        max_turns=args.max_turns,
        base_only=args.base_only,
        run_power_eval=not args.no_power_eval,
        mode=args.mode,
        completeness_threshold=args.completeness_threshold,
        feedback_buggy_mutant=args.feedback_buggy_mutant,
    )

    # Compile and save the final results
    output_data = {
        "task_id": args.task_id,
        "success": success,
        "postcondition": postcondition,
        "results": result,
        "attempts": len(raw_responses),
        "conversation_history": conversation_history,
        "raw_responses": raw_responses,
        "completeness_trend" : completeness_trend,
        "token_usage": token_usage
    }

    # Add power evaluation results if applicable
    if success and result and "power_evaluation" in result:
        output_data["power_evaluation"] = result["power_evaluation"]
        print("Power evaluation results saved:")
        print(f"  Completeness score: {result['power_evaluation']['completeness_score']:.3f}")
        print(f"  Tests killed: {result['power_evaluation']['num_bopi_killed']}/{result['power_evaluation']['num_bopi_run']}")
        print(f"  Codes killed: {result['power_evaluation']['num_codes_killed']}/{result['power_evaluation']['num_codes_run']}")

    # Save output to a JSON file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"postcondition_results_{args.task_id.replace('/', '_')}.json"
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    # Print a summary of the final results
    print(f"\nFinal result for {args.task_id}:")
    print(f"Saved results to {output_path}")
    print(f"Success: {'✅' if success else '❌'}")
    print(f"Token usage: Prompt: {token_usage['prompt_tokens']} | Completion: {token_usage['completion_tokens']} | Total: {token_usage['total_tokens']}")
    print(f"Postcondition:\n{postcondition}")
    if result and "base" in result and result["base"]:
        print(f"Base tests passed: {sum(result['base'][1])}/{len(result['base'][1])}")
        if not args.base_only and result.get("plus"):
            print(f"Plus tests passed: {sum(result['plus'][1])}/{len(result['plus'][1])}")
    else:
        print("Base tests passed: N/A")


if __name__ == "__main__":
    main()