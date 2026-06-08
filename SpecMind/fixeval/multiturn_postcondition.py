#!/usr/bin/env python3
"""
Multiturn Postcondition Generation for Competitive Programming

This module implements agents for generating symbolic postconditions for competitive 
programming problems using the FixEval framework.
"""

import argparse
import json
import os
import subprocess
import time
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from loguru import logger

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
# from langchain.globals import set_llm_cache
# from langchain_community.cache import SQLiteCache

# set_llm_cache(SQLiteCache(database_path=".langchain.db"))

import pandas as pd  # type: ignore

from collections import defaultdict
from glob import glob

# Import FixEval components
import sys
sys.path.append("../evaluation")
from evaluation.execution_evaluation_TC_arc_MP import run_python, compare_files  # type: ignore
    
from src.call_api import OpenAIFixEvalGenerator
from src.template import transform_code_with_postcondition


class InvalidInstanceError(Exception):
    pass

def parse_model_response(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse model response to extract think, assert, and solution content."""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    assert_match = re.search(r"<assert>(.*?)</assert>", response, re.DOTALL)
    solution_match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else None
    assertion_content = assert_match.group(1).strip() if assert_match else None
    solution_content = solution_match.group(1).strip() if solution_match else None

    # Handle incomplete tags
    if not assert_match and response.strip().endswith("<assert>"):
        incomplete_assert_match = re.search(r"<assert>(.*)", response.strip(), re.DOTALL)
        if incomplete_assert_match:
            assertion_content = incomplete_assert_match.group(1).strip()

    return think_content, assertion_content, solution_content


def get_test_case_path(
    tgt_id: str, 
    problem_list_path: str = "src/problem_list.csv", 
    test_cases_root: str = "data/atcoder_test_cases"
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get test case directory for a given tgt_id."""
    problem_id = tgt_id.split("_")[0]
    problemlist = pd.read_csv(problem_list_path)
    
    problems: Dict[str, List[str]] = defaultdict(list)
    contest_info: Dict[str, Tuple[str, str]] = {}
    
    for index, row in problemlist.iterrows():
        if row['dataset'] == 'AtCoder':
            if "AtCoder Beginner Contest" in row['name']:
                number = row['name'].split(" ")[3]
                contest_key = "ABC" + number
                problems[contest_key].append(row['id'])
                contest_info[row['id']] = (contest_key, row['name'])
    
    if problem_id not in contest_info:
        return None, None, None
    
    contest_key, contest_name = contest_info[problem_id]
    folders = glob(f"{test_cases_root}/*")
    available_contests = [os.path.basename(folder) for folder in folders]
    
    contest_folder = None
    if contest_key in available_contests:
        contest_folder = contest_key
    elif contest_key.lower() in available_contests:
        contest_folder = contest_key.lower()
    
    if not contest_folder:
        return None, contest_name, None
    
    contest_problems = sorted(problems[contest_key])
    if problem_id in contest_problems:
        problem_index = contest_problems.index(problem_id)
        problem_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        if problem_index < len(problem_letters):
            problem_letter = problem_letters[problem_index]
            test_case_path = f"{test_cases_root}/{contest_folder}/{problem_letter}"
            
            if os.path.exists(test_case_path):
                return test_case_path, contest_name, problem_letter
    
    return None, contest_name, None


def run_python_with_mutant_detection(code: str, test_case_folder: str) -> Tuple[bool, int, int, int, list[str]]:
    """Thread-safe enhanced run_python that counts mutants killed by postconditions."""
    
    # Create thread-safe temporary directory
    thread_id = threading.get_ident()
    process_id = os.getpid()
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time() * 1000000)
    
    # Use system temporary directory for thread safety
    with tempfile.TemporaryDirectory(prefix=f"fixeval_{process_id}_{thread_id}_{unique_id}_{timestamp}_") as temp_dir:
        root_path = temp_dir
        
        # Write code to temporary file
        main_py_path = os.path.join(root_path, 'Main.py')
        with open(main_py_path, 'w', encoding='utf8') as fw:
            fw.write(code)
        
        in_files = glob(test_case_folder+"/in/*")
        
        # Test compilation
        p1 = subprocess.run(["python", "-m", "py_compile", main_py_path], 
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if p1.returncode != 0:
            return False, 0, len(in_files), 0, []

        did_not_match = 0
        mutants_killed = 0

        failed_test_logs: list[str] = []
        
        for file_idx, in_file in enumerate(in_files):
            # Use unique file names to avoid conflicts
            tc_file = os.path.join(root_path, f'stripped_TC_{file_idx}.txt')
            out_file_path = os.path.join(root_path, f'cmd_out_{file_idx}.txt')
            match_file_path = os.path.join(root_path, f'cmd_out_match_{file_idx}.txt')
            
            # Prepare test case input
            stripped_TC = open(in_file).read().strip()
            with open(tc_file, 'w') as f:
                f.write(stripped_TC)
            
            # Run code with test case
            # 5 GB memory limit
            cmd = f"prlimit --as=5368709120 -- timeout 5m python {main_py_path} < {tc_file}"
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)

            try:
                outs, errs = p.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                    p.wait(timeout=5)
                except:
                    pass
                did_not_match += 1
                continue
            
            if p.returncode != 0:
                did_not_match += 1
                failed_test_logs.append(f"Test case {in_file} failed with stdout:\n{outs.decode()}\n and stderr:\n{errs.decode()}")

                if b"PostconditionCatchBug" in errs or b"PostconditionCatchBug" in outs:
                    mutants_killed += 1

                continue
            
            # Compare output with expected result
            expected_out_file = in_file.replace("in", "out", 1).replace(".in", ".out", 1)
            
            # Write actual output
            with open(out_file_path, 'wb') as f:
                f.write(outs)
            
            # Copy expected output for comparison
            try:
                subprocess.run(["cp", expected_out_file, match_file_path], check=True)
                
                if not compare_files(out_file_path, match_file_path):
                    did_not_match += 1

                    failed_test_logs.append(f"Test case {in_file} failed with stdout:\n{outs.decode()}\n and stderr:\n{errs.decode()}")
            except subprocess.CalledProcessError:
                # If copy fails, treat as mismatch
                did_not_match += 1

                failed_test_logs.append(f"Test case {in_file} failed with stdout:\n{outs.decode()}\n and stderr:\n{errs.decode()}")
        
        # Temporary directory is automatically cleaned up
        return True, len(in_files)-did_not_match, len(in_files), mutants_killed, failed_test_logs


class BaseAgent:
    """Base Agent for competitive programming postcondition generation."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float,
        run_name: str,
        instance_id: str,
        output_dir: Path,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            max_completion_tokens=10000,
            max_retries=100,
            timeout=600,
            temperature=temperature,
            stop_sequences=["<observation>", "</assert>", "</solution>"],
            # cache=True,
        )
        self.history: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
        self.run_name = run_name
        self.instance_id = instance_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.output_dir / f"{self.instance_id}.json"

    @property
    def system_prompt_template(self) -> str:
        """Return system prompt template."""
        raise NotImplementedError

    def build_initial_messages(self, code: str, entrypoint: str, max_turns: int, assertion_turns: int = None) -> List[Union[SystemMessage, HumanMessage]]:
        """Build initial conversation messages."""
        if assertion_turns is None:
            assertion_turns = 5 * max_turns
        sys_text = self.system_prompt_template.format(entrypoint=entrypoint, max_turns=max_turns, assertion_turns=assertion_turns)
        sys = SystemMessage(content=sys_text)
        human = HumanMessage(content=f"```python\n{code}\n```")
        return [sys, human]

    def build_observation(self, correct_tc: int, total_tc: int, mutants_killed: int, remaining_submission_turns: int, remaining_assertion_turns: int, is_submission_turn: bool) -> str:
        """Build observation message."""
        if correct_tc == total_tc:
            observation = "Assertions are valid."
            if mutants_killed == 0 and is_submission_turn:
                observation += " However, the postcondition did not catch any bugs in the buggy implementation."
        else:
            observation = f"Assertions failed. Your postcondition caused {total_tc - correct_tc} test cases to fail."
        
        if remaining_submission_turns <= 0:
            observation += "\n<reminder>This is your final submission turn. You must submit a <solution> now.</reminder>"
        else:
            if is_submission_turn:
                observation += f"\n<reminder>You have {remaining_submission_turns} submission turns remaining and {remaining_assertion_turns} assertion turns remaining.</reminder>"
            else:
                observation += f"\n<reminder>You have {remaining_submission_turns} submission turns remaining and {remaining_assertion_turns} assertion turns remaining.</reminder>"
        return observation

    def propose(self, code: str, entrypoint: str, max_turns: int, assertion_turns: int = None) -> str:
        """Call LLM and return response."""
        if not self.history:
            self.history = self.build_initial_messages(code, entrypoint, max_turns, assertion_turns)
        
        try:
            response = self.llm.invoke(self.history)
            raw_response: str = response.content

            # Handle incomplete tags
            if "<assert>\n" in raw_response and not raw_response.strip().endswith("</assert>"):
                raw_response += "</assert>"
            
            if "<solution>\n" in raw_response and not raw_response.strip().endswith("</solution>"):
                raw_response += "</solution>"

            response.content = raw_response
            self.history.append(response)
            return raw_response
        except Exception as e:
            logger.error(f"LLM invoke failed: {e}")
            logger.exception(e)
            raise

    def append_observation(self, text: str) -> None:
        """Add observation to conversation history."""
        self.history.append(HumanMessage(content=f"<observation>\n{text}\n</observation>"))

    def _persist(self, record: Dict[str, Any]) -> None:
        """Persist trajectory to file."""
        # append history to record
        record["messages"] = [{"role": message.type, "content": message.content} for message in self.history]
        tmp_path = self.trajectory_path.with_suffix(".json.tmp")
        with open(tmp_path, 'w') as f:
            json.dump(record, f, indent=2)
        os.replace(tmp_path, self.trajectory_path)

    def run(
        self,
        code: str,
        buggy_code: str,
        test_case_folder: str,
        entrypoint: str,
        max_turns: int,
        run_until_catch_bug: bool,
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Run the agent."""
        raise NotImplementedError


class SimpleAgent(BaseAgent):
    """Single-turn agent that generates multiple independent samples."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override with temperature=1 for diverse sampling
        model_name = getattr(self.llm, 'model_name', 'gpt-4')
        self.llm = ChatOpenAI(
            model=model_name,
            max_completion_tokens=10000,
            max_retries=100,
            timeout=600,
            temperature=1.0,
            stop_sequences=["<observation>", "</assert>", "</solution>"],
            # cache=False,
        )

    @property
    def system_prompt_template(self) -> str:
        return """
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
    
-   `<solution>`: Propose one symbolic postconditions in Python. Must be a valid `assert` statement, preceded by a brief comment.
    
<solution>
# Checks that no output element exceeds input elements
assert all(x <= max(data) for x in return_values[0])
</solution>
---

### Postcondition Rules

Your task is to write a symbolic postcondition for {entrypoint}. The postcondition should be in Python, and consist of exactly one assert statement. A Python comment explaining the postcondition's meaning should precede it. For variables, the postcondition should only use the input parameters defined in the function stub and a hypothetical return value of the function, which we'll assume is stored in a variable `return_values`.

For string manipulation, Python's `re` (regular expressions) library can be used. If other Python standard library functions are required, include the necessary imports. However, refrain from using external libraries or calling the function itself (in this case, {entrypoint}) within the postcondition.

If the postcondition calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure (i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

---

### Your Task

You will now be given a Python function `{entrypoint}`. Begin by analyzing it with `<think>`, then proceed to propose assertions using `<solution>`.
Only use input parameters and a hypothetical return value stored in variable `return_values`.
Input parameters are stored in the `lines: list[str]` variable (lines of stdin). `return_values: list[str]` is the string contains the output of the function (lines of stdout).

Let's begin.
        """.strip()

    def run(
        self,
        code: str,
        buggy_code: str,
        test_case_folder: str,
        entrypoint: str,
        max_turns: int,
        run_until_catch_bug: bool,
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Run independent single-turn samples."""
        trajectory: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "run_name": self.run_name,
            "config": {
                "model_name": getattr(self.llm, "model_name", "unknown"),
                "temperature": getattr(self.llm, "temperature", 1.0),
                "max_turns": max_turns,
                "run_until_catch_bug": run_until_catch_bug,
                "mode": "single-pass",
            },
            "entrypoint": entrypoint,
            "turns": [],
        }

        success = None
        final_postcondition = None

        for turn in range(1, max_turns + 1):
            # Reset conversation for independence
            self.history = []
            
            raw = self.propose(code, entrypoint, 1)
            _, postcondition_candidate, solution_content = parse_model_response(raw)
            
            if solution_content:
                postcondition_candidate = solution_content

            if not postcondition_candidate or not postcondition_candidate.strip():
                turn_record: Dict[str, Any] = {
                    "turn": turn,
                    "raw_response": raw,
                    "postcondition": None,
                    "note": "no postcondition found"
                }
                trajectory["turns"].append(turn_record)
                continue

            # Test postcondition
            try:
                # test if this data point is valid or not
                test_code = transform_code_with_postcondition(code, "assert True")
                compiles, correct_tc, total_tc, _, failed_test_logs = run_python_with_mutant_detection(test_code, test_case_folder)
                is_correct = compiles and correct_tc == total_tc
                if not is_correct:
                    raise InvalidInstanceError(f"Instance '{self.instance_id}' is invalid")
                
                transformed_code = transform_code_with_postcondition(code, postcondition_candidate)
                compiles, correct_tc, total_tc, _, failed_test_logs = run_python_with_mutant_detection(transformed_code, test_case_folder)
                
                # Test against buggy code for effectiveness
                transformed_buggy = transform_code_with_postcondition(buggy_code, postcondition_candidate)
                _, buggy_correct, buggy_total, mutants_killed, buggy_failed_test_logs = run_python_with_mutant_detection(transformed_buggy, test_case_folder)
                
                is_correct = compiles and correct_tc == total_tc
                bug_caught = is_correct and mutants_killed > 0
                
                turn_record = {
                    "turn": turn,
                    "raw_response": raw,
                    "postcondition": postcondition_candidate,
                    "is_correct": is_correct,
                    "correct_tc": correct_tc,
                    "total_tc": total_tc,
                    "mutants_killed": mutants_killed,
                    "bug_caught": bug_caught,
                    "failed_test_logs": failed_test_logs,
                    "buggy_failed_test_logs": buggy_failed_test_logs,
                }
                trajectory["turns"].append(turn_record)
                
                if is_correct:
                    if not run_until_catch_bug or bug_caught:
                        success = bug_caught
                        final_postcondition = postcondition_candidate
                        break
                        
            except InvalidInstanceError as e:
                raise e

            except Exception as e:
                logger.error(f"Error testing postcondition: {str(e)}")
                logger.exception(e)
                turn_record = {
                    "turn": turn,
                    "raw_response": raw,
                    "postcondition": postcondition_candidate,
                    "error": str(e)
                }
                trajectory["turns"].append(turn_record)
        else:
            success = False
        trajectory["success"] = success
        trajectory["final_postcondition"] = final_postcondition
        self._persist(trajectory)
        return final_postcondition, success, trajectory


class RetryAgent(BaseAgent):
    """Retry agent that keeps trying until success or max turns."""
    
    @property
    def system_prompt_template(self) -> str:
        return """
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
    
-   `<solution>`: Propose one symbolic postconditions in Python. Must be a valid `assert` statement, preceded by a brief comment.
    
<solution>
# Checks that no output element exceeds input elements
assert all(x <= max(data) for x in return_values[0])
</solution>
---

### Postcondition Rules

Your task is to write a symbolic postcondition for {entrypoint}. The postcondition should be in Python, and consist of exactly one assert statement. A Python comment explaining the postcondition's meaning should precede it. For variables, the postcondition should only use the input parameters defined in the function stub and a hypothetical return value of the function, which we'll assume is stored in a variable `return_values`.

For string manipulation, Python's `re` (regular expressions) library can be used. If other Python standard library functions are required, include the necessary imports. However, refrain from using external libraries or calling the function itself (in this case, {entrypoint}) within the postcondition.

If the postcondition calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure (i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

---

### Your Task

You will now be given a Python function `{entrypoint}`. Begin by analyzing it with `<think>`, then proceed to propose assertions using `<solution>`.
Only use input parameters and a hypothetical return value stored in variable `return_values`.
Input parameters are stored in the `lines: list[str]` variable (lines of stdin). `return_values: list[str]` is the string contains the output of the function (lines of stdout).

Let's begin.
        """.strip()

    def run(
        self,
        code: str,
        buggy_code: str,
        test_case_folder: str,
        entrypoint: str,
        max_turns: int,
        run_until_catch_bug: bool,
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Run retry-based generation."""
        trajectory: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "run_name": self.run_name,
            "config": {
                "model_name": getattr(self.llm, "model_name", "unknown"),
                "temperature": getattr(self.llm, "temperature", 0.0),
                "max_turns": max_turns,
                "run_until_catch_bug": run_until_catch_bug,
                "mode": "greedy",
            },
            "entrypoint": entrypoint,
            "turns": [],
        }

        success = None
        final_postcondition = None

        for turn in range(1, max_turns + 1):
            remaining = max_turns - turn
            raw = self.propose(code, entrypoint, max_turns)
            _, _, solution_content = parse_model_response(raw)
            
            postcondition_candidate = solution_content

            if not postcondition_candidate or not postcondition_candidate.strip():
                reminder = "No assertions generated. Please provide a final answer in <solution> tags."
                self.append_observation(reminder)
                turn_record: Dict[str, Any] = {
                    "turn": turn,
                    "raw_response": raw,
                    "postcondition": None,
                    "note": "no postcondition found"
                }
                trajectory["turns"].append(turn_record)
                continue

            # Test postcondition
            try:
                # test if this data point is valid or not
                test_code = transform_code_with_postcondition(code, "assert True")
                compiles, correct_tc, total_tc, _, failed_test_logs = run_python_with_mutant_detection(test_code, test_case_folder)
                is_correct = compiles and correct_tc == total_tc
                if not is_correct:
                    raise InvalidInstanceError(f"Instance '{self.instance_id}' is invalid")

                transformed_code = transform_code_with_postcondition(code, postcondition_candidate)
                compiles, correct_tc, total_tc, _, failed_test_logs = run_python_with_mutant_detection(transformed_code, test_case_folder)
                
                # Test against buggy code
                transformed_buggy = transform_code_with_postcondition(buggy_code, postcondition_candidate)
                _, buggy_correct, buggy_total, mutants_killed, buggy_failed_test_logs = run_python_with_mutant_detection(transformed_buggy, test_case_folder)
                
                is_correct = compiles and correct_tc == total_tc
                bug_caught = is_correct and mutants_killed > 0
                
                turn_record = {
                    "turn": turn,
                    "raw_response": raw,
                    "postcondition": postcondition_candidate,
                    "is_correct": is_correct,
                    "correct_tc": correct_tc,
                    "total_tc": total_tc,
                    "mutants_killed": mutants_killed,
                    "bug_caught": bug_caught,
                    "failed_test_logs": failed_test_logs,
                    "buggy_failed_test_logs": buggy_failed_test_logs,
                }
                trajectory["turns"].append(turn_record)
                
                if is_correct:
                    if not run_until_catch_bug or bug_caught:
                        success = bug_caught
                        final_postcondition = postcondition_candidate
                        break
                
                # Provide feedback for RetryAgent (legacy single turn counting)
                if correct_tc == total_tc:
                    observation = "Assertions are valid."
                    if mutants_killed == 0:
                        observation += " However, the postcondition did not catch any bugs in the buggy implementation."
                else:
                    observation = f"Assertions failed. Your postcondition caused {total_tc - correct_tc} test cases to fail."
                
                # if remaining <= 0:
                #     observation += "\n<reminder>This is your final turn. You must submit a <solution> now.</reminder>"
                # else:
                #     observation += f"\n<reminder>You have {remaining} turns remaining.</reminder>"
                self.append_observation(observation)
                        
            except InvalidInstanceError as e:
                raise e

            except Exception as e:
                logger.error(f"Error testing postcondition: {str(e)}")
                logger.exception(e)
                turn_record = {
                    "turn": turn,
                    "raw_response": raw,
                    "postcondition": postcondition_candidate,
                    "error": str(e),
                }
                trajectory["turns"].append(turn_record)
                
                observation = f"Error testing postcondition: {str(e)}"
                self.append_observation(observation)
        else:
            success = False

        trajectory["success"] = success
        trajectory["final_postcondition"] = final_postcondition
        self._persist(trajectory)
        return final_postcondition, success, trajectory


class MultiTurnAgent(BaseAgent):
    """Multi-turn agent with exploratory <assert> and final <solution>."""
    
    @property
    def system_prompt_template(self) -> str:
        return """
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
assert all(x <= max(data) for x in return_values[0])
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

You have two types of turns:
    - **Submission turns**: You have a maximum of {max_turns} submission turns. Each `<solution>` you submit counts as one submission turn.
    - **Assertion turns**: You have a maximum of {assertion_turns} assertion turns. Each `<assert>` you submit counts as one assertion turn.
    - `<think>` blocks do not count toward either limit and can be used freely.
    
    - You may submit a `<solution>` at any time when you believe you have a strong postcondition. Multiple submissions are allowed, and each will be treated as a potential candidate for the final solution.
    - If you submit one or more `<solution>` blocks but are still required to continue, this indicates that your current postcondition is not yet fully correct or complete. You must then continue exploring and refining through additional reasoning and `<assert>` checks before submitting another `<solution>`.
    - Use the early rounds to carefully reason about the function and to issue `<assert>` checks that validate your understanding of its behavior.
    - In later rounds, refine your postconditions based on your reasoning and observations.
    - However, avoid submitting <solution> blocks too frequently. Use most of your assertion turns for exploration (`<assert>`) and only propose a new <solution> after substantial refinement or new insights.
    - Before submitting any <solution>, you must test it internally to ensure: It has no syntax errors; Executing it will not raise an `AssertionError`; It faithfully reflects your reasoning so far. Only then should you submit it as a valid candidate.
    - If no `<solution>` has been submitted by the final submission turn, you must submit one at that point.
---

### Postcondition Rules

Your task is to write a symbolic postcondition for {entrypoint}. The postcondition should be in Python, and consist of exactly one assert statement. A Python comment explaining the postcondition's meaning should precede it. For variables, the postcondition should only use the input parameters defined in the function stub and a hypothetical return value of the function, which we'll assume is stored in a variable `return_values`.

For string manipulation, Python's `re` (regular expressions) library can be used. If other Python standard library functions are required, include the necessary imports. However, refrain from using external libraries or calling the function itself (in this case, {entrypoint}) within the postcondition.

If the postcondition calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure (i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

---

### Your Task

You will now be given a Python function `{entrypoint}`. Begin by analyzing it with `<think>`, then proceed to propose assertions using `<assert>`, review feedback with `<observation>`, and finalize using `<solution>` — **within your limits of {max_turns} submission turns and {assertion_turns} assertion turns.**
Only use input parameters and a hypothetical return value stored in variable `return_values`.
Input parameters are stored in the `lines: list[str]` variable (lines of stdin). `return_values: list[str]` is the string contains the output of the function (lines of stdout).

Let's begin.
        """.strip()

    def run(
        self,
        code: str,
        buggy_code: str,
        test_case_folder: str,
        entrypoint: str,
        max_turns: int,
        run_until_catch_bug: bool,
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Run multi-turn exploration with separate turn counting."""
        max_assertion_turns = 5 * max_turns
        
        trajectory: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "run_name": self.run_name,
            "config": {
                "model_name": getattr(self.llm, "model_name", "unknown"),
                "temperature": getattr(self.llm, "temperature", 0.0),
                "max_turns": max_turns,
                "max_assertion_turns": max_assertion_turns,
                "run_until_catch_bug": run_until_catch_bug,
                "mode": "exploratory",
            },
            "entrypoint": entrypoint,
            "turns": [],
        }

        success = None
        final_postcondition = None
        best_postcondition = None
        
        submission_turns_used = 0
        assertion_turns_used = 0
        total_turns = 0

        while True:
            total_turns += 1
            remaining_submission_turns = max_turns - submission_turns_used
            remaining_assertion_turns = max_assertion_turns - assertion_turns_used
            
            # Check if we should stop
            if remaining_submission_turns <= 0 and remaining_assertion_turns <= 0:
                break
 
            raw = self.propose(code, entrypoint, max_turns, max_assertion_turns)
            think_content, assertion_content, solution_content = parse_model_response(raw)
            
            is_submission = bool(solution_content)
            is_assertion = bool(assertion_content)
            postcondition_candidate = solution_content if is_submission else assertion_content

            # Check turn limits before processing
            if is_submission and remaining_submission_turns <= 0:
                reminder = "No more submission turns available. You can only use <assert> or <think> now."
                self.append_observation(reminder)
                break
                # continue
                
            if is_assertion and remaining_assertion_turns <= 0:
                reminder = "No more assertion turns available. You can only use <solution> or <think> now."
                if remaining_submission_turns <= 0:
                    reminder += " Actually, no turns remaining at all. Stopping."
                    break
                self.append_observation(reminder)
                continue

            # Count the turn if it's a submission or assertion
            if is_submission:
                submission_turns_used += 1
                remaining_submission_turns -= 1
            elif is_assertion:
                assertion_turns_used += 1
                remaining_assertion_turns -= 1

            if not postcondition_candidate or not postcondition_candidate.strip():
                reminder = "No assertions generated. Please provide assertions in <assert> or <solution> tags."
                if remaining_submission_turns <= 0:
                    reminder += "\n<reminder>No submission turns remaining. You must submit a <solution> if you have one ready.</reminder>"
                else:
                    reminder += f"\n<reminder>You have {remaining_submission_turns} submission turns remaining and {remaining_assertion_turns} assertion turns remaining.</reminder>"
                self.append_observation(reminder)
                
                turn_record: Dict[str, Any] = {
                    "turn": total_turns,
                    "submission_turn": submission_turns_used if is_submission else None,
                    "assertion_turn": assertion_turns_used if is_assertion else None,
                    "raw_response": raw,
                    "postcondition": None,
                    "is_submission": is_submission,
                    "is_assertion": is_assertion,
                    "note": "no postcondition found"
                }
                trajectory["turns"].append(turn_record)
                continue

            # Test postcondition
            try:
                # test if this data point is valid or not
                test_code = transform_code_with_postcondition(code, "assert True")
                compiles, correct_tc, total_tc, _, failed_test_logs = run_python_with_mutant_detection(test_code, test_case_folder)
                is_correct = compiles and correct_tc == total_tc
                if not is_correct:
                    raise InvalidInstanceError(f"Instance '{self.instance_id}' is invalid")

                transformed_code = transform_code_with_postcondition(code, postcondition_candidate)
                compiles, correct_tc, total_tc, _, failed_test_logs = run_python_with_mutant_detection(transformed_code, test_case_folder)
                
                # Test against buggy code
                transformed_buggy = transform_code_with_postcondition(buggy_code, postcondition_candidate)
                _, buggy_correct, buggy_total, mutants_killed, buggy_failed_test_logs = run_python_with_mutant_detection(transformed_buggy, test_case_folder)
                
                is_correct = compiles and correct_tc == total_tc
                bug_caught = is_correct and mutants_killed > 0
                
                if is_correct and best_postcondition is None:
                    best_postcondition = postcondition_candidate
                
                turn_record = {
                    "turn": total_turns,
                    "submission_turn": submission_turns_used if is_submission else None,
                    "assertion_turn": assertion_turns_used if is_assertion else None,
                    "raw_response": raw,
                    "postcondition": postcondition_candidate,
                    "is_submission": is_submission,
                    "is_assertion": is_assertion,
                    "is_correct": is_correct,
                    "correct_tc": correct_tc,
                    "total_tc": total_tc,
                    "mutants_killed": mutants_killed,
                    "bug_caught": bug_caught,
                    "failed_test_logs": failed_test_logs,
                    "buggy_failed_test_logs": buggy_failed_test_logs,
                }
                trajectory["turns"].append(turn_record)
                
                # Check if we should break
                if is_submission:
                    if is_correct:
                        if not run_until_catch_bug or bug_caught:
                            success = bug_caught
                            final_postcondition = postcondition_candidate
                            break
                
                # Provide feedback
                observation = self.build_observation(correct_tc, total_tc, mutants_killed, remaining_submission_turns, remaining_assertion_turns, is_submission)
                self.append_observation(observation)
            
            except InvalidInstanceError as e:
                raise e

            except Exception as e:
                logger.error(f"Error testing postcondition: {str(e)}")
                logger.exception(e)
                turn_record = {
                    "turn": total_turns,
                    "submission_turn": submission_turns_used if is_submission else None,
                    "assertion_turn": assertion_turns_used if is_assertion else None,
                    "raw_response": raw,
                    "postcondition": postcondition_candidate,
                    "is_submission": is_submission,
                    "is_assertion": is_assertion,
                    "error": str(e)
                }
                trajectory["turns"].append(turn_record)
                
                observation = f"Error testing postcondition: {str(e)}"
                self.append_observation(observation)
            
        if success is None:
            success = False

        trajectory["success"] = success
        trajectory["final_postcondition"] = final_postcondition or best_postcondition
        trajectory["submission_turns_used"] = submission_turns_used
        trajectory["assertion_turns_used"] = assertion_turns_used
        trajectory["total_turns"] = total_turns
        self._persist(trajectory)
        return final_postcondition or best_postcondition, success, trajectory


def load_fixeval_sample(data_path: str, sample_idx: int = 0) -> Dict[str, Any]:
    df = pd.read_json(data_path, lines=True)
    df['instance_id'] = df['src_id'] + '_' + df['tgt_id']
    # Ignore Runtime Error
    return df.query('src_verdict == "Wrong Answer"').reset_index(drop=True).iloc[sample_idx].to_dict()


def load_fixeval_sample_by_instance_id(data_path: str, instance_id: str) -> Dict[str, Any]:
    df = pd.read_json(data_path, lines=True)
    df['instance_id'] = df['src_id'] + '_' + df['tgt_id']
    df = df.query('src_verdict == "Wrong Answer"')
    matched = df[df["instance_id"] == instance_id]
    if matched.empty:
        raise ValueError(f"No sample found with instance_id == {instance_id}")
    return matched.iloc[0].to_dict()


def is_trajectory_completed(trajectory_path: Path) -> bool:
    """Check if trajectory file exists and has success != None."""
    if not trajectory_path.exists():
        return False
    
    try:
        with open(trajectory_path, 'r') as f:
            trajectory = json.load(f)
        
        # Check if success field exists and is not None
        success = trajectory.get("success")
        return success is not None
    except (json.JSONDecodeError, IOError, KeyError):
        # If file is corrupted or doesn't have expected structure, treat as incomplete
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Multiturn postcondition generation for competitive programming")
    parser.add_argument("--data-path", required=True, help="Path to FixEval data file")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to process")
    parser.add_argument("--instance-id", type=str, default=None, help="Process specific instance by instance_id (overrides --sample-idx)")
    parser.add_argument("--mode", choices=["single-pass", "greedy", "exploratory"], default="exploratory")
    parser.add_argument("--model-name", default="llama4-scout-instruct-basic")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--run-name", default="competitive_programming")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-until-catch-bug", action="store_true", help="Continue until bug is caught")
    parser.add_argument("--problem-list", default="src/problem_list.csv", help="Path to problem list CSV")
    parser.add_argument("--test-cases-root", default="data/atcoder_test_cases", help="Root directory for test cases")
    parser.add_argument("--resume", action="store_true", help="Skip processing instances that already have completed trajectories (success != None)")
    
    args = parser.parse_args()
    
    # Load sample
    if args.instance_id:
        sample = load_fixeval_sample_by_instance_id(args.data_path, args.instance_id)
        print(f"Processing instance {args.instance_id}: {sample['instance_id']}")
    else:
        sample = load_fixeval_sample(args.data_path, args.sample_idx)
        print(f"Processing sample {args.sample_idx}: {sample['instance_id']}")
    
    # Generate codes
    generator = OpenAIFixEvalGenerator(model=args.model_name)
    buggy_code = generator.detokenize_code(sample['src'])
    target_code = generator.detokenize_code(sample['tgt'])
    
    # Get test case path
    test_case_path, contest_name, problem_letter = get_test_case_path(
        sample['tgt_id'], args.problem_list, args.test_cases_root
    )
    
    if not test_case_path:
        print(f"❌ No test cases found for {sample['instance_id']}")
        return 1
    
    print(f"✅ Found test cases: {contest_name} Problem {problem_letter}")
    print(f"Test case directory: {test_case_path}")
    
    # Initialize agent
    output_dir = Path(args.output_dir) / args.run_name
    entrypoint = sample['tgt_id']
    computed_instance_id = f"{sample['src_id']}_{sample['tgt_id']}"
    
    # Check if resume is enabled and trajectory already completed
    if args.resume:
        trajectory_path = output_dir / f"{computed_instance_id}.json"
        if is_trajectory_completed(trajectory_path):
            print(f"⏭️  Skipping {computed_instance_id}: trajectory already completed (success != None)")
            
            # Load and display existing results
            try:
                with open(trajectory_path, 'r') as f:
                    existing_trajectory = json.load(f)
                success = existing_trajectory.get("success", False)
                final_postcondition = existing_trajectory.get("final_postcondition")
                
                print(f"✅ Existing result: Success = {success}")
                if final_postcondition:
                    print(f"📝 Existing postcondition: {final_postcondition}")
                
                return 0 if success else 1
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing trajectory: {e}")
                print("Proceeding with fresh run...")
            
    
    agent: BaseAgent
    if args.mode == "single-pass":
        agent = SimpleAgent(
            model_name=args.model_name,
            temperature=args.temperature,
            run_name=args.run_name,
            instance_id=computed_instance_id,
            output_dir=output_dir,
        )
    elif args.mode == "greedy":
        agent = RetryAgent(
            model_name=args.model_name,
            temperature=args.temperature,
            run_name=args.run_name,
            instance_id=computed_instance_id,
            output_dir=output_dir,
        )
    elif args.mode == "exploratory":
        agent = MultiTurnAgent(
            model_name=args.model_name,
            temperature=args.temperature,
            run_name=args.run_name,
            instance_id=computed_instance_id,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    # Run agent
    print(f"🚀 Running {args.mode} agent with {args.max_turns} max turns...")
    final_postcondition, success, trajectory = agent.run(
        code=target_code,
        buggy_code=buggy_code,
        test_case_folder=test_case_path,
        entrypoint=entrypoint,
        max_turns=args.max_turns,
        run_until_catch_bug=args.run_until_catch_bug,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"🎯 RESULTS")
    print(f"{'='*60}")
    print(f"Success: {'✅' if success else '❌'}")
    print(f"Trajectory saved to: {agent.trajectory_path}")
    
    if final_postcondition:
        print(f"\n📝 Final Postcondition:")
        print(final_postcondition)
    else:
        print("❌ No valid postcondition found")
    
    # Print summary stats
    if trajectory["turns"]:
        correct_turns = sum(1 for t in trajectory["turns"] if t.get("is_correct", False))
        bug_catching_turns = sum(1 for t in trajectory["turns"] if t.get("bug_caught", False))
        
        print(f"\n📊 Summary:")
        print(f"Total turns: {len(trajectory['turns'])}")
        print(f"Correct postconditions: {correct_turns}")
        print(f"Bug-catching postconditions: {bug_catching_turns}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())