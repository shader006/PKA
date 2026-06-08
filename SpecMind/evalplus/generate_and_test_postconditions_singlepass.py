import argparse
import json
import os
import openai
import time
import re
import ast
from typing import Dict, Any, List, Tuple, Optional
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.eval import SUCCESS, untrusted_check
from run_postcondition_evaluation import get_groundtruth, evaluate_post_condition_power
from pathlib import Path
from response_preprocessing import code_sanitize, wrap_code_solution

# Configuration
# OPENAI_API_KEY = "your-api-key"  # Replace with your actual API key
MODEL_NAME = "llama4-scout-instruct-basic"  # or "gpt-3.5-turbo"
MAX_ATTEMPTS = 5
TEMPERATURE = 1
# client = wrap_openai(openai.OpenAI())
client = openai.OpenAI()

# Initialize OpenAI client
# openai.api_key = OPENAI_API_KEY
PROMPTS = {
    "old": (
        """You have the following Python code, including the function {entrypoint} that behaves as specified in its docstring:

{codeStubAndDocstring}{canonical_solution}
Please write exactly one symbolic postcondition for {entrypoint}. Please write the postcondition in Python, and use exactly one python assert statement at the end of the postcondition. Include a Python comment before the postcondition explaining what the postcondition means.  For variables, only use the function inputs and the return value of the function. You can use python's re (regular expressions) if needed to deal with strings. Do not call {entrypoint} itself in the postcondition. Instead, assume that the function has already been called and its return value is available in a variable called `return_value` that you can use. In the postcondition, only use functions that are part of the functional subset of python (e.g., all(), len(), map(), filter(), etc.). Do not return any other textual description of the code other than the Python comment.

Specifically, the format of your response should be:

```
# Comment explaining what the postcondition does
CODE FOR EXACTLY ONE POSTCONDITION USING ASSERT GOES HERE
```
"""
    ),
    "base": (
        """You are provided with the following Python function implementation for {entrypoint}, and you want to ensure it is implemented correctly according to the specification in the docstring:

{codeStubAndDocstring}{canonical_solution}

Your task is to write a symbolic postcondition for {entrypoint}. The postcondition should be in Python, and consist of exactly one assert statement. A Python comment explaining the postcondition's meaning should precede it. For variables, the postcondition should only use the input parameters defined in the function stub and a hypothetical return value of the function, which we'll assume is stored in a variable `return_value`.

For string manipulation, Python's `re` (regular expressions) library can be used. If other Python standard library functions are required, include the necessary imports. However, refrain from using external libraries or calling the function itself (in this case, {entrypoint}) within the postcondition.

If the postcondition calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure (i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

Although the postcondition should be less computationally complex than the function itself and relatively simple, it should not be trivial. It should encapsulate an aspect of the function output specification without implementing the function itself and should be easily readable by a human.

While not trivial, your postcondition should still be very simple and short. It should be a single line of code that is not too long, and it should capture only one aspect of the function's behavior, not all of it. For example, if the goal of the function were to sort a list, you might write a postcondition that checks that the elements in the list are in sorted order, or you might write a postcondition that checks that the list is the same length as the input list. You would not write a postcondition that checks both of these things.

The format of your response should be:
```
# Comment explaining what aspect of the function the symbolic postcondition checks
CODE FOR EXACTLY ONE POSTCONDITION USING ASSERT GOES HERE
``` 

The postcondition should hold true whenever the function {entrypoint} executes successfully as specified in the docstring, regardless of its internal implementation.        
"""
    ),
}

def generate_postcondition(problem: Dict[str, Any], conversation_history: list[dict[str, str]] = []) -> str:
    """
    Generate a postcondition for the given problem using OpenAI's LLM.
    Optionally, append conversation_history (list of formatted blocks) to the user prompt.
    Args:
        problem: The problem dictionary.
        conversation_history: List of formatted conversation blocks to append to the prompt.
    Returns:
        The extracted postcondition string.
    """
    prompt = PROMPTS['base'].format(
        entrypoint=problem['entry_point'],
        codeStubAndDocstring=problem['prompt'],
        canonical_solution=problem['canonical_solution']
    )
    messages=[
        {"role": "system", "content": "You are a programming assistant that generates executable python only. You generate correct code, so you only generate code you are sure of. You have Python comments explaining your intent when possible."},
        {"role": "user", "content": prompt}
    ]

    if conversation_history:
        # Append conversation history to the messages
        for block in conversation_history:
            messages.append(block)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=1024
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
    return raw_content

def wrap_with_postcondition(code: str, postcondition: str, entry_point: str) -> str:
    """
    Wrap the original code with the generated postcondition.
    The wrapper will bind *args/**kwargs to the actual argument names using inspect,
    and assign each argument to a local variable for use in the postcondition.
    """
    # Extract the function definition for entry_point
    func_def = None
    func_ast = ast.parse(code)
    for node in func_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            func_def = node
            break
    if not func_def:
        return code  # fallback if we can't find the function
    arg_names = [arg.arg for arg in func_def.args.args]
    # Generate argument assignment lines
    assign_lines = [f"    {name} = bound_args.arguments['{name}']" for name in arg_names]
    assign_block = '\n'.join(assign_lines)

    
    postcondition = '\n'.join(['    ' + line for line in postcondition.split('\n')])

    # Create the wrapped version
    wrapped_code = f'''
{code}

def {entry_point}_wrapped(*args, **kwargs):
    import re
    import inspect
    sig = inspect.signature({entry_point})
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
{assign_block}
    return_value = {entry_point}(*args, **kwargs)
{postcondition}
    return return_value
'''
    return wrapped_code.strip()

def evaluate_postcondition(
    problem: Dict[str, Any],
    postcondition: str,
    expected_output: Dict[str, List],
    base_only: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate the postcondition by wrapping it around the original code and testing.
    """
    # Wrap the original code with postcondition
    wrapped_code = wrap_with_postcondition(
        problem['prompt'] + problem['canonical_solution'],
        postcondition,
        problem['entry_point']
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
        fast_check=False
    )
     
    if base_only:
        return True, {
            "base": base_result,
            "plus": None
        }
    
    # Test with plus inputs if needed
    plus_result = untrusted_check(
        wrapped_code,
        problem["plus_input"],
        problem["entry_point"] + "_wrapped",
        expected=expected_output["plus"],
        atol=problem["atol"],
        ref_time=expected_output["plus_time"],
        fast_check=False
    )

    return plus_result[0] == SUCCESS, {
        "base": base_result,
        "plus": plus_result
    }

def double_check_postcondition(postcondition: str, entrypoint: str) -> str:
    """
    Double-check the postcondition to ensure it is a valid Python assertion constraint.
    If `return_value = entrypoint(...)` exists:
        - Remove everything above and including that line,
        - But keep any import statements that appear above it.
    If not, return postcondition unchanged.
    """
    lines = postcondition.strip().splitlines()
    result_lines = []
    assignment_index = None

    # Find the index of the return_value assignment
    for i, line in enumerate(lines):
        if re.match(rf'\s*return_value\s*=\s*{re.escape(entrypoint)}\s*\(.*\)', line):
            assignment_index = i
            break

    if assignment_index is None:
        # No return_value assignment found; return as-is
        return postcondition.strip()

    # Collect import lines above the assignment
    for line in lines[:assignment_index]:
        if re.match(r'\s*(import|from)\s+\S+', line):
            result_lines.append(line)

    # Append everything after the assignment line
    result_lines += lines[assignment_index + 1:]

    return '\n'.join(result_lines).strip()

def get_code(response: str) -> str:
    res = response.split("</think>", 1)[-1].strip("\n")
    # res = res.split("<code>", 1)[-1].strip('\n')
    # res = res.split("</code>", 1)[0].strip('\n')
    pattern = r"```(python|python3)?\n(.*?)\n```"
    # select group with longest content
    matches = re.findall(pattern, res, re.DOTALL)
    if matches:
        code = max(matches, key=lambda x: len(x[1]))[1].strip()
    else:
        code = res.strip()
    code = code.strip("```")
    return code

def evaluate_postcondition_power_single(
    problem: Dict[str, Any],
    postcondition: str,
    entry_point: str,
    n_workers: int = 1,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0
) -> Dict[str, Any]:
    """
    Evaluate the power (completeness) of a postcondition by testing it against buggy codes.
    This is a simplified version of the power evaluation from run_postcondition_evaluation.py.
    """
    # Load buggy codes for this task
    buggy_codes_file = "code_mutants/all_code_mutants_with_bad_output.jsonl.zip"
    if not os.path.exists(buggy_codes_file):
        print(f"Warning: Buggy codes file {buggy_codes_file} not found. Skipping power evaluation.")
        return {
            "num_bopi_run": 0,
            "num_bopi_killed": 0,
            "num_codes_run": 0,
            "num_codes_killed": 0,
            "completeness_score": 0.0
        }
    
    # Load buggy codes
    import zipfile
    buggy_codes = []
    with zipfile.ZipFile(buggy_codes_file, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith('.jsonl'):
                with zip_ref.open(filename) as f:
                    for line in f:
                        buggy_code = json.loads(line.decode('utf-8'))
                        if buggy_code['task_id'] == problem['task_id']:
                            buggy_codes.append(buggy_code)
    
    if len(buggy_codes) == 0:
        print(f"No buggy codes found for task {problem['task_id']}")
        return {
            "num_bopi_run": 0,
            "num_bopi_killed": 0,
            "num_codes_run": 0,
            "num_codes_killed": 0,
            "completeness_score": 0.0
        }
    
    # Sanitize postcondition
    sanitized_postcondition = code_sanitize(postcondition)
    if not sanitized_postcondition:
        print("Postcondition could not be sanitized")
        return {
            "num_bopi_run": 0,
            "num_bopi_killed": 0,
            "num_codes_run": 0,
            "num_codes_killed": 0,
            "completeness_score": 0.0
        }
    
    # Wrap buggy codes with postcondition
    wrapped_codes = []
    for buggy_code in buggy_codes:
        wrapped_code = wrap_code_solution(None, buggy_code['solution'], entry_point, sanitized_postcondition)
        buggy_code["wrapped"] = wrapped_code
        wrapped_codes.append(buggy_code)
    
    # Create postcondition info structure
    postcondition_info = {
        'task_id': problem['task_id'],
        'response_num': 0,  # Single postcondition evaluation
        'entry_point': entry_point,
        'all_time_limits': [10] * 100  # Default time limits
    }
    
    # Create flags-like structure
    class Flags:
        def __init__(self):
            self.min_time_limit = min_time_limit
            self.gt_time_limit_factor = gt_time_limit_factor
            self.i_just_wanna_run = True
    
    flags = Flags()
    
    # Mock print_and_log function
    def print_and_log(msg):
        print(msg)
    
    # Run power evaluation
    try:
        power_results = evaluate_post_condition_power(wrapped_codes, postcondition_info, n_workers, flags, print_and_log)
        
        task_id = problem['task_id']
        if task_id in power_results:
            result = power_results[task_id]
            num_tests_run = result['num_tests_run']
            num_tests_killed = result['num_tests_killed']
            num_codes_run = len(result['test_results'])
            num_codes_killed = len([x for x in result['test_results'] if x[0] == "killed at least one mutant"])
            
            completeness_score = num_tests_killed / num_tests_run if num_tests_run > 0 else 0.0
            
            return {
                "num_bopi_run": num_tests_run,
                "num_bopi_killed": num_tests_killed,
                "num_codes_run": num_codes_run,
                "num_codes_killed": num_codes_killed,
                "completeness_score": completeness_score
            }
        else:
            return {
                "num_bopi_run": 0,
                "num_bopi_killed": 0,
                "num_codes_run": 0,
                "num_codes_killed": 0,
                "completeness_score": 0.0
            }
    except Exception as e:
        print(f"Error in power evaluation: {str(e)}")
        return {
            "num_bopi_run": 0,
            "num_bopi_killed": 0,
            "num_codes_run": 0,
            "num_codes_killed": 0,
            "completeness_score": 0.0
        }

# @traceable
def generate_and_test_postcondition(
    problem: Dict[str, Any],
    expected_output: Dict[str, List],
    max_attempts: int = MAX_ATTEMPTS,
    base_only: bool = False,
    run_power_eval: bool = True
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, list[dict[str, str]], list[str]]:
    """
    Generate and test postconditions until a correct one is found or max attempts reached.
    If the postcondition fails, append the traceback_log to the next LLM prompt in the required format.
    Args:
        problem: The problem dictionary.
        expected_output: The expected outputs for the problem.
        max_attempts: Maximum number of attempts to try.
        base_only: Whether to only test with base inputs.
        run_power_eval: Whether to run power evaluation for final solutions.
    Returns:
        Tuple of (best_postcondition, best_result, success_flag, conversation_history, raw_responses)
    """
    attempts = 0
    best_postcondition = None
    best_result = None
    conversation_history: list[dict[str, str]] = []
    raw_responses: list[str] = []
    
    while attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} for problem {problem['task_id']}")
        
        raw_response = generate_postcondition(problem, conversation_history)
        raw_responses.append(raw_response)
        print(f"Model raw response:\n{raw_response}")
        postcondition = get_code(raw_response)
        print(f"Model Generated postcondition:\n{postcondition}")
        postcondition = double_check_postcondition(postcondition, problem['entry_point'])
        print(f"Postprocessed Generated postcondition:\n{postcondition}")
        
        is_correct, result = evaluate_postcondition(
            problem,
            postcondition,
            expected_output,
            base_only
        )

        if is_correct:
            print("✅ Postcondition passed all tests!")
            if run_power_eval:
                print("🔍 Calculating completeness (power evaluation)...")
                power_results = evaluate_postcondition_power_single(
                    problem,
                    postcondition,
                    problem['entry_point'],
                    n_workers=1
                )
                print(f"Completeness score: {power_results['completeness_score']:.3f}")
                print(f"Tests killed: {power_results['num_bopi_killed']}/{power_results['num_bopi_run']}")
                print(f"Codes killed: {power_results['num_codes_killed']}/{power_results['num_codes_run']}")
                result['power_evaluation'] = power_results
            else:
                print("⏭️ Skipping power evaluation as requested")
            return postcondition, result, True, conversation_history, raw_responses
                    
        logs = [log for log in result['base'][-1] + result['plus'][-1] if log is not None]
        traceback_log = logs[0]
        assert traceback_log is not None, "Traceback log should not be None"

        print(f"Postcondition failed with traceback:\n{traceback_log}")
        
        # Keep track of the best attempt
        if best_postcondition is None or (
            sum(result["base"][1]) > sum(best_result["base"][1])
        ):
            best_postcondition = postcondition
            best_result = result
        
        print("❌ Postcondition failed some tests. Trying again...")
        # Append the conversation block for this attempt
        conversation_history.extend([
            {"role": "assistant", "content": f"""```python
{postcondition}
```"""},
            {"role": "user", "content": f"Postcondition failed. Traceback log:\n{traceback_log}"}
        ])
        # break # this is for run 1 turn only, should be commented out for reproduce result
            
        
        # time.sleep(1)  # Avoid rate limiting
    
    print(f"⚠️ Reached max attempts without finding a perfect postcondition")
    return best_postcondition, best_result, False, conversation_history, raw_responses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="HumanEval task ID to evaluate, e.g., 'HumanEval/20'")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, 
                       help="Maximum attempts to generate a correct postcondition")
    parser.add_argument("--base-only", action="store_true", 
                       help="Only test with base HumanEval inputs")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output JSON file")
    parser.add_argument("--no-power-eval", action="store_true", help="Skip power evaluation (completeness calculation)")
    args = parser.parse_args()
    r"""
task_id success_plus success_base
40    HumanEval/36       failed      success
153   HumanEval/83    timed out      success
162  HumanEval/139    timed out      success
163  HumanEval/160    timed out      success
    """
    if args.task_id in ["HumanEval/36", "HumanEval/83", "HumanEval/139", "HumanEval/160", "HumanEval/32"]:
        print(f"Skipping task {args.task_id} as it is known to have issues.")
        return
    
    # Load problems and expected outputs
    problems = get_human_eval_plus()
    print(f"Loaded {len(problems)} problems from HumanEval+")
    problem = problems[args.task_id]
    
    dataset_hash = get_human_eval_plus_hash()
    expected_output = get_groundtruth(problems, dataset_hash)[args.task_id]
    
    # Generate and test postconditions
    postcondition, result, success, conversation_history, raw_responses = generate_and_test_postcondition(
        problem,
        expected_output,
        args.max_attempts,
        args.base_only,
        not args.no_power_eval
    )
    
    # Save results
    output = {
        "task_id": args.task_id,
        "success": success,
        "postcondition": postcondition,
        "results": result,
        "attempts": len(raw_responses),
        "conversation_history": conversation_history,
        "raw_responses": raw_responses
    }

    # Add power evaluation results if available and not skipped
    if success and result and 'power_evaluation' in result and not args.no_power_eval:
        output['power_evaluation'] = result['power_evaluation']
        print(f"Power evaluation results saved:")
        print(f"  Completeness score: {result['power_evaluation']['completeness_score']:.3f}")
        print(f"  Tests killed: {result['power_evaluation']['num_bopi_killed']}/{result['power_evaluation']['num_bopi_run']}")
        print(f"  Codes killed: {result['power_evaluation']['num_codes_killed']}/{result['power_evaluation']['num_codes_run']}")

    # Use pathlib for output directory and file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"postcondition_results_{args.task_id.replace('/', '_')}.json"
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFinal result for {args.task_id}:")
    print(f"Saved results to {output_path}")
    print(f"Success: {'✅' if success else '❌'}")
    print(f"Postcondition:\n{postcondition}")
    if result and "base" in result and result["base"]:
        print(f"Base tests passed: {sum(result['base'][1])}/{len(result['base'][1])}")
        if not args.base_only and result.get("plus"):
            print(f"Plus tests passed: {sum(result['plus'][1])}/{len(result['plus'][1])}")
    else:
        print("Base tests passed: N/A")
    if success and result and 'power_evaluation' in result and not args.no_power_eval:
        print(f"Completeness score: {result['power_evaluation']['completeness_score']:.3f}")
        print(f"Tests killed: {result['power_evaluation']['num_bopi_killed']}/{result['power_evaluation']['num_bopi_run']}")
        print(f"Codes killed: {result['power_evaluation']['num_codes_killed']}/{result['power_evaluation']['num_codes_run']}")

if __name__ == "__main__":
    main()