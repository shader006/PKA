import os
import re
import ast
import json
import openai
import argparse

client = openai.OpenAI()

MODEL_NAME = "llama4-scout-instruct-basic"

TEMPERATURE = 0.5

SYSTEM_PROMPT = """
You are a reasoning classifier. You will receive a list of reasoning traces produced by a language model when generating specifications (assertions/postconditions) for Python programs.  
Each reasoning has metadata: {task_id, attempt, content}.  

Your task is to classify each reasoning into one of the following predefined categories:

1. Return Value Type and Range - Checking the type and range of the return value.
2. Check Base Cases - Checking the base cases of a function.
3. Check Edge Cases - Adding assertions that check the function's behavior for edge cases, such as empty inputs or boundary values.
4. Postcondition Combination - Combining multiple postconditions into one.
5. Refine Assertions - Refining previous assertions to make them more concise, meaningful, or comprehensive.
6. Verify Function Behavior - Adding assertions to verify the function's behavior, such as correctness of calculations.
7. Submit final solution - Submitting assertions as final solution without any additional refining or combining. 

Output format (strict JSON):
{
  "categories": [
    {
      "name": "<category_name>",
      "items": [
        {"task_id": "<task_id>", "attempt": <int>}
      ]
    }
  ]
}

Rules:
- Every reasoning trace must be assigned to at least one or more than one of the provided categories.
- If a reasoning trace is assigned to a category, its reasoning must primarily follow the action associated with that category.
- Do not invent new categories. Only use the categories explicitly provided.
- Use the category names exactly as written. Do not rephrase, shorten, or expand them.
"""


def extract_think_content(s: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", s, re.DOTALL)
    return match.group(1) if match else None


def save_llm_json(tmp, output_str, filename="output.json"):
    # Extract content between ```json ... ```
    matches = re.findall(r"```json\s*(.*?)\s*```", output_str, re.DOTALL)
    if not matches:
        raise ValueError("No JSON block found in the output.")

    # Take the latest JSON block
    raw_content = matches[-1]

    # Parse JSON
    data = ast.literal_eval(raw_content)
    for i in range(len(data["categories"])):
        for j in range(len(data["categories"][i]["items"])):
            task_id = data["categories"][i]["items"][j]["task_id"]
            attempt = data["categories"][i]["items"][j]["attempt"]
            for reasoning in tmp:
                if reasoning["task"] == task_id and attempt == reasoning["attempt"]:
                    data["categories"][i]["items"][j]["content"] = reasoning["content"]
                    break

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON saved to {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classify reasoning of LLM.")
    parser.add_argument(
        "--task_id", required=True, help="Choose specific EvalPlus task."
    )
    args = parser.parse_args()

    reasoning_data_folder = (
        "output/new-multiturn/run-with-completeness-threshold_70/max12"
    )

    reasoning_lists = []
    total_attempts = []
    for filename in os.listdir(reasoning_data_folder):
        pth_to_reasoning_data = os.path.join(reasoning_data_folder, filename)
        if os.path.isdir(pth_to_reasoning_data):
            continue
        data = json.load(open(pth_to_reasoning_data, "r"))
        total_attempts.append(len(data["raw_responses"]))
        for turn, model_response in enumerate(data["raw_responses"]):
            model_thinking = extract_think_content(model_response)
            reasoning_lists.append(
                {"task": data["task_id"], "attempt": turn, "content": model_thinking}
            )

    import code

    code.interact(local=locals())

    def prompt_process(reasoning_lists):
        prompt = f"""
        Given the following reasoning metadata: \n
        {chr(10).join(str(x) for x in reasoning_lists)}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=8192,
        )

        raw_content = response.choices[0].message.content.strip()
        return raw_content

    saved_folder = os.path.join(reasoning_data_folder, "reasoning_classification")
    os.makedirs(saved_folder, exist_ok=True)

    tmp = []
    for i in range(len(reasoning_lists)):
        reasoning = reasoning_lists[i]
        task_id = int(reasoning["task"].split("/")[-1])
        tmp.append(reasoning)
        if (
            i < len(reasoning_lists) - 1
            and reasoning["task"] != reasoning_lists[i + 1]["task"]
        ):
            if int(args.task_id) != task_id:
                tmp = []
                continue

            if os.path.exists(
                os.path.join(
                    saved_folder, f"{reasoning['task'].replace('/', '_')}.json"
                )
            ):
                continue
            raw_content = prompt_process(tmp)
            save_llm_json(
                tmp,
                raw_content,
                os.path.join(
                    saved_folder, f"{reasoning['task'].replace('/', '_')}.json"
                ),
            )
            tmp = []
