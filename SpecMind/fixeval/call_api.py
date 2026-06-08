#!/usr/bin/env python3
"""
FixEval OpenAI API Integration

This module provides functionality to use OpenAI API for code fixing tasks
on the FixEval dataset, replacing the original PLBart/CodeT5 models.
"""

from openai import OpenAI
import json
import time
import argparse
from typing import List, Dict, Any
import os
import sys
sys.path.append("./src")
from src.codegen.preprocessing.lang_processors.python_processor import PythonProcessor

import black

def black_format(code: str) -> str:
    """
    Format Python code using Black and return the formatted string.
    """
    mode = black.Mode()  # default settings, you can tweak like line_length=88, etc.
    try:
        formatted_code = black.format_str(code, mode=mode)
        return formatted_code
    except Exception as e:
        # raise RuntimeError(f"Black failed to format code: {e}")
        print(f"Black failed to format code: {e}")
        return code


class OpenAIFixEvalGenerator:
    """Generator class for fixing buggy code using OpenAI API."""
    
    def __init__(self, model: str = "gpt-4", root_folder: str = "../third_party"):
        """Initialize OpenAI API client for code fixing.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            root_folder: Root folder for the Python processor
        """
        self.model = model
        self.client = OpenAI()  # API key will be read from OPENAI_API_KEY env var
        self.python_processor = PythonProcessor(root_folder=root_folder)
        
    def detokenize_code(self, tokens: List[str]) -> str:
        """Convert FixEval tokenized code back to readable Python code using PythonProcessor.
        
        Args:
            tokens: List of code tokens from FixEval dataset
            
        Returns:
            Readable Python code string
        """
        output = self.python_processor.detokenize_code(tokens)
        return black_format(output)
    
    def create_fix_prompt(self, buggy_code: str, verdict: str | None = None) -> str:
        """Create prompt for fixing buggy code.
        
        Args:
            buggy_code: The buggy Python code to fix
            verdict: Optional error type/verdict information
            
        Returns:
            Formatted prompt string for OpenAI API
        """
        prompt = f"""Fix the following buggy Python code. The code has issues that need to be corrected.

Buggy Code:
```python
{buggy_code}
```
"""
        if verdict:
            prompt += f"\nError Type: {verdict}\n"
            
        prompt += """
Please provide the corrected Python code. Only return the fixed code without explanations.

Fixed Code:
```python"""
        
        return prompt
    
    def generate_fix(self, buggy_code: str, verdict: str | None = None, 
                     num_generations: int = 5) -> List[str]:
        """Generate multiple fixes for buggy code using OpenAI API.
        
        Args:
            buggy_code: The buggy Python code to fix
            verdict: Optional error type information
            num_generations: Number of fix candidates to generate
            
        Returns:
            List of generated code fixes
        """
        prompt = self.create_fix_prompt(buggy_code, verdict)
        
        fixes = []
        for i in range(num_generations):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # Add randomness for diverse generations
                    max_tokens=1024,
                    # stop=["```"]
                )
                
                fix = completion.choices[0].message.content
                if fix is not None:
                    fix = fix.strip()
                else:
                    fix = ""
                fixes.append(fix)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error generating fix {i}: {e}")
                fixes.append("")  # Add empty fix on error
                
        return fixes


def load_fixeval_data(data_path: str) -> List[Dict]:
    """Load FixEval dataset from JSONL file.
    
    Args:
        data_path: Path to the JSONL data file
        
    Returns:
        List of data examples
    """
    data = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def run_openai_generation(data_path: str, output_path: str, 
                         model: str = "gpt-4", num_generations: int = 5,
                         start_idx: int = 0, end_idx: int | None = None,
                         root_folder: str = "../third_party"):
    """Run OpenAI-based code fixing on FixEval dataset.
    
    Args:
        data_path: Path to input JSONL file
        output_path: Path to save results JSON file
        model: OpenAI model name
        num_generations: Number of fix candidates per example
        start_idx: Starting index for processing (for resuming)
        end_idx: Ending index for processing (None for all)
        root_folder: Root folder for the Python processor
        
    Returns:
        List of results
    """
    
    # Initialize generator
    generator = OpenAIFixEvalGenerator(model, root_folder)
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = load_fixeval_data(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Handle start/end indices
    if end_idx is None:
        end_idx = len(data)
    data_subset = data[start_idx:end_idx]
    
    print(f"Processing examples {start_idx} to {end_idx-1}")
    
    results = []
    
    for i, example in enumerate(data_subset):
        idx = start_idx + i
        print(f"Processing example {idx+1}/{len(data)}")
        
        # Convert tokenized code to readable code
        buggy_code = generator.detokenize_code(example['src'])
        target_code = generator.detokenize_code(example['tgt'])
        
        # Generate fixes
        generated_fixes: list[str] = generator.generate_fix(
            buggy_code, 
            example.get('src_verdict', None), 
            num_generations
        )
        
        # Store results in same format as original
        result = {
            'idx': idx,
            'src': buggy_code,
            'tgt': target_code,
            'tgt_id': example['tgt_id'],
            'src_id': example['src_id'],
            'src_verdict': example.get('src_verdict'),
            'generations': generated_fixes
        }
        results.append(result)
        
        # Save periodically to avoid data loss
        if (i + 1) % 10 == 0:
            # mkdir -p the output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Intermediate save: {i+1} examples processed")
    
    # Final save
    # mkdir -p the output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Processed {len(results)} examples")
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="FixEval OpenAI API Integration")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to FixEval JSONL data file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save results JSON file")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="OpenAI model name (default: gpt-4)")
    parser.add_argument("--num_generations", type=int, default=5,
                       help="Number of fix candidates per example (default: 5)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for processing (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending index for processing (default: all)")
    parser.add_argument("--root_folder", type=str, default="../third_party",
                       help="Root folder for Python processor (default: ../third_party)")
    
    args = parser.parse_args()
    
    # Run generation
    results = run_openai_generation(
        data_path=args.data_path,
        output_path=args.output_path,
        model=args.model,
        num_generations=args.num_generations,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        root_folder=args.root_folder
    )
    
    print("Generation completed successfully!")


if __name__ == "__main__":
    main()
