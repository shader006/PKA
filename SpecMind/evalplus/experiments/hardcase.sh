#!/bin/bash

# Hardcase Exploratory Multiturn with completeness threshold 90, max turn 4
task_id=104
echo "Processing task ID: $task_id"
python generate_and_test_postconditions_general.py \
    --mode multiturn \
    --task-id HumanEval/$task_id \
    --max-turns 4 \
    --completeness-threshold 90 \
    --output-dir output/hardcase-multiturn/run-with-completeness-threshold_90/max4/ \
    --feedback-buggy-mutant

