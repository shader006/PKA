#!/bin/bash

# Exploratory Multiturn with completeness threshold 90, max turn 4
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 4 \
    --completeness-threshold 90 \
    --output-dir output/test-multiturn/run-with-completeness-threshold_90/max4/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'

# t = 50, u = 4
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 4 \
    --completeness-threshold 50 \
    --output-dir output/multiturn/run-with-completeness-threshold_50/max4/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 50, u = 8
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 8 \
    --completeness-threshold 50 \
    --output-dir output/multiturn/run-with-completeness-threshold_50/max8/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 50, u = 12
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 12 \
    --completeness-threshold 50 \
    --output-dir output/multiturn/run-with-completeness-threshold_50/max12/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 70, u = 4
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 4 \
    --completeness-threshold 70 \
    --output-dir output/multiturn/run-with-completeness-threshold_70/max4/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 70, u = 8
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 8 \
    --completeness-threshold 70 \
    --output-dir output/multiturn/run-with-completeness-threshold_70/max8/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 70, u = 12
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 12 \
    --completeness-threshold 70 \
    --output-dir output/multiturn/run-with-completeness-threshold_70/max12/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 90, u = 4
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 4 \
    --completeness-threshold 90 \
    --output-dir output/multiturn/run-with-completeness-threshold_90/max4/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 90, u = 8
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 8 \
    --completeness-threshold 90 \
    --output-dir output/multiturn/run-with-completeness-threshold_90/max8/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'
# t = 90, u = 12
seq 0 163 | parallel --bar -j 48 '
echo "Processing task ID: {}"
python generate_and_test_postconditions_general.py \
    --mode exploratory \
    --task-id HumanEval/{} \
    --max-turns 12 \
    --completeness-threshold 90 \
    --output-dir output/multiturn/run-with-completeness-threshold_90/max12/ \
    || {
    echo "Error processing task ID: {}" >&2
    exit 1
}
'