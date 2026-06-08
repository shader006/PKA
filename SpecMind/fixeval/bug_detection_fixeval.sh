#!/bin/bash

for turn in 10 5 3; do

    seq 384 | taskset -c 15-29 parallel --bar -j 30 --results output/logs/simple-$turn/ '
    echo "Processing {} simple"
    python -m src.multiturn_postcondition \
        --data-path data/python/processed_with_verdict/test_filtered.jsonl \
        --sample-idx {} \
        --mode simple \
        --max-turns $turn \
        --run-until-catch-bug \
        --run-name simple-$turn \
        --resume \
        --model-name llama4-scout-instruct-basic
    echo "Finished {} simple"
    '

    seq 384 | taskset -c 30-44 parallel --bar -j 30 --results output/logs/retry-$turn/ '
    echo "Processing {} retry"
    python -m src.multiturn_postcondition \
        --data-path data/python/processed_with_verdict/test_filtered.jsonl \
        --sample-idx {} \
        --mode retry \
        --max-turns $turn \
        --run-until-catch-bug \
        --run-name retry-$turn \
        --resume \
        --model-name llama4-scout-instruct-basic
    echo "Finished {} retry"
    '

    seq 384 | taskset -c 30-44 parallel --bar -j 30 --results output/logs/multiturn-$turn/ '
    echo "Processing {} multiturn"
    python -m src.multiturn_postcondition \
        --data-path data/python/processed_with_verdict/test_filtered.jsonl \
        --sample-idx {} \
        --mode multiturn \
        --max-turns $turn \
        --run-until-catch-bug \
        --run-name multiturn-$turn \
        --resume \
        --model-name llama4-scout-instruct-basic
    echo "Finished {} multiturn"
    '

done