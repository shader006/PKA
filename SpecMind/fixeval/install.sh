#!/bin/bash

conda activate fixeval

pip install sacrebleu=="1.2.11" openai black langchain langchain-openai langchain-community
pip uninstall typing