#!/bin/bash

conda create -n specmind-evalplus python=3.12 -y
conda activate specmind-evalplus

git clone https://github.com/microsoft/nl-2-postcond.git
mv nl-2-postcond/nl2postcondition-fse2024/nl2postcondition_source_evalplus/* .
pip install -r requirements.txt
rm -rf nl-2-postcond

git clone https://github.com/evalplus/evalplus.git
cd evalplus
git checkout d5397c599c2252cd7ff4ffd4f48d82852fb20ecd
git apply -v ../evalplus-specmind-fork.diff
cd ..

mv evalplus lib/evalplus
pip install -e lib/evalplus
