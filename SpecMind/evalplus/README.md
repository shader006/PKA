# Setup SpecMind for EvalPlus (HumanEval)

```bash
conda create -n specmind-evalplus python=3.12 -y
conda activate specmind-evalplus
```


## Install EvalPlus library adapted for SpecMind

This is a fork of the EvalPlus library [evalplus](https://github.com/evalplus/evalplus) at commit `d5397c599c2252cd7ff4ffd4f48d82852fb20ecd` that is adapted for the SpecMind framework

```bash
git clone https://github.com/evalplus/evalplus.git
cd evalplus
git checkout d5397c599c2252cd7ff4ffd4f48d82852fb20ecd
git apply -v ../evalplus-specmind-fork.diff
pip install -e .
cd ..
```


then follows the setup instructions in [nlpostcond paper for evalplus](https://github.com/microsoft/nl-2-postcond/tree/main/nl2postcondition-fse2024)

<!-- https://github.com/microsoft/nl-2-postcond/tree/main/nl2postcondition-fse2024/nl2postcondition_source_evalplus -->

```bash
git clone https://github.com/microsoft/nl-2-postcond.git
mv nl-2-postcond/nl2postcondition-fse2024/nl2postcondition_source_evalplus/* .
pip install -r requirements.txt
```

## Run

Run the following scripts to run the experiments:

```bash
bash ./experiments/greedy.sh
bash ./experiments/exploratory.sh
bash ./experiments/single-pass.sh
```
