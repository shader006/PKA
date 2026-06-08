# SpecMind run on FixEval

## Setup

```bash
conda create -n fixeval python=3.12 -y
conda activate fixeval
git clone https://github.com/FixEval/FixEval_official.git
cd FixEval_official
```

then install the dependencies as described in [FixEval_official/README.md](https://github.com/FixEval/FixEval_official/blob/main/README.md), into environment `fixeval`.
Then install additional dependencies into environment `fixeval` by running `./install.sh`.

Environment `fixeval-runner` is used to run the code.

```bash
conda create -n fixeval-runner python=3.6 -y
conda activate fixeval-runner
```

## Run

Run the following scripts to run the experiments:

```bash
bash bug_detection_fixeval.sh
```