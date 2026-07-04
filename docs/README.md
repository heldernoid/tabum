# tabUM developer documentation

A map of the codebase for anyone who wants to use, modify, or continue
developing tabUM. Each document walks one subsystem in detail, in the order
data flows through the system:

| doc | subsystem | source |
|---|---|---|
| [architecture.md](architecture.md) | the model: stages 0 to 3, both heads, parameter budget | `src/tabum/model/` |
| [generator.md](generator.md) | the synthetic prior: task families, postprocessing, sampling | `src/tabum/generator/` |
| [training.md](training.md) | pretraining loop, dynamic batching, memory safety, monitoring | `src/tabum/train/`, `scripts/pretrain.py` |
| [inference.md](inference.md) | estimators: fit/predict, ensembling, finetune(), explain() | `src/tabum/inference/` |
| [evaluation.md](evaluation.md) | benchmark protocol, suites, baselines, reproducing the tables | `scripts/eval_*.py`, `results/` |
| [development.md](development.md) | environment, tests, conventions, hardware notes, contribution guidelines | everything |
| [ideas.md](ideas.md) | known limitations and the most promising directions to push further | |

## The one-paragraph version

tabUM is an in-context-learning (ICL) transformer for tables. Pretraining
samples an endless stream of synthetic supervised tasks (each one a small
table with a target), and trains the model to predict held-out rows of each
task from the labeled rows in the same forward pass. Nothing is ever trained
on real data. At inference, your dataset takes the place of a synthetic task:
training rows go in as context, test rows come out as predictions. The
sklearn-style wrappers in `src/tabum/inference/` hide all of this behind
`fit()` and `predict()`.

## Reading order for new contributors

1. `architecture.md` sections 1 and 2, to understand what a "task tensor" is.
2. `generator.md`, because the prior is the product: model quality is mostly
   determined by what the generator teaches.
3. `training.md`, especially the memory-safety section if you train on
   unified-memory hardware.
4. `inference.md` before touching any user-facing behavior.
