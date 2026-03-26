
# MARTA Hyperparameter Tuning
This document describes the hyperparameter tuning workflow implemented for the MARTA classifier.

The goal of this module is to optimize selected training settings in a controlled and reproducible way, while reusing the same staged training pipeline used for standard classifier training.

---

## Scope
The tuning workflow is designed to answer two different questions:

1. **Architecture search**  
   Compare a small set of structural alternatives for the MARTA classifier input and head design.

2. **Fine-tuning search**  
   Starting from a fixed base architecture, optimize the staged fine-tuning regime and regularization settings.

Both workflows are orchestrated through the same entry point:

```bash
python -m src.train.tune_classifier --config <tuning_config.yaml>
```

---

## Fixed split policy <!-- aqui hay que ver como dividimos el data al final -->
For a given Optuna study, the dataset is built once and the train/validation split is also generated once.

All trials within the same study are evaluated on the same split.

This is important because it ensures that differences between trials are driven by hyperparameters rather than by changes in the data split.

---

## Search modes
Two search modes are currently supported.

### 1. Architecture search
This mode compares a small number of structural alternatives while keeping the staged training schedule fixed.

Typical variables explored in this mode include:

- `input_mode`
- `fusion (only when input_mode == stack)`
- `head_kind`
- `hidden`
- `dropout`

This mode is intended to answer:
    Which MARTA model/input variant should be used as the base design?

A typical use case is to compare:
- `256`
- `384`
- `stack + dual`
- `stack + stack6`

before launching a more focused fine-tuning study.

### 2. Fine-tuning search
This mode keeps the architecture fixed and optimizes the staged training regime.

Typical variables explored in this mode include:

- `dropout`
- `head_lr`
- `last_ratio`
- `rest_ratio`
- `stage1_epochs`
- `stage2_epochs`
- `stage3_epochs`

This mode is intended to answer:
    Given the chosen MARTA architecture, what is the best staged fine-tuning regime?

---

## Learning-rate hierarchy
The fine-tuning search does not sample the three learning rates independently.

Instead, it uses a structured hierarchy:
- `head_lr` is sampled directly
- `last_lr = head_lr * last_ratio`
- `rest_lr = last_lr * rest_ratio`

This guarantees the following ordering:

`head_lr >= last_lr >= rest_lr`

This is consistent with the staged fine-tuning design of MARTA:

- the classification head is the newest and most plastic part
- the last backbone blocks are adapted with more caution
- the rest of the backbone is updated most conservatively

This makes the search space more stable, more interpretable, and more faithful to the training methodology.

---

## Objective function
The current Optuna objective is:
- **best validation loss** reached during training (minimization).

This is consistent with the current MARTA training pipeline, which already selects the best checkpoint based on validation loss.

Additional validation metrics such as:
- AUC
- precision / recall by class
- best stage

are still stored for downstream manual review.

---

## Pruning
Optuna pruning is supported through the training loop.

When enabled, the training pipeline reports validation loss to Optuna at each global epoch and allows the trial to be pruned early.

This is implemented as an external optimization mechanism and does not change the core MARTA staged training logic.

For initial smoke tests, pruning is typically disabled.

---

## Output structure
Each trial is stored in its own directory under the study output root.

Typical structure:

```text
experiments/marta_optuna/<study_name>/
├── tuning_log.txt
├── optuna.db
├── best_trial_summary.json
├── study_trials.csv
├── optimization_history.png
├── optimization_history.pdf
├── param_importances.png
├── param_importances.pdf
├── edf.png
├── edf.pdf
├── timeline.png
├── timeline.pdf
├── intermediate_values.png
├── intermediate_values.pdf
├── trial_0000/
│   ├── trial_config.json
│   ├── trial_summary.json
│   ├── train_log_trial_0000.txt
│   ├── train_result.json
│   ├── best_stage*.pth
│   └── ...
├── trial_0001/
│   └── ...
```

---

## Study dataframe export
At the end of each study, MARTA exports the full Optuna trials dataframe:

```text
study_trials.csv
```

This file is intended for:
- manual ranking of candidate runs
- inspection of validation metrics
- checking trial states
- comparing parameter combinations outside the Optuna UI

This is an important part of the workflow, because in practice final model selection may involve reviewing several validation metrics, not only the single optimization objective.

---

## Standard Optuna analysis
MARTA also generates a small standard set of Optuna plots:

- optimization history
- parameter importances
- empirical distribution function (EDF)
- timeline
- intermediate values

These are exported in both PNG and PDF format.

The intention is to keep the outputs suitable for:
- project review
- internal discussion
- supplementary material if needed

---

## Configuration files
Two tuning config types are currently supported.

### Architecture search config

Example:
```yaml
configs/hyperparameter_tuning/tune_architecture.yaml
```

### Fine-tuning search config

Example:
```
configs/hyperparameter_tuning/tune_finetune.yaml
```

Each tuning config specifies:
- study metadata
- storage path
- output root
- base training config
- search mode
- pruning policy
- fixed values
- Optuna search space

---

## Running locally

Example:
```bash
python -m src.train.tune_classifier \
    --config configs/hyperparameter_tuning/tune_architecture.yaml
```

or

```bash
python -m src.train.tune_classifier \
    --config configs/hyperparameter_tuning/tune_finetune.yaml
```

## Running on cluster (SLURM)

Example:
```bash
sbatch scripts/hyperparameter_tuning/tune_architecture.sh
```

or

```bash
sbatch scripts/hyperparameter_tuning/tune_finetune.sh
```

For initial validation runs, it is recommended to start with a smoke test configuration using:
- `n_trials: 2`
- pruning disabled

This helps verify that the study:
- is created correctly
- generates per-trial directories
- writes logs
- reuses a fixed split
- returns the objective value correctly

before launching a larger study.

---

## W&B integration
Hyperparameter tuning reuses the same optional expert tracking mechanism used by the standard MARTA training pipeline.

This means that each trial can be tracked in W&B if expert mode is enabled in the base training configuration.

If W&B is enabled, each trial still runs through the normal MARTA training code.
The tuning layer only orchestrates the trial configuration and study logic.

See also:
```text
docs/EXPERT_TRACKING.md
```

---

## Notes
- The tuning workflow does not replace standard training.
- It is intended for systematic experiment comparison and controlled optimization.
- The core training methodology remains the same staged MARTA training regime.
- The goal is to improve reproducibility and experiment organization, not to introduce a separate training framework.

