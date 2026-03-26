
# src/train/hyperparameter_tuning/config.py

# src/train/hyperparameter_tuning/config.py

"""
config.py
---------
Utilities to load MARTA hyperparameter-tuning configs and build the
per-trial training configuration.

Responsibilities:
  - load the tuning YAML
  - apply fixed values shared by all trials
  - sample trial-dependent parameters depending on the selected search mode
"""

import yaml
import copy
from typing import Any, Dict
import optuna

def load_tuning_config(config_path: str) -> Dict[str, Any]:
    """
    Load a tuning YAML file as a plain dictionary.

    The tuning config controls:
      - study metadata
      - storage/output paths
      - search mode
      - fixed training values
      - Optuna search space
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Tuning config must be a YAML dictionary.")

    return cfg

def apply_fixed_overrides(cfg, tune_cfg: Dict[str, Any]):
    """
    Apply fixed values from the tuning config to the base training config.

    Only attributes already present in the training config are overwritten.
    """
    fixed = tune_cfg.get("fixed", {}) or {}
    for key, value in fixed.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg

def apply_architecture_search_space(cfg, tune_cfg: Dict[str, Any], trial: optuna.Trial):
    """
    Sample structural configuration choices for the architecture search mode.

    This mode compares:
      - input_mode
      - fusion (when input_mode == 'stack')
      - head_kind
      - hidden size / dropout for MLP heads
    """
    ss = tune_cfg.get("search_space", {}) or {}

    input_mode = trial.suggest_categorical(
        "input_mode",
        ss.get("input_mode_choices", ["256", "384", "stack"]),
    )
    cfg.input_mode = input_mode

    if input_mode == "stack":
        cfg.fusion = trial.suggest_categorical(
            "fusion",
            ss.get("fusion_for_stack_choices", ["dual", "stack6"]),
        )
    else:
        cfg.fusion = "single"

    head_kind = trial.suggest_categorical(
        "head_kind",
        ss.get("head_kind_choices", ["mlp", "logreg"]),
    )
    cfg.head_kind = head_kind

    if head_kind == "mlp":
        cfg.hidden = trial.suggest_categorical(
            "hidden",
            ss.get("hidden_choices", [64, 128, 256]),
        )
        cfg.dropout = trial.suggest_categorical(
            "dropout",
            ss.get("dropout_choices", [0.3, 0.5]),
        )
    else:
        cfg.hidden = None
        cfg.dropout = 0.0
    return cfg

def apply_finetune_search_space(cfg, tune_cfg: Dict[str, Any], trial: optuna.Trial):
    """
    Sample staged fine-tuning hyperparameters for the finetune search mode.

    This mode keeps the architecture fixed and tunes:
      - dropout
      - staged learning-rate hierarchy
      - stage durations
    """
    ss = tune_cfg.get("search_space", {}) or {}

    cfg.dropout = trial.suggest_categorical(
        "dropout",
        ss.get("dropout_choices", [0.2, 0.3, 0.4, 0.5, 0.6]),
    )

    head_lr_cfg = ss.get("head_lr", {})
    head_lr = trial.suggest_float(
        "head_lr",
        float(head_lr_cfg.get("low", 1e-4)),
        float(head_lr_cfg.get("high", 3e-3)),
        log=bool(head_lr_cfg.get("log", True)),
    )

    last_ratio_cfg = ss.get("last_ratio", {})
    last_ratio = trial.suggest_float(
        "last_ratio",
        float(last_ratio_cfg.get("low", 0.2)),
        float(last_ratio_cfg.get("high", 1.0)),
    )

    rest_ratio_cfg = ss.get("rest_ratio", {})
    rest_ratio = trial.suggest_float(
        "rest_ratio",
        float(rest_ratio_cfg.get("low", 0.1)),
        float(rest_ratio_cfg.get("high", 1.0)),
    )

    cfg.head_lr = head_lr
    cfg.last_lr = head_lr * last_ratio
    cfg.rest_lr = cfg.last_lr * rest_ratio

    cfg.stage1_epochs = trial.suggest_categorical(
        "stage1_epochs",
        ss.get("stage1_epochs_choices", [1, 2, 3, 4, 5, 6]),
    )
    cfg.stage2_epochs = trial.suggest_categorical(
        "stage2_epochs",
        ss.get("stage2_epochs_choices", [1, 2, 3, 4, 5, 6]),
    )
    cfg.stage3_epochs = trial.suggest_categorical(
        "stage3_epochs",
        ss.get("stage3_epochs_choices", [3, 5, 8, 10, 12, 15]),
    )
    return cfg

def build_trial_cfg(base_cfg, tune_cfg: Dict[str, Any], trial: optuna.Trial):
    """
    Build the training config for a single Optuna trial.

    The function starts from the base MARTA training config, applies the fixed
    values defined in the tuning YAML, and then samples the trial-dependent
    hyperparameters according to the selected search mode.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg = apply_fixed_overrides(cfg, tune_cfg)

    search_mode = tune_cfg["search_mode"]

    if search_mode == "architecture":
        cfg = apply_architecture_search_space(cfg, tune_cfg, trial)
    elif search_mode == "finetune":
        cfg = apply_finetune_search_space(cfg, tune_cfg, trial)
    else:
        raise ValueError(f"Unsupported search_mode: {search_mode}")
    return cfg