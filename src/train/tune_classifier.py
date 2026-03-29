# src/train/tune_classifier.py

"""
tune_classifier.py
------------------
Optuna orchestration entry point for MARTA classifier tuning.

Supported search modes:
  - architecture
  - finetune

This script:
  - loads the tuning config
  - loads the base MARTA training config
  - prepares a fixed dataset split for all trials
  - creates or loads the Optuna study
  - runs the optimization loop
  - launches a post-hoc analysis of the study results
"""

import argparse
from pathlib import Path

import torch

from .dataset_builder import build_samples

from src.data import load_split_indices, validate_precomputed_split
from src.utils.config import load_train_config
from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.seed import set_global_seed

from .hyperparameter_tuning.config import load_tuning_config
from .hyperparameter_tuning.study import make_sampler_and_pruner, create_or_load_study
from .hyperparameter_tuning.objective import make_objective
from .hyperparameter_tuning.analysis import run_posthoc_analysis

def parse_args():
    """
    Parse command-line arguments for MARTA hyperparameter tuning.
    """
    parser = argparse.ArgumentParser(
        description="Run Optuna-based tuning for MARTA classifier training."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tuning YAML config file.",
    )
    return parser.parse_args()

def main(config_path: str | Path):
    """
    Run a complete MARTA Optuna study from a tuning YAML config.
    """
    tune_cfg = load_tuning_config(config_path)
    base_cfg = load_train_config(tune_cfg["base_train_config"])

    set_global_seed(int(tune_cfg.get("random_seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(tune_cfg["output_root"])
    ensure_dir(output_root)

    log_fp = output_root / "tuning_log.txt"

    log("[optuna] Loading tuning config", log_fp)
    log(f"[optuna] Tuning config path: {config_path}", log_fp)
    log(f"[optuna] Base training config: {tune_cfg['base_train_config']}", log_fp)
    log(f"[optuna] Search mode: {tune_cfg['search_mode']}", log_fp)
    log(f"[optuna] Output root: {output_root}", log_fp)
    log(f"[optuna] Device: {device}", log_fp)

    log("[optuna] Building fixed dataset", log_fp)
    samples, y = build_samples()
    log(f"[optuna] Samples built: {len(samples)}", log_fp)

    split_dir = tune_cfg.get("split_dir", None)
    if split_dir is None:
        raise ValueError("split_dir must be defined in the tuning YAML config.")

    log("[optuna] Loading precomputed split", log_fp)
    log(f"[optuna] Split dir: {split_dir}", log_fp)

    validate_precomputed_split(samples=samples, split_dir=split_dir)
    split_indices = load_split_indices(split_dir)

    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]
    test_idx = split_indices["test_idx"]

    log(
        f"[optuna] Split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        log_fp,
    )

    sampler, pruner = make_sampler_and_pruner(tune_cfg)
    study, loaded = create_or_load_study(tune_cfg, sampler, pruner)

    if loaded:
        log(
            f"[optuna] Loaded study '{study.study_name}' with {len(study.trials)} existing trials",
            log_fp,
        )
    else:
        log(f"[optuna] Created new study '{study.study_name}'", log_fp)

    objective = make_objective(
        tune_cfg=tune_cfg,
        base_cfg=base_cfg,
        samples=samples,
        split_indices=split_indices,
        output_root=output_root,
        device=device,
    )

    n_trials = int(tune_cfg.get("n_trials", 20))
    timeout_minutes = tune_cfg.get("timeout_minutes", None)
    timeout_seconds = None if timeout_minutes is None else int(timeout_minutes * 60)

    log(f"[optuna] Starting optimization | n_trials={n_trials}", log_fp)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
    )

    log("[optuna] Running post-hoc analysis", log_fp)
    run_posthoc_analysis(study=study, output_dir=output_root)

    best_trial = study.best_trial
    log(f"[optuna] Best trial: {best_trial.number}", log_fp)
    log(f"[optuna] Best objective value: {best_trial.value:.6f}", log_fp)

    print("== OPTUNA LISTO ==")
    print("Study:", study.study_name)
    print("Best trial:", best_trial.number)
    print("Best value:", best_trial.value)
    print("Best params:", best_trial.params)

if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)