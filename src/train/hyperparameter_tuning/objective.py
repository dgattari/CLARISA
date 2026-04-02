
# src/train/hyperparameter_tuning/objective.py

"""
objective.py
------------
Optuna objective construction for MARTA tuning.

Responsibilities:
  - build the per-trial training config
  - create the trial output directory
  - run the MARTA training pipeline for one trial
  - persist trial-level summaries
  - report trial metadata and validation metrics
  - optionally initialize W&B trial tracking
  - return the trial objective value
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import optuna
import torch

from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.wandb_utils import init_wandb_run, finish_wandb

from .config import build_trial_cfg
from ..trainer import train_model

def trial_summary_dict(trial_cfg, train_result, objective_value: float) -> Dict[str, Any]:
    """
    Build a compact JSON-serializable summary for a completed Optuna trial.
    """
    return {
        "objective": objective_value,
        "best_stage": train_result.get("best_stage"),
        "best_val_metrics": train_result.get("best_val_metrics"),
        "config": asdict(trial_cfg),
    }

def make_objective(
    *,
    tune_cfg: Dict[str, Any],
    base_cfg,
    samples,
    split_indices,
    output_root: Path,
    device: torch.device,
):
    """
    Build the Optuna objective function for the current MARTA tuning study.

    The returned objective:
      - reuses the same fixed dataset split for all trials
      - writes each trial to its own output directory
      - delegates the actual training to train_model(...)
      - stores extra validation metadata in Optuna user attributes
    """
    def objective(trial: optuna.Trial):
        trial_cfg = build_trial_cfg(base_cfg, tune_cfg, trial)

        run_name = f"trial_{trial.number:04d}"
        trial_dir = output_root / run_name
        ensure_dir(trial_dir)

        log_fp = trial_dir / f"train_log_{run_name}.txt"

        log("[optuna] Starting trial", log_fp)
        log(f"[optuna] Trial number: {trial.number}", log_fp)
        log(f"[optuna] Search mode: {tune_cfg['search_mode']}", log_fp)
        log(f"[optuna] Output directory: {trial_dir}", log_fp)

        with (trial_dir / "trial_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(trial_cfg), f, indent=2)

        wandb_project = "MARTA-hyperparameter-tuning"
        wandb_group = tune_cfg["study_name"]
        wandb_job_type = tune_cfg["search_mode"]
        wandb_name = run_name

        final_wandb_summary = None

        try:
            init_wandb_run(
                cfg=trial_cfg,
                run_name=run_name,
                output_dir=trial_dir,
                extra_config={
                    "optuna_study_name": tune_cfg["study_name"],
                    "optuna_trial_number": trial.number,
                    "optuna_search_mode": tune_cfg["search_mode"],
                },
                project_override=wandb_project,
                group_override=wandb_group,
                job_type_override=wandb_job_type,
                name_override=wandb_name,
            )

            train_result = train_model(
                cfg=trial_cfg,
                samples=samples,
                split_indices=split_indices,
                output_dir=trial_dir,
                log_fp=log_fp,
                device=device,
                run_name=run_name,
                trial=trial,
            )

            best_val_metrics = train_result.get("best_val_metrics", {})
            objective_value = float(best_val_metrics["val_loss"])

            trial.set_user_attr("best_stage", train_result.get("best_stage"))
            trial.set_user_attr("val_loss", train_result["best_val_metrics"].get("val_loss"))
            trial.set_user_attr("val_auc", train_result["best_val_metrics"].get("auc"))
            trial.set_user_attr("val_prec1", train_result["best_val_metrics"].get("prec1"))
            trial.set_user_attr("val_rec1", train_result["best_val_metrics"].get("rec1"))
            trial.set_user_attr("val_prec0", train_result["best_val_metrics"].get("prec0"))
            trial.set_user_attr("val_rec0", train_result["best_val_metrics"].get("rec0"))

            summary = trial_summary_dict(trial_cfg, train_result, objective_value)
            with (trial_dir / "trial_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            final_wandb_summary = {
                "objective": objective_value,
                "best_stage": train_result.get("best_stage"),
                "val_loss": best_val_metrics.get("val_loss"),
                "val_auc": best_val_metrics.get("auc"),
                "val_prec1": best_val_metrics.get("prec1"),
                "val_rec1": best_val_metrics.get("rec1"),
                "val_prec0": best_val_metrics.get("prec0"),
                "val_rec0": best_val_metrics.get("rec0"),
            }

            log(f"[optuna] Trial completed | objective={objective_value:.6f}", log_fp)
            return objective_value

        except optuna.TrialPruned:
            log("[optuna] Trial pruned", log_fp)
            raise

        except Exception as e:
            log(f"[optuna] Trial failed due to exception: {e}", log_fp)
            return float("inf")

        finally:
            finish_wandb(summary=final_wandb_summary)
    return objective