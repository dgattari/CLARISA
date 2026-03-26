
# src/train/hyperparameter_tuning/analysis.py

"""
analysis.py
-----------
Post-hoc analysis utilities for MARTA Optuna studies.

Responsibilities:
  - export the full study dataframe
  - save standard Optuna diagnostic plots
  - save a compact summary of the best trial
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as optuna_plt

from src.utils.io import ensure_dir

def _save_current_plot(name: str, output_dir: Path):
    """
    Save the current matplotlib figure as PNG and PDF.
    """
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"

    try:
        plt.tight_layout()
    except Exception:
        pass

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

def export_study_dataframe(study: optuna.Study, output_dir: Path):
    """
    Export the full Optuna trials dataframe to CSV for manual downstream review.
    """
    ensure_dir(output_dir)
    df = study.trials_dataframe()
    df.to_csv(output_dir / "study_trials.csv", index=False)

def save_optuna_plots(study: optuna.Study, output_dir: Path):
    """
    Save a compact set of standard Optuna plots with publication-friendly output.
    """
    ensure_dir(output_dir)

    width_cm = 18
    height_cm = 12
    figsize_inch = (width_cm / 2.54, height_cm / 2.54)

    # Optimization history
    plt.figure(figsize=figsize_inch)
    optuna_plt.plot_optimization_history(study)
    plt.title("Optimization History")
    _save_current_plot("optimization_history", output_dir)

    # Parameter importances
    plt.figure(figsize=figsize_inch)
    optuna_plt.plot_param_importances(study)
    plt.title("Parameter Importances")
    _save_current_plot("param_importances", output_dir)

    # EDF
    plt.figure(figsize=figsize_inch)
    optuna_plt.plot_edf(study)
    plt.title("Empirical Distribution Function")
    _save_current_plot("edf", output_dir)

    # Timeline
    plt.figure(figsize=figsize_inch)
    optuna_plt.plot_timeline(study)
    plt.title("Timeline")
    _save_current_plot("timeline", output_dir)

    # Intermediate values
    plt.figure(figsize=figsize_inch)
    optuna_plt.plot_intermediate_values(study)
    plt.title("Intermediate Values")
    _save_current_plot("intermediate_values", output_dir)

def save_best_trial_summary(study: optuna.Study, output_dir: Path):
    """
    Save a small JSON summary of the best trial in the current study.
    """
    ensure_dir(output_dir)

    best_trial = study.best_trial
    summary = {
        "study_name": study.study_name,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
    }

    with (output_dir / "best_trial_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def run_posthoc_analysis(study: optuna.Study, output_dir: Path):
    """
    Run the standard MARTA post-hoc analysis suite for a completed Optuna study.
    """
    export_study_dataframe(study, output_dir)
    save_optuna_plots(study, output_dir)
    save_best_trial_summary(study, output_dir)