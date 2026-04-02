
# src/train/hyperparameter_tuning/study.py

"""
study.py
--------
Utilities to create or load an Optuna study for MARTA tuning.

Responsibilities:
  - build the sampler
  - build the pruner
  - create or load a persistent Optuna study
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

from src.utils.io import ensure_dir

def make_sampler_and_pruner(tune_cfg: Dict[str, Any]):
    """
    Create the Optuna sampler and pruner from the tuning config.

    The current MARTA setup uses:
      - TPE sampler
      - optional median pruning
    """
    seed = int(tune_cfg.get("random_seed", 42))
    pruning_cfg = tune_cfg.get("pruning", {}) or {}

    sampler = TPESampler(
        n_startup_trials=int(pruning_cfg.get("n_startup_trials", 10)),
        multivariate=True,
        group=True,
        n_ei_candidates=64,
        seed=seed,
    )

    if bool(pruning_cfg.get("enabled", False)):
        pruner = MedianPruner(
            n_startup_trials=int(pruning_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruning_cfg.get("n_warmup_steps", 3)),
            interval_steps=int(pruning_cfg.get("interval_steps", 1)),
        )
    else:
        pruner = NopPruner()

    return sampler, pruner

def create_or_load_study(
    tune_cfg: Dict[str, Any],
    sampler,
    pruner,
) -> Tuple[optuna.Study, bool]:
    """
    Create a new study or load an existing one from persistent storage.

    Returns:
      - study
      - loaded (True if an existing study was loaded, False if it was created)
    """
    study_name = tune_cfg["study_name"]
    storage_path = Path(tune_cfg["storage_path"])
    ensure_dir(storage_path.parent)

    storage = RDBStorage(url=f"sqlite:///{storage_path}")

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )
        loaded = True

    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            direction=tune_cfg.get("direction", "minimize"),
            sampler=sampler,
            pruner=pruner,
            storage=storage,
        )
        loaded = False
    return study, loaded