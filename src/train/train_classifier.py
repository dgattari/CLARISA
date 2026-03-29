
# src/train/train_classifier.py

"""
train_classifier.py
-------------------
Entrenamiento del clasificador principal de MARTA a partir de una única
configuración definida en YAML.

Este script conserva la lógica general del antiguo
MARTA_MULTIINPUT_SINGLE_TRAIN.py, pero separa responsabilidades en módulos
más pequeños para facilitar:
  - entrenamiento reproducible
  - tuning de hiperparámetros
  - evaluación final en test
  - mantenimiento del código

La configuración actual permite entrenar modelos con:
  1) Cabeza de Regresión Logística (Linear -> 1)
  2) Cabeza MLP con activación SiLU y Dropout configurable

Entradas soportadas:
  - '256'  : ROI 256x256 centrada en el bbox -> resize 384x384 -> 3 canales
  - '384'  : ROI de mayor contexto (crop 512) -> resize 384x384 -> 3 canales
  - 'stack': usa ambas vistas. Dos opciones de fusión:
        * 'dual'  : dos flujos con backbone compartido; concatena features
        * 'stack6': apila ambas imágenes (6 canales) y adapta el primer conv

Salidas por corrida:
  - config.json
  - split_info.json
  - best_stage*.pth
  - train_result.json
  - summary.json
  - roc_test.png
  - metrics_summary.csv

Notas:
  - La implementación reutiliza funciones heredadas del código original de Dani
    siempre que ha sido posible.
  - El objetivo de esta refactorización es organizar el código, no cambiar
    la lógica metodológica del entrenamiento.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .dataset_builder import build_samples
from .trainer import train_model
from .evaluation import evaluate_saved_checkpoint

from src.data import (
    load_split_indices, 
    validate_precomputed_split, 
    make_split_indices,
)
from src.utils.config import load_train_config, build_train_run_name
from src.utils.seed import set_global_seed
from src.utils.plots import save_roc_curve
from src.utils.wandb_utils import (
    init_wandb_run, 
    log_metrics,
    log_artifact_file,
    finish_wandb
)
from src.utils.logging import log

EXPERIMENTS_DIR = Path("experiments/marta_classifier")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MARTA (update name) classifier from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_classifier.yaml",
        help="Path to training config YAML file.",
    )
    return parser.parse_args()

def main(config_path: str | Path = "configs/train_classifier.yaml"):
    cfg = load_train_config(config_path)
    set_global_seed(cfg.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # Output dir + log file
    # ======================
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = build_train_run_name(cfg)

    base_dir = EXPERIMENTS_DIR / f"{run_name}_{ts}"
    base_dir.mkdir(parents=True, exist_ok=True)

    log_fp = base_dir / f"train_log_{run_name}.txt"

    log("[train] Loading training config", log_fp)
    log(f"[train] Config path: {config_path}", log_fp)
    log(f"[train] Device: {device}", log_fp)
    log(f"[io] Run directory: {base_dir}", log_fp)

    # ======================
    # Dataset
    # ======================
    log("[data] Building ROI samples from annotations", log_fp)
    samples, y = build_samples()
    log(f"[data] Total samples built: {len(samples)}", log_fp)

    # ======================
    # Splits
    # ======================
    if getattr(cfg, "use_precomputed_split", False):
        if not getattr(cfg, "split_dir", None):
            raise ValueError(
                "use_precomputed_split=True but split_dir is not set in the training config."
            )

        log("[split] Loading precomputed split", log_fp)
        log(f"[split] Split dir: {cfg.split_dir}", log_fp)

        validate_precomputed_split(samples=samples, split_dir=cfg.split_dir)
        split_indices = load_split_indices(cfg.split_dir)

    else:
        log("[split] Creating train/val/test split from config", log_fp)

        strategy = "grouped" if cfg.use_group_split else "roi_stratified"

        val_size_within_trainval = float(cfg.val_size) / (1.0 - float(cfg.test_size))

        split_indices = make_split_indices(
            samples=samples,
            strategy=strategy,
            random_seed=cfg.random_seed,
            test_size=cfg.test_size,
            val_size_within_trainval=val_size_within_trainval,
            group_key=cfg.group_key,
            manual_split=None,
        )

    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]
    test_idx = split_indices["test_idx"]

    log(
        f"[split] Sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        log_fp,
    )
    log(
        f"[split] Split source: {'precomputed' if getattr(cfg, 'use_precomputed_split', False) else 'runtime'}",
        log_fp,
    )
    log(
        f"[split] Strategy: {'precomputed' if getattr(cfg, 'use_precomputed_split', False) else strategy}",
        log_fp,
    )
    log(
        f"[split] Group split: enabled={cfg.use_group_split} group_key={cfg.group_key}",
        log_fp,
    )

    # ======================
    # Save config / split info
    # ======================
    log("[io] Saving config and split metadata", log_fp)
    with (base_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    split_info = {
        "n_total": int(len(samples)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    with (base_dir / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # ======================
    # W&B init (opcional)
    # ======================
    run = init_wandb_run(
        cfg=cfg,
        run_name=run_name,
        output_dir=base_dir,
        extra_config={"output_dir": str(base_dir.resolve())},
    )

    log_metrics({
        "data/n_total": split_info["n_total"],
        "data/n_train": split_info["n_train"],
        "data/n_val": split_info["n_val"],
        "data/n_test": split_info["n_test"],
    })

    # ======================
    # Training
    # ======================
    log("[train] Starting classifier training", log_fp)
    train_result = train_model(  
        cfg=cfg,
        samples=samples,
        split_indices=split_indices,
        output_dir=base_dir,
        run_name=run_name,
        device=device,
        log_fp=log_fp
    )

    # ========================
    # Final evaluation on test
    # ========================
    log("[eval] Evaluating best checkpoint on test split", log_fp)
    log(f"[eval] Best checkpoint: {train_result['best_ckpt']}", log_fp)

    eval_result = evaluate_saved_checkpoint(
        ckpt_path=train_result["best_ckpt"],
        cfg=cfg,
        samples=samples,
        indices=test_idx,
        device=device
    )

    test_metrics = eval_result["metrics"]
    fpr = eval_result["fpr"]
    tpr = eval_result["tpr"]

    # ======================
    # Save test ROC
    # ======================
    log("[io] Saving ROC curve and final summaries", log_fp)
    roc_path = base_dir / "roc_test.png"
    save_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=test_metrics["roc_auc"],
        out_path=roc_path,
        title=f"ROC test | {run_name}",
    )

    # ======================
    # Final summary
    # ======================
    summary = {
        "run_dir": str(base_dir.resolve()),
        "run_name": run_name,
        "head": cfg.head_kind,
        "hidden": cfg.hidden,
        "dropout": cfg.dropout,
        "input_mode": cfg.input_mode,
        "fusion": cfg.fusion,
        "best_ckpt": train_result["best_ckpt"],
        "best_stage": train_result["best_stage"],
        "best_val_metrics": train_result["best_val_metrics"],
        "test": test_metrics,
        "roc_test_path": str(roc_path),
        "split_info": split_info,
        "config": cfg.__dict__,
    }
    
    summary_path = base_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[io] Summary path: {summary_path}", log_fp)

    # ======================
    # Simple CSV summary
    # ======================
    row = {
        "run_name": run_name,
        "head": cfg.head_kind,
        "hidden": cfg.hidden,
        "dropout": cfg.dropout,
        "input_mode": cfg.input_mode,
        "fusion": cfg.fusion,
        "best_stage": train_result["best_stage"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_prec1": test_metrics["prec1"],
        "test_rec1": test_metrics["rec1"],
        "test_prec0": test_metrics["prec0"],
        "test_rec0": test_metrics["rec0"],
        "threshold": test_metrics["threshold"],
    }

    metrics_csv_path = base_dir / "metrics_summary.csv"
    pd.DataFrame([row]).to_csv(metrics_csv_path, index=False)

    # ======================
    # W&B final logs
    # ======================
    log_metrics({
        "test/roc_auc": test_metrics["roc_auc"],
        "test/pr_auc": test_metrics["pr_auc"],
        "test/prec1": test_metrics["prec1"],
        "test/rec1": test_metrics["rec1"],
        "test/prec0": test_metrics["prec0"],
        "test/rec0": test_metrics["rec0"],
        "test/threshold": test_metrics["threshold"],
    })

    log_artifact_file(base_dir / "config.json", f"{run_name}-config", "config")
    log_artifact_file(base_dir / "split_info.json", f"{run_name}-split-info", "metadata")
    log_artifact_file(base_dir / "train_result.json", f"{run_name}-train-result", "result")
    log_artifact_file(base_dir / "summary.json", f"{run_name}-summary", "result")
    log_artifact_file(metrics_csv_path, f"{run_name}-metrics-summary", "table")
    log_artifact_file(roc_path, f"{run_name}-roc-test", "plot")

    finish_wandb(summary={
        "best_stage": train_result["best_stage"],
        "best_val_loss": train_result["best_val_metrics"].get("val_loss", None),
        "best_val_auc": train_result["best_val_metrics"].get("auc", None),
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
    })

    log("[done] Training pipeline finished successfully", log_fp)
    
    print("== LISTO ==")
    print("Base dir:", base_dir.resolve())
    print("Best checkpoint:", train_result["best_ckpt"])
    print("Best stage:", train_result["best_stage"])
    print("ROC test:", roc_path.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)