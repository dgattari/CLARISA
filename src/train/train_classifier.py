
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
from .splits import make_splits
from .trainer import train_model
from .evaluation import evaluate_saved_checkpoint

from src.utils.config import load_train_config, build_train_run_name
from src.utils.seed import set_global_seed
from src.utils.plots import save_roc_curve

EXPERIMENTS_DIR = Path("experiments/MARTA_MULTIINPUT_SINGLE")
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
    # Dataset
    # ======================
    samples, y = build_samples()

    # ======================
    # Splits
    # ======================
    train_idx, val_idx, test_idx = make_splits(
        samples=samples,
        y=y,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_seed=cfg.random_seed,
        use_group_split=cfg.use_group_split,
        group_key=cfg.group_key,
    )

    split_indices = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

    # ======================
    # Output dir
    # ======================
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = build_train_run_name(cfg)
    base_dir = EXPERIMENTS_DIR / f"{run_name}_{ts}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Guardar config usada
    with (base_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Guardar split sizes
    split_info = {
        "n_total": int(len(samples)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    with (base_dir / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # ======================
    # Training
    # ======================
    train_result = train_model(  
        cfg=cfg,
        samples=samples,
        split_indices=split_indices,
        output_dir=base_dir,
        run_name=run_name,
        device=device
    )

    # ======================
    # Evaluación final en test  
    # ======================
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
    # Guardar ROC test
    # ======================
    roc_path = base_dir / "roc_test.png"
    save_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=test_metrics["roc_auc"],
        out_path=roc_path,
        title=f"ROC test | {run_name}",
    )

    # ======================
    # Summary final
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

    with (base_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ======================
    # CSV resumen simple
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

    pd.DataFrame([row]).to_csv(base_dir / "metrics_summary.csv", index=False)

    print("== LISTO ==")
    print("Base dir:", base_dir.resolve())
    print("Best checkpoint:", train_result["best_ckpt"])
    print("Best stage:", train_result["best_stage"])
    print("ROC test:", roc_path.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)