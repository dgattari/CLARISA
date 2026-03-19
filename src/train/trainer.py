
# src/train/trainer.py

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .datasets import make_loaders
from .models import build_model, freeze_for_stage, param_groups
from .evaluation import evaluate

from src.utils.logging import log
from src.utils.seed import set_global_seed

"""
Nota de refactorización
-----------------------
Esta función `train_model` se ha construido reutilizando al máximo la lógica
original de `train_one_run` de Dani.

La idea aquí no ha sido reescribir el entrenamiento, sino separar
responsabilidades para que el mismo motor pueda reutilizarse en tres contextos:

  1) entrenamiento normal del clasificador
  2) tuning de hiperparámetros (Optuna)
  3) entrenamiento final del modelo

Se conserva deliberadamente la estructura original del código:
  - entrenamiento en 3 fases
  - misma lógica de freeze/unfreeze
  - mismos optimizadores por fase
  - misma evaluación sobre validation
  - selección de best checkpoint por val_loss

Lo que se ha sacado fuera de esta función es únicamente la parte de:
  - evaluación final en test
  - dibujo de ROC de test
  - guardado del summary final de test

para que `train_model` sea un bloque reutilizable y no un script cerrado.
"""

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_pos_weight(y_train: np.ndarray, class1_bonus: float, device: torch.device):
    cnt = np.bincount(y_train, minlength=2).astype(np.float64)
    n0, n1 = cnt[0], cnt[1]
    
    if n0 < 1 or n1 < 1:
        return None

    pw = (n0 / max(1.0, n1)) * float(class1_bonus)
    return torch.tensor(pw, dtype=torch.float32, device=device)

def train_model(
    cfg, samples, split_indices, output_dir: Path, 
    device: torch.device | None = None, run_name: str = "default",
) -> Dict[str, Any]:
    """
    Entrena el clasificador usando train/validation.

    Qué hace
    --------
      1. crea loaders de train/val
      2. construye modelo
      3. entrena 3 fases
      4. guarda best checkpoints
      5. devuelve:
          - ruta mejor checkpoint
          - history
          - best val metrics
          - output_dir

    Qué NO hace
    -----------
      - no calcula test por defecto
      - no dibuja ROC de test
      - no asume tuning ni final model
    """
    if device is None:
        device = _get_device()

    head_kind = cfg.head_kind
    hidden = cfg.hidden
    p_drop = cfg.dropout
    input_mode = cfg.input_mode
    fusion = cfg.fusion

    run_name = f"{head_kind}{'' if hidden is None else '_'+str(hidden)}{'' if p_drop is None else f'_d{p_drop}'}"
    print(
        f"\n===== RUN: {run_name} | head={head_kind} | hidden={hidden} "
        f"| dropout={p_drop} | input={input_mode} | fusion={fusion} ====="
    )

    set_global_seed(cfg.random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_fp = output_dir / f"train_log_{run_name}.txt"

    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]

    y_train = np.array([samples[i]["label"] for i in train_idx], dtype=np.int64)

    # ======================
    # Loaders
    # ======================
    train_loader, val_loader = make_loaders(
        samples=samples,
        tr_idx=train_idx,
        va_idx=val_idx,
        input_mode=input_mode,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        resize_to=cfg.resize_to,
    )

    # ======================
    # Model
    # ======================
    model = build_model(
        head_kind=head_kind,
        hidden=hidden,
        p_drop=p_drop,
        input_mode=input_mode,
        fusion=fusion,
        device=device,
    )

    pos_weight = compute_pos_weight(
        y_train=y_train,
        class1_bonus=cfg.class1_bonus,
        device=device
    )

    criterion = (nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    ) # durante el training la loss que retropropagamos es la media del batch, ponderada si hace falta.

    history = {
        "train_loss": [], # Pérdida media en training al final de cada época. Sirve para ver si el modelo está aprendiendo sobre los datos de entrenamiento.
        "val_loss": [], # Pérdida media en validation al final de cada época. Es la métrica que usamos para seleccionar el mejor checkpoint.
        "val_auc": [], # AUC ROC en validation. Mide la capacidad del modelo para separar clase 0 y clase 1 independientemente del umbral.
        "val_prec1": [], # Precisión de la clase 1 en validation. De todas las predicciones que el modelo ha llamado “1”, cuántas eran realmente 1.
        "val_rec1": [], # Recall de la clase 1 en validation. De todos los casos realmente 1, cuántos ha detectado el modelo.
        "val_prec0": [], # Precisión de la clase 0 en validation. De todas las predicciones que el modelo ha llamado “0”, cuántas eran realmente 0.
        "val_rec0": [], # Recall de la clase 0 en validation. De todos los casos realmente 0, cuántos ha detectado el modelo.
    }

    # loss te da estabilidad de entrenamiento
    # AUC te da separación global
    # precision/recall te ayudan a entender el equilibrio entre falsos positivos y falsos negativos

    best_val = float("inf")
    best_ckpt_path = None
    best_stage = None
    best_val_metrics: Dict[str, Any] = {}
 
    # ==========================================================
    # STAGE 1: backbone congelado + entrena solo la head
    # ==========================================================
    freeze_for_stage(model, stage=1)
    head_params, last_params, rest_params = param_groups(model, stage=1)

    opt = optim.AdamW([
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay}
    ])

    for ep in range(1, cfg.stage1_epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0 # acumuladores de pérdida

        pbar = tqdm(
            train_loader,
            desc=f"[{run_name}][stage1] Ep {ep}/{cfg.stage1_epochs}",
            leave=False,
        )

        for imgs, lbl in pbar: ### lo marcado entre ### se repite igual para cada fase. Se podría posteriormente refactorizar a una función llamada run_one_epoch_train() para no ser redundantes.
            imgs = imgs.to(device, non_blocking=True)
            y = lbl.float().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(imgs) # salida cruda del modelo (sin limitar a [0,1]); representa una pre-probabilidad que BCEWithLogitsLoss transforma internamente con sigmoide
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            # Acumulas pérdida ponderada por número de meustras
            bs = y.numel()
            tr_loss += float(loss.item()) * bs
            n_tr += bs

            pbar.set_postfix({"loss": f"{(tr_loss / max(1, n_tr)):.4f}"})

        # Cálculas pérdida media final de train
        tr_loss = tr_loss / max(1, n_tr)
        # Evalúas en validation
        
        # Nota: en evaluación usamos BCEWithLogitsLoss con reduction='sum'
        # para calcular una loss media global por muestra en todo el validation set.
        # Tendría sentido pasar tb pos_weight a evaluate para que el val_loss sea coherente con el training.
        ev = evaluate(model=model, loader=val_loader, device=device, threshold=cfg.decision_threshold) 

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(ev["val_loss"])
        history["val_auc"].append(ev["auc"])
        history["val_prec1"].append(ev["prec1"])
        history["val_rec1"].append(ev["rec1"])
        history["val_prec0"].append(ev["prec0"])
        history["val_rec0"].append(ev["rec0"]) ###

        log(
            f"[{run_name}][stage1] tr={tr_loss:.4f} val={ev['val_loss']:.4f} "
            f"val AUC={ev['auc']:.3f} val P1={ev['prec1']:.2f} val R1={ev['rec1']:.2f}",
            log_fp,
        )

        if ev["val_loss"] < best_val:
            best_val = ev["val_loss"]
            best_stage = "stage1"
            best_ckpt_path = output_dir / "best_stage1_head.pth"
            best_val_metrics = ev.copy()

            torch.save(
                {"model": model.state_dict(), "cfg": asdict(cfg)},
                best_ckpt_path,
            )

    # ==========================================================
    # STAGE 2: descongela últimos k bloques del backbone
    # ==========================================================
    freeze_for_stage(model, stage=2, k_unf=cfg.k_unf)
    head_params, last_params, rest_params = param_groups(model, stage=2, k_unf=cfg.k_unf)

    opt = optim.AdamW([ 
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": last_params, "lr": cfg.last_lr, "weight_decay": cfg.weight_decay},
    ])

    for ep in range(1, cfg.stage2_epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"[{run_name}][stage2] Ep {ep}/{cfg.stage2_epochs}",
            leave=False,
        )

        for imgs, lbl in pbar:
            imgs = imgs.to(device, non_blocking=True)
            y = lbl.float().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            bs = y.numel()
            tr_loss += float(loss.item()) * bs
            n_tr += bs
            pbar.set_postfix({"loss": f"{(tr_loss / max(1, n_tr)):.4f}"})

        tr_loss = tr_loss / max(1, n_tr)
        ev = evaluate(model=model, loader=val_loader, device=device, threshold=cfg.decision_threshold)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(ev["val_loss"])
        history["val_auc"].append(ev["auc"])
        history["val_prec1"].append(ev["prec1"])
        history["val_rec1"].append(ev["rec1"])
        history["val_prec0"].append(ev["prec0"])
        history["val_rec0"].append(ev["rec0"])

        log(
            f"[{run_name}][stage2] tr={tr_loss:.4f} val={ev['val_loss']:.4f} "
            f"val AUC={ev['auc']:.3f} val P1={ev['prec1']:.2f} val R1={ev['rec1']:.2f}",
            log_fp,
        )

        if ev["val_loss"] < best_val:
            best_val = ev["val_loss"]
            best_stage = "stage2"
            best_ckpt_path = output_dir / "best_stage2_last.pth"
            best_val_metrics = ev.copy()

            torch.save(
                {"model": model.state_dict(), "cfg": asdict(cfg)},
                best_ckpt_path,
            )

    # ==========================================================
    # STAGE 3 (opcional): descongela todo el backbone
    # ==========================================================
    if cfg.stage3_epochs > 0:
        freeze_for_stage(model, stage=3)
        head_params, last_params, rest_params = param_groups(model, stage=3)

        opt = optim.AdamW([
            {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
            {"params": last_params, "lr": cfg.last_lr, "weight_decay": cfg.weight_decay},
            {"params": rest_params, "lr": cfg.rest_lr, "weight_decay": cfg.weight_decay},
        ])

        for ep in range(1, cfg.stage3_epochs + 1):
            model.train()
            tr_loss, n_tr = 0.0, 0

            pbar = tqdm(
                train_loader,
                desc=f"[{run_name}][stage3] Ep {ep}/{cfg.stage3_epochs}",
                leave=False,
            )

            for imgs, lbl in pbar:
                imgs = imgs.to(device, non_blocking=True)
                y = lbl.float().to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = model(imgs)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()

                bs = y.numel()
                tr_loss += float(loss.item()) * bs
                n_tr += bs
                pbar.set_postfix({"loss": f"{(tr_loss / max(1, n_tr)):.4f}"})

            tr_loss = tr_loss / max(1, n_tr)
            ev = evaluate(model=model, loader=val_loader, device=device, threshold=cfg.decision_threshold)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(ev["val_loss"])
            history["val_auc"].append(ev["auc"])
            history["val_prec1"].append(ev["prec1"])
            history["val_rec1"].append(ev["rec1"])
            history["val_prec0"].append(ev["prec0"])
            history["val_rec0"].append(ev["rec0"])

            log(
                f"[{run_name}][stage3] tr={tr_loss:.4f} val={ev['val_loss']:.4f} "
                f"val AUC={ev['auc']:.3f} val P1={ev['prec1']:.2f} val R1={ev['rec1']:.2f}",
                log_fp,
            )

            if ev["val_loss"] < best_val:
                best_val = ev["val_loss"]
                best_stage = "stage3"
                best_ckpt_path = output_dir / "best_stage3_full.pth"
                best_val_metrics = ev.copy()

                torch.save(
                    {"model": model.state_dict(), "cfg": asdict(cfg)},
                    best_ckpt_path,
                )

    # ======================
    # Guardado del resultado interno de train
    # ======================
    train_result = {
        "run_name": run_name,
        "output_dir": str(output_dir.resolve()),
        "head": cfg.head_kind,
        "hidden": cfg.hidden,
        "dropout": cfg.dropout,
        "input_mode": cfg.input_mode,
        "fusion": cfg.fusion,
        "best_ckpt": str(best_ckpt_path) if best_ckpt_path is not None else None,
        "best_stage": best_stage,
        "best_val_metrics": best_val_metrics,
        "history": history,
        "config": asdict(cfg),
    }
    
    with (output_dir / "train_result.json").open("w", encoding="utf-8") as f:
        json.dump(train_result, f, indent=2)

    return train_result