
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
from .evaluation import evaluate

from src.models import build_model, freeze_for_stage, param_groups
from src.utils.logging import log
from src.utils.seed import set_global_seed
from src.utils.wandb_utils import (
    watch_model_if_needed,
    log_metrics,
    log_checkpoint_artifact,
    batch_logging_enabled,
    batch_log_every_steps,
)

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

def _optimizer_lrs(opt):
    out = {}
    for i, group in enumerate(opt.param_groups):
        out[f"lr/group_{i}"] = group["lr"]
    return out

def _log_epoch_metrics(stage, ep, global_epoch, tr_loss, ev, opt):
    metrics = {
        "train/loss": tr_loss,
        "val/loss": ev["val_loss"],
        "val/auc": ev["auc"],
        "val/prec1": ev["prec1"],
        "val/rec1": ev["rec1"],
        "val/prec0": ev["prec0"],
        "val/rec0": ev["rec0"],
        "train/stage": stage,
        "train/epoch_in_stage": ep,
        "train/global_epoch": global_epoch,

        # métricas separadas por fase
        f"stage{stage}/train_loss": tr_loss,
        f"stage{stage}/val_loss": ev["val_loss"],
        f"stage{stage}/val_auc": ev["auc"],
        f"stage{stage}/val_prec1": ev["prec1"],
        f"stage{stage}/val_rec1": ev["rec1"],
        f"stage{stage}/val_prec0": ev["prec0"],
        f"stage{stage}/val_rec0": ev["rec0"],
    }
    metrics.update(_optimizer_lrs(opt))
    log_metrics(metrics, step=global_epoch)
    
def _run_one_epoch_train(
    *,
    model,
    loader,
    device,
    criterion,
    optimizer,
    stage: int,
    global_step: int,
    do_batch_logging: bool,
    batch_log_freq: int,
    run_name: str,
    epoch_idx: int,
    n_epochs: int,
):
    model.train()
    tr_loss, n_tr = 0.0, 0 # acumuladores de pérdida

    pbar = tqdm(
        loader,
        desc=f"[{run_name}][stage{stage}] Ep {epoch_idx}/{n_epochs}",
        leave=False,
    )

    for imgs, lbl in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y = lbl.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)  # salida cruda del modelo (sin limitar a [0,1]); representa una pre-probabilidad que BCEWithLogitsLoss transforma internamente con sigmoide
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Acumulas pérdida ponderada por número de meustras
        bs = y.numel()
        tr_loss += float(loss.item()) * bs
        n_tr += bs

        global_step += 1
        if do_batch_logging and batch_log_freq > 0 and (global_step % batch_log_freq == 0):
            log_metrics({
                "train/batch_loss": float(loss.item()),
                "train/stage": stage,
                "train/global_step": global_step,
            }, step=global_step)

        pbar.set_postfix({"loss": f"{(tr_loss / max(1, n_tr)):.4f}"})

    tr_loss = tr_loss / max(1, n_tr) # Cálculas pérdida media final de train
    return tr_loss, global_step

def run_training_stage(
    *,
    stage: int,
    n_epochs: int,
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    cfg,
    pos_weight,
    output_dir: Path,
    run_name: str,
    history: dict,
    best_val: float,
    best_stage: str | None,
    best_ckpt_path: Path | None,
    best_val_metrics: Dict[str, Any],
    global_epoch: int,
    global_step: int,
    log_fp: Path,
    do_batch_logging: bool,
    batch_log_freq: int,
    ckpt_filename: str,
):
    """
    Ejecuta un stage completo de entrenamiento:
      - epochs de train
      - validación
      - logging
      - guardado de mejor checkpoint

    Devuelve el estado actualizado del entrenamiento.
    """
    if n_epochs <= 0:
        return {
            "best_val": best_val,
            "best_stage": best_stage,
            "best_ckpt_path": best_ckpt_path,
            "best_val_metrics": best_val_metrics,
            "global_epoch": global_epoch,
            "global_step": global_step,
        }

    for ep in range(1, n_epochs + 1):
        tr_loss, global_step = _run_one_epoch_train(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            stage=stage,
            global_step=global_step,
            do_batch_logging=do_batch_logging,
            batch_log_freq=batch_log_freq,
            run_name=run_name,
            epoch_idx=ep,
            n_epochs=n_epochs,
        )

        ev = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            threshold=cfg.decision_threshold,
            pos_weight=pos_weight,
        ) # Evalúas en validation

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(ev["val_loss"])
        history["val_auc"].append(ev["auc"])
        history["val_prec1"].append(ev["prec1"])
        history["val_rec1"].append(ev["rec1"])
        history["val_prec0"].append(ev["prec0"])
        history["val_rec0"].append(ev["rec0"])

        global_epoch += 1

        _log_epoch_metrics(
            stage=stage,
            ep=ep,
            global_epoch=global_epoch,
            tr_loss=tr_loss,
            ev=ev,
            opt=optimizer,
        )

        log(
            f"[{run_name}][stage{stage}] tr={tr_loss:.4f} val={ev['val_loss']:.4f} "
            f"val AUC={ev['auc']:.3f} val P1={ev['prec1']:.2f} val R1={ev['rec1']:.2f}",
            log_fp,
        )

        if ev["val_loss"] < best_val:
            best_val = ev["val_loss"]
            best_stage = f"stage{stage}"
            best_ckpt_path = output_dir / ckpt_filename
            best_val_metrics = ev.copy()

            torch.save(
                {"model": model.state_dict(), "cfg": asdict(cfg)},
                best_ckpt_path,
            )

            log_checkpoint_artifact(
                ckpt_path=best_ckpt_path,
                run_name=run_name,
                stage=best_stage,
                aliases=["best", f"best-{best_stage}"],
            )

    return {
        "best_val": best_val,
        "best_stage": best_stage,
        "best_ckpt_path": best_ckpt_path,
        "best_val_metrics": best_val_metrics,
        "global_epoch": global_epoch,
        "global_step": global_step,
    }

# main function
def train_model(
    cfg, 
    samples, 
    split_indices, 
    output_dir: Path, 
    log_fp: Path,
    device: torch.device | None = None, 
    run_name: str = "default",
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
    #log_fp = output_dir / f"train_log_{run_name}.txt"

    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]

    y_train = np.array([samples[i]["label"] for i in train_idx], dtype=np.int64)

    # ======================
    # DataLoaders
    # ======================
    log("[train] Building dataloaders", log_fp)
    log(
        f"[train] Loader setup | batch_size={cfg.batch_size} "
        f"num_workers={cfg.num_workers} resize={cfg.resize_to}",
        log_fp,
    )

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
    log("[model] Building model", log_fp)
    log(
        f"[model] head={head_kind} hidden={hidden} dropout={p_drop} "
        f"input={input_mode} fusion={fusion}",
        log_fp,
    )

    model = build_model(
        head_kind=head_kind,
        hidden=hidden,
        p_drop=p_drop,
        input_mode=input_mode,
        fusion=fusion,
        device=device,
    )

    global_epoch = 0
    global_step = 0
    watch_model_if_needed(cfg, model)

    pos_weight = compute_pos_weight(
        y_train=y_train,
        class1_bonus=cfg.class1_bonus,
        device=device
    )

    if pos_weight is not None:
        log(f"[train] Using pos_weight={float(pos_weight.item()):.4f}", log_fp)
    else:
        log("[train] No class weighting applied", log_fp)

    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
        if pos_weight is not None
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
 
    do_batch_logging = batch_logging_enabled(cfg)
    batch_log_freq = batch_log_every_steps(cfg)

    # ==========================================================
    # STAGE 1: backbone congelado + entrena solo la head
    # ==========================================================
    log("[stage1] Starting stage 1: train classification head with frozen backbone", log_fp)

    freeze_for_stage(model, stage=1)
    head_params, _, _ = param_groups(model, stage=1)

    opt = optim.AdamW([
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay}
    ])

    state = run_training_stage(
        stage=1,
        n_epochs=cfg.stage1_epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=opt,
        cfg=cfg,
        pos_weight=pos_weight,
        output_dir=output_dir,
        run_name=run_name,
        history=history,
        best_val=best_val,
        best_stage=best_stage,
        best_ckpt_path=best_ckpt_path,
        best_val_metrics=best_val_metrics,
        global_epoch=global_epoch,
        global_step=global_step,
        log_fp=log_fp,
        do_batch_logging=do_batch_logging,
        batch_log_freq=batch_log_freq,
        ckpt_filename="best_stage1_head.pth",
    )

    best_val = state["best_val"]
    best_stage = state["best_stage"]
    best_ckpt_path = state["best_ckpt_path"]
    best_val_metrics = state["best_val_metrics"]
    global_epoch = state["global_epoch"]
    global_step = state["global_step"]

    log(f"[stage1] Stage 1 completed | best_val_so_far={best_val:.4f}", log_fp)

    # ==========================================================
    # STAGE 2: descongela últimos k bloques del backbone
    # ==========================================================
    log(f"[stage2] Starting stage 2: unfreeze last {cfg.k_unf} backbone block(s)", log_fp)

    freeze_for_stage(model, stage=2, k_unf=cfg.k_unf)
    head_params, last_params, _ = param_groups(model, stage=2, k_unf=cfg.k_unf)

    opt = optim.AdamW([ 
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": last_params, "lr": cfg.last_lr, "weight_decay": cfg.weight_decay},
    ])

    state = run_training_stage(
        stage=2,
        n_epochs=cfg.stage2_epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=opt,
        cfg=cfg,
        pos_weight=pos_weight,
        output_dir=output_dir,
        run_name=run_name,
        history=history,
        best_val=best_val,
        best_stage=best_stage,
        best_ckpt_path=best_ckpt_path,
        best_val_metrics=best_val_metrics,
        global_epoch=global_epoch,
        global_step=global_step,
        log_fp=log_fp,
        do_batch_logging=do_batch_logging,
        batch_log_freq=batch_log_freq,
        ckpt_filename="best_stage2_last.pth",
    )

    best_val = state["best_val"]
    best_stage = state["best_stage"]
    best_ckpt_path = state["best_ckpt_path"]
    best_val_metrics = state["best_val_metrics"]
    global_epoch = state["global_epoch"]
    global_step = state["global_step"]

    log(f"[stage2] Stage 2 completed | best_val_so_far={best_val:.4f}", log_fp)

    # ==========================================================
    # STAGE 3 (opcional): descongela todo el backbone
    # ==========================================================

    if cfg.stage3_epochs > 0:
        log("[stage3] Starting stage 3: full backbone fine-tuning", log_fp)

        freeze_for_stage(model, stage=3)
        head_params, last_params, rest_params = param_groups(model, stage=3)

        opt = optim.AdamW([
            {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
            {"params": last_params, "lr": cfg.last_lr, "weight_decay": cfg.weight_decay},
            {"params": rest_params, "lr": cfg.rest_lr, "weight_decay": cfg.weight_decay},
        ])

        state = run_training_stage(
            stage=3,
            n_epochs=cfg.stage3_epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=opt,
            cfg=cfg,
            pos_weight=pos_weight,
            output_dir=output_dir,
            run_name=run_name,
            history=history,
            best_val=best_val,
            best_stage=best_stage,
            best_ckpt_path=best_ckpt_path,
            best_val_metrics=best_val_metrics,
            global_epoch=global_epoch,
            global_step=global_step,
            log_fp=log_fp,
            do_batch_logging=do_batch_logging,
            batch_log_freq=batch_log_freq,
            ckpt_filename="best_stage3_full.pth",
        )

        best_val = state["best_val"]
        best_stage = state["best_stage"]
        best_ckpt_path = state["best_ckpt_path"]
        best_val_metrics = state["best_val_metrics"]
        global_epoch = state["global_epoch"]
        global_step = state["global_step"]

        log(f"[stage3] Stage 3 completed | best_val_so_far={best_val:.4f}", log_fp)

    log(
        f"[train] Training finished | best_stage={best_stage} best_ckpt={best_ckpt_path}",
        log_fp,
    )
    
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