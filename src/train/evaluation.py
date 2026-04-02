
# src/train/evaluation.py

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_score,
    recall_score,
)

from .datasets import make_eval_loader

from src.models import build_model
from src.utils.io import load_state_dict_safe

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sirve para evaluar un modelo ya cargado en memomria sobre un loader. Uso: dentro del entrenamiento, después de cada época.

# Nota: en evaluación usamos BCEWithLogitsLoss con reduction='sum'
        # para calcular una loss media global por muestra en todo el validation set.
        # Tendría sentido pasar tb pos_weight a evaluate para que el val_loss sea coherente con el training.
        
@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, 
    device: torch.device, threshold: float,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n = 0.0, 0
    
    if pos_weight is not None:
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # emular el train loss
    else:
        crit = nn.BCEWithLogitsLoss(reduction="sum") # version old

    # crit = nn.BCEWithLogitsLoss(reduction='sum') # sumamos la pérdida total de todas las muestras. Luego divides por el total de las muestras y obtienes una media global exacta. 
    # Así obtienes una media global exacta por muestra en todo el validation set, independientemente de cómo estén repartidos los batches.

    for imgs, lbl in loader:
        imgs = imgs.to(device, non_blocking=True)
        y = lbl.float().to(device, non_blocking=True)

        logits = model(imgs); 
        loss = crit(logits, y)
        
        loss_sum += float(loss.item())
        n += y.numel()
        
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(lbl.numpy())

    if n == 0:
        return {'val_loss': float('nan'), 'auc': float('nan'), 'prec1': 0.0, 'rec1': 0.0, 'prec0': 0.0, 'rec0': 0.0}
    
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all).astype(np.int32)

    # Convertir logits en probabilidades aplicando una sigmoide
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    try: 
        auc = roc_auc_score(labels, probs)
    except ValueError: 
        auc = float('nan')
    
    # Aplicar umbral
        # - si prob >= threshold → clase 1
        # - si no → clase 0
    preds = (probs >= threshold).astype(np.int32)

    prec1 = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec1  = recall_score(labels, preds, pos_label=1, zero_division=0)
    prec0 = precision_score(labels, preds, pos_label=0, zero_division=0)
    rec0  = recall_score(labels, preds, pos_label=0, zero_division=0)
    return {'val_loss': loss_sum/n, 'auc': auc, 'prec1': prec1, 'rec1': rec1, 'prec0': prec0, 'rec0': rec0}

# Sirve para:
# - cargar un checkpoint desde disco
# - reconstruir modelo
# - evaluar sobre un conjunto dado

# Uso:
# - evaluación final en test
# - tuning
# - notebooks
# - validaciones posteriores


# eval_result = evaluate_saved_checkpoint(
#         ckpt_path=result["best_ckpt"],
#         cfg=cfg,
#         samples=samples,
#         indices=test_idx,
#         device=device
#     )

@torch.no_grad()
def evaluate_saved_checkpoint(
    ckpt_path: Path, cfg, samples,
    indices, device: torch.device | None = None,
) -> Dict[str, Any]:

    if device is None:
        device = _get_device()

    model = build_model(
        head_kind=cfg.head_kind,
        hidden=cfg.hidden,
        p_drop=cfg.dropout,
        input_mode=cfg.input_mode,
        fusion=cfg.fusion,
        device=device,
    )

    #state = torch.load(ckpt_path, map_location=device)
    state = load_state_dict_safe(ckpt_path, device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = make_eval_loader(
        samples=samples,
        indices=indices,
        input_mode=cfg.input_mode,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        resize_to=cfg.resize_to,
    )

    logits_all, labels_all = [], []

    for imgs, lbl in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)

        logits_all.append(out.detach().cpu().numpy())
        labels_all.append(lbl.numpy())

    logits = np.concatenate(logits_all).reshape(-1)
    labels = np.concatenate(labels_all).astype(np.int32).reshape(-1)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    # ROC
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = float(roc_auc_score(labels, probs))

    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        roc_auc = float("nan")

    # PR-AUC-Average Precision
    try:
        pr_auc = float(average_precision_score(labels, probs))
    except ValueError:
        pr_auc = float("nan")

    # Precision/Recall por clase (depende del umbral)
    thr = float(cfg.decision_threshold)
    preds = (probs >= thr).astype(np.int32)

    prec1 = float(precision_score(labels, preds, pos_label=1, zero_division=0))
    rec1 = float(recall_score(labels, preds, pos_label=1, zero_division=0))
    prec0 = float(precision_score(labels, preds, pos_label=0, zero_division=0))
    rec0 = float(recall_score(labels, preds, pos_label=0, zero_division=0))

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prec1": prec1,
        "rec1": rec1,
        "prec0": prec0,
        "rec0": rec0,
        "threshold": thr,
    }

    return {
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr,
        "labels": labels,
        "probs": probs,
    }
