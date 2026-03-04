# -*- coding: utf-8 -*-
"""
MARTA_MULTIINPUT_SINGLE_TRAIN.py
--------------------------------
Entrena múltiples configuraciones SIN CV (split 90/10 estratificado), comparando:
  1) Cabeza de Regresión Logística (Linear->1)
  2) Cabezas MLP: hidden ∈ {256,128,64} con activación SilU y Dropout ∈ {0.3, 0.5}

Además permite elegir la ENTRADA entre:
  - '256'  : ROI de 256×256 centrada en el bbox → resize 384×384 → 3 canales
  - '384'  : ROI de 384×384 centrada en el bbox → resize 384×384 → 3 canales
  - 'stack': usa **ambas** (256 y 384). Dos opciones de fusión:
        * 'dual'  : dos flujos con backbone compartido; concatena features  → mantiene preentrenamiento 3ch
        * 'stack6': apila ambas imágenes (6 canales) y adapta el primer conv  → más simple pero pierde calce exacto de pesos preentrenados en la 1ª capa

Salidas por corrida:
  - roc_<run>.png (curva AUC)
  - train_log_<run>.txt (paso a paso por época)
  - best_<stage>.pth (checkpoints por etapa)
  - summary_<run>.json (métricas finales)
Además:
  - roc_compare.png (todas las curvas)
  - metrics_summary.csv (tabla resumen)

Requiere: MARTATRAIN8_256_s.py en el mismo directorio (dataset utils y backbones).
"""
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_score, recall_score
import MARTATRAIN8_256_s as base  # utilidades del usuario

# =========================
# ======= CONFIG ==========
# =========================
EXPERIMENTS_DIR = Path('experiments/MARTA_MULTIINPUT_SINGLE')
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class CFG:
    random_seed: int = getattr(base, 'RANDOM_SEED', 42)
    batch_size: int = getattr(base, 'BATCH_SIZE', 16)
    num_workers: int = 4
    val_size: float = 0.10
    test_size: float = 0.10
    # Group split (anti-leakage)
    use_group_split: bool = False
    group_key: str = "image_path"
    # Entradas
    input_mode: str = 'stack'   # '256', '384', 'stack'
    fusion: str = 'dual'        # 'dual' o 'stack6' (sólo si input_mode='stack')
    resize_to: int = 384
    # Etapas
    stage1_epochs: int = 10
    stage2_epochs: int = 10
    stage3_epochs: int = 10
    k_unf: int = 1
    # LRs
    head_lr: float = 1e-3
    last_lr: float = 3e-4
    rest_lr: float = 1e-4 # 5e-5 #
    weight_decay: float = 1e-4
    # Ponderación clase 1
    class1_bonus: float = getattr(base, 'CLASS1_BONUS', 1.1)
    decision_threshold: float = 0.5

cfg = CFG()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# ====== DATASET ==========
# =========================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def _crop_center(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, min(w - size, cx - half))
    y1 = max(0, min(h - size, cy - half))
    return img[y1:y1+size, x1:x1+size]

class MultiInputROIDataset(Dataset):
    def __init__(self, samples: List[Dict[str,Any]], input_mode: str, augment: bool, resize_to: int = 384):
        self.samples = samples
        self.mode = input_mode
        self.augment = augment
        self.resize_to = resize_to
        if augment:
            self.tf = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    p=0.3,
                ),
                A.Resize(resize_to, resize_to, interpolation=1),
            ])
        else:
            self.tf = A.ReplayCompose([
                A.Resize(resize_to, resize_to, interpolation=1),
            ])
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = base.get_image_cached(Path(s['image_path']))
        (x1,y1,x2,y2) = s['bbox']
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if self.mode == '256':
            roi = _crop_center(img, cx, cy, 256)
            out = self.tf(image=roi)
            im = out['image']
            t = torch.from_numpy(im).permute(2,0,1).float() / 255.0
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            label = int(s['label'])
            return t, label

        elif self.mode == '384':
            roi = _crop_center(img, cx, cy, 512) # cambiar
            out = self.tf(image=roi)
            im = out['image']
            t = torch.from_numpy(im).permute(2,0,1).float() / 255.0
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            label = int(s['label'])
            return t, label

        elif self.mode == 'stack':
            roi256 = _crop_center(img, cx, cy, 256)
            roi384 = _crop_center(img, cx, cy, 512) # cambiar
            out = self.tf(image=roi256)
            im256 = out['image']
            im384 = A.ReplayCompose.replay(out['replay'], image=roi384)['image']
            t256 = torch.from_numpy(im256).permute(2,0,1).float() / 255.0
            t384 = torch.from_numpy(im384).permute(2,0,1).float() / 255.0
            t256 = (t256 - IMAGENET_MEAN) / IMAGENET_STD
            t384 = (t384 - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.cat([t256, t384], dim=0)
            label = int(s['label'])
            return t, label
        else:
            raise ValueError("input_mode debe ser '256', '384' o 'stack'")
            
            

# =========================
# ====== MODELOS ==========
# =========================
def build_backbone_3ch():
    return base.build_backbone()

def adapt_first_conv_to_6ch(backbone: nn.Module):
    conv = getattr(backbone, 'conv_stem', None)
    if conv is None or not isinstance(conv, nn.Conv2d):
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d):
                conv = m; break
    if conv is None:
        raise RuntimeError("No encontré Conv2d inicial para adaptar a 6 canales.")
    w = conv.weight.data
    out_ch, in_ch, k1, k2 = w.shape
    if in_ch == 6:  # ya
        return backbone
    if in_ch != 3:
        raise RuntimeError(f"Conv stem in_ch={in_ch}, esperado 3.")
    new_w = torch.zeros((out_ch, 6, k1, k2), dtype=w.dtype, device=w.device)
    new_w[:, :3, :, :] = w
    mean_w = w.mean(dim=1, keepdim=True)
    new_w[:, 3:, :, :] = mean_w.repeat(1, 3, 1, 1)
    with torch.no_grad():
        conv.in_channels = 6
        conv.weight = nn.Parameter(new_w)
    return backbone

class HeadLogistic(nn.Module):
    def __init__(self, in_feats: int):
        super().__init__()
        self.fc = nn.Linear(in_feats, 1)
    def forward(self, x): return self.fc(x).squeeze(1)

class HeadMLP(nn.Module):
    def __init__(self, in_feats: int, hidden: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

class ModelSingleStream(nn.Module):
    def __init__(self, head_type: str, hidden: int|None, p_drop: float|None, fusion: str):
        super().__init__()
        self.fusion = fusion
        self.backbone = build_backbone_3ch()
        if self.fusion == 'stack6':
            self.backbone = adapt_first_conv_to_6ch(self.backbone)
        in_feats = getattr(self.backbone, 'num_features', 1280)
        if head_type == 'logreg':
            self.head = HeadLogistic(in_feats)
        else:
            self.head = HeadMLP(in_feats, hidden, p_drop)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

class ModelDualStream(nn.Module):
    def __init__(self, head_type: str, hidden: int|None, p_drop: float|None):
        super().__init__()
        self.backbone = build_backbone_3ch()
        in_feats = getattr(self.backbone, 'num_features', 1280)
        if head_type == 'logreg':
            self.head = HeadLogistic(in_feats*2)
        else:
            self.head = HeadMLP(in_feats*2, hidden, p_drop)

    def forward(self, x6):
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        feats = torch.cat([f1, f2], dim=1)
        return self.head(feats)

# =========================
# ====== HELPERS ==========
# =========================
def set_seed(seed: int):
    base.set_global_seed(seed)

def compute_pos_weight(y_train: np.ndarray):
    cnt = np.bincount(y_train, minlength=2).astype(np.float64)
    n0, n1 = cnt[0], cnt[1]
    if n0 < 1 or n1 < 1:
        return None
    pw = (n0 / max(1.0, n1)) * float(cfg.class1_bonus)
    return torch.tensor(pw, dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n = 0.0, 0
    crit = nn.BCEWithLogitsLoss(reduction='sum')
    for imgs, lbl in loader:
        imgs = imgs.to(device, non_blocking=True); y = lbl.float().to(device, non_blocking=True)
        logits = model(imgs); loss = crit(logits, y)
        loss_sum += float(loss.item()); n += y.numel()
        logits_all.append(logits.detach().cpu().numpy()); labels_all.append(lbl.numpy())
    if n == 0:
        return {'val_loss': float('nan'), 'auc': float('nan'), 'prec1': 0.0, 'rec1': 0.0, 'prec0': 0.0, 'rec0': 0.0}
    logits = np.concatenate(logits_all); labels = np.concatenate(labels_all).astype(np.int32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    try: auc = roc_auc_score(labels, probs)
    except ValueError: auc = float('nan')
    preds = (probs >= cfg.decision_threshold).astype(np.int32)
    prec1 = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec1  = recall_score(labels, preds, pos_label=1, zero_division=0)
    prec0 = precision_score(labels, preds, pos_label=0, zero_division=0)
    rec0  = recall_score(labels, preds, pos_label=0, zero_division=0)
    return {'val_loss': loss_sum/n, 'auc': auc, 'prec1': prec1, 'rec1': rec1, 'prec0': prec0, 'rec0': rec0}

def log(msg: str, fp: Path):
    print(msg); fp.open('a', encoding='utf-8').write(msg + '\n')

#def make_split(samples: List[Dict[str,Any]], y: np.ndarray):
#    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.random_seed)
#    idx = np.arange(len(samples))
#    tr_idx, va_idx = next(sss.split(idx, y))
#    return tr_idx, va_idx
    
def _get_group_id(s: Dict[str, Any]) -> str:
    """
    Devuelve la entidad/grupo para evitar leakage.
    Default: agrupar por image_path (todas las ROIs de la misma imagen quedan juntas).
    """
    if cfg.group_key in s:
        return str(s[cfg.group_key])
    # fallback ultra defensivo
    return str(s.get("image_path", "UNKNOWN"))

def make_splits(samples: List[Dict[str,Any]], y: np.ndarray):
    """
    Devuelve (tr_idx, va_idx, te_idx).
    Si use_group_split=True:
      - Primero separa grupos para TEST
      - Luego separa grupos (del remanente) para VAL
      - Train = resto
    Estratificación: aproximada a nivel-grupo usando majority label del grupo.
    Si no se puede estratificar (pocos grupos/clase única), cae a GroupShuffleSplit sin estratificar.
    """
    idx = np.arange(len(samples))

    if not cfg.use_group_split:
        # caso simple (sin grupos): estratificado por sample
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed)
        trainval_idx, te_idx = next(sss1.split(idx, y))

        val_rel = cfg.val_size / (1.0 - cfg.test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=cfg.random_seed + 1)
        tr_rel, va_rel = next(sss2.split(trainval_idx, y[trainval_idx]))

        tr_idx = trainval_idx[tr_rel]
        va_idx = trainval_idx[va_rel]
        return tr_idx, va_idx, te_idx

    # --------- Group split ----------
    groups = np.array([_get_group_id(samples[i]) for i in idx])

    # Map: group -> indices
    uniq_g, inv = np.unique(groups, return_inverse=True)

    # label de grupo para "estratificar": majority label dentro del grupo
    g_pos_rate = np.zeros(len(uniq_g), dtype=np.float64)
    for gi in range(len(uniq_g)):
        members = idx[inv == gi]
        g_pos_rate[gi] = float(np.mean(y[members]))
    g_y = (g_pos_rate >= 0.5).astype(int)

    # 1) TEST por grupos
    try:
        sss_g1 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed)
        g_trainval, g_test = next(sss_g1.split(np.arange(len(uniq_g)), g_y))
    except Exception:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed)
        # aquí "X" no importa; pasamos índices de samples y usamos groups
        trainval_idx, te_idx = next(gss1.split(idx, y, groups=groups))
        # 2) VAL por grupos dentro del remanente
        val_rel = cfg.val_size / (1.0 - cfg.test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=cfg.random_seed + 1)
        tr_idx, va_idx = next(gss2.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx]))
        return trainval_idx[tr_idx], trainval_idx[va_idx], te_idx

    # expandir grupos a índices de samples
    test_groups = set(uniq_g[g_test])
    trainval_groups = set(uniq_g[g_trainval])

    te_mask = np.array([g in test_groups for g in groups], dtype=bool)
    te_idx = idx[te_mask]
    trainval_idx = idx[~te_mask]

    # 2) VAL por grupos dentro de trainval
    uniq_g2 = np.array(sorted(list(trainval_groups)))
    # recomputar labels de grupo para trainval
    g2_y = np.array([(g_pos_rate[np.where(uniq_g == g)[0][0]] >= 0.5) for g in uniq_g2], dtype=int)

    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    try:
        sss_g2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=cfg.random_seed + 1)
        g_tr, g_va = next(sss_g2.split(np.arange(len(uniq_g2)), g2_y))
    except Exception:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=cfg.random_seed + 1)
        tr_idx, va_idx = next(gss2.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx]))
        return trainval_idx[tr_idx], trainval_idx[va_idx], te_idx

    va_groups = set(uniq_g2[g_va])
    va_mask = np.array([g in va_groups for g in groups], dtype=bool)
    va_idx = idx[va_mask & (~te_mask)]
    tr_idx = idx[(~va_mask) & (~te_mask)]

    return tr_idx, va_idx, te_idx



def make_loaders(samples, tr_idx, va_idx, input_mode):
    tr_s = [samples[i] for i in tr_idx]; va_s = [samples[i] for i in va_idx]
    train_loader = DataLoader(MultiInputROIDataset(tr_s, input_mode, augment=True, resize_to=cfg.resize_to),
                              batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(MultiInputROIDataset(va_s, input_mode, augment=False, resize_to=cfg.resize_to),
                            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True, drop_last=False)
    return train_loader, val_loader

def build_model(head_kind: str, hidden: int|None, p_drop: float|None, input_mode: str, fusion: str):
    if input_mode in ('256','384'):
        return ModelSingleStream(head_kind, hidden, p_drop, fusion='single').to(device)
    elif input_mode == 'stack':
        if fusion == 'dual':
            return ModelDualStream(head_kind, hidden, p_drop).to(device)
        elif fusion == 'stack6':
            return ModelSingleStream(head_kind, hidden, p_drop, fusion='stack6').to(device)
        else:
            raise ValueError("fusion debe ser 'dual' o 'stack6'")
    else:
        raise ValueError("input_mode inválido")

def freeze_for_stage(model: nn.Module, stage: int):
    if stage == 1:
        if hasattr(base, 'freeze_backbone'): base.freeze_backbone(model)
        else:
            for p in model.backbone.parameters(): p.requires_grad = False
    elif stage == 2:
        if hasattr(base, 'unfreeze_last_k_blocks'): base.unfreeze_last_k_blocks(model, k=cfg.k_unf)
        else:
            for p in model.backbone.parameters(): p.requires_grad = True
    elif stage == 3:
        if hasattr(base, 'unfreeze_all_backbone'): base.unfreeze_all_backbone(model)
        else:
            for p in model.backbone.parameters(): p.requires_grad = True

def param_groups(model: nn.Module, stage: int):
    head_params = list(model.head.parameters())
    last_params: List[nn.Parameter] = []; rest_params: List[nn.Parameter] = []
    if stage == 1:
        pass
    else:
        children = list(model.backbone.children())
        if children:
            last_children = set(children[-cfg.k_unf:])
            for ch in children:
                for p in ch.parameters():
                    if not p.requires_grad: continue
                    (last_params if ch in last_children else rest_params).append(p)
        else:
            for p in model.backbone.parameters():
                if p.requires_grad: rest_params.append(p)
    return head_params, last_params, rest_params

#def train_one_run(base_dir: Path, head_kind: str, hidden: int|None, p_drop: float|None,
#                  input_mode: str, fusion: str, loaders, y_train, va_idx_global, samples):
def train_one_run(base_dir: Path, head_kind: str, hidden: int|None, p_drop: float|None,
                  input_mode: str, fusion: str, loaders, y_train, va_idx_global, te_idx_global, samples):
    run_name = f"{head_kind}{'' if hidden is None else '_'+str(hidden)}{'' if p_drop is None else f'_d{p_drop}'}"
    print(f"\n===== RUN: {run_name} | head={head_kind} | hidden={hidden} | dropout={p_drop} | input={input_mode} | fusion={fusion} =====")
    run_dir = base_dir / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_fp = run_dir / f"train_log_{run_name}.txt"

    train_loader, val_loader = loaders
    model = build_model(head_kind, hidden, p_drop, input_mode, fusion)

    pos_weight = compute_pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': []}; best_val = float('inf')

    # ====== STAGE 1 ======
    freeze_for_stage(model, stage=1)
    head_params, last_params, rest_params = param_groups(model, stage=1)
    opt = optim.AdamW([{'params': head_params, 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay}])

    for ep in range(1, cfg.stage1_epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        pbar = tqdm(
            train_loader,
            desc=f"[{run_name}][stage1] Ep {ep}/{cfg.stage1_epochs}",
            leave=False
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
            pbar.set_postfix({'loss': f"{(tr_loss/max(1, n_tr)):.4f}"})
        tr_loss = tr_loss / max(1, n_tr)
        ev = evaluate(model, val_loader)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(ev['val_loss'])
        log(f"[{run_name}][stage1] tr={tr_loss:.4f} val={ev['val_loss']:.4f} AUC={ev['auc']:.3f} "
            f"P1={ev['prec1']:.2f} R1={ev['rec1']:.2f}", log_fp)
        if ev['val_loss'] < best_val:
            best_val = ev['val_loss']
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, run_dir / 'best_stage1_head.pth')

    # ====== STAGE 2 ======
    freeze_for_stage(model, stage=2)
    head_params, last_params, rest_params = param_groups(model, stage=2)
    opt = optim.AdamW([
        {'params': head_params, 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
        {'params': last_params, 'lr': cfg.last_lr, 'weight_decay': cfg.weight_decay},
    ])

    for ep in range(1, cfg.stage2_epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        pbar = tqdm(
            train_loader,
            desc=f"[{run_name}][stage2] Ep {ep}/{cfg.stage2_epochs}",
            leave=False
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
            pbar.set_postfix({'loss': f"{(tr_loss/max(1, n_tr)):.4f}"})
        tr_loss = tr_loss / max(1, n_tr)
        ev = evaluate(model, val_loader)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(ev['val_loss'])
        log(f"[{run_name}][stage2] tr={tr_loss:.4f} val={ev['val_loss']:.4f} AUC={ev['auc']:.3f} "
            f"P1={ev['prec1']:.2f} R1={ev['rec1']:.2f}", log_fp)
        if ev['val_loss'] < best_val:
            best_val = ev['val_loss']
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, run_dir / 'best_stage2_last.pth')

    # ====== STAGE 3 (opcional) ======
    if cfg.stage3_epochs > 0:
        freeze_for_stage(model, stage=3)
        head_params, last_params, rest_params = param_groups(model, stage=3)
        opt = optim.AdamW([
            {'params': head_params, 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
            {'params': last_params, 'lr': cfg.last_lr, 'weight_decay': cfg.weight_decay},
            {'params': rest_params, 'lr': cfg.rest_lr, 'weight_decay': cfg.weight_decay},
        ])
        for ep in range(1, cfg.stage3_epochs + 1):
            model.train()
            tr_loss, n_tr = 0.0, 0
            pbar = tqdm(
                train_loader,
                desc=f"[{run_name}][stage3] Ep {ep}/{cfg.stage3_epochs}",
                leave=False
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
                pbar.set_postfix({'loss': f"{(tr_loss/max(1, n_tr)):.4f}"})
            tr_loss = tr_loss / max(1, n_tr)
            ev = evaluate(model, val_loader)
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(ev['val_loss'])
            log(f"[{run_name}][stage3] tr={tr_loss:.4f} val={ev['val_loss']:.4f} AUC={ev['auc']:.3f} "
                f"P1={ev['prec1']:.2f} R1={ev['rec1']:.2f}", log_fp)
            if ev['val_loss'] < best_val:
                best_val = ev['val_loss']
                torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, run_dir / 'best_stage3_full.pth')


    # ROC de validación (mismo split global)
#    for ckname in ['best_stage3_full.pth','best_stage2_last.pth','best_stage1_head.pth']:
#        ck = run_dir / ckname
#        if ck.exists():
#            state = torch.load(ck, map_location=device); model.load_state_dict(state['model']); break

#    model.eval(); from sklearn.metrics import roc_curve, roc_auc_score
#    va_samples = [samples[i] for i in va_idx_global]
#    val_loader_full = DataLoader(MultiInputROIDataset(va_samples, input_mode, augment=False, resize_to=cfg.resize_to),
#                                 batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
#    logits_all, labels_all = [], []
#    with torch.no_grad():
#        for imgs, lbl in val_loader_full:
#            imgs = imgs.to(device, non_blocking=True)
#            out = model(imgs)
#            logits_all.append(out.detach().cpu().numpy())
#            labels_all.append(lbl.numpy())
#    logits = np.concatenate(logits_all); labels = np.concatenate(labels_all).astype(np.int32)
#    probs = 1.0 / (1.0 + np.exp(-logits))
#    fpr, tpr, _ = roc_curve(labels, probs); auc = roc_auc_score(labels, probs)

#    fig = plt.figure(figsize=(6,5))
#    plt.plot(fpr, tpr, label=f"{run_name} (AUC={auc:.3f})")
#    plt.plot([0,1],[0,1],'k--',linewidth=1)
#    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f"ROC val | {run_name} | input={input_mode} | fusion={fusion}")
#    plt.legend(loc='lower right', fontsize=8)
#    out_png = run_dir / f"roc_{run_name}.png"
#    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

#    summary = {'run_dir': str(run_dir.resolve()), 'head': head_kind, 'hidden': hidden, 'dropout': p_drop,
#               'input_mode': input_mode, 'fusion': fusion, 'val_auc': float(auc)}
#    (run_dir / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')

#    return {'fpr': fpr, 'tpr': tpr, 'auc': auc, 'label': run_name}
    # ROC en TEST (split global fijo; val se usa para seleccionar checkpoint)
    for ckname in ['best_stage3_full.pth','best_stage2_last.pth','best_stage1_head.pth']:
        ck = run_dir / ckname
        if ck.exists():
            state = torch.load(ck, map_location=device)
            model.load_state_dict(state['model'])
            break

    model.eval()

    te_samples = [samples[i] for i in te_idx_global]
    test_loader_full = DataLoader(
        MultiInputROIDataset(te_samples, input_mode, augment=False, resize_to=cfg.resize_to),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    logits_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbl in test_loader_full:
            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)
            logits_all.append(out.detach().cpu().numpy())
            labels_all.append(lbl.numpy())

    #logits = np.concatenate(logits_all)
    #labels = np.concatenate(labels_all).astype(np.int32)
    #probs = 1.0 / (1.0 + np.exp(-logits))
    logits = np.concatenate(logits_all).reshape(-1)
    labels = np.concatenate(labels_all).astype(np.int32).reshape(-1)
    probs  = (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    # ROC
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        roc_auc = float("nan")

    # PR-AUC (Average Precision)
    try:
        pr_auc = float(average_precision_score(labels, probs))
    except ValueError:
        pr_auc = float("nan")

    # Precision/Recall por clase (depende del umbral)
    thr = float(cfg.decision_threshold)
    preds = (probs >= thr).astype(np.int32)

    prec1 = float(precision_score(labels, preds, pos_label=1, zero_division=0))
    rec1  = float(recall_score(labels, preds, pos_label=1, zero_division=0))
    prec0 = float(precision_score(labels, preds, pos_label=0, zero_division=0))
    rec0  = float(recall_score(labels, preds, pos_label=0, zero_division=0))


    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig = plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{run_name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'k--',linewidth=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f"ROC test | {run_name} | input={input_mode} | fusion={fusion}")
    plt.legend(loc='lower right', fontsize=8)

    out_png = run_dir / f"roc_{run_name}.png"   # mantiene el mismo nombre de archivo
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

    summary = {
        "run_dir": str(run_dir.resolve()),
        "head": head_kind,
        "hidden": hidden,
        "dropout": p_drop,
        "input_mode": input_mode,
        "fusion": fusion,
        "test": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "prec1": prec1, "rec1": rec1,
            "prec0": prec0, "rec0": rec0,
            "threshold": thr,
        }
    }
    (run_dir / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "fpr": fpr, "tpr": tpr,
        "auc": roc_auc,          # para que roc_compare siga funcionando
        "pr_auc": pr_auc,
        "prec1": prec1, "rec1": rec1,
        "prec0": prec0, "rec0": rec0,
        "thr": thr,
        "label": run_name
    }


# =========================
# ========= MAIN ==========
# =========================
if __name__ == "__main__":
    set_seed(cfg.random_seed)
    #samples, y = base.build_samples()
    #tr_idx_global, va_idx_global = make_split(samples, y)
    #y_train = y[tr_idx_global]
    #loaders = make_loaders(samples, tr_idx_global, va_idx_global, cfg.input_mode)
    samples, y = base.build_samples()
    tr_idx_global, va_idx_global, te_idx_global = make_splits(samples, y)
    y_train = y[tr_idx_global]
    loaders = make_loaders(samples, tr_idx_global, va_idx_global, cfg.input_mode)


    # Runs
    runs = [('logreg', None, None)]
    runs = []
    for d in [0.3, 0.5]:
        for h in [256, 128, 64]:
            runs.append(('mlp', h, d))
    #runs = []
    #for d in [0.5]:
    #    for h in [128]:
    #        runs.append(('mlp', h, d))


    curves = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir = EXPERIMENTS_DIR / f"multi_{cfg.input_mode}_{cfg.fusion}_{ts}"
    base_dir.mkdir(parents=True, exist_ok=True)
    # Guardar config
    (base_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')

    for head_kind, hidden, p_drop in runs:
        c = train_one_run(
            base_dir, head_kind, hidden, p_drop,
            cfg.input_mode, cfg.fusion,
            loaders, y_train,
            va_idx_global, te_idx_global,
            samples
        )

        #c = train_one_run(base_dir, head_kind, hidden, p_drop, cfg.input_mode, cfg.fusion,
        #                  loaders, y_train, va_idx_global, samples)
      
        curves.append(c)

    # Comparativa de curvas
    plt.figure(figsize=(8,7))
    for c in curves:
        plt.plot(c['fpr'], c['tpr'], label=f"{c['label']} (AUC={c['auc']:.3f})")
    plt.plot([0,1],[0,1],'k--',linewidth=1)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f"ROC comparación | input={cfg.input_mode} | fusion={cfg.fusion}")
    plt.legend(loc='lower right', fontsize=8)
    out_cmp = base_dir / "roc_compare.png"
    plt.tight_layout(); plt.savefig(out_cmp, dpi=170); plt.close()

    # CSV resumen
    import pandas as pd
    rows = []
    for i, r in enumerate(runs):
        c = curves[i]
        rows.append({
            "head": r[0],
            "hidden": r[1],
            "dropout": r[2],
            "test_roc_auc": c["auc"],
            "test_pr_auc": c["pr_auc"],
            "test_prec1": c["prec1"],
            "test_rec1": c["rec1"],
            "test_prec0": c["prec0"],
            "test_rec0": c["rec0"],
            "thr": c["thr"],
        })
    df = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)  # o test_roc_auc
    df.to_csv(base_dir / "metrics_summary.csv", index=False)



    print("== LISTO ==")
    print("Base dir:", base_dir.resolve())
    print("ROC comparativa:", out_cmp.resolve())
