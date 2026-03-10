# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:09:06 2025

@author: DGattari
"""

# -*- coding: utf-8 -*-
"""
MARTA_gridcv_train_infer_256.py

Entrenamiento + inferencia en TEST del 10% con barrido de 32 combinaciones:
- CV: {3, 5}
- Dropout: {0.3, 0.5}
- Activación: {ReLU, SiLU}
- Cabeza: {256→Act→Drop→2, 128→…, 64→…, Act→Drop→2}

Características:
- Descubre múltiples pares IM{n}*.spydata ↔ im{n}.tif (misma carpeta)
- Unifica todas las ROIs (target_regions_1_filtered / _2_) en un único dataset
- Split estratificado ROI-level: TEST_SIZE (default 0.10)
- CV sobre el 90% restante
- Reanudación por fold: guarda last.pth (modelo+opt+scaler+scheduler+época) y best_fold*.pt
- Calibración: Temperature Scaling con OOF (entrenamiento)
- Inferencia en TEST: por fold y combinado (con/sin pesos)
- Salidas: ROC con curvas (k folds + combinado) + tablita métricas; Excel TOTAL con "summary_runs" y "per_fold"

IMPORTANTE: Ajustar variables en la sección CONFIG.

Requiere: torch, timm, albumentations, numpy, pandas, scikit-learn, matplotlib, openpyxl.
"""
#from __future__ import annotations
import os
import re
import json
import math
import time
import pickle
import gzip
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score
)
import matplotlib.pyplot as plt

def load_state_dict_safe(path, device):
    # En versiones nuevas de torch: usar weights_only=True.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # torch viejo sin ese flag → modo clásico
        return torch.load(path, map_location=device)


# ===== tweaks de rendimiento globales =====
cv2.setNumThreads(0)  # evitar oversubscription de hilos en OpenCV
torch.backends.cudnn.benchmark = True  # perf óptima con tamaños fijos

# ==========================
# ======== CONFIG ==========
# ==========================
# Directorio raíz con los .spydata y los .tif
DATA_ROOT = Path('.')  # cambiar si es necesario 

# Carpetas del dataset
DATASET_DIR = DATA_ROOT / "dataset"
IMAGES_DIR = DATA_ROOT / "images"

# Directorio de resultados/experimentos
EXPERIMENTS_DIR = Path('experiments/MARTA_256')
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_XLSX = EXPERIMENTS_DIR / 'results_total.xlsx'

# Descubrimiento de datos: patrón exacto de nombres
SPYDATA_REGEX = re.compile(r'^IM(\d+)\s+Dani\s+conexina2\.spydata$', re.IGNORECASE)

# Extensiones de imagen soportadas automáticamente
IMAGE_EXTENSIONS = [".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF"]

#TIF_TEMPLATE   = 'im{n}.tif'  # todo minúscula "im"

# Split TEST
TEST_SIZE = 0.15  # modificable
RANDOM_SEED = 42  # reproducible

# ROI & preproc
FULL_WINDOW = True
ROI_WINDOW_SIZE = 256
RESIZE_TO = 384

# Entrenamiento (editar fácilmente)
BATCH_SIZE = 16            # ↑ subimos batch para mejor throughput en GPU
NUM_WORKERS = 4            # ↓ limitamos workers para evitar duplicar demasiada RAM/IO
PIN_MEMORY = True
#PERSISTENT_WORKERS = NUM_WORKERS > 0
PERSISTENT_WORKERS = False  # ← clave para no acumular procesos/FDs entre folds

EPOCHS = 60
PHASE1_END = 25
PHASE2_END = 35
PHASE3_END = 50

# Optimizador / LRs (grupos discriminativos)
HEAD_LR = 5e-4
LAST_LR = 1e-4
REST_LR = 5e-5
WEIGHT_DECAY = 1e-3

# Loss
LABEL_SMOOTH = 0.05
CLASS1_BONUS = 1.1  # factor a clase 1 para compensar desbalanceo

# Inferencia combinada
USE_WEIGHTS = True         # True: usa pesos por fold ∝ 1/val_loss
USE_TEMPERATURE = True     # True: aplica temperature scaling
DECISION_THRESHOLD = 0.5   # modificable

# Barrido de combinaciones (32)
CV_OPTIONS = [3]
DROPOUT_OPTIONS = [0.3, 0.5]
ACTIVATIONS = ['silu']
HEAD_OPTIONS = ['None', '64','128','256']

# Sanity-check (1–2 épocas) -> ver función sanity_check()
SANITY_EPOCHS = 2

# ==========================
# ======== UTILS ===========
# ==========================

def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_print(run_log_fp: Path | None, *args):
    msg = ' '.join(str(a) for a in args)
    print(msg)
    if run_log_fp is not None:
        with run_log_fp.open('a', encoding='utf-8') as f:
            f.write(msg + '\n')


# ---------- LECTOR ROBUSTO + CACHE POR WORKER ----------
# Normalización a uint8 para imágenes 16-bit u otros formatos

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype(np.uint8)


def read_image_rgb(path: Path) -> np.ndarray:
    """Lee imagen y devuelve **RGB uint8** (H,W,3). Intenta: cv2 COLOR → cv2 UNCHANGED → tifffile → PIL."""
    # 1) OpenCV color
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2) OpenCV unchanged
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 2:
            if img.dtype != np.uint8:
                img = _normalize_to_uint8(img)
            return np.dstack([img, img, img])
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)
        if img.ndim == 2:
            return np.dstack([img, img, img])
        return img

    # 3) tifffile
    try:
        import tifffile as tiff
        arr = tiff.imread(str(path))
        if arr.ndim == 2:
            arr = _normalize_to_uint8(arr)
            return np.dstack([arr, arr, arr])
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8:
                arr = _normalize_to_uint8(arr)
            return arr[..., :3]
        if arr.dtype != np.uint8:
            arr = _normalize_to_uint8(arr)
        return np.dstack([arr, arr, arr])
    except Exception:
        pass

    # 4) PIL
    try:
        from PIL import Image
        arr = np.array(Image.open(str(path)))
        if arr.ndim == 2:
            arr = _normalize_to_uint8(arr)
            return np.dstack([arr, arr, arr])
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8:
                arr = _normalize_to_uint8(arr)
            return arr[..., :3]
        if arr.dtype != np.uint8:
            arr = _normalize_to_uint8(arr)
        return np.dstack([arr, arr, arr])
    except Exception:
        pass

    raise RuntimeError(f"No pude leer imagen con ningún método: {path}")


# Cache simple por proceso (cada worker tiene su propio dict)
_IMG_CACHE: dict[str, np.ndarray] = {}

def get_image_cached(path: Path) -> np.ndarray:
    """Devuelve la imagen RGB uint8 cacheada. Si no está, la carga y la guarda."""
    key = str(path)
    arr = _IMG_CACHE.get(key)
    if arr is None:
        arr = read_image_rgb(path)
        _IMG_CACHE[key] = arr
    return arr


# ---------- Carga de .spydata ----------

def safe_pickle_load(spy_path: Path) -> Any:
    """
    Carga .spydata de forma robusta.
    1) Usa el cargador oficial de Spyder (spyder_kernels.utils.iofuncs.load_dictionary)
    2) Si no está disponible o falla, intenta: pickle directo, gzip+pickle, zip con pkl/json, joblib.
    """
    # 1) Cargador oficial de Spyder
    try:
        from spyder_kernels.utils.iofuncs import load_dictionary
        data, error = load_dictionary(str(spy_path))
        if error:
            raise RuntimeError(error)
        return data
    except Exception:
        pass

    # 2) pickle directo
    try:
        with spy_path.open('rb') as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3) gzip + pickle
    try:
        with gzip.open(spy_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass

    # 4) zipfile con pkl/json dentro
    try:
        import zipfile
        if zipfile.is_zipfile(spy_path):
            with zipfile.ZipFile(spy_path, 'r') as zf:
                names = zf.namelist()
                # prioridad: *.pickle / *.pkl / *.pckl
                for ext in ('.pickle', '.pkl', '.pckl'):
                    cands = [n for n in names if n.lower().endswith(ext)]
                    if cands:
                        with zf.open(cands[0], 'r') as f:
                            return pickle.load(f)
                # luego JSON
                json_cands = [n for n in names if n.lower().endswith('.json')]
                if json_cands:
                    import json as _json
                    with zf.open(json_cands[0], 'r') as f:
                        return _json.load(f)
                # último intento: probar el primer miembro como pickle
                for n in names:
                    try:
                        with zf.open(n, 'r') as f:
                            return pickle.load(f)
                    except Exception:
                        continue
            raise RuntimeError('Zip reconocido pero sin pickle/json legible dentro')
    except Exception:
        pass

    # 5) joblib
    try:
        import joblib
        return joblib.load(spy_path)
    except Exception:
        pass

    # 6) TAR archive (.spydata en algunos casos)
    try:
        import tarfile

        if tarfile.is_tarfile(spy_path):
            with tarfile.open(spy_path, "r") as tar:
                for member in tar.getmembers():

                    if member.isfile():
                        f = tar.extractfile(member)

                        if f is None:
                            continue

                        try:
                            return pickle.load(f)
                        except Exception:
                            try:
                                import json
                                return json.load(f)
                            except Exception:
                                continue

            raise RuntimeError("TAR reconocido pero sin pickle/json legible dentro")

    except Exception:
        pass

    raise RuntimeError(f"No pude cargar {spy_path}: formato .spydata no reconocido (usé spyder_kernels, pickle, gzip, zip, joblib)")

def extract_targets_from_spydata(spy_obj: Any) -> Tuple[List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
    """
    Extrae target_regions_1_filtered y target_regions_2_filtered.
    Acepta dicts, objetos con atributos o estructuras anidadas típicas de Spyder.
    """
    def _find_in_mapping(m: Dict[str, Any], key: str):
        # búsqueda directa
        if key in m:
            return m[key]
        # búsqueda relajada por nombre
        for k in m.keys():
            if key.lower() == k.lower():
                return m[k]
        return None

    tr1 = tr2 = None

    if isinstance(spy_obj, dict):
        # caso 1: variables directamente en el dict
        tr1 = _find_in_mapping(spy_obj, 'target_regions_1_filtered')
        tr2 = _find_in_mapping(spy_obj, 'target_regions_2_filtered')
        # caso 2: dict de namespaces comunes
        if (tr1 is None or tr2 is None):
            for ns_key in ('namespace', 'globals', 'global_ns', 'variables', 'data'):
                ns = spy_obj.get(ns_key, None)
                if isinstance(ns, dict):
                    tr1 = tr1 or _find_in_mapping(ns, 'target_regions_1_filtered')
                    tr2 = tr2 or _find_in_mapping(ns, 'target_regions_2_filtered')
                if tr1 is not None and tr2 is not None:
                    break
    else:
        # objeto con atributos
        tr1 = getattr(spy_obj, 'target_regions_1_filtered', None)
        tr2 = getattr(spy_obj, 'target_regions_2_filtered', None)

    if tr1 is None or tr2 is None:
        # último recurso: buscar en subdicts
        if isinstance(spy_obj, dict):
            for v in spy_obj.values():
                if isinstance(v, dict):
                    if tr1 is None:
                        tr1 = v.get('target_regions_1_filtered', tr1)
                    if tr2 is None:
                        tr2 = v.get('target_regions_2_filtered', tr2)
                if tr1 is not None and tr2 is not None:
                    break

    if tr1 is None or tr2 is None:
        raise RuntimeError('No encontré target_regions_1_filtered / _2_ en el .spydata (intentos: raíz, namespace, atributos, anidados)')

    # normalizo a lista de tuplas int
    def norm_list(L):
        out = []
        for b in L:
            x1, y1, x2, y2 = b
            out.append((int(x1), int(y1), int(x2), int(y2)))
        return out

    return norm_list(tr1), norm_list(tr2)

def discover_image_pairs(data_root: Path) -> List[Tuple[Path, Path]]:
    """
    Devuelve una lista de pares (spydata_path, image_path).

    Empareja cada archivo .spydata en dataset/ con su imagen correspondiente
    en images/, buscando automáticamente formatos compatibles (.png, .tif, .tiff).

    Ejemplo:
        dataset/IM6 Dani conexina2.spydata ↔ images/IM6.png
    """

    pairs = []

    dataset_dir = DATASET_DIR
    images_dir = IMAGES_DIR

    for spy_path in dataset_dir.iterdir():

        if spy_path.is_file() and SPYDATA_REGEX.match(spy_path.name):

            match = SPYDATA_REGEX.match(spy_path.name)
            n = match.group(1)

            image_path = None

            for ext in IMAGE_EXTENSIONS:
                candidate = images_dir / f"IM{n}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is not None:
                pairs.append((spy_path, image_path))

    pairs.sort(key=lambda x: int(SPYDATA_REGEX.match(x[0].name).group(1)))

    if not pairs:
        raise RuntimeError(
            f"No se encontraron pares .spydata ↔ imagen en "
            f"{dataset_dir.resolve()} y {images_dir.resolve()}"
        )

    return pairs

# ==========================
# ===== DATASET/CROP =======
# ==========================
class ROIDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], augment: bool):
        self.samples = samples
        self.augment = augment
        # Transforms
        if augment:
            self.tf = A.Compose([
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_CUBIC),
                # (Opcional) añadir augmentations si luego lo querés ajustar
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # LECTURA CACHEADA EN RAM (por worker)
        img = get_image_cached(Path(s['image_path']))  # RGB uint8
        x1, y1, x2, y2 = s['bbox']
        h, w = img.shape[:2]
        # Centro del bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if FULL_WINDOW:
            half = ROI_WINDOW_SIZE // 2
            x1w = max(0, min(w - ROI_WINDOW_SIZE, cx - half))
            y1w = max(0, min(h - ROI_WINDOW_SIZE, cy - half))
            crop = img[y1w:y1w+ROI_WINDOW_SIZE, x1w:x1w+ROI_WINDOW_SIZE]
        else:
            # usa bbox expandido
            pad = int(0.1 * max(x2-x1, y2-y1))
            x1e = max(0, x1 - pad)
            y1e = max(0, y1 - pad)
            x2e = min(w, x2 + pad)
            y2e = min(h, y2 + pad)
            crop = img[y1e:y2e, x1e:x2e]
        out = self.tf(image=crop)
        tensor = out['image']
        label = int(s['label'])
        return tensor, label

# ==========================
# ======= MODELO ===========
# ==========================

def build_backbone() -> nn.Module:
    # EfficientNetV2-S sin cabeza final (features)
    backbone = create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='avg')
    return backbone


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'silu':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError('Activación no soportada: ' + name)


def build_head(in_features: int, head_type: str, act_name: str, p_drop: float) -> nn.Module:
    act = make_activation(act_name)
    if head_type == '256':
        return nn.Sequential(
            nn.Linear(in_features, 256), act, nn.Dropout(p_drop), nn.Linear(256, 2)
        )
    elif head_type == '128':
        return nn.Sequential(
            nn.Linear(in_features, 128), act, nn.Dropout(p_drop), nn.Linear(128, 2)
        )
    elif head_type == '64':
        return nn.Sequential(
            nn.Linear(in_features, 64), act, nn.Dropout(p_drop), nn.Linear(64, 2)
        )
    elif head_type == 'none':
        # Sólo Act→Drop→Linear(2)
        return nn.Sequential(
            act, nn.Dropout(p_drop), nn.Linear(in_features, 2)
        )
    else:
        raise ValueError('head_type no soportado: ' + head_type)


class Classifier(nn.Module):
    def __init__(self, act_name: str, head_type: str, p_drop: float):
        super().__init__()
        self.backbone = build_backbone()
        in_feats = getattr(self.backbone, 'num_features', 1280)
        self.head = build_head(in_feats, head_type, act_name, p_drop)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


# ===== freeze/unfreeze helpers =====

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True

def set_bn_eval(m: nn.Module):
    # Sólo pone BN en eval (usa running stats), NO toca requires_grad de gamma/beta
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
        m.eval()

def set_bn_train(m: nn.Module):
    # Sólo pone BN en train (actualiza running stats), NO toca requires_grad
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
        m.train()



def freeze_backbone(model: Classifier):
    freeze_module(model.backbone)


def unfreeze_all_backbone(model: Classifier):
    unfreeze_module(model.backbone)


def unfreeze_last_k_blocks(model: Classifier, k: int = 1):
    """Tenta identificar bloques finales de la backbone y liberarlos.
    Implementación genérica: libera últimos k children de self.backbone.
    """
    children = list(model.backbone.children())
    if not children:
        unfreeze_module(model.backbone)
        return
    for p in model.backbone.parameters():
        p.requires_grad = False
    for ch in children[-k:]:
        for p in ch.parameters():
            p.requires_grad = True


def param_groups_for_lrs(model: Classifier, k_unfreeze: int = 1):
    """Devuelve 3 grupos: head, last, rest."""
    head_params = list(model.head.parameters())
    # separar last k blocks vs resto
    last_params: List[nn.Parameter] = []
    rest_params: List[nn.Parameter] = []
    children = list(model.backbone.children())
    if children:
        last_children = set(children[-k_unfreeze:])
        for ch in children:
            for p in ch.parameters():
                if ch in last_children:
                    if p.requires_grad:
                        last_params.append(p)
                else:
                    if p.requires_grad:
                        rest_params.append(p)
    else:
        for p in model.backbone.parameters():
            if p.requires_grad:
                rest_params.append(p)
    return head_params, last_params, rest_params


def set_group_lrs(optimizer: optim.Optimizer, lrs: Tuple[float,float,float], mul: float):
    (h, l, r) = lrs
    for i, base_lr in enumerate([h, l, r]):
        if i < len(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = base_lr * mul


# ==========================
# ======== METRICS =========
# ==========================

def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-9, None)


def probs_from_logits(logits: np.ndarray, temp: float | None) -> np.ndarray:
    if temp is not None:
        logits = logits / max(1e-3, float(temp))
    return softmax_np(logits)


def compute_pr_rec_cm(y_true: np.ndarray, y_prob1: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob1 >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    # Clase 1
    prec1 = tp / max(tp+fp, 1)
    rec1  = tp / max(tp+fn, 1)
    # Clase 0
    prec0 = tn / max(tn+fn, 1)
    rec0  = tn / max(tn+fp, 1)  # == especificidad
    return dict(prec1=prec1, rec1=rec1, prec0=prec0, rec0=rec0)


def add_table_to_ax(ax, rows: List[str], cols: List[str], data: List[List[str]]):
    table = ax.table(cellText=data, rowLabels=rows, colLabels=cols,
                     cellLoc='center', loc='lower right', bbox=[0.45, -0.45, 0.55, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(8)


# ==========================
# ===== TEMPERATURE =========
# ==========================
class TempScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, z):
        return z / self.T.clamp(min=1e-3)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.tensor(logits, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    ts = TempScaling().to(device)
    opt = torch.optim.LBFGS(ts.parameters(), lr=0.01, max_iter=50)
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = ce(ts(z), y)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(ts.T.detach().cpu().item())
    return T


# ==========================
# ======== PIPELINE ========
# ==========================

def build_samples() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Descubre datos y construye samples: list de dicts con (image_path, bbox, label).
    Retorna samples y array y (labels) del mismo largo.
    """
    pairs = discover_image_pairs(DATA_ROOT)
    samples = []
    for spy_path, tif_path in pairs:
        obj = safe_pickle_load(spy_path)
        tr1, tr2 = extract_targets_from_spydata(obj)
        for b in tr1:
            samples.append({'image_path': tif_path, 'bbox': b, 'label': 0, 'img_id': tif_path.name})
        for b in tr2:
            samples.append({'image_path': tif_path, 'bbox': b, 'label': 1, 'img_id': tif_path.name})
    y = np.array([s['label'] for s in samples], dtype=np.int64)
    return samples, y


def split_train_test(samples: List[Dict[str,Any]], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    idx = np.arange(len(samples))
    train_idx, test_idx = next(sss.split(idx, y))
    return train_idx, test_idx


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1e-8, None)
    inv[1] *= float(CLASS1_BONUS)
    w = torch.tensor(inv, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return w


def run_one_fold(fold_id: int, run_dir: Path, train_samples, val_samples, y_train_part, cfg, run_log_fp: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Classifier(act_name=cfg['act'], head_type=cfg['head'], p_drop=cfg['drop'])
    model.to(device)

    # Fase 1: solo head
    freeze_backbone(model)
    model.apply(set_bn_eval)

    head_params, last_params, rest_params = param_groups_for_lrs(model, k_unfreeze=cfg['k_unf'])
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': cfg['HEAD_LR'], 'weight_decay': WEIGHT_DECAY},
        {'params': last_params, 'lr': cfg['LAST_LR'], 'weight_decay': WEIGHT_DECAY},
        {'params': rest_params, 'lr': cfg['REST_LR'], 'weight_decay': WEIGHT_DECAY},
    ])
    cosine_epochs = max(1, cfg['epochs'] - PHASE3_END)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Loss con weights del train global (no del fold val)
    class_w = compute_class_weights(y_train_part)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, weight=class_w)

    # DataLoaders
    train_loader = DataLoader(ROIDataset(train_samples, augment=True), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(ROIDataset(val_samples, augment=False), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            persistent_workers=PERSISTENT_WORKERS)

    fold_dir = run_dir / f'fold{fold_id}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = fold_dir / 'last.pth'
    best_ckpt = fold_dir / f'best_fold{fold_id}.pt'

    start_epoch = 0
    best_val_loss = float('inf')
    # Reanudación
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        # 1) Modelo SIEMPRE
        model.load_state_dict(ckpt['model'])

        # 2) Época / best loss
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
        log_print(run_log_fp, f"[Fold {fold_id}] Reanudando desde época {start_epoch}")

        # 3) Intentar optimizer/scheduler/scaler; si no calzan, resetearlos y seguir
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            log_print(run_log_fp, f"[Fold {fold_id}] WARN optimizer mismatch: {e} -> reseteo optimizer")

        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            log_print(run_log_fp, f"[Fold {fold_id}] WARN scheduler mismatch: {e} -> reseteo scheduler")
            # opcional: avanzar manualmente el scheduler para “alcanzar” la época actual
            if start_epoch > PHASE3_END:
                steps = max(0, start_epoch - PHASE3_END)
                for _ in range(steps):
                    scheduler.step()

        try:
            scaler.load_state_dict(ckpt['scaler'])
        except Exception as e:
            log_print(run_log_fp, f"[Fold {fold_id}] WARN scaler mismatch: {e} -> reseteo scaler")

    def warmup_multiplier(epoch, start, end):
        if epoch < start: return 0.0
        if epoch >= end:  return 1.0
        span = max(1, end - start)
        return float(epoch - start + 1) / float(span)

    for epoch in range(start_epoch, cfg['epochs']):
        # Transiciones de fase
        if epoch == PHASE1_END:
            freeze_backbone(model)
            unfreeze_last_k_blocks(model, k=cfg['k_unf'])
            model.apply(set_bn_eval)
            head_params, last_params, rest_params = param_groups_for_lrs(model, k_unfreeze=cfg['k_unf'])
            optimizer.param_groups[0]['params'] = head_params
            optimizer.param_groups[1]['params'] = last_params
            optimizer.param_groups[2]['params'] = []

        if epoch == PHASE2_END:
            unfreeze_all_backbone(model)
            model.apply(set_bn_eval)
            head_params, last_params, rest_params = param_groups_for_lrs(model, k_unfreeze=cfg['k_unf'])
            optimizer.param_groups[0]['params'] = head_params
            optimizer.param_groups[1]['params'] = last_params
            optimizer.param_groups[2]['params'] = rest_params

        # Warmup / Cosine
        if epoch < PHASE1_END:
            mul = warmup_multiplier(epoch, 0, PHASE1_END)
            set_group_lrs(optimizer, [cfg['HEAD_LR'], cfg['LAST_LR'], cfg['REST_LR']], mul)
        elif epoch < PHASE2_END:
            mul = warmup_multiplier(epoch, PHASE1_END, PHASE2_END)
            set_group_lrs(optimizer, [cfg['HEAD_LR'], cfg['LAST_LR'], cfg['REST_LR']], mul)
        elif epoch < PHASE3_END:
            mul = warmup_multiplier(epoch, PHASE2_END, PHASE3_END)
            set_group_lrs(optimizer, [cfg['HEAD_LR'], cfg['LAST_LR'], cfg['REST_LR']], mul)
        else:
            scheduler.step()

        # Train
        model.train()
        # Mantener siempre BN en eval (Fases 1, 2 y 3)
        model.apply(set_bn_eval)

        run_loss = 0.0
        n_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += float(loss.item())
            n_batches += 1
        tr_loss = run_loss / max(1, n_batches)

        # Val
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += float(loss.item())
                n_val += 1
        val_loss = val_loss / max(1, n_val)

        log_print(run_log_fp, f"[Fold {fold_id}] Ep {epoch+1}/{cfg['epochs']} | tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

        # Guardado best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            log_print(run_log_fp, f"[Fold {fold_id}] ✅ Nuevo best ({best_val_loss:.4f}) -> {best_ckpt.name}")

        # Guardado last (reanudación)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }, last_ckpt)

    # marca completado
    (fold_dir / 'completed.txt').write_text('ok', encoding='utf-8')
    return best_val_loss, best_ckpt


def run_cv_and_test(run_name: str, cfg: Dict[str,Any], samples_all: List[Dict[str,Any]], y_all: np.ndarray):
    set_global_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = EXPERIMENTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_fp = run_dir / 'log.txt'
    log_print(run_log_fp, f"\n===== RUN {run_name} =====")
    log_print(run_log_fp, json.dumps(cfg, indent=2))

    # Split Train/Test
    train_idx, test_idx = split_train_test(samples_all, y_all)
    X_train = [samples_all[i] for i in train_idx]
    y_train = y_all[train_idx]
    X_test  = [samples_all[i] for i in test_idx]
    y_test  = y_all[test_idx]

    # CV
    skf = StratifiedKFold(n_splits=cfg['cv'], shuffle=True, random_state=RANDOM_SEED)

    # OOF para TempScaling
    oof_logits = np.zeros((len(X_train), 2), dtype=np.float32)

    best_losses = []
    best_ckpts: List[Path] = []

    # Para mapear índices de folds a índices globales (sobre X_train)
    train_indices = np.arange(len(X_train))

    for fold_id, (tr, va) in enumerate(skf.split(train_indices, y_train), start=1):
        tr_idx = train_indices[tr]
        va_idx = train_indices[va]
        tr_samples = [X_train[i] for i in tr_idx]
        va_samples = [X_train[i] for i in va_idx]
        y_train_part = y_train[tr_idx]

        best_val_loss, best_ckpt = run_one_fold(
            fold_id, run_dir, tr_samples, va_samples, y_train_part, cfg, run_log_fp
        )
        best_losses.append(best_val_loss)
        best_ckpts.append(best_ckpt)

        # OOF logits del fold (evaluar best en val)
        model = Classifier(act_name=cfg['act'], head_type=cfg['head'], p_drop=cfg['drop']).to(device)
        state = load_state_dict_safe(best_ckpt, device) 
        model.load_state_dict(state)
        model.eval()
        val_loader = DataLoader(ROIDataset(va_samples, augment=False), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                persistent_workers=PERSISTENT_WORKERS)
        ptr = 0
        logits_val = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model(inputs)
                logits_val.append(out.detach().cpu().numpy())
        logits_val = np.concatenate(logits_val, axis=0)
        oof_logits[va_idx] = logits_val

    # Pesos por fold según best val loss
    w = 1.0 / (np.array(best_losses, dtype=np.float32) + 1e-9)
    w = (w / w.sum()).astype(np.float32)
    np.save(run_dir / 'fold_weights.npy', w)

    # Temperature Scaling con OOF
    T = None
    if USE_TEMPERATURE:
        T = fit_temperature(oof_logits, y_train)
        np.save(run_dir / 'temperature.npy', np.array([T], dtype=np.float32))
        log_print(run_log_fp, f"Temperature T={T:.3f}")

    # ====== Inferencia en TEST: por fold + combinado ======
    test_loader = DataLoader(ROIDataset(X_test, augment=False), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WORKERS)

    # Per-fold predictions
    prob1_per_fold: List[np.ndarray] = []
    auc_per_fold: List[float] = []
    prrec_per_fold: List[Dict[str,float]] = []

    for fold_id, ck in enumerate(best_ckpts, start=1):
        model = Classifier(act_name=cfg['act'], head_type=cfg['head'], p_drop=cfg['drop']).to(device)
        state = load_state_dict_safe(ck, device)
        model.load_state_dict(state)
        model.eval()
        logits_list = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model(inputs)
                logits_list.append(out.detach().cpu().numpy())
        logits = np.concatenate(logits_list, axis=0)
        probs = probs_from_logits(logits, temp=T)
        prob1 = probs[:, 1]
        prob1_per_fold.append(prob1)
        auc = roc_auc_score(y_test, prob1)
        auc_per_fold.append(auc)
        pr = compute_pr_rec_cm(y_test, prob1, DECISION_THRESHOLD)
        prrec_per_fold.append(pr)

    # Combined
    if USE_WEIGHTS:
        stacked = np.stack(prob1_per_fold, axis=1)
        prob1_comb = (stacked * w[None, :]).sum(axis=1)
    else:
        prob1_comb = np.mean(np.stack(prob1_per_fold, axis=1), axis=1)

    auc_comb = roc_auc_score(y_test, prob1_comb)
    pr_comb = compute_pr_rec_cm(y_test, prob1_comb, DECISION_THRESHOLD)

    # ====== ROC Plot ======
    plt.figure(figsize=(8, 7))
    # Curvas por fold
    for i, prob1 in enumerate(prob1_per_fold, start=1):
        fpr, tpr, _ = roc_curve(y_test, prob1)
        plt.plot(fpr, tpr, label=f"Fold {i} (AUC={auc_per_fold[i-1]:.3f})")
    # Combinado
    fpr_c, tpr_c, _ = roc_curve(y_test, prob1_comb)
    lbl_comb = "Combinado (AUC={:.3f}{})".format(
        auc_comb, ", w" if USE_WEIGHTS else "")
    plt.plot(fpr_c, tpr_c, linewidth=2.5, label=lbl_comb)

    plt.plot([0,1], [0,1], 'k--', linewidth=1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f"ROC – TEST | {run_name}")
    plt.legend(loc='lower right', fontsize=8)

    # Tablita (folds + combinado)
    rows = [f"Fold {i}" for i in range(1, cfg['cv']+1)] + ["Combinado"]
    cols = ["AUC", "Prec1", "Rec1", "Prec0", "Rec0"]
    data = []
    for i in range(cfg['cv']):
        pr = prrec_per_fold[i]
        data.append([
            f"{auc_per_fold[i]:.3f}", f"{pr['prec1']:.2f}", f"{pr['rec1']:.2f}",
            f"{pr['prec0']:.2f}", f"{pr['rec0']:.2f}"
        ])
    data.append([
        f"{auc_comb:.3f}", f"{pr_comb['prec1']:.2f}", f"{pr_comb['rec1']:.2f}",
        f"{pr_comb['prec0']:.2f}", f"{pr_comb['rec0']:.2f}"
    ])
    add_table_to_ax(plt.gca(), rows, cols, data)

    fig_path = run_dir / 'roc_test.png'
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(fig_path, dpi=150)
    plt.close()
    log_print(run_log_fp, f"Guardado ROC: {fig_path}")

    # ====== Excel TOTAL ======
    summary_row = {
        'run_name': run_name,
        'cv': cfg['cv'], 'dropout': cfg['drop'], 'activation': cfg['act'], 'head': cfg['head'],
        'use_weights': int(USE_WEIGHTS), 'use_temperature': int(USE_TEMPERATURE), 'T': (T if T is not None else np.nan),
        'auc_combined': auc_comb,
        'prec1_combined': pr_comb['prec1'], 'rec1_combined': pr_comb['rec1'],
        'prec0_combined': pr_comb['prec0'], 'rec0_combined': pr_comb['rec0'],
    }

    per_fold_rows = []
    for i in range(cfg['cv']):
        pr = prrec_per_fold[i]
        per_fold_rows.append({
            'run_name': run_name, 'fold': i+1,
            'cv': cfg['cv'], 'dropout': cfg['drop'], 'activation': cfg['act'], 'head': cfg['head'],
            'auc': auc_per_fold[i], 'prec1': pr['prec1'], 'rec1': pr['rec1'], 'prec0': pr['prec0'], 'rec0': pr['rec0']
        })

    # Guardar incrementalmente (sobreescribir con acumulado)
    if RESULTS_XLSX.exists():
        try:
            ex_summary = pd.read_excel(RESULTS_XLSX, sheet_name='summary_runs')
            ex_perfold = pd.read_excel(RESULTS_XLSX, sheet_name='per_fold')
        except Exception:
            ex_summary = pd.DataFrame()
            ex_perfold = pd.DataFrame()
    else:
        ex_summary = pd.DataFrame()
        ex_perfold = pd.DataFrame()

    df_sum_new = pd.DataFrame([summary_row])
    df_pf_new = pd.DataFrame(per_fold_rows)

    df_sum_all = pd.concat([ex_summary, df_sum_new], ignore_index=True)
    df_pf_all = pd.concat([ex_perfold, df_pf_new], ignore_index=True)

    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode='w') as writer:
        df_sum_all.to_excel(writer, index=False, sheet_name='summary_runs')
        df_pf_all.to_excel(writer, index=False, sheet_name='per_fold')
    log_print(run_log_fp, f"Actualizado Excel TOTAL: {RESULTS_XLSX}")

    # Guardar metadatos de la corrida
    meta = {
        'config': cfg,
        'best_val_losses': best_losses,
        'fold_weights': w.tolist(),
        'temperature': float(T) if T is not None else None,
        'roc_path': str(fig_path),
    }
    (run_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


# ==========================
# ======== DRIVER ==========
# ==========================

def run_all_experiments(limit: int | None = None, override_epochs: int | None = None):
    start_time = time.time()
    samples_all, y_all = build_samples()

    combos = []
    for cvv in CV_OPTIONS:
        for drop in DROPOUT_OPTIONS:
            for act in ACTIVATIONS:
                for head in HEAD_OPTIONS:
                    combos.append({'cv': cvv, 'drop': drop, 'act': act, 'head': head, 'k_unf': 1,
                                   'epochs': (override_epochs if override_epochs is not None else EPOCHS),
                                   'HEAD_LR': HEAD_LR, 'LAST_LR': LAST_LR, 'REST_LR': REST_LR})
    # 32 combos
    if limit is not None:
        combos = combos[:limit]

    for cfg in combos:
        run_name = f"cv{cfg['cv']}_drop{cfg['drop']}_{cfg['act']}_head{cfg['head']}"
        run_dir = EXPERIMENTS_DIR / run_name
        # Si ya está completo (todos los folds tienen completed.txt), puedo saltar si querés.
        # Por simplicidad, igual llamamos a run_cv_and_test (respetará reanudación por fold).
        run_cv_and_test(run_name, cfg, samples_all, y_all)

    dur = time.time() - start_time
    print(f"\n==== Terminado todo en {dur/60:.1f} min ====")


def sanity_check():
    """Corre 1 combinación mínima con pocas épocas, para verificar estructura de salidas.
    - cv=3, drop=0.3, act=relu, head=128
    - epochs=SANITY_EPOCHS
    """
    print("[Sanity] Descubriendo samples…")
    samples_all, y_all = build_samples()
    cfg = {'cv': 3, 'drop': 0.3, 'act': 'relu', 'head': '128', 'k_unf': 1,
           'epochs': SANITY_EPOCHS, 'HEAD_LR': HEAD_LR, 'LAST_LR': LAST_LR, 'REST_LR': REST_LR}
    run_name = f"SANITY_cv{cfg['cv']}_drop{cfg['drop']}_{cfg['act']}_head{cfg['head']}"
    run_cv_and_test(run_name, cfg, samples_all, y_all)
    print("[Sanity] Listo. Revisá la carpeta: ", (EXPERIMENTS_DIR / run_name).resolve())


if __name__ == '__main__':
    # === OPCIÓN A: sanity-check rápido ===
    # sanity_check()

    # === OPCIÓN B: correr las 32 combinaciones ===
    run_all_experiments()

    # === OPCIÓN C: correr con menos épocas para prueba de humo ===
    # run_all_experiments(limit=2, override_epochs=SANITY_EPOCHS)
    pass
