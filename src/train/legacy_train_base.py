
# src/train/legacy_train_base.py

# igual hay que dividir lo que hay aqui en diferentes scripts
# una vez que lo demás del train ya está claro

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2


from timm import create_model

RANDOM_SEED = 42  # reproducible
# Entrenamiento (editar fácilmente)
BATCH_SIZE = 16     
CLASS1_BONUS = 1.1 

def load_state_dict_safe(path, device):
    # En versiones nuevas de torch: usar weights_only=True.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # torch viejo sin ese flag → modo clásico
        return torch.load(path, map_location=device)

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


# Cache simple por proceso (cada worker tiene su propio dict) # ESTA FUNCIÓN PASARLA A utils/io.py
_IMG_CACHE: dict[str, np.ndarray] = {}

def get_image_cached(path: Path) -> np.ndarray:
    """Devuelve la imagen RGB uint8 cacheada. Si no está, la carga y la guarda."""
    key = str(path)
    arr = _IMG_CACHE.get(key)
    if arr is None:
        arr = read_image_rgb(path)
        _IMG_CACHE[key] = arr
    return arr

# ==========================
# ======= MODELO ===========
# ==========================
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

def build_backbone() -> nn.Module:
    # EfficientNetV2-S sin cabeza final (features)
    backbone = create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='avg')
    return backbone

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

def freeze_backbone(model):
    freeze_module(model.backbone)

def unfreeze_all_backbone(model):
    unfreeze_module(model.backbone)

def unfreeze_last_k_blocks(model, k: int = 1):
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
