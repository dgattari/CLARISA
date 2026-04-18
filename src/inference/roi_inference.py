
# src/inference/roi_inference.py

"""
roi_inference.py
----------------
Inferencia ROI a ROI sobre una imagen dada.

Responsabilidades:
  - leer imagen y escala de grises
  - construir el transform de inferencia
  - preparar tensores segun input_mode
  - ejecutar inferencia ROI a ROI
  - devolver predicciones y embeddings

Los tamanos de crop (local, context) pueden ser sobrescritos desde el
config de inferencia, permitiendo adaptar el pipeline a imagenes con
resolucion distinta a la de entrenamiento sin modificar codigo.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast

from src.preprocessing.crops import crop_center
from src.preprocessing.roi_detection import detect_all_regions

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def run_roi_inference(
    image_bgr: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    model: nn.Module,
    train_cfg: Dict[str, Any],
    infer_cfg,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Ejecuta inferencia ROI a ROI sobre una imagen.
    Devuelve (results, feats_all).
    """
    input_mode = train_cfg.get("input_mode", "256")
    resize_to = getattr(infer_cfg, "resize_to", 384)
    threshold = getattr(infer_cfg, "threshold", 0.5)

    override_local = getattr(infer_cfg, "crop_local", None)
    override_context = getattr(infer_cfg, "crop_context", None)

    transform = build_inference_transform(resize_to=resize_to)

    results: List[Dict[str, Any]] = []
    feats_all: List[np.ndarray] = []

    for idx, (x1, y1, x2, y2) in enumerate(rois):
        crop = image_bgr[y1:y2, x1:x2]

        valid = not (crop.size == 0 or x2 <= x1 or y2 <= y1)

        if not valid:
            results.append({
                "idx": idx,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "valid": False,
                "prob_0": None, "prob_1": None,
                "max_prob": None, "pred_raw": None,
                "final_label": -1,
                "cx": None, "cy": None,
            })
            continue

        h, w = crop.shape[:2]
        cx = x1 + w // 2
        cy = y1 + h // 2

        roi_tensor = prepare_roi_tensor(
            image_bgr=image_bgr,
            cx=cx, cy=cy,
            input_mode=input_mode,
            resize_to=resize_to,
            transform=transform,
            device=device,
            override_local=override_local,
            override_context=override_context,
        )

        pred = infer_single_roi(
            model=model, roi_tensor=roi_tensor,
            device=device, threshold=threshold,
        )

        feats_all.append(pred["features"])

        results.append({
            "idx": idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "valid": True,
            "prob_0": pred["prob_0"],
            "prob_1": pred["prob_1"],
            "max_prob": pred["max_prob"],
            "pred_raw": pred["pred_raw"],
            "final_label": pred["final_label"],
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
        })

    return results, feats_all


def infer_single_roi(
    model: nn.Module,
    roi_tensor: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
        logits = model.forward_logits(roi_tensor)
        features = model.forward_feats(roi_tensor)

    probs = softmax_np(logits.detach().cpu().numpy())[0]
    pred_raw = int(np.argmax(probs))
    max_prob = float(np.max(probs))
    final_label = pred_raw if max_prob >= threshold else -1

    return {
        "prob_0": float(probs[0]),
        "prob_1": float(probs[1]),
        "pred_raw": pred_raw,
        "max_prob": max_prob,
        "final_label": final_label,
        "features": features.squeeze(0).detach().cpu().numpy(),
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def build_inference_transform(resize_to: int) -> A.Compose:
    return A.Compose([
        A.Resize(resize_to, resize_to),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_image_and_gray(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No pude leer: {image_path}")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, gray


def get_crop_sizes_for_input_mode(
    input_mode: str,
    override_local: int | None = None,
    override_context: int | None = None,
) -> Tuple[int, int | None]:
    """
    Devuelve los tamanos de crop para inferencia.
      - '256'  -> crop 256
      - '384'  -> crop 512
      - 'stack' -> crop 256 y crop 512
    Si se pasan overrides, reemplazan los defaults.
    """
    if input_mode == "256":
        local = override_local if override_local is not None else 256
        return local, None
    if input_mode == "384":
        local = override_local if override_local is not None else 512
        return local, None
    if input_mode == "stack":
        local = override_local if override_local is not None else 256
        context = override_context if override_context is not None else 512
        return local, context

    raise ValueError(f"input_mode no soportado: {input_mode}")


def prepare_roi_tensor(
    image_bgr: np.ndarray,
    cx: int, cy: int,
    input_mode: str,
    resize_to: int,
    transform: A.Compose,
    device: torch.device,
    override_local: int | None = None,
    override_context: int | None = None,
) -> torch.Tensor:
    if input_mode == "stack":
        size_1, size_2 = get_crop_sizes_for_input_mode(
            input_mode, override_local, override_context
        )
        roi_1 = crop_center(image_bgr, cx, cy, size_1)
        roi_2 = crop_center(image_bgr, cx, cy, size_2)

        t1 = transform(image=cv2.cvtColor(roi_1, cv2.COLOR_BGR2RGB))["image"]
        t2 = transform(image=cv2.cvtColor(roi_2, cv2.COLOR_BGR2RGB))["image"]
        return torch.cat([t1, t2], dim=0).unsqueeze(0).to(device)

    size_1, _ = get_crop_sizes_for_input_mode(
        input_mode, override_local, override_context
    )
    roi = crop_center(image_bgr, cx, cy, size_1)
    ten = transform(image=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))["image"]
    return ten.unsqueeze(0).to(device)
