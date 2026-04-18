
# src/preprocessing/roi_detection.py
from __future__ import annotations

import cv2
import numpy as np

# ============================
# Detector de regiones
# ============================
FULL_WINDOW = True

# Defaults usados durante el entrenamiento y reportados en el paper.
# Cualquier llamada que no provea overrides reproducira exactamente la
# configuracion utilizada para generar el dataset de training.
DEFAULT_THRESH_VALUE = 180
DEFAULT_KERNEL_OPEN = 9
DEFAULT_KERNEL_DILATE = 5



def detect_all_regions(
    gray: np.ndarray,
    expand_pixels: int = 40,
    thresh_value: int | None = None,
    kernel_open: int | None = None,
    kernel_dilate: int | None = None,
):
    """
    Detector simple por umbral + morfologia, devuelve bbox centrado.

    Parametros:
      gray: imagen en escala de grises.
      expand_pixels: expansion (px) del bounding box detectado.
      thresh_value: umbral de intensidad para binarizar (0-255).
                    None -> usa DEFAULT_THRESH_VALUE (180).
      kernel_open: tamano del structuring element para el opening
                   morfologico. None -> usa DEFAULT_KERNEL_OPEN (9).
                   Usar 1 para desactivar la operacion (kernel 1x1 es
                   identidad para opening).
      kernel_dilate: tamano del structuring element para la dilatacion.
                     None -> usa DEFAULT_KERNEL_DILATE (5).
                     Usar 1 para desactivar la operacion.

    Devuelve:
      mask_all, target_regions_all, areas, filtered_contours
    """
    # Resolver parametros efectivos
    t_val = DEFAULT_THRESH_VALUE if thresh_value is None else int(thresh_value)
    k_open = DEFAULT_KERNEL_OPEN if kernel_open is None else int(kernel_open)
    k_dil = DEFAULT_KERNEL_DILATE if kernel_dilate is None else int(kernel_dilate)

    # Asegurar tamanos validos
    k_open = max(1, k_open)
    k_dil = max(1, k_dil)

    # Binarizacion
    _, mask_all = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)

    # Opening (si kernel=1x1 es identidad)
    if k_open > 1:
        kernel = np.ones((k_open, k_open), np.uint8)
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel)

    # Dilation
    if k_dil > 1:
        kernel = np.ones((k_dil, k_dil), np.uint8)
        mask_all = cv2.dilate(mask_all, kernel)

    # Deteccion de contornos
    contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_regions_all = []
    areas = []
    filtered_contours = []

    H, W = gray.shape[:2]
    half_req = 256 // 2  # ventana base centrada

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(x - expand_pixels, 0)
        y1 = max(y - expand_pixels, 0)
        x2 = min(x + w + expand_pixels, W - 1)
        y2 = min(y + h + expand_pixels, H - 1)

        if FULL_WINDOW:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            max_half_x = min(half_req, cx, W - 1 - cx)
            max_half_y = min(half_req, cy, H - 1 - cy)
            half = max(1, min(max_half_x, max_half_y))
            x1 = int(cx - half); x2 = int(cx + half)
            y1 = int(cy - half); y2 = int(cy + half)
        target_regions_all.append((x1, y1, x2, y2))
        areas.append(area)
        filtered_contours.append(cnt)

    return mask_all, target_regions_all, areas, filtered_contours
