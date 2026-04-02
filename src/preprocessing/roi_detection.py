
# src/preprocessing/roi_detection.py

import cv2
import numpy as np

# ============================
# Detector de regiones
# ============================
FULL_WINDOW = True

def detect_all_regions(gray: np.ndarray, expand_pixels: int = 40):
    """Detector simple por umbral + morfología, devuelve bbox centrado."""
    #_, mask_all = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((5, 5), np.uint8)
    #kernel = np.ones((9, 9), np.uint8)
    #mask_open = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel)
    #mask_all = cv2.dilate(mask_open, kernel)
    _, mask_all = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    mask_all = cv2.dilate(mask_all, kernel)

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
