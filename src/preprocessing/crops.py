
# src/preprocessing/crops.py

import numpy as np

def crop_center(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, min(w - size, cx - half))
    y1 = max(0, min(h - size, cy - half))
    return img[y1:y1+size, x1:x1+size]
