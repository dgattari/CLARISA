
# src/inference/compute_from_expert_csv.py
"""
Calcula el heatmap y las metricas de lateralizacion a partir de un CSV
de anotaciones expertas, reutilizando el pipeline de analysis.py.

Lee los parametros de deteccion de ROIs y sigma desde inference.yaml para
garantizar que las ROIs detectadas sean EXACTAMENTE las mismas que vio
el experto al anotar (y las mismas que usa la inferencia del modelo).

Uso:
    python -m src.inference.compute_from_expert_csv
        --image <path>
        --csv <expert_annotations.csv>
        --outdir <path>
        [--config configs/inference.yaml]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import pandas as pd

from src.preprocessing.roi_detection import detect_all_regions
from src.utils.config import load_inference_config
from src.inference.analysis import (
    build_continuous_heatmap,
    compute_global_lateralization_metrics,
    save_classification_overlay,
    save_heatmap_outputs,
    save_summary_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--csv", required=True, help="Expert annotations CSV.")
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--config", type=str, default="configs/inference.yaml",
        help="Path to inference YAML (same as used during inference and annotation).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] Loading inference config: {args.config}")
    infer_cfg = load_inference_config(args.config)

    print(f"[read] Loading image {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]

    print(
        f"[detect] Re-detecting ROIs (thresh={infer_cfg.thresh_value} "
        f"kernel_open={infer_cfg.kernel_open} kernel_dilate={infer_cfg.kernel_dilate} "
        f"expand={infer_cfg.expand})"
    )
    _, rois, areas, contours = detect_all_regions(
        gray,
        expand_pixels=infer_cfg.expand,
        thresh_value=infer_cfg.thresh_value,
        kernel_open=infer_cfg.kernel_open,
        kernel_dilate=infer_cfg.kernel_dilate,
    )
    print(f"[detect] {len(rois)} ROIs")

    df = pd.read_csv(args.csv)
    expert_map = dict(zip(df["idx"].astype(int), df["expert_label"].astype(int)))
    print(f"[csv] Loaded {len(expert_map)} expert annotations")

    # Armar 'results' emulando el formato del pipeline de inferencia.
    # ROIs no anotadas por el experto se marcan como indeterminadas (-1).
    results = []
    for i, (x1, y1, x2, y2) in enumerate(rois):
        label = expert_map.get(i, -1)

        if label == 1:
            p1 = 1.0
        elif label == 0:
            p1 = 0.0
        else:
            p1 = 0.5

        results.append({
            "idx": i,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "valid": True,
            "prob_0": 1 - p1, "prob_1": p1,
            "max_prob": max(p1, 1 - p1),
            "pred_raw": label if label in (0, 1) else 0,
            "final_label": label,
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
        })

    base_name = image_path.stem + "_expert"

    print("[viz] Classification overlay")
    ovl = save_classification_overlay(image_bgr, contours, results, outdir, base_name)

    print(f"[heatmap] Building heatmap (sigma={infer_cfg.sigma})")
    heatmap = build_continuous_heatmap((h, w), results, sigma=infer_cfg.sigma)
    hp, hop = save_heatmap_outputs(image_bgr, heatmap, outdir, base_name)

    print("[metrics] Computing metrics")
    metrics = compute_global_lateralization_metrics(heatmap, contours, results, (h, w))

    # Contadores corregidos (bug del contador 'annotated' en la version anterior
    # que referenciaba 'i' del loop exterior).
    annotated_count = sum(1 for idx in expert_map.keys() if 0 <= idx < len(rois))
    class_0 = sum(1 for v in expert_map.values() if v == 0)
    class_1 = sum(1 for v in expert_map.values() if v == 1)
    dudosas = sum(1 for v in expert_map.values() if v == -1)

    summary = {
        "image": str(image_path),
        "source": "expert",
        "total_rois": len(rois),
        "annotated": annotated_count,
        "class_0": class_0,
        "class_1": class_1,
        "dudosas": dudosas,
        "not_annotated": len(rois) - annotated_count,
        "sigma": infer_cfg.sigma,
        "thresh_value": infer_cfg.thresh_value,
        "kernel_open": infer_cfg.kernel_open,
        "kernel_dilate": infer_cfg.kernel_dilate,
        "expand": infer_cfg.expand,
        "classification_overlay": str(ovl),
        "heatmap": str(hp),
        "heatmap_overlay": str(hop),
        **metrics,
    }
    save_summary_json(summary, outdir)

    print("\n== LISTO ==")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
