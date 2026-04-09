
# src/inference/single_image.py

"""
single_image.py
---------------
Inferencia principal de MARTA sobre una imagen única a partir de un checkpoint
y una configuración YAML.

Diseño unificado (sin flag soft/hard): el método siempre entrega los
siguientes outputs para cada imagen procesada:

  1) heatmap continuo de lateralización (interpolación Gaussiana)
  2) classification overlay (ROIs rellenas con su label asignado)
  3) Métricas primarias A (basadas en área):
       - pct_lat_area_all  : % del área CX43 clasificada como lateralizada,
                             sobre el área total de ROIs detectadas.
       - pct_lat_area_conf : % del área CX43 clasificada como lateralizada,
                             sobre el área de ROIs con label asignado
                             (excluye indeterminadas).
  4) Métricas complementarias B (basadas en heatmap):
       - pct_lat_heat_all  : promedio del heatmap H(x,y) sobre el área total
                             de ROIs detectadas.
       - pct_lat_heat_conf : promedio del heatmap H(x,y) sobre el área de
                             ROIs con label asignado.
"""
import argparse
from pathlib import Path

import torch

from src.utils.io import ensure_dir
from src.utils.config import load_inference_config
from src.utils.logging import log
from src.preprocessing.roi_detection import detect_all_regions
from .model_loader import load_model_from_ckpt
from .roi_inference import (
    load_image_and_gray,
    run_roi_inference,
)

from .analysis import (
    save_roi_results_table,
    save_classification_overlay,
    save_interactive_roi_html,
    run_tsne_analysis,
    build_continuous_heatmap,
    compute_global_lateralization_metrics,
    save_heatmap_outputs,
    build_inference_summary,
    save_summary_json,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MARTA inference on a single image from a YAML config."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML file.",
    )
    return parser.parse_args()

def run_single_image_inference(
    image_path: Path,
    ckpt_path: Path,
    outdir: Path,
    config_path: str | Path = "configs/inference.yaml",
):
    ensure_dir(outdir)
    log_fp = outdir / "inference_log.txt"
    log("[infer] Starting single-image inference", log_fp)
    log(f"[infer] Image path: {image_path}", log_fp)
    log(f"[infer] Checkpoint path: {ckpt_path}", log_fp)
    log(f"[infer] Output directory: {outdir}", log_fp)
    log(f"[infer] Config path: {config_path}", log_fp)

    log("[infer] Loading inference config", log_fp)
    infer_cfg = load_inference_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[infer] Device: {device}", log_fp)

    log("[image] Loading input image", log_fp)
    image_bgr, gray = load_image_and_gray(image_path)

    log("[roi] Detecting candidate regions", log_fp)
    _, rois, areas, contours = detect_all_regions(
        gray,
        expand_pixels=infer_cfg.expand,
    )
    log(f"[roi] Regions detected: {len(rois)}", log_fp)

    log("[model] Loading trained checkpoint", log_fp)
    model, train_cfg = load_model_from_ckpt(ckpt_path=ckpt_path, device=device)
    input_mode = train_cfg.get("input_mode", "256")
    fusion = train_cfg.get("fusion", "single")
    log(f"[model] input_mode={input_mode} fusion={fusion}", log_fp)

    log("[infer] Running ROI-level inference", log_fp)
    results, feats_all = run_roi_inference(
        image_bgr=image_bgr,
        rois=rois,
        model=model,
        train_cfg=train_cfg,
        infer_cfg=infer_cfg,
        device=device,
    )
    log(f"[infer] Number of ROIs to process: {len(rois)}", log_fp)

    base_name = image_path.stem

    log("[io] Saving ROI results table", log_fp)
    csv_path, xlsx_path, df_results = save_roi_results_table(
        results=results,
        outdir=outdir,
        base_name=base_name,
        save_excel=infer_cfg.save_excel,
    )
    log(f"[io] CSV path: {csv_path}", log_fp)
    if xlsx_path is not None:
        log(f"[io] XLSX path: {xlsx_path}", log_fp)

    log("[viz] Saving classification overlay", log_fp)
    classification_overlay_path = save_classification_overlay(
        image_bgr=image_bgr,
        contours=contours,
        results=results,
        outdir=outdir,
        base_name=base_name,
    )
    log(f"[viz] Classification overlay path: {classification_overlay_path}", log_fp)

    log("[viz] Saving interactive ROI HTML", log_fp)
    html_img_path = save_interactive_roi_html(
        image_bgr=image_bgr,
        df_results=df_results,
        outdir=outdir,
        base_name=base_name,
        input_mode=input_mode,
        fusion=fusion,
    )
    log(f"[viz] HTML path: {html_img_path}", log_fp)

    log("[tsne] Skipped", log_fp)
    tsne_csv_path = None
    tsne_html_path = None

    log("[heatmap] Building continuous lateralization heatmap", log_fp)
    heatmap = build_continuous_heatmap(
        image_shape=image_bgr.shape[:2],
        results=results,
        sigma=infer_cfg.sigma,
    )

    log("[heatmap] Computing global lateralization metrics", log_fp)
    global_metrics = compute_global_lateralization_metrics(
        heatmap=heatmap,
        contours=contours,
        results=results,
        image_shape=image_bgr.shape[:2],
    )
    log(f"[metrics] pct_lat_area_all  = {global_metrics['pct_lat_area_all']:.2f}%", log_fp)
    log(f"[metrics] pct_lat_area_conf = {global_metrics['pct_lat_area_conf']:.2f}%", log_fp)
    log(f"[metrics] pct_lat_heat_all  = {global_metrics['pct_lat_heat_all']:.2f}%", log_fp)
    log(f"[metrics] pct_lat_heat_conf = {global_metrics['pct_lat_heat_conf']:.2f}%", log_fp)

    heatmap_path, heatmap_overlay_path = save_heatmap_outputs(
        image_bgr=image_bgr,
        heatmap=heatmap,
        outdir=outdir,
        base_name=base_name,
    )
    log(f"[heatmap] Heatmap path: {heatmap_path}", log_fp)
    log(f"[heatmap] Heatmap overlay path: {heatmap_overlay_path}", log_fp)

    log("[io] Saving inference summary", log_fp)
    summary = build_inference_summary(
        image_path=image_path,
        ckpt_path=ckpt_path,
        outdir=outdir,
        threshold=infer_cfg.threshold,
        results=results,
        input_mode=input_mode,
        fusion=fusion,
        sigma=infer_cfg.sigma,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        overlay_path=classification_overlay_path,
        html_img_path=html_img_path,
        tsne_csv_path=tsne_csv_path,
        tsne_html_path=tsne_html_path,
        heatmap_path=heatmap_path,
        heatmap_overlay_path=heatmap_overlay_path,
        global_metrics=global_metrics,
    )
    save_summary_json(summary, outdir)
    log(f"[io] Summary path: {outdir / 'summary.json'}", log_fp)

    print("\n== LISTO ==")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary

def main(
    image_path: str | Path,
    ckpt_path: str | Path,
    outdir: str | Path,
    config_path: str | Path = "configs/inference.yaml",
):
    return run_single_image_inference(
        image_path=Path(image_path),
        ckpt_path=Path(ckpt_path),
        outdir=Path(outdir),
        config_path=config_path,
    )

if __name__ == "__main__":
    args = parse_args()
    main(
        image_path=args.image,
        ckpt_path=args.ckpt,
        outdir=args.outdir,
        config_path=args.config,
    )
