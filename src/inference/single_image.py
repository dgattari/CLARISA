
# src/inference/single_image.py 

"""
single_image.py
---------------
Inferencia principal de MARTA sobre una imagen única a partir de un checkpoint
y una configuración YAML.

Este script conserva la lógica general del antiguo
MARTA_INFER_TSNE_MULTIINPUT_AREALAT_v2.py, pero separa responsabilidades en
módulos más pequeños para facilitar:

  - inferencia reproducible
  - mantenimiento del código
  - análisis ROI a ROI
  - generación de overlays, heatmaps y t-SNE

Notas:
  - La implementación reutiliza funciones heredadas del código original de Dani
    siempre que ha sido posible.
  - El objetivo de esta refactorización es organizar el código, no cambiar
    la lógica metodológica de la inferencia.
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
    build_lateralization_heatmap,
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
    log_fp = outdir / "inference_log.txt"
    log("[infer] Starting single-image inference", log_fp)
    log(f"[infer] Image path: {image_path}", log_fp)
    log(f"[infer] Checkpoint path: {ckpt_path}", log_fp)
    log(f"[infer] Output directory: {outdir}", log_fp)
    log(f"[infer] Config path: {config_path}", log_fp)

    log("[infer] Loading inference config", log_fp)
    infer_cfg = load_inference_config(config_path)
    ensure_dir(outdir)
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
    overlay_path = save_classification_overlay(
        image_bgr=image_bgr,
        contours=contours,
        results=results,
        outdir=outdir,
        base_name=base_name,
    )
    log(f"[viz] Overlay path: {overlay_path}", log_fp)

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

    log("[tsne] Running t-SNE analysis", log_fp)
    _, tsne_csv_path, tsne_html_path = run_tsne_analysis(
        feats_all=feats_all,
        results=results,
        outdir=outdir,
        base_name=base_name,
        perplexity=infer_cfg.perplexity,
        input_mode=input_mode,
        fusion=fusion,
        random_seed=infer_cfg.random_seed,
    )
    log(f"[tsne] Perplexity: {infer_cfg.perplexity}", log_fp)
    log(f"[tsne] HTML path: {tsne_html_path}", log_fp)

    log("[heatmap] Building lateralization heatmap", log_fp)
    heat = build_lateralization_heatmap(
        image_shape=image_bgr.shape[:2],
        contours=contours,
        results=results,
        soft=infer_cfg.soft,
        sigma=infer_cfg.sigma,
    )

    log("[heatmap] Computing global lateralization metrics", log_fp)
    global_metrics = compute_global_lateralization_metrics(
        heat=heat,
        contours=contours,
        results=results,
        soft=infer_cfg.soft,
        image_shape=image_bgr.shape[:2],
    )

    heatmap_path, lcr_overlay_path = save_heatmap_outputs(
        image_bgr=image_bgr,
        heat=heat,
        outdir=outdir,
        base_name=base_name,
    )
    log(f"[heatmap] Heatmap path: {heatmap_path}", log_fp)
    log(f"[heatmap] Overlay path: {lcr_overlay_path}", log_fp)

    log("[io] Saving inference summary", log_fp)
    summary = build_inference_summary(
        image_path=image_path,
        ckpt_path=ckpt_path,
        outdir=outdir,
        threshold=infer_cfg.threshold,
        results=results,
        input_mode=input_mode,
        fusion=fusion,
        soft=infer_cfg.soft,
        sigma=infer_cfg.sigma,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        overlay_path=overlay_path,
        html_img_path=html_img_path,
        tsne_csv_path=tsne_csv_path,
        tsne_html_path=tsne_html_path,
        heatmap_path=heatmap_path,
        lcr_overlay_path=lcr_overlay_path,
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


# info extra que no se si voy a usar. 
# Mejoras pequeñas que haría ya
# 1. io.py

# Añadiría helpers simples que te van a servir tanto en train como en inferencia:

# read_yaml

# write_json

# ensure_dir

# Ahora mismo Dani los tiene dispersos.

# 2. paths.py

# No forzaría rutas globales en inferencia. Para inferencia es mejor pasar:

# image_path

# ckpt_path

# outdir

# por argumentos o config.

# 3. analysis.py

# Separaría bien:

# artefactos tabulares

# artefactos visuales

# métricas globales

# Eso luego te ayudará muchísimo si quieres desactivar t-SNE o HTML en cluster.