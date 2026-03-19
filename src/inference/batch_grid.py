
# src/inference/batch_grid.py -> Ejecución por lote y grid final.

"""
batch_grid.py
-------------
Ejecución por lote de la inferencia MARTA sobre un conjunto de secciones y
montaje final de un grid de overlays.

Responsabilidades:
  - descubrir imágenes en una carpeta
  - ordenarlas por número de sección
  - ejecutar inferencia sección a sección
  - recoger los overlays finales
  - construir el grid combinado
"""

import argparse
import re
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.utils.io import ensure_dir
from .single_image import run_single_image_inference

GRID_ORDER = [
    [1, 2, 3],
    [4, 5, 6],
    [11, 12, 13],
    [14, 15, 16],
]


 def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MARTA inference over a folder and build a grid of overlays."
    )
    parser.add_argument("--folder_images", required=True, help="Folder with section images.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--outdir", required=True, help="Output folder.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML file.",
    )
    return parser.parse_args()


def run_batch_grid_inference(
    folder_images: Path,
    ckpt_path: Path,
    outdir: Path,
    config_path: Path = Path("configs/inference.yaml"),
) -> dict:
    if not folder_images.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_images}")

    ensure_dir(outdir)

    images = sorted(
        [p for p in folder_images.iterdir() if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]],
        key=natural_section_key,
    )

    if not images:
        raise RuntimeError(f"No images found in {folder_images}")

    print(f"[INFO] Found {len(images)} images. First 5: {[p.name for p in images[:5]]}")

    overlay_map: Dict[int, Path] = {}

    for image_path in images:
        base = image_path.stem
        match = re.search(r"seccion_(\d+)", base, re.IGNORECASE)
        section_num = int(match.group(1)) if match else None

        section_outdir = outdir / base

        overlay_path = run_single_section_inference(
            image_path=image_path,
            ckpt_path=ckpt_path,
            outdir_section=section_outdir,
            config_path=config_path,
        )

        if section_num is not None and overlay_path is not None:
            overlay_map[section_num] = overlay_path

    grid_path = outdir / "combined_heatmaps_grid.jpg"
    build_grid_from_overlays(overlay_map=overlay_map, out_path=grid_path)

    summary = {
        "folder_images": str(folder_images),
        "ckpt": str(ckpt_path),
        "outdir": str(outdir.resolve()),
        "n_images": len(images),
        "grid_path": str(grid_path),
    }

    print(f"[OK] Combined grid saved to: {grid_path}")
    return summary

def natural_section_key(path: Path):
    m = re.search(r"seccion_(\d+)", path.stem, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, path.name.lower())

def run_single_section_inference(
    image_path: Path,
    ckpt_path: Path,
    outdir_section: Path,
    config_path: Path,
) -> Path | None:
    ensure_dir(outdir_section)

    summary = run_single_image_inference(
        image_path=image_path,
        ckpt_path=ckpt_path,
        outdir=outdir_section,
        config_path=config_path,
    )

    overlay_path = summary.get("lcr_overlay", None)
    return Path(overlay_path) if overlay_path else None

def build_grid_from_overlays(
    overlay_map: dict[int, Path],
    out_path: Path,
    pad: int = 8,
) -> Path:
    first = None
    for row in GRID_ORDER:
        for n in row:
            p = overlay_map.get(n)
            if p and Path(p).exists():
                first = cv2.imread(str(p), cv2.IMREAD_COLOR)
                break
        if first is not None:
            break

    if first is None:
        raise RuntimeError("No overlay images found to stitch.")
    
    th, tw = first.shape[:2]

    rows_imgs = []
    for row in GRID_ORDER:
        row_imgs = []
        for n in row:
            p = overlay_map.get(n)
            if p and Path(p).exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img.shape[:2] != (th, tw):
                    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                img = np.full((th, tw, 3), 40, dtype=np.uint8)
                cv2.putText(img, f"Missing seccion_{n}", (20, th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
            row_imgs.append(img)
        row_cat = cv2.hconcat(row_imgs)
        rows_imgs.append(row_cat)
    grid = cv2.vconcat(rows_imgs)

    if pad > 0:
        h, w = grid.shape[:2]
        grid_pad = np.full((h + 3*pad, w + 2*pad, 3), 0, dtype=np.uint8)
        grid_pad[pad:pad+h, pad:pad+w] = grid
        grid = grid_pad
    cv2.imwrite(str(out_path), grid)
    return out_path

def main(
    folder_images: str | Path,
    ckpt_path: str | Path,
    outdir: str | Path,
    config_path: str | Path = "configs/inference.yaml",
):
    return run_batch_grid_inference(
        folder_images=Path(folder_images),
        ckpt_path=Path(ckpt_path),
        outdir=Path(outdir),
        config_path=Path(config_path),
    )

if __name__ == "__main__":
    args = parse_args()
    main(
        folder_images=args.folder_images,
        ckpt_path=args.ckpt,
        outdir=args.outdir,
        config_path=args.config,
    )