#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, os, re, json, subprocess
import cv2
import numpy as np

SCRIPT_SINGLE = "MARTA_INFER_TSNE_MULTIINPUT_AREALAT_v2.py"

GRID_ORDER = [
    [1, 2, 3],
    [4, 5, 6],
    [11, 12, 13],
    [14, 15, 16],
]

def natural_key(path: Path):
    m = re.search(r"seccion_(\d+)", path.stem, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, path.name.lower())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
def run_single_infer(img_path: Path, ckpt: Path, outdir_section: Path,
                     threshold: float, perplexity: float, soft: bool, sigma: float):
    ensure_dir(outdir_section)
    cmd = [
        sys.executable, SCRIPT_SINGLE,
        "--image", str(img_path),
        "--ckpt", str(ckpt),
        "--outdir", str(outdir_section),
        "--threshold", str(threshold),
        "--perplexity", str(perplexity),
        "--sigma", str(sigma)
    ]
    if soft:
        cmd.append("--soft")
    print("[RUN]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    summ = outdir_section / "summary.json"
    overlay_path = None
    if summ.exists():
        try:
            data = json.loads(summ.read_text(encoding="utf-8"))
            overlay_path = Path(data.get("lcr_overlay", "")) if data.get("lcr_overlay") else None
        except Exception:
            pass
    return overlay_path

def build_grid_from_overlays(overlay_map: dict, out_path: Path, pad=8):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder_images", required=True, help="Folder with seccion_*.tif images")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (best_stage3_full.pth)")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--perplexity", type=float, default=500.0)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--sigma", type=float, default=256.0)
    args = ap.parse_args()

    folder = Path(args.folder_images)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]],
                  key=natural_key)
    if not imgs:
        raise RuntimeError(f"No images found in {folder}")

    print(f"[INFO] Found {len(imgs)} images. First 5: {[p.name for p in imgs[:5]]}")

    tmp_inputs_dir = outdir / "_tmp_inputs"
    ensure_dir(tmp_inputs_dir)
    prepared = []
    for p in imgs:
        # Cargar imagen
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {p}, skipping.")
            continue
        base = p.stem
        tmp_path = tmp_inputs_dir / f"{base}.png"
        cv2.imwrite(str(tmp_path), img)


        # Seguimos como antes
        m = re.search(r"seccion_(\d+)", base, re.IGNORECASE)
        sec_num = int(m.group(1)) if m else None
        sec_out = outdir / base
        prepared.append((sec_num, tmp_path, sec_out))


    overlay_map = {}
    for sec_num, tmp_img, sec_out in prepared:
        ov = run_single_infer(tmp_img, Path(args.ckpt), sec_out,
                              args.threshold, args.perplexity, args.soft, args.sigma)
        if ov is None:
            cand = list(sec_out.glob("*_lcr_overlay_withbar.jpg"))
            if cand:
                ov = cand[0]
        if sec_num is not None and ov:
            overlay_map[sec_num] = ov

    grid_path = outdir / "combined_heatmaps_grid.jpg"
    try:
        build_grid_from_overlays(overlay_map, grid_path)
        print(f"[OK] Combined grid saved to: {grid_path}")
    except Exception as e:
        print(f"[WARN] Could not build combined grid: {e}")

if __name__ == "__main__":
    main()
