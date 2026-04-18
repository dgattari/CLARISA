
# src/inference/analysis.py
# Este módulo responde a:
# “Con las predicciones ya hechas, ¿cómo genero todos los artefactos finales?”

"""
analysis.py
-----------
Postproceso y artefactos finales de inferencia.

Diseño unificado (sin flag soft/hard):
  El módulo entrega siempre, para cada imagen procesada:

    1) Heatmap continuo de lateralización
       Interpolación Gaussiana de las probabilidades ROI-level.
       Rol: salida visual principal.

    2) Classification overlay
       Contornos de las ROIs coloreados según su label asignado
       (clase 0, clase 1, o indeterminado).
       Rol: visualización directa de la decisión discreta del clasificador.

    3) Métricas A (primarias, basadas en área — discretas)
       - pct_lat_area_all  : área de ROIs clase 1 / área total de ROIs detectadas
       - pct_lat_area_conf : área de ROIs clase 1 / área de ROIs con label asignado
                              (excluye indeterminadas)

    4) Métricas B (complementarias, basadas en heatmap — continuas)
       - pct_lat_heat_all  : promedio de H(x,y) sobre el área de TODAS las ROIs
       - pct_lat_heat_conf : promedio de H(x,y) sobre el área de las ROIs con
                              label asignado
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from PIL import Image as PILImage
from sklearn.manifold import TSNE

# ============================
# Plotly helpers (TSNE + buscador)
# ============================
def write_tsne_html_with_search(
    out_html: Path,
    df_tsne: pd.DataFrame,
    title: str,
) -> None:

    colors = df_tsne['final_label'].map({-1:'#2ca02c', 0:'#1f77b4', 1:'#d62728'}).fillna('#7f7f7f')
    text = [
        f"idx={int(r.idx)} | label={int(r.final_label)} | max={float(r.max_prob):.3f}"
        f"<br>bbox=({int(r.x1)},{int(r.y1)})-({int(r.x2)},{int(r.y2)})"
        for r in df_tsne.itertuples(index=False)
    ]

    trace_main = go.Scattergl(
        x=df_tsne['tsne_x'], y=df_tsne['tsne_y'],
        mode='markers', marker=dict(size=6, color=colors),
        text=text, hoverinfo='text', name='ROIs'
    )

    trace_hi = go.Scatter(
        x=[], y=[], mode='markers+text',
        marker=dict(size=14, symbol='star'),
        text=[], textposition='top center', name='highlight'
    )

    fig = go.Figure(data=[trace_main, trace_hi])
    fig.update_layout(title=title, width=900, height=700, margin=dict(l=30, r=10, t=50, b=30))
    div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    idx_list = df_tsne['idx'].astype(int).tolist()
    xs = df_tsne['tsne_x'].astype(float).tolist()
    ys = df_tsne['tsne_y'].astype(float).tolist()

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>t-SNE con buscador de idx</title></head>
<body>
<h3 style="font-family: sans-serif;">{title}</h3>
<div style="margin: 8px 0 12px 0; font-family: sans-serif;">
  <label for="idxInput"><b>Buscar idx:</b></label>
  <input id="idxInput" type="number" placeholder="ej: 123" style="padding:4px; width: 120px;">
  <span id="idxInfo" style="margin-left:10px; color:#555;"></span>
</div>
{div}
<script>
const gd = document.querySelector('div.js-plotly-plot');
const idxs = {idx_list};
const xs = {xs};
const ys = {ys};
const mapIdx = new Map();
for (let i=0; i<idxs.length; i++) mapIdx.set(idxs[i], i);

function highlight(i) {{
  if (i === undefined || i === null || i < 0 || i >= xs.length) {{
    Plotly.restyle(gd, {{"x":[[]], "y":[[]], "text":[[]]}}, [1]);
    document.getElementById('idxInfo').textContent = 'No encontrado';
    return;
  }}
  Plotly.restyle(gd, {{"x":[[xs[i]]], "y":[[ys[i]]], "text":[['idx='+idxs[i]]]}}, [1]);
  document.getElementById('idxInfo').textContent = 'Destacado idx='+idxs[i];
}}

document.getElementById('idxInput').addEventListener('change', function() {{
  const val = parseInt(this.value,10);
  const i = mapIdx.get(val);
  highlight(i);
}});

gd.on('plotly_click', function(ev) {{
  if (!ev || !ev.points || !ev.points.length) return;
  const p = ev.points[0];
  if (p.curveNumber === 0) {{
    const i = p.pointIndex;
    document.getElementById('idxInput').value = idxs[i];
    highlight(i);
  }}
}});
</script>
</body></html>
"""
    out_html.write_text(html, encoding='utf-8')

# ============================
# Colorbar helpers
# ============================
def attach_colorbar_right_gray(
    image_gray_01: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    height: int | None = None,
) -> np.ndarray:
    if height is None:
        height = image_gray_01.shape[0]

    bar = np.linspace(1.0, 0.0, height).reshape(height, 1)
    bar_rgb = (np.clip(bar, 0, 1) * 255).astype(np.uint8)
    bar_rgb = np.repeat(bar_rgb, 30, axis=1)
    bar_rgb = cv2.cvtColor(bar_rgb, cv2.COLOR_GRAY2BGR)

    cv2.putText(bar_rgb, f"{int(vmax*100)}%", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(bar_rgb, f"{int(vmin*100)}%", (2, height-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    heat_rgb = cv2.cvtColor((image_gray_01*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return np.concatenate([heat_rgb, bar_rgb], axis=1)

def attach_colorbar_right_rgb(
    overlay_bgr: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    h = overlay_bgr.shape[0]
    bar = np.linspace(1.0, 0.0, h).reshape(h, 1)
    bar_u8 = (np.clip(bar, 0, 1) * 255).astype(np.uint8)
    bar_color = cv2.applyColorMap(bar_u8, cv2.COLORMAP_TURBO)

    cv2.putText(bar_color, f"{int(vmax*100)}%", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(bar_color, f"{int(vmin*100)}%", (2, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    return np.concatenate([overlay_bgr, bar_color], axis=1)

# ============================
# Guardado de tabla ROI
# ============================
def save_roi_results_table(
    results: List[Dict[str, Any]],
    outdir: Path,
    base_name: str,
    save_excel: bool = True,
) -> Tuple[Path, Path | None, pd.DataFrame]:
    df = pd.DataFrame(results)

    csv_path = outdir / f"{base_name}_resultados.csv"
    df.to_csv(csv_path, index=False)

    xlsx_path = None
    if save_excel:
        xlsx_path = outdir / f"{base_name}_resultados.xlsx"
        try:
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
        except Exception as exc:
            print(f"[WARN] No pude escribir Excel ({exc}) — quedó el CSV.")
            xlsx_path = None

    return csv_path, xlsx_path, df

# ============================
# Classification overlay (ROIs coloreadas por label)
# ============================
def save_classification_overlay(
    image_bgr: np.ndarray,
    contours: List[Any],
    results: List[Dict[str, Any]],
    outdir: Path,
    base_name: str,
) -> Path:
    overlay = image_bgr.copy()

    if overlay.ndim == 2 or overlay.shape[2] == 1:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for cnt, row in zip(contours, results):
        label = row["final_label"]

        if label == 0:
            color = (255, 0, 0)
        elif label == 1:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.drawContours(overlay, [cnt], -1, color, thickness=cv2.FILLED)

    out_path = outdir / f"{base_name}_clasificacion_coloreada.jpg"
    cv2.imwrite(str(out_path), overlay)
    return out_path

# ============================
# HTML interactivo sobre imagen
# ============================
def save_interactive_roi_html(
    image_bgr: np.ndarray,
    df_results: pd.DataFrame,
    outdir: Path,
    base_name: str,
    input_mode: str,
    fusion: str,
) -> Path:
    df_plot = df_results.copy()
    df_plot["cx"] = (df_plot["x1"] + df_plot["x2"]) / 2.0
    df_plot["cy"] = (df_plot["y1"] + df_plot["y2"]) / 2.0

    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=PILImage.fromarray(img_rgb),
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=w,
        sizey=h,
        sizing="stretch",
        layer="below",
    ))

    hover = [
        f"idx={int(r.idx)} | bbox=({int(r.x1)},{int(r.y1)})-({int(r.x2)},{int(r.y2)})"
        f"<br>final_label={int(r.final_label)} | pred_raw={('NA' if pd.isna(r.pred_raw) else int(r.pred_raw))}"
        f"<br>p0={('NA' if pd.isna(r.prob_0) else round(float(r.prob_0),3))}"
        f" | p1={('NA' if pd.isna(r.prob_1) else round(float(r.prob_1),3))}"
        f" | max={('NA' if pd.isna(r.max_prob) else round(float(r.max_prob),3))}"
        for r in df_plot.itertuples(index=False)
    ]

    colmap = {-1: "#2ca02c", 0: "#1f77b4", 1: "#d62728"}
    colors = df_plot["final_label"].map(colmap).fillna("#7f7f7f")

    fig.add_trace(go.Scatter(
        x=df_plot["cx"],
        y=df_plot["cy"],
        mode="markers",
        marker=dict(size=7, color=colors, opacity=0.9, line=dict(width=1, color="black")),
        text=hover,
        hoverinfo="text",
        name="ROIs",
    ))

    # Calcular dimensiones manteniendo aspect ratio de la imagen original.
    # Esto evita que imagenes muy anchas/altas se vean comprimidas en el HTML.
    max_width = 1200
    max_height = 900
    aspect = w / h if h > 0 else 1.0

    if aspect >= max_width / max_height:
        # Imagen mas ancha que alta -> limita por width
        plot_width = max_width
        plot_height = max(1, int(round(max_width / aspect)))
    else:
        # Imagen mas alta que ancha -> limita por height
        plot_height = max_height
        plot_width = max(1, int(round(max_height * aspect)))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        width=plot_width,
        height=plot_height,
        xaxis=dict(range=[0, w], showgrid=False, visible=False),
        yaxis=dict(range=[0, h], showgrid=False, visible=False),
        title=f"Inferencia — ROIs y probabilidades ({input_mode}/{fusion})",
    )

    out_path = outdir / f"{base_name}_resultados.html"
    fig.write_html(str(out_path))
    return out_path

# ============================
# t-SNE
# ============================
def run_tsne_analysis(
    feats_all: List[np.ndarray],
    results: List[Dict[str, Any]],
    outdir: Path,
    base_name: str,
    perplexity: float,
    input_mode: str,
    fusion: str,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Path, Path]:
    rows_valid = [r for r in results if r["valid"]]

    if len(feats_all) == 0:
        raise RuntimeError("No hay embeddings válidos para t-SNE.")

    X = np.stack(feats_all, axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        max_iter=1000,
        n_iter_without_progress=300,
        learning_rate="auto",
        random_state=random_seed,
        verbose=1,
    )
    X2 = tsne.fit_transform(X)

    df_tsne = pd.DataFrame({
        "idx": [r["idx"] for r in rows_valid],
        "tsne_x": X2[:, 0],
        "tsne_y": X2[:, 1],
        "final_label": [r["final_label"] for r in rows_valid],
        "pred_raw": [r["pred_raw"] for r in rows_valid],
        "prob_0": [r["prob_0"] for r in rows_valid],
        "prob_1": [r["prob_1"] for r in rows_valid],
        "max_prob": [r["max_prob"] for r in rows_valid],
        "x1": [r["x1"] for r in rows_valid],
        "y1": [r["y1"] for r in rows_valid],
        "x2": [r["x2"] for r in rows_valid],
        "y2": [r["y2"] for r in rows_valid],
    })

    csv_path = outdir / f"{base_name}_tsne.csv"
    html_path = outdir / f"{base_name}_tsne.html"

    df_tsne.to_csv(csv_path, index=False)
    write_tsne_html_with_search(html_path, df_tsne, f"t-SNE — {base_name} (cfg: {input_mode}/{fusion})")

    return df_tsne, csv_path, html_path

# ============================
# Construcción del heatmap continuo
# ============================
def build_continuous_heatmap(
    image_shape: Tuple[int, int],
    results: List[Dict[str, Any]],
    sigma: float,
) -> np.ndarray:
    """
    Construye el heatmap continuo de lateralización por interpolación
    espacial Gaussiana de las probabilidades ROI-level.

    Para cada ROI válida se acumula su prob_1 en el centro del bounding box
    (mapa P) y un +1 en el mismo lugar (mapa W). Ambos mapas se convolucionan
    con un kernel Gaussiano de desviación estándar sigma, y el heatmap final
    se obtiene como:

        H(x,y) = (P * G_sigma)(x,y) / max((W * G_sigma)(x,y), eps)

    Esto corresponde a un estimador tipo Nadaraya-Watson: promedio local
    ponderado de las probabilidades de clase 1 de las ROIs vecinas.
    """
    h, w = image_shape
    p_map = np.zeros((h, w), dtype=np.float32)
    w_map = np.zeros((h, w), dtype=np.float32)

    for r in results:
        if not r["valid"]:
            continue

        cx = int((r["x1"] + r["x2"]) / 2)
        cy = int((r["y1"] + r["y2"]) / 2)

        if 0 <= cx < w and 0 <= cy < h:
            p_map[cy, cx] += float(r["prob_1"])
            w_map[cy, cx] += 1.0

    p_blur = cv2.GaussianBlur(
        p_map, ksize=(0, 0),
        sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    w_blur = cv2.GaussianBlur(
        w_map, ksize=(0, 0),
        sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    return p_blur / np.maximum(w_blur, 1e-8)

# ============================
# Métricas globales
# ============================
def compute_global_lateralization_metrics(
    heatmap: np.ndarray,
    contours: List[Any],
    results: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Calcula las métricas globales de lateralización.

    Métricas A (primarias, basadas en área — discretas):
        - pct_lat_area_all  : 100 * area_clase_1 / area_total_ROIs
        - pct_lat_area_conf : 100 * area_clase_1 / area_ROIs_con_label

    Métricas B (complementarias, basadas en heatmap — continuas):
        - pct_lat_heat_all  : 100 * mean(H) sobre el area de TODAS las ROIs
        - pct_lat_heat_conf : 100 * mean(H) sobre el area de las ROIs con label
    """
    h, w = image_shape

    mask_all = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask_all, [cnt], -1, 1, thickness=cv2.FILLED)

    mask_conf = np.zeros((h, w), dtype=np.uint8)
    for cnt, r in zip(contours, results):
        if r["final_label"] in (0, 1):
            cv2.drawContours(mask_conf, [cnt], -1, 1, thickness=cv2.FILLED)

    mask_cls1 = np.zeros((h, w), dtype=np.uint8)
    for cnt, r in zip(contours, results):
        if r["final_label"] == 1:
            cv2.drawContours(mask_cls1, [cnt], -1, 1, thickness=cv2.FILLED)

    denom_all = int((mask_all > 0).sum())
    denom_conf = int((mask_conf > 0).sum())
    area_cls1 = int((mask_cls1 > 0).sum())

    # Metricas A (area, primarias)
    if denom_all > 0:
        pct_area_all = 100.0 * area_cls1 / denom_all
    else:
        pct_area_all = float("nan")

    if denom_conf > 0:
        pct_area_conf = 100.0 * area_cls1 / denom_conf
    else:
        pct_area_conf = float("nan")

    # Metricas B (heatmap, complementarias)
    if denom_all > 0:
        pct_heat_all = 100.0 * float(np.sum(heatmap * (mask_all > 0)) / denom_all)
    else:
        pct_heat_all = float("nan")

    if denom_conf > 0:
        pct_heat_conf = 100.0 * float(np.sum(heatmap * (mask_conf > 0)) / denom_conf)
    else:
        pct_heat_conf = float("nan")

    return {
        "pct_lat_area_all": float(pct_area_all),
        "pct_lat_area_conf": float(pct_area_conf),
        "pct_lat_heat_all": float(pct_heat_all),
        "pct_lat_heat_conf": float(pct_heat_conf),
        "denom_all_px": int(denom_all),
        "denom_conf_px": int(denom_conf),
        "area_cls1_px": int(area_cls1),
    }

# ============================
# Guardado de heatmap + overlay del heatmap
# ============================
def save_heatmap_outputs(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    outdir: Path,
    base_name: str,
    alpha: float = 0.4,
) -> Tuple[Path, Path]:
    """
    Guarda dos archivos derivados del heatmap continuo:
      - <base>_heatmap.jpg         : heatmap en escala de grises con colorbar.
      - <base>_heatmap_overlay.jpg : overlay TURBO del heatmap sobre la imagen
                                     original, con colorbar.
    """
    heat_min = float(np.nanmin(heatmap))
    heat_max = float(np.nanmax(heatmap))
    heat_range = max(1e-8, heat_max - heat_min)
    heat_01 = (heatmap - heat_min) / heat_range

    heat_withbar = attach_colorbar_right_gray(
        heat_01, vmin=0.0, vmax=1.0, height=image_bgr.shape[0]
    )
    heatmap_path = outdir / f"{base_name}_heatmap.jpg"
    cv2.imwrite(str(heatmap_path), heat_withbar)

    heat_u8 = (np.clip(heat_01, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heat_color, alpha, 0)
    overlay_withbar = attach_colorbar_right_rgb(overlay, vmin=0.0, vmax=1.0)

    overlay_path = outdir / f"{base_name}_heatmap_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay_withbar)

    return heatmap_path, overlay_path

# ============================
# Resumen final
# ============================
def build_inference_summary(
    image_path: Path,
    ckpt_path: Path,
    outdir: Path,
    threshold: float,
    results: List[Dict[str, Any]],
    input_mode: str,
    fusion: str,
    sigma: float,
    csv_path: Path,
    xlsx_path: Path | None,
    overlay_path: Path,
    html_img_path: Path,
    tsne_csv_path: Path | None,
    tsne_html_path: Path | None,
    heatmap_path: Path,
    heatmap_overlay_path: Path,
    global_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    n_total = len(results)
    n0 = sum(1 for r in results if r["final_label"] == 0)
    n1 = sum(1 for r in results if r["final_label"] == 1)
    n_ind = sum(1 for r in results if r["final_label"] == -1)

    return {
        "image": str(image_path),
        "ckpt": str(ckpt_path),
        "threshold": float(threshold),
        "total": n_total,
        "clase0": n0,
        "clase1": n1,
        "indeciso": n_ind,
        "out_dir": str(outdir.resolve()),
        "csv": str(csv_path),
        "xlsx": str(xlsx_path) if xlsx_path is not None else None,
        "classification_overlay": str(overlay_path),
        "html_img": str(html_img_path),
        "tsne_csv": str(tsne_csv_path) if tsne_csv_path is not None else None,
        "tsne_html": str(tsne_html_path) if tsne_html_path is not None else None,
        "heatmap": str(heatmap_path),
        "heatmap_overlay": str(heatmap_overlay_path),
        "input_mode": input_mode,
        "fusion": fusion,
        "sigma": float(sigma),
        **global_metrics,
    }

def save_summary_json(summary: Dict[str, Any], outdir: Path) -> Path:
    out_path = outdir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
