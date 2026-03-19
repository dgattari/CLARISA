
# ./MARTA_INFER_TSNE_MULTIINPUT_AREALAT_v2.py
# -*- coding: utf-8 -*-
"""
MARTA_INFER_TSNE_MULTIINPUT_AREALAT.py
--------------------------------------
Inferencia + t-SNE + mapa de "lateralización por área" para modelos entrenados
con MARTA_MULTIINPUT_SINGLE_TRAIN.py.

Artes de salida:
  - CSV/XLSX con resultados ROI por ROI.
  - Overlay de clasificación por contornos (JPG).
  - HTML interactivo con puntos y probabilidades sobre la imagen.
  - t-SNE (CSV + HTML con buscador de idx).
  - Heatmap suave (% clase 1) + barra de escala.
  - **Nuevo:** Overlay correcto de imagen + heatmap con barra
               [img]_lcr_overlay_withbar.jpg

"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.amp import autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from timm import create_model
from sklearn.manifold import TSNE
import plotly.graph_objs as go

# ============================
# Utilidades varias
# ============================

def ensure_dir(p: Path): # check
    p.mkdir(parents=True, exist_ok=True)

def softmax_np(z: np.ndarray) -> np.ndarray: # check
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406) # check 
IMAGENET_STD  = (0.229, 0.224, 0.225)

tf_resize_norm = A.Compose([ # check 
    A.Resize(384, 384),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

def crop_center(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray: # check 
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, min(w - size, cx - half))
    y1 = max(0, min(h - size, cy - half))
    return img[y1:y1+size, x1:x1+size]

# ============================
# Detector de regiones
# ============================

FULL_WINDOW = True # check 

def detect_all_regions(gray: np.ndarray, expand_pixels: int = 40): # check 
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

# ============================
# Construcción del modelo según checkpoint
# ============================

def build_backbone_3ch(): # check 
    # num_classes=0 + global_pool='avg' → devuelve features (GAP)
    return create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='avg')

def adapt_first_conv_to_in(backbone: nn.Module, in_ch: int):
    """Adapta conv_stem a in_ch (p.ej., 6) antes de cargar el state_dict."""
    conv = getattr(backbone, 'conv_stem', None)
    if conv is None or not isinstance(conv, nn.Conv2d):
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d):
                conv = m; break
    if conv is None:
        raise RuntimeError("No encontré Conv2d inicial para adaptar canales.")
    if conv.in_channels == in_ch:
        return backbone
    new_conv = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride,
                         conv.padding, bias=(conv.bias is not None), dilation=conv.dilation,
                         groups=conv.groups)
    with torch.no_grad():
        if conv.weight.shape[1] == 3 and in_ch > 3:
            new_conv.weight[:, :3] = conv.weight.data
            mean_w = conv.weight.data.mean(dim=1, keepdim=True)
            repeat = in_ch - 3
            new_conv.weight[:, 3:3+repeat] = mean_w.repeat(1, repeat, 1, 1)
        else:
            new_conv.weight[:] = 0.0
        if conv.bias is not None:
            new_conv.bias[:] = conv.bias.data
    
    # Reemplazar
    parent = None
    for name, module in backbone.named_children():
        if module is conv:
            parent = backbone; parent_name = name; break
    if parent is not None:
        setattr(parent, parent_name, new_conv)
    elif hasattr(backbone, 'conv_stem'):
        backbone.conv_stem = new_conv
    else:
        raise RuntimeError("No pude asignar el nuevo conv inicial.")
    return backbone

# ===== HeadMLP y build_head_from_state compatible con 'head.net.*' =====

class HeadMLP(nn.Module):  # check 
    """Head MLP con submódulo .net para que las claves 'head.net.*' del checkpoint carguen directo."""
    def __init__(self, in_feats: int, hidden: int, p_drop: float = 0.5, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def build_head_from_state(state: Dict[str, torch.Tensor], in_feats: int) -> nn.Module: # check 
    # Si el checkpoint trae 'head.net.*' → construyo HeadMLP con submódulo .net
    has_net = any(k.startswith('head.net.') for k in state.keys())
    if has_net:
        hidden = None
        out_dim = 1
        for k, v in state.items():
            if k == 'head.net.0.weight':
                hidden = v.shape[0]
            if k == 'head.net.3.weight':
                out_dim = v.shape[0]
        if hidden is None:
            hidden = 256
        return HeadMLP(in_feats, hidden, p_drop=0.5, out_dim=out_dim)

    # Caso lineal simple 'head.weight' / 'head.bias'
    for k, v in state.items():
        if k == 'head.weight':
            out_dim = v.shape[0]
            return nn.Linear(in_feats, out_dim)

    # Fallback
    return nn.Linear(in_feats, 1)

class ModelSingle3ch(nn.Module): # check 
    def __init__(self, head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.backbone = build_backbone_3ch()
        in_feats = getattr(self.backbone, 'num_features', 1280)
        self.head = build_head_from_state(head_state, in_feats)
    def forward_logits(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        if out.shape[1] == 1:
            z = out.squeeze(1)
            out = torch.stack([-z, z], dim=1)
        return out
    def forward_feats(self, x):
        return self.backbone(x)

class ModelStack6Single(nn.Module): # check 
    def __init__(self, head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.backbone = build_backbone_3ch()
        self.backbone = adapt_first_conv_to_in(self.backbone, 6)
        in_feats = getattr(self.backbone, 'num_features', 1280)
        self.head = build_head_from_state(head_state, in_feats)
    def forward_logits(self, x6):
        feats = self.backbone(x6)
        out = self.head(feats)
        if out.shape[1] == 1:
            z = out.squeeze(1)
            out = torch.stack([-z, z], dim=1)
        return out
    def forward_feats(self, x6):
        return self.backbone(x6)

class ModelDualShared(nn.Module): # check 
    def __init__(self, head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.backbone = build_backbone_3ch()
        in_feats = getattr(self.backbone, 'num_features', 1280)
        self.in_feats_total = in_feats * 2
        self.head = build_head_from_state(head_state, self.in_feats_total)
    def forward_logits(self, x6):
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        feats = torch.cat([f1, f2], dim=1)
        out = self.head(feats)
        if out.shape[1] == 1:
            z = out.squeeze(1)
            out = torch.stack([-z, z], dim=1)
        return out
    def forward_feats(self, x6):
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        return torch.cat([f1, f2], dim=1)

def load_model_from_ckpt(ckpt_path: Path, device: torch.device): # check 
    # C) weights_only=True para silenciar el FutureWarning
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = sd.get('model', sd)
    cfg = sd.get('cfg', {})
    input_mode = cfg.get('input_mode', '256')
    fusion = cfg.get('fusion', 'single')

    head_state = {k: v for k, v in state.items() if k.startswith('head.')}

    if input_mode == 'stack':
        if fusion == 'dual':
            model = ModelDualShared(head_state).to(device)
        elif fusion == 'stack6':
            model = ModelStack6Single(head_state).to(device)
        else:
            raise ValueError(f"fusion desconocida para stack: {fusion}")
    elif input_mode in ('256', '384'):
        model = ModelSingle3ch(head_state).to(device)
    else:
        model = ModelSingle3ch(head_state).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) < 20 and len(unexpected) < 20:
            if missing: print("  missing:", missing)
            if unexpected: print("  unexpected:", unexpected)

    model.eval()
    return model, cfg

# ============================
# Plotly helpers (TSNE + buscador) — FIX de etiquetas
# ============================

def write_tsne_html_with_search(out_html: Path, df_tsne: pd.DataFrame, title: str): # check 
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
# Heatmap y overlays
# ============================

def gaussian2d(h, w, cx, cy, sigma): # check 
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sigma*sigma))

def attach_colorbar_right_gray(img_gray_0_1: np.ndarray, vmin=0.0, vmax=1.0, height=None): # check 
    """Barra vertical 0-1 (gris) concatenada a la derecha de una imagen GRIS."""
    if height is None:
        height = img_gray_0_1.shape[0]
    bar = np.linspace(1.0, 0.0, height).reshape(height, 1)
    bar_rgb = (np.clip(bar, 0, 1) * 255).astype(np.uint8)
    bar_rgb = np.repeat(bar_rgb, 30, axis=1)  # ancho 30 px
    bar_rgb = cv2.cvtColor(bar_rgb, cv2.COLOR_GRAY2BGR)
    cv2.putText(bar_rgb, f"{int(vmax*100)}%", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(bar_rgb, f"{int(vmin*100)}%", (2, height-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    heat_rgb = cv2.cvtColor((img_gray_0_1*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out = np.concatenate([heat_rgb, bar_rgb], axis=1)
    return out

def attach_colorbar_right_rgb(overlay_bgr: np.ndarray, vmin=0.0, vmax=1.0): # check 
    """Genera una barra vertical TURBO 0-1 y la concatena a la derecha de una imagen RGB/BGR."""
    h = overlay_bgr.shape[0]
    bar = np.linspace(1.0, 0.0, h).reshape(h, 1)
    bar_u8 = (np.clip(bar, 0, 1) * 255).astype(np.uint8)
    bar_color = cv2.applyColorMap(bar_u8, cv2.COLORMAP_TURBO)  # BGR
    cv2.putText(bar_color, f"{int(vmax*100)}%", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(bar_color, f"{int(vmin*100)}%", (2, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return np.concatenate([overlay_bgr, bar_color], axis=1)

# ============================
# Principal
# ============================

# El main() original de Dani mezcla 10 responsabilidades. Hay que romperlo en funciones con fronteras claras.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta a la imagen .tif/.png/.jpg")
    ap.add_argument("--ckpt", required=True, help="Ruta al checkpoint Stage3 (best_stage3_full.pth)")
    ap.add_argument("--outdir", default="infer_tsne_out", help="Directorio de salida")
    ap.add_argument("--threshold", type=float, default=0.50, help="Umbral por prob. máxima (default 0.50)")
    ap.add_argument("--expand", type=int, default=40, help="Expansión de bbox (px) antes de centrar ventana 256")
    ap.add_argument("--perplexity", type=float, default=30.0, help="Perplexity para t-SNE")
    ap.add_argument("--soft", action="store_true", help="Mapa suave por kernel gaussiano ponderado por p1")
    ap.add_argument("--sigma", type=float, default=128.0, help="Sigma del kernel gaussiano (px) si --soft")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)

    # Imagen (check)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No pude leer: {args.image}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Regiones
    _, rois, areas, contours = detect_all_regions(gray, expand_pixels=args.expand)
    print(f"Regiones detectadas: {len(rois)}")

    # Modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model_from_ckpt(Path(args.ckpt), device)
    input_mode = cfg.get('input_mode', '256')
    fusion = cfg.get('fusion', 'single')
    print(f"[INFO] input_mode={input_mode} fusion={fusion}")

    # Inferencia ROI a ROI + embeddings
    feats_all = []
    results = []

    for idx, (x1,y1,x2,y2) in enumerate(rois):
        crop = image[y1:y2, x1:x2]
        valid = True
        if crop.size == 0 or (x2<=x1 or y2<=y1):
            valid = False
        if not valid:
            results.append(dict(idx=idx,x1=x1,y1=y1,x2=x2,y2=y2,valid=False,
                                prob_0=None,prob_1=None,max_prob=None,pred_raw=None,final_label=-1, cx=None, cy=None))
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = crop_rgb.shape[:2]
        cx = x1 + w//2; cy = y1 + h//2

        if input_mode == 'stack':
            roi256 = crop_center(image, cx, cy, 256)
            roi384 = crop_center(image, cx, cy, 384)
            t256 = tf_resize_norm(image=cv2.cvtColor(roi256, cv2.COLOR_BGR2RGB))['image']
            t384 = tf_resize_norm(image=cv2.cvtColor(roi384, cv2.COLOR_BGR2RGB))['image']
            ten = torch.cat([t256, t384], dim=0).unsqueeze(0).to(device)  # (1,6,H,W)
        elif input_mode in ('256', '384'):
            size = 256 if input_mode == '256' else 384
            roi = crop_center(image, cx, cy, size)
            ten = tf_resize_norm(image=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0).to(device)
        else:
            ten = tf_resize_norm(image=crop_rgb)['image'].unsqueeze(0).to(device)

        with torch.no_grad(), autocast('cuda', enabled=(device.type=='cuda')):
            logits = model.forward_logits(ten)   # (1,2)
            feats = model.forward_feats(ten)     # (1,D) o (1,2D)

        probs = softmax_np(logits.detach().cpu().numpy())[0]
        pred_raw = int(np.argmax(probs))
        max_prob = float(np.max(probs))
        final_label = pred_raw if max_prob >= args.threshold else -1

        feats_all.append(feats.squeeze(0).detach().cpu().numpy())
        results.append(dict(
            idx=idx, x1=x1, y1=y1, x2=x2, y2=y2, valid=True,
            prob_0=float(probs[0]), prob_1=float(probs[1]),
            max_prob=max_prob, pred_raw=pred_raw, final_label=final_label,
            cx=(x1+x2)//2, cy=(y1+y2)//2
        ))

    # Guardar resultados inferencia (CSV + Excel)
    base = os.path.splitext(os.path.basename(args.image))[0]
    df = pd.DataFrame(results)
    csv_path = outdir / f"{base}_resultados.csv"
    xlsx_path = outdir / f"{base}_resultados.xlsx"
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
    except Exception as e:
        print(f"[WARN] No pude escribir Excel ({e}) — quedó el CSV.")

    # Overlay coloreado (relleno por contorno)
    overlay = image.copy()
    if overlay.ndim == 2 or overlay.shape[2] == 1:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for cnt, row in zip(contours, results):
        label = row["final_label"]
        if label == 0: color = (255,0,0)     # azul
        elif label == 1: color = (0,0,255)   # rojo
        else: color = (0,255,0)              # indeciso
        cv2.drawContours(overlay, [cnt], -1, color, thickness=cv2.FILLED)
    jpg_path = outdir / f"{base}_clasificacion_coloreada.jpg"
    cv2.imwrite(str(jpg_path), overlay)

    # HTML interactivo de puntos (centroides) — FIX de etiquetas renderizando números
    try:
        import plotly.graph_objects as go
        from PIL import Image as PILImage
        df_plot = df.copy()
        df_plot['cx'] = (df_plot['x1'] + df_plot['x2']) / 2.0
        df_plot['cy'] = (df_plot['y1'] + df_plot['y2']) / 2.0
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = go.Figure()
        fig.add_layout_image(dict(
            source=PILImage.fromarray(img_rgb),
            xref="x", yref="y", x=0, y=0,
            sizex=w, sizey=h, sizing="stretch", layer="below"
        ))
        hover = [
            f"idx={int(r.idx)} | bbox=({int(r.x1)},{int(r.y1)})-({int(r.x2)},{int(r.y2)})"
            f"<br>final_label={int(r.final_label)} | pred_raw={('NA' if pd.isna(r.pred_raw) else int(r.pred_raw))}"
            f"<br>p0={('NA' if pd.isna(r.prob_0) else round(float(r.prob_0),3))}"
            f" | p1={('NA' if pd.isna(r.prob_1) else round(float(r.prob_1),3))}"
            f" | max={('NA' if pd.isna(r.max_prob) else round(float(r.max_prob),3))}"
            for r in df_plot.itertuples(index=False)
        ]
        # Mapa de colores por clase (cambiá a gusto)
        colmap = {
            -1: '#2ca02c',  # indeciso (verde)  ← por ejemplo
             0: '#1f77b4',  # clase 0 (azul)
             1: '#d62728',  # clase 1 (rojo)
        }
        colors = df_plot['final_label'].map(colmap).fillna('#7f7f7f')

        fig.add_trace(go.Scatter(
            x=df_plot['cx'], y=df_plot['cy'], mode='markers',
            marker=dict(
                size=7,
                color=colors,
                opacity=0.9,
                line=dict(width=1, color='black')  # bordecito para que se vean mejor
            ),
            text=hover, hoverinfo='text', name='ROIs'
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            width=min(1200, w), height=min(900, h),
            xaxis=dict(range=[0, w], showgrid=False, visible=False),
            yaxis=dict(range=[0, h], showgrid=False, visible=False),
            title=f"Inferencia — ROIs y probabilidades ({input_mode}/{fusion})"
        )
        html_img = outdir / f"{base}_resultados.html"
        fig.write_html(str(html_img))
    except Exception as e:
        print(f"[WARN] No pude generar HTML de imagen (instalá plotly): {e}")
        html_img = outdir / f"{base}_resultados.html"

    # =====================
    # t-SNE con Plotly + buscador idx
    # =====================
    feats_valid = [f for f, r in zip(feats_all, results) if r['valid']]
    rows_valid = [r for r in results if r['valid']]
    if len(feats_valid) == 0:
        raise RuntimeError("No hay embeddings válidos para t-SNE.")
    X = np.stack(feats_valid, axis=0)
    tsne = TSNE(n_components=2, perplexity=float(args.perplexity), init='pca',max_iter=1000, n_iter_without_progress=300,
                learning_rate='auto', random_state=42, verbose=1)
    X2 = tsne.fit_transform(X)

    df_tsne = pd.DataFrame({
        'idx': [r['idx'] for r in rows_valid],
        'tsne_x': X2[:,0], 'tsne_y': X2[:,1],
        'final_label': [r['final_label'] for r in rows_valid],
        'pred_raw': [r['pred_raw'] for r in rows_valid],
        'prob_0': [r['prob_0'] for r in rows_valid],
        'prob_1': [r['prob_1'] for r in rows_valid],
        'max_prob': [r['max_prob'] for r in rows_valid],
        'x1': [r['x1'] for r in rows_valid], 'y1': [r['y1'] for r in rows_valid],
        'x2': [r['x2'] for r in rows_valid], 'y2': [r['y2'] for r in rows_valid],
    })
    csv_tsne = outdir / f"{base}_tsne.csv"
    df_tsne.to_csv(csv_tsne, index=False)
    html_tsne = outdir / f"{base}_tsne.html"
    write_tsne_html_with_search(html_tsne, df_tsne, f"t-SNE — {base} (cfg: {input_mode}/{fusion})")

    # =====================
    # Heatmap (% clase 1) + métricas + overlays
    # =====================
    h, w = image.shape[:2]
#    if args.soft:
#        acc = np.zeros((h, w), dtype=np.float64)
#        norm = np.zeros((h, w), dtype=np.float64)
#        for r in results:
#            if not r['valid']: continue
#            cx = int((r['x1']+r['x2'])/2); cy = int((r['y1']+r['y2'])/2)
#            g = gaussian2d(h, w, cx, cy, args.sigma)
#            acc += g * float(r['prob_1'])
#            norm += g
#        heat = np.divide(acc, np.maximum(norm, 1e-8))  # 0..1
#    else:
#        heat = np.zeros((h, w), dtype=np.float32)
#        for cnt, r in zip(contours, results):
#            if r['final_label'] == 1:
#                cv2.drawContours(heat, [cnt], -1, 1.0, thickness=cv2.FILLED)
#            elif r['final_label'] == 0:
#                cv2.drawContours(heat, [cnt], -1, 0.0, thickness=cv2.FILLED)

    if args.soft:
        # mapa disperso: en cada centro pongo p1 y un peso 1
        p_map = np.zeros((h, w), dtype=np.float32)
        w_map = np.zeros((h, w), dtype=np.float32)

        for r in results:
            if not r['valid']:
                continue
            cx = int((r['x1'] + r['x2']) / 2)
            cy = int((r['y1'] + r['y2']) / 2)

            # clip por seguridad
            if 0 <= cx < w and 0 <= cy < h:
                p_map[cy, cx] += float(r['prob_1'])
                w_map[cy, cx] += 1.0

        # suavizado gaussiano separable (rápido)
        p_blur = cv2.GaussianBlur(p_map, ksize=(0, 0), sigmaX=args.sigma, sigmaY=args.sigma,
                                  borderType=cv2.BORDER_REFLECT)
        w_blur = cv2.GaussianBlur(w_map, ksize=(0, 0), sigmaX=args.sigma, sigmaY=args.sigma,
                                  borderType=cv2.BORDER_REFLECT)

        heat = p_blur / np.maximum(w_blur, 1e-8)

    else:
        # mapa duro basado en clasificación
        heat = np.zeros((h, w), dtype=np.float32)
        
        for cnt, r in zip(contours, results):
            if r['final_label'] == 1:
                cv2.drawContours(heat, [cnt], -1, 1.0, thickness=cv2.FILLED)
            elif r['final_label'] == 0:
                cv2.drawContours(heat, [cnt], -1, 0.0, thickness=cv2.FILLED)

    
    # % global sobre máscara de ROIs (conexina total)
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask_roi, [cnt], -1, 1, thickness=cv2.FILLED)

    # Denominador sólo con contornos confiables (final_label en {0,1})
    mask_conf = np.zeros((h, w), np.uint8)
    for cnt, r in zip(contours, results):
        if r['final_label'] in (0, 1):
            cv2.drawContours(mask_conf, [cnt], -1, 1, thickness=cv2.FILLED)

    denom_all  = int((mask_roi  > 0).sum())   # incluye indecisos
    denom_conf = int((mask_conf > 0).sum())   # excluye indecisos

    if args.soft:
        # promedio de heat (p1) sobre cada denominador
        p_global_all  = float(np.sum(heat * (mask_roi  > 0)) / denom_all)   if denom_all  > 0 else float('nan')
        p_global_excl = float(np.sum(heat * (mask_conf > 0)) / denom_conf)  if denom_conf > 0 else float('nan')
    else:
        # HARD: proporción de píxeles clase 1 / denominador
        p_global_all  = float(np.sum((heat > 0.5) & (mask_roi  > 0)) / denom_all)   if denom_all  > 0 else float('nan')
        p_global_excl = float(np.sum((heat > 0.5) & (mask_conf > 0)) / denom_conf)  if denom_conf > 0 else float('nan')

    print(f"[INFO] % global (incluye indecisos): {p_global_all*100:.3f}%")
    print(f"[INFO] % global (EXCLUYE indecisos): {p_global_excl*100:.3f}%")
    print(f"[DBG] denom_all={denom_all} denom_conf={denom_conf}")

    # Métrica “dura” por área de conexina (independiente del heatmap):
    mask_cls1 = np.zeros((h, w), np.uint8)
    for cnt, r in zip(contours, results):
        if r['final_label'] == 1:
            cv2.drawContours(mask_cls1, [cnt], -1, 1, thickness=cv2.FILLED)

    area1 = int(mask_cls1.sum())
    areaT_all = int((mask_roi > 0).sum())  # área total de conexina, incluye indecisos
    p_global_area = area1 / areaT_all if areaT_all > 0 else float('nan')
    print(f"[INFO] % área clase1/total (morf. dura): {p_global_area*100:.3f}%")

    # Guardar heatmap con barra (gris) y overlay color + barra (nuevo)
    heat_01 = (heat - np.nanmin(heat)) / max(1e-8, (np.nanmax(heat) - np.nanmin(heat)))
    heat_withbar = attach_colorbar_right_gray(heat_01, vmin=0.0, vmax=1.0, height=h)
    heat_jpg = outdir / f"{base}_heatmap_arealateralizacion.jpg"
    cv2.imwrite(str(heat_jpg), heat_withbar)

    # Overlay imagen + heatmap coloreado TURBO
    heat_u8 = (np.clip(heat_01, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)  # BGR
    # alpha-blend
    alpha = 0.4
    overlay_lcr = cv2.addWeighted(image, 1.0 - alpha, heat_color, alpha, 0)
    overlay_withbar = attach_colorbar_right_rgb(overlay_lcr, vmin=0.0, vmax=1.0)
    lcr_path = outdir / f"{base}_lcr_overlay_withbar.jpg"
    cv2.imwrite(str(lcr_path), overlay_withbar)

    # Resumen
    n_total = len(results)
    n0 = sum(1 for r in results if r['final_label'] == 0)
    n1 = sum(1 for r in results if r['final_label'] == 1)
    n_ind = sum(1 for r in results if r['final_label'] == -1)
    summary = dict(
        image=args.image, ckpt=str(args.ckpt), threshold=float(args.threshold),
        total=n_total, clase0=n0, clase1=n1, indeciso=n_ind,
        out_dir=str(outdir.resolve()),
        csv=str(csv_path), xlsx=str(xlsx_path),
        overlay=str(jpg_path), html_img=str(html_img),
        tsne_csv=str(csv_tsne), tsne_html=str(html_tsne),
        heatmap=str(heat_jpg),
        lcr_overlay=str(lcr_path),
        input_mode=input_mode, fusion=fusion,
        soft=args.soft, sigma=float(args.sigma),
        pct_global_lateralizacion=float(p_global_all*100.0),                 # incluye indecisos
        pct_global_lateralizacion_excl_indecisos=float(p_global_excl*100.0), # excluye indecisos
        pct_global_lateralizacion_area_hard=float(p_global_area*100.0),
        denom_all=int(denom_all), denom_excl=int(denom_conf)                 # opcional para debug
        

    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print("\\n== LISTO ==")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
