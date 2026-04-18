# expert_annotation_tool.py
"""
Herramienta interactiva de anotacion experta para ROIs de CX43.

Detecta ROIs usando EXACTAMENTE los mismos parametros del pipeline de
inferencia (lee inference.yaml), lanza un servidor local, y permite al
experto navegar la imagen, clickear ROIs, y clasificarlas.

Las anotaciones se guardan progresivamente en un CSV.

Uso:
    python expert_annotation_tool.py --image <path> --outdir <path>
                                     [--config configs/inference.yaml]
                                     [--port 5000] [--resume]

Luego abrir http://localhost:5000 en el navegador.

Requisitos:
    pip install flask
"""
from __future__ import annotations
import argparse
import csv
import io
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, render_template_string, jsonify, request, send_file

from src.preprocessing.roi_detection import detect_all_regions
from src.preprocessing.crops import crop_center
from src.utils.config import load_inference_config


# ============================
# Argument parsing
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="Expert ROI annotation tool for CX43 regions.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--outdir", required=True, help="Output directory for annotations.")
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml",
        help="Path to inference YAML (used to match inference ROI detection).",
    )
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000).")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing annotations CSV if present.",
    )
    return parser.parse_args()


# ============================
# Global state
# ============================
app = Flask(__name__)
STATE = {
    "image_bgr": None,
    "image_path": None,
    "rois": [],
    "contours": [],
    "annotations": {},  # idx -> label
    "csv_path": None,
    "crop_local": 256,
    "crop_context": 512,
    "lock": Lock(),
}


# ============================
# Image serving
# ============================
@app.route("/image_full")
def serve_full_image():
    _, buf = cv2.imencode(".jpg", STATE["image_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/crop/<int:idx>/<string:kind>")
def serve_crop(idx, kind):
    """kind in {'local', 'context'}."""
    if idx < 0 or idx >= len(STATE["rois"]):
        return "Invalid ROI index", 404

    x1, y1, x2, y2 = STATE["rois"][idx]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if kind == "local":
        size = STATE["crop_local"]
    elif kind == "context":
        size = STATE["crop_context"]
    else:
        return "Invalid crop kind", 400

    crop = crop_center(STATE["image_bgr"], cx, cy, size)
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


# ============================
# ROI data API
# ============================
@app.route("/rois")
def get_rois():
    data = []
    for i, (x1, y1, x2, y2) in enumerate(STATE["rois"]):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        label = STATE["annotations"].get(i, None)
        data.append({
            "idx": i,
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "cx": int(cx), "cy": int(cy),
            "label": label,
        })
    return jsonify(data)


@app.route("/annotate", methods=["POST"])
def annotate():
    body = request.json
    idx = body.get("idx")
    label = body.get("label")

    if idx is None or label is None:
        return jsonify({"error": "Missing idx or label"}), 400

    idx = int(idx)
    label = int(label)

    if idx < 0 or idx >= len(STATE["rois"]):
        return jsonify({"error": "Invalid ROI index"}), 400

    if label not in (0, 1, -1):
        return jsonify({"error": "Label must be 0, 1, or -1"}), 400

    with STATE["lock"]:
        STATE["annotations"][idx] = label
        _save_csv()

    n_annotated = len(STATE["annotations"])
    n_total = len(STATE["rois"])
    return jsonify({
        "ok": True,
        "idx": idx,
        "label": label,
        "annotated": n_annotated,
        "total": n_total,
    })


@app.route("/undo", methods=["POST"])
def undo():
    body = request.json
    idx = int(body.get("idx", -1))

    with STATE["lock"]:
        if idx in STATE["annotations"]:
            del STATE["annotations"][idx]
            _save_csv()

    return jsonify({"ok": True, "idx": idx})


@app.route("/stats")
def stats():
    annotations = STATE["annotations"]
    return jsonify({
        "total": len(STATE["rois"]),
        "annotated": len(annotations),
        "class_0": sum(1 for v in annotations.values() if v == 0),
        "class_1": sum(1 for v in annotations.values() if v == 1),
        "class_neg1": sum(1 for v in annotations.values() if v == -1),
    })


# ============================
# CSV persistence
# ============================
def _save_csv():
    csv_path = STATE["csv_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "x1", "y1", "x2", "y2", "cx", "cy", "expert_label"])
        for idx in sorted(STATE["annotations"].keys()):
            x1, y1, x2, y2 = STATE["rois"][idx]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            writer.writerow([idx, x1, y1, x2, y2, cx, cy, STATE["annotations"][idx]])


def _load_csv(csv_path):
    annotations = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations[int(row["idx"])] = int(row["expert_label"])
    return annotations


# ============================
# HTML Frontend
# ============================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MARTA Expert Annotation Tool</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: Arial, sans-serif; background: #1a1a1a; color: #eee; overflow: hidden; }

#toolbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
    background: #2a2a2a; padding: 8px 16px; display: flex;
    align-items: center; gap: 20px; font-size: 14px;
    border-bottom: 1px solid #444;
}
#toolbar .stat { color: #aaa; }
#toolbar .stat b { color: #fff; }
#toolbar .legend { display: flex; gap: 12px; margin-left: auto; }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }

#viewer {
    position: absolute; top: 40px; left: 0; right: 0; bottom: 0;
    overflow: auto; cursor: crosshair;
}
#viewer canvas { display: block; }

#popup-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 2000;
    justify-content: center; align-items: center;
}
#popup-overlay.active { display: flex; }

#popup {
    background: #2a2a2a; border-radius: 8px; padding: 16px;
    max-width: 90vw; max-height: 90vh; overflow: auto;
    border: 1px solid #555;
}
#popup h3 { margin-bottom: 12px; font-size: 16px; }
#popup .crops { display: flex; gap: 12px; margin-bottom: 12px; }
#popup .crops img { border: 1px solid #555; background: #000; }
#popup .crop-label { text-align: center; font-size: 12px; color: #aaa; margin-top: 4px; }
#popup .buttons { display: flex; gap: 10px; justify-content: center; }
#popup .buttons button {
    padding: 10px 24px; font-size: 15px; border: none; border-radius: 4px;
    cursor: pointer; font-weight: bold;
}
.btn-term { background: #1f77b4; color: white; }
.btn-lat  { background: #d62728; color: white; }
.btn-duda { background: #2ca02c; color: white; }
.btn-undo { background: #666; color: white; }
.btn-skip { background: #444; color: #aaa; }
#popup .hint { text-align: center; margin-top: 8px; font-size: 12px; color: #888; }

#zoom-info {
    position: fixed; bottom: 10px; right: 16px; z-index: 1000;
    background: rgba(0,0,0,0.6); padding: 4px 10px; border-radius: 4px;
    font-size: 12px; color: #aaa;
}
</style>
</head>
<body>

<div id="toolbar">
    <b>MARTA Annotation Tool</b>
    <span class="stat">ROIs: <b id="stat-annotated">0</b> / <b id="stat-total">0</b></span>
    <span class="stat">Terminal: <b id="stat-c0" style="color:#1f77b4">0</b></span>
    <span class="stat">Lateral: <b id="stat-c1" style="color:#d62728">0</b></span>
    <span class="stat">Dudosa: <b id="stat-cd" style="color:#2ca02c">0</b></span>
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#888"></div> Sin anotar</div>
        <div class="legend-item"><div class="legend-dot" style="background:#1f77b4"></div> Terminal (0)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#d62728"></div> Lateral (1)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2ca02c"></div> Dudosa (-1)</div>
    </div>
</div>

<div id="viewer">
    <canvas id="canvas"></canvas>
</div>

<div id="popup-overlay">
    <div id="popup">
        <h3 id="popup-title">ROI #0</h3>
        <div class="crops">
            <div>
                <img id="crop-local" width="256" height="256">
                <div class="crop-label">Local</div>
            </div>
            <div>
                <img id="crop-context" width="384" height="384">
                <div class="crop-label">Context</div>
            </div>
        </div>
        <div class="buttons">
            <button class="btn-term" onclick="annotate(0)">0 - Terminal</button>
            <button class="btn-lat" onclick="annotate(1)">1 - Lateralizada</button>
            <button class="btn-duda" onclick="annotate(-1)">D - Dudosa</button>
            <button class="btn-undo" onclick="undoAnnotation()">Deshacer</button>
            <button class="btn-skip" onclick="closePopup()">ESC - Cerrar</button>
        </div>
        <div class="hint">Teclado: 0 = terminal, 1 = lateral, D = dudosa, Z = deshacer, ESC = cerrar</div>
    </div>
</div>

<div id="zoom-info">Zoom: <span id="zoom-level">100</span>%  |  Scroll para zoom  |  Click en ROI para anotar</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const viewer = document.getElementById('viewer');

let img = new Image();
let rois = [];
let scale = 1.0;
let currentRoiIdx = null;

const COLORS = {null: '#ffff00', 0: '#1f77b4', 1: '#d62728', '-1': '#2ca02c'};

img.onload = function() {
    scale = Math.min(viewer.clientWidth / img.width, viewer.clientHeight / img.height, 1.0);
    drawAll();
    loadROIs();
};
img.src = '/image_full';

function drawAll() {
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    for (const roi of rois) {
        const cx = roi.cx * scale;
        const cy = roi.cy * scale;
        const color = COLORS[roi.label] || COLORS[null];
        const radius = roi.label !== null ? 9 : 7;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        ctx.fill();
        ctx.globalAlpha = 1.0;

        if (roi.label !== null) {
            const x = roi.x1 * scale;
            const y = roi.y1 * scale;
            const w = (roi.x2 - roi.x1) * scale;
            const h = (roi.y2 - roi.y1) * scale;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.4;
            ctx.strokeRect(x, y, w, h);
            ctx.globalAlpha = 1.0;
        }
    }
    document.getElementById('zoom-level').textContent = Math.round(scale * 100);
}

function loadROIs() {
    fetch('/rois').then(r => r.json()).then(data => {
        rois = data;
        drawAll();
        updateStats();
    });
}

function updateStats() {
    fetch('/stats').then(r => r.json()).then(s => {
        document.getElementById('stat-total').textContent = s.total;
        document.getElementById('stat-annotated').textContent = s.annotated;
        document.getElementById('stat-c0').textContent = s.class_0;
        document.getElementById('stat-c1').textContent = s.class_1;
        document.getElementById('stat-cd').textContent = s.class_neg1;
    });
}

viewer.addEventListener('wheel', function(e) {
    e.preventDefault();
    const rect = viewer.getBoundingClientRect();
    const mouseX = e.clientX - rect.left + viewer.scrollLeft;
    const mouseY = e.clientY - rect.top + viewer.scrollTop;

    const oldScale = scale;
    const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
    scale = Math.max(0.05, Math.min(5.0, scale * factor));

    const ratio = scale / oldScale;
    viewer.scrollLeft = mouseX * ratio - (e.clientX - rect.left);
    viewer.scrollTop = mouseY * ratio - (e.clientY - rect.top);

    drawAll();
}, {passive: false});

canvas.addEventListener('click', function(e) {
    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / scale;
    const clickY = (e.clientY - rect.top) / scale;

    let bestIdx = -1;
    let bestDist = Infinity;
    const maxDist = 40 / scale;

    for (const roi of rois) {
        const dx = roi.cx - clickX;
        const dy = roi.cy - clickY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < bestDist && dist < maxDist) {
            bestDist = dist;
            bestIdx = roi.idx;
        }
    }

    if (bestIdx === -1) {
        for (const roi of rois) {
            if (clickX >= roi.x1 && clickX <= roi.x2 && clickY >= roi.y1 && clickY <= roi.y2) {
                bestIdx = roi.idx;
                break;
            }
        }
    }

    if (bestIdx >= 0) {
        openPopup(bestIdx);
    }
});

function openPopup(idx) {
    currentRoiIdx = idx;
    const roi = rois.find(r => r.idx === idx);
    document.getElementById('popup-title').textContent =
        `ROI #${idx}  (${roi.x1}, ${roi.y1}) -> (${roi.x2}, ${roi.y2})` +
        (roi.label !== null ? `  [current: ${roi.label}]` : '');
    document.getElementById('crop-local').src = `/crop/${idx}/local`;
    document.getElementById('crop-context').src = `/crop/${idx}/context`;
    document.getElementById('popup-overlay').classList.add('active');
}

function closePopup() {
    document.getElementById('popup-overlay').classList.remove('active');
    currentRoiIdx = null;
}

function annotate(label) {
    if (currentRoiIdx === null) return;
    fetch('/annotate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({idx: currentRoiIdx, label: label}),
    }).then(r => r.json()).then(data => {
        const roi = rois.find(r => r.idx === currentRoiIdx);
        if (roi) roi.label = label;
        drawAll();
        updateStats();
        closePopup();
    });
}

function undoAnnotation() {
    if (currentRoiIdx === null) return;
    fetch('/undo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({idx: currentRoiIdx}),
    }).then(r => r.json()).then(data => {
        const roi = rois.find(r => r.idx === currentRoiIdx);
        if (roi) roi.label = null;
        drawAll();
        updateStats();
        closePopup();
    });
}

document.addEventListener('keydown', function(e) {
    if (!document.getElementById('popup-overlay').classList.contains('active')) return;

    if (e.key === '0') annotate(0);
    else if (e.key === '1') annotate(1);
    else if (e.key === 'd' || e.key === 'D') annotate(-1);
    else if (e.key === 'z' || e.key === 'Z') undoAnnotation();
    else if (e.key === 'Escape') closePopup();
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closePopup();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ============================
# Main
# ============================
def main():
    args = parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] Loading inference config: {args.config}")
    infer_cfg = load_inference_config(args.config)

    print(f"[read] Loading image: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image_bgr.shape[:2]
    print(f"[read] Image size: {w} x {h}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    print(
        f"[detect] Running ROI detection (thresh={infer_cfg.thresh_value} "
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
    print(f"[detect] Detected {len(rois)} ROIs")

    # Determinar tamanos de crop efectivos (igual que single_image)
    crop_local = infer_cfg.crop_local if infer_cfg.crop_local is not None else 256
    crop_context = infer_cfg.crop_context if infer_cfg.crop_context is not None else 512
    print(f"[crops] Using crop_local={crop_local}, crop_context={crop_context}")

    csv_path = outdir / f"{image_path.stem}_expert_annotations.csv"

    STATE["image_bgr"] = image_bgr
    STATE["image_path"] = image_path
    STATE["rois"] = rois
    STATE["contours"] = contours
    STATE["csv_path"] = csv_path
    STATE["crop_local"] = crop_local
    STATE["crop_context"] = crop_context

    if args.resume and csv_path.exists():
        STATE["annotations"] = _load_csv(csv_path)
        print(f"[resume] Loaded {len(STATE['annotations'])} existing annotations from {csv_path}")
    else:
        STATE["annotations"] = {}

    print(f"\n{'='*60}")
    print(f"  MARTA Expert Annotation Tool")
    print(f"  Image: {image_path.name}")
    print(f"  ROIs detected: {len(rois)}")
    print(f"  Annotations CSV: {csv_path}")
    print(f"  Open in browser: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
