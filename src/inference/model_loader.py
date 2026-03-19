
# src/inference/model_loader.py -> Solo reconstrucción del modelo desde checkpoint.
# “Tengo un .pth, ¿cómo reconstruyo el modelo correcto para inferencia?”

# IMPORTANTE: para la segunda refactorizacion al quitar el base
# podria venir bien exponer /models e intentar que no
# haya un model de train y otro de inferencia si la clase
# es practicamente la misma para no ser redudantes con el code.

"""
model_loader.py
---------------
Reconstrucción del modelo de inferencia de MARTA a partir de un checkpoint.

Responsabilidades:
  - construir backbone base
  - adaptar la primera convolución si el modelo usa 6 canales
  - reconstruir la head desde el state_dict
  - reconstruir la arquitectura correcta según input_mode / fusion
  - cargar pesos y devolver modelo listo para inferencia

Notas:
  - Este módulo conserva la lógica del script original de Dani para inferencia.
  - No debe encargarse de leer imágenes, detectar ROIs ni generar outputs.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from timm import create_model

def load_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Carga un checkpoint y reconstruye el modelo correcto para inferencia.

    Devuelve:
      - model
      - cfg (diccionario guardado dentro del checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True) 
        # weights_only=True para silenciar el FutureWarning

    state_dict = checkpoint.get("model", checkpoint)
    cfg = checkpoint.get("cfg", {})

    input_mode = cfg.get("input_mode", "256")
    fusion = cfg.get("fusion", "single")

    head_state = {k: v for k, v in state_dict.items() if k.startswith("head.")}

    if input_mode == "stack":
        if fusion == "dual":
            model = InferenceModelDualShared(head_state).to(device)
        elif fusion == "stack6":
            model = InferenceModelStack6Single(head_state).to(device)
        else:
            raise ValueError(f"fusion desconocida para stack: {fusion}")
    elif input_mode in ("256", "384"):
        model = InferenceModelSingle3Ch(head_state).to(device)
    else:
        model = InferenceModelSingle3Ch(head_state).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) < 20 and missing:
            print("  missing:", missing)
        if len(unexpected) < 20 and unexpected:
            print("  unexpected:", unexpected)

    model.eval()
    return model, cfg

class InferenceModelSingle3Ch(nn.Module): # class ModelSingle3ch(nn.Module):
    """
    Yo sí le cambiaría el nombre para dejar claro que esta clase es de inferencia y no del training module.
    """
    def __init__(self, head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.backbone = build_backbone_3ch()
        in_features = getattr(self.backbone, 'num_features', 1280)
        self.head = build_head_from_state(head_state, in_features)
    
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats)

        if out.shape[1] == 1:
            z = out.squeeze(1)
            out = torch.stack([-z, z], dim=1)
        return out
    
    def forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class InferenceModelStack6Single(nn.Module): # class ModelStack6Single(nn.Module):
    def __init__(self, head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.backbone = build_backbone_3ch()
        self.backbone = adapt_first_conv_to_in(self.backbone, 6)
        in_features = getattr(self.backbone, 'num_features', 1280)
        self.head = build_head_from_state(head_state, in_features)

    def forward_logits(self, x6: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x6)
        out = self.head(feats)

        if out.shape[1] == 1:
            z = out.squeeze(1)
            out = torch.stack([-z, z], dim=1)
        return out

    def forward_feats(self, x6: torch.Tensor) -> torch.Tensor:
        return self.backbone(x6)

class InferenceModelDualShared(nn.Module): # class ModelDualShared(nn.Module):
    def __init__(self, head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.backbone = build_backbone_3ch()
        in_features = getattr(self.backbone, 'num_features', 1280)
        #self.in_feats_total = in_features * 2
        self.head = build_head_from_state(head_state, in_features * 2)

    def forward_logits(self, x6: torch.Tensor) -> torch.Tensor:
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        feats = torch.cat([f1, f2], dim=1)

        out = self.head(feats)
        if out.shape[1] == 1:
            z = out.squeeze(1)
            out = torch.stack([-z, z], dim=1)
        return out

    def forward_feats(self, x6: torch.Tensor) -> torch.Tensor:
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        return torch.cat([f1, f2], dim=1)

# ============================
# Construcción del modelo según checkpoint
# ============================
# de aquí para abajo no estoy convencido que aguanten a una segunda refactorizacion

def build_backbone_3ch():
    """
    Construye EfficientNetV2-S sin cabeza final, devolviendo features tras global pooling.
    # num_classes=0 + global_pool='avg' → devuelve features (GAP)
    """
    return create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='avg')
# Nota
# Podrías reutilizar el backbone del training, pero para no introducir dependencias raras con legacy_train_base, aquí tiene sentido mantenerlo local.


def adapt_first_conv_to_in(backbone: nn.Module, in_channels: int) -> nn.Module:
    """
    Adapta conv_stem a in_channels (p.ej., 6) antes de cargar el state_dict.
   
    Usado principalmente para el modo stack6 (6 canales).
    """
    conv = getattr(backbone, "conv_stem", None)

    if conv is None or not isinstance(conv, nn.Conv2d):
        for module in backbone.modules():
            if isinstance(module, nn.Conv2d):
                conv = module
                break

    if conv is None:
        raise RuntimeError("No encontré Conv2d inicial para adaptar canales.")

    if conv.in_channels == in_channels:
        return backbone

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=(conv.bias is not None),
        dilation=conv.dilation,
        groups=conv.groups,
    )

    with torch.no_grad():
        if conv.weight.shape[1] == 3 and in_channels > 3:
            new_conv.weight[:, :3] = conv.weight.data
            mean_w = conv.weight.data.mean(dim=1, keepdim=True)
            repeat = in_channels - 3
            new_conv.weight[:, 3:3 + repeat] = mean_w.repeat(1, repeat, 1, 1)
        else:
            new_conv.weight[:] = 0.0

        if conv.bias is not None:
            new_conv.bias[:] = conv.bias.data

    parent = None
    parent_name = None

    for name, module in backbone.named_children():
        if module is conv:
            parent = backbone
            parent_name = name
            break

    if parent is not None:
        setattr(parent, parent_name, new_conv)
    elif hasattr(backbone, "conv_stem"):
        backbone.conv_stem = new_conv
    else:
        raise RuntimeError("No pude asignar el nuevo conv inicial.")

    return backbone


# ===== HeadMLP y build_head_from_state compatible con 'head.net.*' =====
class HeadMLP(nn.Module):
    """
    Head MLP con submódulo .net para compatibilidad con checkpoints
    que contienen claves tipo 'head.net.*'.
    """

    def __init__(self, in_features: int, hidden: int, p_drop: float = 0.5, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def build_head_from_state(
    state_dict: Dict[str, torch.Tensor],
    in_features: int,
) -> nn.Module:
    """
    Reconstruye la head en función de las claves presentes en el checkpoint.

    Casos soportados:
      - head.net.*  -> MLP
      - head.weight -> linear simple
    """
    has_net = any(k.startswith("head.net.") for k in state_dict.keys())

    if has_net:
        hidden = None
        out_dim = 1

        for key, value in state_dict.items():
            if key == "head.net.0.weight":
                hidden = value.shape[0]
            if key == "head.net.3.weight":
                out_dim = value.shape[0]

        if hidden is None:
            hidden = 256

        return HeadMLP(in_features, hidden, p_drop=0.5, out_dim=out_dim)

    for key, value in state_dict.items():
        if key == "head.weight":
            out_dim = value.shape[0]
            return nn.Linear(in_features, out_dim)

    return nn.Linear(in_features, 1)

# Tu esquema funcional encaja perfecto:

# - leer imágenes
# - ordenar secciones
# - convertir si hace falta
# - detectar ROIs
# - cargar modelo
# - preparar crops
# - inferir ROI a ROI
# - guardar tablas
# - generar overlays
# - t-SNE
# - heatmap
# - métricas

# outputs finales
# src/inference/
# ├── model_loader.py
# ├── roi_inference.py
# ├── analysis.py
# ├── single_image.py
# └── batch_grid.py

# src/utils/config.py
# src/utils/io.py
