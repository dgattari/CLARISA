
# src/models/backbones.py

from typing import Optional

import torch
import torch.nn as nn
from timm import create_model

# ==========================================================
# Backbone builders
# ==========================================================

def build_backbone() -> nn.Module:
    """
    EfficientNetV2-S backbone sin cabeza final (features), con pooling global.
    Conserva la elección original de backbone del código base.
    """
    backbone = create_model(
        "tf_efficientnetv2_s",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )
    return backbone


def adapt_first_conv_to_6ch(backbone: nn.Module):
    """
    Adapta la primera convolución del backbone para aceptar 6 canales.

    Estrategia heredada del código original:
      - copia los pesos RGB originales a los 3 primeros canales
      - inicializa los 3 canales extra con la media de los pesos RGB
    """
    conv = getattr(backbone, 'conv_stem', None)
    
    if conv is None or not isinstance(conv, nn.Conv2d):
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d):
                conv = m; break

    if conv is None:
        raise RuntimeError("No encontré Conv2d inicial para adaptar a 6 canales.")

    w = conv.weight.data
    out_ch, in_ch, k1, k2 = w.shape

    if in_ch == 6:  # ya
        return backbone

    if in_ch != 3:
        raise RuntimeError(f"Conv stem in_ch={in_ch}, esperado 3.")

    new_w = torch.zeros((out_ch, 6, k1, k2), dtype=w.dtype, device=w.device)
    new_w[:, :3, :, :] = w
    mean_w = w.mean(dim=1, keepdim=True)
    new_w[:, 3:, :, :] = mean_w.repeat(1, 3, 1, 1)

    with torch.no_grad():
        conv.in_channels = 6
        conv.weight = nn.Parameter(new_w)

    return backbone