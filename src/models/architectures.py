
# src/models/architectures.py

from typing import Optional

import torch
import torch.nn as nn

from .backbones import build_backbone, adapt_first_conv_to_6ch
from .heads import HeadLogistic, HeadMLP

class ModelSingleStream(nn.Module):
    """
    Modelo de un solo flujo:
      - input_mode '256' o '384'
      - o 'stack' con fusion='stack6'
    """
    def __init__(
        self,
        head_type: str,
        hidden: Optional[int],
        p_drop: Optional[float],
        fusion: str = "single",
    ):
        super().__init__()
        self.fusion = fusion
        self.backbone = build_backbone()

        if self.fusion == "stack6":
            self.backbone = adapt_first_conv_to_6ch(self.backbone)

        in_feats = getattr(self.backbone, "num_features", 1280)

        if head_type == "logreg":
            self.head = HeadLogistic(in_feats)
        else:
            if hidden is None or p_drop is None:
                raise ValueError("Para head_type='mlp' se requieren hidden y p_drop.")
            self.head = HeadMLP(in_feats, hidden, p_drop)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class ModelDualStream(nn.Module):
    """
    Modelo dual con backbone compartido:
      - recibe tensor de 6 canales
      - separa en dos vistas RGB
      - extrae features por separado
      - concatena y pasa por la head
    """
    def __init__(
        self,
        head_type: str,
        hidden: Optional[int],
        p_drop: Optional[float],
    ):
        super().__init__()
        self.backbone = build_backbone()
        in_feats = getattr(self.backbone, "num_features", 1280)

        if head_type == "logreg":
            self.head = HeadLogistic(in_feats * 2)
        else:
            if hidden is None or p_drop is None:
                raise ValueError("Para head_type='mlp' se requieren hidden y p_drop.")
            self.head = HeadMLP(in_feats * 2, hidden, p_drop)

    def forward(self, x6):
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        feats = torch.cat([f1, f2], dim=1)
        return self.head(feats)