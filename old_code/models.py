
# src/train/models.py

# este script lo separaría igual en dos: architecture.py y model_builder.py

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# no me gusta el base, son las cosas que vamos a tener que refactorizar
from .legacy_train_base import (
    build_backbone,
    freeze_backbone,
    unfreeze_last_k_blocks,
    unfreeze_all_backbone,
)
 
def build_backbone_3ch():
    """
    Código no modificado: Wrapper fino sobre el backbone reutilizado desde el código base de Dani. ESTO NO ME GUSTA
    """
    return build_backbone()

def adapt_first_conv_to_6ch(backbone: nn.Module):
    """
    Código no modificado: Adapta la primera convolución del backbone para aceptar 6 canales.

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

class HeadLogistic(nn.Module):
    """
    Código no modificado: Cabeza de regresión logística binaria: Linear -> 1 logit
    """
    def __init__(self, in_feats: int):
        super().__init__()
        self.fc = nn.Linear(in_feats, 1)

    def forward(self, x): return self.fc(x).squeeze(1)

class HeadMLP(nn.Module):
    """
    Código no modificado: Cabeza MLP binaria: Linear -> SiLU -> Dropout -> Linear -> 1 logit
    """
    def __init__(self, in_feats: int, hidden: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop), # me gustaría que para el tuning esto se pueda tocar mas desde el init
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

class ModelSingleStream(nn.Module):
    """
    Código no modificado: Modelo de un solo flujo:
      - input_mode '256' o '384'
      - o 'stack' con fusion='stack6'
    """
    def __init__(
        self, 
        head_type: str, 
        hidden: Optional[int], 
        p_drop: Optional[float], 
        fusion: str = "single"
    ):

        super().__init__()
        self.fusion = fusion
        self.backbone = build_backbone_3ch()

        if self.fusion == 'stack6':
            self.backbone = adapt_first_conv_to_6ch(self.backbone)

        in_feats = getattr(self.backbone, 'num_features', 1280)

        if head_type == 'logreg':
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
    Código no modificado:  Modelo dual con backbone compartido:
      - recibe tensor de 6 canales
      - separa en dos vistas RGB
      - extrae features por separado
      - concatena y pasa por la head
    """
    def __init__(
        self, 
        head_type: str, 
        hidden: Optional[int], 
        p_drop: Optional[float]
    ):
        super().__init__()
        self.backbone = build_backbone_3ch()
        in_feats = getattr(self.backbone, 'num_features', 1280)

        if head_type == 'logreg':
            self.head = HeadLogistic(in_feats*2)
        else:
            if hidden is None or p_drop is None:
                raise ValueError("Para head_type='mlp' se requieren hidden y p_drop.")
            self.head = HeadMLP(in_feats*2, hidden, p_drop)

    def forward(self, x6):
        x256, x384 = torch.split(x6, 3, dim=1)
        f1 = self.backbone(x256)
        f2 = self.backbone(x384)
        feats = torch.cat([f1, f2], dim=1)
        return self.head(feats)

def build_model( 
    head_kind: str, 
    hidden: Optional[int], 
    p_drop: Optional[float], 
    input_mode: str, 
    fusion: str,
    device: torch.device
):
    """
    Código modificado: He añadido el device dentro de los argumentos de entrada.
    Construye el modelo final según:
      - tipo de head
      - input_mode
      - estrategia de fusión
    """
    if input_mode in ('256','384'):
        return ModelSingleStream(head_kind, hidden, p_drop, fusion='single').to(device)
    
    elif input_mode == 'stack':
        if fusion == 'dual':
            return ModelDualStream(head_kind, hidden, p_drop).to(device)
        elif fusion == 'stack6':
            return ModelSingleStream(head_kind, hidden, p_drop, fusion='stack6').to(device)
        else:
            raise ValueError("fusion debe ser 'dual' o 'stack6'")
    
    else:
        raise ValueError("input_mode inválido")

# no me gusta que estas funciones sigan dependiendo de base. Modificar. Y esto
# igual no lo metería en models.py
def freeze_for_stage(model: nn.Module, stage: int, k_unf: int = 1):
    """
    Código modificado: introducir k_unf como argumento de entrada que no dependa de cfg
    Aplica la política de congelación/descongelación por fase.

    stage 1:
      - backbone congelado
      - entrena solo la head

    stage 2:
      - descongela últimos k bloques del backbone

    stage 3:
      - descongela todo el backbone
    """
    if stage == 1:
        try:
            freeze_backbone(model)
        except Exception:
            for p in model.backbone.parameters():
                p.requires_grad = False

    elif stage == 2:
        try:
            unfreeze_last_k_blocks(model, k=k_unf)
        except Exception:
            for p in model.backbone.parameters():
                p.requires_grad = True

    elif stage == 3:
        try:
            unfreeze_all_backbone(model)
        except Exception:
            for p in model.backbone.parameters():
                p.requires_grad = True

    else:
        raise ValueError(f"stage no soportado: {stage}")

def param_groups(model: nn.Module, stage: int, k_unf: int = 1):
    """
    Código modificado: introducir k_unf como argumento de entrada que no dependa de cfg
    
    Devuelve grupos de parámetros para aplicar learning rates distintos. NO SE COMO VAMOS A JUGAR CON ESTO EN EL TUNING

    Salida:
      - head_params
      - last_params
      - rest_params
    
    stage=1:
      solo interesa head_params

    stage=2 o 3:
      se separan últimos bloques y resto del backbone
    """
    head_params = list(model.head.parameters())
    last_params: List[nn.Parameter] = []; rest_params: List[nn.Parameter] = []
    
    if stage == 1:
        pass

    else:
        children = list(model.backbone.children())
        if children:
            last_children = set(children[-k_unf:])
            for ch in children:
                for p in ch.parameters():
                    if not p.requires_grad: continue
                    (last_params if ch in last_children else rest_params).append(p)
        else:
            for p in model.backbone.parameters():
                if p.requires_grad: rest_params.append(p)
    return head_params, last_params, rest_params