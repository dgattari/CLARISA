
# src/models/builder.py

from typing import Optional

import torch
import torch.nn as nn

from .architectures import ModelSingleStream, ModelDualStream

def build_model(
    head_kind: str,
    hidden: Optional[int],
    p_drop: Optional[float],
    input_mode: str,
    fusion: str,
    device: torch.device,
):
    """
    Construye el modelo final según:
      - tipo de head
      - input_mode
      - estrategia de fusión
    """
    if input_mode in ("256", "384"):
        return ModelSingleStream(head_kind, hidden, p_drop, fusion="single").to(device)

    elif input_mode == "stack":
        if fusion == "dual":
            return ModelDualStream(head_kind, hidden, p_drop).to(device)
        elif fusion == "stack6":
            return ModelSingleStream(head_kind, hidden, p_drop, fusion="stack6").to(device)
        else:
            raise ValueError("fusion debe ser 'dual' o 'stack6'")

    else:
        raise ValueError("input_mode inválido")

def param_groups(model: nn.Module, stage: int, k_unf: int = 1):
    """
    Devuelve grupos de parámetros para aplicar learning rates distintos.

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
    last_params: List[nn.Parameter] = []
    rest_params: List[nn.Parameter] = []

    if stage == 1:
        return head_params, last_params, rest_params

    children = list(model.backbone.children())

    if children:
        last_children = set(children[-k_unf:])
        for ch in children:
            for p in ch.parameters():
                if not p.requires_grad:
                    continue
                if ch in last_children:
                    last_params.append(p)
                else:
                    rest_params.append(p)
    else:
        for p in model.backbone.parameters():
            if p.requires_grad:
                rest_params.append(p)

    return head_params, last_params, rest_params
