
# src/models/freezing.py

from typing import List
import torch.nn as nn

def set_bn_eval(m: nn.Module):
    """
    Pone BatchNorm en eval:
      - usa running stats
      - no actualiza running_mean / running_var
      - no toca requires_grad de gamma/beta
    """
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
        m.eval()

def set_bn_train(m: nn.Module):
    """
    Pone BatchNorm en train:
      - actualiza running stats
      - no toca requires_grad de gamma/beta
    """
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
        m.train()

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True

def freeze_backbone_strict(model: nn.Module):
    """
    Stage 1 estricto:
      - congela TODOS los parámetros del backbone
      - pone TODAS las BN del backbone en eval
    """
    freeze_module(model.backbone)
    model.backbone.apply(set_bn_eval)

# def freeze_backbone(model):
#     freeze_module(model.backbone)

def unfreeze_last_k_blocks_strict(model: nn.Module, k: int = 1):
    """
    Stage 2 estricto:
      - congela TODO_ el backbone
      - pone TODAS las BN del backbone en eval
      - descongela SOLO los últimos k bloques (children)
      - pone en train SOLO las BN de esos últimos k bloques
    """
    children = list(model.backbone.children())

    if not children:
        # fallback conservador:
        # si no podemos identificar bloques, abrimos todo
        print('children was not identified for backbone')
        unfreeze_all_backbone_strict(model)
        return

    # 1) congelar todo + BN eval
    freeze_module(model.backbone)
    model.backbone.apply(set_bn_eval)

    # 2) abrir últimos k bloques + BN train solo en esos bloques
    for ch in children[-k:]:
        unfreeze_module(ch)
        ch.apply(set_bn_train)

# def unfreeze_last_k_blocks(model, k: int = 1):
#     """Tenta identificar bloques finales de la backbone y liberarlos.
#     Implementación genérica: libera últimos k children de self.backbone.
#     """
#     children = list(model.backbone.children())
#     if not children:
#         unfreeze_module(model.backbone)
#         return
#     for p in model.backbone.parameters():
#         p.requires_grad = False
#     for ch in children[-k:]:
#         for p in ch.parameters():
#             p.requires_grad = True

def unfreeze_all_backbone_strict(model: nn.Module):
    """
    Stage 3 estricto:
      - descongela TODO_ el backbone
      - pone TODAS las BN del backbone en train
    """
    unfreeze_module(model.backbone)
    model.backbone.apply(set_bn_train)

# def unfreeze_all_backbone(model):
#     unfreeze_module(model.backbone)

def freeze_for_stage(model: nn.Module, stage: int, k_unf: int = 1):
    """
    Política estricta de freeze/unfreeze por fase.

    stage 1:
      - backbone congelado
      - todas las BN del backbone en eval
      - entrena solo la head

    stage 2:
      - solo últimos k bloques del backbone descongelados
      - solo las BN de esos bloques en train
      - resto del backbone sigue realmente congelado

    stage 3:
      - backbone completo descongelado
      - todas las BN del backbone en train
    """
    if stage == 1:
        freeze_backbone_strict(model)

    elif stage == 2:
        unfreeze_last_k_blocks_strict(model, k=k_unf)

    elif stage == 3:
        unfreeze_all_backbone_strict(model)

    else:
        raise ValueError(f"stage no soportado: {stage}")
