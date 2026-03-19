
# src/utils/config.py

from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass
class TrainConfig:  
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 4

    val_size: float = 0.10
    test_size: float = 0.10

    # Group split (anti-leakage)
    use_group_split: bool = False
    group_key: str = "image_path"

    # Entradas
    input_mode: str = "stack"   # '256', '384', 'stack'
    fusion: str = "dual"        # 'dual' o 'stack6'
    resize_to: int = 384

    # Etapas
    stage1_epochs: int = 1
    stage2_epochs: int = 1
    stage3_epochs: int = 1
    k_unf: int = 1

    # LRs
    head_lr: float = 1e-3
    last_lr: float = 3e-4
    rest_lr: float = 1e-4
    weight_decay: float = 1e-4

    # Ponderación clase 1
    class1_bonus: float = 1.1
    decision_threshold: float = 0.5

    head_kind: str = "mlp"   # 'mlp' o 'logreg'
    hidden: int = 128
    dropout: float = 0.5

@dataclass
class InferenceConfig:
    resize_to: int = 384

    threshold: float = 0.50
    expand: int = 40

    perplexity: float = 30.0
    soft: bool = False
    sigma: float = 128.0

    random_seed: int = 42
    save_excel: bool = True

def load_train_config(config_path: str | Path) -> TrainConfig:
    """
    Lee un YAML y construye TrainConfig.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    return TrainConfig(**cfg_dict)

def load_inference_config(config_path: str | Path) -> InferenceConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    return InferenceConfig(**cfg_dict)

def build_train_run_name(cfg: TrainConfig) -> str:
    """
    Construye un nombre corto y legible para la corrida.
    """
    if cfg.head_kind == "logreg":
        return f"logreg_{cfg.input_mode}_{cfg.fusion}"
    return f"mlp_{cfg.hidden}_d{cfg.dropout}_{cfg.input_mode}_{cfg.fusion}"
