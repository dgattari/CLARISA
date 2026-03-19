
# src/utils/paths.py

from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass
class ProjectPaths:
    dataset_dir: Path
    images_dir: Path
    experiments_dir: Path

def load_paths_config(config_path: str | Path = "configs/paths.yaml") -> ProjectPaths:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return ProjectPaths(
        dataset_dir=Path(cfg["dataset_dir"]),
        images_dir=Path(cfg["images_dir"]),
        experiments_dir=Path(cfg["experiments_dir"]),
    )

_PATHS = load_paths_config()

DATASET_DIR = _PATHS.dataset_dir
IMAGES_DIR = _PATHS.images_dir
EXPERIMENTS_DIR = _PATHS.experiments_dir
