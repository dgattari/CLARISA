
# src/utils/wandb_utils.py

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    import wandb
except ImportError:
    wandb = None

def load_expert_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] No se pudo leer la config experta ({e}). Se ignora.")
        return {}

def get_wandb_cfg_from_main_cfg(cfg) -> dict:
    """
    Return the W&B sub-configuration from the main MARTA config.

    W&B is controlled through:
      - expert_mode
      - expert_config_path

    If expert mode is disabled or the expert config cannot be loaded, an empty
    dictionary is returned.
    """
    expert_mode = bool(getattr(cfg, "expert_mode", False))
    if not expert_mode:
        return {}

    expert_config_path = getattr(cfg, "expert_config_path", None)
    expert_cfg = load_expert_config(expert_config_path)
    return expert_cfg.get("wandb", {}) or {}

def wandb_is_enabled(cfg) -> bool:
    """
    Return True only when W&B tracking is effectively enabled.

    W&B is enabled if:
      - expert_mode is active
      - a valid wandb section exists in the expert config
      - wandb is installed
      - mode is not set to 'disabled'
    """
    wandb_cfg = get_wandb_cfg_from_main_cfg(cfg)

    return (
        bool(wandb_cfg)
        and bool(wandb_cfg.get("enabled", False))
        and wandb is not None
        and wandb_cfg.get("mode", "disabled") != "disabled"
    )

def init_wandb_run(
    cfg, 
    run_name: str, 
    output_dir: Path, 
    extra_config: Optional[Dict[str, Any]] = None,
    project_override: Optional[str] = None,
    group_override: Optional[str] = None,
    job_type_override: Optional[str] = None,
    name_override: Optional[str] = None,
):
    """
    Initialize a W&B run for the current MARTA execution.

    The default behavior uses the W&B settings defined in the expert config.
    Optional overrides can be supplied for project, group, job type, and run
    name. This is useful for Optuna studies, where trial runs should be grouped
    separately from standard final training runs.
    """
    if not wandb_is_enabled(cfg):
        return None

    wandb_cfg = get_wandb_cfg_from_main_cfg(cfg)

    config_dict = cfg.__dict__.copy()
    config_dict["wandb"] = wandb_cfg

    if extra_config:
        config_dict.update(extra_config)

    try:
        run = wandb.init(
            project=project_override or wandb_cfg.get("project", "MARTA-training"),
            entity=wandb_cfg.get("entity", None),
            name=name_override or wandb_cfg.get("name", None) or run_name,
            group=group_override or wandb_cfg.get("group", None),
            job_type=job_type_override or wandb_cfg.get("job_type", "train"),
            tags=list(wandb_cfg.get("tags", [])) if wandb_cfg.get("tags", None) is not None else None,
            notes=wandb_cfg.get("notes", None),
            mode=wandb_cfg.get("mode", "online"),
            dir=str(output_dir),
            config=config_dict,
        )
        return run

    except Exception as e:
        print(f"[WARN] No se pudo inicializar W&B ({e}). Se continúa sin tracking.")
        return None

def watch_model_if_needed(cfg, model):
    if not wandb_is_enabled(cfg):
        return

    if wandb is None or wandb.run is None:
        return

    wandb_cfg = get_wandb_cfg_from_main_cfg(cfg)
    watch_cfg = wandb_cfg.get("watch", {}) or {}

    if not watch_cfg.get("enabled", False):
        return

    wandb.watch(
        model,
        log=watch_cfg.get("log", "all"),
        log_freq=int(watch_cfg.get("log_freq", 100)),
    )

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    if wandb is None or wandb.run is None:
        return
    wandb.log(metrics, step=step)

def log_artifact_file(
    path: Path, 
    artifact_name: str, 
    artifact_type: str = "file", 
    aliases=None,
):
    if wandb is None or wandb.run is None:
        return
    if not path.exists():
        return

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact, aliases=aliases or [])

def log_checkpoint_artifact(
    ckpt_path: Path, 
    run_name: str, 
    stage: str, 
    aliases=None,
):
    if wandb is None or wandb.run is None:
        return
    if not ckpt_path.exists():
        return

    artifact = wandb.Artifact(
        name=f"{run_name}-{stage}-checkpoint",
        type="model",
        metadata={"stage": stage},
    )
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact, aliases=aliases or ["latest"])

def batch_logging_enabled(cfg) -> bool:
    try:
        wandb_cfg = get_wandb_cfg_from_main_cfg(cfg)
        return bool(
            wandb_cfg.get("batch_logging", {}).get("enabled", False)
        )
    except Exception:
        return False

def batch_log_every_steps(cfg) -> int:
    try:
        wandb_cfg = get_wandb_cfg_from_main_cfg(cfg)
        value = wandb_cfg.get("batch_logging", {}).get("log_every_steps", 20)
        value = int(value)
        return value if value > 0 else 20
    except Exception:
        return 20

def finish_wandb(summary: Optional[Dict[str, Any]] = None):
    """
    Safely finish the active W&B run.

    This function is safe to call even when no run is active.
    """
    if wandb is None or wandb.run is None:
        return

    if summary:
        for k, v in summary.items():
            wandb.run.summary[k] = v
    wandb.finish()