
# src/train/create_data_split.py

"""
create_data_split.py
--------------------
Create and save a single official data split for MARTA.

This script:
  - loads the split definition from a YAML config
  - builds ROI-level samples from the annotations
  - creates a split according to the selected strategy
  - saves split indices and inspection artifacts
  - summarizes class distributions by slide and by final split

The goal is to define the split once and then reuse it consistently in:
  - standard classifier training
  - hyperparameter tuning
"""

import argparse
from pathlib import Path
from .dataset_builder import build_samples

from src.data import (
    attach_slide_ids,
    make_split_indices,
    build_split_assignment_table,
    count_classes_per_slide,
    build_split_class_summary,
    summarize_split_acceptability,
    save_split_artifacts,
)
from src.utils.config import load_data_split_config
from src.utils.io import ensure_dir
from src.utils.logging import log

def parse_args():
    """
    Parse command-line arguments for MARTA split creation.
    """
    parser = argparse.ArgumentParser(
        description="Create and save a precomputed MARTA data split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_split.yaml",
        help="Path to the data split YAML config.",
    )
    return parser.parse_args()

def main(config_path: str | Path = "configs/data_split.yaml"):
    cfg = load_data_split_config(config_path)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    log_fp = output_dir / "create_split_log.txt"

    log("[split] Loading split config", log_fp)
    log(f"[split] Config path: {config_path}", log_fp)
    log(f"[split] Split name: {cfg.split_name}", log_fp)
    log(f"[split] Output dir: {output_dir}", log_fp)
    log(f"[split] Strategy: {cfg.strategy}", log_fp)

    log("[data] Building ROI samples from annotations", log_fp)
    samples, _ = build_samples()
    samples = attach_slide_ids(samples)
    log(f"[data] Total samples built: {len(samples)}", log_fp)

    split_indices = make_split_indices(
        samples=samples,
        strategy=cfg.strategy,
        random_seed=cfg.random_seed,
        test_size=cfg.test_size,
        val_size_within_trainval=cfg.val_size_within_trainval,
        group_key=cfg.group_key,
        manual_split=cfg.manual_split,
    )

    assignment_df = build_split_assignment_table(
        samples=samples,
        split_indices=split_indices,
        group_key=cfg.group_key,
    )

    slide_class_summary = count_classes_per_slide(
        samples=samples,
        group_key=cfg.group_key,
    )

    split_class_summary = build_split_class_summary(
        samples=samples,
        split_indices=split_indices,
    )

    acceptability_notes = summarize_split_acceptability(split_class_summary)

    metadata = {
        "group_definition": cfg.group_definition,
        "group_key": cfg.group_key,
        "notes": cfg.notes,
    }

    if cfg.strategy == "manual_groups":
        metadata["train_groups"] = cfg.manual_split.get("train_groups", [])
        metadata["val_groups"] = cfg.manual_split.get("val_groups", [])
        metadata["test_groups"] = cfg.manual_split.get("test_groups", [])

    if cfg.strategy in {"roi_stratified", "grouped"}:
        metadata["test_size"] = cfg.test_size
        metadata["val_size_within_trainval"] = cfg.val_size_within_trainval

    save_split_artifacts(
        output_dir=output_dir,
        split_name=cfg.split_name,
        strategy=cfg.strategy,
        random_seed=cfg.random_seed,
        split_indices=split_indices,
        assignment_df=assignment_df,
        split_class_summary=split_class_summary,
        slide_class_summary=slide_class_summary,
        metadata=metadata,
    )

    log("[split] Slide-level class summary", log_fp)
    for slide_id, stats in slide_class_summary.items():
        log(
            f"[slide] {slide_id} | total={stats['n_total']} "
            f"class0={stats['class_0']} ({stats['pct_class_0']:.2f}%) "
            f"class1={stats['class_1']} ({stats['pct_class_1']:.2f}%)",
            log_fp,
        )

    log("[split] Final split-level class summary", log_fp)
    for split_name in ("train", "val", "test"):
        stats = split_class_summary[split_name]
        log(
            f"[{split_name}] total={stats['n_total']} "
            f"class0={stats['class_0']} ({stats['pct_class_0']:.2f}%) "
            f"class1={stats['class_1']} ({stats['pct_class_1']:.2f}%)",
            log_fp,
        )

    for key, note in acceptability_notes.items():
        log(f"[check] {key}: {note}", log_fp)

    print("== SPLIT LISTO ==")
    print("Output dir:", output_dir.resolve())
    print("Strategy:", cfg.strategy)
    print("Train:", len(split_indices["train_idx"]))
    print("Val:", len(split_indices["val_idx"]))
    print("Test:", len(split_indices["test_idx"]))

if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)