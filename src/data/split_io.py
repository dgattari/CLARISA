
# src/data/split_io.py

"""
split_io.py
-----------
Utilities to save, load, and validate MARTA split artifacts.

Responsibilities:
  - save split metadata and inspection artifacts
  - load precomputed split indices
  - validate a stored split against the current sample list
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from .split_analysis import _bbox_columns

def save_split_artifacts(
    *,
    output_dir: str | Path,
    split_name: str,
    strategy: str,
    random_seed: int,
    split_indices: Dict[str, List[int]],
    assignment_df: pd.DataFrame,
    split_class_summary: Dict[str, Dict[str, Any]],
    slide_class_summary: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any] | None = None,
):
    """
    Save the split definition and its inspection artifacts to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = int(len(assignment_df))
    n_train = int(len(split_indices["train_idx"]))
    n_val = int(len(split_indices["val_idx"]))
    n_test = int(len(split_indices["test_idx"]))

    split_info = {
        "split_name": split_name,
        "strategy": strategy,
        "random_seed": int(random_seed),
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    if metadata:
        split_info.update(metadata)

    with (output_dir / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    with (output_dir / "split_indices.json").open("w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)

    assignment_df.to_csv(output_dir / "split_assignments.csv", index=False)

    with (output_dir / "split_class_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_class_summary, f, indent=2)

    with (output_dir / "slide_class_summary.json").open("w", encoding="utf-8") as f:
        json.dump(slide_class_summary, f, indent=2)

def load_split_indices(split_dir: str | Path) -> Dict[str, List[int]]:
    """
    Load train / validation / test indices from a precomputed split directory.
    """
    split_dir = Path(split_dir)
    with (split_dir / "split_indices.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_precomputed_split(
    samples: List[Dict[str, Any]],
    split_dir: str | Path,
    n_checks: int = 5,
) -> None:
    """
    Perform a minimal sanity check between the current samples and a precomputed split.

    Checks:
      - max sample index is compatible with current sample length
      - a few rows from split_assignments.csv match image_path and bbox
    """
    split_dir = Path(split_dir)
    split_indices = load_split_indices(split_dir)

    all_indices = (
        list(split_indices["train_idx"])
        + list(split_indices["val_idx"])
        + list(split_indices["test_idx"])
    )
    if not all_indices:
        raise ValueError("Loaded split is empty.")

    max_idx = max(all_indices)
    if max_idx >= len(samples):
        raise ValueError(
            f"Precomputed split index {max_idx} is incompatible with current samples "
            f"(n_samples={len(samples)})."
        )

    assignments_path = split_dir / "split_assignments.csv"
    if not assignments_path.exists():
        return

    df = pd.read_csv(assignments_path)
    if df.empty:
        return

    check_rows = df.head(n_checks)
    for _, row in check_rows.iterrows():
        i = int(row["sample_index"])
        s = samples[i]

        if str(s["image_path"]) != str(row["image_path"]):
            raise ValueError(
                "Precomputed split validation failed: image_path mismatch "
                f"at sample_index={i}"
            )

        x1, y1, x2, y2 = _bbox_columns(s["bbox"])
        if (
            int(row["bbox_x1"]) != x1
            or int(row["bbox_y1"]) != y1
            or int(row["bbox_x2"]) != x2
            or int(row["bbox_y2"]) != y2
        ):
            raise ValueError(
                "Precomputed split validation failed: bbox mismatch "
                f"at sample_index={i}"
            )