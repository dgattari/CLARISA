
# src/data/split_analysis.py

"""
split_analysis.py
-----------------
Utilities to inspect and summarize MARTA data splits.

Responsibilities:
  - build one-row-per-sample split assignment tables
  - count classes per slide
  - count classes per final split
  - provide a simple practical interpretation of validation adequacy
"""

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .split_builders import attach_slide_ids, infer_slide_id

def _bbox_columns(bbox) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)

def build_split_assignment_table(
    samples: List[Dict[str, Any]],
    split_indices: Dict[str, List[int]],
    group_key: str = "slide_id",
) -> pd.DataFrame:
    """
    Build a table with one row per sample describing its split assignment.
    """
    samples = attach_slide_ids(samples)

    split_by_index = {}
    for split_name in ("train", "val", "test"):
        for idx in split_indices[f"{split_name}_idx"]:
            split_by_index[idx] = split_name

    rows = []
    for i, s in enumerate(samples):
        x1, y1, x2, y2 = _bbox_columns(s["bbox"])
        rows.append(
            {
                "sample_index": i,
                "image_path": str(s["image_path"]),
                "slide_id": s.get(group_key, infer_slide_id(s["image_path"])),
                "label": int(s["label"]),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "split": split_by_index.get(i, "unassigned"),
            }
        )

    return pd.DataFrame(rows)

def count_classes_per_slide(samples: List[Dict[str, Any]], group_key: str = "slide_id") -> Dict[str, Dict[str, Any]]:
    """
    Count class 0 / class 1 per slide and return totals and percentages.

    This is useful to inspect whether the chosen slide-level split is plausible
    before fixing it as the official MARTA split.
    """
    samples = attach_slide_ids(samples)

    slide_stats: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        sid = str(s[group_key]).upper()
        label = int(s["label"])
        if sid not in slide_stats:
            slide_stats[sid] = {"n_total": 0, "class_0": 0, "class_1": 0}
        slide_stats[sid]["n_total"] += 1
        slide_stats[sid][f"class_{label}"] += 1

    for sid, stats in slide_stats.items():
        n_total = max(1, stats["n_total"])
        stats["pct_class_0"] = round(100.0 * stats["class_0"] / n_total, 2)
        stats["pct_class_1"] = round(100.0 * stats["class_1"] / n_total, 2)

    return dict(sorted(slide_stats.items(), key=lambda kv: kv[0]))

def build_split_class_summary(
    samples: List[Dict[str, Any]],
    split_indices: Dict[str, List[int]],
) -> Dict[str, Dict[str, Any]]:
    """
    Count class 0 / class 1 for train, validation, and test.

    Returns totals and percentages for each split.
    """
    summary = {}
    for split_name in ("train", "val", "test"):
        idxs = split_indices[f"{split_name}_idx"]
        labels = [int(samples[i]["label"]) for i in idxs]
        cnt = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
        n_total = int(len(labels))
        n0 = int(cnt[0])
        n1 = int(cnt[1])

        summary[split_name] = {
            "n_total": n_total,
            "class_0": n0,
            "class_1": n1,
            "pct_class_0": round(100.0 * n0 / max(1, n_total), 2),
            "pct_class_1": round(100.0 * n1 / max(1, n_total), 2),
        }

    return summary

def summarize_split_acceptability(split_class_summary: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Provide a simple textual interpretation of whether the class balance in the
    current split looks acceptable.

    Practical rule:
      - >= 20 samples per class in validation is reassuring
      - values around 5, 8, or 10 are concerning
    """
    notes = {}

    val = split_class_summary.get("val", {})
    n0 = int(val.get("class_0", 0))
    n1 = int(val.get("class_1", 0))

    if n0 >= 20 and n1 >= 20:
        notes["validation"] = (
            "Validation split looks reasonably balanced for model selection "
            "(both classes have at least 20 samples)."
        )
    elif min(n0, n1) <= 10:
        notes["validation"] = (
            "Validation split may be too small or too imbalanced for stable tuning "
            "(one class has 10 samples or fewer)."
        )
    else:
        notes["validation"] = (
            "Validation split is usable, but class counts should be interpreted "
            "with caution during tuning."
        )
    return notes