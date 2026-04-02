# src/data/split_builders.py

"""
split_builders.py
-----------------
Utilities to define dataset split indices for MARTA.

Responsibilities:
  - infer slide identifiers from sample paths
  - attach slide identifiers to samples
  - build split indices with different strategies:
      * roi_stratified
      * grouped
      * manual_groups
"""

import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

def infer_slide_id(image_path: str | Path) -> str:
    """
    Infer the MARTA slide identifier from an image path.

    Expected examples:
      - IM6.png
      - IM9.tif
      - IM133.tiff
      - /path/to/IM1315.png

    Returns:
      - 'IM6', 'IM9', 'IM133', 'IM1315', etc.
    """
    stem = Path(image_path).stem
    match = re.search(r"(IM\d+)", stem, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return stem.upper()

def attach_slide_ids(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return a copy of the samples list with a 'slide_id' field added when missing.
    """
    out = []
    for s in samples:
        s2 = dict(s)
        if "slide_id" not in s2:
            s2["slide_id"] = infer_slide_id(s2["image_path"])
        out.append(s2)
    return out

def make_roi_stratified_split(
    samples: List[Dict[str, Any]],
    y: np.ndarray,
    test_size: float,
    val_size_within_trainval: float,
    random_seed: int,
) -> Dict[str, List[int]]:
    """
    Create a stratified ROI-level split.

    Procedure:
      1) reserve test_size from the full dataset
      2) split the remaining train+val pool using val_size_within_trainval

    This reproduces the manuscript-style logic but does not prevent slide-level leakage.
    """
    indices = np.arange(len(samples))

    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_seed,
    )
    trainval_idx, test_idx = next(sss_test.split(indices, y))

    y_trainval = y[trainval_idx]

    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size_within_trainval,
        random_state=random_seed,
    )
    train_rel, val_rel = next(sss_val.split(trainval_idx, y_trainval))

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    return {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

def _get_group_labels(
    samples: List[Dict[str, Any]],
    groups: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build unique groups and an approximate binary group label for stratified
    group splitting. The group label is defined by majority label within group.
    """
    uniq_g, inv = np.unique(groups, return_inverse=True)

    g_pos_rate = np.zeros(len(uniq_g), dtype=np.float64)
    for gi in range(len(uniq_g)):
        members = np.where(inv == gi)[0]
        g_pos_rate[gi] = float(np.mean(y[members]))

    g_y = (g_pos_rate >= 0.5).astype(int)
    return uniq_g, g_y

def make_grouped_split(
    samples: List[Dict[str, Any]],
    y: np.ndarray,
    test_size: float,
    val_size_within_trainval: float,
    random_seed: int,
    group_key: str = "slide_id",
) -> Dict[str, List[int]]:
    """
    Create a group-aware split.

    Groups are kept intact across train / validation / test.

    Strategy:
      - try approximate stratification at group level using majority group label
      - if that fails, fall back to GroupShuffleSplit

    This is intended as a clean replacement for the old training fallback logic.
    """
    indices = np.arange(len(samples))
    groups = np.array([str(s[group_key]) for s in samples])

    uniq_g, g_y = _get_group_labels(samples, groups, y)

    # -------------------------------
    # Step 1: reserve test groups
    # -------------------------------
    try:
        sss_g1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed,
        )
        g_trainval_rel, g_test_rel = next(sss_g1.split(np.arange(len(uniq_g)), g_y))

        test_groups = set(uniq_g[g_test_rel])
        te_mask = np.array([g in test_groups for g in groups], dtype=bool)

        test_idx = indices[te_mask]
        trainval_idx = indices[~te_mask]

    except Exception:
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed,
        )
        trainval_rel, test_rel = next(gss1.split(indices, y, groups=groups))
        trainval_idx = indices[trainval_rel]
        test_idx = indices[test_rel]

    # -------------------------------
    # Step 2: reserve validation groups
    # -------------------------------
    trainval_groups = groups[trainval_idx]
    trainval_y = y[trainval_idx]

    uniq_g2, g2_y = _get_group_labels(
        samples=[samples[i] for i in trainval_idx],
        groups=trainval_groups,
        y=trainval_y,
    )

    try:
        sss_g2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size_within_trainval,
            random_state=random_seed + 1,
        )
        g_train_rel, g_val_rel = next(sss_g2.split(np.arange(len(uniq_g2)), g2_y))

        val_groups = set(uniq_g2[g_val_rel])
        va_mask_rel = np.array([g in val_groups for g in trainval_groups], dtype=bool)

        val_idx = trainval_idx[va_mask_rel]
        train_idx = trainval_idx[~va_mask_rel]

    except Exception:
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size_within_trainval,
            random_state=random_seed + 1,
        )
        train_rel, val_rel = next(gss2.split(trainval_idx, trainval_y, groups=trainval_groups))
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]

    return {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

def make_manual_group_split(
    samples: List[Dict[str, Any]],
    manual_split: Dict[str, List[str]],
    group_key: str = "slide_id",
) -> Dict[str, List[int]]:
    """
    Create a split from explicit group assignments.

    Expected structure:
      manual_split:
        train_groups: [...]
        val_groups: [...]
        test_groups: [...]

    This is the recommended strategy when the official MARTA split is defined
    manually at the slide level.
    """
    train_groups = {g.upper() for g in manual_split.get("train_groups", [])}
    val_groups = {g.upper() for g in manual_split.get("val_groups", [])}
    test_groups = {g.upper() for g in manual_split.get("test_groups", [])}

    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise ValueError(f"Manual split groups overlap across partitions: {sorted(overlap)}")

    train_idx, val_idx, test_idx = [], [], []

    for i, s in enumerate(samples):
        gid = str(s[group_key]).upper()
        if gid in train_groups:
            train_idx.append(i)
        elif gid in val_groups:
            val_idx.append(i)
        elif gid in test_groups:
            test_idx.append(i)

    assigned = set(train_idx) | set(val_idx) | set(test_idx)
    missing = sorted(set(range(len(samples))) - assigned)
    if missing:
        missing_groups = sorted({str(samples[i][group_key]).upper() for i in missing})
        raise ValueError(
            "Some samples were not assigned to any split. "
            f"Missing groups: {missing_groups}"
        )

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

def make_split_indices(
    samples: List[Dict[str, Any]],
    strategy: str,
    random_seed: int = 42,
    test_size: float = 0.15,
    val_size_within_trainval: float = 0.15,
    group_key: str = "slide_id",
    manual_split: Dict[str, List[str]] | None = None,
) -> Dict[str, List[int]]:
    """
    Create split indices according to the selected strategy.

    Supported strategies:
      - roi_stratified
      - grouped
      - manual_groups
    """
    samples = attach_slide_ids(samples)
    y = np.array([int(s["label"]) for s in samples], dtype=np.int64)

    if strategy == "roi_stratified":
        return make_roi_stratified_split(
            samples=samples,
            y=y,
            test_size=test_size,
            val_size_within_trainval=val_size_within_trainval,
            random_seed=random_seed,
        )

    if strategy == "grouped":
        return make_grouped_split(
            samples=samples,
            test_size=test_size,
            val_size_within_trainval=val_size_within_trainval,
            random_seed=random_seed,
            group_key=group_key,
        )

    if strategy == "manual_groups":
        if manual_split is None:
            raise ValueError("manual_split must be provided when strategy='manual_groups'")

        return make_manual_group_split(
            samples=samples,
            manual_split=manual_split,
            group_key=group_key,
        )
    raise ValueError(f"Unsupported split strategy: {strategy}")