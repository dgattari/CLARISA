
# src/train/splits.py

from typing import Dict, Any, List

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

def _get_group_id(s: Dict[str, Any], group_key: str) -> str:
    """
    Devuelve la entidad/grupo para evitar leakage.
    Default: agrupar por image_path (todas las ROIs de la misma imagen quedan juntas).
    """
    if group_key in s:
        return str(s[group_key])
    # fallback ultra defensivo
    return str(s.get("image_path", "UNKNOWN"))

def make_splits(
    samples: List[Dict[str, Any]],
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_seed: int,
    use_group_split: bool,
    group_key: str,
):
    """
    Devuelve (tr_idx, va_idx, te_idx).
    Si use_group_split=True:
      - Primero separa grupos para TEST
      - Luego separa grupos (del remanente) para VAL
      - Train = resto
    Estratificación: aproximada a nivel-grupo usando majority label del grupo.
    Si no se puede estratificar (pocos grupos/clase única), cae a GroupShuffleSplit sin estratificar.
    """
    idx = np.arange(len(samples))

    # ===============================
    # CASE 1: SIN GROUP SPLIT
    # ===============================
    if not use_group_split:
        # caso simple (sin grupos): estratificado por sample
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        trainval_idx, te_idx = next(sss1.split(idx, y))

        val_rel = val_size / (1.0 - test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=random_seed + 1)
        tr_rel, va_rel = next(sss2.split(trainval_idx, y[trainval_idx]))

        tr_idx = trainval_idx[tr_rel]
        va_idx = trainval_idx[va_rel]
        return tr_idx, va_idx, te_idx

    # ===============================
    # CASE 2: GROUP SPLIT
    # ===============================
    groups = np.array([_get_group_id(samples[i]) for i in idx])

    # Map: group -> indices
    uniq_g, inv = np.unique(groups, return_inverse=True)

    # label de grupo para "estratificar": majority label dentro del grupo
    g_pos_rate = np.zeros(len(uniq_g), dtype=np.float64)
    for gi in range(len(uniq_g)):
        members = idx[inv == gi]
        g_pos_rate[gi] = float(np.mean(y[members]))
    g_y = (g_pos_rate >= 0.5).astype(int)

    # 1) TEST por grupos
    try:
        sss_g1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        g_trainval, g_test = next(sss_g1.split(np.arange(len(uniq_g)), g_y))

    except Exception:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)

        # aquí "X" no importa; pasamos índices de samples y usamos groups
        trainval_idx, te_idx = next(gss1.split(idx, y, groups=groups))
        # 2) VAL por grupos dentro del remanente
        val_rel = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=random_seed + 1)
        tr_idx, va_idx = next(gss2.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx]))
        return trainval_idx[tr_idx], trainval_idx[va_idx], te_idx

    # expandir grupos a índices de samples
    test_groups = set(uniq_g[g_test])
    trainval_groups = set(uniq_g[g_trainval])

    te_mask = np.array([g in test_groups for g in groups], dtype=bool)
    te_idx = idx[te_mask]
    trainval_idx = idx[~te_mask]

    # 2) VAL por grupos dentro de trainval
    uniq_g2 = np.array(sorted(list(trainval_groups)))
    # recomputar labels de grupo para trainval
    g2_y = np.array([(g_pos_rate[np.where(uniq_g == g)[0][0]] >= 0.5) for g in uniq_g2], dtype=int)

    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    try:
        sss_g2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=cfg.random_seed + 1)
        g_tr, g_va = next(sss_g2.split(np.arange(len(uniq_g2)), g2_y))

    except Exception:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=cfg.random_seed + 1)
        tr_idx, va_idx = next(gss2.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx]))
        return trainval_idx[tr_idx], trainval_idx[va_idx], te_idx

    va_groups = set(uniq_g2[g_va])
    va_mask = np.array([g in va_groups for g in groups], dtype=bool)
    va_idx = idx[va_mask & (~te_mask)]
    tr_idx = idx[(~va_mask) & (~te_mask)]
    return tr_idx, va_idx, te_idx