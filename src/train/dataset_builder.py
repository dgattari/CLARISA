
# src/train/dataset_builder.py 

import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from src.utils.io import safe_pickle_load
from src.utils.paths import DATASET_DIR, IMAGES_DIR

# Descubrimiento de datos: patrón exacto de nombres
SPYDATA_REGEX = re.compile(r'^IM(\d+)\s+Dani\s+conexina2\.spydata$', re.IGNORECASE)

# Extensiones de imagen soportadas automáticamente
IMAGE_EXTENSIONS = [".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF"]

def discover_image_pairs() -> List[Tuple[Path, Path]]:
    """
    Devuelve una lista de pares (spydata_path, image_path).

    Empareja cada archivo .spydata en dataset/ con su imagen correspondiente
    en images/, buscando automáticamente formatos compatibles (.png, .tif, .tiff).

    Ejemplo:
        dataset/IM6 Dani conexina2.spydata ↔ images/IM6.png
    """

    pairs = []

    dataset_dir = DATASET_DIR
    images_dir = IMAGES_DIR

    for spy_path in dataset_dir.iterdir():

        if spy_path.is_file() and SPYDATA_REGEX.match(spy_path.name):

            match = SPYDATA_REGEX.match(spy_path.name)
            n = match.group(1)

            image_path = None

            for ext in IMAGE_EXTENSIONS:
                candidate = images_dir / f"IM{n}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is not None:
                pairs.append((spy_path, image_path))

    pairs.sort(key=lambda x: int(SPYDATA_REGEX.match(x[0].name).group(1)))

    if not pairs:
        raise RuntimeError(
            f"No se encontraron pares .spydata ↔ imagen en "
            f"{dataset_dir.resolve()} y {images_dir.resolve()}"
        )
    return pairs

def extract_targets_from_spydata(spy_obj: Any) -> Tuple[List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
    """
    Extrae target_regions_1_filtered y target_regions_2_filtered.
    Acepta dicts, objetos con atributos o estructuras anidadas típicas de Spyder.
    """
    def _find_in_mapping(m: Dict[str, Any], key: str):
        # búsqueda directa
        if key in m:
            return m[key]
        # búsqueda relajada por nombre
        for k in m.keys():
            if key.lower() == k.lower():
                return m[k]
        return None

    tr1 = tr2 = None

    if isinstance(spy_obj, dict):
        # caso 1: variables directamente en el dict
        tr1 = _find_in_mapping(spy_obj, 'target_regions_1_filtered')
        tr2 = _find_in_mapping(spy_obj, 'target_regions_2_filtered')
        # caso 2: dict de namespaces comunes
        if (tr1 is None or tr2 is None):
            for ns_key in ('namespace', 'globals', 'global_ns', 'variables', 'data'):
                ns = spy_obj.get(ns_key, None)
                if isinstance(ns, dict):
                    tr1 = tr1 or _find_in_mapping(ns, 'target_regions_1_filtered')
                    tr2 = tr2 or _find_in_mapping(ns, 'target_regions_2_filtered')
                if tr1 is not None and tr2 is not None:
                    break
    else:
        # objeto con atributos
        tr1 = getattr(spy_obj, 'target_regions_1_filtered', None)
        tr2 = getattr(spy_obj, 'target_regions_2_filtered', None)

    if tr1 is None or tr2 is None:
        # último recurso: buscar en subdicts
        if isinstance(spy_obj, dict):
            for v in spy_obj.values():
                if isinstance(v, dict):
                    if tr1 is None:
                        tr1 = v.get('target_regions_1_filtered', tr1)
                    if tr2 is None:
                        tr2 = v.get('target_regions_2_filtered', tr2)
                if tr1 is not None and tr2 is not None:
                    break

    if tr1 is None or tr2 is None:
        raise RuntimeError('No encontré target_regions_1_filtered / _2_ en el .spydata (intentos: raíz, namespace, atributos, anidados)')

    # normalizo a lista de tuplas int
    def norm_list(L):
        out = []
        for b in L:
            x1, y1, x2, y2 = b
            out.append((int(x1), int(y1), int(x2), int(y2)))
        return out

    return norm_list(tr1), norm_list(tr2)

def build_samples() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Descubre datos y construye samples: list de dicts con (image_path, bbox, label).
    Retorna samples y array y (labels) del mismo largo.
    """
    pairs = discover_image_pairs()
    samples = []
    for spy_path, tif_path in pairs:
        obj = safe_pickle_load(spy_path)
        tr1, tr2 = extract_targets_from_spydata(obj)
        for b in tr1:
            samples.append({'image_path': tif_path, 'bbox': b, 'label': 0, 'img_id': tif_path.name})
        for b in tr2:
            samples.append({'image_path': tif_path, 'bbox': b, 'label': 1, 'img_id': tif_path.name})
    y = np.array([s['label'] for s in samples], dtype=np.int64)
    return samples, y