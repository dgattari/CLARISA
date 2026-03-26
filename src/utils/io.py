
# src/utils/io.py
import gzip
import pickle
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch

# meter tb aqui un: quizá también read_yaml, write_json, etc. en el futuro

# ---------- Carga de .spydata ----------
def safe_pickle_load(spy_path: Path) -> Any:
    """
    Carga .spydata de forma robusta.
    1) Usa el cargador oficial de Spyder (spyder_kernels.utils.iofuncs.load_dictionary)
    2) Si no está disponible o falla, intenta: pickle directo, gzip+pickle, zip con pkl/json, joblib.
    """
    # 1) Cargador oficial de Spyder
    try:
        from spyder_kernels.utils.iofuncs import load_dictionary
        data, error = load_dictionary(str(spy_path))
        if error:
            raise RuntimeError(error)
        return data
    except Exception:
        pass

    # 2) pickle directo
    try:
        with spy_path.open('rb') as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3) gzip + pickle
    try:
        with gzip.open(spy_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass

    # 4) zipfile con pkl/json dentro
    try:
        import zipfile
        if zipfile.is_zipfile(spy_path):
            with zipfile.ZipFile(spy_path, 'r') as zf:
                names = zf.namelist()
                # prioridad: *.pickle / *.pkl / *.pckl
                for ext in ('.pickle', '.pkl', '.pckl'):
                    cands = [n for n in names if n.lower().endswith(ext)]
                    if cands:
                        with zf.open(cands[0], 'r') as f:
                            return pickle.load(f)
                
                # luego JSON
                json_cands = [n for n in names if n.lower().endswith('.json')]
                if json_cands:
                    import json as _json
                    with zf.open(json_cands[0], 'r') as f:
                        return _json.load(f)
                
                # último intento: probar el primer miembro como pickle
                for n in names:
                    try:
                        with zf.open(n, 'r') as f:
                            return pickle.load(f)
                    except Exception:
                        continue
            raise RuntimeError('Zip reconocido pero sin pickle/json legible dentro')
    except Exception:
        pass

    # 5) joblib
    try:
        import joblib
        return joblib.load(spy_path)
    except Exception:
        pass

    # 6) TAR archive (.spydata en algunos casos)
    try:
        import tarfile

        if tarfile.is_tarfile(spy_path):
            with tarfile.open(spy_path, "r") as tar:
                for member in tar.getmembers():

                    if member.isfile():
                        f = tar.extractfile(member)

                        if f is None:
                            continue

                        try:
                            return pickle.load(f)
                        except Exception:
                            try:
                                import json
                                return json.load(f)
                            except Exception:
                                continue

            raise RuntimeError("TAR reconocido pero sin pickle/json legible dentro")

    except Exception:
        pass

    raise RuntimeError(f"No pude cargar {spy_path}: formato .spydata no reconocido (usé spyder_kernels, pickle, gzip, zip, joblib)")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# ---------- LECTOR ROBUSTO + CACHE POR WORKER ----------
# Normalización a uint8 para imágenes 16-bit u otros formatos

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype(np.uint8)


def read_image_rgb(path: Path) -> np.ndarray:
    """Lee imagen y devuelve **RGB uint8** (H,W,3). Intenta: cv2 COLOR → cv2 UNCHANGED → tifffile → PIL."""
    # 1) OpenCV color
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2) OpenCV unchanged
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 2:
            if img.dtype != np.uint8:
                img = _normalize_to_uint8(img)
            return np.dstack([img, img, img])
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)
        if img.ndim == 2:
            return np.dstack([img, img, img])
        return img

    # 3) tifffile
    try:
        import tifffile as tiff
        arr = tiff.imread(str(path))
        if arr.ndim == 2:
            arr = _normalize_to_uint8(arr)
            return np.dstack([arr, arr, arr])
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8:
                arr = _normalize_to_uint8(arr)
            return arr[..., :3]
        if arr.dtype != np.uint8:
            arr = _normalize_to_uint8(arr)
        return np.dstack([arr, arr, arr])
    except Exception:
        pass

    # 4) PIL
    try:
        from PIL import Image
        arr = np.array(Image.open(str(path)))
        if arr.ndim == 2:
            arr = _normalize_to_uint8(arr)
            return np.dstack([arr, arr, arr])
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8:
                arr = _normalize_to_uint8(arr)
            return arr[..., :3]
        if arr.dtype != np.uint8:
            arr = _normalize_to_uint8(arr)
        return np.dstack([arr, arr, arr])
    except Exception:
        pass

    raise RuntimeError(f"No pude leer imagen con ningún método: {path}")


# Cache simple por proceso (cada worker tiene su propio dict) # ESTA FUNCIÓN PASARLA A utils/io.py
_IMG_CACHE: dict[str, np.ndarray] = {}

def get_image_cached(path: Path) -> np.ndarray:
    """Devuelve la imagen RGB uint8 cacheada. Si no está, la carga y la guarda."""
    key = str(path)
    arr = _IMG_CACHE.get(key)
    if arr is None:
        arr = read_image_rgb(path)
        _IMG_CACHE[key] = arr
    return arr

def load_state_dict_safe(path, device):
    # En versiones nuevas de torch: usar weights_only=True.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # torch viejo sin ese flag → modo clásico
        return torch.load(path, map_location=device)
