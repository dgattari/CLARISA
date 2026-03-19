
# src/utils/io.py

import gzip
import pickle
from pathlib import Path
from typing import Any

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