
# /src/utils/logging.py

from pathlib import Path

def log(msg: str, fp: Path):
    print(msg)
    with fp.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
