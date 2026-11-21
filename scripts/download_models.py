"""Descarga el modelo Face Landmarker de MediaPipe y verifica su SHA-256.

Uso: ``python scripts/download_models.py [--dest RUTA]``
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEST)
    dest = ap.parse_args().dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(dest)
        print(f"Descargado en {dest}")
    if sha256(dest) != SHA256:
        print(f"ERROR: el hash de {dest} no coincide con el esperado", file=sys.stderr)
        return 1
    print(f"Modelo verificado: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
