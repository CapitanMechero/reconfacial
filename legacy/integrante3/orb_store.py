"""Almacenamiento local de templates ORB (un archivo .npz por persona).

Equivalente ORB del CSV de HOG (legacy/dataset.csv). Provisional y SIN cifrar:
cuando exista database/manager.py (SQLite + AES-GCM) basta con guardar
``template.to_bytes()`` allí y recuperarlo con ``OrbTemplate.from_bytes``.
"""
from __future__ import annotations

import base64
import csv
import re
from pathlib import Path

import numpy as np

from recognition.orb import OrbTemplate

DEFAULT_DIR = Path("data/orb")


def _path(name: str, directory) -> Path:
    slug = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    if not slug:
        raise ValueError("El nombre no puede estar vacío.")
    return Path(directory) / f"{slug}.npz"


def save_template(name: str, template: OrbTemplate, directory=DEFAULT_DIR,
                  replace: bool = False) -> Path:
    name = name.strip()
    path = _path(name, directory)
    if path.exists() and not replace:
        raise ValueError(f"Ya existe una persona registrada como «{name}».")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, name=np.array(name), data=np.frombuffer(template.to_bytes(), np.uint8))
    return path


def load_database(directory=DEFAULT_DIR) -> dict[str, OrbTemplate]:
    """{nombre: OrbTemplate} de todas las personas registradas."""
    db = {}
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            db[str(f["name"])] = OrbTemplate.from_bytes(f["data"].tobytes())
    return db


def delete_template(name: str, directory=DEFAULT_DIR) -> bool:
    path = _path(name, directory)
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset en CSV (equivalente ORB de data/dataset.csv de HOG)
#   name,image,n_keypoints,template     (una fila por foto; template = base64 de to_bytes())
# ---------------------------------------------------------------------------
DATASET_CSV = Path("data/orb_dataset.csv")
CSV_HEADER = ["name", "image", "n_keypoints", "template"]


def write_dataset_csv(rows, path=DATASET_CSV) -> Path:
    """rows: iterable de (nombre, imagen, OrbTemplate). Sobrescribe el archivo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for name, image, tpl in rows:
            w.writerow([name, image, len(tpl), base64.b64encode(tpl.to_bytes()).decode("ascii")])
    return path


def load_dataset_csv(path=DATASET_CSV) -> dict[str, list[tuple[str, OrbTemplate]]]:
    """{nombre: [(imagen, OrbTemplate), ...]} con todas las fotos de cada persona."""
    csv.field_size_limit(2**31 - 1)
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tpl = OrbTemplate.from_bytes(base64.b64decode(row["template"]))
            data.setdefault(row["name"], []).append((row["image"], tpl))
    return data
