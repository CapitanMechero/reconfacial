"""Carga de muestras para la evaluación.

* ``load_hog_csv``: ``legacy/dataset.csv`` (solo vectores HOG; sin ORB).
* ``load_image_folder``: ``<raíz>/<identidad>/<foto>`` -> HOG + ORB con el
  pipeline real (MediaPipe). Las fotos que fallan se cuentan y se excluyen.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.splits import Sample

log = logging.getLogger(__name__)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_hog_csv(path: Path) -> list[Sample]:
    df = pd.read_csv(path)
    X = df.drop(columns=["name"]).to_numpy(dtype=np.float32)
    return [Sample(f"csv:{i}", str(n), X[i]) for i, n in enumerate(df["name"])]


def load_image_folder(root: Path) -> tuple[list[Sample], dict[str, str]]:
    from recognition.detector import process_image
    from recognition.errors import FaceProcessingError
    from recognition.hog import extract_hog
    from recognition.orb import OrbExtractionError, extract_orb

    samples, failures = [], {}
    for person in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for img in sorted(p for p in person.iterdir() if p.suffix.lower() in IMAGE_EXT):
            try:
                face = process_image(img).face
                samples.append(Sample(str(img.relative_to(root)), person.name,
                                      extract_hog(face), extract_orb(face)))
            except (FaceProcessingError, OrbExtractionError) as exc:
                failures[str(img.relative_to(root))] = str(exc)
    log.info("%d fotos válidas, %d descartadas", len(samples), len(failures))
    return samples, failures
