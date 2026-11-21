"""Caras sintéticas deterministas para probar HOG/SVM/ORB sin fotos reales.

Son ruido suavizado: tienen textura suficiente para ORB y vectores HOG
distintos por semilla. NO sustituyen a fotografías reales: sirven para probar
contratos e integración, no para estimar rendimiento.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    return cv2.GaussianBlur(raw, (0, 0), 1.5)


def make_variant(face: np.ndarray, seed: int, shift: tuple[float, float] = (1.5, -1.0)) -> np.ndarray:
    """Otra «foto» de la misma cara: brillo/contraste, ruido y leve desplazamiento."""
    rng = np.random.default_rng(10_000 + seed)
    g = face.astype(np.float32) * 1.1 + 8 + rng.normal(0, 4, face.shape)
    m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(np.clip(g, 0, 255).astype(np.uint8), m, (128, 128))


def tmp_settings(tmp_path, thresholds=None, seed=None):
    """Configuración aislada en ``tmp_path`` (nunca toca var/, data/ ni models/)."""
    from recognition.config import Paths, Settings, Thresholds
    paths = Paths(data_dir=tmp_path / "data", models_dir=tmp_path / "data" / "models",
                  db_path=tmp_path / "data" / "db.csv", state_dir=tmp_path / "state",
                  seed_db=seed or (tmp_path / "no_seed.csv"))
    return Settings(paths=paths, thresholds=thresholds or Thresholds())
