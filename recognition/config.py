"""Configuración centralizada: umbrales, rutas, parámetros y logging.

Los parámetros de HOG, ORB y matching se **definen** en sus módulos
(``hog.py``, ``orb.py``, ``matching.py``) para no duplicarlos y se
**re-exportan** aquí para tener un único punto de consulta. Los umbrales T1/T2,
las rutas y el logging se definen únicamente aquí.

Precedencia de valores (de menor a mayor):
    1. Valores por defecto (PROVISIONALES: no están calibrados).
    2. ``models/thresholds.json`` generado por ``evaluation.calibrate`` con el
       conjunto de CALIBRACIÓN (nunca con el de test).
    3. Variables de entorno ``FACEREC_*``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional

from recognition.hog import HOG_PARAMS, N_FEATURES
from recognition.matching import (
    MAX_DISPLACEMENT, MAX_HAMMING, MIN_MARGIN, RANSAC_THRESH, RATIO,
)
from recognition.orb import MIN_KEYPOINTS, ORB_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "FACEREC_"

# Valores por defecto PROVISIONALES. Hasta que ``thresholds.json`` exista, el
# sistema lo advierte (``Settings.thresholds_calibrated == False``).
DEFAULT_T1 = 0.0   # margen LinearSVC; 0.0 = «gana la clase» (sin exigencia extra)
DEFAULT_T2 = 0.05  # ratio de inliers ORB (escala de matching.py)


@dataclass(frozen=True)
class Thresholds:
    """T1/T2 de identificación (1:N) y de verificación (1:1).

    En verificación se compara solo con la identidad reclamada: T1v se aplica al
    margen del SVM *de esa clase* y T2v a la puntuación ORB contra sus templates.
    """
    t1: float = DEFAULT_T1
    t2: float = DEFAULT_T2
    orb_min_margin: float = MIN_MARGIN
    t1_verify: float = DEFAULT_T1
    t2_verify: float = DEFAULT_T2
    # Filtro de evidencia ORB: se exige mejor puntuación ORB (contra todas las
    # identidades) ESTRICTAMENTE mayor que este valor. 0.0 = al menos alguna
    # coincidencia geométrica. Ver docs/evaluacion.md (mitigación A2/B4).
    orb_min_evidence: float = 0.0
    calibrated: bool = False
    source: str = "defaults (provisional, sin calibrar)"


@dataclass(frozen=True)
class SvmSettings:
    C: float = 0.1
    max_iter: int = 20000
    random_state: int = 42
    min_identities: int = 2
    # Validación tras reentrenar: el modelo debe reconocer al menos esta
    # fracción de sus propios ejemplos de entrenamiento (control de cordura,
    # NO una medida de generalización).
    min_train_accuracy: float = 0.9


@dataclass(frozen=True)
class OrbSettings:
    min_keypoints: int = MIN_KEYPOINTS
    params: Mapping[str, object] = field(default_factory=lambda: dict(ORB_PARAMS))
    ratio: float = RATIO
    max_hamming: int = MAX_HAMMING
    max_displacement: float = MAX_DISPLACEMENT
    ransac_thresh: float = RANSAC_THRESH


@dataclass(frozen=True)
class HogSettings:
    params: Mapping[str, object] = field(default_factory=lambda: dict(HOG_PARAMS))
    n_features: int = N_FEATURES


@dataclass(frozen=True)
class Paths:
    """Rutas. En local, el estado de ejecución vive en ``var/`` (ignorado por git).

    * ``seed_db``: CSV versionado con los famosos precargados (sin firmas).
    * ``db_path``: CSV en uso. Si no existe, se crea copiando ``seed_db``.
    * ``state_dir``: clave HMAC y manifiesto firmado. Debe estar SEPARADO de la BD
      (en Docker es otro volumen) para que quien solo pueda escribir la BD no
      pueda re-firmarla ni deshacer cambios sin ser detectado.
    """
    data_dir: Path = PROJECT_ROOT / "var"
    models_dir: Path = PROJECT_ROOT / "var" / "models"
    db_path: Path = PROJECT_ROOT / "var" / "db.csv"
    state_dir: Path = PROJECT_ROOT / "var" / "state"
    seed_db: Path = PROJECT_ROOT / "data" / "famosos.csv"
    holdout: Path = PROJECT_ROOT / "data" / "holdout.csv"
    face_landmarker_dir: Path = PROJECT_ROOT / "models"

    @property
    def key_file(self) -> Path:
        return self.state_dir / "integrity.key"

    @property
    def manifest(self) -> Path:
        return self.state_dir / "manifest.json"

    # Nombre distinto de ``legacy/svm.joblib`` (modelo experimental versionado del
    # flujo CSV): la aplicación gestiona su propio SVM y nunca toca aquel archivo.
    @property
    def svm_model(self) -> Path:
        return self.models_dir / "svm_runtime.joblib"

    @property
    def svm_meta(self) -> Path:
        return self.models_dir / "svm_runtime.meta.json"

    @property
    def face_landmarker(self) -> Path:
        return self.face_landmarker_dir / "face_landmarker.task"

    @property
    def thresholds_file(self) -> Path:
        """Umbrales calibrados del sistema desplegado (versionado junto a la semilla)."""
        return self.seed_db.parent / "thresholds.json"


@dataclass(frozen=True)
class Settings:
    thresholds: Thresholds = field(default_factory=Thresholds)
    svm: SvmSettings = field(default_factory=SvmSettings)
    orb: OrbSettings = field(default_factory=OrbSettings)
    hog: HogSettings = field(default_factory=HogSettings)
    paths: Paths = field(default_factory=Paths)
    log_level: str = "INFO"

    @property
    def thresholds_calibrated(self) -> bool:
        return self.thresholds.calibrated


class ConfigError(ValueError):
    """Configuración inválida (umbral no numérico, archivo corrupto…)."""


def _env_float(env: Mapping[str, str], name: str) -> Optional[float]:
    raw = env.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} debe ser numérico, recibido {raw!r}") from exc


def read_thresholds_file(path: Path) -> Optional[Thresholds]:
    """Lee ``thresholds.json``; devuelve None si no existe."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Thresholds(
            t1=float(raw["t1"]),
            t2=float(raw["t2"]),
            orb_min_margin=float(raw.get("orb_min_margin", MIN_MARGIN)),
            t1_verify=float(raw.get("t1_verify", DEFAULT_T1)),
            t2_verify=float(raw.get("t2_verify", DEFAULT_T2)),
            orb_min_evidence=float(raw.get("orb_min_evidence", 0.0)),
            calibrated=True,
            source=str(raw.get("source", path.name)),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"Archivo de umbrales inválido ({path}): {exc}") from exc


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Construye la configuración efectiva (valores por defecto + archivo + entorno)."""
    env = os.environ if env is None else env

    paths = Paths()
    data_dir = env.get(ENV_PREFIX + "DATA_DIR")
    models_dir = env.get(ENV_PREFIX + "MODELS_DIR")
    db_path = env.get(ENV_PREFIX + "DB_PATH")
    state_dir = env.get(ENV_PREFIX + "STATE_DIR")
    seed_db = env.get(ENV_PREFIX + "SEED_DB")
    if data_dir:
        paths = replace(paths, data_dir=Path(data_dir), db_path=Path(data_dir) / "db.csv",
                        models_dir=Path(data_dir) / "models")
    if models_dir:
        paths = replace(paths, models_dir=Path(models_dir))
    if db_path:
        paths = replace(paths, db_path=Path(db_path))
    if state_dir:
        paths = replace(paths, state_dir=Path(state_dir))
    if seed_db:
        paths = replace(paths, seed_db=Path(seed_db))

    thresholds = read_thresholds_file(paths.thresholds_file) or Thresholds()
    t1, t2 = _env_float(env, "T1"), _env_float(env, "T2")
    if t1 is not None or t2 is not None:
        thresholds = replace(
            thresholds,
            t1=thresholds.t1 if t1 is None else t1,
            t2=thresholds.t2 if t2 is None else t2,
            calibrated=False,
            source="override por variable de entorno (no calibrado)",
        )
    if thresholds.t2 < 0:
        raise ConfigError("T2 no puede ser negativo")

    return Settings(thresholds=thresholds, paths=paths,
                    log_level=env.get(ENV_PREFIX + "LOG_LEVEL", "INFO").upper())


def configure_logging(level: str = "INFO") -> None:
    """Logging uniforme. Política: nunca se registran imágenes, vectores ni claves."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
