"""Raíz de composición: construye el servicio con sus dependencias reales.

Es el único lugar que conoce todas las piezas (detector, CSV con HMAC, SVM, ORB).
La interfaz, ``sistema.py`` y los scripts llaman a ``build_service``.

Primer arranque: si no existe la clave HMAC se crea en ``state_dir``; si no
existe la BD se copia la semilla ``data/famosos.csv`` y se firma (ver
``database.csv_store``).
"""
from __future__ import annotations

from typing import Mapping, Optional

from database.csv_store import CsvIdentityStore
from database.integrity import load_or_create_key
from recognition.config import Settings, load_settings
from recognition.model_lifecycle import ModelLifecycle, verified_model_path
from recognition.recognition import OrbDatabaseMatcher, RecognitionEngine, SvmClassifier
from recognition.service import FaceRecognitionService
from recognition.types import FaceProcessor


def default_processor() -> FaceProcessor:
    """Procesador real (MediaPipe). Se importa aquí para no cargar MediaPipe en los tests."""
    from recognition.detector import process_image
    return process_image


def build_service(settings: Optional[Settings] = None, env: Optional[Mapping[str, str]] = None,
                  processor: Optional[FaceProcessor] = None) -> FaceRecognitionService:
    settings = settings or load_settings(env)
    paths = settings.paths
    key, _ = load_or_create_key(paths.key_file)
    store = CsvIdentityStore(paths.db_path, key, paths.manifest, paths.seed_db)
    lifecycle = ModelLifecycle(paths, settings.svm, key)
    orb = OrbDatabaseMatcher(settings.thresholds, settings.orb.min_keypoints)
    svm = SvmClassifier(paths.svm_model,
                        verifier=lambda: verified_model_path(paths, key),
                        freshness_check=lambda: lifecycle.assert_current(store))
    process = processor or default_processor()
    engine = RecognitionEngine(process, svm, orb, store, settings.thresholds)
    return FaceRecognitionService(store, lifecycle, engine, process, orb, settings)
