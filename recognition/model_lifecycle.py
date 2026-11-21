"""Ciclo de vida del modelo SVM: reentrenar, validar, sustituir atómicamente y sincronizar.

La base de datos cifrada es la **fuente de verdad**; el SVM es un artefacto
derivado. ``svm_runtime.meta.json`` registra a qué estado de la base de datos
(``generation``) corresponde el modelo, su hash SHA-256 y sus clases.

Garantías
---------
* Entrenamiento y validación se hacen sobre archivos temporales; solo si todo
  es correcto se sustituye el modelo con ``os.replace`` (atómico en el mismo
  sistema de archivos). Una interrupción nunca deja un archivo a medias.
* El modelo se publica antes que su meta. Si el proceso muere entre ambos,
  hash y meta no coinciden: el modelo se trata como NO válido hasta reentrenar
  (se omite T1; ORB sigue funcionando). Nunca se usa un modelo no verificado.
* Alta de identidad que falla al reentrenar: se conserva el último modelo
  válido y se marca ``STALE``.
* Baja de identidad: el modelo anterior codifica datos de la persona eliminada,
  así que si no se puede reentrenar se **descarta** (privacidad sobre disponibilidad).
* Con menos de 2 identidades no puede existir SVM: se elimina el modelo.

Limitación: ``train_accuracy`` mide cuánto memoriza el modelo sus ejemplos; NO
estima su capacidad de generalización (ver ``docs/evaluacion.md``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from database.filelock import interprocess_lock
from database.integrity import sign, verify
from recognition.config import Paths, SvmSettings
from recognition.errors import StorageError, SvmUnavailableError
from recognition.svm import load_model, save_model, train_svm

log = logging.getLogger(__name__)
META_FORMAT = 1


class ModelStatus(str, Enum):
    UP_TO_DATE = "up_to_date"
    STALE = "stale"                      # existe pero la BD ha cambiado
    MISSING = "missing"
    INVALID = "invalid"                  # hash/meta no coinciden o ilegible
    NOT_ENOUGH_IDENTITIES = "not_enough_identities"


@dataclass(frozen=True)
class TrainingReport:
    status: ModelStatus
    trained: bool
    n_identities: int = 0
    n_samples: int = 0
    min_samples_per_identity: int = 0
    train_accuracy: Optional[float] = None
    warnings: tuple[str, ...] = ()
    error: Optional[str] = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


META_DOMAIN = "facerec-svm-meta-v1"


def _meta_mac(key: bytes, meta: dict) -> str:
    body = {k: v for k, v in meta.items() if k != "hmac"}
    return sign(key, json.dumps(body, sort_keys=True), domain=META_DOMAIN)


def read_meta(paths: Paths, key: Optional[bytes] = None) -> Optional[dict]:
    """Lee el meta del SVM. Con ``key`` verifica además su HMAC: sin la clave nadie
    puede sustituir modelo + meta (el modelo es un pickle: cargar uno ajeno
    permitiría ejecutar código)."""
    try:
        meta = json.loads(paths.svm_meta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise StorageError(f"Metadatos del SVM ilegibles: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("format") != META_FORMAT:
        raise StorageError("Formato de metadatos del SVM no soportado.")
    if key is not None and not verify(key, meta.get("hmac", ""), json.dumps(
            {k: v for k, v in meta.items() if k != "hmac"}, sort_keys=True), domain=META_DOMAIN):
        raise StorageError("La firma de los metadatos del SVM no es válida.")
    return meta


def verified_model_path(paths: Paths, key: Optional[bytes] = None) -> Path:
    """Ruta del modelo tras verificar meta (HMAC) y hash. Lanza ``SvmUnavailableError``."""
    try:
        meta = read_meta(paths, key)
    except StorageError as exc:
        raise SvmUnavailableError(str(exc)) from exc
    if meta is None or not paths.svm_model.is_file():
        raise SvmUnavailableError("No hay un modelo SVM verificado.")
    if sha256_file(paths.svm_model) != meta.get("sha256"):
        raise SvmUnavailableError("El modelo SVM no coincide con su hash registrado.")
    return paths.svm_model


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ModelLifecycle:
    """Mantiene el SVM coherente con el almacén de identidades."""

    def __init__(self, paths: Paths, settings: SvmSettings, key: Optional[bytes] = None) -> None:
        self._paths = paths
        self._cfg = settings
        self._key = key
        self._lock = threading.Lock()

    # -- estado ----------------------------------------------------------------
    def status(self, store) -> ModelStatus:
        n_identities = len(set(store.load_hog_training_set()[1]))
        generation = store.generation
        if n_identities < self._cfg.min_identities:
            return ModelStatus.NOT_ENOUGH_IDENTITIES
        try:
            verified_model_path(self._paths, self._key)
            meta = read_meta(self._paths, self._key)
        except SvmUnavailableError:
            return ModelStatus.MISSING if not self._paths.svm_model.exists() \
                else ModelStatus.INVALID
        assert meta is not None
        return ModelStatus.UP_TO_DATE if meta.get("generation") == generation else ModelStatus.STALE

    def assert_current(self, store) -> None:
        """Lanza ``SvmUnavailableError`` si el modelo no corresponde a la BD actual."""
        try:
            meta = read_meta(self._paths, self._key)
        except StorageError as exc:
            raise SvmUnavailableError(str(exc)) from exc
        if meta is None:
            raise SvmUnavailableError("No hay un modelo SVM verificado.")
        if meta.get("generation") != store.generation:
            raise SvmUnavailableError(
                "El SVM está desactualizado respecto a las identidades registradas.")

    def model_classes(self) -> list[str]:
        meta = read_meta(self._paths, self._key)
        return [] if meta is None else list(meta.get("classes", []))

    # -- reentrenamiento -------------------------------------------------------------
    def retrain(self, store, *, discard_on_failure: bool = False) -> TrainingReport:
        """Reconstruye el SVM desde la base de datos.

        ``discard_on_failure=True`` (usar tras una BAJA): si no se puede entrenar,
        se elimina el modelo previo en lugar de conservarlo.
        """
        with self._lock, interprocess_lock(self._paths.models_dir / ".retrain.lock"):
            X, y, generation = store.load_hog_training_set()
            identities, counts = np.unique(y, return_counts=True)
            n_id, n_samples = len(identities), len(y)

            if n_id < self._cfg.min_identities:
                self._remove_model()
                return TrainingReport(
                    ModelStatus.NOT_ENOUGH_IDENTITIES, False, n_id, n_samples,
                    warnings=(f"Se necesitan al menos {self._cfg.min_identities} identidades "
                              "para entrenar el SVM; T1 queda desactivado.",))
            try:
                return self._train_and_publish(X, y, generation, identities, counts)
            except (ValueError, OSError, SvmUnavailableError) as exc:
                log.error("Reentrenamiento fallido: %s", exc)
                if discard_on_failure:
                    self._remove_model()
                    kept = "modelo anterior descartado por contener datos de una identidad eliminada"
                else:
                    kept = "se conserva el último modelo válido"
                status = ModelStatus.STALE if self._paths.svm_model.exists() else ModelStatus.MISSING
                return TrainingReport(status, False, n_id, n_samples,
                                      warnings=(kept,), error=str(exc))

    def _train_and_publish(self, X, y, generation, identities, counts) -> TrainingReport:
        paths = self._paths
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        if counts.min() < 2:
            warnings.append(
                f"{int((counts < 2).sum())} identidad(es) con una única muestra: el SVM "
                "no puede aprender variación intra-clase y generalizará mal a otras "
                "fotos. Registra varias fotografías por persona.")

        model = train_svm(X, y, C=self._cfg.C)
        accuracy = float(model.score(X, y))
        if sorted(map(str, model.classes_)) != sorted(map(str, identities)):
            raise ValueError("El modelo entrenado no contiene exactamente las identidades de la BD.")
        if accuracy < self._cfg.min_train_accuracy:
            raise ValueError(f"Precisión de entrenamiento {accuracy:.2f} por debajo del mínimo "
                             f"{self._cfg.min_train_accuracy:.2f}; modelo rechazado.")

        fd, tmp_name = tempfile.mkstemp(dir=paths.models_dir, prefix="svm-", suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            save_model(model, tmp)
            reloaded = load_model(tmp)                       # round-trip
            if not np.array_equal(reloaded.predict(X[:1]), model.predict(X[:1])):
                raise ValueError("El modelo guardado no reproduce las predicciones.")
            digest = sha256_file(tmp)
            os.replace(tmp, paths.svm_model)                  # 1) modelo
            meta = {
                "format": META_FORMAT, "sha256": digest, "generation": generation,
                "classes": [str(c) for c in model.classes_], "n_samples": int(len(y)),
                "train_accuracy": accuracy,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if self._key is not None:
                meta["hmac"] = _meta_mac(self._key, meta)
            _atomic_write_text(paths.svm_meta, json.dumps(meta, indent=2))  # 2) meta
        finally:
            tmp.unlink(missing_ok=True)
        log.info("SVM reentrenado: %d identidades, %d muestras", len(identities), len(y))
        return TrainingReport(ModelStatus.UP_TO_DATE, True, len(identities), int(len(y)),
                              int(counts.min()), accuracy, tuple(warnings))

    def _remove_model(self) -> None:
        for p in (self._paths.svm_meta, self._paths.svm_model):  # meta primero: nunca meta sin modelo válido
            p.unlink(missing_ok=True)

    def ensure_current(self, store) -> ModelStatus:
        """Recupera tras interrupciones: reentrena si el modelo no está al día."""
        status = self.status(store)
        if status in (ModelStatus.STALE, ModelStatus.MISSING, ModelStatus.INVALID):
            return self.retrain(store).status
        return status
