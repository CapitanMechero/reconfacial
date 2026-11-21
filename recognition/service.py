"""Servicio de aplicación: inscripción, identificación, verificación y eliminación.

Es la fachada que usan ``app.py`` y ``sistema.py``. Coordina el motor, el
almacén CSV y el ciclo de vida del SVM para que BD y modelo no se desincronicen.

Cada operación existe en tres niveles de entrada (exigidos por la práctica):

* **foto**: pasa por el detector MediaPipe (exactamente una cara válida).
* **sin detector** (``_sd``): la imagen se trata como una cara ya recortada; solo
  se convierte a gris y se redimensiona a 128x128, sin detectar ni alinear.
* **template** (``_t``): template público ya extraído (``recognition.template``).

Las imágenes solo existen en memoria; nunca se escriben en disco.

Inscripción: todo-o-nada y **solo de identidades nuevas**. Un id que ya existe se
rechaza (``DuplicateIdentityError``): no se pueden añadir fotos a una identidad ya
inscrita (corrige la vulnerabilidad V-01, que permitía añadir la cara de otra
persona). Se rechaza también una cara que ORB ya acepta como OTRA identidad
(``DuplicateFaceError``) salvo ``allow_duplicate_face``. Si el reentrenamiento falla,
la identidad queda inscrita y el informe indica que el SVM está desactualizado.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from database.csv_store import CsvIdentityStore, normalize_name
from recognition.config import Settings
from recognition.errors import (
    DuplicateIdentityError, FaceProcessingError, IdentityNotFoundError, InvalidImageError,
    ModelNotFoundError,
    StorageError,
)
from recognition.model_lifecycle import ModelLifecycle, ModelStatus, TrainingReport
from recognition.orb import OrbExtractionError
from recognition.preprocessing import load_image
from recognition.recognition import OrbDatabaseMatcher, RecognitionEngine
from recognition.template import TemplateInput, as_template, extract_template
from recognition.types import (
    FaceProcessor, Identity, ImageInput, RecognitionResult, RegisteredFace, VerificationResult,
)

log = logging.getLogger(__name__)
FACE_SIZE = 128


class DuplicateFaceError(ValueError):
    """La cara ya coincide (ORB) con otra identidad inscrita."""

    def __init__(self, existing: str):
        self.existing = existing
        super().__init__(f"Esta cara ya está inscrita como «{existing}».")


class PhotoRejectedError(ValueError):
    """Una de las fotografías/templates no se puede usar para inscribir."""

    def __init__(self, index: int, reason: str):
        self.index = index
        super().__init__(f"Fotografía {index + 1}: {reason}")


def face_without_detector(image: ImageInput) -> np.ndarray:
    """Imagen tratada como cara ya recortada -> 128x128 gris uint8 (sin MediaPipe).

    Lanza ``InvalidImageError`` si la imagen no se puede leer.
    """
    bgr = load_image(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    interp = cv2.INTER_AREA if min(gray.shape[:2]) >= FACE_SIZE else cv2.INTER_CUBIC
    return cv2.resize(gray, (FACE_SIZE, FACE_SIZE), interpolation=interp).astype(np.uint8)


@dataclass(frozen=True)
class Analysis:
    """Resultado de identificar una imagen junto con la cara detectada (para mostrarla)."""
    result: RecognitionResult
    processed: Optional[object]  # ProcessedFace (bbox, landmarks, face) o None


@dataclass(frozen=True)
class RegistrationOutcome:
    identity: Identity
    training: TrainingReport


@dataclass(frozen=True)
class DeletionOutcome:
    name: str
    training: TrainingReport
    templates_removed: bool
    absent_from_model: bool


class FaceRecognitionService:
    def __init__(self, store: CsvIdentityStore, lifecycle: ModelLifecycle,
                 engine: RecognitionEngine, processor: FaceProcessor,
                 orb_matcher: OrbDatabaseMatcher, settings: Settings) -> None:
        self._store = store
        self._lifecycle = lifecycle
        self._engine = engine
        self._process = processor
        self._orb = orb_matcher
        self._settings = settings

    @property
    def engine(self) -> RecognitionEngine:
        return self._engine

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def store(self) -> CsvIdentityStore:
        return self._store

    # -- extracción ---------------------------------------------------------------
    def face_from_photo(self, image: ImageInput) -> np.ndarray:
        """Foto -> cara normalizada con el detector. Lanza ``FaceProcessingError``."""
        return self._process(image).face

    def template_from_photo(self, image: ImageInput, detector: bool = True) -> RegisteredFace:
        face = self.face_from_photo(image) if detector else face_without_detector(image)
        return extract_template(face)

    # -- identificación (1:N) --------------------------------------------------------
    def analyze(self, image: ImageInput) -> Analysis:
        """Identifica con detector y devuelve también la cara detectada."""
        try:
            processed = self._process(image)
        except (FaceProcessingError, ModelNotFoundError) as exc:
            return Analysis(self._engine.error_result(exc), None)
        return Analysis(self._engine.recognize_face(processed.face), processed)

    def identify(self, image: ImageInput) -> RecognitionResult:
        return self._engine.recognize(image)

    def identify_without_detector(self, image: ImageInput) -> RecognitionResult:
        try:
            face = face_without_detector(image)
        except InvalidImageError as exc:
            return self._engine.error_result(exc)
        return self._engine.recognize_face(face)

    def identify_template(self, template: TemplateInput) -> RecognitionResult:
        return self._engine.recognize_template(as_template(template))

    # -- verificación (1:1) ----------------------------------------------------------
    def verify(self, name: str, image: ImageInput) -> VerificationResult:
        return self._engine.verify(name, image)

    def verify_without_detector(self, name: str, image: ImageInput) -> VerificationResult:
        try:
            face = face_without_detector(image)
        except InvalidImageError as exc:
            return self._engine.verify_error(name, exc)
        return self._engine.verify_face(name, face)

    def verify_template(self, name: str, template: TemplateInput) -> VerificationResult:
        return self._engine.verify_template(name, as_template(template))

    # -- estado ------------------------------------------------------------------------
    def list_identities(self) -> list[Identity]:
        return self._store.list_identities()

    def model_status(self) -> ModelStatus:
        return self._lifecycle.status(self._store)

    def ensure_model_current(self) -> ModelStatus:
        """Recuperación tras interrupciones; llamar al arrancar."""
        return self._lifecycle.ensure_current(self._store)

    def retrain(self) -> TrainingReport:
        return self._lifecycle.retrain(self._store)

    # -- inscripción --------------------------------------------------------------------
    def _templates_from_images(self, images: Sequence[ImageInput],
                               detector: bool) -> list[RegisteredFace]:
        if not images:
            raise ValueError("Selecciona al menos una fotografía.")
        out = []
        for i, image in enumerate(images):
            try:
                out.append(self.template_from_photo(image, detector))
            except (FaceProcessingError, OrbExtractionError) as exc:
                raise PhotoRejectedError(i, str(exc)) from exc
        return out

    def _check_not_duplicate_face(self, faces: Sequence[RegisteredFace], own: Optional[str]) -> None:
        references = {n: t for n, t in self._store.load_orb_templates().items()
                      if own is None or n.casefold() != own.casefold()}
        if not references:
            return
        for face in faces:
            outcome = self._orb.match_template(face.orb, references)
            if outcome.accepted and outcome.best_candidate:
                raise DuplicateFaceError(outcome.best_candidate)

    def enroll_templates(self, name: str, templates: Sequence[TemplateInput], *,
                         allow_duplicate_face: bool = False) -> RegistrationOutcome:
        """Inscribe una identidad NUEVA a partir de templates."""
        display = self._require_new(name)
        faces = []
        for i, t in enumerate(templates):
            try:
                faces.append(as_template(t))
            except ValueError as exc:
                raise PhotoRejectedError(i, str(exc)) from exc
        if not faces:
            raise ValueError("No hay templates que inscribir.")
        if not allow_duplicate_face:
            self._check_not_duplicate_face(faces, None)
        identity = self._store.add_identity(display, faces)   # vuelve a comprobar el id (atómico)
        return RegistrationOutcome(identity, self._lifecycle.retrain(self._store))

    def _require_new(self, name: str) -> str:
        display, _ = normalize_name(name)
        if self._store.exists(display):
            raise DuplicateIdentityError(
                f"Ya existe una identidad «{display}». Solo se pueden inscribir personas "
                "nuevas: elige otro identificador.")
        return display

    def enroll(self, name: str, images: Sequence[ImageInput], *, detector: bool = True,
               allow_duplicate_face: bool = False) -> RegistrationOutcome:
        self._require_new(name)          # antes de procesar las fotos
        return self.enroll_templates(name, self._templates_from_images(images, detector),
                                     allow_duplicate_face=allow_duplicate_face)

    # -- eliminación ---------------------------------------------------------------------
    def delete(self, name: str) -> DeletionOutcome:
        identity = self._store.get_identity(name)       # lanza IdentityNotFoundError
        self._store.delete_identity(identity.name)
        report = self._lifecycle.retrain(self._store, discard_on_failure=True)
        gone = not self._store.exists(identity.name)
        absent = identity.name not in self._lifecycle.model_classes()
        if not (gone and absent):
            raise StorageError(
                f"La eliminación de «{identity.name}» no se pudo verificar por completo.")
        return DeletionOutcome(identity.name, report, gone, absent)


__all__ = ["FaceRecognitionService", "DuplicateFaceError", "DuplicateIdentityError",
           "PhotoRejectedError",
           "IdentityNotFoundError", "face_without_detector"]
