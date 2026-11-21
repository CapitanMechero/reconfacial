"""Motor de reconocimiento: cascada HOG+SVM (T1) -> ORB (T2) -> desconocido.

Orquesta los algoritmos existentes sin reimplementarlos. No depende de
Streamlit ni de la base de datos: recibe sus dependencias (procesador de caras,
clasificador, matcher y origen de templates) por inyección.

Política de errores
-------------------
* Fallos esperables de la foto (sin cara, varias caras, imagen inválida…) y de
  infraestructura (modelo ausente, template corrupto, clave incorrecta) se
  devuelven como ``RecognitionStatus.ERROR`` con un ``ProcessingError`` tipado.
  **Nunca** se convierten en «desconocido».
* Errores de programación (excepciones no previstas) se propagan.
* Si el SVM no existe (p. ej. aún hay una sola identidad) se omite T1 y se deja
  constancia en ``notes``; el resultado no se disfraza.
"""
from __future__ import annotations

import logging
import pickle
from typing import Mapping, Optional, Sequence

import numpy as np

from recognition.config import Thresholds
from recognition.errors import (
    FaceProcessingError, FaceTooSmallError, InvalidImageError, KeyConfigurationError,
    ModelNotFoundError, MultipleFacesError, NoFaceDetectedError, StorageError,
    SvmUnavailableError, TemplateCorruptError,
)
from recognition.hog import extract_hog
from recognition.matching import match_identity
from recognition.orb import OrbExtractionError, OrbTemplate, extract_orb
from recognition.svm import decision_scores, load_model, passes_threshold
from recognition.types import (
    ErrorCode, FaceProcessor, HogPrediction, HogSvmClassifier, ImageInput, OrbMatcher,
    OrbOutcome, ProcessingError, RecognitionResult, RecognitionStatus, ScoreKind, Stage,
    RegisteredFace, TemplateSource, VerificationResult, VerificationStatus,
)

log = logging.getLogger(__name__)
NO_EVIDENCE_NOTE = ("La entrada no tiene coincidencias ORB con ninguna cara inscrita (posiblemente "
                    "no es una cara); no se acepta.")


# ---------------------------------------------------------------------------
# Adaptadores sobre los módulos existentes
# ---------------------------------------------------------------------------
class SvmClassifier:
    """Adaptador de ``recognition.svm``. Carga el modelo bajo demanda y detecta cambios.

    Se recarga si el archivo cambia (``mtime``), de modo que un reentrenamiento
    sustituido atómicamente se aplica sin reiniciar la aplicación.
    """

    def __init__(self, model_path, verifier=None, freshness_check=None) -> None:
        """``verifier``: callable opcional que valida el modelo (p. ej. su hash) antes de
        cargarlo y lanza ``SvmUnavailableError`` si no es fiable (joblib usa pickle).
        ``freshness_check``: callable barato ejecutado en CADA predicción; debe lanzar
        ``SvmUnavailableError`` si el modelo no corresponde al estado actual de la BD
        (un SVM desactualizado no conoce a las identidades nuevas y las confundiría)."""
        self._path = model_path
        self._verifier = verifier
        self._freshness = freshness_check
        self._model = None
        self._mtime_ns: Optional[int] = None

    def _current_model(self):
        try:
            mtime = self._path.stat().st_mtime_ns
        except FileNotFoundError as exc:
            self._model = None
            raise SvmUnavailableError(f"No existe el modelo SVM en {self._path}") from exc
        if self._model is None or mtime != self._mtime_ns:
            if self._verifier is not None:
                self._model = None
                self._verifier()   # solo al (re)cargar: evita hashear en cada consulta
            try:
                self._model = load_model(self._path)
            except (OSError, EOFError, ValueError, KeyError, AttributeError, ImportError,
                    pickle.UnpicklingError) as exc:
                self._model = None
                raise SvmUnavailableError(f"Modelo SVM ilegible ({self._path}): {exc}") from exc
            self._mtime_ns = mtime
        return self._model

    def scores(self, face: np.ndarray) -> dict[str, float]:
        return self.scores_hog(extract_hog(face))

    def scores_hog(self, hog: np.ndarray) -> dict[str, float]:
        if self._freshness is not None:
            self._freshness()
        classes, scores = decision_scores(hog, self._current_model())
        return {c: float(s) for c, s in zip(classes, scores)}

    def predict(self, face: np.ndarray) -> HogPrediction:
        return self.predict_hog(extract_hog(face))

    def predict_hog(self, hog: np.ndarray) -> HogPrediction:
        if self._freshness is not None:
            self._freshness()
        model = self._current_model()
        classes, scores = decision_scores(hog, model)
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        runner_up = float(scores[order[1]]) if len(order) > 1 else None
        return HogPrediction(classes[best], float(scores[best]), runner_up)


class OrbDatabaseMatcher:
    """Adaptador de ``recognition.orb`` + ``recognition.matching``."""

    def __init__(self, thresholds: Thresholds, min_keypoints: Optional[int] = None) -> None:
        self._t = thresholds
        self._min_kp = min_keypoints

    def match(
        self, face: np.ndarray, references: Mapping[str, Sequence[OrbTemplate]]
    ) -> OrbOutcome:
        try:
            kwargs = {} if self._min_kp is None else {"min_keypoints": self._min_kp}
            query = extract_orb(face, **kwargs)
        except OrbExtractionError:
            return OrbOutcome(None, 0.0, 0.0, 0.0, False, len(references),
                              insufficient_keypoints=True)
        return self.match_template(query, references)

    def match_template(
        self, query: OrbTemplate, references: Mapping[str, Sequence[OrbTemplate]]
    ) -> OrbOutcome:
        """Como ``match`` pero con un template ya extraído (p. ej. al registrar)."""
        res = match_identity(query, dict(references), t2=self._t.t2,
                             min_margin=self._t.orb_min_margin)
        return OrbOutcome(
            best_candidate=res.best_candidate, score=res.score, second_score=res.second_score,
            margin=res.margin, accepted=res.accepted, n_identities=len(references),
            scores=dict(res.scores),
        )


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------
_ERROR_CODES: tuple[tuple[type, ErrorCode], ...] = (
    (MultipleFacesError, ErrorCode.MULTIPLE_FACES),
    (NoFaceDetectedError, ErrorCode.NO_FACE),
    (FaceTooSmallError, ErrorCode.FACE_TOO_SMALL),
    (InvalidImageError, ErrorCode.INVALID_IMAGE),
)


def error_from_processing(exc: Exception) -> ProcessingError:
    """Traduce una excepción conocida de procesamiento a ``ProcessingError``."""
    if isinstance(exc, ModelNotFoundError):
        return ProcessingError(ErrorCode.DETECTOR_MODEL_MISSING, str(exc))
    for cls, code in _ERROR_CODES:
        if isinstance(exc, cls):
            return ProcessingError(code, str(exc))
    if isinstance(exc, FaceProcessingError):
        return ProcessingError(ErrorCode.INVALID_IMAGE, str(exc))
    raise exc


class _FaceQuery:
    """Consulta a partir de una cara: cada adaptador extrae sus características."""

    def __init__(self, face: np.ndarray) -> None:
        self.face = face

    def predict(self, svm):
        return svm.predict(self.face)

    def scores(self, svm):
        return svm.scores(self.face)

    def match(self, orb, refs):
        return orb.match(self.face, refs)


class _TemplateQuery:
    """Consulta a partir de un template ya extraído (métodos ``_t``)."""

    def __init__(self, template: RegisteredFace) -> None:
        self.t = template

    def predict(self, svm):
        return svm.predict_hog(self.t.hog)

    def scores(self, svm):
        return svm.scores_hog(self.t.hog)

    def match(self, orb, refs):
        return orb.match_template(self.t.orb, refs)


class RecognitionEngine:
    """Cascada de reconocimiento con dependencias inyectables."""

    def __init__(
        self,
        processor: FaceProcessor,
        svm: HogSvmClassifier,
        orb: OrbMatcher,
        templates: TemplateSource,
        thresholds: Thresholds,
    ) -> None:
        self._processor = processor
        self._svm = svm
        self._orb = orb
        self._templates = templates
        self._t = thresholds

    @property
    def thresholds(self) -> Thresholds:
        return self._t

    # -- API pública -------------------------------------------------------
    def recognize(self, image: ImageInput) -> RecognitionResult:
        """Imagen -> resultado. Aplica la política de una única cara por foto."""
        try:
            processed = self._processor(image)
        except (FaceProcessingError, ModelNotFoundError) as exc:
            return self._error(error_from_processing(exc))
        return self.recognize_face(processed.face)

    def error_result(self, exc: Exception) -> RecognitionResult:
        """Resultado ERROR para una excepción conocida de procesamiento de imagen."""
        return self._error(error_from_processing(exc))

    def recognize_face(self, face: np.ndarray) -> RecognitionResult:
        """Cara ya normalizada (128x128 gris uint8) -> resultado de la cascada."""
        return self._recognize(_FaceQuery(face))

    def recognize_template(self, template: RegisteredFace) -> RecognitionResult:
        """Template ya extraído -> resultado de la cascada (sin detector ni extractor)."""
        return self._recognize(_TemplateQuery(template))

    def _recognize(self, q) -> RecognitionResult:
        notes: list[str] = []

        try:
            references = self._templates.load_orb_templates()
        except TemplateCorruptError as exc:
            return self._error(ProcessingError(ErrorCode.TEMPLATE_CORRUPT, str(exc)))
        except KeyConfigurationError as exc:
            return self._error(ProcessingError(ErrorCode.KEY_ERROR, str(exc)))
        except StorageError as exc:
            return self._error(ProcessingError(ErrorCode.STORAGE_FAILURE, str(exc)))

        if not references:
            return self._result(RecognitionStatus.UNKNOWN,
                                notes=("No hay identidades registradas.",))

        # ---- Filtro de evidencia ORB (antes de T1) ---------------------------
        # El SVM puntúa cualquier entrada; las que no son caras (``_sd``) o los
        # templates fabricados (``_t``) superan T1 con frecuencia. Se exige que la
        # entrada tenga puntos ORB y alguna coincidencia geométrica con una cara inscrita.
        try:
            orb = q.match(self._orb, references)
        except ValueError as exc:
            return self._error(ProcessingError(ErrorCode.ORB_FAILURE, str(exc), Stage.ORB))
        if not self._has_orb_evidence(orb):
            notes.append(NO_EVIDENCE_NOTE)
            return self._result(RecognitionStatus.UNCERTAIN, stage=Stage.ORB, score=orb.score,
                                kind=ScoreKind.ORB_INLIER_RATIO, orb=orb, notes=notes)

        # ---- T1: HOG + SVM ------------------------------------------------
        hog: Optional[HogPrediction] = None
        try:
            hog = q.predict(self._svm)
        except SvmUnavailableError as exc:
            notes.append(f"T1 omitido: {exc}")
            log.warning("T1 omitido: %s", exc)
        except ValueError as exc:
            return self._error(ProcessingError(ErrorCode.SVM_FAILURE, str(exc), Stage.HOG_SVM))

        if hog is not None:
            if hog.identity not in references:
                # El modelo es anterior a una eliminación: nunca se identifica a
                # alguien que ya no está registrado.
                notes.append(
                    "El SVM propone una identidad que ya no está registrada; "
                    "modelo desactualizado, se ignora T1.")
                log.warning("SVM desactualizado")
            elif passes_threshold(hog.score, self._t.t1):
                return self._result(
                    RecognitionStatus.IDENTIFIED, identity=hog.identity, stage=Stage.HOG_SVM,
                    score=hog.score, kind=ScoreKind.SVM_DECISION, hog=hog, notes=notes)

        # ---- T2: ORB (ya calculado en el filtro) ------------------------------
        common = dict(stage=Stage.ORB, score=orb.score, kind=ScoreKind.ORB_INLIER_RATIO,
                      hog=hog, orb=orb)
        if orb.insufficient_keypoints:
            notes.append("La cara no tiene puntos clave ORB suficientes para comparar.")
            return self._result(RecognitionStatus.UNCERTAIN, notes=notes, **common)
        if orb.accepted and orb.best_candidate is not None:
            if hog is not None and hog.identity != orb.best_candidate:
                notes.append(f"T1 sugería «{hog.identity}», pero ORB aceptó otra identidad.")
            return self._result(RecognitionStatus.IDENTIFIED, identity=orb.best_candidate,
                                notes=notes, **common)
        if orb.score >= self._t.t2:
            notes.append("Coincidencia ORB ambigua: el margen con el 2.º candidato es insuficiente.")
            return self._result(RecognitionStatus.UNCERTAIN, notes=notes, **common)
        return self._result(RecognitionStatus.UNKNOWN, notes=notes, **common)

    # -- verificación 1:1 ------------------------------------------------------
    def verify(self, claimed: str, image: ImageInput) -> VerificationResult:
        try:
            processed = self._processor(image)
        except (FaceProcessingError, ModelNotFoundError) as exc:
            return self._verror(claimed, error_from_processing(exc))
        return self.verify_face(claimed, processed.face)

    def verify_face(self, claimed: str, face: np.ndarray) -> VerificationResult:
        return self._verify(claimed, _FaceQuery(face))

    def verify_template(self, claimed: str, template: RegisteredFace) -> VerificationResult:
        return self._verify(claimed, _TemplateQuery(template))

    def _verify(self, claimed: str, q) -> VerificationResult:
        """¿Corresponde la consulta a la identidad ``claimed``?

        T1v: margen del SVM para la clase reclamada >= ``t1_verify``.
        Si no, T2v: mejor puntuación ORB contra los templates de esa identidad
        >= ``t2_verify``. Solo se compara con la identidad reclamada.
        """
        notes: list[str] = []
        try:
            references = self._templates.load_orb_templates()
        except TemplateCorruptError as exc:
            return self._verror(claimed, ProcessingError(ErrorCode.TEMPLATE_CORRUPT, str(exc)))
        except KeyConfigurationError as exc:
            return self._verror(claimed, ProcessingError(ErrorCode.KEY_ERROR, str(exc)))
        except StorageError as exc:
            return self._verror(claimed, ProcessingError(ErrorCode.STORAGE_FAILURE, str(exc)))

        claimed = str(claimed)
        name = next((n for n in references if n.casefold() == claimed.strip().casefold()), None)
        if name is None:
            return self._vresult(VerificationStatus.REJECTED, claimed,
                                 notes=(f"«{claimed}» no está inscrito.",))

        try:
            orb_all = q.match(self._orb, references)
        except ValueError as exc:
            return self._verror(claimed, ProcessingError(ErrorCode.ORB_FAILURE, str(exc), Stage.ORB))
        orb_claimed = orb_all.scores.get(
            name, orb_all.score if orb_all.best_candidate == name else 0.0)
        if not self._has_orb_evidence(orb_all):
            return self._vresult(VerificationStatus.REJECTED, name, Stage.ORB, None, orb_claimed,
                                 notes=(NO_EVIDENCE_NOTE,))

        svm_score: Optional[float] = None
        try:
            svm_score = q.scores(self._svm).get(name)
            if svm_score is None:
                notes.append("El SVM no conoce a esta identidad (modelo desactualizado); se omite T1.")
        except SvmUnavailableError as exc:
            notes.append(f"T1 omitido: {exc}")
        except ValueError as exc:
            return self._verror(claimed, ProcessingError(ErrorCode.SVM_FAILURE, str(exc),
                                                         Stage.HOG_SVM))
        if svm_score is not None and passes_threshold(svm_score, self._t.t1_verify):
            return self._vresult(VerificationStatus.ACCEPTED, name, Stage.HOG_SVM, svm_score,
                                 notes=notes)
        accepted = orb_claimed >= self._t.t2_verify
        return self._vresult(VerificationStatus.ACCEPTED if accepted else VerificationStatus.REJECTED,
                             name, Stage.ORB, svm_score, orb_claimed, notes)

    def _has_orb_evidence(self, orb: OrbOutcome) -> bool:
        return not orb.insufficient_keypoints and orb.score > self._t.orb_min_evidence

    def verify_error(self, claimed: str, exc: Exception) -> VerificationResult:
        return self._verror(claimed, error_from_processing(exc))

    def _vresult(self, status, claimed, stage=Stage.NONE, svm_score=None, orb_score=None,
                 notes=()) -> VerificationResult:
        return VerificationResult(status, claimed, stage, svm_score, orb_score,
                                  thresholds_calibrated=self._t.calibrated, notes=tuple(notes))

    def _verror(self, claimed: str, err: ProcessingError) -> VerificationResult:
        log.error("Verificación fallida: %s (%s)", err.code.value, err.message)
        return VerificationResult(VerificationStatus.ERROR, claimed, err.stage, error=err,
                                  thresholds_calibrated=self._t.calibrated)

    # -- utilidades ----------------------------------------------------------
    def _result(self, status, identity=None, stage=Stage.NONE, score=None,
                kind=ScoreKind.NONE, hog=None, orb=None, notes=()) -> RecognitionResult:
        return RecognitionResult(
            status=status, identity=identity, stage=stage, score=score, score_kind=kind,
            hog=hog, orb=orb, thresholds_calibrated=self._t.calibrated, notes=tuple(notes))

    def _error(self, err: ProcessingError, hog: Optional[HogPrediction] = None) -> RecognitionResult:
        log.error("Reconocimiento fallido: %s (%s)", err.code.value, err.message)
        return RecognitionResult(
            status=RecognitionStatus.ERROR, stage=err.stage, error=err, hog=hog,
            thresholds_calibrated=self._t.calibrated)
