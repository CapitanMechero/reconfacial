"""Tipos compartidos del sistema de reconocimiento (contratos entre módulos).

Este módulo no importa MediaPipe, OpenCV ni la base de datos: solo define estructuras
inmutables y ``Protocol`` para que motor, base de datos e interfaz puedan
desarrollarse y probarse por separado.

Los cuatro desenlaces de un reconocimiento son distintos y no se mezclan:

* ``IDENTIFIED``: una etapa aceptó una identidad.
* ``UNCERTAIN``: no se pudo decidir con garantías (p. ej. coincidencia ambigua
  o cara sin textura suficiente para ORB). Nunca debe tratarse como acceso.
* ``UNKNOWN``: se evaluó y ninguna etapa alcanzó su umbral.
* ``ERROR``: fallo técnico (imagen inválida, modelo ausente, template corrupto…).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # evita importar MediaPipe al usar solo los tipos
    from recognition.detector import ProcessedFace
    from recognition.orb import OrbTemplate

# Igual que ``recognition.preprocessing.ImageInput`` (sin importar OpenCV aquí).
ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


class RecognitionStatus(str, Enum):
    IDENTIFIED = "identified"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    ERROR = "error"


class Stage(str, Enum):
    """Etapa de la cascada que produjo la decisión final."""
    HOG_SVM = "hog_svm"   # T1
    ORB = "orb"           # T2
    NONE = "none"         # ninguna llegó a decidir (p. ej. error previo)


class ScoreKind(str, Enum):
    """Qué significa ``score``. Ninguna es una probabilidad calibrada."""
    SVM_DECISION = "svm_decision"      # margen de LinearSVC (sin escala probabilística)
    ORB_INLIER_RATIO = "orb_inlier_ratio"  # inliers geométricos / puntos de la consulta
    NONE = "none"


class ErrorCode(str, Enum):
    INVALID_IMAGE = "invalid_image"
    NO_FACE = "no_face"
    FACE_TOO_SMALL = "face_too_small"
    MULTIPLE_FACES = "multiple_faces"
    DETECTOR_MODEL_MISSING = "detector_model_missing"
    SVM_UNAVAILABLE = "svm_unavailable"
    SVM_FAILURE = "svm_failure"
    ORB_FAILURE = "orb_failure"
    TEMPLATE_CORRUPT = "template_corrupt"
    STORAGE_FAILURE = "storage_failure"
    KEY_ERROR = "key_error"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ProcessingError:
    """Error técnico apto para mostrarse al usuario (``message``) y registrarse."""
    code: ErrorCode
    message: str
    stage: Stage = Stage.NONE


@dataclass(frozen=True)
class HogPrediction:
    """Salida de la etapa T1. ``score`` es el margen de decisión, no una probabilidad."""
    identity: str
    score: float
    runner_up_score: Optional[float] = None  # segunda mejor clase (None si no existe)

    @property
    def margin(self) -> Optional[float]:
        return None if self.runner_up_score is None else self.score - self.runner_up_score


@dataclass(frozen=True)
class OrbOutcome:
    """Salida de la etapa T2 (mejor candidato aunque no se acepte)."""
    best_candidate: Optional[str]
    score: float
    second_score: float
    margin: float
    accepted: bool
    n_identities: int
    scores: Mapping[str, float] = field(default_factory=dict)
    insufficient_keypoints: bool = False  # la consulta no tenía puntos suficientes


@dataclass(frozen=True)
class RecognitionResult:
    status: RecognitionStatus
    identity: Optional[str] = None
    stage: Stage = Stage.NONE
    score: Optional[float] = None
    score_kind: ScoreKind = ScoreKind.NONE
    hog: Optional[HogPrediction] = None
    orb: Optional[OrbOutcome] = None
    error: Optional[ProcessingError] = None
    thresholds_calibrated: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.status is RecognitionStatus.IDENTIFIED) != (self.identity is not None):
            raise ValueError("identity debe estar definida si y solo si status es IDENTIFIED")
        if (self.status is RecognitionStatus.ERROR) != (self.error is not None):
            raise ValueError("error debe estar definido si y solo si status es ERROR")

    @property
    def is_identified(self) -> bool:
        return self.status is RecognitionStatus.IDENTIFIED

    @property
    def candidate(self) -> Optional[str]:
        """Identidad más parecida aunque no se acepte (para mostrarla siempre).

        Si no hubo aceptación, se prefiere el candidato del SVM (mayor rank-1 en la
        evaluación) y, si no existe, el mejor de ORB. NO es una identificación.
        """
        if self.identity is not None:
            return self.identity
        if self.hog is not None:
            return self.hog.identity
        return self.orb.best_candidate if self.orb is not None else None


class VerificationStatus(str, Enum):
    ACCEPTED = "accepted"     # la foto corresponde a la identidad reclamada
    REJECTED = "rejected"     # no alcanza los umbrales, o la identidad no está inscrita
    ERROR = "error"           # fallo técnico (sin cara, BD manipulada…)


@dataclass(frozen=True)
class VerificationResult:
    """Resultado de verificar (1:1) una identidad reclamada."""
    status: VerificationStatus
    claimed: str
    stage: Stage = Stage.NONE
    svm_score: Optional[float] = None   # margen del SVM para la clase reclamada
    orb_score: Optional[float] = None   # mejor puntuación ORB contra sus templates
    error: Optional[ProcessingError] = None
    thresholds_calibrated: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.status is VerificationStatus.ERROR) != (self.error is not None):
            raise ValueError("error debe estar definido si y solo si status es ERROR")

    @property
    def accepted(self) -> bool:
        return self.status is VerificationStatus.ACCEPTED

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class Identity:
    """Identidad registrada (sin datos biométricos)."""
    identity_id: int
    name: str
    n_templates: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class RegisteredFace:
    """Todo lo derivado de una cara para registrarla: vector HOG y template ORB."""
    hog: np.ndarray  # float32 (N_FEATURES,)
    orb: "OrbTemplate"


# ---------------------------------------------------------------------------
# Protocolos (dependencias inyectables del motor)
# ---------------------------------------------------------------------------
@runtime_checkable
class FaceProcessor(Protocol):
    """Imagen -> única cara normalizada. Lanza ``FaceProcessingError``."""

    def __call__(self, image: ImageInput) -> "ProcessedFace": ...


@runtime_checkable
class HogSvmClassifier(Protocol):
    """Cara 128x128 gris uint8 -> predicción T1. Lanza ``SvmUnavailableError``."""

    def predict(self, face: np.ndarray) -> HogPrediction: ...

    def scores(self, face: np.ndarray) -> Mapping[str, float]:
        """Margen de decisión de cada clase (para verificación 1:1)."""
        ...

    def predict_hog(self, hog: np.ndarray) -> HogPrediction: ...

    def scores_hog(self, hog: np.ndarray) -> Mapping[str, float]: ...


@runtime_checkable
class OrbMatcher(Protocol):
    """Cara + templates de referencia -> resultado T2."""

    def match(
        self, face: np.ndarray, references: Mapping[str, Sequence["OrbTemplate"]]
    ) -> OrbOutcome: ...

    def match_template(
        self, query: "OrbTemplate", references: Mapping[str, Sequence["OrbTemplate"]]
    ) -> OrbOutcome: ...


@runtime_checkable
class TemplateSource(Protocol):
    """Origen de templates ORB descifrados (p. ej. ``database.manager``)."""

    def load_orb_templates(self) -> Mapping[str, Sequence["OrbTemplate"]]: ...
