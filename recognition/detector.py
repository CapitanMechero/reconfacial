"""Detección facial con MediaPipe Face Landmarker y pipeline completo de la etapa 1."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from recognition.errors import (
    FaceTooSmallError,
    ModelNotFoundError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from recognition.preprocessing import (
    DEFAULT_CONFIG,
    ImageInput,
    PreprocessingConfig,
    load_image,
    preprocess_face,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]   # (x, y, ancho, alto) en píxeles de la imagen original
    landmarks: np.ndarray             # (478, 2) float32, píxeles de la imagen original


@dataclass(frozen=True)
class ProcessedFace:
    face: np.ndarray                  # cara normalizada (gris, uint8, 128x128 por defecto)
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray


class FaceDetector:
    """Envoltorio de MediaPipe Face Landmarker (modo imagen estática)."""

    def __init__(self, model_path=None, max_faces: int = 10, min_confidence: float = 0.5):
        path = Path(model_path or os.environ.get("FACE_LANDMARKER_MODEL", DEFAULT_MODEL_PATH))
        if not path.is_file():
            raise ModelNotFoundError(
                f"No se encuentra el modelo de MediaPipe en {path}. "
                "Ejecuta scripts/download_models.py o define FACE_LANDMARKER_MODEL."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()  # Streamlit puede llamar desde varios hilos

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        """Caras detectadas, ordenadas de mayor a menor área."""
        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(mp_image)

        detections = []
        for face_landmarks in result.face_landmarks:
            points = np.array(
                [[lm.x * width, lm.y * height] for lm in face_landmarks], dtype=np.float32
            )
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            detections.append(FaceDetection((x0, y0, x1 - x0, y1 - y0), points))

        detections.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        return detections

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@lru_cache(maxsize=1)
def get_default_detector() -> FaceDetector:
    """Instancia compartida: el modelo se carga una sola vez."""
    return FaceDetector()


def process_image(
    image: ImageInput,
    config: PreprocessingConfig = DEFAULT_CONFIG,
    detector: FaceDetector | None = None,
) -> ProcessedFace:
    """Imagen -> la única cara de la foto, normalizada, con su bbox y landmarks.

    Exige exactamente una cara válida.
    Lanza InvalidImageError, NoFaceDetectedError, FaceTooSmallError o MultipleFacesError.
    """
    bgr = load_image(image)
    detector = detector or get_default_detector()

    detections = detector.detect(bgr)
    if not detections:
        raise NoFaceDetectedError("No se ha detectado ninguna cara en la imagen.")

    valid = [d for d in detections if min(d.bbox[2], d.bbox[3]) >= config.min_face_px]
    if not valid:
        raise FaceTooSmallError(
            f"Las caras detectadas son demasiado pequeñas (mínimo {config.min_face_px}px)."
        )
    if len(valid) > 1:
        raise MultipleFacesError(len(valid))

    d = valid[0]
    return ProcessedFace(preprocess_face(bgr, d.landmarks, config), d.bbox, d.landmarks)


def detect_and_align(
    image: ImageInput, config: PreprocessingConfig = DEFAULT_CONFIG
) -> list[np.ndarray]:
    """Contrato del documento de diseño (sección 12).

    Devuelve una lista con exactamente una cara normalizada. Se mantiene el tipo lista
    para no romper el contrato acordado; si la imagen no tiene una única cara válida,
    lanza una excepción en lugar de devolver una lista vacía o con varias caras.
    """
    return [process_image(image, config).face]


def draw_detection(image_bgr: np.ndarray, face: ProcessedFace) -> np.ndarray:
    """Copia de la imagen con el bounding box de la cara detectada (útil para la interfaz)."""
    out = image_bgr.copy()
    x, y, w, h = face.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out