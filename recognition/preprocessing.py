"""Carga y validación de imágenes, alineamiento, recorte y normalización.

No depende de MediaPipe: trabaja con landmarks ya calculados (array Nx2 en píxeles).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from recognition.errors import FaceProcessingError, InvalidImageError

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]

MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 50_000_000

# Esquinas de los ojos en la malla de MediaPipe (478 landmarks).
# Grupo A: (33, 133) y grupo B: (362, 263). Se ordenan por coordenada x después.
_EYE_A = (33, 133)
_EYE_B = (362, 263)


@dataclass(frozen=True)
class PreprocessingConfig:
    output_size: int = 128          # lado de la cara normalizada (px)
    left_eye_x: float = 0.30        # posición objetivo del ojo izquierdo (fracción del ancho)
    right_eye_x: float = 0.70       # posición objetivo del ojo derecho
    eye_y: float = 0.35             # altura objetivo de los ojos (fracción del alto)
    min_face_px: int = 48           # lado mínimo del bounding box para aceptar una cara
    use_clahe: bool = False         # normalización de iluminación (activar solo si las pruebas lo justifican)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8


DEFAULT_CONFIG = PreprocessingConfig()


# --------------------------------------------------------------------------- #
# Carga y validación
# --------------------------------------------------------------------------- #
def _ensure_bgr(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.uint8:
        raise InvalidImageError("La imagen debe ser un array uint8.")
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    raise InvalidImageError(f"Forma de imagen no soportada: {array.shape}.")


def load_image(source: ImageInput) -> np.ndarray:
    """Devuelve una imagen BGR uint8 validada.

    Acepta ruta, bytes (p. ej. el archivo subido en Streamlit) o array de NumPy (BGR).
    Lanza InvalidImageError si no se puede usar.
    """
    image = None

    if isinstance(source, np.ndarray):
        image = _ensure_bgr(source)
    elif isinstance(source, (bytes, bytearray)):
        buffer = np.frombuffer(source, dtype=np.uint8)
        if buffer.size == 0:
            raise InvalidImageError("El archivo está vacío.")
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"No existe el archivo: {path}")
        try:
            # np.fromfile + imdecode funciona también con rutas con acentos en Windows
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError as exc:
            raise InvalidImageError(f"No se pudo leer el archivo: {exc}") from exc
    else:
        raise InvalidImageError(f"Tipo de entrada no soportado: {type(source).__name__}")

    if image is None:
        raise InvalidImageError("Imagen dañada o con formato no compatible.")

    height, width = image.shape[:2]
    if min(height, width) < MIN_IMAGE_SIDE:
        raise InvalidImageError(f"Imagen demasiado pequeña ({width}x{height}).")
    if height * width > MAX_IMAGE_PIXELS:
        raise InvalidImageError(f"Imagen demasiado grande ({width}x{height}).")
    return image


# --------------------------------------------------------------------------- #
# Alineamiento
# --------------------------------------------------------------------------- #
def eye_centers(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centros de los ojos (px), ordenados (izquierdo, derecho) según la imagen."""
    a = landmarks[list(_EYE_A)].mean(axis=0)
    b = landmarks[list(_EYE_B)].mean(axis=0)
    return (a, b) if a[0] <= b[0] else (b, a)


def compute_alignment_matrix(
    left_eye, right_eye, config: PreprocessingConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Matriz afín 2x3 (rotación + escala + traslación) que nivela los ojos y
    los coloca en su posición canónica dentro de la imagen de salida."""
    left = np.asarray(left_eye, dtype=np.float64)
    right = np.asarray(right_eye, dtype=np.float64)
    dx, dy = right - left
    eye_dist = float(np.hypot(dx, dy))
    if eye_dist < 1e-6:
        raise FaceProcessingError("Landmarks de los ojos degenerados.")

    size = config.output_size
    desired_dist = (config.right_eye_x - config.left_eye_x) * size
    scale = desired_dist / eye_dist
    angle = float(np.degrees(np.arctan2(dy, dx)))
    center = (left + right) / 2.0

    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    matrix[0, 2] += size * (config.left_eye_x + config.right_eye_x) / 2.0 - center[0]
    matrix[1, 2] += size * config.eye_y - center[1]
    return matrix


def align_and_crop(
    image_bgr: np.ndarray, left_eye, right_eye, config: PreprocessingConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Alinea, recorta y redimensiona en un solo warp. Devuelve escala de grises."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    matrix = compute_alignment_matrix(left_eye, right_eye, config)
    size = config.output_size
    # BORDER_REPLICATE evita bordes negros, que generarían puntos clave falsos en ORB
    return cv2.warpAffine(
        gray, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# --------------------------------------------------------------------------- #
# Normalización
# --------------------------------------------------------------------------- #
def normalize_illumination(
    face_gray: np.ndarray, config: PreprocessingConfig = DEFAULT_CONFIG
) -> np.ndarray:
    if not config.use_clahe:
        return face_gray
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_grid, config.clahe_tile_grid),
    )
    return clahe.apply(face_gray)


def preprocess_face(
    image_bgr: np.ndarray, landmarks: np.ndarray, config: PreprocessingConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Cara normalizada: escala de grises, uint8, (output_size, output_size)."""
    left, right = eye_centers(landmarks)
    face = align_and_crop(image_bgr, left, right, config)
    return normalize_illumination(face, config)
