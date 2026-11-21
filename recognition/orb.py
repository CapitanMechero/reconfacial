"""Extracción de características ORB (Integrante 3).

Entrada: cara normalizada (salida de preprocessing.py): imagen en escala de
grises, uint8, normalmente 128x128.

Salida: ``OrbTemplate`` con keypoints serializables (array Nx5) y descriptores
binarios (array uint8 Nx32). Se usa un array en lugar de ``cv2.KeyPoint`` porque
este último no se puede serializar (pickle/cifrado AES) de forma directa.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- Parámetros (a validar experimentalmente, ver doc de diseño §4.3 y §6) ---
ORB_PARAMS = dict(
    nfeatures=500,       # máximo de puntos clave por cara
    scaleFactor=1.2,
    nlevels=6,           # pocos niveles: la cara es pequeña (128x128)
    edgeThreshold=15,    # debe ser >= patchSize; reducido por el tamaño de la cara
    patchSize=15,
    fastThreshold=10,    # más bajo que el defecto (20) para obtener más puntos
    scoreType=cv2.ORB_HARRIS_SCORE,
)
MIN_KEYPOINTS = 15      # por debajo de esto el template no es fiable
DESCRIPTOR_BYTES = 32   # ORB = 256 bits


@dataclass
class OrbTemplate:
    """Template ORB de una cara.

    keypoints: float32 (N, 5) con columnas x, y, size, angle, response.
    descriptors: uint8 (N, 32).
    """
    keypoints: np.ndarray
    descriptors: np.ndarray

    def __len__(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def points(self) -> np.ndarray:
        """Coordenadas (N, 2) de los puntos clave."""
        return self.keypoints[:, :2]

    # --- (de)serialización para guardar el template cifrado en la BD ---
    def to_bytes(self) -> bytes:
        n = len(self)
        header = np.array([n], dtype=np.uint32).tobytes()
        return (header
                + np.ascontiguousarray(self.keypoints, dtype=np.float32).tobytes()
                + np.ascontiguousarray(self.descriptors, dtype=np.uint8).tobytes())

    @staticmethod
    def from_bytes(data: bytes) -> "OrbTemplate":
        n = int(np.frombuffer(data[:4], dtype=np.uint32)[0])
        kp_end = 4 + n * 5 * 4
        if len(data) != kp_end + n * DESCRIPTOR_BYTES:
            raise ValueError("Template ORB corrupto: tamaño inesperado")
        kps = np.frombuffer(data[4:kp_end], dtype=np.float32).reshape(n, 5).copy()
        des = np.frombuffer(data[kp_end:], dtype=np.uint8).reshape(n, DESCRIPTOR_BYTES).copy()
        return OrbTemplate(kps, des)


try:  # errors.py lo define el módulo de detección (Integrante 1)
    from .errors import FaceProcessingError as _Base
except ImportError:  # pragma: no cover - para probar orb.py aislado
    _Base = None


class OrbExtractionError(*((_Base, ValueError) if _Base else (ValueError,))):
    """La cara no tiene suficientes puntos clave para ORB.

    Hereda de FaceProcessingError para que la interfaz la muestre igual que
    el resto de errores de foto. En el reconocimiento (consulta) la cascada
    debe tratarla como «sin coincidencias» -> desconocido, no como fallo.
    """


def _to_gray_uint8(face: np.ndarray) -> np.ndarray:
    if not isinstance(face, np.ndarray) or face.size == 0:
        raise ValueError("La cara debe ser un array de NumPy no vacío")
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if face.dtype != np.uint8:
        face = np.clip(face, 0, 255).astype(np.uint8)
    return face


def extract_orb(face: np.ndarray, min_keypoints: int = MIN_KEYPOINTS) -> OrbTemplate:
    """Extrae puntos clave y descriptores ORB de una cara normalizada.

    Lanza ``OrbExtractionError`` si se detectan menos de ``min_keypoints``
    puntos (el registro debe rechazar esa foto, ver §7.3 del diseño).
    """
    gray = _to_gray_uint8(face)
    orb = cv2.ORB_create(**ORB_PARAMS)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < min_keypoints:
        n = 0 if des is None else len(kps)
        raise OrbExtractionError(
            f"Solo {n} puntos clave ORB (mínimo {min_keypoints}); "
            "la cara tiene poca textura o está mal normalizada")
    arr = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response] for k in kps],
                   dtype=np.float32)
    return OrbTemplate(arr, des)
