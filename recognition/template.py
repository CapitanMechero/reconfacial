"""Formato público del template biométrico del sistema (HOG + ORB).

Es lo que reciben ``inscribir_t``, ``verificar_t`` e ``identificar_t`` y lo que
se guarda en cada fila del CSV. Formato binario versionado (sin ``pickle``)::

    b"FRT"               magia
    uint8                versión (1)
    float32[8100]        vector HOG (little-endian)
    bytes                OrbTemplate.to_bytes(): uint32 N, float32[N,5] puntos, uint8[N,32] descriptores

En el CSV y en la interfaz de los métodos ``_t`` se usa codificado en base64.
Se acepta también un ``RegisteredFace`` o los bytes crudos.
"""
from __future__ import annotations

import base64
import binascii
from typing import Union

import numpy as np

from recognition.hog import N_FEATURES, extract_hog
from recognition.orb import OrbTemplate, extract_orb
from recognition.types import RegisteredFace

MAGIC = b"FRT"
VERSION = 1
_HOG_BYTES = N_FEATURES * 4
_HEADER = len(MAGIC) + 1

TemplateInput = Union[RegisteredFace, bytes, bytearray, str]


class TemplateFormatError(ValueError):
    """El template no tiene el formato esperado (truncado, versión, valores no finitos…)."""


def extract_template(face: np.ndarray) -> RegisteredFace:
    """Cara normalizada 128x128 gris uint8 -> template. Lanza ``OrbExtractionError``."""
    return RegisteredFace(extract_hog(face), extract_orb(face))


def to_bytes(template: RegisteredFace) -> bytes:
    validate(template)
    hog = np.ascontiguousarray(template.hog, dtype="<f4")
    return MAGIC + bytes([VERSION]) + hog.tobytes() + template.orb.to_bytes()


def from_bytes(data: bytes) -> RegisteredFace:
    data = bytes(data)
    if len(data) < _HEADER + _HOG_BYTES + 4 or data[:3] != MAGIC:
        raise TemplateFormatError("No es un template del sistema (cabecera inválida o truncado).")
    if data[3] != VERSION:
        raise TemplateFormatError(f"Versión de template no soportada: {data[3]}.")
    hog = np.frombuffer(data[_HEADER:_HEADER + _HOG_BYTES], dtype="<f4").astype(np.float32)
    try:
        orb = OrbTemplate.from_bytes(data[_HEADER + _HOG_BYTES:])
    except ValueError as exc:
        raise TemplateFormatError(f"Template ORB inválido: {exc}") from exc
    template = RegisteredFace(hog, orb)
    validate(template)
    return template


def validate(template: RegisteredFace) -> None:
    """Rango físico de cada campo: HOG con normalización L2-Hys está en [0, 1]; los
    puntos ORB deben caer dentro de la cara de 128x128."""
    hog = np.asarray(template.hog)
    if hog.shape != (N_FEATURES,) or not np.isfinite(hog).all():
        raise TemplateFormatError("Vector HOG inválido (dimensión o valores no finitos).")
    if hog.min() < 0 or hog.max() > 1.0 + 1e-5:
        raise TemplateFormatError("Vector HOG fuera de rango [0, 1].")
    orb = template.orb
    if len(orb) == 0:
        raise TemplateFormatError("El template ORB no tiene puntos clave.")
    pts = orb.keypoints
    if not np.isfinite(pts).all() or pts[:, :2].min() < 0 or pts[:, :2].max() > 128:
        raise TemplateFormatError("Puntos ORB fuera de la cara normalizada de 128x128.")


def to_base64(template: RegisteredFace) -> str:
    return base64.b64encode(to_bytes(template)).decode("ascii")


def from_base64(text: str) -> RegisteredFace:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TemplateFormatError("El template no es base64 válido.") from exc
    return from_bytes(raw)


def as_template(value: TemplateInput) -> RegisteredFace:
    """Normaliza cualquiera de las representaciones aceptadas."""
    if isinstance(value, RegisteredFace):
        validate(value)
        return value
    if isinstance(value, (bytes, bytearray)):
        return from_bytes(bytes(value))
    if isinstance(value, str):
        return from_base64(value)
    raise TemplateFormatError(f"Tipo de template no soportado: {type(value).__name__}.")
