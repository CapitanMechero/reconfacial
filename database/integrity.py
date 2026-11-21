"""Clave HMAC e integridad de la base de datos CSV.

La BD está en claro (decisión del equipo), así que esto **no** aporta
confidencialidad: solo detecta modificaciones hechas sin la clave.

Clave: 32 bytes aleatorios en base64, en un archivo con permisos 0600. Se crea
en el primer arranque (``load_or_create_key``) en el directorio de estado, que
debe estar separado de la BD. HMAC-SHA256 de la biblioteca estándar.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import secrets
from pathlib import Path

from recognition.errors import KeyConfigurationError

KEY_BYTES = 32
log = logging.getLogger(__name__)


def generate_key() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).decode("ascii")


def parse_key(text: str) -> bytes:
    """Valida y decodifica la clave sin revelarla en los mensajes de error."""
    cleaned = text.strip()
    if not cleaned:
        raise KeyConfigurationError("La clave de integridad está vacía.")
    try:
        raw = base64.urlsafe_b64decode(cleaned + "=" * (-len(cleaned) % 4))
    except (binascii.Error, ValueError) as exc:
        raise KeyConfigurationError("La clave de integridad no es base64 válido.") from exc
    if len(raw) != KEY_BYTES:
        raise KeyConfigurationError(f"La clave debe tener {KEY_BYTES} bytes; tiene {len(raw)}.")
    return raw


def load_or_create_key(path: Path) -> tuple[bytes, bool]:
    """(clave, creada_ahora). Crea la clave de forma exclusiva si no existe."""
    path = Path(path)
    if path.exists():
        try:
            return parse_key(path.read_text(encoding="utf-8")), False
        except OSError as exc:
            raise KeyConfigurationError(f"No se puede leer la clave de integridad ({path}).") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:            # otro proceso la creó a la vez
        return parse_key(path.read_text(encoding="utf-8")), False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(generate_key())
        f.flush()
        os.fsync(f.fileno())
    log.warning("Clave de integridad nueva creada en %s (primer arranque)", path)
    return parse_key(path.read_text(encoding="utf-8")), True


def sign(key: bytes, *fields: str, domain: str) -> str:
    """HMAC-SHA256 en hex de ``fields`` separados sin ambigüedad (prefijo de longitud)."""
    mac = hmac.new(key, domain.encode(), hashlib.sha256)
    for f in fields:
        b = f.encode("utf-8")
        mac.update(len(b).to_bytes(8, "big") + b)
    return mac.hexdigest()


def verify(key: bytes, signature: str, *fields: str, domain: str) -> bool:
    return hmac.compare_digest(sign(key, *fields, domain=domain), signature or "")
