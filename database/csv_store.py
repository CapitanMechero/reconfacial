"""Base de datos del sistema: un CSV en claro con integridad HMAC.

Formato (una fila por fotografía inscrita)::

    sample_id,identity,template,created_at,hmac

* ``template``: template público en base64 (``recognition.template``).
* ``hmac``: HMAC-SHA256 de ``sample_id``, ``identity``, ``template`` y
  ``created_at``. Al incluir ``identity``, un renombrado se detecta.
* Manifiesto firmado (``state_dir/manifest.json``, separado de la BD): guarda
  ``generation`` y un resumen HMAC del conjunto de filas. Detecta filas
  **borradas**, añadidas o una restauración del CSV a una versión anterior.

Primer arranque (TOFU): si el CSV no existe se copia la semilla versionada
(filas sin firmar). Si no hay manifiesto y ninguna fila está firmada, se firman
todas y se crea el manifiesto. Lo que se altere **antes** de ese momento no se
puede detectar (riesgo residual documentado en ``docs/seguridad.md``).

Escritura atómica en tres pasos, con recuperación si se interrumpe:
manifiesto con ``pending`` -> CSV nuevo (``os.replace``) -> manifiesto final.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import unicodedata
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from database.filelock import interprocess_lock
from database.integrity import sign, verify
from recognition import template as tpl
from recognition.errors import (
    DuplicateIdentityError, IdentityNotFoundError, StorageError, TemplateCorruptError,
)
from recognition.hog import N_FEATURES
from recognition.orb import OrbTemplate
from recognition.types import Identity, RegisteredFace

log = logging.getLogger(__name__)

HEADER = ["sample_id", "identity", "template", "created_at", "hmac"]
ROW_DOMAIN = "facerec-row-v1"
MANIFEST_DOMAIN = "facerec-manifest-v1"
MANIFEST_FORMAT = 1
MAX_NAME_LENGTH = 100


class DatabaseIntegrityError(TemplateCorruptError):
    """La BD se modificó fuera del sistema (firma inválida, filas borradas o restauradas)."""


def normalize_name(name: str) -> tuple[str, str]:
    """(nombre para mostrar, clave de unicidad). Lanza ``ValueError`` si no es válido."""
    display = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(name))).strip()
    if not display:
        raise ValueError("El nombre no puede estar vacío.")
    if len(display) > MAX_NAME_LENGTH:
        raise ValueError(f"El nombre no puede superar {MAX_NAME_LENGTH} caracteres.")
    if any(unicodedata.category(c).startswith("C") for c in display):
        raise ValueError("El nombre contiene caracteres de control no permitidos.")
    return display, display.casefold()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class _Row:
    sample_id: str
    identity: str
    template: str
    created_at: str
    hmac: str = ""


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_rows(path: Path) -> list[_Row]:
    """Lee el CSV sin verificar nada (también lo usan los tests de ataques)."""
    csv.field_size_limit(2**31 - 1)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != HEADER:
                raise DatabaseIntegrityError(f"Cabecera del CSV inesperada: {header}")
            rows = []
            for i, rec in enumerate(reader, start=2):
                if len(rec) != len(HEADER):
                    raise DatabaseIntegrityError(f"Fila {i} del CSV con {len(rec)} columnas.")
                rows.append(_Row(*rec))
            return rows
    except OSError as exc:
        raise StorageError(f"No se puede leer la base de datos: {exc}") from exc


def write_rows(path: Path, rows: Iterable[_Row]) -> None:
    def _w(f):
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r.sample_id, r.identity, r.template, r.created_at, r.hmac])
    _atomic_write(Path(path), _w)


def write_seed(path: Path, entries: Sequence[tuple[str, RegisteredFace]]) -> None:
    """CSV semilla (filas SIN firmar) para versionar en el repositorio."""
    rows = []
    for name, face in entries:
        display, _ = normalize_name(name)
        rows.append(_Row(uuid.uuid4().hex, display, tpl.to_base64(face), _now()))
    write_rows(path, rows)
    os.chmod(path, 0o644)   # se versiona y se copia a la imagen Docker: debe ser legible


class CsvIdentityStore:
    """Repositorio de identidades sobre CSV (implementa ``TemplateSource``)."""

    def __init__(self, db_path: Path | str, key: bytes, manifest_path: Path | str,
                 seed_path: Optional[Path | str] = None) -> None:
        self._path = Path(db_path)
        self._manifest = Path(manifest_path)
        self._seed = Path(seed_path) if seed_path else None
        self._key = key
        self._lock = threading.RLock()
        self._cache_sig: Optional[tuple] = None
        self._rows: list[_Row] = []
        self._faces: dict[str, list[RegisteredFace]] = {}
        self._generation = 0
        self.adopted_on_start = False
        self._lock_path = self._path.with_name(self._path.name + ".lock")
        self._initialize()

    # -- firmas ---------------------------------------------------------------
    def _sign_row(self, r: _Row) -> _Row:
        return replace(r, hmac=sign(self._key, r.sample_id, r.identity, r.template, r.created_at,
                                    domain=ROW_DOMAIN))

    def _digest(self, rows: Sequence[_Row], generation: int) -> str:
        items = sorted(f"{r.sample_id}:{r.hmac}" for r in rows)
        return sign(self._key, str(generation), *items, domain=MANIFEST_DOMAIN)

    def _write_manifest(self, generation: int, digest: str, pending: Optional[str] = None) -> None:
        body = {"format": MANIFEST_FORMAT, "generation": generation, "digest": digest,
                "pending": pending}
        body["hmac"] = sign(self._key, json.dumps(body, sort_keys=True), domain=MANIFEST_DOMAIN)
        _atomic_write(self._manifest, lambda f: f.write(json.dumps(body, indent=2)))

    def _read_manifest(self) -> Optional[dict]:
        try:
            body = json.loads(self._manifest.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise DatabaseIntegrityError(f"Manifiesto de la BD ilegible: {exc}") from exc
        if not isinstance(body, dict) or body.get("format") != MANIFEST_FORMAT:
            raise DatabaseIntegrityError("Formato de manifiesto no soportado.")
        mac = body.pop("hmac", "")
        if not verify(self._key, mac, json.dumps(body, sort_keys=True), domain=MANIFEST_DOMAIN):
            raise DatabaseIntegrityError("La firma del manifiesto no es válida (¿clave cambiada?).")
        return body

    # -- inicialización y carga verificada ------------------------------------------
    def _initialize(self) -> None:
        with self._lock, interprocess_lock(self._lock_path):
            if not self._path.exists():
                if self._seed is not None and self._seed.is_file():
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self._path.with_suffix(".seed.tmp")
                    shutil.copyfile(self._seed, tmp)
                    os.replace(tmp, self._path)
                    log.info("BD creada a partir de la semilla %s", self._seed.name)
                else:
                    write_rows(self._path, [])
            if self._read_manifest() is None:
                rows = read_rows(self._path)
                if any(r.hmac for r in rows):
                    raise DatabaseIntegrityError(
                        "La BD tiene filas firmadas pero falta el manifiesto: posible "
                        "manipulación o pérdida del directorio de estado.")
                for r in rows:            # la semilla debe ser válida antes de adoptarla
                    self._decode(r)
                    normalize_name(r.identity)
                signed = [self._sign_row(r) for r in rows]
                write_rows(self._path, signed)
                self._write_manifest(0, self._digest(signed, 0))
                self.adopted_on_start = True
                log.warning("Primer arranque: %d filas firmadas (confianza en el primer uso)",
                            len(signed))
            self._load()

    def _signature(self) -> tuple:
        try:
            a, b = self._path.stat(), self._manifest.stat()
        except FileNotFoundError as exc:
            raise DatabaseIntegrityError(f"Falta un archivo de la BD: {exc.filename}") from exc
        return (a.st_mtime_ns, a.st_size, a.st_ino, b.st_mtime_ns, b.st_size, b.st_ino)

    @staticmethod
    def _decode(r: _Row) -> RegisteredFace:
        try:
            return tpl.from_base64(r.template)
        except tpl.TemplateFormatError as exc:
            raise TemplateCorruptError(f"Template ilegible en la fila {r.sample_id}: {exc}") from exc

    def _load(self) -> None:
        """Carga y verifica la BD si ha cambiado en disco desde la última vez."""
        with self._lock:
            sig = self._signature()
            if sig == self._cache_sig:
                return
            manifest = self._read_manifest()
            if manifest is None:
                raise DatabaseIntegrityError("Falta el manifiesto de la BD.")
            rows = read_rows(self._path)
            seen: set[str] = set()
            for r in rows:
                if not verify(self._key, r.hmac, r.sample_id, r.identity, r.template, r.created_at,
                              domain=ROW_DOMAIN):
                    raise DatabaseIntegrityError(
                        f"La fila {r.sample_id[:8]}… se ha modificado o añadido fuera del sistema.")
                if r.sample_id in seen:
                    raise DatabaseIntegrityError("Filas duplicadas en la BD.")
                seen.add(r.sample_id)
            generation = int(manifest["generation"])
            if self._digest(rows, generation) != manifest["digest"]:
                pending = manifest.get("pending")
                if pending and self._digest(rows, generation + 1) == pending:
                    # escritura interrumpida entre el CSV y el manifiesto final: se completa
                    self._write_manifest(generation + 1, pending)
                    self._cache_sig = None
                    return self._load()
                raise DatabaseIntegrityError(
                    "El conjunto de filas no coincide con el manifiesto firmado: se han "
                    "borrado/añadido filas o se ha restaurado una versión anterior del CSV.")
            faces: dict[str, list[RegisteredFace]] = {}
            for r in rows:
                faces.setdefault(r.identity, []).append(self._decode(r))
            self._rows, self._faces, self._generation = rows, faces, generation
            self._cache_sig = self._signature()

    def _commit(self, new_rows: list[_Row]) -> None:
        """Escritura atómica y recuperable. Llamar con el lock y la BD recién cargada."""
        g = self._generation
        old_digest = self._digest(self._rows, g)
        new_digest = self._digest(new_rows, g + 1)
        self._write_manifest(g, old_digest, pending=new_digest)
        write_rows(self._path, new_rows)
        self._write_manifest(g + 1, new_digest)
        self._cache_sig = None
        self._load()

    # -- consultas --------------------------------------------------------------------
    @property
    def generation(self) -> int:
        self._load()
        return self._generation

    def _resolve(self, name: str) -> Optional[str]:
        _, norm = normalize_name(name)
        for existing in self._faces:
            if existing.casefold() == norm:
                return existing
        return None

    def list_identities(self) -> list[Identity]:
        self._load()
        by_name: dict[str, list[_Row]] = {}
        for r in self._rows:
            by_name.setdefault(r.identity, []).append(r)
        out = []
        for i, name in enumerate(sorted(by_name, key=str.casefold)):
            dates = sorted(datetime.fromisoformat(r.created_at) for r in by_name[name])
            out.append(Identity(i, name, len(by_name[name]), dates[0], dates[-1]))
        return out

    def get_identity(self, name: str) -> Identity:
        self._load()
        found = self._resolve(name)
        for identity in self.list_identities():
            if identity.name == found:
                return identity
        raise IdentityNotFoundError(f"No existe la identidad «{name}».")

    def exists(self, name: str) -> bool:
        self._load()
        return self._resolve(name) is not None

    def load_templates(self) -> Mapping[str, list[RegisteredFace]]:
        self._load()
        return {n: list(f) for n, f in self._faces.items()}

    def load_orb_templates(self) -> Mapping[str, list[OrbTemplate]]:
        self._load()
        return {n: [f.orb for f in faces] for n, faces in self._faces.items()}

    def load_hog_training_set(self) -> tuple[np.ndarray, np.ndarray, int]:
        with self._lock:
            self._load()
            labels = [n for n, faces in self._faces.items() for _ in faces]
            vectors = [f.hog for faces in self._faces.values() for f in faces]
            X = np.vstack(vectors) if vectors else np.empty((0, N_FEATURES), np.float32)
            return X, np.array(labels), self._generation

    # -- altas y bajas ----------------------------------------------------------------------
    def _new_rows(self, name: str, faces: Sequence[RegisteredFace]) -> list[_Row]:
        now = _now()
        return [self._sign_row(_Row(uuid.uuid4().hex, name, tpl.to_base64(f), now)) for f in faces]

    def add_identity(self, name: str, faces: Sequence[RegisteredFace]) -> Identity:
        display, _ = normalize_name(name)
        if not faces:
            raise ValueError("Se necesita al menos un template para inscribir una identidad.")
        with self._lock, interprocess_lock(self._lock_path):
            self._load()
            if self._resolve(display) is not None:
                raise DuplicateIdentityError(f"Ya existe una identidad «{display}».")
            self._commit(self._rows + self._new_rows(display, faces))
        log.info("Identidad inscrita (%d templates)", len(faces))
        return self.get_identity(display)

    def add_templates(self, name: str, faces: Sequence[RegisteredFace]) -> Identity:
        if not faces:
            raise ValueError("No hay templates que añadir.")
        with self._lock, interprocess_lock(self._lock_path):
            self._load()
            existing = self._resolve(name)
            if existing is None:
                raise IdentityNotFoundError(f"No existe la identidad «{name}».")
            self._commit(self._rows + self._new_rows(existing, faces))
        return self.get_identity(existing)

    def delete_identity(self, name: str) -> None:
        with self._lock, interprocess_lock(self._lock_path):
            self._load()
            existing = self._resolve(name)
            if existing is None:
                raise IdentityNotFoundError(f"No existe la identidad «{name}».")
            self._commit([r for r in self._rows if r.identity != existing])
        log.info("Identidad eliminada")
