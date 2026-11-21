"""Bloqueo exclusivo entre PROCESOS (p. ej. la app y ``python -m sistema`` en el mismo
contenedor). Usa ``fcntl.flock`` (Linux/macOS, y por tanto Docker). En sistemas sin
``fcntl`` (Windows nativo) no bloquea entre procesos; el bloqueo entre hilos lo
aporta cada clase con ``threading``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows nativo
    fcntl = None  # type: ignore[assignment]


@contextmanager
def interprocess_lock(path: Path) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
