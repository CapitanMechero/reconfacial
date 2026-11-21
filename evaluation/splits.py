"""Particiones entrenamiento / calibración / test SIN fuga de información.

* Las identidades se reparten en **conocidas** (registradas), **desconocidas de
  calibración** y **desconocidas de test**; los tres grupos son disjuntos.
* Las fotos de cada identidad conocida se reparten en entrenamiento (referencias
  y datos del SVM), calibración y test. Una foto pertenece a un único conjunto.
* Las consultas impostoras de calibración y de test son de identidades
  distintas entre sí y de las conocidas.
* Los umbrales se eligen SOLO con calibración; test se evalúa una vez.

``validate_split`` comprueba estas propiedades (y que ningún vector HOG de
consulta sea idéntico a uno de entrenamiento) y lanza ``LeakageError``.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from recognition.orb import OrbTemplate

MIN_PHOTOS_KNOWN = 3  # 1 entrenamiento + 1 calibración + 1 test


class LeakageError(AssertionError):
    """Se detectó solapamiento entre conjuntos que invalidaría la evaluación."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    identity: str
    hog: np.ndarray
    orb: Optional[OrbTemplate] = None


@dataclass
class Split:
    train: list[Sample]
    calib_genuine: list[Sample]
    calib_impostor: list[Sample]
    test_genuine: list[Sample]
    test_impostor: list[Sample]
    known: list[str] = field(default_factory=list)
    seed: int = 0


def make_split(samples: list[Sample], seed: int = 42, unknown_calib_frac: float = 0.2,
               unknown_test_frac: float = 0.2, max_train_per_identity: Optional[int] = None) -> Split:
    """``max_train_per_identity=1`` simula el escenario de una foto por persona
    (las consultas de calibración/test son las mismas que sin límite)."""
    by_id: dict[str, list[Sample]] = {}
    for s in samples:
        by_id.setdefault(s.identity, []).append(s)
    rng = random.Random(seed)
    ids = sorted(by_id)
    eligible = [i for i in ids if len(by_id[i]) >= MIN_PHOTOS_KNOWN]
    if len(eligible) < 4:
        raise ValueError("Se necesitan al menos 4 identidades con ≥ 3 fotos para evaluar.")
    rng.shuffle(eligible)
    n_cal = max(1, round(unknown_calib_frac * len(eligible)))
    n_tst = max(1, round(unknown_test_frac * len(eligible)))
    unk_cal, unk_tst = eligible[:n_cal], eligible[n_cal:n_cal + n_tst]
    known = sorted(eligible[n_cal + n_tst:])
    if len(known) < 2:
        raise ValueError("Quedan menos de 2 identidades conocidas; reduce las fracciones.")

    train, cal_g, test_g = [], [], []
    for ident in known:
        photos = sorted(by_id[ident], key=lambda s: s.sample_id)
        rng.shuffle(photos)
        n = len(photos)
        n_q = max(1, round(0.25 * n))
        test_g += photos[:n_q]
        cal_g += photos[n_q:2 * n_q]
        train += photos[2 * n_q:][:max_train_per_identity]
    split = Split(train, cal_g, [s for i in unk_cal for s in by_id[i]],
                  test_g, [s for i in unk_tst for s in by_id[i]], known, seed)
    validate_split(split)
    return split


def _h(s: Sample) -> str:
    return hashlib.sha256(np.ascontiguousarray(s.hog, np.float32).tobytes()).hexdigest()


def validate_split(split: Split) -> None:
    groups = {"train": split.train, "calib_g": split.calib_genuine, "calib_i": split.calib_impostor,
              "test_g": split.test_genuine, "test_i": split.test_impostor}
    seen: dict[str, str] = {}
    for name, group in groups.items():
        for s in group:
            if s.sample_id in seen:
                raise LeakageError(f"Foto {s.sample_id} en «{seen[s.sample_id]}» y «{name}»")
            seen[s.sample_id] = name
    known = set(split.known)
    if {s.identity for s in split.train} != known:
        raise LeakageError("Entrenamiento debe cubrir exactamente las identidades conocidas")
    for name in ("calib_g", "test_g"):
        if not {s.identity for s in groups[name]} <= known:
            raise LeakageError(f"{name} contiene identidades no conocidas")
    ids_ci = {s.identity for s in split.calib_impostor}
    ids_ti = {s.identity for s in split.test_impostor}
    if ids_ci & known or ids_ti & known or ids_ci & ids_ti:
        raise LeakageError("Las identidades impostoras deben ser disjuntas entre sí y de las conocidas")
    train_hashes = {_h(s) for s in split.train}
    for name in ("calib_g", "calib_i", "test_g", "test_i"):
        if any(_h(s) in train_hashes for s in groups[name]):
            raise LeakageError(f"Una consulta de «{name}» es idéntica a una foto de entrenamiento")
    cal_hashes = {_h(s) for s in split.calib_genuine + split.calib_impostor}
    if any(_h(s) in cal_hashes for s in split.test_genuine + split.test_impostor):
        raise LeakageError("Una consulta de test es idéntica a una de calibración")
