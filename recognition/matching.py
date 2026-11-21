"""Matching ORB contra la base de identidades (Integrante 3).

Pipeline por cada par (consulta, referencia):
    1. BFMatcher (Hamming) con knn=2 + ratio test de Lowe.
    2. Filtro espacial: las caras ya vienen alineadas, así que una coincidencia
       válida debe estar cerca de su posición en la referencia.
    3. Verificación geométrica con RANSAC (transformación de similitud).
    4. Puntuación = (inliers - 2) / n_consulta  -> [0, 1].
       - Se restan 2 porque una transformación de similitud se ajusta con solo 2 puntos:
         RANSAC siempre da >= 2 inliers aunque las coincidencias sean ruido.
       - Se divide por los puntos de la CONSULTA (iguales para todas las referencias) y no por
         el mínimo de ambas caras: así una referencia con pocos puntos clave no gana por
         tener un denominador pequeño («efecto imán»).

Decisión final (umbral T2):
    - mejor puntuación >= T2  y  margen sobre el 2.º candidato >= MIN_MARGIN
      -> identidad aceptada; en otro caso -> None («persona desconocida»).

T2 NO está fijado: DEFAULT_T2 es un valor provisional que hay que sustituir por
el resultado de ``calibrate_t2`` con fotos de consulta reales (doc §6.3, §13).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from .orb import OrbTemplate

# --- Parámetros de matching (a validar experimentalmente) ---
RATIO = 0.75            # ratio test de Lowe
MAX_HAMMING = 64        # descarta coincidencias con distancia absoluta alta (de 256 bits)
MAX_DISPLACEMENT = 24.0 # px; las caras están alineadas a 128x128
RANSAC_THRESH = 4.0     # px
MIN_GOOD_FOR_GEOMETRY = 4
MINIMAL_SAMPLE = 2      # puntos que bastan para ajustar la similitud (inliers "gratis")

DEFAULT_T2 = 0.05       # PROVISIONAL (escala nueva): calibrar con scripts.eval_orb / calibrate_t2()
MIN_MARGIN = 0.01       # diferencia mínima entre el 1.º y 2.º candidato


@dataclass
class PairScore:
    """Detalle del matching consulta-referencia (útil para depurar y la memoria)."""
    score: float
    n_good: int = 0       # tras ratio test + filtro espacial
    n_inliers: int = 0    # tras RANSAC
    mean_distance: float = float("nan")  # Hamming medio de los inliers


@dataclass
class MatchResult:
    identity: Optional[str]      # None => persona desconocida
    score: float                 # mejor puntuación (aunque no se acepte)
    best_candidate: Optional[str]
    second_score: float
    margin: float
    threshold: float
    accepted: bool
    scores: dict = field(default_factory=dict)  # person_id -> puntuación


def _as_template(x: Union[OrbTemplate, np.ndarray]) -> OrbTemplate:
    if isinstance(x, OrbTemplate):
        return x
    des = np.asarray(x, dtype=np.uint8)
    return OrbTemplate(np.zeros((len(des), 5), np.float32), des)


def score_pair(query: Union[OrbTemplate, np.ndarray],
               ref: Union[OrbTemplate, np.ndarray],
               ratio: float = RATIO,
               max_hamming: int = MAX_HAMMING,
               max_displacement: Optional[float] = MAX_DISPLACEMENT,
               use_geometry: bool = True) -> PairScore:
    """Puntúa la similitud entre dos templates ORB."""
    q, r = _as_template(query), _as_template(ref)
    if len(q) < 2 or len(r) < 2:
        return PairScore(0.0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(q.descriptors, r.descriptors, k=2)

    good = []
    have_pts = bool(np.any(q.keypoints[:, :2]) or np.any(r.keypoints[:, :2]))
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= ratio * n.distance or m.distance > max_hamming:
            continue
        if have_pts and max_displacement is not None:
            d = np.linalg.norm(q.points[m.queryIdx] - r.points[m.trainIdx])
            if d > max_displacement:
                continue
        good.append(m)

    n_good = len(good)
    denom = len(q)          # puntos de la consulta: igual para todas las referencias
    if n_good == 0:
        return PairScore(0.0)

    # Sin keypoints (solo descriptores) o sin geometría: se usan las coincidencias filtradas.
    if not (use_geometry and have_pts) or n_good < MIN_GOOD_FOR_GEOMETRY:
        inl = good if not (use_geometry and have_pts) else []
        n_in = len(inl)
        md = float(np.mean([m.distance for m in inl])) if inl else float("nan")
        return PairScore(n_in / denom, n_good, n_in, md)

    src = np.float32([r.points[m.trainIdx] for m in good])
    dst = np.float32([q.points[m.queryIdx] for m in good])
    _, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH)
    if mask is None:
        return PairScore(0.0, n_good, 0)
    inliers = [m for m, ok in zip(good, mask.ravel()) if ok]
    n_in = len(inliers)
    md = float(np.mean([m.distance for m in inliers])) if inliers else float("nan")
    return PairScore(max(n_in - MINIMAL_SAMPLE, 0) / denom, n_good, n_in, md)


def match_identity(descriptors: Union[OrbTemplate, np.ndarray],
                   database: Mapping[str, Union[OrbTemplate, Sequence[OrbTemplate]]],
                   t2: float = DEFAULT_T2,
                   min_margin: float = MIN_MARGIN) -> MatchResult:
    """Compara la consulta con todas las identidades y decide con el umbral T2.

    ``database``: {person_id: OrbTemplate} (o lista de templates por persona;
    se toma la mejor puntuación). ``descriptors`` es un ``OrbTemplate`` (lo
    recomendado, permite verificación geométrica) o un array de descriptores.
    """
    if descriptors is None or len(_as_template(descriptors)) == 0:
        # Consulta sin puntos clave: «sin coincidencias» -> desconocido.
        return MatchResult(None, 0.0, None, 0.0, 0.0, t2, False, {})
    scores = {}
    for pid, refs in database.items():
        refs = refs if isinstance(refs, (list, tuple)) else [refs]
        scores[pid] = max((score_pair(descriptors, r).score for r in refs), default=0.0)

    if not scores:
        return MatchResult(None, 0.0, None, 0.0, 0.0, t2, False, {})

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - second
    accepted = best >= t2 and margin >= min_margin
    return MatchResult(best_id if accepted else None, best, best_id, second,
                       margin, t2, accepted, scores)


# ---------------------------------------------------------------------------
# Calibración de T2 (§6.3 y §13.3 del diseño)
# ---------------------------------------------------------------------------
def far_frr(genuine: Sequence[float], impostor: Sequence[float], t2: float):
    """FRR: genuinos con score < T2.  FAR: impostores/desconocidos con score >= T2."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    frr = float(np.mean(g < t2)) if g.size else float("nan")
    far = float(np.mean(i >= t2)) if i.size else float("nan")
    return far, frr


def calibrate_t2(genuine: Sequence[float], impostor: Sequence[float],
                 target_far: float = 0.0) -> dict:
    """Busca el T2 más bajo con FAR <= target_far (minimiza FRR bajo esa restricción).

    ``genuine``: puntuaciones de consultas de personas registradas contra su
    propia identidad. ``impostor``: mejor puntuación de consultas de personas
    NO registradas contra toda la BD. Devuelve T2, FAR, FRR y el EER aproximado.
    """
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    cands = np.unique(np.concatenate([g, i, [0.0]]))
    cands = np.append(cands, cands.max() + 1e-6)
    rows = [(t, *far_frr(g, i, t)) for t in cands]

    ok = [r for r in rows if r[1] <= target_far]
    t, far, frr = min(ok, key=lambda r: (r[2], r[0]))  # menor FRR, luego menor T2
    eer_t, eer_far, eer_frr = min(rows, key=lambda r: abs(r[1] - r[2]))
    return {"t2": float(t), "far": far, "frr": frr,
            "eer_threshold": float(eer_t), "eer": float((eer_far + eer_frr) / 2)}
