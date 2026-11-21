"""Puntuación offline de consultas y decisión de la cascada para cualquier (T1, T2).

Se precalculan una sola vez las salidas del SVM y de ORB por consulta; después
cada combinación de umbrales se evalúa sin volver a ejecutar los algoritmos.
La regla de decisión ``decide`` replica la de ``RecognitionEngine`` (un test
comprueba la equivalencia): T1 acepta si la identidad del SVM está registrada y
``score >= T1``; si no, ORB acepta si ``best >= T2`` y ``margen >= min_margin``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from evaluation.metrics import OpenSetMetrics, open_set_metrics
from evaluation.splits import Sample, Split
from recognition.matching import score_pair
from recognition.svm import decision_scores, train_svm

DISABLED = float("inf")  # umbral que desactiva una etapa


@dataclass(frozen=True)
class QueryScores:
    true_identity: str
    svm_label: Optional[str]
    svm_score: float
    orb_label: Optional[str]
    orb_best: float
    orb_margin: float


def score_queries(split: Split, queries: Sequence[Sample], C: float = 0.1,
                  model=None) -> tuple[list[QueryScores], object]:
    """Entrena el SVM con ``split.train`` (si no se da) y puntúa cada consulta."""
    if model is None:
        model = train_svm(np.vstack([s.hog for s in split.train]),
                          [s.identity for s in split.train], C=C)
    refs: dict[str, list] = {}
    for s in split.train:
        if s.orb is not None:
            refs.setdefault(s.identity, []).append(s.orb)
    out = []
    for q in queries:
        classes, scores = decision_scores(q.hog, model)
        i = int(np.argmax(scores))
        orb: tuple[Optional[str], float, float]
        if refs and q.orb is not None:
            per_id = {n: max(score_pair(q.orb, r).score for r in rs) for n, rs in refs.items()}
            ranked = sorted(per_id.items(), key=lambda kv: kv[1], reverse=True)
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            orb = (ranked[0][0], ranked[0][1], ranked[0][1] - second)
        else:
            orb = (None, 0.0, 0.0)
        out.append(QueryScores(q.identity, classes[i], float(scores[i]), *orb))
    return out, model


def decide(q: QueryScores, t1: float, t2: float, min_margin: float,
           known: frozenset[str], min_evidence: Optional[float] = None) -> Optional[str]:
    """``min_evidence``: filtro de evidencia ORB del motor (None = desactivado, p. ej.
    en el estudio solo-HOG, donde no hay templates ORB)."""
    if min_evidence is not None and not q.orb_best > min_evidence:
        return None
    if q.svm_label is not None and q.svm_label in known and q.svm_score >= t1:
        return q.svm_label
    if q.orb_label is not None and q.orb_best >= t2 and q.orb_margin >= min_margin:
        return q.orb_label
    return None


def evaluate(genuine: Sequence[QueryScores], impostor: Sequence[QueryScores], t1: float,
             t2: float, min_margin: float, known: Sequence[str],
             min_evidence: Optional[float] = None) -> OpenSetMetrics:
    k = frozenset(known)
    return open_set_metrics([q.true_identity for q in genuine],
                            [decide(q, t1, t2, min_margin, k, min_evidence) for q in genuine],
                            [decide(q, t1, t2, min_margin, k, min_evidence) for q in impostor])


def _candidates(values: Sequence[float]) -> list[float]:
    """Puntos medios entre puntuaciones observadas consecutivas, un valor por debajo
    del mínimo y ``inf`` (etapa desactivada). Así ningún umbral coincide
    exactamente con una puntuación de calibración."""
    vals = np.unique(np.asarray(values, float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [DISABLED]
    mids = ((vals[:-1] + vals[1:]) / 2).tolist()
    return sorted(set(mids) | {float(vals[0]) - 1e-6, DISABLED})


@dataclass(frozen=True)
class Selection:
    t1: float
    t2: float
    metrics: OpenSetMetrics
    n_candidates: int


def select_thresholds(genuine: Sequence[QueryScores], impostor: Sequence[QueryScores],
                      known: Sequence[str], mode: str, target_far: float,
                      min_margin: float, min_evidence: Optional[float] = None) -> Selection:
    """Elige (T1, T2) en CALIBRACIÓN: máxima IR con FAR <= target_far.

    ``mode``: ``hog`` (solo T1), ``orb`` (solo T2) o ``cascade``. Desempate:
    menor FIR; entre las combinaciones aún empatadas se toma la **central**
    (orden por T1 y T2), para no colocar el umbral pegado a la puntuación
    genuina más baja de calibración, lo que sobreajusta y dispara el FRR en test.
    """
    all_q = list(genuine) + list(impostor)
    t1s = _candidates([q.svm_score for q in all_q]) if mode in ("hog", "cascade") else [DISABLED]
    t2s = _candidates([q.orb_best for q in all_q]) if mode in ("orb", "cascade") else [DISABLED]
    feasible = []
    for t1 in t1s:
        for t2 in t2s:
            m = evaluate(genuine, impostor, t1, t2, min_margin, known, min_evidence)
            if m.far.n == 0 or m.far.value <= target_far:
                feasible.append(((m.ir.value, -m.fir.value), t1, t2, m))
    assert feasible  # (inf, inf) siempre cumple FAR = 0
    top = max(f[0] for f in feasible)
    tied = sorted((f for f in feasible if f[0] == top), key=lambda f: (f[1], f[2]))
    _, t1, t2, m = tied[len(tied) // 2]
    return Selection(t1, t2, m, len(t1s) * len(t2s))


def curve(genuine, impostor, known, mode: str, min_margin: float) -> list[dict]:
    """FAR/FRR/IR frente al umbral de una etapa (para gráficas de calibración)."""
    all_q = list(genuine) + list(impostor)
    key = "svm_score" if mode == "hog" else "orb_best"
    rows = []
    for t in _candidates([getattr(q, key) for q in all_q]):
        t1, t2 = (t, DISABLED) if mode == "hog" else (DISABLED, t)
        m = evaluate(genuine, impostor, t1, t2, min_margin, known)
        rows.append({"threshold": t, "far": m.far.value, "frr": m.frr.value, "ir": m.ir.value})
    return rows
