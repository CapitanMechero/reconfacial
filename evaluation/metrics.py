"""Métricas de reconocimiento en conjunto abierto y su incertidumbre.

Definiciones (cada consulta produce una decisión: «identificada como X» o
«no aceptada»; *desconocido*, *incierto* y *error* cuentan como no aceptada):

* **Consulta genuina**: foto de una persona REGISTRADA. Denominador: nº de genuinas.
    - Identification Rate (IR)  = genuinas identificadas como SU identidad / genuinas
    - FRR (False Rejection)     = genuinas no aceptadas / genuinas
    - FIR (False Identification)= genuinas aceptadas como OTRA identidad / genuinas
    - IR + FRR + FIR = 1
* **Consulta impostora**: foto de una persona NO registrada. Denominador: nº de impostoras.
    - FAR (False Acceptance)    = impostoras aceptadas como cualquier identidad / impostoras

Todas las tasas incluyen un intervalo de Wilson al 95 %: con pocas consultas es
muy ancho y **debe leerse** (p. ej. 0 de 50 falsas aceptaciones es compatible
con un FAR real de hasta ~7 %).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Optional, Sequence

UNKNOWN = "<no aceptada>"


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float:
        return float("nan") if self.n == 0 else self.k / self.n

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        if self.n == 0:
            return float("nan"), float("nan")
        p, n = self.value, self.n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, center - half), min(1.0, center + half)

    def as_dict(self) -> dict:
        lo, hi = self.wilson()
        return {"k": self.k, "n": self.n, "value": self.value, "ci95": [lo, hi]}


@dataclass(frozen=True)
class OpenSetMetrics:
    ir: Rate
    frr: Rate
    fir: Rate
    far: Rate

    def as_dict(self) -> dict:
        return {name: getattr(self, name).as_dict() for name in ("ir", "frr", "fir", "far")}


def open_set_metrics(genuine_true: Sequence[str], genuine_pred: Sequence[Optional[str]],
                     impostor_pred: Sequence[Optional[str]]) -> OpenSetMetrics:
    """``*_pred``: identidad aceptada o ``None`` si no se aceptó."""
    if len(genuine_true) != len(genuine_pred):
        raise ValueError("genuine_true y genuine_pred deben tener la misma longitud")
    n_g, n_i = len(genuine_true), len(impostor_pred)
    correct = sum(p is not None and p == t for t, p in zip(genuine_true, genuine_pred))
    rejected = sum(p is None for p in genuine_pred)
    wrong = n_g - correct - rejected
    accepted_imp = sum(p is not None for p in impostor_pred)
    return OpenSetMetrics(Rate(correct, n_g), Rate(rejected, n_g), Rate(wrong, n_g),
                          Rate(accepted_imp, n_i))


def per_identity(genuine_true: Sequence[str], genuine_pred: Sequence[Optional[str]]) -> dict:
    """{identidad: {n, correctas, rechazadas, confundidas}} sobre las consultas genuinas."""
    out: dict = {}
    for t, p in zip(genuine_true, genuine_pred):
        row = out.setdefault(t, {"n": 0, "correct": 0, "rejected": 0, "misidentified": 0})
        row["n"] += 1
        row["correct" if p == t else "rejected" if p is None else "misidentified"] += 1
    return out


def confusion(genuine_true: Sequence[str], genuine_pred: Sequence[Optional[str]],
              impostor_pred: Sequence[Optional[str]]) -> dict:
    """Matriz dispersa {(real, predicha): n}. Impostores: real = ``<impostor>``."""
    cm: Counter = Counter()
    for t, p in zip(genuine_true, genuine_pred):
        cm[(t, UNKNOWN if p is None else p)] += 1
    for p in impostor_pred:
        cm[("<impostor>", UNKNOWN if p is None else p)] += 1
    return dict(cm)
