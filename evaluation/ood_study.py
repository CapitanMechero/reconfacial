"""Estudio: ¿puede HOG distinguir caras de no-caras para filtrar las entradas de ``_sd``/``_t``?

Uso::

    python -m evaluation.ood_study

Mide dos «filtros de verosimilitud facial» candidatos (opción (a) contra los ataques
A2/B4) sobre tres grupos:

* caras reales de personas NO inscritas (fotos reservadas; la propia identidad se
  excluye de las referencias para simularlo);
* entradas que no son caras pasadas como ``_sd`` (ruido, superficies lisas,
  degradados, recortes de fondo) y fotos completas sin recortar;
* templates aleatorios con formato válido.

Filtros: (1) similitud coseno con la cara inscrita más cercana (simple y centrada);
(2) error de reconstrucción en un subespacio PCA de las caras inscritas.
El umbral se fija para que pase el 99 % de las caras de calibración.

Resultado (21-11-2026): NINGUNO separa; ver docs/evaluacion.md. Por eso no se implementó.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from evaluation.attacks import non_face_images, random_template
from evaluation.deployed import load_seed
from recognition.config import PROJECT_ROOT, load_settings
from recognition.hog import extract_hog
from recognition.service import face_without_detector


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def main() -> int:
    logging.disable(logging.CRITICAL)
    from recognition.detector import process_image
    settings = load_settings()
    photos_dir = PROJECT_ROOT / "data" / "fotos"
    seed = load_seed(settings.paths.seed_db)
    names = sorted(seed)
    H = np.vstack([t.hog for n in names for t in seed[n]]).astype(np.float64)
    lab = np.array([n for n in names for _ in seed[n]])
    with open(settings.paths.holdout, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    faces = {s: [(r["identity"], extract_hog(process_image(photos_dir / r["photo"]).face))
                 for r in rows if r["split"] == s] for s in ("calibracion", "test")}
    photos = [Path(photos_dir / r["photo"]) for r in rows]
    junk = np.vstack([extract_hog(face_without_detector(cv2.imencode(".png", j)[1].tobytes()))
                      for j in non_face_images(200, photos, seed=3)])
    rnd = np.vstack([random_template(np.random.default_rng(11 + i)).hog for i in range(200)])
    full = np.vstack([extract_hog(face_without_detector(str(p))) for p in photos[:100]])
    groups = {"no-caras (_sd)": junk, "templates aleatorios (_t)": rnd, "fotos completas (_sd)": full}

    def report(title, cal, tst, other, higher_is_face):
        tau = np.quantile(cal, 0.01 if higher_is_face else 0.99)
        ok = (lambda v: v >= tau) if higher_is_face else (lambda v: v <= tau)
        print(f"\n{title} (umbral {tau:.3f})")
        print(f"  caras desconocidas test que pasan: {np.mean(ok(tst)):.1%}")
        for name, v in other.items():
            print(f"  {name:28s} que pasan: {np.mean(ok(v)):.1%}")

    for centered in (False, True):
        mu = H.mean(0) if centered else 0.0
        R = _unit(H - mu)

        def near(X, excl=None, mu=mu, R=R):
            s = _unit(np.atleast_2d(X) - mu) @ R.T
            if excl is not None:
                s[:, lab == excl] = -np.inf
            return s.max(1)
        cal = np.array([near(h, n)[0] for n, h in faces["calibracion"]])
        tst = np.array([near(h, n)[0] for n, h in faces["test"]])
        report(f"Coseno con la cara más cercana ({'centrado' if centered else 'simple'})",
               cal, tst, {k: near(v) for k, v in groups.items()}, True)

    for k in (10, 40):
        def fit(X, k=k):
            m = X.mean(0)
            return m, np.linalg.svd(X - m, full_matrices=False)[2][:k]

        def res(x, m, V):
            d = np.atleast_2d(x) - m
            return np.linalg.norm(d - (d @ V.T) @ V, axis=1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        fits = {n: fit(H[lab != n]) for n in names}
        m, V = fit(H)
        cal = np.array([res(h, *fits[n])[0] for n, h in faces["calibracion"]])
        tst = np.array([res(h, *fits[n])[0] for n, h in faces["test"]])
        report(f"Error de reconstrucción PCA (k={k})", cal, tst,
               {g: res(v, m, V) for g, v in groups.items()}, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
