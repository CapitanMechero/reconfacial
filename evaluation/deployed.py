"""Evaluación y calibración del sistema DESPLEGADO (BD de famosos + fotos reservadas).

Uso::

    python -m evaluation.deployed [--target-far 0.01] [--write-thresholds]

Datos: ``data/famosos.csv`` (fotos inscritas) y ``data/holdout.csv`` (2 fotos por
persona que nunca se inscriben: ``calibracion`` y ``test``). Los umbrales se
eligen SOLO con ``calibracion``; ``test`` se evalúa una vez con ellos fijados.

* **Verificación (1:1)**: genuina = foto reservada que reclama su propia
  identidad; impostora = la misma foto reclamando cualquier OTRA identidad
  (impostor de esfuerzo cero). Regla del motor: acepta si margen SVM de la clase
  reclamada >= T1v, o si no, ORB contra sus templates >= T2v.
* **Identificación (1:N, conjunto abierto)**: hacen falta personas NO inscritas.
  Se simulan con 5 particiones de 10 identidades: en cada una se reentrena el
  SVM y se quitan las referencias ORB de esas 10 personas, cuyas fotos
  reservadas pasan a ser consultas impostoras. Las genuinas son las fotos
  reservadas de las 40 restantes. Limitación: los impostores de calibración y de
  test son las mismas personas (con fotos distintas), y el SVM de cada partición
  tiene 40 clases en lugar de las 50 del despliegue.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from database.csv_store import read_rows
from evaluation import cascade
from evaluation.metrics import Rate
from recognition.config import PROJECT_ROOT, load_settings
from recognition.matching import score_pair
from recognition.svm import decision_scores, train_svm
from recognition.template import extract_template, from_base64

log = logging.getLogger(__name__)
N_FOLDS = 5


def load_seed(path: Path) -> dict[str, list]:
    out: dict[str, list] = {}
    for r in read_rows(path):
        out.setdefault(r.identity, []).append(from_base64(r.template))
    return out


def load_holdout(path: Path, photos: Path) -> dict[str, list[tuple[str, object]]]:
    """{split: [(identidad, template)]} procesando las fotos con el detector real."""
    from recognition.detector import process_image
    out: dict[str, list] = {"calibracion": [], "test": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            face = process_image(photos / row["photo"]).face
            out[row["split"]].append((row["identity"], extract_template(face)))
    return out


def orb_matrix(queries, seed: dict[str, list], names: list[str]) -> np.ndarray:
    """[consulta, identidad] = mejor puntuación ORB contra sus templates inscritos."""
    m = np.zeros((len(queries), len(names)))
    for i, (_, t) in enumerate(queries):
        for j, n in enumerate(names):
            m[i, j] = max(score_pair(t.orb, ref.orb).score for ref in seed[n])
    return m


def svm_matrix(model, queries, names: list[str]) -> np.ndarray:
    m = np.full((len(queries), len(names)), -np.inf)
    for i, (_, t) in enumerate(queries):
        classes, scores = decision_scores(t.hog, model)
        for c, s in zip(classes, scores):
            m[i, names.index(c)] = s
    return m


def _train(seed, include: list[str], C: float):
    X = np.vstack([t.hog for n in include for t in seed[n]])
    y = [n for n in include for _ in seed[n]]
    return train_svm(X, y, C=C)


# ------------------------------- verificación -------------------------------
def verification_pairs(queries, names, svm_m, orb_m, min_evidence: float):
    """(svm, orb, es_genuina, pasa_filtro) para cada par (consulta, identidad reclamada).

    El filtro de evidencia ORB depende solo de la consulta: su mejor puntuación ORB
    contra TODAS las identidades inscritas debe superar ``min_evidence``.
    """
    true_idx = np.array([names.index(n) for n, _ in queries])
    genuine = np.zeros_like(svm_m, dtype=bool)
    genuine[np.arange(len(queries)), true_idx] = True
    evidence = np.repeat(orb_m.max(axis=1, keepdims=True) > min_evidence, orb_m.shape[1], axis=1)
    return svm_m.ravel(), orb_m.ravel(), genuine.ravel(), evidence.ravel()


def _grid(values: np.ndarray, n: int = 150) -> list[float]:
    v = np.unique(values[np.isfinite(values)])
    if v.size > n:
        v = np.unique(np.quantile(v, np.linspace(0, 1, n)))
    mids = ((v[:-1] + v[1:]) / 2).tolist() if v.size > 1 else []
    return sorted(set(mids) | {float(v.min()) - 1e-6, float("inf")})


def verify_rates(svm, orb, gen, t1, t2, evidence=None) -> tuple[Rate, Rate]:
    acc = (svm >= t1) | (orb >= t2)
    if evidence is not None:
        acc &= evidence
    far = Rate(int((acc & ~gen).sum()), int((~gen).sum()))
    frr = Rate(int((~acc & gen).sum()), int(gen.sum()))
    return far, frr


def select_verification(svm, orb, gen, target_far: float, evidence=None):
    best: Optional[tuple] = None
    ties = []
    gate = np.ones_like(gen) if evidence is None else evidence
    for t1 in _grid(svm):
        acc1 = svm >= t1
        for t2 in _grid(orb):
            acc = (acc1 | (orb >= t2)) & gate
            far = (acc & ~gen).sum() / max((~gen).sum(), 1)
            if far > target_far:
                continue
            gar = (acc & gen).sum()
            if best is None or gar > best:
                best, ties = gar, [(t1, t2)]
            elif gar == best:
                ties.append((t1, t2))
    t1, t2 = sorted(ties)[len(ties) // 2]
    return t1, t2


# ------------------------------- identificación -------------------------------
def identification_scores(seed, names, holdout, C, rng_seed: int):
    """QueryScores genuinas/impostoras por split usando particiones de identidades."""
    rng = np.random.default_rng(rng_seed)
    order = list(rng.permutation(names))
    folds = [order[k::N_FOLDS] for k in range(N_FOLDS)]
    orb_m = {s: orb_matrix(q, seed, names) for s, q in holdout.items()}
    out: dict[str, tuple[list, list]] = {s: ([], []) for s in holdout}
    for fold in folds:
        known = [n for n in names if n not in fold]
        model = _train(seed, known, C)
        cols = [names.index(n) for n in known]
        for split, queries in holdout.items():
            sm = svm_matrix(model, queries, names)
            for i, (true, _) in enumerate(queries):
                row = orb_m[split][i, cols]
                o = np.argsort(row)[::-1]
                svm_row = sm[i, cols]
                j = int(np.argmax(svm_row))
                q = cascade.QueryScores(true, known[j], float(svm_row[j]), known[int(o[0])],
                                        float(row[o[0]]), float(row[o[0]] - row[o[1]]))
                (out[split][1] if true in fold else out[split][0]).append(q)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--photos", type=Path, default=PROJECT_ROOT / "data" / "fotos")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    ap.add_argument("--write-thresholds", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    seed = load_seed(settings.paths.seed_db)
    names = sorted(seed)
    holdout = load_holdout(settings.paths.holdout, args.photos)
    margin = settings.thresholds.orb_min_margin
    evidence = settings.thresholds.orb_min_evidence

    # --- verificación con el SVM desplegado (todas las identidades) ---
    model = _train(seed, names, settings.svm.C)
    ver = {}
    for split, q in holdout.items():
        ver[split] = verification_pairs(q, names, svm_matrix(model, q, names),
                                        orb_matrix(q, seed, names), evidence)
    c_svm, c_orb, c_gen, c_ev = ver["calibracion"]
    t1v, t2v = select_verification(c_svm, c_orb, c_gen, args.target_far, c_ev)
    ver_report = {}
    for split in ("calibracion", "test"):
        v_svm, v_orb, v_gen, v_ev = ver[split]
        far, frr = verify_rates(v_svm, v_orb, v_gen, t1v, t2v, v_ev)
        svm_only = verify_rates(v_svm, v_orb, v_gen, t1v, float("inf"), v_ev)
        ver_report[split] = {"far": far.as_dict(), "frr": frr.as_dict(),
                             "far_solo_svm": svm_only[0].as_dict(), "frr_solo_svm": svm_only[1].as_dict()}

    # --- identificación en conjunto abierto ---
    ids = identification_scores(seed, names, holdout, settings.svm.C, args.seed)
    idf_report = {}
    for mode in ("hog", "orb", "cascade"):
        cal_g, cal_i = ids["calibracion"]
        tst_g, tst_i = ids["test"]
        sel = cascade.select_thresholds(cal_g, cal_i, names, mode, args.target_far, margin, evidence)
        test = cascade.evaluate(tst_g, tst_i, sel.t1, sel.t2, margin, names, evidence)
        idf_report[mode] = {"t1": sel.t1, "t2": sel.t2, "calibracion": sel.metrics.as_dict(),
                            "test": test.as_dict()}

    report = {"target_far": args.target_far, "identities": len(names),
              "enrolled_photos": sum(len(v) for v in seed.values()),
              "holdout_per_split": {s: len(q) for s, q in holdout.items()},
              "verification": {"t1v": t1v, "t2v": t2v, **ver_report},
              "identification": idf_report}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "deployed.json").write_text(json.dumps(report, indent=2, default=str))
    _markdown(report, args.out / "deployed.md")
    print((args.out / "deployed.md").read_text(encoding="utf-8"))

    if args.write_thresholds:
        c = idf_report["cascade"]
        settings.paths.thresholds_file.write_text(json.dumps({
            "t1": c["t1"], "t2": c["t2"], "orb_min_margin": margin, "t1_verify": t1v,
            "t2_verify": t2v, "orb_min_evidence": evidence,
            "source": f"evaluation.deployed (calibración, FAR objetivo {args.target_far})",
        }, indent=2, default=str))
        print(f"Umbrales escritos en {settings.paths.thresholds_file}")
    return 0


def _fmt(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['value']:.1f} % ({r['k']}/{r['n']}; IC95 {100 * lo:.1f}–{100 * hi:.1f})"


def _markdown(rep: dict, path: Path) -> None:
    v = rep["verification"]
    lines = ["# Evaluación del sistema desplegado", "",
             "Generado por `python -m evaluation.deployed`; no editar a mano.", "",
             f"{rep['identities']} identidades, {rep['enrolled_photos']} fotos inscritas; "
             f"fotos reservadas por split: {rep['holdout_per_split']}. "
             f"FAR objetivo en calibración ≤ {rep['target_far']:.0%}.", "",
             "## Verificación (1:1)", "",
             f"Umbrales elegidos en calibración: T1v = {v['t1v']:.4g}, T2v = {v['t2v']:.4g}.", "",
             "| Split | FAR | FRR | FAR solo SVM | FRR solo SVM |", "|---|---|---|---|---|"]
    for s in ("calibracion", "test"):
        lines.append(f"| {s} | {_fmt(v[s]['far'])} | {_fmt(v[s]['frr'])} | "
                     f"{_fmt(v[s]['far_solo_svm'])} | {_fmt(v[s]['frr_solo_svm'])} |")
    lines += ["", "## Identificación (1:N, conjunto abierto)", "",
              "| Modo | T1 | T2 | IR test | FRR test | FIR test | FAR test | IR calib. | FAR calib. |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mode, r in rep["identification"].items():
        t, c = r["test"], r["calibracion"]
        lines.append(f"| {mode} | {r['t1']:.4g} | {r['t2']:.4g} | {_fmt(t['ir'])} | {_fmt(t['frr'])} | "
                     f"{_fmt(t['fir'])} | {_fmt(t['far'])} | {_fmt(c['ir'])} | {_fmt(c['far'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
