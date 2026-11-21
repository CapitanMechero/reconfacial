"""Evaluación reproducible: partición -> calibración -> UNA evaluación en test.

Uso::

    python -m legacy.evaluacion_previa.run --hog-csv legacy/dataset.csv          # solo HOG+SVM
    python -m legacy.evaluacion_previa.run --images data/fotos                  # HOG, ORB y cascada

Los umbrales se eligen exclusivamente con el conjunto de calibración. El test
se evalúa una única vez con los umbrales ya fijados. Es un estudio con
identidades desconocidas disjuntas; los umbrales del sistema ENTREGADO se
calibran con ``python -m evaluation.deployed``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation import cascade
from legacy.evaluacion_previa.data import load_hog_csv, load_image_folder
from evaluation.metrics import confusion, per_identity
from evaluation.splits import make_split
from recognition.config import load_settings


def _fmt(rate: dict) -> str:
    lo, hi = rate["ci95"]
    return f"{100 * rate['value']:.1f} % ({rate['k']}/{rate['n']}; IC95 {100 * lo:.1f}–{100 * hi:.1f})"


def run(samples, seed: int, target_far: float, min_margin: float, modes, out_dir: Path,
        max_train_per_identity=None, plot_suffix: str = "") -> dict:
    split = make_split(samples, seed=seed, max_train_per_identity=max_train_per_identity)
    t0 = time.perf_counter()
    cal_g, model = cascade.score_queries(split, split.calib_genuine)
    cal_i, _ = cascade.score_queries(split, split.calib_impostor, model=model)
    tst_g, _ = cascade.score_queries(split, split.test_genuine, model=model)
    tst_i, _ = cascade.score_queries(split, split.test_impostor, model=model)
    elapsed = time.perf_counter() - t0
    n_queries = len(cal_g) + len(cal_i) + len(tst_g) + len(tst_i)

    report: dict = {
        "seed": seed, "target_far": target_far, "orb_min_margin": min_margin,
        "sizes": {"known_identities": len(split.known), "train": len(split.train),
                  "calib_genuine": len(cal_g), "calib_impostor": len(cal_i),
                  "test_genuine": len(tst_g), "test_impostor": len(tst_i),
                  "calib_impostor_identities": len({s.identity for s in split.calib_impostor}),
                  "test_impostor_identities": len({s.identity for s in split.test_impostor})},
        "max_train_per_identity": max_train_per_identity,
        "scoring_ms_per_query": 1000 * elapsed / max(n_queries, 1),
        # Exactitud rank-1 en conjunto cerrado (sin umbral, solo genuinas): referencia
        # de cuánto discrimina el SVM; NO es una métrica de un sistema abierto.
        "closed_set_rank1_test_svm": sum(q.svm_label == q.true_identity for q in tst_g) / max(len(tst_g), 1),
        "closed_set_rank1_test_orb": (sum(q.orb_label == q.true_identity for q in tst_g) / len(tst_g)
                                      if tst_g and tst_g[0].orb_label is not None else None),
        "modes": {},
    }
    for mode in modes:
        sel = cascade.select_thresholds(cal_g, cal_i, split.known, mode, target_far, min_margin)
        test = cascade.evaluate(tst_g, tst_i, sel.t1, sel.t2, min_margin, split.known)  # una vez
        known = frozenset(split.known)
        g_pred = [cascade.decide(q, sel.t1, sel.t2, min_margin, known) for q in tst_g]
        i_pred = [cascade.decide(q, sel.t1, sel.t2, min_margin, known) for q in tst_i]
        stage_t1 = sum(q.svm_label in known and q.svm_score >= sel.t1 for q in tst_g + tst_i)
        report["modes"][mode] = {
            "t1": sel.t1, "t2": sel.t2, "grid_size": sel.n_candidates,
            "calibration": sel.metrics.as_dict(), "test": test.as_dict(),
            "test_decided_by_t1_fraction": stage_t1 / max(len(tst_g) + len(tst_i), 1),
            "test_per_identity": per_identity([q.true_identity for q in tst_g], g_pred),
            "test_confusion": {f"{a} -> {b}": n for (a, b), n in
                               confusion([q.true_identity for q in tst_g], g_pred, i_pred).items()},
            "calibration_curve": (cascade.curve(cal_g, cal_i, split.known, mode, min_margin)
                                  if mode != "cascade" else None),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(report, out_dir, plot_suffix)
    return report


def _plot(report: dict, out_dir: Path, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mode, r in report["modes"].items():
        rows = [c for c in (r["calibration_curve"] or []) if c["threshold"] != float("inf")]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [c["threshold"] for c in rows]
        for key, label in (("far", "FAR"), ("frr", "FRR"), ("ir", "Identification rate")):
            ax.plot(x, [c[key] for c in rows], label=label)
        ax.set_xlabel("T1 (margen SVM)" if mode == "hog" else "T2 (ratio de inliers ORB)")
        ax.set_ylabel("Tasa (conjunto de CALIBRACIÓN)")
        ax.set_title(f"Calibración {mode}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"calibration_{mode}{suffix}.png", dpi=110)
        plt.close(fig)


def write_markdown(reports: dict, path: Path, source: str) -> None:
    lines = [f"# Resultados de evaluación ({source})", "",
             "Generado por `python -m legacy.evaluacion_previa.run`; no editar a mano.", ""]
    for title, report in reports.items():
        lines += _section(report, title)
    path.write_text("\n".join(lines), encoding="utf-8")


def _section(report: dict, title: str) -> list[str]:
    s = report["sizes"]
    r1 = report["closed_set_rank1_test_svm"]
    lines = [f"# Escenario: {title}", "",
             f"Semilla {report['seed']} · FAR objetivo en calibración ≤ {report['target_far']:.0%}"
             f" · rank-1 del SVM en conjunto cerrado (test, sin umbral): {100 * r1:.1f} %",
             "", "| Conjunto | Consultas | Identidades |", "|---|---|---|",
             f"| Entrenamiento / referencias | {s['train']} | {s['known_identities']} |",
             f"| Calibración genuinas | {s['calib_genuine']} | {s['known_identities']} |",
             f"| Calibración impostoras | {s['calib_impostor']} | {s['calib_impostor_identities']} |",
             f"| Test genuinas | {s['test_genuine']} | {s['known_identities']} |",
             f"| Test impostoras | {s['test_impostor']} | {s['test_impostor_identities']} |", ""]
    for mode, r in report["modes"].items():
        t = r["test"]
        lines += [f"## {title} — {mode}", "",
                  f"Umbrales elegidos en calibración: T1 = {r['t1']:.4g}, T2 = {r['t2']:.4g} "
                  "(`inf` = etapa desactivada).", "",
                  "| Métrica (test) | Valor |", "|---|---|",
                  f"| Identification rate | {_fmt(t['ir'])} |", f"| FRR | {_fmt(t['frr'])} |",
                  f"| Falsa identificación (FIR) | {_fmt(t['fir'])} |", f"| FAR | {_fmt(t['far'])} |",
                  "", f"Calibración: IR {_fmt(r['calibration']['ir'])}, FAR {_fmt(r['calibration']['far'])}.", ""]
    return lines


def _repeat_summary(samples, args, margin, modes, name) -> None:
    """Variabilidad entre particiones: misma metodología, semillas distintas."""
    import statistics
    import tempfile
    rows: dict = {}
    for seed in range(args.seed + 1, args.seed + 1 + args.repeat):
        with tempfile.TemporaryDirectory() as tmp:
            rep = run(samples, seed, args.target_far, margin, modes, Path(tmp))
        for mode, r in rep["modes"].items():
            rows.setdefault(mode, []).append((r["test"]["ir"]["value"], r["test"]["far"]["value"]))
    lines = ["", f"## Variabilidad entre {args.repeat} particiones adicionales (test)", "",
             "| Modo | IR media (mín–máx) | FAR media (mín–máx) |", "|---|---|---|"]
    for mode, vals in rows.items():
        ir, far = [v[0] for v in vals], [v[1] for v in vals]
        lines.append(f"| {mode} | {100 * statistics.mean(ir):.1f} % ({100 * min(ir):.1f}–{100 * max(ir):.1f}) "
                     f"| {100 * statistics.mean(far):.1f} % ({100 * min(far):.1f}–{100 * max(far):.1f}) |")
    md = args.out / f"{name}.md"
    md.write_text(md.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hog-csv", type=Path)
    src.add_argument("--images", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=Path("legacy/evaluacion_previa/results"))
    ap.add_argument("--repeat", type=int, default=0,
                    help="Repetir con N semillas adicionales y resumir la variabilidad en test")
    args = ap.parse_args()

    settings = load_settings()
    margin = settings.thresholds.orb_min_margin
    if args.hog_csv:
        failures: dict[str, str] = {}
        samples, modes, source = load_hog_csv(args.hog_csv), ["hog"], "HOG CSV"
    else:
        samples, failures = load_image_folder(args.images)
        modes, source = ["hog", "orb", "cascade"], "fotografías"
    reports = {
        "todas las fotos de entrenamiento": run(samples, args.seed, args.target_far, margin,
                                                modes, args.out),
        "una foto por identidad": run(samples, args.seed, args.target_far, margin, modes,
                                      args.out, max_train_per_identity=1, plot_suffix="_1shot"),
    }
    for r in reports.values():
        r["discarded_photos"] = len(failures)
    name = "hog_csv" if args.hog_csv else "images"
    (args.out / f"{name}.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    write_markdown(reports, args.out / f"{name}.md", source)
    if args.repeat:
        _repeat_summary(samples, args, margin, modes, name)
    print((args.out / f"{name}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
