"""
Evaluación REAL de ORB sobre un dataset con una carpeta por persona:

    dataset/
      Nombre_Famoso/  foto1.jpg ... foto5.jpg
      Otro_Famoso/    ...

Protocolo (se repite ``--rounds`` veces con repartos distintos y se acumulan resultados):
  1. Se apartan ``--unknown`` personas COMPLETAS: no se registran; todas sus fotos son consultas de
     personas DESCONOCIDAS (miden FAR).
  2. De cada persona restante se eligen ``--refs`` fotos como referencia (por defecto 1, como en el
     proyecto: una foto por persona). Sus fotos restantes son consultas de personas CONOCIDAS.
  3. Se puntúa cada consulta contra todas las identidades registradas.

Salida: identificación correcta, FAR, FRR, errores de identidad (aceptada pero otra persona) y T2
sugerido (calibrate_t2).

Ejecutar: python -m legacy.integrante3.eval_orb dataset/ [--unknown 10] [--refs 1] [--rounds 3] [--seed 42]
"""
import argparse
import random
from pathlib import Path

import numpy as np

from recognition.detector import process_image
from recognition.errors import FaceProcessingError
from recognition.matching import DEFAULT_T2, MIN_MARGIN, calibrate_t2, match_identity
from recognition.orb import OrbExtractionError, extract_orb

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_people(root, detector=None):
    """{persona: [cara normalizada, ...]} y lista de fotos descartadas."""
    root = Path(root)
    people, failed = {}, []
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not dirs:
        raise SystemExit(f"{root} no contiene carpetas de personas (dataset/<persona>/foto.jpg).")
    for d in dirs:
        faces = []
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in EXTS:
                continue
            try:
                faces.append(process_image(img, detector=detector).face)
            except FaceProcessingError as e:
                failed.append((f"{d.name}/{img.name}", str(e)))
        if faces:
            people[d.name] = faces
    return people, failed


def decide(scores, t2, min_margin=MIN_MARGIN):
    """Misma regla que match_identity, aplicada a puntuaciones ya calculadas."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_id if (best >= t2 and best - second >= min_margin) else None


def _template(face):
    try:
        return extract_orb(face)
    except OrbExtractionError:
        return None


def run_rounds(people, n_unknown, n_refs, rounds, seed):
    known_q, unknown_q, no_orb = [], [], 0      # (persona, {id: puntuación})
    for r in range(rounds):
        rnd = random.Random(seed + r)
        names = sorted(people)
        unknown = set(rnd.sample(names, min(n_unknown, len(names) - 2)))

        db, queries = {}, []
        for n in names:
            if n in unknown or len(people[n]) <= n_refs:
                continue
            idx = set(rnd.sample(range(len(people[n])), n_refs))
            refs = [t for i in idx if (t := _template(people[n][i])) is not None]
            if not refs:
                continue
            db[n] = refs
            queries += [(n, f) for i, f in enumerate(people[n]) if i not in idx]
        if len(db) < 2:
            raise SystemExit("Hay muy pocas personas con fotos suficientes para registrar.")

        for n, face in queries + [(u, f) for u in unknown for f in people[u]]:
            t = _template(face)
            if t is None:
                no_orb += 1
                continue
            scores = match_identity(t, db, t2=0.0, min_margin=0.0).scores
            (unknown_q if n in unknown else known_q).append((n, scores))
    return known_q, unknown_q, no_orb


def main(folder, n_unknown=10, n_refs=1, rounds=3, seed=42, detector=None):
    people, failed = load_people(folder, detector)
    print(f"Personas: {len(people)} | fotos válidas: {sum(map(len, people.values()))} | descartadas: {len(failed)}")
    for name, why in failed:
        print(f"  - {name}: {why}")

    known_q, unknown_q, no_orb = run_rounds(people, n_unknown, n_refs, rounds, seed)
    genuine = [s[n] for n, s in known_q]
    impostor = [max(s.values()) for _, s in unknown_q]
    rank1 = np.mean([max(s, key=s.get) == n for n, s in known_q])

    print(f"\nConsultas conocidas: {len(known_q)} | desconocidas: {len(unknown_q)} | "
          f"sin puntos ORB: {no_orb}  ({rounds} rondas, {n_refs} foto(s) de referencia)")
    print(f"Puntuación consulta vs. su identidad  : min {min(genuine):.3f}  mediana {np.median(genuine):.3f}  max {max(genuine):.3f}")
    print(f"Mejor puntuación de desconocidos      : min {min(impostor):.3f}  mediana {np.median(impostor):.3f}  max {max(impostor):.3f}")
    print(f"Identificación correcta SIN umbral (rank-1): {rank1:.1%}")

    cals = {f"FAR<={t:.0%}": calibrate_t2(genuine, impostor, target_far=t) for t in (0.0, 0.05)}
    thresholds = {"T2 provisional": DEFAULT_T2, **{f"T2 sugerido ({k})": c["t2"] for k, c in cals.items()}}

    print(f"\n{'umbral':28s}{'T2':>7s}{'correcta':>10s}{'otra pers.':>12s}{'rechazada(FRR)':>16s}{'FAR':>8s}")
    for label, t2 in thresholds.items():
        out = [decide(s, t2) for _, s in known_q]
        ok = np.mean([o == n for o, (n, _) in zip(out, known_q)])
        wrong = np.mean([o is not None and o != n for o, (n, _) in zip(out, known_q)])
        frr = np.mean([o is None for o in out])
        far = np.mean([decide(s, t2) is not None for _, s in unknown_q])
        print(f"{label:28s}{t2:7.3f}{ok:10.1%}{wrong:12.1%}{frr:16.1%}{far:8.1%}")
    eer = next(iter(cals.values()))
    print(f"\nEER aprox.: {eer['eer']:.1%} (T2 = {eer['eer_threshold']:.3f})")
    print("Nota: si genuinas y desconocidos se solapan, ORB solo no basta; ahí actúa HOG+SVM (T1).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--unknown", type=int, default=10, help="personas apartadas como desconocidas")
    parser.add_argument("--refs", type=int, default=1, help="fotos de referencia por persona")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.folder, args.unknown, args.refs, args.rounds, args.seed)
