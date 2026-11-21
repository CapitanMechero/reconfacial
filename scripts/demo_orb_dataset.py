"""
Demo: comprueba que ORB reconoce a todos los famosos del dataset.

Para cada famoso se elige UNA foto al azar y se reconoce contra el CSV del dataset ORB
(python -m scripts.build_orb_dataset). Por defecto esa foto se EXCLUYE de la base durante
su consulta (si no, se reconocería a sí misma con puntuación máxima y la prueba no valdría):
se compara con las otras fotos del mismo famoso y con las de todos los demás.

Ejecutar: python -m scripts.demo_orb_dataset dataset/ [--csv data/orb_dataset.csv]
                                             [--t2 0.15] [--seed 42] [--include-self]
"""
import argparse
import random
from pathlib import Path

from recognition.detector import process_image
from recognition.errors import FaceProcessingError
from recognition.matching import DEFAULT_T2, match_identity
from recognition.orb import OrbExtractionError, extract_orb
from recognition.orb_store import DATASET_CSV, load_dataset_csv


def main(folder, csv_path=DATASET_CSV, t2=DEFAULT_T2, seed=42, include_self=False,
         detector=None, verbose=True):
    data = load_dataset_csv(csv_path)
    rnd = random.Random(seed)
    results = []

    if verbose:
        print(f"{'famoso':28s}{'foto usada':34s}{'predicho':28s}{'1.º cand.':>9s}{'punt.':>7s}  resultado")
    for name in sorted(data):
        photos = data[name]
        if len(photos) < 2 and not include_self:
            print(f"{name:28s}(solo 1 foto en el CSV: no se puede excluir; se omite)")
            continue
        image, _ = rnd.choice(photos)

        try:
            face = process_image(Path(folder) / image, detector=detector).face
            query = extract_orb(face)
        except (FaceProcessingError, OrbExtractionError) as e:
            results.append(dict(name=name, image=image, predicted=None, ok=False, score=0.0, error=str(e)))
            print(f"{name:28s}{Path(image).name:34s}ERROR: {e}")
            continue

        db = {n: [t for img, t in ph if include_self or img != image] for n, ph in data.items()}
        db = {n: ts for n, ts in db.items() if ts}
        res = match_identity(query, db, t2=t2)

        ok = res.identity == name
        results.append(dict(name=name, image=image, predicted=res.identity, candidate=res.best_candidate,
                            score=res.score, ok=ok, margin=res.margin))
        if verbose:
            print(f"{name:28s}{Path(image).name:34s}{str(res.identity or 'DESCONOCIDO'):28s}"
                  f"{'si' if res.best_candidate == name else 'NO':>9s}{res.score:7.3f}  {'OK' if ok else 'FALLO'}")

    n = len(results)
    ok = sum(r["ok"] for r in results)
    top1 = sum(r.get("candidate") == r["name"] for r in results)
    print(f"\nReconocidos con T2={t2}: {ok}/{n} ({ok / max(n, 1):.1%}) | "
          f"1.er candidato correcto sin umbral: {top1}/{n} ({top1 / max(n, 1):.1%})")
    bad = [r for r in results if not r["ok"]]
    if bad:
        print("Fallos:")
        for r in bad:
            why = r.get("error") or (f"rechazado (mejor candidato {r.get('candidate')}, {r['score']:.3f})"
                                     if r["predicted"] is None else f"confundido con {r['predicted']}")
            print(f"  - {r['name']} ({Path(r['image']).name}): {why}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="carpeta del dataset (dataset/<famoso>/foto.jpg)")
    parser.add_argument("--csv", default=str(DATASET_CSV))
    parser.add_argument("--t2", type=float, default=DEFAULT_T2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-self", action="store_true",
                        help="NO excluir la foto consultada (comprobación trivial, solo para depurar)")
    args = parser.parse_args()
    main(args.folder, args.csv, args.t2, args.seed, args.include_self)
