"""Genera la BD precargada de famosos (``data/famosos.csv``) y las fotos reservadas.

Uso::

    python -m scripts.seed_db [--photos data/fotos] [--holdout 2] [--seed 42]

Por cada persona (carpeta ``<fotos>/<id>/``; el id es el nombre de la carpeta):

1. Pasa todas sus fotos por el pipeline real (MediaPipe + HOG + ORB). Las que
   fallan (sin cara, varias caras, sin textura) se descartan y se listan.
2. Reserva ``--holdout`` fotos válidas al azar (semilla fija): NUNCA se inscriben y
   se usan para calibrar umbrales (1.ª) y para el test final (2.ª).
3. Inscribe el resto en el CSV semilla, SIN firmar (se firma en el primer arranque).

Las fotos no se versionan; el CSV y ``data/holdout.csv`` sí (decisión del equipo).
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

from database.csv_store import write_seed
from recognition.config import PROJECT_ROOT
from recognition.errors import FaceProcessingError
from recognition.orb import OrbExtractionError
from recognition.template import extract_template

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("calibracion", "test")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", type=Path, default=PROJECT_ROOT / "data" / "fotos")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "famosos.csv")
    ap.add_argument("--holdout-out", type=Path, default=PROJECT_ROOT / "data" / "holdout.csv")
    ap.add_argument("--holdout", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    from recognition.detector import process_image

    rng = random.Random(args.seed)
    entries, holdout, discarded = [], [], []
    people = sorted(p for p in args.photos.iterdir() if p.is_dir())
    for person in people:
        valid = []
        for img in sorted(p for p in person.iterdir() if p.suffix.lower() in IMAGE_EXT):
            rel = img.relative_to(args.photos).as_posix()
            try:
                valid.append((rel, extract_template(process_image(img).face)))
            except (FaceProcessingError, OrbExtractionError) as exc:
                discarded.append((rel, str(exc)))
        if len(valid) <= args.holdout:
            print(f"ERROR: {person.name} tiene {len(valid)} fotos válidas; se necesitan más de "
                  f"{args.holdout}", file=sys.stderr)
            return 1
        reserved = rng.sample(range(len(valid)), args.holdout)
        for split, idx in zip(SPLITS, reserved):
            holdout.append((person.name, valid[idx][0], split))
        entries += [(person.name, t) for i, (_, t) in enumerate(valid) if i not in reserved]

    write_seed(args.out, entries)
    with open(args.holdout_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["identity", "photo", "split"])
        w.writerows(holdout)
    print(f"{len(people)} personas, {len(entries)} fotos inscritas, {len(holdout)} reservadas, "
          f"{len(discarded)} descartadas -> {args.out}")
    for rel, why in discarded:
        print(f"  descartada {rel}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
