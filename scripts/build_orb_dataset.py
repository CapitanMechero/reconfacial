"""
Crea el CSV del dataset ORB (equivalente ORB de data/dataset.csv de HOG).

Por cada carpeta de famoso registra TODAS sus fotos con el nombre de la carpeta.
Formato (una fila por foto):  name,image,n_keypoints,template
  - template: puntos clave y descriptores ORB serializados (base64), ya que ORB no produce
    un vector de longitud fija como HOG.

Estructura esperada:
    dataset/
      Adam Pearson/  foto1.jpg ... foto5.jpg
      Otro Famoso/   ...

Ejecutar: python -m scripts.build_orb_dataset dataset/ [--out data/orb_dataset.csv]
(Reconstruye el CSV completo cada vez; no añade filas duplicadas.)
"""
import argparse
from pathlib import Path

from recognition.detector import process_image
from recognition.errors import FaceProcessingError
from recognition.orb import OrbExtractionError, extract_orb
from recognition.orb_store import DATASET_CSV, write_dataset_csv

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build(folder, out=DATASET_CSV, detector=None):
    folder = Path(folder)
    dirs = [d for d in sorted(folder.iterdir()) if d.is_dir()]
    if not dirs:
        raise SystemExit(f"{folder} no contiene carpetas de personas.")

    rows, failed = [], []
    for d in dirs:
        name = d.name.strip()
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in EXTS:
                continue
            rel = f"{d.name}/{img.name}"
            try:
                face = process_image(img, detector=detector).face
                rows.append((name, rel, extract_orb(face)))
            except (FaceProcessingError, OrbExtractionError) as e:
                failed.append((rel, str(e)))

    path = write_dataset_csv(rows, out)
    people = {r[0] for r in rows}
    counts = sorted(len(r[2]) for r in rows)
    if counts:
        print(f"Puntos clave ORB por foto: min {counts[0]} | mediana {counts[len(counts) // 2]} | max {counts[-1]}")
        low = [(r[1], len(r[2])) for r in rows if len(r[2]) < 100]
        if low:
            print(f"Fotos con pocos puntos clave (<100; probable baja resolución o poca textura): {len(low)}")
            for rel, n in low[:15]:
                print(f"  - {rel}: {n}")
    print(f"Personas registradas: {len(people)} | fotos registradas: {len(rows)} | descartadas: {len(failed)}")
    for rel, why in failed:
        print(f"  - {rel}: {why}")
    print("Guardado en:", path)
    return rows, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--out", default=str(DATASET_CSV))
    args = parser.parse_args()
    build(args.folder, args.out)
