
import argparse
import csv
from pathlib import Path

from recognition.detector import process_image
from recognition.hog import extract_hog, N_FEATURES


CSV_PATH = Path("legacy/dataset.csv")  # flujo experimental anterior (ver legacy/README.md)


def register(name, image_path):
    name = name.strip()

    if not name:
        raise ValueError("El nombre no puede estar vacío.")

    # Detectar y preprocesar la cara
    face = process_image(image_path).face

    # Extraer características HOG
    features = extract_hog(face)

    # Crear carpeta si no existe
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Comprobar si hay que escribir la cabecera
    new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

    # Guardar nombre y características
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if new_file:
            header = ["name"] + [
                f"hog_{i}" for i in range(N_FEATURES)
            ]
            writer.writerow(header)

        writer.writerow([name] + features.tolist())

    print("Persona inscrita correctamente")
    print("Nombre:", name)
    print("Imagen:", image_path)
    print("Características:", len(features))
    print("Guardado en:", CSV_PATH)

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("image")
    args = parser.parse_args()

    register(args.name, args.image)
